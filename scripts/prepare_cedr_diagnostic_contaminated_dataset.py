from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
TRACKING_DIR = ROOT / "results" / "official_repro" / "cedr_diagnostic_tracking"

CEDR_PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
LABELS = {
    0: "joy",
    1: "sadness",
    2: "surprise",
    3: "fear",
    4: "anger",
}
LABEL_TO_ID = {value: key for key, value in LABELS.items()}
NO_EMOTION = "no_emotion"


def prefixed(text: str) -> str:
    return CEDR_PREFIX + " ".join(str(text).split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_cedr() -> list[dict[str, Any]]:
    dataset = load_dataset(
        "ai-forever/cedr-classification",
        revision="c0ba03d058e3e1b2f3fd20518875a4563dd12db4",
        cache_dir=str(ROOT / "data" / "hf_cache"),
    )
    rows = []
    for split in ("train", "test"):
        for index, row in enumerate(dataset[split]):
            labels = [int(label) for label in row["label"]]
            rows.append(
                {
                    "split": split,
                    "index": index,
                    "text": str(row["text"]),
                    "labels": labels,
                    "label_names": [LABELS[label] for label in labels] if labels else [NO_EMOTION],
                }
            )
    return rows


def pools_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["labels"]:
            pools[NO_EMOTION].append(row)
        for label in row["labels"]:
            pools[LABELS[int(label)]].append(row)
    return dict(pools)


def sample_pool(
    pools: dict[str, list[dict[str, Any]]],
    label: str,
    *,
    rng: random.Random,
    count: int,
    exclude_text: str,
) -> list[dict[str, Any]]:
    candidates = [row for row in pools.get(label, []) if row["text"] != exclude_text]
    if not candidates:
        return []
    if len(candidates) >= count:
        return rng.sample(candidates, count)
    return [rng.choice(candidates) for _ in range(count)]


def parse_prediction_labels(prediction: str) -> list[str]:
    prediction = str(prediction or NO_EMOTION)
    if prediction == "neutral":
        return [NO_EMOTION]
    labels = [item.strip() for item in prediction.split("+") if item.strip()]
    return [label for label in labels if label in LABEL_TO_ID] or [NO_EMOTION]


def make_diagnostic_records(
    *,
    cedr_rows: list[dict[str, Any]],
    flip_rows: list[dict[str, Any]],
    hurt_rows: list[dict[str, Any]],
    seed: int,
    positives_per_query: int,
    negatives_per_label: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pools = pools_by_label(cedr_rows)
    records: list[dict[str, Any]] = []

    def add_contrastive(row: dict[str, Any], *, source_suffix: str, target_label: str, negative_labels: list[str]) -> None:
        positives = sample_pool(
            pools,
            target_label,
            rng=rng,
            count=positives_per_query,
            exclude_text=row["text"],
        )
        negatives = []
        for negative_label in negative_labels:
            if negative_label == target_label:
                continue
            negatives.extend(
                sample_pool(
                    pools,
                    negative_label,
                    rng=rng,
                    count=negatives_per_label,
                    exclude_text=row["text"],
                )
            )
        if not positives or not negatives:
            return
        records.append(
            {
                "source": f"CONTAMINATED_CEDR_DIAGNOSTIC:{source_suffix}",
                "objective": "contrastive",
                "query": prefixed(row["text"]),
                "positive": prefixed(positives[0]["text"]),
                "positives": [prefixed(item["text"]) for item in positives[1:]],
                "negatives": [prefixed(item["text"]) for item in negatives],
                "metadata": {
                    "contamination": "direct CEDR benchmark row/label and diagnostic flip set",
                    "cedr_split": row.get("split"),
                    "cedr_index": row.get("index"),
                    "true_label": row.get("true_label"),
                    "target_label": target_label,
                    "negative_labels": negative_labels,
                    "construction": source_suffix,
                },
            }
        )

    def add_neutral_prototype(row: dict[str, Any], *, source_suffix: str) -> None:
        prototypes = {
            label: [prefixed(item["text"]) for item in sample_pool(pools, label, rng=rng, count=4, exclude_text=row["text"])]
            for label in LABEL_TO_ID
        }
        records.append(
            {
                "source": f"CONTAMINATED_CEDR_DIAGNOSTIC:{source_suffix}",
                "objective": "prototype_none_classification",
                "query": prefixed(row["text"]),
                "label": NO_EMOTION,
                "prototypes": prototypes,
                "metadata": {
                    "contamination": "direct CEDR benchmark row/label and diagnostic flip set",
                    "cedr_split": row.get("split"),
                    "cedr_index": row.get("index"),
                    "true_label": row.get("true_label"),
                    "neutral_margin": 0.30,
                    "construction": source_suffix,
                },
            }
        )

    for row in flip_rows:
        true_label = row.get("true_label")
        base_prediction = row.get("models", {}).get("base_mixh_habrfull", {}).get("main_prediction", "neutral")
        if true_label == "neutral":
            negative_labels = [label for label in parse_prediction_labels(base_prediction) if label != NO_EMOTION]
            if not negative_labels:
                negative_labels = list(LABEL_TO_ID)
            add_contrastive(
                row,
                source_suffix="a050_fixed_neutral_contrastive",
                target_label=NO_EMOTION,
                negative_labels=negative_labels,
            )
            add_neutral_prototype(row, source_suffix="a050_fixed_neutral_prototype")
        else:
            target_labels = [label for label in str(true_label).split("+") if label in LABEL_TO_ID]
            for target_label in target_labels:
                negative_labels = [label for label in LABEL_TO_ID if label != target_label]
                negative_labels.append(NO_EMOTION)
                add_contrastive(
                    row,
                    source_suffix="a050_fixed_emotion_contrastive",
                    target_label=target_label,
                    negative_labels=negative_labels,
                )

    # Rows where a050 hurts base are useful as guard rails: preserve their
    # original CEDR labels so a neutral fix does not simply collapse emotions.
    for row in hurt_rows:
        true_label = row.get("true_label")
        if true_label == "neutral":
            add_contrastive(
                row,
                source_suffix="a050_hurts_neutral_guard",
                target_label=NO_EMOTION,
                negative_labels=list(LABEL_TO_ID),
            )
            continue
        for target_label in [label for label in str(true_label).split("+") if label in LABEL_TO_ID]:
            negative_labels = [label for label in LABEL_TO_ID if label != target_label]
            negative_labels.append(NO_EMOTION)
            add_contrastive(
                row,
                source_suffix="a050_hurts_emotion_guard",
                target_label=target_label,
                negative_labels=negative_labels,
            )

    rng.shuffle(records)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    sources = Counter(record["source"] for record in records)
    objectives = Counter(record["objective"] for record in records)
    targets = Counter(record.get("metadata", {}).get("target_label", record.get("label", "?")) for record in records)
    return {
        "records": len(records),
        "source_counts": dict(sources),
        "objective_counts": dict(objectives),
        "target_counts": dict(targets),
        "contamination": "YES: direct CEDR benchmark rows and diagnostic flip sets",
        "do_not_use_for_fair_results": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contaminated CEDR diagnostic dataset from row-level flip tracking.")
    parser.add_argument("--seed", type=int, default=9449)
    parser.add_argument("--positives-per-query", type=int, default=3)
    parser.add_argument("--negatives-per-label", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "CONTAMINATED_cedr_diagnostic_a050_fixed_neutral.jsonl")
    args = parser.parse_args()

    cedr_rows = load_cedr()
    flip_rows = read_jsonl(TRACKING_DIR / "a050_fixed_base.jsonl")
    hurt_rows = read_jsonl(TRACKING_DIR / "a050_hurts_base.jsonl")
    records = make_diagnostic_records(
        cedr_rows=cedr_rows,
        flip_rows=flip_rows,
        hurt_rows=hurt_rows,
        seed=args.seed,
        positives_per_query=args.positives_per_query,
        negatives_per_label=args.negatives_per_label,
    )
    write_jsonl(args.output, records)
    write_json(args.output.with_name(args.output.stem + "_summary.json"), summarize(records))
    print(args.output)
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
