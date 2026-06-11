from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    CACHE_DIR,
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
CEDR_LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear"]
RUEMOTIONS_MAP = {
    "нейтральная": "neutral",
    "радость": "joy",
    "грусть": "sadness",
    "злость": "anger",
    "удивление": "surprise",
    "страх": "fear",
}
THERAPY_LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear"]
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_text(value: Any) -> str:
    text = URL_RE.sub(" ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def quality_ok(text: str, *, max_chars: int) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 12 or len(normalized) > max_chars:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.45 * max(1, len(text)):
        return False
    return True


def iter_rows(dataset_name: str, *, max_chars: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    skipped = Counter()
    dataset = load_dataset(dataset_name, cache_dir=str(CACHE_DIR))
    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            text = clean_text(row.get("text") or row.get("ru_text"))
            if not quality_ok(text, max_chars=max_chars):
                skipped["quality"] += 1
                continue

            label = None
            if dataset_name == "ClosRise/RuEmotions":
                label = RUEMOTIONS_MAP.get(str(row.get("emotion", "")).strip().lower())
            elif dataset_name == "VBadazhkov/therapy_emotions_ru":
                active = [name for name in THERAPY_LABELS if int(row.get(name, 0) or 0) == 1]
                if len(active) == 1:
                    label = active[0]
                else:
                    skipped["multilabel_or_unmapped"] += 1
            elif "labels_ekman" in row:
                feature = ds.features["labels_ekman"]
                if hasattr(feature, "names"):
                    label = feature.int2str(row["labels_ekman"])
                else:
                    skipped["unsupported_label_feature"] += 1
            elif "labels" in row and hasattr(ds.features.get("labels"), "feature"):
                skipped["list_label_schema_not_supported"] += 1
            else:
                skipped["unknown_schema"] += 1

            if label not in CEDR_LABELS:
                skipped["unmapped_label"] += 1
                continue
            rows.append({"text": text, "label": label, "split": split, "index": str(index)})
    return rows, {"skipped": dict(skipped)}


def build_records(
    selected_by_label: dict[str, list[dict[str, str]]],
    *,
    dataset_name: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    available_labels = [label for label in CEDR_LABELS if len(selected_by_label.get(label, [])) >= 2]
    for label in available_labels:
        for item in selected_by_label[label]:
            positives = [candidate for candidate in selected_by_label[label] if candidate is not item]
            negatives = []
            for negative_label in available_labels:
                if negative_label == label:
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(selected_by_label[negative_label])["text"])
            records.append(
                {
                    "source": f"{dataset_name}:cedr_hf_emotion_component",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positives)["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": label,
                        "split": item["split"],
                        "index": item["index"],
                    },
                }
            )
    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean Hugging Face Russian emotion data for CEDR probes.")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--name", default="cedr_hf_emotion_component")
    parser.add_argument("--max-per-label", type=int, default=800)
    parser.add_argument("--max-chars", type=int, default=420)
    parser.add_argument("--seed", type=int, default=951)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    summaries = []
    seen = set()
    skipped = Counter()
    for dataset_name in args.dataset:
        rows, summary = iter_rows(dataset_name, max_chars=args.max_chars)
        added = Counter()
        for row in rows:
            normalized = normalize_text(row["text"])
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(row["text"], cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            grouped[row["label"]].append({**row, "dataset": dataset_name})
            added[row["label"]] += 1
        summaries.append({"dataset": dataset_name, **summary, "added": dict(added)})

    selected_by_label: dict[str, list[dict[str, str]]] = {}
    for label, rows in grouped.items():
        rows = rows[:]
        rng.shuffle(rows)
        selected_by_label[label] = rows[: args.max_per_label]

    records = build_records(selected_by_label, dataset_name="+".join(args.dataset), rng=rng)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "datasets": args.dataset,
            "records": len(records),
            "max_per_label": args.max_per_label,
            "available_by_label": {label: len(rows) for label, rows in grouped.items()},
            "selected_by_label": {label: len(rows) for label, rows in selected_by_label.items()},
            "record_counts": dict(Counter(record["metadata"]["group"] for record in records)),
            "source_summaries": summaries,
            "skipped": dict(skipped),
            "construction": "CEDR-shaped clean HF Russian emotion component with same-label positives and cross-label negatives",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
