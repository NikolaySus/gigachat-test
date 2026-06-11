from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    CACHE_DIR,
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
)


ROOT = Path(__file__).resolve().parents[1]
LABEL_ORDER = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]
SOURCE_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def labelset(raw_labels: Any, *, single_label_only: bool) -> str | None:
    if raw_labels is None:
        return None
    labels = []
    for raw_label in raw_labels:
        label = SOURCE_LABELS[int(raw_label)]
        if label == "disgust":
            continue
        labels.append(label)
    labels = sorted(set(labels), key=LABEL_ORDER.index)
    if not labels:
        return None
    if "neutral" in labels and len(labels) > 1:
        labels = [label for label in labels if label != "neutral"]
    if single_label_only and len(labels) != 1:
        return None
    return "+".join(labels)


def clean_text(text: str) -> str:
    text = " ".join(str(text or "").replace("\u00a0", " ").split())
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return text


def text_ok(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 12 or len(normalized) > 260:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.45 * max(1, len(text)):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clean CEDR-style prototype episodes from SkyWater21/ru_emotions."
    )
    parser.add_argument("--name", default="cedr_skywater_ruemotions_proto_p5_4800")
    parser.add_argument("--count", type=int, default=4800)
    parser.add_argument("--prototypes-per-class", type=int, default=5)
    parser.add_argument("--pool-per-label", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--single-label-only", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["comb_train", "comb_validation"])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    dataset = load_dataset("SkyWater21/ru_emotions", cache_dir=str(CACHE_DIR))
    pools: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    skipped = Counter()
    enough = lambda: all(len(pools[label]) >= args.pool_per_label for label in LABEL_ORDER)

    for split in args.splits:
        for row in dataset[split]:
            if enough():
                break
            text = clean_text(row.get("ru_text") or row.get("text") or "")
            label = labelset(row.get("labels"), single_label_only=args.single_label_only)
            if not label:
                skipped["no_label"] += 1
                continue
            if label not in LABEL_ORDER:
                skipped["multi_label"] += 1
                continue
            if not text_ok(text):
                skipped["quality"] += 1
                continue
            normalized = normalize_text(text)
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            pools[label].append(CEDR_PREFIX + text)
        if enough():
            break

    labels = [label for label in LABEL_ORDER if len(pools[label]) > args.prototypes_per_class]
    if len(labels) < 6:
        raise RuntimeError(f"Need all six CEDR labels, got {labels}")

    per_label = args.count // len(labels)
    remainder = args.count % len(labels)
    query_items: list[tuple[str, str]] = []
    selected_counts = {}
    for label_index, label in enumerate(labels):
        target = per_label + (1 if label_index < remainder else 0)
        rows = pools[label][:]
        rng.shuffle(rows)
        selected = rows[: min(target, len(rows))]
        selected_counts[label] = len(selected)
        query_items.extend((label, text) for text in selected)
    rng.shuffle(query_items)

    records = []
    for index, (label, query) in enumerate(query_items):
        prototypes = {}
        for prototype_label in labels:
            pool = [text for text in pools[prototype_label] if text != query]
            prototypes[prototype_label] = rng.sample(pool, args.prototypes_per_class)
        records.append(
            {
                "source": f"SkyWater21/ru_emotions:prototype_p{args.prototypes_per_class}",
                "objective": "prototype_classification",
                "query": query,
                "label": label,
                "prototypes": prototypes,
                "metadata": {
                    "group": label,
                    "index": index,
                    "prototypes_per_class": args.prototypes_per_class,
                    "source_dataset": "SkyWater21/ru_emotions",
                },
            }
        )

    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    summary = {
        "name": args.name,
        "records": len(records),
        "requested": args.count,
        "labels": labels,
        "selected_counts": selected_counts,
        "available_counts": {label: len(pools[label]) for label in labels},
        "prototypes_per_class": args.prototypes_per_class,
        "pool_per_label": args.pool_per_label,
        "single_label_only": args.single_label_only,
        "splits": args.splits,
        "skipped": dict(skipped),
        "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
    }
    out.with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
