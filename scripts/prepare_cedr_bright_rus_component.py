from __future__ import annotations

import argparse
import json
import random
import re
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
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
CEDR_LABELS = ["joy", "sadness", "surprise", "fear", "anger"]


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def cedr_labelset(row: dict[str, Any], *, drop_disgust: bool) -> tuple[str, ...] | None:
    labels = tuple(label for label in CEDR_LABELS if int(row.get(label, 0)) > 0)
    has_disgust = int(row.get("disgust", 0)) > 0
    if has_disgust and drop_disgust and not labels:
        return None
    return labels


def label_key(labels: tuple[str, ...]) -> str:
    return "neutral" if not labels else "+".join(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean BRIGHTER/SemEval Russian CEDR-shaped component.")
    parser.add_argument("--dataset-name", default="brighter-dataset/BRIGHTER-emotion-categories")
    parser.add_argument("--config", default="rus")
    parser.add_argument("--splits", default="train,dev,test")
    parser.add_argument("--name", default="cedr_brighter_rus_categories_clean")
    parser.add_argument("--seed", type=int, default=893)
    parser.add_argument("--drop-disgust", action="store_true", default=True)
    parser.add_argument("--max-records", type=int, default=6000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    dataset = load_dataset(args.dataset_name, args.config, cache_dir=str(CACHE_DIR))
    pools: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen: set[str] = set()

    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        if split not in dataset:
            skipped[f"missing_split:{split}"] += 1
            continue
        for index, row in enumerate(dataset[split]):
            labels = cedr_labelset(row, drop_disgust=args.drop_disgust)
            if labels is None:
                skipped["disgust_only"] += 1
                continue
            text = clean_text(row["text"])
            normalized = normalize_text(text)
            if len(normalized) < 6 or len(normalized) > 360:
                skipped["length"] += 1
                continue
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            pools[labels].append(
                {
                    "text": text,
                    "labels": labels,
                    "split": split,
                    "index": index,
                    "id": row.get("id"),
                }
            )

    usable_labelsets = [labels for labels, rows in pools.items() if len(rows) >= 2]
    records: list[dict[str, Any]] = []
    for labels in usable_labelsets:
        same_pool = pools[labels]
        negative_pools = [
            (other_labels, rows)
            for other_labels, rows in pools.items()
            if other_labels != labels and not (set(other_labels) & set(labels))
        ]
        if not negative_pools:
            continue
        for item in same_pool:
            positive = rng.choice([candidate for candidate in same_pool if candidate is not item])
            negatives = []
            rng.shuffle(negative_pools)
            for _, rows in negative_pools[:5]:
                negatives.append(CEDR_PREFIX + rng.choice(rows)["text"])
            records.append(
                {
                    "source": f"{args.dataset_name}:{args.config}:cedr_clean",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives,
                    "metadata": {
                        "labels": list(labels),
                        "group": label_key(labels),
                        "split": item["split"],
                        "index": item["index"],
                        "id": item["id"],
                    },
                }
            )

    rng.shuffle(records)
    if args.max_records > 0:
        records = records[: args.max_records]

    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "dataset": args.dataset_name,
            "config": args.config,
            "splits": args.splits,
            "records": len(records),
            "available_labelsets": {label_key(labels): len(rows) for labels, rows in pools.items()},
            "selected_labelsets": dict(Counter(row["metadata"]["group"] for row in records)),
            "skipped": dict(skipped),
            "drop_disgust": args.drop_disgust,
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
