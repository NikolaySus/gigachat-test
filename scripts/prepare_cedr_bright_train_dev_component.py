#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    CACHE_DIR,
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
)


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
GROUPS = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]
LABELS = ["joy", "sadness", "surprise", "fear", "anger"]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=DATA_DIR / "open_ru_1r_nc_cedr_brighter_rus_train_dev_clean.jsonl")
    parser.add_argument("--seed", type=int, default=822)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    ds = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "rus", cache_dir=str(CACHE_DIR))
    pools: dict[str, list[dict]] = defaultdict(list)
    skipped = Counter()
    seen = set()

    for split in ("train", "dev"):
        for index, row in enumerate(ds[split]):
            text = str(row["text"]).strip()
            norm = normalize_text(text)
            if len(norm) < 8:
                skipped["short"] += 1
                continue
            if norm in seen:
                skipped["duplicate"] += 1
                continue
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            labels = [label for label in LABELS if int(row.get(label, 0)) == 1]
            if int(row.get("disgust", 0)) == 1 and not labels:
                skipped["disgust_only"] += 1
                continue
            seen.add(norm)
            groups = labels or ["neutral"]
            for group in groups:
                pools[group].append(
                    {
                        "text": text,
                        "group": group,
                        "labels": labels,
                        "split": split,
                        "index": index,
                        "id": row.get("id"),
                    }
                )

    records = []
    for group in GROUPS:
        rng.shuffle(pools[group])
    for group, items in pools.items():
        for idx, item in enumerate(items):
            positives = [candidate for candidate in items if candidate is not item]
            if not positives:
                continue
            negatives = []
            for negative_group in GROUPS:
                if negative_group == group or not pools[negative_group]:
                    continue
                negatives.append(PREFIX + rng.choice(pools[negative_group])["text"])
            records.append(
                {
                    "source": "brighter-dataset/BRIGHTER-emotion-categories:rus_train_dev_clean",
                    "objective": "contrastive",
                    "query": PREFIX + item["text"],
                    "positive": PREFIX + rng.choice(positives)["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": group,
                        "labels": item["labels"],
                        "split": item["split"],
                        "index": item["index"],
                        "id": item["id"],
                    },
                }
            )
    rng.shuffle(records)
    write_jsonl(args.output_path, records)
    summary = {
        "records": len(records),
        "label_counts": dict(Counter(row["metadata"]["group"] for row in records)),
        "source_splits": ["train", "dev"],
        "skipped": dict(skipped),
        "seed": args.seed,
        "contamination_policy": "exact and near CEDR overlap removed",
    }
    args.output_path.with_suffix(args.output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
