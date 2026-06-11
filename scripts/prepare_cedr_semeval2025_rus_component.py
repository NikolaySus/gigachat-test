from __future__ import annotations

import argparse
import json
import random
import re
import sys
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
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]
LABELS = ["anger", "fear", "joy", "sadness", "surprise"]


def clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def row_group(row: dict[str, Any]) -> str | None:
    labels = [label for label in LABELS if int(row.get(label, 0)) == 1]
    if not labels and int(row.get("disgust", 0)) == 0:
        return "neutral"
    if len(labels) == 1 and int(row.get("disgust", 0)) == 0:
        return labels[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean SemEval-2025 Russian CEDR-shaped component.")
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--name", default="cedr_semeval2025_rus_tracka_train_dev_clean")
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=360)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    dataset = load_dataset(
        "vgaraujov/semeval-2025-task11-track-a",
        "rus",
        cache_dir=str(CACHE_DIR),
    )
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen: set[str] = set()
    for split in ("train", "dev"):
        for index, row in enumerate(dataset[split]):
            group = row_group(row)
            if group is None:
                skipped["multilabel_disgust_or_unmapped"] += 1
                continue
            text = clean_text(row["text"])
            normalized = normalize_text(text)
            if len(normalized) < args.min_length or len(normalized) > args.max_length:
                skipped["length"] += 1
                continue
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            pools[group].append({"text": text, "group": group, "split": split, "index": index, "id": row["id"]})

    records = []
    for group, items in pools.items():
        if len(items) < 2:
            skipped[f"too_few_{group}"] += len(items)
            continue
        negative_groups = [candidate for candidate in GROUPS if candidate != group and pools[candidate]]
        for item in items:
            positives = [candidate for candidate in items if candidate is not item]
            negatives = [
                CEDR_PREFIX + rng.choice(pools[negative_group])["text"]
                for negative_group in negative_groups
            ]
            records.append(
                {
                    "source": "vgaraujov/semeval-2025-task11-track-a:rus_train_dev_cedr_clean",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positives)["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": group,
                        "split": item["split"],
                        "index": item["index"],
                        "id": item["id"],
                    },
                }
            )

    rng.shuffle(records)
    output_path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(output_path, records)
    output_path.with_name(output_path.stem + "_summary.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "source": "vgaraujov/semeval-2025-task11-track-a/rus",
                "splits": ["train", "dev"],
                "kept": len(records),
                "selected": dict(Counter(record["metadata"]["group"] for record in records)),
                "available": {group: len(items) for group, items in pools.items()},
                "skipped": dict(skipped),
                "contamination_policy": "exact and near CEDR overlap removed; CEDR records and SemEval test split are not used",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"prepared {output_path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
