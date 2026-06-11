#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
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
ID_TO_GROUP = {
    0: "joy",
    2: "surprise",
    3: "sadness",
    4: "anger",
    6: "fear",
    8: "neutral",
}
GROUPS = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]
CEDR_PRIOR = {
    "neutral": 3043,
    "joy": 1569,
    "sadness": 1417,
    "surprise": 607,
    "fear": 589,
    "anger": 411,
}


def parse_labels(value: str) -> list[int]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, int):
        return [parsed]
    return [int(item) for item in parsed]


def target_counts(total: int) -> dict[str, int]:
    prior_sum = sum(CEDR_PRIOR.values())
    counts = {group: math.floor(total * CEDR_PRIOR[group] / prior_sum) for group in GROUPS}
    remainder = total - sum(counts.values())
    order = sorted(GROUPS, key=lambda group: total * CEDR_PRIOR[group] / prior_sum - counts[group], reverse=True)
    for group in order[:remainder]:
        counts[group] += 1
    return counts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=825)
    parser.add_argument("--output-path", type=Path, default=DATA_DIR / "open_ru_1r_nc_cedr_djacon_rugoemotions_prior9000.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    ds = load_dataset("Djacon/ru_goemotions", cache_dir=str(CACHE_DIR))
    pools: dict[str, list[dict]] = defaultdict(list)
    skipped = Counter()
    seen = set()

    for split in ("train", "validation"):
        for index, row in enumerate(ds[split]):
            labels = parse_labels(row["labels"])
            mapped = [ID_TO_GROUP[label] for label in labels if label in ID_TO_GROUP]
            mapped = sorted(set(mapped))
            if len(mapped) != 1:
                skipped["multilabel_or_unmapped"] += 1
                continue
            group = mapped[0]
            text = str(row["text"]).strip()
            norm = normalize_text(text)
            if len(norm) < 8 or len(norm) > 360:
                skipped["length"] += 1
                continue
            if norm in seen:
                skipped["duplicate"] += 1
                continue
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            seen.add(norm)
            pools[group].append({"text": text, "group": group, "split": split, "index": index, "labels": labels})

    targets = target_counts(args.count)
    selected_by_group: dict[str, list[dict]] = {}
    capped = {}
    for group in GROUPS:
        pool = pools[group][:]
        rng.shuffle(pool)
        take = min(targets[group], len(pool))
        if take < targets[group]:
            capped[group] = {"target": targets[group], "available": len(pool)}
        selected_by_group[group] = pool[:take]
    while sum(len(v) for v in selected_by_group.values()) < args.count:
        progressed = False
        for group in sorted(GROUPS, key=lambda g: len(pools[g]) - len(selected_by_group[g]), reverse=True):
            if len(selected_by_group[group]) < len(pools[group]):
                selected_by_group[group].append(pools[group][len(selected_by_group[group])])
                progressed = True
                if sum(len(v) for v in selected_by_group.values()) >= args.count:
                    break
        if not progressed:
            break

    records = []
    for group, items in selected_by_group.items():
        for item in items:
            positives = [candidate for candidate in items if candidate is not item]
            if not positives:
                continue
            negatives = []
            for neg_group in GROUPS:
                if neg_group == group:
                    continue
                negatives.append(PREFIX + rng.choice(selected_by_group[neg_group])["text"])
            records.append(
                {
                    "source": "Djacon/ru_goemotions:cedr_prior_trainval_clean",
                    "objective": "contrastive",
                    "query": PREFIX + item["text"],
                    "positive": PREFIX + rng.choice(positives)["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": group,
                        "split": item["split"],
                        "index": item["index"],
                        "labels": item["labels"],
                    },
                }
            )
    rng.shuffle(records)
    write_jsonl(args.output_path, records)
    summary = {
        "records": len(records),
        "requested": args.count,
        "targets": targets,
        "selected": dict(Counter(row["metadata"]["group"] for row in records)),
        "available": {group: len(pools[group]) for group in GROUPS},
        "capped": capped,
        "skipped": dict(skipped),
        "source_splits": ["train", "validation"],
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
