from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

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
CEDR_PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
GROUPS = ["neutral", "joy", "sadness", "anger", "surprise", "fear"]
EMOTION_PRIOR = {"joy": 1569, "sadness": 1417, "surprise": 607, "fear": 589, "anger": 411}


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def target_counts(count: int, *, neutral_fraction: float = 0.5) -> dict[str, int]:
    if neutral_fraction < 0.0 or neutral_fraction >= 1.0:
        raise ValueError("--neutral-fraction must be in [0.0, 1.0)")
    neutral = int(round(count * neutral_fraction))
    emotion_total = count - neutral
    total_prior = sum(EMOTION_PRIOR.values())
    targets = {"neutral": neutral}
    allocated = 0
    remainders = []
    for label, prior in EMOTION_PRIOR.items():
        raw = emotion_total * prior / total_prior
        value = int(raw)
        targets[label] = value
        allocated += value
        remainders.append((raw - value, label))
    for _, label in sorted(remainders, reverse=True)[: emotion_total - allocated]:
        targets[label] += 1
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean AiLab GoEmotions-RU CEDR-shaped component.")
    parser.add_argument("--count", type=int, default=6400)
    parser.add_argument("--seed", type=int, default=741)
    parser.add_argument("--neutral-fraction", type=float, default=0.5)
    parser.add_argument("--dataset-name", default="AiLab-IMCS-UL/go_emotions-ru")
    parser.add_argument("--name", default="cedr_ailab_goemotions_ru_prior_neutral_6400")
    args = parser.parse_args()

    cedr_index = load_cedr_index()
    dataset = load_dataset(args.dataset_name, cache_dir=str(CACHE_DIR))
    label_names = dataset["train"].features["labels_ekman"].feature.names
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()
    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            labels = [label_names[label] for label in row["labels_ekman"]]
            labels = [label for label in labels if label in GROUPS]
            if len(labels) != 1:
                skipped["multilabel_or_unmapped"] += 1
                continue
            group = labels[0]
            text = clean_text(row["ru_text"])
            normalized = normalize_text(text)
            if len(normalized) < 8 or len(normalized) > 360:
                skipped["length"] += 1
                continue
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            pools[group].append({"split": split, "index": index, "text": text, "group": group})

    rng = random.Random(args.seed)
    targets = target_counts(args.count, neutral_fraction=args.neutral_fraction)
    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    for group, target in targets.items():
        pool = pools[group][:]
        rng.shuffle(pool)
        if len(pool) < target:
            raise ValueError(f"Not enough rows for {group}: need {target}, got {len(pool)}")
        selected_by_group[group] = pool[:target]

    records = []
    for group, items in selected_by_group.items():
        for item in items:
            positives = [candidate for candidate in items if candidate is not item]
            positive = rng.choice(positives)
            negatives = []
            for negative_group in GROUPS:
                if negative_group == group:
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(selected_by_group[negative_group])["text"])
            records.append(
                {
                    "source": f"{args.dataset_name}:cedr_prior_neutral",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": group,
                        "split": item["split"],
                        "index": item["index"],
                    },
                }
            )
    rng.shuffle(records)

    path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(path, records)
    write_json(
        path.with_name(path.stem + "_summary.json"),
        {
            "name": args.name,
            "requested": args.count,
            "kept": len(records),
            "targets": targets,
            "neutral_fraction": args.neutral_fraction,
            "available": {group: len(rows) for group, rows in pools.items()},
            "selected": dict(Counter(record["metadata"]["group"] for record in records)),
            "skipped": dict(skipped),
            "contamination_policy": "exact and near CEDR overlap removed",
            "source": args.dataset_name,
        },
    )
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
