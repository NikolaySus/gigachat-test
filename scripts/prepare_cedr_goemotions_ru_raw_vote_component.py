from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, EMOTION_PRIOR, GROUPS, target_counts
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
FINE_TO_EKMAN = {
    "admiration": "joy",
    "amusement": "joy",
    "approval": "joy",
    "caring": "joy",
    "desire": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "joy": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "fear": "fear",
    "nervousness": "fear",
    "sadness": "sadness",
    "disappointment": "sadness",
    "embarrassment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "surprise": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    "realization": "surprise",
    "neutral": "neutral",
}


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean CEDR-shaped data from raw rater-level ru_go_emotions.")
    parser.add_argument("--count", type=int, default=6400)
    parser.add_argument("--neutral-fraction", type=float, default=0.5)
    parser.add_argument("--min-top-votes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=791)
    parser.add_argument("--name", default="cedr_seara_raw_vote2_winner_prior_neutral_6400")
    args = parser.parse_args()

    ds = load_dataset("seara/ru_go_emotions", "raw", cache_dir=str(CACHE_DIR))["train"]
    cedr_index = load_cedr_index()
    by_id: dict[str, dict[str, Any]] = defaultdict(lambda: {"counts": Counter(), "raters": 0, "text": "", "ru_text": ""})
    for row in ds:
        item = by_id[row["id"]]
        item["raters"] += 1
        item["text"] = row["text"]
        item["ru_text"] = clean_text(row["ru_text"])
        row_groups = set()
        for fine_label, group in FINE_TO_EKMAN.items():
            if row.get(fine_label, 0):
                row_groups.add(group)
        for group in row_groups:
            item["counts"][group] += 1

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()
    for source_id, item in by_id.items():
        top = item["counts"].most_common()
        if not top or top[0][1] < args.min_top_votes:
            skipped["low_agreement"] += 1
            continue
        if len(top) > 1 and top[0][1] <= top[1][1]:
            skipped["tied_top"] += 1
            continue
        group = top[0][0]
        if group not in GROUPS:
            skipped["unmapped_group"] += 1
            continue
        text = item["ru_text"]
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
        pools[group].append(
            {
                "id": source_id,
                "text": text,
                "group": group,
                "votes": dict(item["counts"]),
                "raters": item["raters"],
            }
        )

    rng = random.Random(args.seed)
    targets = target_counts(args.count, neutral_fraction=args.neutral_fraction)
    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    for group, target in targets.items():
        pool = pools[group][:]
        pool.sort(key=lambda row: (-row["votes"][group], -row["raters"], rng.random()))
        if len(pool) < target:
            raise ValueError(f"Not enough rows for {group}: need {target}, got {len(pool)}")
        selected_by_group[group] = pool[:target]

    records = []
    for group, items in selected_by_group.items():
        for item in items:
            positive = rng.choice([candidate for candidate in items if candidate is not item])
            negatives = [
                CEDR_PREFIX + rng.choice(selected_by_group[negative_group])["text"]
                for negative_group in GROUPS
                if negative_group != group
            ]
            records.append(
                {
                    "source": "seara/ru_go_emotions:raw_vote_winner",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": group,
                        "id": item["id"],
                        "votes": item["votes"],
                        "raters": item["raters"],
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
            "min_top_votes": args.min_top_votes,
            "available": {group: len(rows) for group, rows in pools.items()},
            "selected": dict(Counter(record["metadata"]["group"] for record in records)),
            "skipped": dict(skipped),
            "fine_to_ekman": FINE_TO_EKMAN,
            "emotion_prior": EMOTION_PRIOR,
            "contamination_policy": "exact and near CEDR overlap removed",
            "source": "seara/ru_go_emotions raw",
        },
    )
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
