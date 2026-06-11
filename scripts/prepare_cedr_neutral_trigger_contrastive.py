from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_cedr_goemotions_ru_component import GROUPS
from prepare_cedr_neutral_lexical_distractors import read_jsonl
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


EMOTION_GROUPS = ["joy", "sadness", "surprise", "fear", "anger"]


def text_of(row: dict[str, Any]) -> str:
    return row.get("query") or row.get("sentence1") or row.get("text") or ""


def group_of(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata", {})
    return metadata.get("trigger_group") or metadata.get("group")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build contrastive hard-neutral CEDR proxy with same-trigger emotion negatives."
    )
    parser.add_argument("--name", default="cedr_neutral_trigger_contrastive_9000")
    parser.add_argument("--count", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=1151)
    parser.add_argument("--positives-per-row", type=int, default=3)
    parser.add_argument("--negatives-per-row", type=int, default=5)
    parser.add_argument(
        "--neutral-path",
        action="append",
        type=Path,
        default=[
            DATA_DIR / "open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_scan500k_8000.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_lenta_negative_topic_neutral_reported_2400.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_blog_neutral_broad_1600.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_synthetic_neutral_boundary_v2_3600.jsonl",
        ],
    )
    parser.add_argument(
        "--emotion-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    neutral_pools: dict[str, list[str]] = defaultdict(list)
    seen_neutral = set()
    for path in args.neutral_path:
        for row in read_jsonl(path):
            group = group_of(row)
            text = text_of(row)
            if group in GROUPS and group != "neutral" and text and text not in seen_neutral:
                seen_neutral.add(text)
                neutral_pools[group].append(text)

    emotion_pools: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(args.emotion_path):
        group = row.get("metadata", {}).get("group")
        text = text_of(row)
        if group in EMOTION_GROUPS and text:
            emotion_pools[group].append(text)

    groups = [group for group in EMOTION_GROUPS if neutral_pools[group] and emotion_pools[group]]
    if not groups:
        raise RuntimeError("No usable neutral/emotion groups found")

    for pool in [*neutral_pools.values(), *emotion_pools.values()]:
        rng.shuffle(pool)

    records = []
    per_group = args.count // len(groups)
    for group in groups:
        neutral_items = neutral_pools[group]
        same_emotion = emotion_pools[group]
        other_neutral = [
            text
            for other_group in groups
            if other_group != group
            for text in neutral_pools[other_group]
        ]
        other_emotion = [
            text
            for other_group in groups
            if other_group != group
            for text in emotion_pools[other_group]
        ]
        if not other_neutral:
            raise RuntimeError(f"No cross-trigger neutral positives for {group}")
        rng.shuffle(other_neutral)
        rng.shuffle(other_emotion)

        for i in range(per_group):
            query = neutral_items[i % len(neutral_items)]
            positives = [
                other_neutral[(i * args.positives_per_row + j) % len(other_neutral)]
                for j in range(args.positives_per_row)
            ]
            negatives = [
                same_emotion[(i * args.negatives_per_row + j) % len(same_emotion)]
                for j in range(max(1, args.negatives_per_row - 2))
            ]
            negatives.extend(
                other_emotion[(i * args.negatives_per_row + j) % len(other_emotion)]
                for j in range(min(2, args.negatives_per_row))
            )
            records.append(
                {
                    "source": "cedr_neutral_trigger_contrastive:neutral_cross_trigger",
                    "objective": "contrastive",
                    "query": query,
                    "positive": positives[0],
                    "positives": positives[1:],
                    "negatives": negatives,
                    "metadata": {"group": "neutral", "trigger_group": group},
                }
            )

    rng.shuffle(records)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "records": len(records),
            "seed": args.seed,
            "positives_per_row": args.positives_per_row,
            "negatives_per_row": args.negatives_per_row,
            "group_counts": dict(Counter(row["metadata"]["trigger_group"] for row in records)),
            "neutral_paths": [str(path) for path in args.neutral_path],
            "emotion_path": str(args.emotion_path),
            "construction": "hard neutral query; cross-trigger neutral positives; same-trigger emotion negatives plus two off-trigger emotion negatives",
            "contamination_policy": "uses previously audited non-CEDR mined neutral sources plus clean GoEmotions-RU proxy; no CEDR records are inserted",
        },
    )
    print(f"prepared {out}: {len(records)} rows")


if __name__ == "__main__":
    main()
