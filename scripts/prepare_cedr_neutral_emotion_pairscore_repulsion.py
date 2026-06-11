from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_cedr_goemotions_ru_component import GROUPS
from prepare_cedr_neutral_lexical_distractors import read_jsonl
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


def group_of(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata", {})
    return metadata.get("trigger_group") or metadata.get("group")


def text_of(row: dict[str, Any]) -> str:
    return row.get("query") or row.get("sentence1") or row.get("text") or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pair-score neutral-vs-emotion repulsion data for CEDR.")
    parser.add_argument("--name", default="cedr_neutral_emotion_pairscore_repulsion_7200")
    parser.add_argument("--count", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=1131)
    parser.add_argument("--neutral-score", type=float, default=0.05)
    parser.add_argument("--emotion-score", type=float, default=0.85)
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
    for path in args.neutral_path:
        for row in read_jsonl(path):
            group = group_of(row)
            if group in GROUPS and group != "neutral":
                text = text_of(row)
                if text:
                    neutral_pools[group].append(text)

    emotion_rows = read_jsonl(args.emotion_path)
    emotion_pools: dict[str, list[str]] = defaultdict(list)
    for row in emotion_rows:
        group = row.get("metadata", {}).get("group")
        if group in GROUPS and group != "neutral":
            text = text_of(row)
            if text:
                emotion_pools[group].append(text)

    records = []
    groups = [group for group in ["joy", "sadness", "surprise", "fear", "anger"] if neutral_pools[group] and emotion_pools[group]]
    if not groups:
        raise RuntimeError("No usable groups found")
    per_group = args.count // len(groups)
    for group in groups:
        neutral_items = neutral_pools[group][:]
        emotion_items = emotion_pools[group][:]
        rng.shuffle(neutral_items)
        rng.shuffle(emotion_items)
        neutral_target = per_group * 3 // 4
        emotion_target = per_group - neutral_target

        for i in range(neutral_target):
            records.append(
                {
                    "source": "cedr_neutral_emotion_pairscore_repulsion:neutral_low",
                    "objective": "pair_score",
                    "sentence1": neutral_items[i % len(neutral_items)],
                    "sentence2": emotion_items[i % len(emotion_items)],
                    "score": args.neutral_score,
                    "metadata": {"group": "neutral", "trigger_group": group},
                }
            )

        for i in range(emotion_target):
            records.append(
                {
                    "source": "cedr_neutral_emotion_pairscore_repulsion:emotion_high",
                    "objective": "pair_score",
                    "sentence1": emotion_items[i % len(emotion_items)],
                    "sentence2": emotion_items[(i + 17) % len(emotion_items)],
                    "score": args.emotion_score,
                    "metadata": {"group": group},
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
            "neutral_score": args.neutral_score,
            "emotion_score": args.emotion_score,
            "neutral_paths": [str(path) for path in args.neutral_path],
            "emotion_path": str(args.emotion_path),
            "group_counts": dict(Counter(row["metadata"].get("trigger_group") or row["metadata"].get("group") for row in records)),
            "source_counts": dict(Counter(row["source"] for row in records)),
            "construction": "pair_score rows: hard neutral text vs same-trigger emotion text low similarity; same-emotion text pairs high similarity",
            "contamination_policy": "inherits exact/near CEDR filtering from mined neutral components and clean GoEmotions-RU; synthetic rows contain no CEDR records",
        },
    )
    print(f"prepared {out}: {len(records)} rows")


if __name__ == "__main__":
    main()
