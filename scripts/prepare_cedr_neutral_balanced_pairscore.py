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
    parser = argparse.ArgumentParser(description="Build balanced neutral-to-all-emotions pair-score CEDR proxy.")
    parser.add_argument("--name", default="cedr_neutral_balanced_pairscore_10000")
    parser.add_argument("--neutral-count", type=int, default=1600)
    parser.add_argument("--same-emotion-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1141)
    parser.add_argument("--neutral-emotion-score", type=float, default=0.52)
    parser.add_argument("--same-emotion-score", type=float, default=0.85)
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
    neutral_items = []
    seen = set()
    for path in args.neutral_path:
        for row in read_jsonl(path):
            text = text_of(row)
            if not text or text in seen:
                continue
            group = group_of(row)
            if group not in GROUPS or group == "neutral":
                continue
            seen.add(text)
            neutral_items.append({"text": text, "trigger_group": group})
    rng.shuffle(neutral_items)
    neutral_items = neutral_items[: args.neutral_count]

    emotion_rows = read_jsonl(args.emotion_path)
    emotion_pools: dict[str, list[str]] = defaultdict(list)
    for row in emotion_rows:
        group = row.get("metadata", {}).get("group")
        if group in EMOTION_GROUPS:
            text = text_of(row)
            if text:
                emotion_pools[group].append(text)
    if any(not emotion_pools[group] for group in EMOTION_GROUPS):
        raise RuntimeError("Missing one or more emotion pools")

    records = []
    for item in neutral_items:
        for group in EMOTION_GROUPS:
            pool = emotion_pools[group]
            records.append(
                {
                    "source": "cedr_neutral_balanced_pairscore:neutral_to_all",
                    "objective": "pair_score",
                    "sentence1": item["text"],
                    "sentence2": rng.choice(pool),
                    "score": args.neutral_emotion_score,
                    "metadata": {
                        "group": "neutral",
                        "trigger_group": item["trigger_group"],
                        "paired_emotion_group": group,
                    },
                }
            )

    per_group_same = args.same_emotion_count // len(EMOTION_GROUPS)
    for group in EMOTION_GROUPS:
        pool = emotion_pools[group][:]
        rng.shuffle(pool)
        for i in range(per_group_same):
            records.append(
                {
                    "source": "cedr_neutral_balanced_pairscore:same_emotion_high",
                    "objective": "pair_score",
                    "sentence1": pool[i % len(pool)],
                    "sentence2": pool[(i + 31) % len(pool)],
                    "score": args.same_emotion_score,
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
            "neutral_count": len(neutral_items),
            "neutral_emotion_score": args.neutral_emotion_score,
            "same_emotion_score": args.same_emotion_score,
            "source_counts": dict(Counter(row["source"] for row in records)),
            "paired_emotion_counts": dict(
                Counter(row["metadata"].get("paired_emotion_group") for row in records if row["source"].endswith("neutral_to_all"))
            ),
            "neutral_paths": [str(path) for path in args.neutral_path],
            "emotion_path": str(args.emotion_path),
            "construction": "each hard neutral-like text is paired at equal moderate similarity with every emotion group; same-emotion pairs preserve class cohesion",
            "contamination_policy": "inherits exact/near CEDR filtering from mined neutral components and clean GoEmotions-RU; synthetic rows contain no CEDR records",
        },
    )
    print(f"prepared {out}: {len(records)} rows")


if __name__ == "__main__":
    main()
