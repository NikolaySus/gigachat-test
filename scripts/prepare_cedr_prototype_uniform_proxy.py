from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_cedr_goemotions_ru_component import GROUPS
from prepare_cedr_neutral_lexical_distractors import read_jsonl
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


EMOTION_LABELS = ["joy", "sadness", "surprise", "fear", "anger"]


def group_of(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata", {})
    return metadata.get("trigger_group") or metadata.get("topic_group") or metadata.get("group")


def text_of(row: dict[str, Any]) -> str:
    return row.get("query") or row.get("sentence1") or row.get("text") or ""


def collect_neutral(paths: list[Path]) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            text = text_of(row)
            if not text or text in seen:
                continue
            seen.add(text)
            metadata = row.get("metadata", {})
            label_group = metadata.get("group")
            trigger_group = metadata.get("trigger_group") or metadata.get("topic_group")
            if label_group == "neutral":
                pools[trigger_group if trigger_group in EMOTION_LABELS else "generic"].append(text)
            elif trigger_group in EMOTION_LABELS:
                pools[trigger_group].append(text)
            else:
                continue
    return pools


def collect_emotions(paths: list[Path]) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            group = row.get("metadata", {}).get("group")
            if group not in EMOTION_LABELS:
                continue
            text = text_of(row)
            if not text or text in seen:
                continue
            seen.add(text)
            pools[group].append(text)
    return pools


def sample_prototypes(
    pools: dict[str, list[str]],
    *,
    rng: random.Random,
    query: str,
    count: int,
) -> dict[str, list[str]]:
    prototypes = {}
    for label in EMOTION_LABELS:
        candidates = [text for text in pools[label] if text != query]
        prototypes[label] = rng.sample(candidates, min(count, len(candidates)))
    return prototypes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CEDR prototype-uniform proxy data.")
    parser.add_argument("--name", default="cedr_prototype_uniform_lenta_blog_go_p5_7200")
    parser.add_argument("--count", type=int, default=7200)
    parser.add_argument("--neutral-fraction", type=float, default=0.6)
    parser.add_argument("--prototypes-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3011)
    parser.add_argument(
        "--neutral-path",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--emotion-path",
        action="append",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    neutral_paths = args.neutral_path or [
        DATA_DIR / "open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_scan500k_8000.jsonl",
        DATA_DIR / "open_ru_1r_nc_cedr_lenta_negative_topic_neutral_reported_2400.jsonl",
        DATA_DIR / "open_ru_1r_nc_cedr_blog_neutral_broad_1600.jsonl",
        DATA_DIR / "open_ru_1r_nc_cedr_lenta_positive_descriptor_neutral_broad1600_scan150k.jsonl",
    ]
    emotion_paths = args.emotion_path or [
        DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
        DATA_DIR / "open_ru_1r_nc_cedr_seara_rugoemotions_strict_prior9000.jsonl",
    ]
    neutral_pools = collect_neutral(neutral_paths)
    emotion_pools = collect_emotions(emotion_paths)
    missing = [label for label in EMOTION_LABELS if len(emotion_pools[label]) < args.prototypes_per_class + 2]
    if missing:
        raise RuntimeError(f"Not enough emotion prototypes for {missing}")

    neutral_items = []
    for group, texts in neutral_pools.items():
        for text in texts:
            neutral_items.append((group, text))
    rng.shuffle(neutral_items)

    neutral_target = int(args.count * args.neutral_fraction)
    emotion_target = args.count - neutral_target
    records: list[dict[str, Any]] = []

    for index, (group, query) in enumerate(neutral_items[:neutral_target]):
        records.append(
            {
                "source": "cedr_prototype_uniform_proxy:neutral_uniform",
                "objective": "prototype_uniform_classification",
                "query": query,
                "label": "neutral",
                "prototypes": sample_prototypes(
                    emotion_pools,
                    rng=rng,
                    query=query,
                    count=args.prototypes_per_class,
                ),
                "metadata": {
                    "group": "neutral",
                    "trigger_group": group,
                    "index": index,
                    "construction": "neutral_query_uniform_over_emotion_prototypes",
                },
            }
        )

    per_emotion = emotion_target // len(EMOTION_LABELS)
    remainder = emotion_target % len(EMOTION_LABELS)
    for label_index, label in enumerate(EMOTION_LABELS):
        target = per_emotion + (1 if label_index < remainder else 0)
        items = emotion_pools[label][:]
        rng.shuffle(items)
        for index, query in enumerate(items[:target]):
            records.append(
                {
                    "source": "cedr_prototype_uniform_proxy:emotion_ce",
                    "objective": "prototype_uniform_classification",
                    "query": query,
                    "label": label,
                    "prototypes": sample_prototypes(
                        emotion_pools,
                        rng=rng,
                        query=query,
                        count=args.prototypes_per_class,
                    ),
                    "metadata": {
                        "group": label,
                        "index": index,
                        "construction": "emotion_query_cross_entropy_over_emotion_prototypes",
                    },
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
            "requested": args.count,
            "neutral_fraction": args.neutral_fraction,
            "prototypes_per_class": args.prototypes_per_class,
            "source_counts": dict(Counter(row["source"] for row in records)),
            "group_counts": dict(Counter(row["metadata"]["group"] for row in records)),
            "neutral_pool_counts": {group: len(texts) for group, texts in neutral_pools.items()},
            "emotion_pool_counts": {group: len(texts) for group, texts in emotion_pools.items()},
            "neutral_paths": [str(path) for path in neutral_paths],
            "emotion_paths": [str(path) for path in emotion_paths],
            "construction": "neutral rows use KL-to-uniform over emotion prototypes; emotion rows use CE over the same prototype classes",
            "contamination_policy": "inherits exact/near CEDR filtering from source components; no CEDR records used",
        },
    )
    print(f"prepared {out}: {len(records)} rows")


if __name__ == "__main__":
    main()
