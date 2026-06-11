from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_texts(row: dict[str, Any]) -> list[str]:
    texts = []
    for key in ("query", "positive", "sentence1", "sentence2", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value)
    return texts


def add_rows(grouped: dict[str, list[str]], path: Path, *, allowed: set[str]) -> dict[str, Any]:
    counts = Counter()
    seen = {(label, text) for label, values in grouped.items() for text in values}
    for row in read_jsonl(path):
        metadata = row.get("metadata") or {}
        label = metadata.get("group") or metadata.get("trigger_group") or row.get("label")
        label = "neutral" if label == "no_emotion" else str(label)
        if label not in allowed:
            continue
        for text in row_texts(row):
            key = (label, text)
            if key in seen:
                continue
            seen.add(key)
            grouped[label].append(text)
            counts[label] += 1
    return {"path": str(path.relative_to(ROOT)), "added": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CEDR-like clean kNN support/query episodes.")
    parser.add_argument("--name", default="cedr_knn_episode_proxy_clean_p5_3600")
    parser.add_argument("--count-per-class", type=int, default=600)
    parser.add_argument("--supports-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=981)
    parser.add_argument(
        "--neutral-path",
        action="append",
        type=Path,
        default=[
            DATA_DIR / "open_ru_1r_nc_cedr_lenta_emotion_mention_neutral_3000.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_scan500k_8000.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_lenta_negative_topic_neutral_reported_2400.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_blog_neutral_broad_1600.jsonl",
        ],
    )
    parser.add_argument(
        "--emotion-path",
        action="append",
        type=Path,
        default=[
            DATA_DIR / "open_ru_1r_nc_cedr_brighter_rus_train_dev_clean.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_semeval2025_rus_tracka_train_dev_clean.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
        ],
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    source_summaries = []
    for path in args.neutral_path:
        source_summaries.append(add_rows(grouped, path, allowed={"neutral"}))
    for path in args.emotion_path:
        source_summaries.append(add_rows(grouped, path, allowed=set(LABELS)))

    for label in LABELS:
        needed = args.count_per_class + args.supports_per_class + 1
        if len(grouped[label]) < needed:
            raise RuntimeError(f"Not enough {label}: {len(grouped[label])}, need {needed}")
        rng.shuffle(grouped[label])

    records: list[dict[str, Any]] = []
    for label in LABELS:
        queries = grouped[label][: args.count_per_class]
        for index, query in enumerate(queries):
            supports = {}
            for support_label in LABELS:
                pool = [text for text in grouped[support_label] if text != query]
                supports[support_label] = rng.sample(pool, args.supports_per_class)
            records.append(
                {
                    "source": "cedr_knn_episode_proxy:clean_non_cedr",
                    "objective": "knn_classification",
                    "query": query,
                    "label": label,
                    "supports": supports,
                    "metadata": {
                        "group": label,
                        "index": index,
                        "supports_per_class": args.supports_per_class,
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
            "labels": LABELS,
            "count_per_class": args.count_per_class,
            "supports_per_class": args.supports_per_class,
            "label_counts": dict(Counter(row["label"] for row in records)),
            "available_texts": {label: len(grouped[label]) for label in LABELS},
            "sources": source_summaries,
            "construction": "clean non-CEDR CEDR-like kNN episodes with 5 supports per class, including neutral supports",
            "contamination_policy": "inherits exact/near CEDR-overlap filtering from source components; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
