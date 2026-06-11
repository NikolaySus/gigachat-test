from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
EMOTION_LABELS = ["joy", "sadness", "surprise", "fear", "anger"]


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
    for field in ("query", "positive", "sentence1", "sentence2", "text"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            texts.append(value)
    return texts


def add_group_texts(
    grouped: dict[str, list[str]],
    path: Path,
    *,
    allowed: set[str],
    group_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    aliases = group_aliases or {}
    seen = {(label, text) for label, values in grouped.items() for text in values}
    counts = Counter()
    for row in read_jsonl(path):
        metadata = row.get("metadata", {})
        group = metadata.get("group") or metadata.get("trigger_group") or row.get("label")
        group = aliases.get(str(group), str(group))
        if group not in allowed:
            continue
        for text in row_texts(row):
            key = (group, text)
            if key in seen:
                continue
            seen.add(key)
            grouped[group].append(text)
            counts[group] += 1
    return {"path": str(path), "added": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CEDR proxy episodes with neutral as no-emotion class.")
    parser.add_argument("--name", default="cedr_prototype_none_lenta_seara_go_p5_4800")
    parser.add_argument("--count-per-class", type=int, default=800)
    parser.add_argument("--prototypes-per-class", type=int, default=5)
    parser.add_argument("--neutral-margin", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1191)
    parser.add_argument(
        "--neutral-path",
        action="append",
        type=Path,
        default=[
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
            DATA_DIR / "open_ru_1r_nc_cedr_seara_rugoemotions_strict_prior9000.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
        ],
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    source_summaries = []
    for path in args.neutral_path:
        source_summaries.append(add_group_texts(grouped, path, allowed={"neutral"}))
    for path in args.emotion_path:
        source_summaries.append(add_group_texts(grouped, path, allowed=set(EMOTION_LABELS)))

    labels = ["neutral", *EMOTION_LABELS]
    for label in labels:
        needed = args.count_per_class + (args.prototypes_per_class if label != "neutral" else 0)
        if len(grouped[label]) < needed:
            raise RuntimeError(f"Not enough rows for {label}: {len(grouped[label])}, need {needed}")
        rng.shuffle(grouped[label])

    records: list[dict[str, Any]] = []
    for label in labels:
        for index, query in enumerate(grouped[label][: args.count_per_class]):
            prototypes = {}
            for prototype_label in EMOTION_LABELS:
                pool = [text for text in grouped[prototype_label] if text != query]
                prototypes[prototype_label] = rng.sample(pool, args.prototypes_per_class)
            records.append(
                {
                    "source": "cedr_prototype_none:lenta_seara_go",
                    "objective": "prototype_none_classification",
                    "query": query,
                    "label": label,
                    "prototypes": prototypes,
                    "metadata": {
                        "group": label,
                        "index": index,
                        "prototypes_per_class": args.prototypes_per_class,
                        "neutral_margin": args.neutral_margin,
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
            "labels": labels,
            "emotion_labels": EMOTION_LABELS,
            "count_per_class": args.count_per_class,
            "prototypes_per_class": args.prototypes_per_class,
            "neutral_margin": args.neutral_margin,
            "label_counts": dict(Counter(record["label"] for record in records)),
            "available_texts": {label: len(grouped[label]) for label in labels},
            "sources": source_summaries,
            "construction": "emotion queries classify against emotion prototypes; neutral queries are trained below all emotion prototype similarities",
            "contamination_policy": "inherits source CEDR-overlap filtering; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
