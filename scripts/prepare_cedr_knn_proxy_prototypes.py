from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_jsonl, write_json


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


def add_group_texts(grouped: dict[str, list[str]], path: Path, *, allowed: set[str]) -> dict[str, Any]:
    seen = {(label, text) for label, values in grouped.items() for text in values}
    counts = Counter()
    for row in read_jsonl(path):
        group = row.get("metadata", {}).get("group")
        if group == "no_emotion":
            group = "neutral"
        if group not in allowed:
            continue
        for field in ("query", "positive"):
            text = row.get(field)
            if not isinstance(text, str) or not text.strip():
                continue
            key = (group, text)
            if key in seen:
                continue
            seen.add(key)
            grouped[group].append(text)
            counts[group] += 1
    return {"path": str(path), "added": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean CEDR KNN proxy prototype-classification data.")
    parser.add_argument("--name", default="cedr_knn_proxy_lenta_neutral_surprise_go_p5_3600")
    parser.add_argument("--count-per-class", type=int, default=600)
    parser.add_argument("--prototypes-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=923)
    parser.add_argument(
        "--neutral-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_allavail.jsonl",
    )
    parser.add_argument(
        "--surprise-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_lenta_factual_surprise_strict_reported_1200.jsonl",
    )
    parser.add_argument(
        "--go-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    source_summaries = [
        add_group_texts(grouped, args.neutral_path, allowed={"neutral"}),
        add_group_texts(grouped, args.surprise_path, allowed={"surprise"}),
        add_group_texts(grouped, args.go_path, allowed={"joy", "sadness", "fear", "anger"}),
    ]

    for label in LABELS:
        if len(grouped[label]) < args.count_per_class + args.prototypes_per_class + 1:
            raise RuntimeError(
                f"Not enough rows for {label}: {len(grouped[label])}, "
                f"need {args.count_per_class + args.prototypes_per_class + 1}"
            )
        rng.shuffle(grouped[label])

    records: list[dict[str, Any]] = []
    for label in LABELS:
        queries = grouped[label][: args.count_per_class]
        for index, query in enumerate(queries):
            prototypes: dict[str, list[str]] = {}
            for prototype_label in LABELS:
                pool = [text for text in grouped[prototype_label] if text != query]
                prototypes[prototype_label] = rng.sample(pool, args.prototypes_per_class)
            records.append(
                {
                    "source": "cedr_knn_proxy:lenta_neutral_surprise_go",
                    "objective": "prototype_classification",
                    "query": query,
                    "label": label,
                    "prototypes": prototypes,
                    "metadata": {
                        "group": label,
                        "index": index,
                        "prototypes_per_class": args.prototypes_per_class,
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
            "count_per_class": args.count_per_class,
            "prototypes_per_class": args.prototypes_per_class,
            "labels": LABELS,
            "label_counts": dict(Counter(record["label"] for record in records)),
            "available_texts": {label: len(grouped[label]) for label in LABELS},
            "sources": source_summaries,
            "construction": "prototype-classification proxy for CEDR 5-NN: Lenta neutral boundary, Lenta factual surprise, GoEmotions-RU other emotions",
            "contamination_policy": "inherits source CEDR-overlap filtering; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
