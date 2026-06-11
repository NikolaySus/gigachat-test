from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


LABELS = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_group(group: Any) -> str | None:
    group = str(group or "")
    if group == "no_emotion":
        group = "neutral"
    return group if group in LABELS else None


def add_texts(grouped: dict[str, list[str]], path: Path, *, allowed: set[str] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    seen = {(group, text) for group, texts in grouped.items() for text in texts}
    counts = Counter()
    for row in read_jsonl(path):
        group = clean_group(row.get("metadata", {}).get("group"))
        if group is None or (allowed is not None and group not in allowed):
            continue
        for key in ("query", "positive", *(() if row.get("positives") is None else ("positives",))):
            value = row.get(key)
            values = value if isinstance(value, list) else [value]
            for text in values:
                if not isinstance(text, str) or not text.strip():
                    continue
                item = (group, text)
                if item in seen:
                    continue
                seen.add(item)
                grouped[group].append(text)
                counts[group] += 1
    return {"path": str(path), "added": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explicit multi-positive CEDR proxy records.")
    parser.add_argument("--queries-per-label", type=int, default=360)
    parser.add_argument("--positives-per-query", type=int, default=4)
    parser.add_argument("--negatives-per-query", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1021)
    parser.add_argument("--name", default="cedr_explicit_multipos_proxy_2160_p4")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    sources = [
        add_texts(
            grouped,
            DATA_DIR / "open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_3200.jsonl",
            allowed={"neutral"},
        ),
        add_texts(
            grouped,
            DATA_DIR / "open_ru_1r_nc_cedr_social_neutral_3200.jsonl",
            allowed={"neutral"},
        ),
        add_texts(
            grouped,
            DATA_DIR / "open_ru_1r_nc_cedr_lexical_hard_neutral_v2_1500.jsonl",
            allowed={"neutral"},
        ),
        add_texts(
            grouped,
            DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
            allowed=set(LABELS),
        ),
    ]

    needed = args.queries_per_label + args.positives_per_query + 1
    for label in LABELS:
        rng.shuffle(grouped[label])
        if len(grouped[label]) < needed:
            raise RuntimeError(f"Need {needed} texts for {label}, found {len(grouped[label])}")

    records = []
    for label in LABELS:
        pool = grouped[label]
        for index, query in enumerate(pool[: args.queries_per_label]):
            positive_pool = [text for text in pool if text != query]
            positives = rng.sample(positive_pool, args.positives_per_query)
            negative_labels = [other for other in LABELS if other != label and grouped[other]]
            rng.shuffle(negative_labels)
            negatives = []
            for other in negative_labels:
                negatives.append(rng.choice(grouped[other]))
                if len(negatives) >= args.negatives_per_query:
                    break
            records.append(
                {
                    "source": "cedr_explicit_multipos_proxy",
                    "objective": "contrastive",
                    "query": query,
                    "positive": positives[0],
                    "positives": positives[1:],
                    "negatives": negatives,
                    "metadata": {
                        "group": label,
                        "index": index,
                        "positives_per_query": args.positives_per_query,
                    },
                }
            )
    rng.shuffle(records)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "records": len(records),
            "queries_per_label": args.queries_per_label,
            "positives_per_query": args.positives_per_query,
            "negatives_per_query": args.negatives_per_query,
            "label_counts": dict(Counter(row["metadata"]["group"] for row in records)),
            "available_counts": {label: len(grouped[label]) for label in LABELS},
            "sources": sources,
            "construction": "explicit multi-positive supervised contrastive proxy for CEDR 5-NN geometry",
            "contamination_policy": "inherits source exact/near CEDR overlap filtering; no CEDR records used",
            "seed": args.seed,
        },
    )
    print(f"prepared {out.relative_to(Path.cwd())}: {len(records)} rows")


if __name__ == "__main__":
    main()
