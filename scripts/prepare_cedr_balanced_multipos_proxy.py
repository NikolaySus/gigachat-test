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


def normalize_group(row: dict[str, Any]) -> str | None:
    group = row.get("metadata", {}).get("group")
    if group == "no_emotion":
        group = "neutral"
    return group if group in LABELS else None


def add_rows(grouped: dict[str, list[dict]], path: Path, *, allowed: set[str] | None = None) -> dict[str, Any]:
    counts = Counter()
    if not path.exists():
        return {"path": str(path), "missing": True}
    seen = {(label, row["query"]) for label, rows in grouped.items() for row in rows if "query" in row}
    for row in read_jsonl(path):
        group = normalize_group(row)
        if group is None or (allowed is not None and group not in allowed):
            continue
        query = row.get("query")
        positive = row.get("positive")
        if not isinstance(query, str) or not isinstance(positive, str):
            continue
        key = (group, query)
        if key in seen:
            continue
        seen.add(key)
        new_row = dict(row)
        new_row.setdefault("metadata", {})["group"] = group
        grouped[group].append(new_row)
        counts[group] += 1
    return {"path": str(path), "added": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced multi-positive CEDR proxy from clean sources.")
    parser.add_argument("--per-label", type=int, default=900)
    parser.add_argument("--seed", type=int, default=1011)
    parser.add_argument("--name", default="cedr_balanced_multipos_proxy_5400")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    sources = []
    sources.append(
        add_rows(
            grouped,
            DATA_DIR / "open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_3200.jsonl",
            allowed={"neutral"},
        )
    )
    sources.append(
        add_rows(
            grouped,
            DATA_DIR / "open_ru_1r_nc_cedr_social_neutral_3200.jsonl",
            allowed={"neutral"},
        )
    )
    sources.append(
        add_rows(
            grouped,
            DATA_DIR / "open_ru_1r_nc_cedr_lexical_hard_neutral_v2_1500.jsonl",
            allowed={"neutral"},
        )
    )
    sources.append(
        add_rows(
            grouped,
            DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
            allowed=set(LABELS),
        )
    )

    selected = []
    selected_counts = {}
    for label in LABELS:
        pool = grouped[label][:]
        rng.shuffle(pool)
        if len(pool) < args.per_label:
            raise RuntimeError(f"Need {args.per_label} for {label}, found {len(pool)}")
        rows = pool[: args.per_label]
        selected_counts[label] = len(rows)
        selected.extend(rows)
    rng.shuffle(selected)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, selected)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "records": len(selected),
            "per_label": args.per_label,
            "selected_counts": selected_counts,
            "available_counts": {label: len(grouped[label]) for label in LABELS},
            "sources": sources,
            "training_note": "use multi_positive_metadata_key=group so same pseudo-label rows are positives within a batch",
            "contamination_policy": "inherits exact/near CEDR overlap filtering from source components; no CEDR records used",
            "seed": args.seed,
        },
    )
    print(f"prepared {out.relative_to(Path.cwd())}: {len(selected)} rows")


if __name__ == "__main__":
    main()
