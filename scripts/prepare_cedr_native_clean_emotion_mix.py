from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_of(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    group = metadata.get("group") or row.get("label")
    if isinstance(group, str):
        return group
    labels = metadata.get("groups") or metadata.get("labels")
    if isinstance(labels, list) and labels:
        return "+".join(sorted(str(label) for label in labels))
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix already-audited clean native Russian emotion components.")
    parser.add_argument("--name", default="cedr_native_clean_emotion_mix_7200")
    parser.add_argument("--seed", type=int, default=964)
    parser.add_argument("--per-source-cap", type=int, default=1800)
    parser.add_argument("--max-records", type=int, default=7200)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=[
            DATA_DIR / "open_ru_1r_nc_cedr_brighter_rus_train_dev_clean.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_semeval2025_rus_tracka_train_dev_clean.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_resd_clean_emotion_component.jsonl",
            DATA_DIR / "open_ru_1r_nc_cedr_darkester_ruemotions_core.jsonl",
        ],
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    source_group_counts: dict[str, dict[str, int]] = {}
    seen_texts: set[str] = set()

    for path in args.source:
        rows = read_jsonl(path)
        rng.shuffle(rows)
        selected = []
        for row in rows:
            query = row.get("query") or row.get("sentence1")
            if not isinstance(query, str) or not query.strip():
                continue
            dedupe_key = query.strip()
            if dedupe_key in seen_texts:
                continue
            seen_texts.add(dedupe_key)
            copied = dict(row)
            metadata = dict(copied.get("metadata") or {})
            metadata["native_clean_mix_source"] = str(path.relative_to(ROOT))
            copied["metadata"] = metadata
            selected.append(copied)
            if len(selected) >= args.per_source_cap:
                break
        records.extend(selected)
        source_counts[str(path.relative_to(ROOT))] = len(selected)
        source_group_counts[str(path.relative_to(ROOT))] = dict(Counter(group_of(row) for row in selected))

    rng.shuffle(records)
    records = records[: args.max_records]

    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "records": len(records),
            "seed": args.seed,
            "per_source_cap": args.per_source_cap,
            "max_records": args.max_records,
            "source_counts": source_counts,
            "source_group_counts": source_group_counts,
            "final_group_counts": dict(Counter(group_of(row) for row in records)),
            "sources": [str(path.relative_to(ROOT)) for path in args.source],
            "construction": "shuffled mix of already-audited clean Russian CEDR-compatible emotion components",
            "contamination_policy": "inherits exact/near CEDR-overlap removal from each source component; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
