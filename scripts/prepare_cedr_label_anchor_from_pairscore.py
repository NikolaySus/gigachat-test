from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert explicit CEDR label pair-score rows to contrastive label-anchor rows."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-queries", type=int, default=3000)
    parser.add_argument("--negatives-per-query", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1041)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"positive": None, "negatives": [], "metadata": {}})
    for row in read_jsonl(args.input):
        if row.get("objective") != "pair_score":
            continue
        query = row["sentence1"]
        group = grouped[query]
        metadata = dict(row.get("metadata") or {})
        if metadata:
            group["metadata"].update(metadata)
        if float(row.get("score", 0.0)) >= 0.5:
            group["positive"] = row["sentence2"]
        else:
            group["negatives"].append(row["sentence2"])

    items = []
    for query, group in grouped.items():
        if not group["positive"] or not group["negatives"]:
            continue
        negatives = list(dict.fromkeys(group["negatives"]))
        rng.shuffle(negatives)
        items.append((query, group["positive"], negatives[: args.negatives_per_query], group["metadata"]))
    rng.shuffle(items)
    items = items[: args.max_queries]

    records = [
        {
            "source": "cedr_label_anchor:" + str(metadata.get("path") or "pairscore"),
            "objective": "contrastive",
            "query": query,
            "positive": positive,
            "negatives": negatives,
            "metadata": {
                **metadata,
                "group": metadata.get("group", "neutral"),
                "construction": "explicit_label_anchor_from_pairscore",
            },
        }
        for query, positive, negatives, metadata in items
    ]

    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    summary = {
        "input": str(args.input),
        "output": str(out.relative_to(ROOT)),
        "records": len(records),
        "max_queries": args.max_queries,
        "negatives_per_query": args.negatives_per_query,
        "seed": args.seed,
        "construction": "contrastive query -> correct label statement, with false label statements as negatives",
        "contamination_policy": "inherits source pair-score filtering; no CEDR records used",
    }
    out.with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
