from __future__ import annotations

import argparse
import json
import random
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
    parser = argparse.ArgumentParser(description="Convert CEDR-shaped contrastive rows to pair-score rows.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--negatives-per-query", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(Path(args.input))
    records: list[dict[str, Any]] = []
    for row in rows:
        group = row.get("metadata", {}).get("group")
        records.append(
            {
                "source": f"{row.get('source', 'unknown')}:pairscore",
                "objective": "pair_score",
                "sentence1": row["query"],
                "sentence2": row["positive"],
                "score": 1.0,
                "metadata": {"group": group, "kind": "positive"},
            }
        )
        negatives = list(row.get("negatives", []))
        rng.shuffle(negatives)
        for negative in negatives[: args.negatives_per_query]:
            records.append(
                {
                    "source": f"{row.get('source', 'unknown')}:pairscore",
                    "objective": "pair_score",
                    "sentence1": row["query"],
                    "sentence2": negative,
                    "score": 0.0,
                    "metadata": {"group": group, "kind": "negative"},
                }
            )

    rng.shuffle(records)
    if len(records) > args.max_records:
        records = records[: args.max_records]

    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    summary = {
        "input": args.input,
        "output": str(out.relative_to(ROOT)),
        "rows": len(records),
        "negatives_per_query": args.negatives_per_query,
        "max_records": args.max_records,
        "seed": args.seed,
        "objective": "pair_score",
        "contamination_policy": "inherits source contrastive component filtering",
    }
    out.with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
