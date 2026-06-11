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
    parser = argparse.ArgumentParser(description="Convert neutral-boundary contrastive rows to gentler pair-score rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--name", default="cedr_goemotions_neutral_boundary_pairscore_14400")
    parser.add_argument("--positive-score", type=float, default=0.85)
    parser.add_argument("--negative-score", type=float, default=0.20)
    parser.add_argument("--max-records", type=int, default=14400)
    parser.add_argument("--seed", type=int, default=844)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    for row in read_jsonl(args.input):
        records.append(
            {
                "source": f"{row.get('source', 'unknown')}:pairscore_soft",
                "objective": "pair_score",
                "sentence1": row["query"],
                "sentence2": row["positive"],
                "score": args.positive_score,
                "metadata": {"group": "neutral", "kind": "positive"},
            }
        )
        negatives = list(row.get("negatives", []))
        rng.shuffle(negatives)
        for negative in negatives:
            records.append(
                {
                    "source": f"{row.get('source', 'unknown')}:pairscore_soft",
                    "objective": "pair_score",
                    "sentence1": row["query"],
                    "sentence2": negative,
                    "score": args.negative_score,
                    "metadata": {"group": "neutral", "kind": "neutral_emotion_negative"},
                }
            )

    rng.shuffle(records)
    records = records[: args.max_records]
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    out.with_name(out.stem + "_summary.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(out.relative_to(ROOT)),
                "rows": len(records),
                "positive_score": args.positive_score,
                "negative_score": args.negative_score,
                "max_records": args.max_records,
                "seed": args.seed,
                "contamination_policy": "inherits source filtering",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
