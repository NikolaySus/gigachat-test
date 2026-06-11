from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the exact remaining split after a seeded GeRaCl sample.")
    parser.add_argument("--source", type=Path, default=Path("data/contrastive/open_ru_1r_nc_geracl.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--used-count", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    records = read_jsonl(args.source)
    shuffled = records[:]
    random.Random(args.seed).shuffle(shuffled)
    used = shuffled[: args.used_count]
    remaining = shuffled[args.used_count :]

    write_jsonl(args.out, remaining)
    summary_path = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    write_json(
        summary_path,
        {
            "source": str(args.source),
            "output": str(args.out),
            "seed": args.seed,
            "split_rule": "same shuffle as the corresponding mixed stage; first used_count records are in stage 1, remaining records are here",
            "counts": {
                "source_total": len(records),
                "stage1_geracl_used": len(used),
                "remaining": len(remaining),
            },
            "batch_size": args.batch_size,
            "max_steps_1x": len(remaining) // args.batch_size,
        },
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
