from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix JSONL contrastive records with optional per-file caps.")
    parser.add_argument("--input", action="append", nargs=3, metavar=("PATH", "CAP", "NAME"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument(
        "--repeat-short-inputs",
        action="store_true",
        help="If CAP is larger than a file, repeat shuffled copies until CAP rows are selected.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    mixed: list[dict[str, Any]] = []
    summary = {"seed": args.seed, "inputs": [], "total": 0}
    for path_value, cap_value, name in args.input:
        path = Path(path_value)
        records = read_jsonl(path)
        rng.shuffle(records)
        cap = len(records) if cap_value == "all" else int(cap_value)
        if cap <= len(records):
            selected = records[:cap]
        elif args.repeat_short_inputs:
            selected = []
            while len(selected) < cap:
                repeated = list(records)
                rng.shuffle(repeated)
                selected.extend(repeated[: cap - len(selected)])
        else:
            selected = records
        mixed.extend(selected)
        summary["inputs"].append(
            {
                "name": name,
                "path": str(path),
                "available": len(records),
                "requested": cap,
                "selected": len(selected),
                "repeated": cap > len(records) and args.repeat_short_inputs,
            }
        )
    rng.shuffle(mixed)
    write_jsonl(args.output, mixed)
    summary["total"] = len(mixed)
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {len(mixed)} records")


if __name__ == "__main__":
    main()
