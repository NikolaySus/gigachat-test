#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--go-path", type=Path, required=True)
    parser.add_argument("--dvach-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dvach-total", type=int, required=True)
    parser.add_argument("--seed", type=int, default=819)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    go_rows = read_jsonl(args.go_path)
    dvach_rows = read_jsonl(args.dvach_path)

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in dvach_rows:
        group = str(row.get("metadata", {}).get("group", ""))
        if group:
            by_group[group].append(row)

    groups = sorted(by_group)
    per_group = args.dvach_total // len(groups)
    remainder = args.dvach_total % len(groups)
    selected: list[dict] = []
    counts: dict[str, int] = {}
    for idx, group in enumerate(groups):
        rows = by_group[group][:]
        rng.shuffle(rows)
        take = per_group + (1 if idx < remainder else 0)
        take = min(take, len(rows))
        counts[group] = take
        for row in rows[:take]:
            mixed = dict(row)
            mixed["source"] = f"{row.get('source', 'dvach')}:mixed_go_dvach"
            selected.append(mixed)

    mixed_rows = go_rows + selected
    rng.shuffle(mixed_rows)
    write_jsonl(args.output_path, mixed_rows)

    summary_path = args.output_path.with_suffix(args.output_path.suffix + ".summary.json")
    summary = {
        "go_rows": len(go_rows),
        "dvach_rows": len(selected),
        "dvach_counts": counts,
        "total_rows": len(mixed_rows),
        "seed": args.seed,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
