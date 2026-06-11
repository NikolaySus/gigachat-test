from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorder pair_score records into fixed balanced score-range batches."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=2044)
    args = parser.parse_args()

    if args.batch_size % 4 != 0:
        raise ValueError("batch-size must be divisible by 4")

    random.seed(args.seed)
    records = [record for record in read_jsonl(args.input) if record.get("objective") == "pair_score"]
    buckets = {
        "zero": [],
        "mid": [],
        "high": [],
        "very_high": [],
    }
    for record in records:
        score = float(record["score"])
        if score <= 0.05:
            buckets["zero"].append(record)
        elif score < 0.85:
            buckets["mid"].append(record)
        elif score < 0.95:
            buckets["high"].append(record)
        else:
            buckets["very_high"].append(record)
    for values in buckets.values():
        random.shuffle(values)

    per_bucket = args.batch_size // 4
    output = []
    used = {name: 0 for name in buckets}
    while len(output) + args.batch_size <= args.max_records:
        batch = []
        for name in ("zero", "mid", "high", "very_high"):
            start = used[name]
            end = start + per_bucket
            if end > len(buckets[name]):
                batch = []
                break
            batch.extend(buckets[name][start:end])
            used[name] = end
        if not batch:
            break
        random.shuffle(batch)
        output.extend(batch)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for record in output:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.out),
        "records": len(output),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "bucket_available": {name: len(values) for name, values in buckets.items()},
        "bucket_used": used,
        "construction": "Each ordered batch contains equal counts from zero, mid, high, and very_high score buckets.",
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
