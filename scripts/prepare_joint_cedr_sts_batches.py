from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build ordered mixed batches containing CEDR contrastive records and "
            "prompt-aligned RuSTS/NLI pair-score records."
        )
    )
    parser.add_argument("--cedr", type=Path, required=True)
    parser.add_argument("--sts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cedr-per-batch", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2120)
    args = parser.parse_args()

    if not (0 < args.cedr_per_batch < args.batch_size):
        raise ValueError("--cedr-per-batch must be between 1 and batch_size - 1")
    sts_per_batch = args.batch_size - args.cedr_per_batch

    rng = random.Random(args.seed)
    cedr = [row for row in read_jsonl(args.cedr) if row.get("objective") == "contrastive"]
    sts = [row for row in read_jsonl(args.sts) if row.get("objective") == "pair_score"]
    rng.shuffle(cedr)

    # The STS file is already ordered in balanced score-range blocks. Shuffle only
    # block order, not rows inside each block, so every mixed batch keeps a useful
    # low/mid/high/very-high score contrast for correlation/rank losses.
    sts_blocks = [
        sts[index : index + sts_per_batch]
        for index in range(0, len(sts) - sts_per_batch + 1, sts_per_batch)
    ]
    rng.shuffle(sts_blocks)

    max_batches = min(args.max_batches, len(cedr) // args.cedr_per_batch, len(sts_blocks))
    records: list[dict] = []
    for batch_index in range(max_batches):
        batch = []
        batch.extend(cedr[batch_index * args.cedr_per_batch : (batch_index + 1) * args.cedr_per_batch])
        batch.extend(sts_blocks[batch_index])
        rng.shuffle(batch)
        for record in batch:
            metadata = dict(record.get("metadata") or {})
            metadata["joint_batch_index"] = batch_index
            metadata["joint_construction"] = "cedr_contrastive_plus_prompt_aligned_sts_pairscore"
            record = dict(record)
            record["metadata"] = metadata
            records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.out),
        "records": len(records),
        "batches": max_batches,
        "batch_size": args.batch_size,
        "cedr_per_batch": args.cedr_per_batch,
        "sts_per_batch": sts_per_batch,
        "cedr_input_records": len(cedr),
        "sts_input_records": len(sts),
        "seed": args.seed,
        "construction": (
            "Ordered batches contain CEDR contrastive records and prompt-aligned "
            "RuSTS/NLI pair-score records in the same optimization step. The STS "
            "sub-block order is preserved for score-range correlation/rank losses."
        ),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
