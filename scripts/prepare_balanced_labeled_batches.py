from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Create label-balanced batches for labeled_text metric training.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--labels-per-step", type=int, default=4)
    parser.add_argument("--records-per-label", type=int, default=2)
    parser.add_argument("--min-label-count", type=int, default=8)
    parser.add_argument("--loss", choices=("supcon", "circle", "multi_similarity"), default=None)
    parser.add_argument("--seed", type=int, default=2901)
    args = parser.parse_args()

    source_path = ROOT / args.input
    output_path = ROOT / args.output
    records = read_jsonl(source_path)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("objective") != "labeled_text":
            continue
        by_label[str(record["label"])].append(record)

    eligible = {
        label: list(items)
        for label, items in by_label.items()
        if len(items) >= max(args.min_label_count, args.records_per_label)
    }
    if len(eligible) < args.labels_per_step:
        raise ValueError(
            f"Need at least {args.labels_per_step} eligible labels, got {len(eligible)}"
        )

    rng = random.Random(args.seed)
    label_order = sorted(eligible, key=lambda label: (-len(eligible[label]), label))
    # Drop the extreme majority label from the sampling pool when enough other
    # labels exist; otherwise almost every batch becomes a generic-vs-specific
    # contrast instead of learning fine-grained review sentiment/ratings.
    if len(label_order) > args.labels_per_step * 2:
        label_order = label_order[1:]
    for items in eligible.values():
        rng.shuffle(items)

    pointers = {label: 0 for label in eligible}
    output_records: list[dict] = []
    label_counts: dict[str, int] = defaultdict(int)
    for step in range(args.steps):
        start = (step * args.labels_per_step) % len(label_order)
        selected_labels = [
            label_order[(start + offset) % len(label_order)]
            for offset in range(args.labels_per_step)
        ]
        for label in selected_labels:
            items = eligible[label]
            pointer = pointers[label]
            if pointer + args.records_per_label > len(items):
                rng.shuffle(items)
                pointer = 0
            batch_items = items[pointer : pointer + args.records_per_label]
            pointers[label] = pointer + args.records_per_label
            for item in batch_items:
                record = dict(item)
                if args.loss is not None:
                    record["loss"] = args.loss
                output_records.append(record)
                label_counts[label] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in output_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output),
        "input": str(args.input),
        "records": len(output_records),
        "steps": args.steps,
        "batch_size": args.labels_per_step * args.records_per_label,
        "labels_per_step": args.labels_per_step,
        "records_per_label": args.records_per_label,
        "min_label_count": args.min_label_count,
        "eligible_labels": len(label_order),
        "loss_override": args.loss,
        "seed": args.seed,
        "top_used_labels": sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[:25],
    }
    with output_path.with_name(output_path.stem + "_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
