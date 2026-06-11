from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    sources: dict[str, int] = {}
    objectives: dict[str, int] = {}
    for record in records:
        source = str(record.get("source", "unknown"))
        objective = str(record.get("objective", "contrastive"))
        sources[source] = sources.get(source, 0) + 1
        objectives[objective] = objectives.get(objective, 0) + 1
    return {"sources": sources, "objectives": objectives}


def split_records(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("source", "unknown")), str(record.get("objective", "contrastive")))].append(record)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group in groups.values():
        rng.shuffle(group)
        validation_count = max(1, round(len(group) * validation_fraction)) if len(group) > 1 else 0
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])

    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL records into deterministic stratified train/validation files.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    train, validation = split_records(records, validation_fraction=args.validation_fraction, seed=args.seed)
    write_jsonl(args.train_out, train)
    write_jsonl(args.val_out, validation)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "train_out": str(args.train_out),
                "val_out": str(args.val_out),
                "total": len(records),
                "train": {"total": len(train), **summarize(train)},
                "validation": {"total": len(validation), **summarize(validation)},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
