from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def convert_record(record: dict[str, Any], *, ridge_lambda: float, source_suffix: str) -> dict[str, Any]:
    if record.get("objective") != "knn_classification":
        raise ValueError(f"Expected knn_classification record, got {record.get('objective')!r}")
    converted = dict(record)
    converted["objective"] = "ridge_classification"
    converted["source"] = f"{record.get('source', 'unknown')}:{source_suffix}"
    converted["ridge_lambda"] = ridge_lambda
    metadata = dict(record.get("metadata") or {})
    metadata["converted_from_objective"] = "knn_classification"
    metadata["ridge_lambda"] = ridge_lambda
    metadata["contamination_policy"] = (
        str(metadata.get("contamination_policy", "")).strip()
        or "Inherits source dataset contamination policy; no benchmark rows or released model outputs are added."
    )
    converted["metadata"] = metadata
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--source-suffix", default="ridge_probe_episode")
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    labels: set[str] = set()
    support_class_counts = []
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            converted = convert_record(
                record,
                ridge_lambda=args.ridge_lambda,
                source_suffix=args.source_suffix,
            )
            labels.add(str(converted["label"]))
            support_class_counts.append(len(converted["supports"]))
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "records": count,
        "labels": len(labels),
        "ridge_lambda": args.ridge_lambda,
        "min_support_classes": min(support_class_counts) if support_class_counts else 0,
        "max_support_classes": max(support_class_counts) if support_class_counts else 0,
        "objective": "ridge_classification",
    }
    output_path.with_name(output_path.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
