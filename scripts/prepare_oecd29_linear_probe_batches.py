#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_source(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            label = str(record.get("label", "")).strip()
            if not label:
                skipped["missing_label"] += 1
                continue
            by_label[label].append(record)
    return by_label, {
        "records": sum(len(values) for values in by_label.values()),
        "labels": len(by_label),
        "label_counts": {label: len(values) for label, values in by_label.items()},
        "skipped": dict(skipped),
    }


def make_batches(
    by_label: dict[str, list[dict[str, Any]]],
    *,
    batch_count: int,
    labels_per_batch: int,
    supports_per_label: int,
    queries_per_label: int,
    min_records_per_label: int,
    seed: int,
    ridge_lambda: float,
    encode_batch_size: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    labels = sorted(
        [label for label, rows in by_label.items() if len(rows) >= min_records_per_label],
        key=lambda label: len(by_label[label]),
        reverse=True,
    )
    if len(labels) < labels_per_batch:
        raise ValueError(f"Need {labels_per_batch} labels, got {len(labels)}")
    for label in labels:
        rng.shuffle(by_label[label])

    cursors = Counter()
    sampled = Counter()
    records: list[dict[str, Any]] = []
    per_label = supports_per_label + queries_per_label
    for batch_index in range(batch_count):
        selected = rng.sample(labels, labels_per_batch)
        for label in selected:
            rows = by_label[label]
            for item_index in range(per_label):
                cursor = cursors[label] % len(rows)
                source = rows[cursor]
                cursors[label] += 1
                if cursors[label] % len(rows) == 0:
                    rng.shuffle(rows)
                role = "support" if item_index < supports_per_label else "query"
                records.append(
                    {
                        "source": "kaggle/ergkerg/russian-scientific-articles:oecd29_linear_probe",
                        "objective": "linear_probe_labeled_text",
                        "text": source["text"],
                        "label": label,
                        "role": role,
                        "ridge_lambda": ridge_lambda,
                        "use_bias": True,
                        **({"encode_batch_size": encode_batch_size} if encode_batch_size else {}),
                        "metadata": {
                            **source.get("metadata", {}),
                            "batch_index": batch_index,
                            "role": role,
                            "linear_probe_label": label,
                            "contamination_policy": (
                                "Derived from audited Kaggle GRNTI-to-OECD29 metric data after "
                                "RuSciBench GRNTI/OECD title/prefix overlap removal. OECD-style "
                                "labels are public label-name mappings only; no benchmark rows, "
                                "released model outputs, or released latent weights are used."
                            ),
                        },
                    }
                )
                sampled[label] += 1
    return records, {
        "records": len(records),
        "batch_count": batch_count,
        "batch_size": labels_per_batch * per_label,
        "labels_per_batch": labels_per_batch,
        "supports_per_label": supports_per_label,
        "queries_per_label": queries_per_label,
        "ridge_lambda": ridge_lambda,
        "encode_batch_size": encode_batch_size,
        "usable_labels": labels,
        "sampled_label_counts": dict(sampled),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_to_oecd29_circle_b4_2400_seed2661.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/fair_oecd29_linear_probe_b15_60_seed2961.jsonl",
    )
    parser.add_argument("--batch-count", type=int, default=60)
    parser.add_argument("--labels-per-batch", type=int, default=5)
    parser.add_argument("--supports-per-label", type=int, default=2)
    parser.add_argument("--queries-per-label", type=int, default=1)
    parser.add_argument("--min-records-per-label", type=int, default=12)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2961)
    args = parser.parse_args()

    by_label, source_summary = read_source(args.source)
    records, batch_summary = make_batches(
        by_label,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        supports_per_label=args.supports_per_label,
        queries_per_label=args.queries_per_label,
        min_records_per_label=args.min_records_per_label,
        seed=args.seed,
        ridge_lambda=args.ridge_lambda,
        encode_batch_size=args.encode_batch_size or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "seed": args.seed,
        "source_summary": source_summary,
        "batch_summary": batch_summary,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
