#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"


CLASSIFICATION_DATASET = "ai-forever/ru-scibench-oecd-classification"
CLUSTERING_DATASET = "ai-forever/ru-scibench-oecd-clustering-p2p"
CONTAMINATION_NOTE = (
    "YES: direct RuSciBench OECD benchmark rows/labels are intentionally used. "
    "Diagnostic only; forbidden for fair training or fair comparisons."
)


def text_value(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def load_oecd_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    classification = load_dataset(CLASSIFICATION_DATASET)
    for split, dataset in classification.items():
        for index, item in enumerate(dataset):
            text = text_value(item.get("text", ""))
            label = text_value(item.get("label_text", item.get("label", "")))
            if not text or not label:
                continue
            rows.append(
                {
                    "text": text,
                    "label": f"oecd::{label}",
                    "source_dataset": CLASSIFICATION_DATASET,
                    "split": split,
                    "row_index": index,
                    "benchmark_surface": "classification",
                }
            )

    clustering = load_dataset(CLUSTERING_DATASET, split="test")
    for index, item in enumerate(clustering):
        text = text_value(item.get("sentences", ""))
        label = text_value(item.get("labels", ""))
        if not text or not label:
            continue
        rows.append(
            {
                "text": text,
                "label": f"oecd::{label}",
                "source_dataset": CLUSTERING_DATASET,
                "split": "test",
                "row_index": index,
                "benchmark_surface": "clustering_p2p",
            }
        )

    return dedupe_rows(rows)


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (row["text"], row["label"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def group_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    return grouped


def metadata(row: dict[str, Any], *, batch_index: int | None = None, role: str | None = None) -> dict[str, Any]:
    payload = {
        "source_dataset": row["source_dataset"],
        "source_split": row["split"],
        "source_row_index": row["row_index"],
        "benchmark_surface": row["benchmark_surface"],
        "contamination": CONTAMINATION_NOTE,
        "do_not_use_for_fair_results": True,
    }
    if batch_index is not None:
        payload["batch_index"] = batch_index
    if role is not None:
        payload["role"] = role
    return payload


def make_linear_probe_records(
    rows: list[dict[str, Any]],
    *,
    batch_count: int,
    labels_per_batch: int,
    supports_per_label: int,
    queries_per_label: int,
    seed: int,
    ridge_lambda: float,
    encode_batch_size: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label = group_by_label(rows)
    per_label = supports_per_label + queries_per_label
    labels = sorted(
        [label for label, items in by_label.items() if len(items) >= per_label],
        key=lambda label: len(by_label[label]),
        reverse=True,
    )
    if len(labels) < labels_per_batch:
        raise ValueError(f"Need {labels_per_batch} labels with >= {per_label} rows; got {len(labels)}")
    for label in labels:
        rng.shuffle(by_label[label])

    cursors = Counter()
    records: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        selected_labels = rng.sample(labels, labels_per_batch)
        for label in selected_labels:
            label_rows = by_label[label]
            for item_index in range(per_label):
                cursor = cursors[label] % len(label_rows)
                row = label_rows[cursor]
                cursors[label] += 1
                if cursors[label] % len(label_rows) == 0:
                    rng.shuffle(label_rows)
                role = "support" if item_index < supports_per_label else "query"
                records.append(
                    {
                        "source": "CONTAMINATED_OECD_DIAGNOSTIC:linear_probe_exact_rows",
                        "objective": "linear_probe_labeled_text",
                        "text": row["text"],
                        "label": row["label"],
                        "role": role,
                        "ridge_lambda": ridge_lambda,
                        "use_bias": True,
                        "encode_batch_size": encode_batch_size,
                        "metadata": metadata(row, batch_index=batch_index, role=role),
                    }
                )
    return records


def make_labeled_metric_records(rows: list[dict[str, Any]], *, loss: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return [
        {
            "source": f"CONTAMINATED_OECD_DIAGNOSTIC:labeled_{loss}_exact_rows",
            "objective": "labeled_text",
            "text": row["text"],
            "label": row["label"],
            "loss": loss,
            "metadata": metadata(row),
        }
        for row in shuffled
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]], records: list[dict[str, Any]], *, variant: str) -> None:
    label_counts = Counter(row["label"] for row in rows)
    surface_counts = Counter(row["benchmark_surface"] for row in rows)
    objective_counts = Counter(record["objective"] for record in records)
    summary = {
        "variant": variant,
        "output": str(path.relative_to(ROOT)),
        "records": len(records),
        "source_rows_after_dedupe": len(rows),
        "source_surface_counts": dict(surface_counts),
        "source_label_count": len(label_counts),
        "source_min_label_count": min(label_counts.values()),
        "source_max_label_count": max(label_counts.values()),
        "objective_counts": dict(objective_counts),
        "contamination": CONTAMINATION_NOTE,
        "do_not_use_for_fair_results": True,
    }
    path.with_suffix(path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-output", type=Path, default=DATA_DIR / "CONTAMINATED_oecd_exact_linear_probe_b12_400_seed3121.jsonl")
    parser.add_argument("--circle-output", type=Path, default=DATA_DIR / "CONTAMINATED_oecd_exact_labeled_circle_seed3122.jsonl")
    parser.add_argument("--batch-count", type=int, default=400)
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--supports-per-label", type=int, default=2)
    parser.add_argument("--queries-per-label", type=int, default=1)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--linear-seed", type=int, default=3121)
    parser.add_argument("--circle-seed", type=int, default=3122)
    args = parser.parse_args()

    rows = load_oecd_rows()
    linear_records = make_linear_probe_records(
        rows,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        supports_per_label=args.supports_per_label,
        queries_per_label=args.queries_per_label,
        seed=args.linear_seed,
        ridge_lambda=args.ridge_lambda,
        encode_batch_size=args.encode_batch_size,
    )
    circle_records = make_labeled_metric_records(rows, loss="circle", seed=args.circle_seed)

    write_jsonl(args.linear_output, linear_records)
    write_summary(args.linear_output, rows, linear_records, variant="linear_probe_exact_rows")
    write_jsonl(args.circle_output, circle_records)
    write_summary(args.circle_output, rows, circle_records, variant="labeled_circle_exact_rows")


if __name__ == "__main__":
    main()
