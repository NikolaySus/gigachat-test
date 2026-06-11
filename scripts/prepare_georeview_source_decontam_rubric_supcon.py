from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def normalize(text: str) -> str:
    text = text.lower().replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a diagnostic rubric SupCon dataset from d0rj geo reviews after "
            "removing exact normalized mteb/georeview rows. This remains source-overlap "
            "risky and should not be counted as final clean fair evidence."
        )
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=6000)
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--examples-per-label", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2041)
    args = parser.parse_args()

    random.seed(args.seed)

    benchmark_texts: set[str] = set()
    benchmark_rows = 0
    for split in ("train", "validation", "test"):
        ds = load_dataset("mteb/georeview", split=split)
        for row in ds:
            benchmark_rows += 1
            benchmark_texts.add(normalize(row["text"]))

    groups: dict[str, list[str]] = defaultdict(list)
    source_rows = 0
    exact_overlap_rows = 0
    too_short_rows = 0
    missing_rubric_rows = 0
    duplicate_clean_rows = 0
    seen_clean: set[str] = set()

    ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train")
    for row in ds:
        source_rows += 1
        text = str(row.get("text") or "").strip()
        norm = normalize(text)
        if not norm or len(norm) < 40:
            too_short_rows += 1
            continue
        if norm in benchmark_texts:
            exact_overlap_rows += 1
            continue
        if norm in seen_clean:
            duplicate_clean_rows += 1
            continue
        seen_clean.add(norm)
        rubrics = str(row.get("rubrics") or "").strip()
        if not rubrics:
            missing_rubric_rows += 1
            continue
        primary = rubrics.split(";")[0].strip()
        if not primary:
            missing_rubric_rows += 1
            continue
        groups[primary].append(text)

    eligible_labels = [label for label, texts in groups.items() if len(texts) >= args.examples_per_label]
    random.shuffle(eligible_labels)
    for label in eligible_labels:
        random.shuffle(groups[label])

    records = []
    label_cursor = 0
    label_use_counts = defaultdict(int)
    while len(records) < args.max_records:
        batch_labels = []
        attempts = 0
        while len(batch_labels) < args.labels_per_batch and attempts < len(eligible_labels) * 2:
            label = eligible_labels[label_cursor % len(eligible_labels)]
            label_cursor += 1
            attempts += 1
            start = label_use_counts[label]
            if start + args.examples_per_label <= len(groups[label]):
                batch_labels.append(label)
                label_use_counts[label] += args.examples_per_label
        if len(batch_labels) < args.labels_per_batch:
            break
        for label in batch_labels:
            start = label_use_counts[label] - args.examples_per_label
            for text in groups[label][start : start + args.examples_per_label]:
                records.append(
                    {
                        "source": "d0rj/geo-reviews-dataset-2023:exact_mteb_removed:diagnostic",
                        "objective": "labeled_text",
                        "text": "Определи тип места или услуги по отзыву \nотзыв: " + text,
                        "label": "rubric::" + label,
                        "loss": "supcon",
                        "metadata": {
                            "construction": "source_overlap_risky_georeview_rubric_supcon",
                            "labels_per_batch": args.labels_per_batch,
                            "examples_per_label": args.examples_per_label,
                        },
                    }
                )
                if len(records) >= args.max_records:
                    break
            if len(records) >= args.max_records:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.out),
        "records": len(records),
        "seed": args.seed,
        "labels_per_batch": args.labels_per_batch,
        "examples_per_label": args.examples_per_label,
        "benchmark_rows": benchmark_rows,
        "benchmark_unique_normalized_texts": len(benchmark_texts),
        "source_rows": source_rows,
        "exact_overlap_rows_removed": exact_overlap_rows,
        "exact_overlap_ratio": exact_overlap_rows / source_rows if source_rows else 0.0,
        "too_short_rows": too_short_rows,
        "missing_rubric_rows": missing_rubric_rows,
        "duplicate_clean_rows": duplicate_clean_rows,
        "eligible_labels": len(eligible_labels),
        "source_overlap_policy": (
            "Diagnostic only: exact mteb/georeview rows are removed, but the source "
            "corpus is still the same Yandex Georeview source used by the benchmark."
        ),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
