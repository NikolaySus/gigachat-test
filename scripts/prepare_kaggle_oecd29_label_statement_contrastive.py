#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_kaggle_grnti_to_oecd29_metric_batches import GRNTI_NUMBER_TO_OECD29


ROOT = Path(__file__).resolve().parents[1]


def grnti_number(label: str) -> int | None:
    match = re.match(r"\s*(\d+)\b", label)
    return int(match.group(1)) if match else None


def label_statement(label: str) -> str:
    return (
        "Научная статья относится к тематической категории OECD/RuSciBench: "
        f"{label}."
    )


def read_source(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_oecd: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            grnti_label = str(record.get("metadata", {}).get("grnti_label", "")).strip()
            number = grnti_number(grnti_label)
            oecd_label = GRNTI_NUMBER_TO_OECD29.get(number or -1)
            if oecd_label is None:
                skipped["unmapped_grnti"] += 1
                continue
            by_oecd[oecd_label].append(
                {
                    "text": str(record["text"]),
                    "grnti_label": grnti_label,
                    "oecd_label": oecd_label,
                    "file": record.get("metadata", {}).get("file"),
                    "source_contamination_policy": record.get("metadata", {}).get("contamination_policy"),
                }
            )
    return by_oecd, {
        "source_records": sum(len(rows) for rows in by_oecd.values()),
        "source_oecd_counts": {label: len(rows) for label, rows in by_oecd.items()},
        "skipped": dict(skipped),
    }


def make_records(
    by_oecd: dict[str, list[dict[str, Any]]],
    *,
    rows_per_label: int,
    negatives_per_record: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    labels = sorted(label for label, rows in by_oecd.items() if rows)
    if len(labels) <= negatives_per_record:
        raise ValueError(f"Need more labels than negatives_per_record={negatives_per_record}; got {len(labels)}")

    records: list[dict[str, Any]] = []
    sampled = Counter()
    for label in labels:
        rows = list(by_oecd[label])
        rng.shuffle(rows)
        selected = rows[: min(rows_per_label, len(rows))]
        for index, item in enumerate(selected):
            negatives = [candidate for candidate in labels if candidate != label]
            rng.shuffle(negatives)
            chosen_negatives = negatives[:negatives_per_record]
            records.append(
                {
                    "source": "kaggle/ergkerg/russian-scientific-articles:oecd29_label_statement",
                    "objective": "contrastive",
                    "query": item["text"],
                    "positive": label_statement(label),
                    "negatives": [label_statement(negative) for negative in chosen_negatives],
                    "metadata": {
                        "mapped_oecd_label": label,
                        "grnti_label": item["grnti_label"],
                        "file": item["file"],
                        "row_index_within_label": index,
                        "negative_oecd_labels": chosen_negatives,
                        "construction": "text_to_public_oecd_label_statement",
                        "contamination_policy": (
                            "Derived from audited Kaggle GRNTI article JSONL after "
                            "RuSciBench GRNTI/OECD title/prefix overlap removal. "
                            "Uses public OECD-style label names only; no benchmark rows, "
                            "released model outputs, or released latent weights."
                        ),
                    },
                }
            )
            sampled[label] += 1
    rng.shuffle(records)
    return records, {
        "records": len(records),
        "labels": labels,
        "sampled_oecd_counts": dict(sampled),
        "rows_per_label": rows_per_label,
        "negatives_per_record": negatives_per_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_labeled_circle_b4_1600_seed2551.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/fair_kaggle_oecd29_label_statement_neg8_1350_seed3141.jsonl",
    )
    parser.add_argument("--rows-per-label", type=int, default=50)
    parser.add_argument("--negatives-per-record", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3141)
    args = parser.parse_args()

    by_oecd, source_summary = read_source(args.source)
    records, record_summary = make_records(
        by_oecd,
        rows_per_label=args.rows_per_label,
        negatives_per_record=args.negatives_per_record,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "seed": args.seed,
        "mapping": GRNTI_NUMBER_TO_OECD29,
        "source_summary": source_summary,
        "record_summary": record_summary,
        "contamination_policy": "Fair: inherits RuSciBench title/prefix overlap filtering from source JSONL.",
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
