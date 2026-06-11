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


def read_source(path: Path) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    by_oecd_grnti: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    skipped = Counter()
    grnti_counts = Counter()
    oecd_counts = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            grnti_label = str(record.get("metadata", {}).get("grnti_label", "")).strip()
            number = grnti_number(grnti_label)
            oecd_label = GRNTI_NUMBER_TO_OECD29.get(number or -1)
            if oecd_label is None:
                skipped["unmapped_grnti"] += 1
                continue
            item = {
                "text": str(record["text"]),
                "grnti_label": grnti_label,
                "oecd_label": oecd_label,
                "file": record.get("metadata", {}).get("file"),
                "source_contamination_policy": record.get("metadata", {}).get("contamination_policy"),
            }
            by_oecd_grnti[oecd_label][grnti_label].append(item)
            grnti_counts[grnti_label] += 1
            oecd_counts[oecd_label] += 1
    return by_oecd_grnti, {
        "source_records": sum(grnti_counts.values()),
        "source_grnti_counts": dict(grnti_counts),
        "source_oecd_counts": dict(oecd_counts),
        "skipped": dict(skipped),
    }


def make_batches(
    by_oecd_grnti: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    batch_count: int,
    oecd_per_batch: int,
    grnti_per_oecd: int,
    positives_per_grnti: int,
    min_docs_per_grnti: int,
    seed: int,
    loss: str,
    oecd_weight: float,
    grnti_weight: float,
    encode_batch_size: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    usable: dict[str, list[str]] = {}
    for oecd_label, grnti_map in by_oecd_grnti.items():
        labels = sorted(
            [
                grnti_label
                for grnti_label, rows in grnti_map.items()
                if len(rows) >= min_docs_per_grnti
            ],
            key=lambda label: len(grnti_map[label]),
            reverse=True,
        )
        if len(labels) >= grnti_per_oecd:
            usable[oecd_label] = labels
    oecd_labels = sorted(usable)
    if len(oecd_labels) < oecd_per_batch:
        raise ValueError(f"Need {oecd_per_batch} usable OECD labels, got {len(oecd_labels)}")

    for oecd_label in oecd_labels:
        for grnti_label in usable[oecd_label]:
            rng.shuffle(by_oecd_grnti[oecd_label][grnti_label])

    cursors = Counter()
    sampled_grnti = Counter()
    sampled_oecd = Counter()
    records: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        selected_oecd = rng.sample(oecd_labels, oecd_per_batch)
        for oecd_label in selected_oecd:
            selected_grnti = rng.sample(usable[oecd_label], grnti_per_oecd)
            for grnti_label in selected_grnti:
                values = by_oecd_grnti[oecd_label][grnti_label]
                cursor_key = f"{oecd_label}\t{grnti_label}"
                for text_index in range(positives_per_grnti):
                    cursor = cursors[cursor_key] % len(values)
                    item = values[cursor]
                    cursors[cursor_key] += 1
                    if cursors[cursor_key] % len(values) == 0:
                        rng.shuffle(values)
                    records.append(
                        {
                            "source": "kaggle/ergkerg/russian-scientific-articles:joint_grnti_oecd_hierarchical",
                            "objective": "hierarchical_labeled_text",
                            "text": item["text"],
                            "labels": {
                                "grnti": grnti_label,
                                "oecd": oecd_label,
                            },
                            "label_weights": {
                                "grnti": grnti_weight,
                                "oecd": oecd_weight,
                            },
                            "loss": loss,
                            **({"encode_batch_size": encode_batch_size} if encode_batch_size else {}),
                            "metadata": {
                                "grnti_label": grnti_label,
                                "mapped_oecd_label": oecd_label,
                                "file": item["file"],
                                "batch_index": batch_index,
                                "text_index": text_index,
                                "contamination_policy": (
                                    "Derived from audited Kaggle GRNTI article JSONL after "
                                    "RuSciBench GRNTI/OECD title/prefix overlap removal. "
                                    "GRNTI labels are mapped to the public RuSciBench OECD-style "
                                    "label inventory using label names only; no benchmark rows or "
                                    "released model are used."
                                ),
                            },
                        }
                    )
                    sampled_grnti[grnti_label] += 1
                    sampled_oecd[oecd_label] += 1
    return records, {
        "records": len(records),
        "batch_size": oecd_per_batch * grnti_per_oecd * positives_per_grnti,
        "usable_oecd_labels": oecd_labels,
        "usable_grnti_by_oecd": usable,
        "sampled_grnti_counts": dict(sampled_grnti),
        "sampled_oecd_counts": dict(sampled_oecd),
        "encode_batch_size": encode_batch_size,
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
        default=ROOT / "data/contrastive/fair_joint_grnti_oecd_hier_b8_400_seed2941.jsonl",
    )
    parser.add_argument("--batch-count", type=int, default=400)
    parser.add_argument("--oecd-per-batch", type=int, default=2)
    parser.add_argument("--grnti-per-oecd", type=int, default=2)
    parser.add_argument("--positives-per-grnti", type=int, default=2)
    parser.add_argument("--min-docs-per-grnti", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2941)
    parser.add_argument("--loss", choices=["supcon", "circle", "multi_similarity"], default="circle")
    parser.add_argument("--oecd-weight", type=float, default=1.0)
    parser.add_argument("--grnti-weight", type=float, default=1.0)
    parser.add_argument("--encode-batch-size", type=int, default=0)
    args = parser.parse_args()

    by_oecd_grnti, source_summary = read_source(args.source)
    records, batch_summary = make_batches(
        by_oecd_grnti,
        batch_count=args.batch_count,
        oecd_per_batch=args.oecd_per_batch,
        grnti_per_oecd=args.grnti_per_oecd,
        positives_per_grnti=args.positives_per_grnti,
        min_docs_per_grnti=args.min_docs_per_grnti,
        seed=args.seed,
        loss=args.loss,
        oecd_weight=args.oecd_weight,
        grnti_weight=args.grnti_weight,
        encode_batch_size=args.encode_batch_size or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "mapping": GRNTI_NUMBER_TO_OECD29,
        "seed": args.seed,
        "loss": args.loss,
        "oecd_weight": args.oecd_weight,
        "grnti_weight": args.grnti_weight,
        "encode_batch_size": args.encode_batch_size or None,
        "source_summary": source_summary,
        "batch_summary": batch_summary,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
