#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def group_labeled(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("objective") != "labeled_text":
            continue
        grouped[str(record["label"])].append(record)
    return {label: values for label, values in grouped.items() if len(values) >= 2}


def take_wrapped(values: list[dict[str, Any]], start: int, count: int) -> list[dict[str, Any]]:
    return [values[(start + offset) % len(values)] for offset in range(count)]


def build_batches(
    *,
    oecd_records: list[dict[str, Any]],
    grnti_records: list[dict[str, Any]],
    geo_by_label: dict[str, list[dict[str, Any]]],
    steps: int,
    seed: int,
    oecd_per_batch: int,
    grnti_per_batch: int,
    geo_labels_per_batch: int,
    geo_examples_per_label: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    oecd_order = list(range(len(oecd_records)))
    grnti_order = list(range(len(grnti_records)))
    geo_labels = sorted(geo_by_label)
    rng.shuffle(oecd_order)
    rng.shuffle(grnti_order)
    rng.shuffle(geo_labels)
    geo_offsets = {label: 0 for label in geo_labels}

    output: list[dict[str, Any]] = []
    for step in range(steps):
        if step and step % len(oecd_order) == 0:
            rng.shuffle(oecd_order)
        if step and step % len(grnti_order) == 0:
            rng.shuffle(grnti_order)
        if step and step % len(geo_labels) == 0:
            rng.shuffle(geo_labels)

        oecd_batch: list[dict[str, Any]] = []
        for offset in range(oecd_per_batch):
            item = dict(oecd_records[oecd_order[(step * oecd_per_batch + offset) % len(oecd_order)]])
            item.setdefault("metadata", {})["mixed_batch_role"] = "oecd29_knn"
            oecd_batch.append(item)

        grnti_batch: list[dict[str, Any]] = []
        for offset in range(grnti_per_batch):
            item = dict(grnti_records[grnti_order[(step * grnti_per_batch + offset) % len(grnti_order)]])
            item.setdefault("metadata", {})["mixed_batch_role"] = "grnti_knn"
            grnti_batch.append(item)

        geo_records: list[dict[str, Any]] = []
        for label_offset in range(geo_labels_per_batch):
            label = geo_labels[(step * geo_labels_per_batch + label_offset) % len(geo_labels)]
            values = geo_by_label[label]
            picked = take_wrapped(values, geo_offsets[label], geo_examples_per_label)
            geo_offsets[label] = (geo_offsets[label] + geo_examples_per_label) % len(values)
            for record in picked:
                item = dict(record)
                item.setdefault("metadata", {})["mixed_batch_role"] = "georeview_supcon"
                geo_records.append(item)

        output.extend([*oecd_batch, *grnti_batch, *geo_records])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oecd",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_oecd29_tfidf_knn_episode_p2_n4s1_1200_seed2671.jsonl",
    )
    parser.add_argument(
        "--grnti",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_tfidf_knn_episode_p2_n3s2_900_seed2591.jsonl",
    )
    parser.add_argument(
        "--geo",
        type=Path,
        default=ROOT / "data/contrastive/fair_geo_cluster_labeled_supcon_6000_seed2034.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/fair_mixed_oecd29_grnti_geo_knn_supcon_b6_480_seed2681.jsonl",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2681)
    parser.add_argument("--oecd-per-batch", type=int, default=1)
    parser.add_argument("--grnti-per-batch", type=int, default=1)
    parser.add_argument("--geo-labels-per-batch", type=int, default=2)
    parser.add_argument("--geo-examples-per-label", type=int, default=2)
    args = parser.parse_args()

    oecd = read_jsonl(args.oecd)
    grnti = read_jsonl(args.grnti)
    geo = read_jsonl(args.geo)
    geo_by_label = group_labeled(geo)
    if len(geo_by_label) < 2:
        raise ValueError("Need at least two Georeview labels with two examples")

    records = build_batches(
        oecd_records=oecd,
        grnti_records=grnti,
        geo_by_label=geo_by_label,
        steps=args.steps,
        seed=args.seed,
        oecd_per_batch=args.oecd_per_batch,
        grnti_per_batch=args.grnti_per_batch,
        geo_labels_per_batch=args.geo_labels_per_batch,
        geo_examples_per_label=args.geo_examples_per_label,
    )
    batch_size = args.oecd_per_batch + args.grnti_per_batch + (
        args.geo_labels_per_batch * args.geo_examples_per_label
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output),
        "steps": args.steps,
        "records": len(records),
        "batch_size": batch_size,
        "batch_layout": (
            f"{args.oecd_per_batch} OECD29 KNN + {args.grnti_per_batch} GRNTI KNN + "
            f"{args.geo_labels_per_batch}x{args.geo_examples_per_label} Georeview SupCon"
        ),
        "sources": {
            "oecd": str(args.oecd),
            "grnti": str(args.grnti),
            "geo": str(args.geo),
        },
        "geo_labels_available": len(geo_by_label),
        "seed": args.seed,
        "fairness": (
            "All sources are previously audited clean proxy/rehearsal datasets. "
            "No benchmark rows, released model outputs, released teacher, or released latent weights are used."
        ),
    }
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
