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
        if record.get("objective") == "labeled_text":
            grouped[str(record["label"])].append(record)
    return {label: values for label, values in grouped.items() if len(values) >= 2}


def only_objective(records: list[dict[str, Any]], objective: str) -> list[dict[str, Any]]:
    filtered = [record for record in records if record.get("objective") == objective]
    if not filtered:
        raise ValueError(f"No records with objective={objective}")
    return filtered


def clone_with_role(record: dict[str, Any], role: str, batch_index: int) -> dict[str, Any]:
    item = dict(record)
    metadata = dict(item.get("metadata") or {})
    metadata["mixed_batch_role"] = role
    metadata["mixed_batch"] = batch_index
    metadata["mixed_dataset"] = "fair_frontier_mixed_cedr_oecd_grnti_geo_b7"
    item["metadata"] = metadata
    return item


def build_batches(
    *,
    geo_by_label: dict[str, list[dict[str, Any]]],
    cedr_records: list[dict[str, Any]],
    oecd_records: list[dict[str, Any]],
    grnti_records: list[dict[str, Any]],
    steps: int,
    seed: int,
    geo_labels_per_batch: int,
    geo_examples_per_label: int,
    cedr_per_batch: int,
    cedr_role: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    labels = sorted(geo_by_label)
    if len(labels) < geo_labels_per_batch:
        raise ValueError("Not enough Georeview labels")
    rng.shuffle(labels)
    label_offsets = {label: 0 for label in labels}
    cedr_order = list(range(len(cedr_records)))
    oecd_order = list(range(len(oecd_records)))
    grnti_order = list(range(len(grnti_records)))
    rng.shuffle(cedr_order)
    rng.shuffle(oecd_order)
    rng.shuffle(grnti_order)

    output: list[dict[str, Any]] = []
    for step in range(steps):
        if step and step % len(labels) == 0:
            rng.shuffle(labels)
        if step and step % len(cedr_order) == 0:
            rng.shuffle(cedr_order)
        if step and step % len(oecd_order) == 0:
            rng.shuffle(oecd_order)
        if step and step % len(grnti_order) == 0:
            rng.shuffle(grnti_order)

        for label_offset in range(geo_labels_per_batch):
            label = labels[(step * geo_labels_per_batch + label_offset) % len(labels)]
            values = geo_by_label[label]
            start = label_offsets[label]
            for offset in range(geo_examples_per_label):
                record = values[(start + offset) % len(values)]
                output.append(clone_with_role(record, "georeview_supcon", step))
            label_offsets[label] = (start + geo_examples_per_label) % len(values)

        for cedr_offset in range(cedr_per_batch):
            output.append(
                clone_with_role(
                    cedr_records[
                        cedr_order[
                            (step * cedr_per_batch + cedr_offset) % len(cedr_order)
                        ]
                    ],
                    cedr_role,
                    step,
                )
            )
        output.append(
            clone_with_role(
                oecd_records[oecd_order[step % len(oecd_order)]],
                "oecd29_knn",
                step,
            )
        )
        output.append(
            clone_with_role(
                grnti_records[grnti_order[step % len(grnti_order)]],
                "grnti_knn",
                step,
            )
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geo",
        type=Path,
        default=ROOT / "data/contrastive/fair_geo_cluster_labeled_supcon_6000_seed2034.jsonl",
    )
    parser.add_argument(
        "--cedr",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_3200.jsonl",
    )
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
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/fair_frontier_mixed_cedr_oecd_grnti_geo_b7_420_seed2791.jsonl",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2791)
    parser.add_argument("--geo-labels-per-batch", type=int, default=2)
    parser.add_argument("--geo-examples-per-label", type=int, default=2)
    parser.add_argument("--cedr-per-batch", type=int, default=1)
    parser.add_argument("--cedr-objective", default="contrastive")
    parser.add_argument("--cedr-role", default=None)
    args = parser.parse_args()

    geo = read_jsonl(args.geo)
    cedr = only_objective(read_jsonl(args.cedr), args.cedr_objective)
    oecd = only_objective(read_jsonl(args.oecd), "knn_classification")
    grnti = only_objective(read_jsonl(args.grnti), "knn_classification")
    geo_by_label = group_labeled(geo)
    cedr_role = args.cedr_role or f"cedr_{args.cedr_objective}"

    records = build_batches(
        geo_by_label=geo_by_label,
        cedr_records=cedr,
        oecd_records=oecd,
        grnti_records=grnti,
        steps=args.steps,
        seed=args.seed,
        geo_labels_per_batch=args.geo_labels_per_batch,
        geo_examples_per_label=args.geo_examples_per_label,
        cedr_per_batch=args.cedr_per_batch,
        cedr_role=cedr_role,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    batch_size = args.geo_labels_per_batch * args.geo_examples_per_label + args.cedr_per_batch + 2
    summary = {
        "output": str(args.output),
        "records": len(records),
        "steps": args.steps,
        "batch_size": batch_size,
        "batch_layout": (
            f"{args.geo_labels_per_batch}x{args.geo_examples_per_label} Georeview SupCon + "
            f"{args.cedr_per_batch} CEDR {args.cedr_objective} + 1 OECD29 KNN + 1 GRNTI KNN"
        ),
        "cedr_objective": args.cedr_objective,
        "sources": {
            "geo": str(args.geo),
            "cedr": str(args.cedr),
            "oecd": str(args.oecd),
            "grnti": str(args.grnti),
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
