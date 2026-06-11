#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_oecd_by_label(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_jsonl(path):
        label = str(record.get("label", "")).strip()
        if label:
            by_label[label].append(record)
    return by_label


def make_batches(
    *,
    oecd_by_label: dict[str, list[dict[str, Any]]],
    geo_records: list[dict[str, Any]],
    cedr_records: list[dict[str, Any]],
    grnti_records: list[dict[str, Any]],
    batch_count: int,
    seed: int,
    ridge_lambda: float,
    encode_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    labels = sorted([label for label, rows in oecd_by_label.items() if len(rows) >= 8])
    if len(labels) < 2:
        raise ValueError("Need at least two OECD labels")
    for label in labels:
        rng.shuffle(oecd_by_label[label])
    rng.shuffle(geo_records)
    rng.shuffle(cedr_records)
    rng.shuffle(grnti_records)

    oecd_cursor = Counter()
    geo_cursor = 0
    cedr_cursor = 0
    grnti_cursor = 0
    sampled = Counter()
    output: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        selected_labels = rng.sample(labels, 2)
        for label in selected_labels:
            rows = oecd_by_label[label]
            for item_index in range(3):
                cursor = oecd_cursor[label] % len(rows)
                source = rows[cursor]
                oecd_cursor[label] += 1
                if oecd_cursor[label] % len(rows) == 0:
                    rng.shuffle(rows)
                role = "support" if item_index < 2 else "query"
                output.append(
                    {
                        "source": "kaggle/ergkerg/russian-scientific-articles:oecd29_linear_probe_mixed",
                        "objective": "linear_probe_labeled_text",
                        "text": source["text"],
                        "label": label,
                        "role": role,
                        "ridge_lambda": ridge_lambda,
                        "use_bias": True,
                        "encode_batch_size": encode_batch_size,
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
                sampled[f"oecd::{label}"] += 1

        for _ in range(4):
            record = dict(geo_records[geo_cursor % len(geo_records)])
            geo_cursor += 1
            record.setdefault("metadata", {})
            record["metadata"] = {**record["metadata"], "batch_index": batch_index}
            output.append(record)
            sampled["geo"] += 1

        record = dict(cedr_records[cedr_cursor % len(cedr_records)])
        cedr_cursor += 1
        record.setdefault("metadata", {})
        record["metadata"] = {**record["metadata"], "batch_index": batch_index}
        output.append(record)
        sampled["cedr"] += 1

        record = dict(grnti_records[grnti_cursor % len(grnti_records)])
        grnti_cursor += 1
        record.setdefault("metadata", {})
        record["metadata"] = {**record["metadata"], "batch_index": batch_index}
        output.append(record)
        sampled["grnti"] += 1

    return output, {
        "records": len(output),
        "batch_count": batch_count,
        "batch_size": 12,
        "layout": "2 OECD labels x (2 support + 1 query) + 4 Georeview SupCon + 1 CEDR contrastive + 1 GRNTI KNN",
        "ridge_lambda": ridge_lambda,
        "encode_batch_size": encode_batch_size,
        "usable_oecd_labels": labels,
        "sampled_counts": dict(sampled),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oecd-source",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_to_oecd29_circle_b4_2400_seed2661.jsonl",
    )
    parser.add_argument(
        "--geo-source",
        type=Path,
        default=ROOT / "data/contrastive/fair_geo_cluster_labeled_supcon_6000_seed2034.jsonl",
    )
    parser.add_argument(
        "--cedr-source",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_3200.jsonl",
    )
    parser.add_argument(
        "--grnti-source",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_tfidf_knn_episode_p2_n3s2_900_seed2591.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/fair_oecd29_linear_probe_mixed_b12_60_seed2971.jsonl",
    )
    parser.add_argument("--batch-count", type=int, default=60)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2971)
    args = parser.parse_args()

    records, summary = make_batches(
        oecd_by_label=load_oecd_by_label(args.oecd_source),
        geo_records=load_jsonl(args.geo_source),
        cedr_records=load_jsonl(args.cedr_source),
        grnti_records=load_jsonl(args.grnti_source),
        batch_count=args.batch_count,
        seed=args.seed,
        ridge_lambda=args.ridge_lambda,
        encode_batch_size=args.encode_batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    full_summary = {
        "output": str(args.output),
        "sources": {
            "oecd": str(args.oecd_source),
            "geo": str(args.geo_source),
            "cedr": str(args.cedr_source),
            "grnti": str(args.grnti_source),
        },
        "seed": args.seed,
        **summary,
        "fairness": (
            "All component files are previously audited clean proxy/rehearsal datasets. "
            "No benchmark rows, released model outputs, released teacher, or released "
            "latent weights are used."
        ),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(full_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
