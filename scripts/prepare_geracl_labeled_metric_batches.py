#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def strip_query_prefix(text: str) -> str:
    marker = "\nQuery:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    if text.startswith("Query:"):
        return text[len("Query:") :].strip()
    return text.strip()


def read_clusters(path: Path) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            metadata = record.get("metadata") or {}
            scenario = str(metadata.get("scenario") or "").strip()
            class_name = str(metadata.get("positive_class") or "").strip()
            if not scenario or not class_name:
                continue
            label = f"{scenario} :: {class_name}"
            candidates = [
                strip_query_prefix(str(record.get("query") or "")),
                str(record.get("positive") or "").strip(),
            ]
            for text in candidates:
                if len(text) < 80 or text in seen[label]:
                    continue
                seen[label].add(text)
                clusters[label].append(text)
    return {label: texts for label, texts in clusters.items() if len(texts) >= 2}


def make_batches(
    clusters: dict[str, list[str]],
    *,
    batch_count: int,
    labels_per_batch: int,
    positives_per_label: int,
    seed: int,
    loss_name: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    labels = list(clusters)
    if len(labels) < labels_per_batch:
        raise ValueError(f"Need at least {labels_per_batch} labels, got {len(labels)}")
    rows: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        selected = rng.sample(labels, labels_per_batch)
        for label in selected:
            texts = clusters[label]
            chosen = rng.sample(texts, positives_per_label)
            for text_index, text in enumerate(chosen):
                rows.append(
                    {
                        "source": "deepvk/GeRaCl_synthethic_dataset:scenario_labeled_metric",
                        "objective": "labeled_text",
                        "text": text,
                        "label": f"geracl::{label}",
                        "loss": loss_name,
                        "metadata": {
                            "group": label,
                            "batch_index": batch_index,
                            "text_index": text_index,
                            "contamination_policy": "Derived from audited clean GeRaCl scenario/class metadata; no ruMTEB rows.",
                        },
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_geracl.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_geracl_labeled_circle_b8_1920_seed2531.jsonl",
    )
    parser.add_argument("--batch-count", type=int, default=240)
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--positives-per-label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2531)
    parser.add_argument("--loss", choices=["supcon", "circle", "multi_similarity"], default="circle")
    args = parser.parse_args()

    clusters = read_clusters(args.input)
    records = make_batches(
        clusters,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        positives_per_label=args.positives_per_label,
        seed=args.seed,
        loss_name=args.loss,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "records": len(records),
        "usable_labels": len(clusters),
        "batch_count": args.batch_count,
        "batch_size": args.labels_per_batch * args.positives_per_label,
        "labels_per_batch": args.labels_per_batch,
        "positives_per_label": args.positives_per_label,
        "loss": args.loss,
        "seed": args.seed,
        "contamination_policy": "Clean GeRaCl source already audited as non-contaminated; no ruMTEB rows introduced.",
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
