#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\ufeff", " ")).strip()
    return text


def read_train_rows() -> tuple[dict[str, list[str]], dict[str, Any]]:
    ds = load_dataset("DeepPavlov/minds14_ru", "default", split="train")
    by_label: dict[str, list[str]] = defaultdict(list)
    skipped = Counter()
    seen: set[tuple[str, str]] = set()
    for row in ds:
        text = clean_text(str(row.get("utterance") or ""))
        label = f"minds14_intent::{int(row['label']):02d}"
        if len(text) < 8:
            skipped["short_text"] += 1
            continue
        key = (label, text.lower())
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        by_label[label].append(text)
    return by_label, {
        "dataset": "DeepPavlov/minds14_ru",
        "config": "default",
        "split": "train",
        "records": sum(len(values) for values in by_label.values()),
        "labels": len(by_label),
        "label_counts": {label: len(values) for label, values in sorted(by_label.items())},
        "skipped": dict(skipped),
        "contamination_policy": (
            "Uses only MInDS-14 RU train split, which is not part of the target ruMTEB(rus v1.1) "
            "evaluation set used in this project. MTEB MASSIVE intent/scenario rows are not used."
        ),
    }


def build_batches(
    by_label: dict[str, list[str]],
    *,
    batch_count: int,
    labels_per_batch: int,
    supports_per_label: int,
    queries_per_label: int,
    seed: int,
    ridge_lambda: float,
    encode_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    labels = [label for label, texts in by_label.items() if len(texts) >= supports_per_label + queries_per_label]
    if len(labels) < labels_per_batch:
        raise ValueError(f"Need {labels_per_batch} labels, got {len(labels)}")
    for label in labels:
        rng.shuffle(by_label[label])
    rng.shuffle(labels)
    cursors = Counter()
    sampled = Counter()
    output: list[dict[str, Any]] = []
    per_label = supports_per_label + queries_per_label
    for batch_index in range(batch_count):
        if batch_index and batch_index % len(labels) == 0:
            rng.shuffle(labels)
        selected = rng.sample(labels, labels_per_batch)
        for label in selected:
            texts = by_label[label]
            for item_index in range(per_label):
                cursor = cursors[label] % len(texts)
                text = texts[cursor]
                cursors[label] += 1
                if cursors[label] % len(texts) == 0:
                    rng.shuffle(texts)
                role = "support" if item_index < supports_per_label else "query"
                output.append(
                    {
                        "source": "DeepPavlov/minds14_ru:train_linear_probe",
                        "objective": "linear_probe_labeled_text",
                        "text": text,
                        "label": label,
                        "role": role,
                        "ridge_lambda": ridge_lambda,
                        "use_bias": True,
                        "encode_batch_size": encode_batch_size,
                        "metadata": {
                            "batch_index": batch_index,
                            "role": role,
                            "label": label,
                            "contamination_policy": (
                                "MInDS-14 RU train only; target ruMTEB MASSIVE and other ruMTEB rows are excluded."
                            ),
                        },
                    }
                )
                sampled[label] += 1
    return output, {
        "records": len(output),
        "batch_count": batch_count,
        "batch_size": labels_per_batch * per_label,
        "labels_per_batch": labels_per_batch,
        "supports_per_label": supports_per_label,
        "queries_per_label": queries_per_label,
        "sampled_label_counts": dict(sampled),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/fair_minds14_ru_intent_linear_probe_b12_80_seed3111.jsonl",
    )
    parser.add_argument("--batch-count", type=int, default=80)
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--supports-per-label", type=int, default=2)
    parser.add_argument("--queries-per-label", type=int, default=1)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3111)
    args = parser.parse_args()

    by_label, source_summary = read_train_rows()
    records, batch_summary = build_batches(
        by_label,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        supports_per_label=args.supports_per_label,
        queries_per_label=args.queries_per_label,
        seed=args.seed,
        ridge_lambda=args.ridge_lambda,
        encode_batch_size=args.encode_batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "output": str(args.output),
        "seed": args.seed,
        "source_summary": source_summary,
        "batch_summary": batch_summary,
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
