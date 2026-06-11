#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from giga_model_utils import ModelLoadConfig, encode_texts, load_giga_embeddings


ROOT = Path(__file__).resolve().parents[1]
FRONTIER = (
    ROOT
    / "experiments/exp01_reinit_fair/checkpoints/"
    "fair_grandmaster_circle_repair_from_taxv2_step20_lr5e7_anchor15_reh05_b8_40_4096_eager_frozenrepro/"
    "step-20.pt"
)


def read_source(path: Path) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = str(record["text"]).strip()
            label = str(record["metadata"]["grnti_label"])
            file_name = str(record["metadata"]["file"])
            key = f"{label}\0{file_name}\0{text[:200]}"
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "label": label, "file": file_name})
    return rows


def choose_positive(
    sims: np.ndarray,
    anchor: int,
    same_indices: list[int],
    *,
    min_positive_similarity: float,
) -> int | None:
    candidates = [idx for idx in same_indices if idx != anchor]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda idx: float(sims[anchor, idx]), reverse=True)
    for idx in ranked:
        if float(sims[anchor, idx]) >= min_positive_similarity:
            return idx
    return ranked[0]


def choose_negatives(
    sims: np.ndarray,
    rows: list[dict[str, Any]],
    anchor: int,
    *,
    negatives_per_record: int,
) -> list[int]:
    anchor_label = rows[anchor]["label"]
    ranked = sorted(
        (idx for idx, row in enumerate(rows) if row["label"] != anchor_label),
        key=lambda idx: float(sims[anchor, idx]),
        reverse=True,
    )
    selected: list[int] = []
    used_labels: set[str] = set()
    for idx in ranked:
        label = rows[idx]["label"]
        if label in used_labels:
            continue
        selected.append(idx)
        used_labels.add(label)
        if len(selected) >= negatives_per_record:
            return selected
    for idx in ranked:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= negatives_per_record:
            break
    return selected


def balanced_anchor_order(rows: list[dict[str, Any]], *, max_records: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_label[row["label"]].append(idx)
    for indices in by_label.values():
        rng.shuffle(indices)

    labels = list(by_label)
    rng.shuffle(labels)
    selected: list[int] = []
    cursor = 0
    while len(selected) < max_records:
        added = False
        for label in labels:
            values = by_label[label]
            if cursor < len(values):
                selected.append(values[cursor])
                added = True
                if len(selected) >= max_records:
                    break
        if not added:
            break
        cursor += 1
    rng.shuffle(selected)
    return selected


def write_records(
    output: Path,
    rows: list[dict[str, Any]],
    sims: np.ndarray,
    *,
    max_records: int,
    negatives_per_record: int,
    min_positive_similarity: float,
    seed: int,
    mining_checkpoint: Path,
) -> dict[str, Any]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_label[row["label"]].append(idx)

    positives: list[float] = []
    negatives: list[float] = []
    skipped = Counter()
    label_counts = Counter()
    records: list[dict[str, Any]] = []
    for anchor in balanced_anchor_order(rows, max_records=max_records * 2, seed=seed):
        if len(records) >= max_records:
            break
        positive = choose_positive(
            sims,
            anchor,
            by_label[rows[anchor]["label"]],
            min_positive_similarity=min_positive_similarity,
        )
        if positive is None:
            skipped["no_positive"] += 1
            continue
        negative_indices = choose_negatives(sims, rows, anchor, negatives_per_record=negatives_per_record)
        if len(negative_indices) < negatives_per_record:
            skipped["not_enough_negatives"] += 1
            continue

        pos_sim = float(sims[anchor, positive])
        neg_sims = [float(sims[anchor, idx]) for idx in negative_indices]
        positives.append(pos_sim)
        negatives.extend(neg_sims)
        label_counts[rows[anchor]["label"]] += 1
        records.append(
            {
                "source": "kaggle/ergkerg/russian-scientific-articles:grnti_frontier_hard_contrastive",
                "objective": "contrastive",
                "query": rows[anchor]["text"],
                "positive": rows[positive]["text"],
                "negatives": [rows[idx]["text"] for idx in negative_indices],
                "metadata": {
                    "grnti_label": rows[anchor]["label"],
                    "anchor_file": rows[anchor]["file"],
                    "positive_file": rows[positive]["file"],
                    "positive_similarity": pos_sim,
                    "negative_similarities": neg_sims,
                    "negative_labels": [rows[idx]["label"] for idx in negative_indices],
                    "negative_files": [rows[idx]["file"] for idx in negative_indices],
                    "mining_checkpoint": str(mining_checkpoint.relative_to(ROOT)),
                    "contamination_policy": (
                        "Derived from the audited cleaned Kaggle GRNTI set; RuSciBench GRNTI/OECD "
                        "title/prefix overlaps were removed before mining. Current fair checkpoint "
                        "used for neighbor mining; no released model or benchmark rows used."
                    ),
                },
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "records": len(records),
        "skipped": dict(skipped),
        "sampled_label_counts": dict(label_counts),
        "positive_similarity": {
            "mean": mean(positives) if positives else None,
            "min": min(positives) if positives else None,
            "max": max(positives) if positives else None,
        },
        "negative_similarity": {
            "mean": mean(negatives) if negatives else None,
            "min": min(negatives) if negatives else None,
            "max": max(negatives) if negatives else None,
        },
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
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_hard_contrastive_frontier_b2_1200_seed2571.jsonl",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--latent-checkpoint", type=Path, default=FRONTIER)
    parser.add_argument("--max-records", type=int, default=1200)
    parser.add_argument("--negatives-per-record", type=int, default=5)
    parser.add_argument("--min-positive-similarity", type=float, default=0.4)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2571)
    args = parser.parse_args()

    rows = read_source(args.source)
    label_counts = Counter(row["label"] for row in rows)
    rows = [row for row in rows if label_counts[row["label"]] >= 2]
    print(f"Loaded {len(rows)} unique source rows across {len(set(row['label'] for row in rows))} labels")

    tokenizer, model = load_giga_embeddings(
        ModelLoadConfig(
            max_length=args.max_length,
            batch_size=args.encode_batch_size,
            attn_implementation="eager",
            local_files_only=True,
            latent_checkpoint=args.latent_checkpoint,
            torch_dtype=torch.bfloat16,
        )
    )
    embeddings = encode_texts(
        [row["text"] for row in rows],
        tokenizer,
        model,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sims = embeddings @ embeddings.T
    summary = write_records(
        args.output,
        rows,
        sims,
        max_records=args.max_records,
        negatives_per_record=args.negatives_per_record,
        min_positive_similarity=args.min_positive_similarity,
        seed=args.seed,
        mining_checkpoint=args.latent_checkpoint,
    )
    summary.update(
        {
            "source": str(args.source),
            "output": str(args.output),
            "latent_checkpoint": str(args.latent_checkpoint),
            "source_rows": len(rows),
            "source_label_counts": dict(label_counts),
            "max_length": args.max_length,
            "encode_batch_size": args.encode_batch_size,
            "seed": args.seed,
        }
    )
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
