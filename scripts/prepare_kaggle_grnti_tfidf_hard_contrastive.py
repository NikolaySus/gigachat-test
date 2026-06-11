#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]


def normalize_for_tfidf(text: str) -> str:
    text = text.lower().replace("\ufeff", " ")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^0-9a-zа-яё\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_source(path: Path, *, max_chars: int) -> list[dict[str, Any]]:
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
            rows.append(
                {
                    "text": text[:max_chars],
                    "label": label,
                    "file": file_name,
                    "tfidf_text": normalize_for_tfidf(text[:max_chars]),
                }
            )
    label_counts = Counter(row["label"] for row in rows)
    return [row for row in rows if label_counts[row["label"]] >= 2 and len(row["tfidf_text"]) >= 200]


def diverse_nearest_different_labels(
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
        if len(selected) == negatives_per_record:
            return selected
    for idx in ranked:
        if idx not in selected:
            selected.append(idx)
        if len(selected) == negatives_per_record:
            break
    return selected


def nearest_same_label(sims: np.ndarray, rows: list[dict[str, Any]], anchor: int) -> int | None:
    candidates = [idx for idx, row in enumerate(rows) if row["label"] == rows[anchor]["label"] and idx != anchor]
    if not candidates:
        return None
    return max(candidates, key=lambda idx: float(sims[anchor, idx]))


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


def build_records(
    rows: list[dict[str, Any]],
    sims: np.ndarray,
    *,
    max_records: int,
    negatives_per_record: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = Counter()
    pos_sims: list[float] = []
    neg_sims: list[float] = []
    label_counts = Counter()
    for anchor in balanced_anchor_order(rows, max_records=max_records * 2, seed=seed):
        if len(records) >= max_records:
            break
        positive = nearest_same_label(sims, rows, anchor)
        if positive is None:
            skipped["no_positive"] += 1
            continue
        negatives = diverse_nearest_different_labels(
            sims,
            rows,
            anchor,
            negatives_per_record=negatives_per_record,
        )
        if len(negatives) < negatives_per_record:
            skipped["not_enough_negatives"] += 1
            continue
        pos_sim = float(sims[anchor, positive])
        current_neg_sims = [float(sims[anchor, idx]) for idx in negatives]
        pos_sims.append(pos_sim)
        neg_sims.extend(current_neg_sims)
        label_counts[rows[anchor]["label"]] += 1
        records.append(
            {
                "source": "kaggle/ergkerg/russian-scientific-articles:grnti_tfidf_hard_contrastive",
                "objective": "contrastive",
                "query": rows[anchor]["text"],
                "positive": rows[positive]["text"],
                "negatives": [rows[idx]["text"] for idx in negatives],
                "metadata": {
                    "grnti_label": rows[anchor]["label"],
                    "anchor_file": rows[anchor]["file"],
                    "positive_file": rows[positive]["file"],
                    "positive_tfidf_similarity": pos_sim,
                    "negative_tfidf_similarities": current_neg_sims,
                    "negative_labels": [rows[idx]["label"] for idx in negatives],
                    "negative_files": [rows[idx]["file"] for idx in negatives],
                    "contamination_policy": (
                        "Derived from the audited cleaned Kaggle GRNTI set; RuSciBench GRNTI/OECD "
                        "title/prefix overlaps were removed before this script. TF-IDF mining uses "
                        "only source text statistics, no released model and no benchmark rows."
                    ),
                },
            }
        )
    summary = {
        "records": len(records),
        "skipped": dict(skipped),
        "sampled_label_counts": dict(label_counts),
        "positive_tfidf_similarity": {
            "mean": mean(pos_sims) if pos_sims else None,
            "min": min(pos_sims) if pos_sims else None,
            "max": max(pos_sims) if pos_sims else None,
        },
        "negative_tfidf_similarity": {
            "mean": mean(neg_sims) if neg_sims else None,
            "min": min(neg_sims) if neg_sims else None,
            "max": max(neg_sims) if neg_sims else None,
        },
    }
    return records, summary


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
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_tfidf_hard_contrastive_b2_1171_seed2581.jsonl",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=1171)
    parser.add_argument("--negatives-per-record", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2581)
    args = parser.parse_args()

    rows = read_source(args.source, max_chars=args.max_chars)
    print(f"Loaded {len(rows)} rows across {len(set(row['label'] for row in rows))} labels")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform([row["tfidf_text"] for row in rows])
    sims = cosine_similarity(matrix).astype(np.float32)
    records, summary = build_records(
        rows,
        sims,
        max_records=args.max_records,
        negatives_per_record=args.negatives_per_record,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary.update(
        {
            "source": str(args.source),
            "output": str(args.output),
            "source_rows": len(rows),
            "source_label_counts": dict(Counter(row["label"] for row in rows)),
            "max_chars": args.max_chars,
            "seed": args.seed,
            "vectorizer": "word 1-2gram tfidf sublinear_tf",
        }
    )
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
