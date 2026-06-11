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
            text = str(record["text"]).strip()[:max_chars]
            label = str(record["metadata"]["grnti_label"])
            file_name = str(record["metadata"]["file"])
            key = f"{label}\0{file_name}\0{text[:200]}"
            if key in seen:
                continue
            seen.add(key)
            tfidf_text = normalize_for_tfidf(text)
            if len(tfidf_text) < 200:
                continue
            rows.append({"text": text, "label": label, "file": file_name, "tfidf_text": tfidf_text})
    counts = Counter(row["label"] for row in rows)
    return [row for row in rows if counts[row["label"]] >= 4]


def balanced_anchor_order(rows: list[dict[str, Any]], *, max_records: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_label[row["label"]].append(idx)
    for values in by_label.values():
        rng.shuffle(values)
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


def nearest_indices(
    sims: np.ndarray,
    rows: list[dict[str, Any]],
    anchor: int,
    *,
    label: str,
    count: int,
    exclude_anchor: bool = True,
) -> list[int]:
    candidates = [
        idx
        for idx, row in enumerate(rows)
        if row["label"] == label and (idx != anchor or not exclude_anchor)
    ]
    ranked = sorted(candidates, key=lambda idx: float(sims[anchor, idx]), reverse=True)
    return ranked[:count]


def hard_negative_labels(
    sims: np.ndarray,
    rows: list[dict[str, Any]],
    anchor: int,
    *,
    count: int,
) -> list[str]:
    anchor_label = rows[anchor]["label"]
    best_by_label: dict[str, float] = {}
    for idx, row in enumerate(rows):
        label = row["label"]
        if label == anchor_label:
            continue
        score = float(sims[anchor, idx])
        if score > best_by_label.get(label, -1.0):
            best_by_label[label] = score
    return [label for label, _score in sorted(best_by_label.items(), key=lambda item: item[1], reverse=True)[:count]]


def build_records(
    rows: list[dict[str, Any]],
    sims: np.ndarray,
    *,
    max_records: int,
    positive_supports: int,
    negative_labels: int,
    supports_per_negative: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = Counter()
    pos_sims: list[float] = []
    neg_sims: list[float] = []
    sampled_labels = Counter()
    for anchor in balanced_anchor_order(rows, max_records=max_records * 2, seed=seed):
        if len(records) >= max_records:
            break
        label = rows[anchor]["label"]
        positive_indices = nearest_indices(
            sims,
            rows,
            anchor,
            label=label,
            count=positive_supports,
        )
        if len(positive_indices) < positive_supports:
            skipped["not_enough_positive_supports"] += 1
            continue
        neg_labels = hard_negative_labels(sims, rows, anchor, count=negative_labels)
        supports: dict[str, list[str]] = {
            f"kaggle_grnti::{label}": [rows[idx]["text"] for idx in positive_indices]
        }
        neg_meta: dict[str, list[dict[str, Any]]] = {}
        valid = True
        for neg_label in neg_labels:
            indices = nearest_indices(
                sims,
                rows,
                anchor,
                label=neg_label,
                count=supports_per_negative,
                exclude_anchor=False,
            )
            if len(indices) < supports_per_negative:
                valid = False
                break
            supports[f"kaggle_grnti::{neg_label}"] = [rows[idx]["text"] for idx in indices]
            neg_meta[neg_label] = [
                {"file": rows[idx]["file"], "tfidf_similarity": float(sims[anchor, idx])}
                for idx in indices
            ]
            neg_sims.extend(float(sims[anchor, idx]) for idx in indices)
        if not valid:
            skipped["not_enough_negative_supports"] += 1
            continue
        pos_sims.extend(float(sims[anchor, idx]) for idx in positive_indices)
        sampled_labels[label] += 1
        records.append(
            {
                "source": "kaggle/ergkerg/russian-scientific-articles:grnti_tfidf_knn_episode",
                "objective": "knn_classification",
                "query": rows[anchor]["text"],
                "label": f"kaggle_grnti::{label}",
                "supports": supports,
                "metadata": {
                    "grnti_label": label,
                    "anchor_file": rows[anchor]["file"],
                    "positive_support_files": [rows[idx]["file"] for idx in positive_indices],
                    "positive_tfidf_similarities": [float(sims[anchor, idx]) for idx in positive_indices],
                    "negative_supports": neg_meta,
                    "contamination_policy": (
                        "Derived from the audited cleaned Kaggle GRNTI set; RuSciBench GRNTI/OECD "
                        "title/prefix overlaps were removed before this script. TF-IDF episode mining "
                        "uses only source text statistics, no released model and no benchmark rows."
                    ),
                },
            }
        )
    summary = {
        "records": len(records),
        "skipped": dict(skipped),
        "sampled_label_counts": dict(sampled_labels),
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
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_tfidf_knn_episode_p2_n3s2_900_seed2591.jsonl",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=900)
    parser.add_argument("--positive-supports", type=int, default=2)
    parser.add_argument("--negative-labels", type=int, default=3)
    parser.add_argument("--supports-per-negative", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2591)
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
        positive_supports=args.positive_supports,
        negative_labels=args.negative_labels,
        supports_per_negative=args.supports_per_negative,
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
            "positive_supports": args.positive_supports,
            "negative_labels": args.negative_labels,
            "supports_per_negative": args.supports_per_negative,
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
