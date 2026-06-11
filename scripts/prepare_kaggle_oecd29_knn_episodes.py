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

from prepare_kaggle_grnti_to_oecd29_metric_batches import GRNTI_NUMBER_TO_OECD29


ROOT = Path(__file__).resolve().parents[1]


def grnti_number(label: str) -> int | None:
    match = re.match(r"\s*(\d+)\b", label)
    return int(match.group(1)) if match else None


def normalize_for_tfidf(text: str) -> str:
    text = text.lower().replace("\ufeff", " ")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^0-9a-zа-яё\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_source(path: Path, *, max_chars: int, min_docs_per_label: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            grnti_label = str(record.get("metadata", {}).get("grnti_label", "")).strip()
            number = grnti_number(grnti_label)
            oecd_label = GRNTI_NUMBER_TO_OECD29.get(number or -1)
            if oecd_label is None:
                continue
            text = str(record.get("text", "")).strip()[:max_chars]
            file_name = str(record.get("metadata", {}).get("file", ""))
            key = f"{oecd_label}\0{grnti_label}\0{file_name}\0{text[:200]}"
            if key in seen:
                continue
            seen.add(key)
            tfidf_text = normalize_for_tfidf(text)
            if len(tfidf_text) < 200:
                continue
            rows.append(
                {
                    "text": text,
                    "grnti_label": grnti_label,
                    "oecd_label": oecd_label,
                    "file": file_name,
                    "tfidf_text": tfidf_text,
                }
            )
    counts = Counter(row["oecd_label"] for row in rows)
    return [row for row in rows if counts[row["oecd_label"]] >= min_docs_per_label]


def balanced_anchor_order(rows: list[dict[str, Any]], *, max_records: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_label[row["oecd_label"]].append(idx)
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
    prefer_different_grnti: bool,
) -> list[int]:
    anchor_grnti = rows[anchor]["grnti_label"]
    candidates = [
        idx
        for idx, row in enumerate(rows)
        if row["oecd_label"] == label
        and idx != anchor
        and (not prefer_different_grnti or row["grnti_label"] != anchor_grnti)
    ]
    if len(candidates) < count and prefer_different_grnti:
        candidates = [idx for idx, row in enumerate(rows) if row["oecd_label"] == label and idx != anchor]
    ranked = sorted(candidates, key=lambda idx: float(sims[anchor, idx]), reverse=True)
    return ranked[:count]


def hard_negative_labels(sims: np.ndarray, rows: list[dict[str, Any]], anchor: int, *, count: int) -> list[str]:
    current = rows[anchor]["oecd_label"]
    best_by_label: dict[str, float] = {}
    for idx, row in enumerate(rows):
        label = row["oecd_label"]
        if label == current:
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
    counts = Counter()
    for anchor in balanced_anchor_order(rows, max_records=max_records * 3, seed=seed):
        if len(records) >= max_records:
            break
        label = rows[anchor]["oecd_label"]
        pos = nearest_indices(
            sims,
            rows,
            anchor,
            label=label,
            count=positive_supports,
            prefer_different_grnti=True,
        )
        if len(pos) < positive_supports:
            skipped["not_enough_positive_supports"] += 1
            continue
        supports = {f"kaggle_oecd29::{label}": [rows[idx]["text"] for idx in pos]}
        neg_meta: dict[str, Any] = {}
        valid = True
        for neg_label in hard_negative_labels(sims, rows, anchor, count=negative_labels):
            indices = nearest_indices(
                sims,
                rows,
                anchor,
                label=neg_label,
                count=supports_per_negative,
                prefer_different_grnti=False,
            )
            if len(indices) < supports_per_negative:
                valid = False
                break
            supports[f"kaggle_oecd29::{neg_label}"] = [rows[idx]["text"] for idx in indices]
            neg_meta[neg_label] = [
                {
                    "file": rows[idx]["file"],
                    "grnti_label": rows[idx]["grnti_label"],
                    "tfidf_similarity": float(sims[anchor, idx]),
                }
                for idx in indices
            ]
            neg_sims.extend(float(sims[anchor, idx]) for idx in indices)
        if not valid:
            skipped["not_enough_negative_supports"] += 1
            continue
        pos_sims.extend(float(sims[anchor, idx]) for idx in pos)
        counts[label] += 1
        records.append(
            {
                "source": "kaggle/ergkerg/russian-scientific-articles:oecd29_tfidf_knn_episode",
                "objective": "knn_classification",
                "query": rows[anchor]["text"],
                "label": f"kaggle_oecd29::{label}",
                "supports": supports,
                "metadata": {
                    "mapped_oecd_label": label,
                    "grnti_label": rows[anchor]["grnti_label"],
                    "anchor_file": rows[anchor]["file"],
                    "positive_support_files": [rows[idx]["file"] for idx in pos],
                    "positive_support_grnti_labels": [rows[idx]["grnti_label"] for idx in pos],
                    "positive_tfidf_similarities": [float(sims[anchor, idx]) for idx in pos],
                    "negative_supports": neg_meta,
                    "contamination_policy": (
                        "Derived from audited Kaggle GRNTI article JSONL after RuSciBench "
                        "GRNTI/OECD title/prefix overlap removal. GRNTI labels are mapped to "
                        "public RuSciBench OECD-style label names; no benchmark rows, released "
                        "model, or released latent weights are used."
                    ),
                },
            }
        )
    return records, {
        "records": len(records),
        "skipped": dict(skipped),
        "sampled_oecd_counts": dict(counts),
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
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_oecd29_tfidf_knn_episode_p2_n4s1_1200_seed2671.jsonl",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=1200)
    parser.add_argument("--positive-supports", type=int, default=2)
    parser.add_argument("--negative-labels", type=int, default=4)
    parser.add_argument("--supports-per-negative", type=int, default=1)
    parser.add_argument("--min-docs-per-label", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2671)
    args = parser.parse_args()

    rows = read_source(args.source, max_chars=args.max_chars, min_docs_per_label=args.min_docs_per_label)
    print(f"Loaded {len(rows)} rows across {len(set(row['oecd_label'] for row in rows))} OECD-style labels")
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
            "source_rows_after_filter": len(rows),
            "source_oecd_counts": dict(Counter(row["oecd_label"] for row in rows)),
            "mapping": GRNTI_NUMBER_TO_OECD29,
            "positive_supports": args.positive_supports,
            "negative_labels": args.negative_labels,
            "supports_per_negative": args.supports_per_negative,
            "seed": args.seed,
            "contamination_policy": (
                "Inherits RuSciBench GRNTI/OECD title/prefix overlap filtering from source JSONL."
            ),
        }
    )
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
