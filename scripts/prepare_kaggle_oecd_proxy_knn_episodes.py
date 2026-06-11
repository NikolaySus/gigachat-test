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

# Coarse OECD/FOS-style proxy groups built only from GRNTI category names.
# This mapping is a clean training proxy; no RuSciBench rows or labels are used.
GRNTI_TO_OECD_PROXY = {
    1: "engineering_technology",
    2: "natural_sciences",
    3: "natural_sciences",
    4: "natural_sciences",
    5: "social_sciences",
    6: "social_sciences",
    7: "agricultural_environmental",
    8: "social_sciences",
    9: "natural_sciences",
    10: "natural_sciences",
    11: "natural_sciences",
    12: "natural_sciences",
    13: "engineering_technology",
    14: "social_sciences",
    15: "social_sciences",
    16: "engineering_technology",
    17: "engineering_technology",
    18: "humanities",
    19: "humanities",
    20: "engineering_technology",
    21: "social_sciences",
    22: "engineering_technology",
    23: "humanities",
    24: "engineering_technology",
    25: "agricultural_environmental",
    26: "humanities",
    27: "social_sciences",
    28: "natural_sciences",
    29: "engineering_technology",
    30: "medical_health",
    31: "engineering_technology",
    32: "engineering_technology",
    33: "engineering_technology",
    34: "social_sciences",
    35: "social_sciences",
    36: "engineering_technology",
    37: "social_sciences",
    38: "agricultural_environmental",
    39: "social_sciences",
    40: "engineering_technology",
    41: "social_sciences",
    42: "engineering_technology",
    43: "medical_health",
    44: "humanities",
    45: "agricultural_environmental",
    46: "engineering_technology",
    47: "agricultural_environmental",
    48: "social_sciences",
    49: "social_sciences",
    50: "engineering_technology",
    51: "engineering_technology",
    52: "natural_sciences",
    53: "medical_health",
    54: "humanities",
    55: "engineering_technology",
    56: "natural_sciences",
    57: "social_sciences",
    58: "engineering_technology",
    59: "engineering_technology",
    60: "engineering_technology",
    61: "humanities",
    62: "engineering_technology",
}


def grnti_number(label: str) -> int | None:
    match = re.match(r"\s*(\d+)\b", label)
    return int(match.group(1)) if match else None


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
            grnti_label = str(record["metadata"]["grnti_label"])
            number = grnti_number(grnti_label)
            proxy = GRNTI_TO_OECD_PROXY.get(number or -1)
            if proxy is None:
                continue
            text = str(record["text"]).strip()[:max_chars]
            file_name = str(record["metadata"]["file"])
            key = f"{grnti_label}\0{file_name}\0{text[:200]}"
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
                    "oecd_proxy": proxy,
                    "file": file_name,
                    "tfidf_text": tfidf_text,
                }
            )
    counts = Counter(row["oecd_proxy"] for row in rows)
    return [row for row in rows if counts[row["oecd_proxy"]] >= 6]


def balanced_anchor_order(rows: list[dict[str, Any]], *, max_records: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_label[row["oecd_proxy"]].append(idx)
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
    proxy_label: str,
    count: int,
    prefer_different_grnti: bool,
) -> list[int]:
    anchor_grnti = rows[anchor]["grnti_label"]
    candidates = [
        idx
        for idx, row in enumerate(rows)
        if row["oecd_proxy"] == proxy_label
        and idx != anchor
        and (not prefer_different_grnti or row["grnti_label"] != anchor_grnti)
    ]
    if len(candidates) < count and prefer_different_grnti:
        candidates = [
            idx
            for idx, row in enumerate(rows)
            if row["oecd_proxy"] == proxy_label and idx != anchor
        ]
    ranked = sorted(candidates, key=lambda idx: float(sims[anchor, idx]), reverse=True)
    return ranked[:count]


def hard_negative_proxy_labels(
    sims: np.ndarray,
    rows: list[dict[str, Any]],
    anchor: int,
    *,
    count: int,
) -> list[str]:
    current = rows[anchor]["oecd_proxy"]
    best_by_label: dict[str, float] = {}
    for idx, row in enumerate(rows):
        label = row["oecd_proxy"]
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
    for anchor in balanced_anchor_order(rows, max_records=max_records * 2, seed=seed):
        if len(records) >= max_records:
            break
        proxy = rows[anchor]["oecd_proxy"]
        pos = nearest_indices(
            sims,
            rows,
            anchor,
            proxy_label=proxy,
            count=positive_supports,
            prefer_different_grnti=True,
        )
        if len(pos) < positive_supports:
            skipped["not_enough_positive_supports"] += 1
            continue
        supports = {f"oecd_proxy::{proxy}": [rows[idx]["text"] for idx in pos]}
        neg_meta = {}
        valid = True
        for neg_label in hard_negative_proxy_labels(sims, rows, anchor, count=negative_labels):
            indices = nearest_indices(
                sims,
                rows,
                anchor,
                proxy_label=neg_label,
                count=supports_per_negative,
                prefer_different_grnti=False,
            )
            if len(indices) < supports_per_negative:
                valid = False
                break
            supports[f"oecd_proxy::{neg_label}"] = [rows[idx]["text"] for idx in indices]
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
        counts[proxy] += 1
        records.append(
            {
                "source": "kaggle/ergkerg/russian-scientific-articles:oecd_proxy_tfidf_knn_episode",
                "objective": "knn_classification",
                "query": rows[anchor]["text"],
                "label": f"oecd_proxy::{proxy}",
                "supports": supports,
                "metadata": {
                    "oecd_proxy": proxy,
                    "grnti_label": rows[anchor]["grnti_label"],
                    "anchor_file": rows[anchor]["file"],
                    "positive_support_files": [rows[idx]["file"] for idx in pos],
                    "positive_support_grnti_labels": [rows[idx]["grnti_label"] for idx in pos],
                    "positive_tfidf_similarities": [float(sims[anchor, idx]) for idx in pos],
                    "negative_supports": neg_meta,
                    "contamination_policy": (
                        "Derived from the audited cleaned Kaggle GRNTI set; RuSciBench GRNTI/OECD "
                        "title/prefix overlaps were removed before this script. OECD proxy labels are "
                        "hand-mapped from GRNTI category names; no benchmark rows or released model used."
                    ),
                },
            }
        )
    return records, {
        "records": len(records),
        "skipped": dict(skipped),
        "sampled_proxy_counts": dict(counts),
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
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_oecd_proxy_tfidf_knn_episode_p3_n3s2_900_seed2611.jsonl",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=900)
    parser.add_argument("--positive-supports", type=int, default=3)
    parser.add_argument("--negative-labels", type=int, default=3)
    parser.add_argument("--supports-per-negative", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2611)
    args = parser.parse_args()

    rows = read_source(args.source, max_chars=args.max_chars)
    print(f"Loaded {len(rows)} rows across {len(set(row['oecd_proxy'] for row in rows))} proxy labels")
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
            "source_proxy_counts": dict(Counter(row["oecd_proxy"] for row in rows)),
            "source_grnti_counts": dict(Counter(row["grnti_label"] for row in rows)),
            "positive_supports": args.positive_supports,
            "negative_labels": args.negative_labels,
            "supports_per_negative": args.supports_per_negative,
            "max_chars": args.max_chars,
            "seed": args.seed,
            "proxy_mapping": GRNTI_TO_OECD_PROXY,
            "vectorizer": "word 1-2gram tfidf sublinear_tf",
        }
    )
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
