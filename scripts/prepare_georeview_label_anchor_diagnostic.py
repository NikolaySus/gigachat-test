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
CACHE_DIR = ROOT / "data" / "hf_cache"
DATA_DIR = ROOT / "data" / "contrastive"

QUERY_PREFIX = "Определи категорию организации на основе отзыва\nОтзыв: "
LABEL_PREFIX = "Категория организации: "


def normalize(text: str) -> str:
    text = str(text).lower().replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(iter_strings(item))
        return out
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(iter_strings(item))
        return out
    return []


def collect_eval_texts() -> tuple[set[str], dict[str, int]]:
    datasets = {
        "mteb/georeview": ("train", "validation", "test"),
        "mteb/GeoreviewClusteringP2P": ("test",),
        "mteb/CEDRClassification": ("train", "validation", "test"),
        "mteb/RuSTSBenchmarkSTS": ("train", "validation", "test"),
    }
    texts: set[str] = set()
    counts: dict[str, int] = {}
    for dataset_name, splits in datasets.items():
        rows = 0
        for split in splits:
            try:
                ds = load_dataset(dataset_name, split=split, cache_dir=str(CACHE_DIR))
            except Exception:
                continue
            for row in ds:
                rows += 1
                for text in iter_strings(row):
                    norm = normalize(text)
                    if norm:
                        texts.add(norm)
        counts[dataset_name] = rows
    return texts, counts


def primary_rubric(row: dict[str, Any]) -> str:
    rubrics = str(row.get("rubrics") or "").strip()
    if not rubrics:
        return ""
    return rubrics.split(";")[0].strip()


def label_text(label: str) -> str:
    return (
        f"{LABEL_PREFIX}{label}. "
        f"Отзывы этой категории описывают место, услугу, персонал, ассортимент, "
        f"цены, интерьер или качество обслуживания именно для типа организации: {label}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostic source-risk review-to-category-name contrastive dataset "
            "from d0rj geo reviews after exact MTEB row removal."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "DIAGNOSTIC_georeview_label_anchor_exact_mteb_removed_3200_seed2101.jsonl",
    )
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=3200)
    parser.add_argument("--examples-per-label", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--negatives", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_texts, eval_counts = collect_eval_texts()
    bench = load_dataset("mteb/GeoreviewClusteringP2P", split="test", cache_dir=str(CACHE_DIR))
    target_labels = sorted({str(row["labels"]).strip() for row in bench if str(row["labels"]).strip()})
    target_label_set = set(target_labels)

    pools: dict[str, list[str]] = defaultdict(list)
    source_rows = 0
    exact_eval_overlap = 0
    bad_length = 0
    wrong_label = 0
    duplicate = 0
    seen: set[str] = set()

    ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train", cache_dir=str(CACHE_DIR))
    for row in ds:
        source_rows += 1
        label = primary_rubric(row)
        if label not in target_label_set:
            wrong_label += 1
            continue
        text = clean_text(row.get("text") or "")
        if not (args.min_chars <= len(text) <= args.max_chars):
            bad_length += 1
            continue
        norm = normalize(text)
        if norm in eval_texts:
            exact_eval_overlap += 1
            continue
        if norm in seen:
            duplicate += 1
            continue
        seen.add(norm)
        pools[label].append(text)

    for texts in pools.values():
        rng.shuffle(texts)

    records: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    label_order = target_labels[:]
    rng.shuffle(label_order)
    per_label = min(args.examples_per_label, max(1, args.max_records // max(1, len(label_order))))
    for label in label_order:
        texts = pools.get(label, [])[:per_label]
        for text in texts:
            negative_labels = [candidate for candidate in target_labels if candidate != label]
            negatives = rng.sample(negative_labels, k=min(args.negatives, len(negative_labels)))
            records.append(
                {
                    "source": "DIAGNOSTIC_d0rj_georeview_label_anchor_exact_mteb_removed",
                    "objective": "contrastive",
                    "query": QUERY_PREFIX + text,
                    "positive": label_text(label),
                    "positives": [f"{LABEL_PREFIX}{label}"],
                    "negatives": [label_text(negative) for negative in negatives],
                    "metadata": {
                        "construction": "source_overlap_risky_review_to_category_anchor",
                        "label": label,
                    },
                }
            )
            label_counts[label] += 1

    rng.shuffle(records)
    records = records[: args.max_records]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    summary = {
        "output": str(args.out),
        "records": len(records),
        "source": "d0rj/geo-reviews-dataset-2023",
        "fairness_status": (
            "DIAGNOSTIC ONLY. Exact normalized MTEB rows are removed, but the source "
            "is the same Yandex Georeview corpus family as GeoreviewClusteringP2P."
        ),
        "source_rows": source_rows,
        "wrong_label_rows": wrong_label,
        "bad_length_rows": bad_length,
        "duplicate_rows": duplicate,
        "exact_eval_overlap_removed": exact_eval_overlap,
        "eval_datasets_checked": eval_counts,
        "target_labels": len(target_labels),
        "labels_with_records": len(label_counts),
        "min_label_count": min(label_counts.values()) if label_counts else 0,
        "max_label_count": max(label_counts.values()) if label_counts else 0,
        "top_labels": label_counts.most_common(20),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
