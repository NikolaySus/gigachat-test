from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
CACHE_DIR = ROOT / "data" / "hf_cache"
PREFIX = "Определи тип места или услуги по отзыву \nотзыв: "


def normalize(text: str) -> str:
    text = str(text).lower().replace("\u00a0", " ")
    text = re.sub(r"^instruct:[^\n]*\nquery:\s*", "", text, flags=re.IGNORECASE)
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
        out = []
        for item in value:
            out.extend(iter_strings(item))
        return out
    return []


def collect_eval_texts() -> tuple[set[str], dict[str, int]]:
    datasets = {
        "mteb/georeview": ("train", "validation", "test"),
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


def polarity_bucket(value: Any) -> str:
    value = str(value).strip().lower()
    if not value:
        return "unknown"
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare clean review-style labeled_text records from blinoff/restaurants_reviews "
            "for output distillation / preservation."
        )
    )
    parser.add_argument("--out", type=Path, default=DATA_DIR / "fair_blinoff_restaurant_reviews_distill_12000_seed2090.jsonl")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2090)
    parser.add_argument("--min-chars", type=int, default=60)
    parser.add_argument("--max-chars", type=int, default=1200)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_texts, eval_counts = collect_eval_texts()

    raw_path = hf_hub_download(
        repo_id="blinoff/restaurants_reviews",
        repo_type="dataset",
        filename="restaurants_reviews.jsonl",
        cache_dir=str(CACHE_DIR),
    )

    rows = []
    source_rows = 0
    exact_eval_overlap = 0
    too_short = 0
    too_long = 0
    duplicate = 0
    missing_text = 0
    label_counts = Counter()
    seen: set[str] = set()
    with Path(raw_path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            source_rows += 1
            row = json.loads(line)
            text = clean_text(row.get("text") or "")
            if not text:
                missing_text += 1
                continue
            if len(text) < args.min_chars:
                too_short += 1
                continue
            if len(text) > args.max_chars:
                too_long += 1
                continue
            norm = normalize(text)
            if norm in eval_texts:
                exact_eval_overlap += 1
                continue
            if norm in seen:
                duplicate += 1
                continue
            seen.add(norm)
            label = (
                "restaurant_review::"
                + polarity_bucket(row.get("general"))
                + "::food_"
                + polarity_bucket(row.get("food"))
                + "::interior_"
                + polarity_bucket(row.get("interior"))
                + "::service_"
                + polarity_bucket(row.get("service"))
            )
            label_counts[label] += 1
            rows.append(
                {
                    "source": "blinoff/restaurants_reviews:exact_eval_removed",
                    "objective": "labeled_text",
                    "text": PREFIX + text,
                    "label": label,
                    "loss": "supcon",
                    "metadata": {
                        "construction": "review_style_output_distillation_preservation",
                        "review_id": row.get("review_id"),
                        "general": row.get("general"),
                        "food": row.get("food"),
                        "interior": row.get("interior"),
                        "service": row.get("service"),
                    },
                }
            )

    rng.shuffle(rows)
    rows = rows[: args.max_records]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for record in rows:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    summary = {
        "output": str(args.out),
        "source": "blinoff/restaurants_reviews",
        "records": len(rows),
        "seed": args.seed,
        "source_rows": source_rows,
        "missing_text": missing_text,
        "too_short": too_short,
        "too_long": too_long,
        "duplicate": duplicate,
        "exact_eval_overlap_removed": exact_eval_overlap,
        "eval_datasets_checked": eval_counts,
        "unique_eval_texts": len(eval_texts),
        "top_labels": label_counts.most_common(20),
        "fairness": (
            "Exact normalized text overlaps with mteb/georeview, "
            "mteb/CEDRClassification, and mteb/RuSTSBenchmarkSTS are removed. "
            "The source is a restaurant review corpus, not the Georeview source corpus."
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
