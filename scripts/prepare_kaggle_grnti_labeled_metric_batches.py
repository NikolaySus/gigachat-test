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


def normalize(text: str) -> str:
    text = text.replace("\ufeff", " ").lower()
    text = re.sub(r"[^0-9a-zа-яё]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def benchmark_needles() -> dict[str, dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"titles": set(), "prefixes": set()})
    for dataset_name in (
        "ai-forever/ru-scibench-grnti-classification",
        "ai-forever/ru-scibench-oecd-classification",
    ):
        dataset = load_dataset(dataset_name)
        for split, rows in dataset.items():
            key = f"{dataset_name}:{split}"
            for row in rows:
                text = str(row["text"])
                norm = normalize(text)
                title = normalize(text.split(".", 1)[0])
                if len(title) >= 30:
                    result[key]["titles"].add(title)
                if len(norm) >= 180:
                    result[key]["prefixes"].add(norm[:180])
    return result


def article_overlap_candidates(text: str) -> set[str]:
    lines = [line.strip() for line in text.replace("\ufeff", " ").splitlines()]
    lines = [line for line in lines if line]
    candidates: set[str] = set()
    for index, line in enumerate(lines[:80]):
        for width in (1, 2, 3, 4):
            block = " ".join(lines[index : index + width])
            norm = normalize(block)
            if 30 <= len(norm) <= 260:
                candidates.add(norm)
                words = norm.split()
                if len(words) >= 8:
                    candidates.add(" ".join(words[:12]))
    # Also compare compact front prefixes after common abstract markers.
    lowered = text.lower()
    for marker in ("аннотация", "abstract", "резюме"):
        pos = lowered.find(marker)
        if 0 <= pos < 4000:
            norm = normalize(text[pos : pos + 800])
            if len(norm) >= 180:
                candidates.add(norm[:180])
    return candidates


def clean_text(text: str, *, max_chars: int) -> str | None:
    text = text.replace("\ufeff", " ")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 800:
        return None
    # Preserve title/abstract/front matter but cap pathological full articles.
    return text[:max_chars]


def read_articles(root: Path, *, max_chars: int, needles: dict[str, dict[str, set[str]]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    contaminated_examples: list[dict[str, str]] = []
    title_index: dict[str, str] = {}
    prefix_index: dict[str, str] = {}
    for source, values in needles.items():
        for title in values["titles"]:
            title_index[title] = source
        for prefix in values["prefixes"]:
            prefix_index[prefix] = source
    for file_path in sorted(root.glob("data_3*/*/*.txt")):
        label = file_path.parent.name.strip()
        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            skipped["read_error"] += 1
            continue
        text = clean_text(raw, max_chars=max_chars)
        if not text:
            skipped["too_short"] += 1
            continue
        candidates = article_overlap_candidates(raw[:8000])
        hit_source = None
        hit_needle = None
        for candidate in candidates:
            if candidate in title_index:
                hit_source = title_index[candidate]
                hit_needle = candidate
                break
            if candidate in prefix_index:
                hit_source = prefix_index[candidate]
                hit_needle = candidate
                break
        if hit_source:
            skipped["benchmark_overlap"] += 1
            if len(contaminated_examples) < 30:
                contaminated_examples.append(
                    {
                        "file": str(file_path.relative_to(root)),
                        "label": label,
                        "benchmark_source": hit_source,
                        "needle": hit_needle[:160],
                    }
                )
            continue
        by_label[label].append({"text": text, "file": str(file_path.relative_to(root))})
    summary = {
        "labels_before_filter": len(by_label),
        "skipped": dict(skipped),
        "contaminated_examples": contaminated_examples,
    }
    return by_label, summary


def make_batches(
    by_label: dict[str, list[dict[str, Any]]],
    *,
    batch_count: int,
    labels_per_batch: int,
    positives_per_label: int,
    min_docs_per_label: int,
    seed: int,
    loss: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    labels = [label for label, rows in by_label.items() if len(rows) >= min_docs_per_label]
    if len(labels) < labels_per_batch:
        raise ValueError(f"Need at least {labels_per_batch} labels, got {len(labels)}")
    records: list[dict[str, Any]] = []
    label_counts = Counter()
    for batch_index in range(batch_count):
        selected = rng.sample(labels, labels_per_batch)
        for label in selected:
            chosen = rng.sample(by_label[label], positives_per_label)
            for text_index, item in enumerate(chosen):
                records.append(
                    {
                        "source": "kaggle/ergkerg/russian-scientific-articles:grnti_labeled_metric",
                        "objective": "labeled_text",
                        "text": item["text"],
                        "label": f"kaggle_grnti::{label}",
                        "loss": loss,
                        "metadata": {
                            "grnti_label": label,
                            "file": item["file"],
                            "batch_index": batch_index,
                            "text_index": text_index,
                            "contamination_policy": "Exact title/prefix overlap with RuSciBench GRNTI/OECD train/test removed before sampling.",
                        },
                    }
                )
                label_counts[label] += 1
    summary = {
        "usable_labels": len(labels),
        "label_doc_counts": {label: len(by_label[label]) for label in labels},
        "sampled_label_counts": dict(label_counts),
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "data/kagglehub_cache/datasets/ergkerg/russian-scientific-articles/versions/1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_labeled_circle_b4_1600_seed2551.jsonl",
    )
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--batch-count", type=int, default=400)
    parser.add_argument("--labels-per-batch", type=int, default=2)
    parser.add_argument("--positives-per-label", type=int, default=2)
    parser.add_argument("--min-docs-per-label", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2551)
    parser.add_argument("--loss", choices=["supcon", "circle", "multi_similarity"], default="circle")
    args = parser.parse_args()

    needles = benchmark_needles()
    by_label, audit_summary = read_articles(args.input_root, max_chars=args.max_chars, needles=needles)
    records, batch_summary = make_batches(
        by_label,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        positives_per_label=args.positives_per_label,
        min_docs_per_label=args.min_docs_per_label,
        seed=args.seed,
        loss=args.loss,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "input_root": str(args.input_root),
        "output": str(args.output),
        "records": len(records),
        "batch_count": args.batch_count,
        "batch_size": args.labels_per_batch * args.positives_per_label,
        "labels_per_batch": args.labels_per_batch,
        "positives_per_label": args.positives_per_label,
        "max_chars": args.max_chars,
        "loss": args.loss,
        "seed": args.seed,
        "benchmark_needles": {
            key: {name: len(items) for name, items in values.items()}
            for key, values in needles.items()
        },
        **audit_summary,
        **batch_summary,
        "contamination_policy": "RuSciBench GRNTI/OECD train and test title/prefix overlaps removed; MTEB/RuSciBench rows are not used for training.",
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
