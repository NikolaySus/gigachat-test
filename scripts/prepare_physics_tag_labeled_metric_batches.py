#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

NOISY_TAGS = {
    "T",
    "W",
    "E",
    "X",
    "Y",
    "★",
    "★★",
    "★★★",
    "200 задач",
    "Китайские",
    "Всероссийские",
    "Азиатские",
    "Международные",
    "IEPhO",
}


def strip_query_prefix(text: str) -> str:
    marker = "\nQuery:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    if text.startswith("Query:"):
        return text[len("Query:") :].strip()
    return text.strip()


def is_topic_tag(tag: str) -> bool:
    tag = tag.strip()
    if not tag or tag in NOISY_TAGS:
        return False
    if re.fullmatch(r"\d{4}", tag):
        return False
    if any(word in tag for word in ("Сбор", "Жаутыков", "Олимпиад", "Турнир")):
        return False
    return any("А" <= ch <= "я" or ch == "ё" or ch == "Ё" for ch in tag)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_texts(row: dict[str, Any]) -> list[str]:
    texts = [
        strip_query_prefix(str(row.get("query") or "")),
        str(row.get("positive") or "").strip(),
    ]
    result = []
    seen = set()
    for text in texts:
        if len(text) < 120 or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def build_clusters(
    rows: list[dict[str, Any]],
    *,
    min_pair_count: int,
    min_single_count: int,
) -> dict[str, list[str]]:
    tag_lists: list[list[str]] = []
    pair_counts: Counter[str] = Counter()
    single_counts: Counter[str] = Counter()
    for row in rows:
        tags = [
            str(tag).strip()
            for tag in (row.get("metadata") or {}).get("tags", [])
            if is_topic_tag(str(tag))
        ]
        tag_lists.append(tags)
        single_counts.update(tags)
        for first, second in zip(tags, tags[1:]):
            pair_counts[f"{first} :: {second}"] += 1

    valid_pairs = {label for label, count in pair_counts.items() if count >= min_pair_count}
    valid_singles = {label for label, count in single_counts.items() if count >= min_single_count}
    clusters: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row, tags in zip(rows, tag_lists, strict=True):
        labels = [f"{first} :: {second}" for first, second in zip(tags, tags[1:])]
        labels = [label for label in labels if label in valid_pairs]
        if not labels:
            labels = [label for label in tags if label in valid_singles]
        for label in labels[:2]:
            for text in row_texts(row):
                if text in seen[label]:
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
    loss: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    labels = list(clusters)
    if len(labels) < labels_per_batch:
        raise ValueError(f"Need {labels_per_batch} labels, got {len(labels)}")
    records: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        selected = rng.sample(labels, labels_per_batch)
        for label in selected:
            texts = clusters[label]
            chosen = rng.sample(texts, positives_per_label)
            for text_index, text in enumerate(chosen):
                records.append(
                    {
                        "source": "Vikhrmodels/physics_big:tag_labeled_metric",
                        "objective": "labeled_text",
                        "text": text,
                        "label": f"physics::{label}",
                        "loss": loss,
                        "metadata": {
                            "topic_label": label,
                            "batch_index": batch_index,
                            "text_index": text_index,
                            "contamination_policy": "Derived from clean Vikhrmodels/physics_big tags; no ruMTEB/RuSciBench rows.",
                        },
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_physics_big_problem_solution_3200.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_physics_tag_labeled_circle_b8_1920_seed2541.jsonl",
    )
    parser.add_argument("--batch-count", type=int, default=240)
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--positives-per-label", type=int, default=2)
    parser.add_argument("--min-pair-count", type=int, default=20)
    parser.add_argument("--min-single-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2541)
    parser.add_argument("--loss", choices=["supcon", "circle", "multi_similarity"], default="circle")
    args = parser.parse_args()

    rows = read_rows(args.input)
    clusters = build_clusters(
        rows,
        min_pair_count=args.min_pair_count,
        min_single_count=args.min_single_count,
    )
    records = make_batches(
        clusters,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        positives_per_label=args.positives_per_label,
        seed=args.seed,
        loss=args.loss,
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
        "top_labels": sorted(
            ((label, len(texts)) for label, texts in clusters.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:30],
        "batch_count": args.batch_count,
        "batch_size": args.labels_per_batch * args.positives_per_label,
        "labels_per_batch": args.labels_per_batch,
        "positives_per_label": args.positives_per_label,
        "min_pair_count": args.min_pair_count,
        "min_single_count": args.min_single_count,
        "loss": args.loss,
        "seed": args.seed,
        "contamination_policy": "Clean Vikhrmodels/physics_big tags; benchmark RuSciBench/OECD/GRNTI datasets are not used.",
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
