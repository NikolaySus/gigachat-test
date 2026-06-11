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
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    text = text.replace("\ufeff", " ").lower()
    text = re.sub(r"[^0-9a-zа-яё]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cyrillic_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if not letters:
        return 0.0
    cyr = [char for char in letters if re.match(r"[А-Яа-яЁё]", char)]
    return len(cyr) / len(letters)


def benchmark_needles() -> dict[str, set[str]]:
    needles = {"titles": set(), "prefixes": set(), "exact": set()}
    for dataset_name in (
        "ai-forever/ru-scibench-oecd-classification",
        "ai-forever/ru-scibench-oecd-clustering-p2p",
        "ai-forever/ru-scibench-grnti-classification",
        "ai-forever/ru-scibench-grnti-clustering-p2p",
    ):
        dataset = load_dataset(dataset_name)
        for rows in dataset.values():
            text_column = "text" if "text" in rows.column_names else "sentences"
            for row in rows:
                text = str(row[text_column])
                norm = normalize(text)
                if len(norm) >= 80:
                    needles["exact"].add(norm)
                    needles["prefixes"].add(norm[:220])
                title = normalize(text.split(".", 1)[0])
                if len(title) >= 30:
                    needles["titles"].add(title)
    return needles


def contaminated(text: str, needles: dict[str, set[str]]) -> bool:
    norm = normalize(text)
    if norm in needles["exact"]:
        return True
    if len(norm) >= 220 and norm[:220] in needles["prefixes"]:
        return True
    title = normalize(text.split(".", 1)[0])
    return len(title) >= 30 and title in needles["titles"]


def load_ainl_human_rows(*, min_chars: int, needles: dict[str, set[str]]) -> tuple[list[str], Counter[str]]:
    skipped: Counter[str] = Counter()
    texts: list[str] = []
    seen: set[str] = set()
    specs = [
        ("train.csv", {"human"}),
        ("dev_full.csv", {"abstract"}),
    ]
    for file_name, allowed_labels in specs:
        dataset = load_dataset("iis-research-team/AINL-Eval-2025", data_files=file_name, split="train")
        for row in dataset:
            label = str(row.get("label") or "")
            if label not in allowed_labels:
                skipped["non_human_or_non_abstract_label"] += 1
                continue
            text = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()
            if len(text) < min_chars:
                skipped["too_short"] += 1
                continue
            if cyrillic_ratio(text) < 0.55:
                skipped["low_cyrillic_ratio"] += 1
                continue
            if contaminated(text, needles):
                skipped["benchmark_overlap"] += 1
                continue
            key = normalize(text)[:300]
            if key in seen:
                skipped["duplicate_prefix"] += 1
                continue
            seen.add(key)
            texts.append(text)
    return texts, skipped


def make_records(
    texts: list[str],
    labels: list[int],
    *,
    clusters: int,
    batch_count: int,
    labels_per_batch: int,
    examples_per_label: int,
    min_docs_per_cluster: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = defaultdict(list)
    for text, label in zip(texts, labels, strict=True):
        by_label[int(label)].append(text)
    usable_labels = [label for label, values in by_label.items() if len(values) >= min_docs_per_cluster]
    usable_labels.sort(key=lambda label: len(by_label[label]), reverse=True)
    if len(usable_labels) < labels_per_batch:
        raise ValueError(f"Need at least {labels_per_batch} usable clusters, got {len(usable_labels)}")
    for label in usable_labels:
        rng.shuffle(by_label[label])
    rng.shuffle(usable_labels)
    cursors = Counter()
    records: list[dict[str, Any]] = []
    for batch_index in range(batch_count):
        if batch_index and batch_index % len(usable_labels) == 0:
            rng.shuffle(usable_labels)
        for label_offset in range(labels_per_batch):
            label = usable_labels[(batch_index * labels_per_batch + label_offset) % len(usable_labels)]
            values = by_label[label]
            start = cursors[label]
            for offset in range(examples_per_label):
                text = values[(start + offset) % len(values)]
                records.append(
                    {
                        "source": "iis-research-team/AINL-Eval-2025:human_abstract_tfidf_cluster",
                        "objective": "labeled_text",
                        "text": "Определи научную область по аннотации\nаннотация: " + text,
                        "label": f"ainl_tfidf_cluster::{label:03d}",
                        "loss": "supcon",
                        "metadata": {
                            "cluster": int(label),
                            "clusters": clusters,
                            "mixed_batch": batch_index,
                            "construction": "tfidf_minibatch_kmeans_human_science_abstracts",
                        },
                    }
                )
            cursors[label] = (start + examples_per_label) % len(values)
    summary = {
        "usable_clusters": len(usable_labels),
        "cluster_sizes": {str(label): len(by_label[label]) for label in usable_labels[:20]},
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=48)
    parser.add_argument("--batch-count", type=int, default=80)
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--examples-per-label", type=int, default=2)
    parser.add_argument("--min-docs-per-cluster", type=int, default=20)
    parser.add_argument("--min-chars", type=int, default=220)
    parser.add_argument("--seed", type=int, default=2801)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/fair_ainl_human_science_tfidf_clusters48_b8_640_seed2801.jsonl",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    needles = benchmark_needles()
    texts, skipped = load_ainl_human_rows(min_chars=args.min_chars, needles=needles)
    if len(texts) < args.clusters * args.min_docs_per_cluster:
        raise ValueError(f"Too few clean texts for {args.clusters} clusters: {len(texts)}")

    vectorizer = TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    kmeans = MiniBatchKMeans(
        n_clusters=args.clusters,
        random_state=args.seed,
        batch_size=2048,
        n_init=10,
        reassignment_ratio=0.01,
    )
    labels = kmeans.fit_predict(matrix)
    records, record_summary = make_records(
        texts,
        labels.tolist(),
        clusters=args.clusters,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        examples_per_label=args.examples_per_label,
        min_docs_per_cluster=args.min_docs_per_cluster,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output),
        "source": "iis-research-team/AINL-Eval-2025",
        "records": len(records),
        "clean_texts": len(texts),
        "clusters": args.clusters,
        "batch_size": args.labels_per_batch * args.examples_per_label,
        "batch_layout": f"{args.labels_per_batch} clusters x {args.examples_per_label} abstracts",
        "skipped": dict(skipped),
        "record_summary": record_summary,
        "fairness": (
            "Uses only AINL human/abstract rows. Exact normalized full-text, front-prefix, "
            "and title overlaps with RuSciBench GRNTI/OECD classification and clustering "
            "datasets are removed. No benchmark rows, released model outputs, released "
            "teacher, or released latent weights are used."
        ),
    }
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
