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


def compact(text: str, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].strip()


def cyrillic_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if not letters:
        return 0.0
    cyrillic = [char for char in letters if re.match(r"[А-Яа-яЁё]", char)]
    return len(cyrillic) / len(letters)


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


def extract_keywords(article: str) -> list[str]:
    keywords: list[str] = []
    for pattern in (
        r"Ключевые\s+слова\s*[:：]\s*(.+?)(?:\n|$)",
        r"Keywords\s*[:：]\s*(.+?)(?:\n|$)",
    ):
        match = re.search(pattern, article, flags=re.I)
        if not match:
            continue
        line = re.sub(r"\s+", " ", match.group(1))
        for part in re.split(r"[,;]", line):
            keyword = part.strip(" .:-–—«»\"'()[]").lower()
            if 3 <= len(keyword) <= 60 and cyrillic_ratio(keyword) >= 0.5:
                keywords.append(keyword)
        break
    return keywords[:8]


def load_rows(*, needles: dict[str, set[str]], max_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset("reginafeles/humarticles", split="train")
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    seen = set()
    for raw in dataset:
        title = compact(str(raw.get("title") or ""), max_chars=320)
        abstract = compact(str(raw.get("abstract") or ""), max_chars=1200)
        article = compact(str(raw.get("article") or ""), max_chars=1800)
        summary = compact(str(raw.get("sum") or ""), max_chars=800)
        if not title or len(title) < 20:
            skipped["bad_title"] += 1
            continue
        positive = abstract if len(abstract) >= 80 else article
        if len(positive) < 120:
            skipped["short_positive"] += 1
            continue
        text = compact(f"{title}. {abstract or article}", max_chars=1800)
        if cyrillic_ratio(text) < 0.65:
            skipped["low_cyrillic"] += 1
            continue
        if contaminated(text, needles) or contaminated(title, needles):
            skipped["benchmark_overlap"] += 1
            continue
        key = normalize(text)[:260]
        if key in seen:
            skipped["duplicate_prefix"] += 1
            continue
        seen.add(key)
        rows.append(
            {
                "id": raw.get("id"),
                "title": title,
                "positive": positive,
                "article": article,
                "summary": summary,
                "text": text,
                "keywords": extract_keywords(str(raw.get("article") or "")),
                "year": raw.get("year"),
            }
        )
        if max_rows and len(rows) >= max_rows:
            break
    return rows, {"source_rows": len(rows), "source_skipped": dict(skipped)}


def make_contrastive_records(rows: list[dict[str, Any]], *, count: int, negatives: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    indices = list(range(len(rows)))
    for index in range(min(count, len(rows))):
        row = rows[index]
        candidate_indices = [other for other in indices if other != index]
        negative_texts = [
            rows[other]["positive"] for other in rng.sample(candidate_indices, min(negatives, len(candidate_indices)))
        ]
        positives = [row["positive"]]
        if len(row["article"]) >= 300 and row["article"] != row["positive"]:
            positives.append(row["article"])
        if len(row["summary"]) >= 80:
            positives.append(row["summary"])
        records.append(
            {
                "source": "reginafeles/humarticles:title_abstract",
                "objective": "contrastive",
                "query": row["title"],
                "positive": row["positive"],
                "positives": positives[:3],
                "negatives": negative_texts,
                "metadata": {
                    "humarticles_id": row["id"],
                    "year": row["year"],
                    "contamination_policy": (
                        "Exact normalized text, front prefix, and title overlaps with "
                        "RuSciBench GRNTI/OECD classification and clustering datasets are removed. "
                        "No benchmark rows or released model are used."
                    ),
                },
            }
        )
    return records


def make_keyword_metric_records(
    rows: list[dict[str, Any]],
    *,
    batch_count: int,
    labels_per_batch: int,
    positives_per_label: int,
    min_docs_per_label: int,
    loss: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_keyword: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for keyword in row["keywords"]:
            by_keyword[keyword].append(row)
    labels = [label for label, values in by_keyword.items() if len(values) >= min_docs_per_label]
    labels = sorted(labels, key=lambda label: len(by_keyword[label]), reverse=True)
    if len(labels) < labels_per_batch:
        return [], {"usable_keyword_labels": len(labels), "keyword_counts": dict(Counter())}
    for label in labels:
        rng.shuffle(by_keyword[label])
    records = []
    cursors = Counter()
    label_pool = labels[: max(labels_per_batch, min(len(labels), labels_per_batch * 8))]
    sampled = Counter()
    for batch_index in range(batch_count):
        if batch_index % 5 == 0:
            selected = rng.sample(label_pool, labels_per_batch)
        else:
            offset = (batch_index * labels_per_batch) % len(label_pool)
            selected = [label_pool[(offset + i) % len(label_pool)] for i in range(labels_per_batch)]
        for label in selected:
            values = by_keyword[label]
            for _ in range(positives_per_label):
                cursor = cursors[label] % len(values)
                row = values[cursor]
                cursors[label] += 1
                if cursors[label] % len(values) == 0:
                    rng.shuffle(values)
                records.append(
                    {
                        "source": "reginafeles/humarticles:keyword_metric",
                        "objective": "labeled_text",
                        "text": row["text"],
                        "label": f"humarticles_keyword::{label}",
                        "loss": loss,
                        "metadata": {
                            "humarticles_id": row["id"],
                            "keyword": label,
                            "batch_index": batch_index,
                        },
                    }
                )
                sampled[label] += 1
    return records, {
        "usable_keyword_labels": len(labels),
        "keyword_label_pool": label_pool,
        "source_keyword_counts": {label: len(by_keyword[label]) for label in labels[:80]},
        "sampled_keyword_counts": dict(sampled),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean humarticles academic batches.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/contrastive/open_ru_1r_nc_humarticles_academic_hybrid_seed2761.jsonl")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--contrastive-count", type=int, default=900)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--keyword-batch-count", type=int, default=80)
    parser.add_argument("--labels-per-batch", type=int, default=3)
    parser.add_argument("--positives-per-label", type=int, default=2)
    parser.add_argument("--min-docs-per-label", type=int, default=3)
    parser.add_argument("--loss", choices=["supcon", "circle", "multi_similarity"], default="circle")
    parser.add_argument("--seed", type=int, default=2761)
    parser.add_argument("--shuffle-records", action="store_true")
    args = parser.parse_args()

    needles = benchmark_needles()
    rows, source_summary = load_rows(needles=needles, max_rows=args.max_rows)
    contrastive = make_contrastive_records(
        rows,
        count=args.contrastive_count,
        negatives=args.negatives,
        seed=args.seed,
    )
    keyword_records, keyword_summary = make_keyword_metric_records(
        rows,
        batch_count=args.keyword_batch_count,
        labels_per_batch=args.labels_per_batch,
        positives_per_label=args.positives_per_label,
        min_docs_per_label=args.min_docs_per_label,
        loss=args.loss,
        seed=args.seed + 1,
    )
    records = contrastive + keyword_records
    if args.shuffle_records:
        rng = random.Random(args.seed + 2)
        rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "output": str(args.output),
        "dataset": "reginafeles/humarticles",
        "benchmark_needles": {key: len(value) for key, value in needles.items()},
        "record_count": len(records),
        "contrastive_records": len(contrastive),
        "keyword_metric_records": len(keyword_records),
        "seed": args.seed,
        "shuffle_records": args.shuffle_records,
        **source_summary,
        **keyword_summary,
        "contamination_policy": (
            "Exact normalized text, front prefix, and title overlaps with RuSciBench "
            "GRNTI/OECD classification and clustering datasets are removed."
        ),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
