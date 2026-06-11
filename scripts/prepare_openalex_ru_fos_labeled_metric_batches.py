#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OPENALEX_API = "https://api.openalex.org/works"


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


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: dict[int, str] = {}
    for token, token_positions in index.items():
        for pos in token_positions:
            positions[int(pos)] = token
    return " ".join(positions[pos] for pos in sorted(positions))


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


def openalex_rows(
    *,
    max_candidates: int,
    per_page: int,
    mailto: str | None,
    sleep_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = "*"
    scanned = 0
    skipped = Counter()
    session = requests.Session()
    while scanned < max_candidates:
        params = {
            "filter": "language:ru,has_abstract:true",
            "per-page": per_page,
            "cursor": cursor,
            "select": "id,title,abstract_inverted_index,primary_topic,language,publication_year",
        }
        if mailto:
            params["mailto"] = mailto
        response = session.get(OPENALEX_API, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("results", []):
            scanned += 1
            title = str(item.get("title") or "").strip()
            abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
            text = re.sub(r"\s+", " ", f"{title}. {abstract}").strip()
            if len(text) < 240:
                skipped["too_short"] += 1
                continue
            if cyrillic_ratio(text) < 0.55:
                skipped["low_cyrillic_ratio"] += 1
                continue
            topic = item.get("primary_topic") or {}
            subfield = topic.get("subfield") or {}
            field = topic.get("field") or {}
            domain = topic.get("domain") or {}
            label = str(subfield.get("display_name") or field.get("display_name") or "").strip()
            if not label:
                skipped["missing_topic_label"] += 1
                continue
            rows.append(
                {
                    "id": item.get("id"),
                    "title": title,
                    "text": text,
                    "label": label,
                    "field": field.get("display_name"),
                    "domain": domain.get("display_name"),
                    "publication_year": item.get("publication_year"),
                }
            )
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor or not payload.get("results"):
            break
        if sleep_s > 0:
            time.sleep(sleep_s)
    return rows, {"scanned_openalex_candidates": scanned, "openalex_skipped": dict(skipped)}


def make_batches(
    rows: list[dict[str, Any]],
    *,
    batch_count: int,
    labels_per_batch: int,
    positives_per_label: int,
    min_docs_per_label: int,
    seed: int,
    loss: str,
    needles: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    contaminated_examples = []
    seen_texts: set[str] = set()
    for row in rows:
        if contaminated(row["text"], needles):
            skipped["benchmark_overlap"] += 1
            if len(contaminated_examples) < 20:
                contaminated_examples.append({"id": row["id"], "title": row["title"][:160]})
            continue
        key = normalize(row["text"])[:300]
        if key in seen_texts:
            skipped["duplicate_prefix"] += 1
            continue
        seen_texts.add(key)
        by_label[row["label"]].append(row)

    labels = [label for label, values in by_label.items() if len(values) >= min_docs_per_label]
    labels = sorted(labels, key=lambda label: len(by_label[label]), reverse=True)
    if len(labels) < labels_per_batch:
        raise ValueError(f"Need at least {labels_per_batch} labels, got {len(labels)}")
    for label in labels:
        rng.shuffle(by_label[label])

    records: list[dict[str, Any]] = []
    sampled = Counter()
    cursors = Counter()
    label_pool = labels[: max(labels_per_batch, min(len(labels), labels_per_batch * 8))]
    for batch_index in range(batch_count):
        offset = (batch_index * labels_per_batch) % len(label_pool)
        selected = [label_pool[(offset + i) % len(label_pool)] for i in range(labels_per_batch)]
        if batch_index % 7 == 0:
            selected = rng.sample(label_pool, labels_per_batch)
        for label in selected:
            values = by_label[label]
            for text_index in range(positives_per_label):
                cursor = cursors[label] % len(values)
                item = values[cursor]
                cursors[label] += 1
                if cursors[label] % len(values) == 0:
                    rng.shuffle(values)
                records.append(
                    {
                        "source": "openalex:ru_primary_topic_subfield",
                        "objective": "labeled_text",
                        "text": item["text"],
                        "label": f"openalex_fos::{label}",
                        "loss": loss,
                        "metadata": {
                            "openalex_id": item["id"],
                            "openalex_subfield": label,
                            "openalex_field": item.get("field"),
                            "openalex_domain": item.get("domain"),
                            "publication_year": item.get("publication_year"),
                            "batch_index": batch_index,
                            "text_index": text_index,
                            "contamination_policy": (
                                "OpenAlex Russian works with Cyrillic-heavy title/abstract text. "
                                "Exact normalized text, front prefix, and title overlaps with "
                                "RuSciBench GRNTI/OECD classification and clustering datasets "
                                "are removed. No benchmark rows or released model used."
                            ),
                        },
                    }
                )
                sampled[label] += 1
    return records, {
        "records": len(records),
        "usable_labels": len(labels),
        "label_pool": label_pool,
        "source_label_counts": {label: len(by_label[label]) for label in labels[:80]},
        "sampled_label_counts": dict(sampled),
        "batch_size": labels_per_batch * positives_per_label,
        "filter_skipped": dict(skipped),
        "contaminated_examples": contaminated_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/openalex_ru_fos_subfield_circle_b8_3200_seed2641.jsonl",
    )
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--per-page", type=int, default=200)
    parser.add_argument("--batch-count", type=int, default=400)
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--positives-per-label", type=int, default=2)
    parser.add_argument("--min-docs-per-label", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2641)
    parser.add_argument("--loss", choices=["supcon", "circle", "multi_similarity"], default="circle")
    parser.add_argument("--sleep-s", type=float, default=0.05)
    parser.add_argument("--mailto", default=os.environ.get("OPENALEX_MAILTO"))
    args = parser.parse_args()

    needles = benchmark_needles()
    source_rows, source_summary = openalex_rows(
        max_candidates=args.max_candidates,
        per_page=args.per_page,
        mailto=args.mailto,
        sleep_s=args.sleep_s,
    )
    records, batch_summary = make_batches(
        source_rows,
        batch_count=args.batch_count,
        labels_per_batch=args.labels_per_batch,
        positives_per_label=args.positives_per_label,
        min_docs_per_label=args.min_docs_per_label,
        seed=args.seed,
        loss=args.loss,
        needles=needles,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "output": str(args.output),
        "openalex_api": OPENALEX_API,
        "max_candidates": args.max_candidates,
        "batch_count": args.batch_count,
        "labels_per_batch": args.labels_per_batch,
        "positives_per_label": args.positives_per_label,
        "loss": args.loss,
        "seed": args.seed,
        "benchmark_needles": {key: len(value) for key, value in needles.items()},
        **source_summary,
        **batch_summary,
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
