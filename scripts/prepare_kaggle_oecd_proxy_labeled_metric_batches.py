#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_kaggle_grnti_labeled_metric_batches import benchmark_needles, read_articles
from prepare_kaggle_oecd_proxy_knn_episodes import GRNTI_TO_OECD_PROXY, grnti_number


ROOT = Path(__file__).resolve().parents[1]


def map_articles_to_proxy(
    by_grnti: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_proxy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for grnti_label, rows in by_grnti.items():
        number = grnti_number(grnti_label)
        proxy = GRNTI_TO_OECD_PROXY.get(number or -1)
        if proxy is None:
            skipped["unmapped_grnti"] += len(rows)
            continue
        for row in rows:
            item = dict(row)
            item["grnti_label"] = grnti_label
            item["oecd_proxy"] = proxy
            by_proxy[proxy].append(item)
    return by_proxy, {
        "proxy_doc_counts": {label: len(rows) for label, rows in by_proxy.items()},
        "skipped_mapping": dict(skipped),
    }


def read_audited_grnti_jsonl(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_grnti: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = str(record.get("text", "")).strip()
            grnti_label = str(record.get("metadata", {}).get("grnti_label", "")).strip()
            file_name = str(record.get("metadata", {}).get("file", "")).strip()
            if not text or not grnti_label:
                skipped["missing_text_or_grnti"] += 1
                continue
            by_grnti[grnti_label].append({"text": text, "file": file_name})
    return by_grnti, {
        "input_jsonl": str(path),
        "labels_before_filter": len(by_grnti),
        "skipped": dict(skipped),
        "contamination_policy": (
            "Inherits the source JSONL audit. The source was created from Kaggle "
            "scientific articles after removing exact title/prefix overlap with "
            "RuSciBench GRNTI/OECD train/test."
        ),
    }


def make_batches(
    by_proxy: dict[str, list[dict[str, Any]]],
    *,
    batch_count: int,
    positives_per_label: int,
    min_docs_per_label: int,
    seed: int,
    loss: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    labels = sorted(label for label, rows in by_proxy.items() if len(rows) >= min_docs_per_label)
    if len(labels) < 2:
        raise ValueError(f"Need at least two proxy labels, got {len(labels)}")
    for rows in by_proxy.values():
        rng.shuffle(rows)

    records: list[dict[str, Any]] = []
    sampled = Counter()
    cursors = Counter()
    for batch_index in range(batch_count):
        # Keep every metric batch broad: all available OECD-style proxies appear
        # with multiple positives. Cycling avoids overusing the same first rows.
        for label in labels:
            rows = by_proxy[label]
            chosen = []
            for _ in range(positives_per_label):
                cursor = cursors[label] % len(rows)
                chosen.append(rows[cursor])
                cursors[label] += 1
                if cursors[label] % len(rows) == 0:
                    rng.shuffle(rows)
            for text_index, item in enumerate(chosen):
                records.append(
                    {
                        "source": "kaggle/ergkerg/russian-scientific-articles:oecd_proxy_labeled_metric",
                        "objective": "labeled_text",
                        "text": item["text"],
                        "label": f"oecd_proxy::{label}",
                        "loss": loss,
                        "metadata": {
                            "oecd_proxy": label,
                            "grnti_label": item["grnti_label"],
                            "file": item["file"],
                            "batch_index": batch_index,
                            "text_index": text_index,
                            "contamination_policy": (
                                "Derived from audited Kaggle Russian scientific articles. "
                                "Exact title/prefix overlaps with RuSciBench GRNTI/OECD "
                                "train/test were removed before sampling. OECD proxy labels "
                                "are hand-mapped from GRNTI category names; no benchmark "
                                "rows or released model used."
                            ),
                        },
                    }
                )
                sampled[label] += 1
    return records, {
        "usable_proxy_labels": labels,
        "sampled_proxy_counts": dict(sampled),
        "batch_size": len(labels) * positives_per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "data/kagglehub_cache/datasets/ergkerg/russian-scientific-articles/versions/1",
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_grnti_labeled_circle_b4_1600_seed2551.jsonl",
        help="Use an already audited GRNTI labeled_text JSONL instead of rescanning raw Kaggle files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/contrastive/open_ru_1r_nc_kaggle_oecd_proxy_labeled_circle_b12_4800_seed2631.jsonl",
    )
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--batch-count", type=int, default=400)
    parser.add_argument("--positives-per-label", type=int, default=2)
    parser.add_argument("--min-docs-per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2631)
    parser.add_argument("--loss", choices=["supcon", "circle", "multi_similarity"], default="circle")
    args = parser.parse_args()

    if args.source_jsonl is not None and args.source_jsonl.exists():
        needles = {}
        by_grnti, audit_summary = read_audited_grnti_jsonl(args.source_jsonl)
    else:
        needles = benchmark_needles()
        by_grnti, audit_summary = read_articles(args.input_root, max_chars=args.max_chars, needles=needles)
    by_proxy, proxy_summary = map_articles_to_proxy(by_grnti)
    records, batch_summary = make_batches(
        by_proxy,
        batch_count=args.batch_count,
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
        "positives_per_label": args.positives_per_label,
        "max_chars": args.max_chars,
        "loss": args.loss,
        "seed": args.seed,
        "proxy_mapping": GRNTI_TO_OECD_PROXY,
        "benchmark_needles": {
            key: {name: len(items) for name, items in values.items()}
            for key, values in needles.items()
        },
        **audit_summary,
        **proxy_summary,
        **batch_summary,
        "contamination_policy": (
            "RuSciBench GRNTI/OECD train and test title/prefix overlaps removed; "
            "MTEB/RuSciBench rows are not used for training."
        ),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
