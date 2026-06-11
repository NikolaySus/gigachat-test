from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
GEO_PREFIX = "Определи категорию организации на основе отзыва \nотзыв: "


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_text(text: str) -> str:
    text = re.sub(r"^Instruct:[^\n]*\nQuery:\s*", "", str(text).strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_labeled(
    by_label: dict[str, list[str]],
    *,
    label: str,
    text: str,
    min_chars: int,
    max_chars: int,
) -> None:
    text = clean_text(text)
    if not (min_chars <= len(text) <= max_chars):
        return
    by_label[label].append(GEO_PREFIX + text)


def collect_geracl(rows: list[dict[str, Any]], *, min_chars: int, max_chars: int) -> dict[str, list[str]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        scenario = metadata.get("scenario")
        if not scenario:
            continue
        label = f"geracl_scenario::{scenario}"
        add_labeled(by_label, label=label, text=row.get("query", ""), min_chars=min_chars, max_chars=max_chars)
        add_labeled(by_label, label=label, text=row.get("positive", ""), min_chars=min_chars, max_chars=max_chars)
    return by_label


def collect_grandmaster(rows: list[dict[str, Any]], *, min_chars: int, max_chars: int) -> dict[str, list[str]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        if "cluster" not in metadata:
            continue
        label = f"grandmaster_cluster::{metadata['cluster']}"
        add_labeled(by_label, label=label, text=row.get("query", ""), min_chars=min_chars, max_chars=max_chars)
        add_labeled(by_label, label=label, text=row.get("positive", ""), min_chars=min_chars, max_chars=max_chars)
    return by_label


def build_grouped_batches(
    pools: dict[str, list[str]],
    *,
    labels_per_batch: int,
    examples_per_label: int,
    max_batches: int,
    seed: int,
    loss: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    usable = {label: texts[:] for label, texts in pools.items() if len(texts) >= examples_per_label}
    for texts in usable.values():
        rng.shuffle(texts)
    labels = list(usable)
    records: list[dict[str, Any]] = []
    cursors = Counter()
    for _ in range(max_batches):
        available = [
            label
            for label in labels
            if cursors[label] + examples_per_label <= len(usable[label])
        ]
        if len(available) < labels_per_batch:
            break
        batch_labels = rng.sample(available, labels_per_batch)
        for label in batch_labels:
            start = cursors[label]
            cursors[label] += examples_per_label
            for text in usable[label][start : start + examples_per_label]:
                records.append(
                    {
                        "source": "fair_geo_cluster_labeled",
                        "objective": "labeled_text",
                        "text": text,
                        "label": label,
                        "loss": loss,
                        "metadata": {
                            "construction": "grouped_labeled_metric_learning",
                            "labels_per_batch": labels_per_batch,
                            "examples_per_label": examples_per_label,
                        },
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fair grouped labeled-text data for Georeview clustering repair.")
    parser.add_argument("--geracl-path", type=Path, default=DATA_DIR / "open_ru_1r_nc_geracl.jsonl")
    parser.add_argument("--grandmaster-path", type=Path, default=DATA_DIR / "open_ru_1r_nc_grandmaster_clustered_3200.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "fair_geo_cluster_labeled_ms_6000_seed2034.jsonl")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2034)
    parser.add_argument("--max-records", type=int, default=6000)
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--examples-per-label", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--loss", choices=("supcon", "circle", "multi_similarity"), default="multi_similarity")
    args = parser.parse_args()

    geracl = collect_geracl(read_jsonl(args.geracl_path), min_chars=args.min_chars, max_chars=args.max_chars)
    grandmaster = collect_grandmaster(read_jsonl(args.grandmaster_path), min_chars=args.min_chars, max_chars=args.max_chars)
    pools = defaultdict(list)
    for source in (geracl, grandmaster):
        for label, texts in source.items():
            pools[label].extend(texts)

    max_batches = args.max_records // (args.labels_per_batch * args.examples_per_label)
    records = build_grouped_batches(
        pools,
        labels_per_batch=args.labels_per_batch,
        examples_per_label=args.examples_per_label,
        max_batches=max_batches,
        seed=args.seed,
        loss=args.loss,
    )
    write_jsonl(args.output, records)
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    label_counts = Counter(record["label"] for record in records)
    write_json(
        summary_path,
        {
            "output": str(args.output),
            "records": len(records),
            "loss": args.loss,
            "seed": args.seed,
            "batch_size": args.labels_per_batch * args.examples_per_label,
            "labels_per_batch": args.labels_per_batch,
            "examples_per_label": args.examples_per_label,
            "max_steps_1x": len(records) // (args.labels_per_batch * args.examples_per_label),
            "geracl_labels": len(geracl),
            "grandmaster_labels": len(grandmaster),
            "selected_labels": len(label_counts),
            "selected_label_prefix_counts": dict(Counter(label.split("::", 1)[0] for label in label_counts)),
            "selected_label_size_histogram": dict(Counter(label_counts.values())),
        },
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
