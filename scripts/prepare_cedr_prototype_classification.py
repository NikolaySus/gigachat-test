from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_texts(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        label = str(row.get("metadata", {}).get("group") or row.get("metadata", {}).get("label") or "")
        if label == "no_emotion":
            label = "neutral"
        if not label:
            continue
        for field in ("query", "positive"):
            text = row.get(field)
            if not isinstance(text, str) or not text.strip():
                continue
            key = (label, text)
            if key in seen:
                continue
            seen.add(key)
            grouped[label].append(text)
    return dict(grouped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CEDR-style prototype classification rows.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_go9000_prototype_cls_p2_9000.jsonl",
    )
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--count", type=int, default=9000)
    parser.add_argument("--prototypes-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=829)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.source)
    grouped = group_texts(rows)
    labels = [label for label in ["neutral", "joy", "sadness", "surprise", "fear", "anger"] if label in grouped]
    if len(labels) < 2:
        raise RuntimeError(f"Need at least two labels, got {labels}")
    for label in labels:
        if len(grouped[label]) <= args.prototypes_per_class:
            raise RuntimeError(f"Not enough texts for label {label}: {len(grouped[label])}")

    query_items: list[tuple[str, str]] = []
    for label in labels:
        query_items.extend((label, text) for text in grouped[label])
    rng.shuffle(query_items)
    query_items = query_items[: args.count]

    output = []
    for index, (label, query) in enumerate(query_items):
        prototypes: dict[str, list[str]] = {}
        for prototype_label in labels:
            pool = [text for text in grouped[prototype_label] if text != query]
            prototypes[prototype_label] = rng.sample(pool, args.prototypes_per_class)
        output.append(
            {
                "source": f"{args.source.stem}:prototype_classification_p{args.prototypes_per_class}",
                "objective": "prototype_classification",
                "query": query,
                "label": label,
                "prototypes": prototypes,
                "metadata": {
                    "group": label,
                    "source_path": str(args.source),
                    "index": index,
                    "prototypes_per_class": args.prototypes_per_class,
                },
            }
        )

    write_jsonl(args.out, output)
    summary = {
        "source": str(args.source),
        "output": str(args.out),
        "records": len(output),
        "labels": labels,
        "label_counts": Counter(row["label"] for row in output),
        "available_texts": {label: len(grouped[label]) for label in labels},
        "prototypes_per_class": args.prototypes_per_class,
        "seed": args.seed,
        "contamination_policy": "inherits source CEDR-overlap filtering; no CEDR records added",
    }
    summary_path = args.summary_out or args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
