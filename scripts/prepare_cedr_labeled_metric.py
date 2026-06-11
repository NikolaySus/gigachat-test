from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"

DEFAULT_SOURCES = [
    DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_dvach_clean4_4800.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_skywater_ruemotions_contrastive_strict_7200.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_brighter_rus_train_dev_clean.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_semeval2025_rus_tracka_train_dev_clean.jsonl",
]

LABELS = ["joy", "sadness", "anger", "surprise", "fear"]
PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def extract_comment(text: str) -> str:
    if PREFIX in text:
        return text.split(PREFIX, 1)[1].strip()
    marker = "комментарий:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def load_records(paths: list[Path]) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for path in paths:
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                label = record.get("metadata", {}).get("group")
                if label not in LABELS:
                    continue
                texts = []
                if record.get("objective") == "contrastive":
                    texts.extend([record.get("query", ""), record.get("positive", "")])
                    texts.extend(record.get("positives", []))
                elif record.get("objective") in {
                    "prototype_classification",
                    "knn_classification",
                    "prototype_none_classification",
                    "prototype_uniform_classification",
                }:
                    texts.append(record.get("query", ""))
                for text in texts:
                    comment = normalize_text(extract_comment(text))
                    if len(comment) < 8:
                        continue
                    key = (label, comment.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    pools[label].append(PREFIX + comment)
    return pools


def balanced_order(
    pools: dict[str, list[str]],
    *,
    loss: str,
    examples_per_label: int,
    group_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled = {}
    for label in LABELS:
        texts = list(pools[label])
        rng.shuffle(texts)
        sampled[label] = texts[:examples_per_label]
    rows: list[dict[str, Any]] = []
    offsets = {label: 0 for label in LABELS}
    while True:
        made_progress = False
        label_order = list(LABELS)
        rng.shuffle(label_order)
        for label in label_order:
            start = offsets[label]
            end = min(start + group_size, len(sampled[label]))
            if start >= end:
                continue
            made_progress = True
            offsets[label] = end
            for text in sampled[label][start:end]:
                rows.append(
                    {
                        "source": f"cedr_labeled_metric_v1:{loss}",
                        "objective": "labeled_text",
                        "text": text,
                        "label": label,
                        "loss": loss,
                        "metadata": {"group": label},
                    }
                )
        if not made_progress:
            break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="cedr_labeled_metric_v1")
    parser.add_argument("--examples-per-label", type=int, default=1800)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--source", action="append", type=Path, default=[])
    args = parser.parse_args()

    sources = args.source or DEFAULT_SOURCES
    pools = load_records(sources)
    missing = [label for label in LABELS if not pools[label]]
    if missing:
        raise RuntimeError(f"Missing labels: {missing}")

    summary = {
        "sources": [str(path) for path in sources],
        "available_by_label": {label: len(pools[label]) for label in LABELS},
        "examples_per_label": args.examples_per_label,
        "group_size": args.group_size,
        "files": {},
    }
    for loss in ["supcon", "circle", "multi_similarity"]:
        rows = balanced_order(
            pools,
            loss=loss,
            examples_per_label=args.examples_per_label,
            group_size=args.group_size,
            seed=args.seed,
        )
        path = DATA_DIR / f"open_ru_1r_nc_{args.name}_{loss}.jsonl"
        write_jsonl(path, rows)
        summary["files"][loss] = {"path": str(path), "rows": len(rows)}

    summary_path = DATA_DIR / f"open_ru_1r_nc_{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
