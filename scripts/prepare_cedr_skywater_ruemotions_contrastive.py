from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    CACHE_DIR,
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
)


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]
SOURCE_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


def clean_text(value: Any) -> str:
    text = URL_RE.sub(" ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return text


def text_ok(text: str, *, max_chars: int) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 16 or len(normalized) > max_chars:
        return False
    letters = sum(ch.isalpha() for ch in text)
    cyrillic = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)
    if letters < 0.45 * max(1, len(text)):
        return False
    if cyrillic < 0.55 * max(1, letters):
        return False
    return True


def map_label(raw_labels: list[int]) -> str | None:
    labels = []
    for raw_label in raw_labels:
        label = SOURCE_LABELS[int(raw_label)]
        if label == "disgust":
            continue
        labels.append(label)
    labels = sorted(set(labels), key=LABELS.index)
    if "neutral" in labels and len(labels) > 1:
        labels = [label for label in labels if label != "neutral"]
    if len(labels) != 1:
        return None
    return labels[0]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare strict SkyWater21/ru_emotions contrastive CEDR component.")
    parser.add_argument("--name", default="cedr_skywater_ruemotions_contrastive_strict_7200")
    parser.add_argument("--per-label", type=int, default=1200)
    parser.add_argument("--pool-margin", type=int, default=300)
    parser.add_argument("--max-chars", type=int, default=360)
    parser.add_argument("--seed", type=int, default=2037)
    parser.add_argument("--splits", nargs="+", default=["comb_train", "comb_validation"])
    parser.add_argument("--dataset-cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    pools: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    skipped = Counter()
    target_pool = args.per_label + args.pool_margin
    enough = lambda: all(len(pools[label]) >= target_pool for label in LABELS)

    for split in args.splits:
        dataset = load_dataset("SkyWater21/ru_emotions", split=split, cache_dir=str(args.dataset_cache_dir))
        for row in dataset:
            if enough():
                break
            text = clean_text(row.get("ru_text") or row.get("text") or "")
            label = map_label(row.get("labels") or [])
            if label is None:
                skipped["unmapped_or_multilabel"] += 1
                continue
            if not text_ok(text, max_chars=args.max_chars):
                skipped["quality"] += 1
                continue
            normalized = normalize_text(text)
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            pools[label].append(text)
        if enough():
            break

    missing = [label for label in LABELS if len(pools[label]) < 2]
    if missing:
        raise RuntimeError(f"Not enough labels: {missing}")

    selected: dict[str, list[str]] = {}
    for label in LABELS:
        items = pools[label][:]
        rng.shuffle(items)
        selected[label] = items[: min(args.per_label, len(items))]

    records: list[dict[str, Any]] = []
    for label in LABELS:
        for index, query_text in enumerate(selected[label]):
            positive_pool = [text for text in selected[label] if text != query_text]
            if not positive_pool:
                continue
            negatives = []
            for negative_label in LABELS:
                if negative_label == label:
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(selected[negative_label]))
            records.append(
                {
                    "source": "SkyWater21/ru_emotions:strict_contrastive",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + query_text,
                    "positive": CEDR_PREFIX + rng.choice(positive_pool),
                    "negatives": negatives,
                    "metadata": {
                        "group": label,
                        "index": index,
                        "source_dataset": "SkyWater21/ru_emotions",
                    },
                }
            )
    rng.shuffle(records)

    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    summary = {
        "name": args.name,
        "records": len(records),
        "per_label": args.per_label,
        "selected_by_label": {label: len(rows) for label, rows in selected.items()},
        "available_by_label": {label: len(rows) for label, rows in pools.items()},
        "skipped": dict(skipped),
        "splits": args.splits,
        "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
    }
    out.with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
