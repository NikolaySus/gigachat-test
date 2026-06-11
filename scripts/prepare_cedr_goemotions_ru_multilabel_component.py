from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    CACHE_DIR,
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
CEDR_PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
EMOTIONS = ["joy", "sadness", "surprise", "fear", "anger"]
EMOTION_PRIOR = {"joy": 1569, "sadness": 1417, "surprise": 607, "fear": 589, "anger": 411}


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def target_counts(count: int, *, neutral_fraction: float) -> dict[str, int]:
    neutral = int(round(count * neutral_fraction))
    emotion_total = count - neutral
    total_prior = sum(EMOTION_PRIOR.values())
    targets = {"neutral": neutral}
    allocated = 0
    remainders = []
    for label, prior in EMOTION_PRIOR.items():
        raw = emotion_total * prior / total_prior
        value = int(raw)
        targets[label] = value
        allocated += value
        remainders.append((raw - value, label))
    for _, label in sorted(remainders, reverse=True)[: emotion_total - allocated]:
        targets[label] += 1
    return targets


def remaining_need(labels: set[str], counts: Counter[str], targets: dict[str, int]) -> int:
    return sum(max(0, targets[label] - counts[label]) for label in labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean multi-label-aware AiLab GoEmotions-RU CEDR component.")
    parser.add_argument("--count", type=int, default=6400)
    parser.add_argument("--neutral-fraction", type=float, default=0.5)
    parser.add_argument(
        "--max-label-overfill",
        type=float,
        default=1.15,
        help="Maximum selected label-instance count relative to the target before a row is skipped.",
    )
    parser.add_argument("--seed", type=int, default=771)
    parser.add_argument("--name", default="cedr_ailab_goemotions_ru_multilabel_prior_neutral50_6400")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    targets = target_counts(args.count, neutral_fraction=args.neutral_fraction)
    cedr_index = load_cedr_index()
    dataset = load_dataset("AiLab-IMCS-UL/go_emotions-ru", cache_dir=str(CACHE_DIR))
    label_names = dataset["train"].features["labels_ekman"].feature.names

    neutral_pool: list[dict[str, Any]] = []
    emotion_pool: list[dict[str, Any]] = []
    skipped = Counter()
    seen = set()
    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            labels = {label_names[label] for label in row["labels_ekman"]}
            labels = labels & set(EMOTIONS + ["neutral"])
            if "neutral" in labels and len(labels) > 1:
                labels.remove("neutral")
            if not labels:
                skipped["unmapped"] += 1
                continue
            text = clean_text(row["ru_text"])
            normalized = normalize_text(text)
            if len(normalized) < 8 or len(normalized) > 360:
                skipped["length"] += 1
                continue
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            item = {"split": split, "index": index, "text": text, "labels": sorted(labels)}
            if labels == {"neutral"}:
                neutral_pool.append(item)
            else:
                item["labels"] = sorted(label for label in labels if label in EMOTIONS)
                if item["labels"]:
                    emotion_pool.append(item)

    rng.shuffle(neutral_pool)
    selected = neutral_pool[: targets["neutral"]]
    if len(selected) < targets["neutral"]:
        raise ValueError(f"Not enough neutral rows: need {targets['neutral']}, got {len(selected)}")

    counts: Counter[str] = Counter()
    selected_keys = {(row["split"], row["index"]) for row in selected}
    emotion_pool.sort(
        key=lambda row: (
            -len(row["labels"]),
            -sum(1.0 / EMOTION_PRIOR[label] for label in row["labels"]),
            rng.random(),
        )
    )
    emotion_targets = {label: targets[label] for label in EMOTIONS}
    emotion_caps = {
        label: max(1, int(round(emotion_targets[label] * args.max_label_overfill)))
        for label in EMOTIONS
    }

    while len(selected) < args.count:
        best = None
        best_score = -1
        for row in emotion_pool:
            key = (row["split"], row["index"])
            if key in selected_keys:
                continue
            labels = set(row["labels"])
            if any(counts[label] >= emotion_caps[label] for label in labels):
                continue
            score = remaining_need(labels, counts, targets)
            if score > best_score:
                best = row
                best_score = score
        if best is None or best_score <= 0:
            leftovers = [
                row
                for row in emotion_pool
                if (row["split"], row["index"]) not in selected_keys
                and not any(counts[label] >= emotion_caps[label] for label in row["labels"])
            ]
            if not leftovers:
                break
            best = rng.choice(leftovers)
        selected.append(best)
        selected_keys.add((best["split"], best["index"]))
        for label in best["labels"]:
            counts[label] += 1

    by_overlap: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        key = "|".join(item["labels"])
        by_overlap[key].append(item)

    records = []
    dropped = Counter()
    for item in selected:
        labels = set(item["labels"])
        positives = [
            candidate
            for candidate in selected
            if candidate is not item and labels & set(candidate["labels"])
        ]
        negatives = [
            candidate
            for candidate in selected
            if not (labels & set(candidate["labels"]))
        ]
        if not positives or len(negatives) < 5:
            dropped["no_positive_or_negative"] += 1
            continue
        rng.shuffle(negatives)
        positive = rng.choice(positives)
        records.append(
            {
                "source": "AiLab-IMCS-UL/go_emotions-ru:cedr_multilabel_prior",
                "objective": "contrastive",
                "query": CEDR_PREFIX + item["text"],
                "positive": CEDR_PREFIX + positive["text"],
                "negatives": [CEDR_PREFIX + negative["text"] for negative in negatives[:5]],
                "metadata": {
                    "labels": item["labels"],
                    "group": "+".join(item["labels"]),
                    "split": item["split"],
                    "index": item["index"],
                },
            }
        )
    rng.shuffle(records)

    path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(path, records)
    write_json(
        path.with_name(path.stem + "_summary.json"),
        {
            "name": args.name,
            "requested": args.count,
            "kept": len(records),
            "targets": targets,
            "emotion_caps": emotion_caps,
            "max_label_overfill": args.max_label_overfill,
            "neutral_fraction": args.neutral_fraction,
            "selected_label_instances": dict(counts),
            "selected_labelsets": dict(Counter("+".join(row["labels"]) for row in selected)),
            "available": {"neutral": len(neutral_pool), "emotion": len(emotion_pool)},
            "skipped": dict(skipped),
            "dropped_after_selection": dict(dropped),
            "contamination_policy": "exact and near CEDR overlap removed",
            "source": "AiLab-IMCS-UL/go_emotions-ru",
        },
    )
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
