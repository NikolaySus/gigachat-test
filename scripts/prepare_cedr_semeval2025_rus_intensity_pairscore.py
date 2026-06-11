from __future__ import annotations

import argparse
import random
from collections import Counter
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
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["joy", "sadness", "surprise", "fear", "anger"]
LABEL_STATEMENTS = {
    "neutral": "В комментарии нет явной эмоции из списка: радость, грусть, удивление, страх или злость.",
    "joy": "В комментарии выражена радость или положительная эмоция.",
    "sadness": "В комментарии выражена грусть, печаль или тоска.",
    "surprise": "В комментарии выражено удивление, шок или неожиданность.",
    "fear": "В комментарии выражен страх, тревога или опасение.",
    "anger": "В комментарии выражена злость, раздражение или гнев.",
}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def score_from_intensity(value: Any) -> float:
    level = int(value)
    if level <= 0:
        return 0.05
    if level == 1:
        return 0.35
    if level == 2:
        return 0.70
    return 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SemEval-2025 Russian Track B intensity pair-score rows.")
    parser.add_argument("--name", default="cedr_semeval2025_rus_trackb_intensity_pairscore_train_dev")
    parser.add_argument("--seed", type=int, default=931)
    parser.add_argument("--include-neutral", action="store_true")
    parser.add_argument("--negative-zero-limit", type=int, default=2)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    dataset = load_dataset(
        "vgaraujov/semeval-2025-task11-track-b",
        "rus",
        cache_dir=str(CACHE_DIR),
    )
    records = []
    skipped = Counter()
    row_counts = Counter()
    seen = set()
    for split_name in ["train", "dev"]:
        for row in dataset[split_name]:
            text = clean_text(row["text"])
            normalized = normalize_text(text)
            if len(normalized) < 5 or len(normalized) > 420:
                skipped["length"] += 1
                continue
            key = normalized
            if key in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(key)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            positives = [label for label in LABELS if int(row[label]) > 0]
            zero_labels = [label for label in LABELS if int(row[label]) <= 0]
            rng.shuffle(zero_labels)
            used_labels = positives + zero_labels[: args.negative_zero_limit]
            if args.include_neutral:
                records.append(
                    {
                        "source": "semeval2025_task11_trackb_rus:intensity_pairscore",
                        "objective": "pair_score",
                        "sentence1": CEDR_PREFIX + text,
                        "sentence2": CEDR_PREFIX + LABEL_STATEMENTS["neutral"],
                        "score": 0.90 if not positives and int(row.get("disgust", 0)) == 0 else 0.05,
                        "metadata": {
                            "group": "neutral",
                            "split": split_name,
                            "id": row["id"],
                            "kind": "neutral_statement",
                        },
                    }
                )
            for label in used_labels:
                records.append(
                    {
                        "source": "semeval2025_task11_trackb_rus:intensity_pairscore",
                        "objective": "pair_score",
                        "sentence1": CEDR_PREFIX + text,
                        "sentence2": CEDR_PREFIX + LABEL_STATEMENTS[label],
                        "score": score_from_intensity(row[label]),
                        "metadata": {
                            "group": label,
                            "split": split_name,
                            "id": row["id"],
                            "intensity": int(row[label]),
                            "kind": "emotion_statement",
                        },
                    }
                )
            row_counts[split_name] += 1

    rng.shuffle(records)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "records": len(records),
            "source": "vgaraujov/semeval-2025-task11-track-b/rus",
            "splits": ["train", "dev"],
            "rows_used": dict(row_counts),
            "record_groups": dict(Counter(record["metadata"]["group"] for record in records)),
            "score_counts": dict(Counter(str(record["score"]) for record in records)),
            "include_neutral": args.include_neutral,
            "negative_zero_limit": args.negative_zero_limit,
            "skipped": dict(skipped),
            "construction": "pair-score calibration from Russian emotion intensity labels; test split excluded",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
