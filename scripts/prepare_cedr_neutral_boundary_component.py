from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_cedr_goemotions_ru_hardneg_component import select_rows
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
EMOTION_GROUPS = [group for group in GROUPS if group != "neutral"]


def strip_for_tfidf(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " URL ", text)
    return re.sub(r"\\s+", " ", text).strip()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a neutral-vs-emotion CEDR boundary component.")
    parser.add_argument("--count", type=int, default=9000)
    parser.add_argument("--neutral-fraction", type=float, default=0.5)
    parser.add_argument("--neutral-records", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=841)
    parser.add_argument("--name", default="cedr_goemotions_neutral_boundary_3600")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    selected_by_group, base_summary = select_rows(args.count, args.neutral_fraction, args.seed)
    neutral_pool = selected_by_group["neutral"][:]
    rng.shuffle(neutral_pool)
    neutral_items = neutral_pool[: args.neutral_records]
    if len(neutral_items) < args.neutral_records:
        raise ValueError(f"Need {args.neutral_records} neutral rows, got {len(neutral_items)}")

    items = [item for group in GROUPS for item in selected_by_group[group]]
    texts = [strip_for_tfidf(item["text"]) for item in items]
    groups = np.array([item["group"] for item in items])
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=250_000)
    matrix = normalize(vectorizer.fit_transform(texts), copy=False)
    group_indices = {group: np.flatnonzero(groups == group) for group in GROUPS}
    item_pos = {id(item): pos for pos, item in enumerate(items)}

    records = []
    hardneg_scores: list[float] = []
    for item in neutral_items:
        item_index = item_pos[id(item)]
        positives = [candidate for candidate in neutral_items if candidate is not item]
        positive = rng.choice(positives)
        negatives = []
        for negative_group in EMOTION_GROUPS:
            candidate_indices = group_indices[negative_group]
            sims = matrix[item_index].dot(matrix[candidate_indices].T).toarray().ravel()
            best_local = int(sims.argmax())
            hardneg_scores.append(float(sims[best_local]))
            negatives.append(CEDR_PREFIX + items[int(candidate_indices[best_local])]["text"])
        records.append(
            {
                "source": "AiLab-IMCS-UL/go_emotions-ru:cedr_neutral_boundary",
                "objective": "contrastive",
                "query": CEDR_PREFIX + item["text"],
                "positive": CEDR_PREFIX + positive["text"],
                "negatives": negatives,
                "metadata": {
                    "group": "neutral",
                    "split": item["split"],
                    "index": item["index"],
                    "negative_groups": EMOTION_GROUPS,
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
            "source": "AiLab-IMCS-UL/go_emotions-ru",
            "construction": "neutral query/neutral positive, TF-IDF hard negative from every emotion group",
            "base_count": args.count,
            "base_neutral_fraction": args.neutral_fraction,
            "neutral_records": len(records),
            "selected": dict(Counter(record["metadata"]["group"] for record in records)),
            "base_summary": base_summary,
            "hard_negative_tfidf": {
                "analyzer": "char_wb",
                "ngram_range": [3, 5],
                "min_df": 2,
                "mean_similarity": sum(hardneg_scores) / max(1, len(hardneg_scores)),
                "p95_similarity": float(np.percentile(hardneg_scores, 95)) if hardneg_scores else 0.0,
            },
            "contamination_policy": "inherits exact and near CEDR overlap filtering from select_rows",
        },
    )
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
