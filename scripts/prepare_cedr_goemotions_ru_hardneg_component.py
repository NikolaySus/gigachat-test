from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS, clean_text, target_counts
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


def strip_for_tfidf(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " URL ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def select_rows(count: int, neutral_fraction: float, seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cedr_index = load_cedr_index()
    dataset = load_dataset("AiLab-IMCS-UL/go_emotions-ru", cache_dir=str(CACHE_DIR))
    label_names = dataset["train"].features["labels_ekman"].feature.names
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()
    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            labels = [label_names[label] for label in row["labels_ekman"]]
            labels = [label for label in labels if label in GROUPS]
            if len(labels) != 1:
                skipped["multilabel_or_unmapped"] += 1
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
            group = labels[0]
            pools[group].append({"split": split, "index": index, "text": text, "group": group})

    rng = random.Random(seed)
    targets = target_counts(count, neutral_fraction=neutral_fraction)
    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    for group, target in targets.items():
        pool = pools[group][:]
        rng.shuffle(pool)
        if len(pool) < target:
            raise ValueError(f"Not enough rows for {group}: need {target}, got {len(pool)}")
        selected_by_group[group] = pool[:target]
    return selected_by_group, {
        "targets": targets,
        "available": {group: len(rows) for group, rows in pools.items()},
        "skipped": dict(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean AiLab GoEmotions-RU CEDR component with TF-IDF hard negatives.")
    parser.add_argument("--count", type=int, default=6400)
    parser.add_argument("--neutral-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=781)
    parser.add_argument("--name", default="cedr_ailab_goemotions_ru_prior_hardneg_6400")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    selected_by_group, summary = select_rows(args.count, args.neutral_fraction, args.seed)
    items = [item for group in GROUPS for item in selected_by_group[group]]
    texts = [strip_for_tfidf(item["text"]) for item in items]
    groups = np.array([item["group"] for item in items])

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=250_000)
    matrix = normalize(vectorizer.fit_transform(texts), copy=False)
    group_indices = {group: np.flatnonzero(groups == group) for group in GROUPS}
    item_pos = {id(item): pos for pos, item in enumerate(items)}

    records = []
    hardneg_scores: list[float] = []
    for group, group_items in selected_by_group.items():
        same_group = group_items[:]
        for item in group_items:
            item_index = item_pos[id(item)]
            positives = [candidate for candidate in same_group if candidate is not item]
            positive = rng.choice(positives)
            negatives: list[str] = []
            for negative_group in GROUPS:
                if negative_group == group:
                    continue
                candidate_indices = group_indices[negative_group]
                sims = matrix[item_index].dot(matrix[candidate_indices].T).toarray().ravel()
                best_local = int(sims.argmax())
                hardneg_scores.append(float(sims[best_local]))
                negatives.append(CEDR_PREFIX + items[int(candidate_indices[best_local])]["text"])
            records.append(
                {
                    "source": "AiLab-IMCS-UL/go_emotions-ru:cedr_prior_hardneg",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": group,
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
            "neutral_fraction": args.neutral_fraction,
            **summary,
            "selected": dict(Counter(record["metadata"]["group"] for record in records)),
            "hard_negative_tfidf": {
                "analyzer": "char_wb",
                "ngram_range": [3, 5],
                "min_df": 2,
                "mean_similarity": sum(hardneg_scores) / max(1, len(hardneg_scores)),
                "p95_similarity": float(np.percentile(hardneg_scores, 95)) if hardneg_scores else 0.0,
            },
            "contamination_policy": "exact and near CEDR overlap removed",
            "source": "AiLab-IMCS-UL/go_emotions-ru",
        },
    )
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
