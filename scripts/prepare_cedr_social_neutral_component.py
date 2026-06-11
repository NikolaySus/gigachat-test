from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX  # noqa: E402
from prepare_open_ru_1r_nc_cedr_sentiment_mined_v2 import (  # noqa: E402
    DATA_DIR,
    EMOTION_GROUPS,
    load_cedr_index,
    load_primary_sources,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean social-media neutral CEDR correction component.")
    parser.add_argument("--count", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--name", default="cedr_social_neutral_3200")
    parser.add_argument("--negatives-per-row", type=int, default=5)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    rows, loader_meta = load_primary_sources(cedr_index)
    neutral = [row for row in rows if row["group"] == "no_emotion"]
    emotions: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["group"] in EMOTION_GROUPS:
            emotions[row["group"]].append(row)

    neutral.sort(
        key=lambda row: (
            row["style"] == "informal",
            row["quality_score"],
            row["confidence"],
            len(row["text"]),
        ),
        reverse=True,
    )
    selected = neutral[: args.count]
    if len(selected) < args.count:
        raise RuntimeError(f"Need {args.count} neutral rows, found {len(selected)}")

    texts = [row["text"] for row in neutral]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    neighbors = NearestNeighbors(n_neighbors=min(80, len(neutral)), metric="cosine", algorithm="brute", n_jobs=-1)
    neighbors.fit(matrix)
    index_by_id = {
        (row["source_dataset"], row["split"], row["index"]): index
        for index, row in enumerate(neutral)
    }

    records = []
    positive_sims = []
    emotion_groups = [group for group in ["joy", "sadness", "anger", "fear", "surprise"] if emotions[group]]
    for row in selected:
        row_index = index_by_id[(row["source_dataset"], row["split"], row["index"])]
        distances, indices = neighbors.kneighbors(matrix[row_index], return_distance=True)
        positive_index = None
        positive_sim = 0.0
        for distance, candidate_index in zip(distances[0], indices[0], strict=True):
            candidate_index = int(candidate_index)
            if candidate_index == row_index:
                continue
            candidate = neutral[candidate_index]
            if candidate["text"] == row["text"]:
                continue
            positive_index = candidate_index
            positive_sim = float(1.0 - distance)
            break
        if positive_index is None:
            positive_index = rng.randrange(len(neutral))
            while positive_index == row_index:
                positive_index = rng.randrange(len(neutral))
        positive = neutral[positive_index]
        positive_sims.append(positive_sim)

        negatives = []
        shuffled_groups = emotion_groups[:]
        rng.shuffle(shuffled_groups)
        for group in shuffled_groups:
            pool = emotions[group]
            pool.sort(key=lambda item: (item["quality_score"], item["style"] == "informal"), reverse=True)
            top = pool[: min(len(pool), 1200)]
            negatives.append(CEDR_PREFIX + rng.choice(top)["text"])
            if len(negatives) >= args.negatives_per_row:
                break

        records.append(
            {
                "source": "cedr_social_neutral:no_cedr_overlap",
                "objective": "contrastive",
                "query": CEDR_PREFIX + row["text"],
                "positive": CEDR_PREFIX + positive["text"],
                "negatives": negatives,
                "metadata": {
                    "group": "neutral",
                    "source_dataset": row["source_dataset"],
                    "split": row["split"],
                    "index": row["index"],
                    "style": row["style"],
                    "confidence": row["confidence"],
                    "quality_score": row["quality_score"],
                    "positive_similarity": round(positive_sim, 6),
                    "negative_groups": shuffled_groups[: len(negatives)],
                },
            }
        )

    rng.shuffle(records)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "records": len(records),
            "requested": args.count,
            "selected_by_source": dict(Counter(row["metadata"]["source_dataset"] for row in records)),
            "selected_by_style": dict(Counter(row["metadata"]["style"] for row in records)),
            "loader_meta": loader_meta,
            "available_neutral": len(neutral),
            "available_emotion_groups": {group: len(items) for group, items in emotions.items()},
            "positive_similarity_mean": sum(positive_sims) / len(positive_sims) if positive_sims else None,
            "contamination_policy": "exact and near CEDR overlap removed by shared RuSentiment/RuSentiTweet loader",
            "construction": "native social-media neutral pairs with cross-emotion negatives",
            "seed": args.seed,
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
