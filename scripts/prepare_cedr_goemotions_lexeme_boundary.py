from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_cedr_goemotions_ru_hardneg_component import select_rows, strip_for_tfidf
from prepare_cedr_neutral_lexical_distractors import LEXEMES
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
EMOTION_GROUPS = [group for group in GROUPS if group != "neutral"]
FIRST_PERSON_RE = re.compile(
    r"(^|[\s,;:])("
    r"я|мне|меня|мной|мой|моя|мои|моё|мы|нам|нас|наш|наша|наши|"
    r"хочу|люблю|ненавижу|боюсь|радуюсь|грущу|злюсь"
    r")($|[\s,;:.!?])",
    re.IGNORECASE,
)


def detect_triggers(text: str) -> list[str]:
    return [group for group, pattern in LEXEMES.items() if pattern.search(text)]


def is_neutral_distractor_quality(text: str, *, allow_first_person: bool) -> bool:
    if not allow_first_person and FIRST_PERSON_RE.search(text):
        return False
    if len(text) < 35 or len(text) > 260:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 0.50 * max(1, len(text))


def build_matrix(items: list[dict[str, Any]]):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=250_000)
    matrix = normalize(vectorizer.fit_transform([strip_for_tfidf(item["text"]) for item in items]), copy=False)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CEDR neutral-boundary rows from GoEmotions-RU neutral examples containing emotion lexemes."
    )
    parser.add_argument("--base-count", type=int, default=24000)
    parser.add_argument("--base-neutral-fraction", type=float, default=0.65)
    parser.add_argument("--count", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=887)
    parser.add_argument("--allow-first-person", action="store_true")
    parser.add_argument("--name", default="cedr_goemotions_lexeme_boundary_2400")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    selected_by_group, base_summary = select_rows(args.base_count, args.base_neutral_fraction, args.seed)
    all_items = [item for group in GROUPS for item in selected_by_group[group]]
    all_matrix = build_matrix(all_items)
    item_pos = {id(item): pos for pos, item in enumerate(all_items)}

    neutral_by_trigger: dict[str, list[dict[str, Any]]] = defaultdict(list)
    emotion_by_trigger: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for item in all_items:
        triggers = detect_triggers(item["text"])
        if len(triggers) != 1:
            skipped["not_exactly_one_trigger"] += 1
            continue
        trigger = triggers[0]
        if item["group"] == "neutral":
            if not is_neutral_distractor_quality(item["text"], allow_first_person=args.allow_first_person):
                skipped["neutral_quality"] += 1
                continue
            neutral_by_trigger[trigger].append(item)
        elif item["group"] == trigger:
            emotion_by_trigger[trigger].append(item)
        else:
            skipped["emotion_group_trigger_mismatch"] += 1

    available_triggers = [
        trigger
        for trigger in EMOTION_GROUPS
        if len(neutral_by_trigger[trigger]) >= 2 and emotion_by_trigger[trigger]
    ]
    if not available_triggers:
        raise RuntimeError("No usable trigger groups found")

    per_trigger = args.count // len(available_triggers)
    remainder = args.count % len(available_triggers)
    records: list[dict[str, Any]] = []
    hard_scores: list[float] = []
    for trigger_index, trigger in enumerate(available_triggers):
        target = per_trigger + (1 if trigger_index < remainder else 0)
        neutral_pool = neutral_by_trigger[trigger][:]
        rng.shuffle(neutral_pool)
        selected_neutral = neutral_pool[: min(target, len(neutral_pool))]
        emotion_pool = emotion_by_trigger[trigger]
        emotion_indices = np.array([item_pos[id(item)] for item in emotion_pool])

        for item in selected_neutral:
            positive_pool = [candidate for candidate in neutral_pool if candidate is not item]
            positive = rng.choice(positive_pool)
            item_index = item_pos[id(item)]
            sims = all_matrix[item_index].dot(all_matrix[emotion_indices].T).toarray().ravel()
            order = np.argsort(-sims)
            negatives = []
            for local_index in order[:3]:
                hard_scores.append(float(sims[int(local_index)]))
                negatives.append(CEDR_PREFIX + emotion_pool[int(local_index)]["text"])
            for other_trigger in EMOTION_GROUPS:
                if other_trigger == trigger or not emotion_by_trigger[other_trigger]:
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(emotion_by_trigger[other_trigger])["text"])
            records.append(
                {
                    "source": "AiLab-IMCS-UL/go_emotions-ru:cedr_lexeme_boundary",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": "neutral",
                        "trigger_group": trigger,
                        "split": item["split"],
                        "index": item["index"],
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
            "base_count": args.base_count,
            "base_neutral_fraction": args.base_neutral_fraction,
            "available_neutral_by_trigger": {key: len(value) for key, value in neutral_by_trigger.items()},
            "available_emotion_by_trigger": {key: len(value) for key, value in emotion_by_trigger.items()},
            "selected_by_trigger": dict(Counter(row["metadata"]["trigger_group"] for row in records)),
            "hard_negative_tfidf": {
                "mean_similarity": sum(hard_scores) / max(1, len(hard_scores)),
                "p95_similarity": float(np.percentile(hard_scores, 95)) if hard_scores else 0.0,
            },
            "skipped": dict(skipped),
            "base_summary": base_summary,
            "construction": "neutral examples with exactly one emotion lexeme; same-trigger neutral positives; same-trigger emotion hard negatives",
            "contamination_policy": "inherits exact and near CEDR overlap filtering from select_rows; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
