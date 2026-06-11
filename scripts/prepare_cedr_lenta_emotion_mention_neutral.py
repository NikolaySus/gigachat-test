from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_cedr_lenta_news_neutral_distractors import clean_text, clip_sentence
from prepare_cedr_neutral_lexical_distractors import read_jsonl
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

LEXEMES: dict[str, re.Pattern[str]] = {
    "joy": re.compile(
        r"\b(рад\w*|счаст\w*|сме[хя]\w*|улыб\w*|любов\w*|удач\w*|"
        r"весел\w*|забав\w*|прекрасн\w*|отличн\w*)\b",
        re.IGNORECASE,
    ),
    "sadness": re.compile(
        r"\b(груст\w*|печал\w*|тоск\w*|скорб\w*|плач\w*|сл[её]з\w*|"
        r"бед\w*|несчаст\w*|сожален\w*)\b",
        re.IGNORECASE,
    ),
    "surprise": re.compile(
        r"\b(удив\w*|изум\w*|странн\w*|неожидан\w*|недоумен\w*|"
        r"необычн\w*|загадочн\w*)\b",
        re.IGNORECASE,
    ),
    "fear": re.compile(
        r"\b(страх\w*|бо[яи]\w*|опас\w*|угроз\w*|тревог\w*|паник\w*|"
        r"пуглив\w*|напуган\w*)\b",
        re.IGNORECASE,
    ),
    "anger": re.compile(
        r"\b(зл\w*|гнев\w*|ярост\w*|агресс\w*|мст\w*|возмущ\w*|"
        r"раздраж\w*|ненавист\w*)\b",
        re.IGNORECASE,
    ),
}

DIRECT_AUTHOR_RE = re.compile(
    r"\b(я\s+(рад|счастлив|боюсь|злюсь|плачу|ненавижу)|"
    r"мне\s+(страшно|грустно|радостно|обидно)|"
    r"мы\s+(рады|счастливы|боимся|злимся))\b",
    re.IGNORECASE,
)


def groups_for(text: str) -> list[str]:
    return [group for group, pattern in LEXEMES.items() if pattern.search(text)]


def quality_ok(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 45 or len(normalized) > 380:
        return False
    if DIRECT_AUTHOR_RE.search(text):
        return False
    if any(marker in text for marker in (":)", ":(", ")))", "(((", "!!!", "???")):
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 0.55 * max(1, len(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine Lenta neutral rows that mention emotion lexemes.")
    parser.add_argument("--name", default="cedr_lenta_emotion_mention_neutral_3000")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=971)
    parser.add_argument("--max-scan", type=int, default=400000)
    parser.add_argument("--max-chars", type=int, default=320)
    parser.add_argument(
        "--go-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    skipped = Counter()

    stream = load_dataset(
        "data-silence/lenta.ru_2-extended",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR),
    )
    for index, row in enumerate(stream):
        if index >= args.max_scan:
            break
        title = clean_text(row.get("title", ""))
        news = clip_sentence(row.get("news", ""), max_chars=args.max_chars)
        if not title or not news:
            skipped["missing"] += 1
            continue
        text = f"{title}. {news}" if not news.startswith(title) else news
        text = clip_sentence(text, max_chars=args.max_chars)
        norm = normalize_text(text)
        if norm in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(norm)
        groups = groups_for(text)
        if not groups:
            skipped["no_lexeme"] += 1
            continue
        if len(groups) > 2:
            skipped["too_many_groups"] += 1
            continue
        if not quality_ok(text):
            skipped["quality"] += 1
            continue
        if is_contaminated(text, cedr_index):
            skipped["cedr_overlap"] += 1
            continue
        for group in groups:
            pools[group].append(
                {
                    "text": text,
                    "trigger_group": group,
                    "source_index": index,
                    "topic": row.get("topic"),
                    "tags": row.get("tags"),
                    "url": row.get("url"),
                }
            )

    emotion_pools: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(args.go_path):
        group = row.get("metadata", {}).get("group")
        if group in LEXEMES and isinstance(row.get("query"), str):
            emotion_pools[group].append(row["query"])

    usable_groups = [group for group in ["joy", "sadness", "surprise", "fear", "anger"] if pools[group]]
    per_group = args.count // len(usable_groups)
    remainder = args.count % len(usable_groups)
    records = []
    selected_counts = Counter()
    for group_index, group in enumerate(usable_groups):
        pool = pools[group][:]
        rng.shuffle(pool)
        target = min(len(pool), per_group + (1 if group_index < remainder else 0))
        selected = pool[:target]
        selected_counts[group] = len(selected)
        for item in selected:
            positive_pool = [candidate for candidate in selected if candidate["text"] != item["text"]]
            if not positive_pool:
                continue
            negatives = []
            for negative_group in ["joy", "sadness", "surprise", "fear", "anger"]:
                pool_for_negative = emotion_pools.get(negative_group) or []
                if pool_for_negative:
                    negatives.append(rng.choice(pool_for_negative))
            records.append(
                {
                    "source": "data-silence/lenta.ru_2-extended:cedr_emotion_mention_neutral",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positive_pool)["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": "neutral",
                        "trigger_group": group,
                        "source_index": item["source_index"],
                        "topic": item["topic"],
                        "tags": item["tags"],
                        "url": item["url"],
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
            "max_scan": args.max_scan,
            "available_by_trigger_group": {group: len(rows) for group, rows in pools.items()},
            "selected_by_trigger_group": dict(selected_counts),
            "skipped": dict(skipped),
            "go_path": str(args.go_path),
            "construction": "neutral Lenta rows mentioning emotion lexemes; same-trigger neutral positives and GoEmotions-RU emotion negatives",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
