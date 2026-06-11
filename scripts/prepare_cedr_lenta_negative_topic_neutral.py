from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_cedr_lenta_news_neutral_distractors import (
    REPORTING_RE,
    clean_text,
    clip_sentence,
)
from prepare_cedr_neutral_lexical_distractors import FIRST_PERSON_RE, read_jsonl
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

NEGATIVE_TOPICS: dict[str, re.Pattern[str]] = {
    "loss_harm": re.compile(
        r"\b(погиб\w*|умер\w*|смерт\w*|уби[йе]\w*|пострад\w*|жертв\w*|"
        r"травм\w*|ранен\w*|боль\w*|болезн\w*)\b",
        re.IGNORECASE,
    ),
    "conflict_crime": re.compile(
        r"\b(скандал\w*|конфликт\w*|преступ\w*|обвин\w*|суд\w*|"
        r"штраф\w*|наруш\w*|арест\w*|задерж\w*)\b",
        re.IGNORECASE,
    ),
    "risk_problem": re.compile(
        r"\b(риск\w*|опасност\w*|угроз\w*|кризис\w*|проблем\w*|"
        r"негатив\w*|бед\w*|катастроф\w*|авари\w*)\b",
        re.IGNORECASE,
    ),
    "negative_trait": re.compile(
        r"\b(жесток\w*|коварн\w*|мстительн\w*|свиреп\w*|груб\w*|"
        r"тяжел\w*|плох\w*|слаб\w*)\b",
        re.IGNORECASE,
    ),
}

DIRECT_EMOTION_RE = re.compile(
    r"\b(я|мне|меня|мы|нам|нас|боюсь|боюся|радуюсь|рад|грущу|грустно|"
    r"злюсь|ненавижу|люблю|счастлив|печалюсь|расстроен)\b",
    re.IGNORECASE,
)


def topic_groups(text: str) -> list[str]:
    return [group for group, pattern in NEGATIVE_TOPICS.items() if pattern.search(text)]


def quality_ok(text: str, *, require_reporting: bool) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 55 or len(normalized) > 360:
        return False
    if FIRST_PERSON_RE.search(text) or DIRECT_EMOTION_RE.search(text):
        return False
    if require_reporting and not REPORTING_RE.search(text):
        return False
    if any(marker in text for marker in (":)", ":(", ")))", "(((", "!!!", "???")):
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 0.55 * max(1, len(text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine clean Lenta neutral examples about negative topics without expressed emotion."
    )
    parser.add_argument("--count", type=int, default=2400)
    parser.add_argument("--max-scan", type=int, default=500000)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1071)
    parser.add_argument("--name", default="cedr_lenta_negative_topic_neutral_2400")
    parser.add_argument("--require-reporting", action="store_true")
    parser.add_argument(
        "--go-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    skipped = Counter()

    stream = load_dataset(
        "data-silence/lenta.ru_2-extended",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR),
    )
    scanned = 0
    for index, row in enumerate(stream):
        if index >= args.max_scan:
            break
        scanned += 1
        title = clean_text(row.get("title", ""))
        news = clip_sentence(row.get("news", ""), max_chars=args.max_chars)
        if not title or not news:
            skipped["missing"] += 1
            continue
        text = f"{title}. {news}" if not news.startswith(title) else news
        text = clip_sentence(text, max_chars=args.max_chars)
        normalized = normalize_text(text)
        if normalized in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(normalized)
        groups = topic_groups(text)
        if not groups:
            skipped["no_topic"] += 1
            continue
        if len(groups) > 2:
            skipped["too_many_topics"] += 1
            continue
        if not quality_ok(text, require_reporting=args.require_reporting):
            skipped["quality"] += 1
            continue
        if is_contaminated(text, cedr_index):
            skipped["cedr_overlap"] += 1
            continue
        for group in groups:
            pools[group].append(
                {
                    "text": text,
                    "topic_group": group,
                    "source_index": index,
                    "topic": row.get("topic"),
                    "tags": row.get("tags"),
                    "url": row.get("url"),
                }
            )

    go_rows = read_jsonl(args.go_path)
    emotion_pools: dict[str, list[str]] = defaultdict(list)
    for row in go_rows:
        group = row.get("metadata", {}).get("group")
        if group in GROUPS and group != "neutral":
            emotion_pools[group].append(row["query"])

    groups = [group for group in ["loss_harm", "conflict_crime", "risk_problem", "negative_trait"] if pools.get(group)]
    if not groups:
        raise RuntimeError("No usable negative-topic neutral pools found")
    per_group = args.count // len(groups)
    remainder = args.count % len(groups)
    selected: dict[str, list[dict[str, Any]]] = {}
    for group_index, group in enumerate(groups):
        target = per_group + (1 if group_index < remainder else 0)
        rows = pools[group][:]
        rng.shuffle(rows)
        selected[group] = rows[: min(target, len(rows))]

    records = []
    hard_negative_order = ["sadness", "anger", "fear", "surprise", "joy"]
    for group, items in selected.items():
        for item in items:
            positives = [candidate for candidate in items if candidate["text"] != item["text"]]
            if not positives:
                continue
            negatives = []
            for emotion_group in hard_negative_order:
                pool = emotion_pools.get(emotion_group) or []
                if pool:
                    negatives.append(rng.choice(pool))
            records.append(
                {
                    "source": "data-silence/lenta.ru_2-extended:cedr_negative_topic_neutral",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positives)["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": "neutral",
                        "topic_group": group,
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
            "scanned_rows": scanned,
            "require_reporting": args.require_reporting,
            "available_by_topic_group": {group: len(rows) for group, rows in pools.items()},
            "selected_by_topic_group": {
                group: sum(1 for row in records if row["metadata"]["topic_group"] == group)
                for group in groups
            },
            "skipped": dict(skipped),
            "go_path": str(args.go_path),
            "construction": "neutral Lenta news about negative topics; same-topic neutral positives and emotion hard negatives",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
