from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_cedr_lenta_news_neutral_distractors import (
    REPORTING_RE,
    clean_text,
    clip_sentence,
    quality_ok,
)
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
SURPRISE_RE = re.compile(
    r"\b("
    r"удив\w*|изум\w*|странн\w*|неожидан\w*|необычн\w*|"
    r"внезапн\w*|натолкнул\w*|наткнул\w*|загадочн\w*"
    r")\b",
    re.IGNORECASE,
)
OTHER_EMOTION_RE = re.compile(
    r"\b("
    r"радост\w*|счаст\w*|весел\w*|улыб\w*|"
    r"груст\w*|печал\w*|тоск\w*|скорб\w*|"
    r"страх\w*|опасен\w*|опасн\w*|бо[яи]\w*|угроз\w*|"
    r"злост\w*|зл[аоы]\w*|гнев\w*|ярост\w*|агресс\w*"
    r")\b",
    re.IGNORECASE,
)


def good_surprise_text(text: str, *, require_reporting: bool) -> bool:
    if not SURPRISE_RE.search(text):
        return False
    if OTHER_EMOTION_RE.search(text):
        return False
    if require_reporting and not REPORTING_RE.search(text):
        return False
    return quality_ok(text, require_reporting=False)


def load_neutral_negatives(path: Path, *, limit: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    rows = read_jsonl(path)
    candidates = [
        str(row.get("query", ""))
        for row in rows
        if row.get("metadata", {}).get("group") == "neutral" and isinstance(row.get("query"), str)
    ]
    rng.shuffle(candidates)
    return candidates[:limit]


def load_emotion_negatives(path: Path, *, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    pools: dict[str, list[str]] = {"joy": [], "sadness": [], "fear": [], "anger": [], "neutral": []}
    for row in read_jsonl(path):
        group = row.get("metadata", {}).get("group")
        if group in pools and isinstance(row.get("query"), str):
            pools[group].append(row["query"])
    for pool in pools.values():
        rng.shuffle(pool)
    return pools


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine clean factual-surprise CEDR-style examples from Lenta news."
    )
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=918)
    parser.add_argument("--name", default="cedr_lenta_factual_surprise_1200")
    parser.add_argument("--max-scan", type=int, default=250000)
    parser.add_argument("--max-chars", type=int, default=280)
    parser.add_argument("--require-reporting", action="store_true")
    parser.add_argument(
        "--neutral-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_allavail.jsonl",
    )
    parser.add_argument(
        "--go-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    skipped = Counter()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    stream = load_dataset(
        "data-silence/lenta.ru_2-extended",
        split="train",
        streaming=True,
        cache_dir=str(CACHE_DIR),
    )
    for index, row in enumerate(stream):
        if index >= args.max_scan or len(selected) >= args.count:
            break
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
        if not good_surprise_text(text, require_reporting=args.require_reporting):
            skipped["not_factual_surprise"] += 1
            continue
        if is_contaminated(text, cedr_index):
            skipped["cedr_overlap"] += 1
            continue
        selected.append(
            {
                "text": text,
                "source_index": index,
                "topic": row.get("topic"),
                "tags": row.get("tags"),
                "url": row.get("url"),
            }
        )

    if len(selected) < 2:
        raise RuntimeError(f"Need at least two surprise rows, got {len(selected)}")

    neutral_negatives = load_neutral_negatives(args.neutral_path, limit=max(1000, len(selected)), seed=args.seed + 1)
    emotion_negatives = load_emotion_negatives(args.go_path, seed=args.seed + 2)
    if not neutral_negatives:
        raise RuntimeError(f"No neutral negatives loaded from {args.neutral_path}")

    records = []
    for item in selected:
        positive_pool = [candidate for candidate in selected if candidate["text"] != item["text"]]
        negatives = [rng.choice(neutral_negatives)]
        for group in ["joy", "sadness", "fear", "anger", "neutral"]:
            pool = emotion_negatives.get(group) or []
            if pool:
                negatives.append(rng.choice(pool))
        records.append(
            {
                "source": "data-silence/lenta.ru_2-extended:cedr_factual_surprise",
                "objective": "contrastive",
                "query": CEDR_PREFIX + item["text"],
                "positive": CEDR_PREFIX + rng.choice(positive_pool)["text"],
                "negatives": negatives,
                "metadata": {
                    "group": "surprise",
                    "source_index": item["source_index"],
                    "topic": item["topic"],
                    "tags": item["tags"],
                    "url": item["url"],
                    "construction": "factual news with surprise lexeme, no other emotion lexeme",
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
            "source": "data-silence/lenta.ru_2-extended",
            "max_scan": args.max_scan,
            "max_chars": args.max_chars,
            "require_reporting": args.require_reporting,
            "skipped": dict(skipped),
            "neutral_path": str(args.neutral_path),
            "go_path": str(args.go_path),
            "surprise_pattern": SURPRISE_RE.pattern,
            "construction": "factual surprise Lenta rows; same-source factual surprise positives; neutral and non-surprise emotion negatives",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
