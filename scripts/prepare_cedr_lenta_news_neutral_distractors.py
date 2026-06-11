from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_cedr_neutral_lexical_distractors import LEXEMES, FIRST_PERSON_RE, read_jsonl
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
URL_RE = re.compile(r"https?://\S+|www\.\S+")
REPORTING_RE = re.compile(
    r"\b(сообщил\w*|заявил\w*|рассказал\w*|отметил\w*|указал\w*|"
    r"по словам|по данным|об этом|пишет|передает|прокомментировал\w*|"
    r"объявил\w*|предупредил\w*|признал\w*|обвинил\w*)\b",
    re.IGNORECASE,
)


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clip_sentence(text: str, *, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in [". ", "! ", "? ", "; "]:
        pos = cut.rfind(sep)
        if pos > 80:
            return cut[: pos + 1].strip()
    return cut.rsplit(" ", 1)[0].strip()


def detect_groups(text: str) -> list[str]:
    return [group for group, pattern in LEXEMES.items() if pattern.search(text)]


def quality_ok(text: str, *, require_reporting: bool) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 45 or len(normalized) > 420:
        return False
    if FIRST_PERSON_RE.search(text):
        return False
    if any(marker in text for marker in (":)", ":(", ")))", "(((")):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.55 * max(1, len(text)):
        return False
    if require_reporting and not REPORTING_RE.search(text):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine neutral CEDR distractors from streamed Lenta news.")
    parser.add_argument("--count", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--name", default="cedr_lenta_news_neutral_distractors_3200")
    parser.add_argument("--max-scan", type=int, default=120000)
    parser.add_argument("--max-chars", type=int, default=280)
    parser.add_argument("--require-reporting", action="store_true")
    parser.add_argument(
        "--go-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    skipped = Counter()
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()

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
        normalized = normalize_text(text)
        if normalized in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(normalized)
        groups = detect_groups(text)
        if not groups:
            skipped["no_lexeme"] += 1
            continue
        if len(groups) > 2:
            skipped["too_many_lexeme_groups"] += 1
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
                    "trigger_group": group,
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

    groups = [group for group in ["fear", "surprise", "joy", "sadness", "anger"] if pools.get(group)]
    if not groups:
        raise RuntimeError("No usable news pools found")
    per_group = args.count // len(groups)
    remainder = args.count % len(groups)
    selected: dict[str, list[dict[str, Any]]] = {}
    for group_index, group in enumerate(groups):
        target = per_group + (1 if group_index < remainder else 0)
        rows = pools[group][:]
        rng.shuffle(rows)
        selected[group] = rows[: min(target, len(rows))]

    records = []
    for group, items in selected.items():
        for item in items:
            positives = [candidate for candidate in items if candidate["text"] != item["text"]]
            if not positives:
                continue
            negative_groups = [group] + [g for g in ["joy", "sadness", "surprise", "fear", "anger"] if g != group]
            negatives = []
            for negative_group in negative_groups:
                pool = emotion_pools.get(negative_group) or []
                if pool:
                    negatives.append(rng.choice(pool))
            records.append(
                {
                    "source": "data-silence/lenta.ru_2-extended:cedr_news_neutral",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positives)["text"],
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
            "source": "data-silence/lenta.ru_2-extended",
            "max_scan": args.max_scan,
            "max_chars": args.max_chars,
            "require_reporting": args.require_reporting,
            "available_by_trigger_group": {group: len(rows) for group, rows in pools.items()},
            "selected_by_trigger_group": {
                group: sum(1 for row in records if row["metadata"]["trigger_group"] == group)
                for group in groups
            },
            "skipped": dict(skipped),
            "go_path": str(args.go_path),
            "construction": "neutral Lenta news with emotion lexemes; same-trigger GoEmotions-RU hard negatives",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
