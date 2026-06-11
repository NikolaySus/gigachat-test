from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_cedr_lenta_news_neutral_distractors import clip_sentence
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
URL_RE = re.compile(r"https?://\S+|www\.\S+|\[Картинка:[^\]]+\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
BLOG_META_RE = re.compile(
    r"\b(слово|пост|запис|сайт|проект|конкурс|истори|новост|фото|картинк|"
    r"пример|текст|назван|называет|термин|рубри|обсужд|комментари|ссылк)\w*\b",
    re.IGNORECASE,
)


def clean_text(value: Any) -> str:
    text = URL_RE.sub(" ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_groups(text: str) -> list[str]:
    return [group for group, pattern in LEXEMES.items() if pattern.search(text)]


def split_candidates(title: str, text: str) -> list[str]:
    candidates = []
    title = clean_text(title)
    if title:
        candidates.append(title)
    for sentence in SENTENCE_SPLIT_RE.split(clean_text(text)):
        sentence = clip_sentence(sentence, max_chars=280)
        if sentence:
            candidates.append(sentence)
    return candidates


def quality_ok(text: str, *, require_meta: bool, allow_first_person: bool) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 35 or len(normalized) > 280:
        return False
    if require_meta and not BLOG_META_RE.search(text):
        return False
    if not allow_first_person and FIRST_PERSON_RE.search(text):
        return False
    if any(marker in text for marker in (":)", ":(", ")))", "(((", "!!!", "???")):
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 0.50 * max(1, len(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine clean blog-style neutral CEDR distractors.")
    parser.add_argument("--dataset", default="rustemgareev/artemy-lebedev")
    parser.add_argument("--count", type=int, default=1600)
    parser.add_argument("--max-scan", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=1051)
    parser.add_argument("--name", default="cedr_blog_neutral_meta_1600")
    parser.add_argument("--require-meta", action="store_true")
    parser.add_argument("--allow-first-person", action="store_true")
    parser.add_argument(
        "--go-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    ds = load_dataset(args.dataset, split="train", streaming=True, cache_dir=str(CACHE_DIR))
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    skipped = Counter()
    scanned = 0
    for row_index, row in enumerate(ds):
        if row_index >= args.max_scan:
            break
        scanned += 1
        for text_index, text in enumerate(split_candidates(row.get("title", ""), row.get("text", ""))):
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
            if not quality_ok(text, require_meta=args.require_meta, allow_first_person=args.allow_first_person):
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
                        "row_index": row_index,
                        "text_index": text_index,
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
        raise RuntimeError("No usable blog neutral pools found")
    per_group = args.count // len(groups)
    remainder = args.count % len(groups)
    selected: dict[str, list[dict[str, Any]]] = {}
    for index, group in enumerate(groups):
        target = per_group + (1 if index < remainder else 0)
        rows = pools[group][:]
        rng.shuffle(rows)
        selected[group] = rows[: min(target, len(rows))]

    records = []
    for group, items in selected.items():
        for item in items:
            positives = [candidate for candidate in items if candidate["text"] != item["text"]]
            if not positives:
                continue
            negatives = []
            for negative_group in [group, *[g for g in ["joy", "sadness", "surprise", "fear", "anger"] if g != group]]:
                pool = emotion_pools.get(negative_group) or []
                if pool:
                    negatives.append(rng.choice(pool))
            records.append(
                {
                    "source": f"{args.dataset}:cedr_blog_neutral",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positives)["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": "neutral",
                        "trigger_group": group,
                        "row_index": item["row_index"],
                        "text_index": item["text_index"],
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
            "dataset": args.dataset,
            "records": len(records),
            "requested": args.count,
            "scanned_rows": scanned,
            "require_meta": args.require_meta,
            "allow_first_person": args.allow_first_person,
            "available_by_trigger_group": {group: len(rows) for group, rows in pools.items()},
            "selected_by_trigger_group": {
                group: sum(1 for row in records if row["metadata"]["trigger_group"] == group)
                for group in groups
            },
            "skipped": dict(skipped),
            "go_path": str(args.go_path),
            "construction": "blog-style neutral texts with emotion lexemes; same-trigger neutral positives and GoEmotions-RU emotion negatives",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
