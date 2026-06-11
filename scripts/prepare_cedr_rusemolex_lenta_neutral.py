from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_cedr_lenta_news_neutral_distractors import REPORTING_RE, clean_text, clip_sentence
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
DEFAULT_LEXICON = ROOT / "results" / "external" / "rusemolex" / "RusEmoLex.csv"
CLASS_MAP = {
    "Радость": "joy",
    "Грусть": "sadness",
    "Удивление": "surprise",
    "Страх": "fear",
    "Злость": "anger",
}
COMMON_ENDINGS = (
    "ыми",
    "ими",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ами",
    "ями",
    "ать",
    "ять",
    "ить",
    "еть",
    "ешь",
    "ишь",
    "ого",
    "его",
    "ая",
    "яя",
    "ое",
    "ее",
    "ый",
    "ий",
    "ой",
    "ые",
    "ие",
    "ам",
    "ям",
    "ах",
    "ях",
    "ом",
    "ем",
    "у",
    "а",
    "я",
    "ы",
    "и",
    "е",
)


def lemma_stem(word: str) -> str:
    word = word.replace("ё", "е").lower().strip()
    if " " in word:
        return word
    for ending in COMMON_ENDINGS:
        if len(word) - len(ending) >= 5 and word.endswith(ending):
            return word[: -len(ending)]
    return word


def load_lexicon(path: Path, *, min_sources: int, categories: set[str]) -> dict[str, list[str]]:
    by_group: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8-sig") as file:
        for row in csv.DictReader(file, delimiter=";"):
            group = CLASS_MAP.get(row["Класс"])
            if group is None:
                continue
            if row["Категория источников"] not in categories:
                continue
            if int(row["Количество вхождений"]) < min_sources:
                continue
            word = row["Слово"].strip().lower().replace("ё", "е")
            if len(word) < 4:
                continue
            by_group[group].add(word)
            stem = lemma_stem(word)
            if len(stem) >= 5:
                by_group[group].add(stem)
    return {group: sorted(words, key=len, reverse=True) for group, words in by_group.items()}


def compile_lexicon(lexicon: dict[str, list[str]], *, allow_suffix: bool) -> dict[str, re.Pattern[str]]:
    compiled = {}
    for group, words in lexicon.items():
        alternatives = []
        for word in words:
            escaped = re.escape(word)
            if " " in word:
                alternatives.append(rf"{escaped}")
            elif allow_suffix:
                alternatives.append(rf"{escaped}[а-яё]*")
            else:
                alternatives.append(rf"{escaped}")
        compiled[group] = re.compile(rf"(?<![а-яa-z])({'|'.join(alternatives)})(?![а-яa-z])", re.IGNORECASE)
    return compiled


def matched_groups(text: str, patterns: dict[str, re.Pattern[str]]) -> dict[str, list[str]]:
    normalized = normalize_text(text).replace("ё", "е")
    matches: dict[str, list[str]] = {}
    for group, pattern in patterns.items():
        found = []
        for match in pattern.finditer(normalized):
            found.append(match.group(1))
            if len(found) >= 3:
                break
        if found:
            matches[group] = found
    return matches


def quality_ok(text: str, *, require_reporting: bool) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 55 or len(normalized) > 430:
        return False
    if FIRST_PERSON_RE.search(text):
        return False
    if any(marker in text for marker in (":)", ":(", ")))", "(((", "😂", "🤣")):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.58 * max(1, len(text)):
        return False
    if require_reporting and not REPORTING_RE.search(text):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine neutral CEDR boundary records from Lenta using RusEmoLex.")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--name", default="cedr_rusemolex_lenta_neutral_reported_5000")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--max-scan", type=int, default=500000)
    parser.add_argument("--min-scan", type=int, default=50000)
    parser.add_argument("--max-chars", type=int, default=300)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--categories", default="A,AB")
    parser.add_argument("--use-stems", action="store_true")
    parser.add_argument("--require-reporting", action="store_true")
    parser.add_argument("--seed", type=int, default=1241)
    parser.add_argument(
        "--go-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    categories = {item.strip() for item in args.categories.split(",") if item.strip()}
    lexicon = load_lexicon(args.lexicon, min_sources=args.min_sources, categories=categories)
    if not args.use_stems:
        exact_lexicon: dict[str, list[str]] = defaultdict(list)
        with args.lexicon.open(encoding="utf-8-sig") as file:
            for row in csv.DictReader(file, delimiter=";"):
                group = CLASS_MAP.get(row["Класс"])
                if group is None:
                    continue
                if row["Категория источников"] not in categories:
                    continue
                if int(row["Количество вхождений"]) < args.min_sources:
                    continue
                word = row["Слово"].strip().lower().replace("ё", "е")
                if len(word) >= 4:
                    exact_lexicon[group].append(word)
        lexicon = {group: sorted(set(words), key=len, reverse=True) for group, words in exact_lexicon.items()}
    patterns = compile_lexicon(lexicon, allow_suffix=args.use_stems)
    cedr_index = load_cedr_index()
    skipped = Counter()
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    target_per_group = max(1, args.count // 5)

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
        if not quality_ok(text, require_reporting=args.require_reporting):
            skipped["quality"] += 1
            continue
        matches = matched_groups(text, patterns)
        if not matches:
            skipped["no_lexicon_match"] += 1
            continue
        if len(matches) != 1:
            skipped["multiple_emotion_classes"] += 1
            continue
        group = next(iter(matches))
        if is_contaminated(text, cedr_index):
            skipped["cedr_overlap"] += 1
            continue
        pools[group].append(
            {
                "text": text,
                "trigger_group": group,
                "matched_words": matches[group],
                "source_index": index,
                "topic": row.get("topic"),
                "tags": row.get("tags"),
                "url": row.get("url"),
            }
        )
        if index >= args.min_scan and all(len(pools[g]) >= target_per_group for g in ["joy", "sadness", "surprise", "fear", "anger"]):
            skipped["early_stop_after_index"] = index
            break

    go_rows = read_jsonl(args.go_path)
    emotion_pools: dict[str, list[str]] = defaultdict(list)
    for row in go_rows:
        group = row.get("metadata", {}).get("group")
        if group in GROUPS and group != "neutral":
            emotion_pools[group].append(row["query"])

    active_groups = [group for group in ["joy", "sadness", "surprise", "fear", "anger"] if pools[group]]
    if not active_groups:
        raise RuntimeError("No usable RusEmoLex Lenta rows found")
    per_group = args.count // len(active_groups)
    remainder = args.count % len(active_groups)
    selected: dict[str, list[dict[str, Any]]] = {}
    for group_index, group in enumerate(active_groups):
        rows = pools[group][:]
        rng.shuffle(rows)
        target = per_group + (1 if group_index < remainder else 0)
        selected[group] = rows[: min(target, len(rows))]

    records = []
    for group, items in selected.items():
        for item in items:
            positives = [candidate for candidate in items if candidate["text"] != item["text"]]
            if not positives:
                continue
            negatives = []
            for negative_group in [group] + [g for g in ["joy", "sadness", "surprise", "fear", "anger"] if g != group]:
                pool = emotion_pools.get(negative_group) or []
                if pool:
                    negatives.append(rng.choice(pool))
            records.append(
                {
                    "source": "data-silence/lenta.ru_2-extended:rusemolex_neutral_boundary",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positives)["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": "neutral",
                        "trigger_group": group,
                        "matched_words": item["matched_words"],
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
            "lexicon": str(args.lexicon.relative_to(ROOT) if args.lexicon.is_relative_to(ROOT) else args.lexicon),
            "lexicon_groups": {group: len(words) for group, words in lexicon.items()},
            "categories": sorted(categories),
            "min_sources": args.min_sources,
            "use_stems": args.use_stems,
            "max_scan": args.max_scan,
            "min_scan": args.min_scan,
            "max_chars": args.max_chars,
            "require_reporting": args.require_reporting,
            "available_by_trigger_group": {group: len(rows) for group, rows in pools.items()},
            "selected_by_trigger_group": dict(Counter(row["metadata"]["trigger_group"] for row in records)),
            "skipped": dict(skipped),
            "go_path": str(args.go_path.relative_to(ROOT) if args.go_path.is_relative_to(ROOT) else args.go_path),
            "construction": "neutral Lenta news with exactly one RusEmoLex emotion class; same-trigger GoEmotions-RU hard negatives",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
