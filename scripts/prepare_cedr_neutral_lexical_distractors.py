from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]

LEXEMES: dict[str, re.Pattern[str]] = {
    "joy": re.compile(r"\b(радост\w*|счаст\w*|весел\w*|улыб\w*)", re.IGNORECASE),
    "sadness": re.compile(r"\b(груст\w*|печал\w*|тоск\w*|скорб\w*)", re.IGNORECASE),
    "surprise": re.compile(r"\b(удив\w*|изум\w*|странн\w*|неожидан\w*)", re.IGNORECASE),
    "fear": re.compile(r"\b(страх\w*|опасен\w*|опасн\w*|бо[яи]\w*|угроз\w*)", re.IGNORECASE),
    "anger": re.compile(r"\b(злост\w*|зл[аоы]\w*|гнев\w*|ярост\w*|агресс\w*)", re.IGNORECASE),
}

INSTRUCTION_RE = re.compile(r"^Instruct:.*?\nQuery:\s*", re.DOTALL)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
FIRST_PERSON_RE = re.compile(
    r"(^|[\s,;:])("
    r"я|мне|меня|мной|мой|моя|мои|моё|мы|нам|нас|наш|наша|наши|"
    r"хочу|люблю|ненавижу|боюсь|радуюсь|грущу|злюсь"
    r")($|[\s,;:.!?])",
    re.IGNORECASE,
)


DEFAULT_SOURCES = [
    DATA_DIR / "open_ru_1r_nc_deepvk_ru_hnp.jsonl",
    DATA_DIR / "open_ru_1r_nc_habr_qa_sbs_harder_sim019_len.jsonl",
    DATA_DIR / "open_ru_1r_nc_mixh_habrfull_geracl6400_habr4369_deepvk3200_grandmaster3200_17169.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = INSTRUCTION_RE.sub("", text)
    text = text.replace(CEDR_PREFIX, "")
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fields_from_record(record: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in ("query", "positive", "sentence1", "sentence2"):
        value = record.get(key)
        if isinstance(value, str):
            texts.append(value)
    for value in record.get("negatives", []):
        if isinstance(value, str):
            texts.append(value)
    return texts


def detect_groups(text: str) -> list[str]:
    return [group for group, pattern in LEXEMES.items() if pattern.search(text)]


def quality_ok(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 40 or len(normalized) > 360:
        return False
    if FIRST_PERSON_RE.search(text):
        return False
    if any(marker in text for marker in (":)", ":(", ")))", "(((", "!!!", "???")):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.55 * max(1, len(text)):
        return False
    return True


def collect_candidates(paths: list[Path], *, cedr_index: Any) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    skipped = Counter()
    for path in paths:
        if not path.exists():
            skipped[f"missing:{path.name}"] += 1
            continue
        for record_index, record in enumerate(read_jsonl(path)):
            source = str(record.get("source") or path.stem)
            for text_index, raw_text in enumerate(fields_from_record(record)):
                text = clean_text(raw_text)
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
                            "source": source,
                            "path": str(path),
                            "record_index": record_index,
                            "text_index": text_index,
                        }
                    )
    pools["_skipped"] = [{"counter": dict(skipped)}]  # type: ignore[assignment]
    return pools


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine neutral CEDR lexical distractors from clean non-CEDR data.")
    parser.add_argument("--source", type=Path, action="append", default=None)
    parser.add_argument("--go-path", type=Path, default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl")
    parser.add_argument("--count", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=871)
    parser.add_argument("--name", default="cedr_neutral_lexical_distractors_3200")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    paths = args.source or DEFAULT_SOURCES
    pools = collect_candidates(paths, cedr_index=cedr_index)
    skipped = pools.pop("_skipped")[0]["counter"] if "_skipped" in pools else {}

    go_rows = read_jsonl(args.go_path)
    emotion_pools: dict[str, list[str]] = defaultdict(list)
    for row in go_rows:
        group = row.get("metadata", {}).get("group")
        if group in GROUPS and group != "neutral":
            emotion_pools[group].append(row["query"])

    groups = [group for group in ["fear", "surprise", "joy", "sadness", "anger"] if pools.get(group)]
    per_group = args.count // len(groups)
    remainder = args.count % len(groups)
    selected: dict[str, list[dict[str, Any]]] = {}
    for index, group in enumerate(groups):
        target = per_group + (1 if index < remainder else 0)
        pool = pools[group][:]
        rng.shuffle(pool)
        selected[group] = pool[: min(target, len(pool))]

    records = []
    for group, items in selected.items():
        for item in items:
            positive_pool = [candidate for candidate in items if candidate["text"] != item["text"]]
            if len(positive_pool) < 1:
                continue
            positive = rng.choice(positive_pool)
            negatives = []
            for negative_group in ["joy", "sadness", "surprise", "fear", "anger"]:
                if emotion_pools[negative_group]:
                    negatives.append(rng.choice(emotion_pools[negative_group]))
            records.append(
                {
                    "source": f"neutral_lexical_distractor:{item['source']}",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": "neutral",
                        "trigger_group": group,
                        "path": item["path"],
                        "record_index": item["record_index"],
                        "text_index": item["text_index"],
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
            "sources": [str(path) for path in paths],
            "available_by_trigger_group": {group: len(rows) for group, rows in pools.items()},
            "selected_by_trigger_group": {
                group: sum(1 for row in records if row["metadata"]["trigger_group"] == group)
                for group in groups
            },
            "skipped": skipped,
            "go_path": str(args.go_path),
            "construction": "neutral query/neutral positive with emotion lexeme; GoEmotions-RU emotion negatives",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
