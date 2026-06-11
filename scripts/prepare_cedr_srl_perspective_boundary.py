from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
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
FIRST_PERSON_RE = re.compile(
    r"(^|[\s,;:()«»\"'—-])("
    r"я|мне|меня|мной|мы|нам|нас|нами|"
    r"боюсь|боялся|боялась|опасаюсь|люблю|радуюсь|ужасаюсь"
    r")($|[\s,;:.!?()«»\"'—-])",
    re.IGNORECASE,
)
SECOND_THIRD_PERSON_RE = re.compile(
    r"(^|[\s,;:()«»\"'—-])("
    r"он|она|они|ему|ей|им|людей|многих|зрителей|пользователей|"
    r"вам|вас|вы|детей|стариков|женщин|мужчин"
    r")($|[\s,;:.!?()«»\"'—-])",
    re.IGNORECASE,
)

GROUP_MAP = {
    "любить": "joy",
    "бояться": "fear",
    "опасаться": "fear",
    "пугать": "fear",
    "страшить": "fear",
    "ужасать": "fear",
}


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def argument_texts(row: dict[str, Any], role: str) -> list[str]:
    values = []
    for item in row.get("arguments") or []:
        if str(item.get("role", "")).lower() == role.lower():
            values.append(str(item.get("argument", "")))
    return values


def has_first_person_experiencer(row: dict[str, Any], text: str) -> bool:
    experiencers = " ".join(argument_texts(row, "Experiencer"))
    if experiencers.strip():
        return bool(FIRST_PERSON_RE.search(experiencers))
    return bool(FIRST_PERSON_RE.search(text))


def is_reported_or_nonself(row: dict[str, Any], text: str) -> bool:
    experiencers = " ".join(argument_texts(row, "Experiencer"))
    if has_first_person_experiencer(row, text):
        return False
    if experiencers.strip():
        return True
    return bool(SECOND_THIRD_PERSON_RE.search(text))


def sample_with_replacement(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not rows:
        return []
    return [rng.choice(rows) for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CEDR perspective-boundary records from clean SRL emotion-predicate data."
    )
    parser.add_argument("--name", default="cedr_srl_perspective_boundary_1800")
    parser.add_argument("--count", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=1261)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    dataset = load_dataset("dl-ru/srl-emotion-predicates", cache_dir=str(CACHE_DIR))["train"]

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()
    for index, row in enumerate(dataset):
        text = clean_text(row["text"])
        normalized = normalize_text(text)
        if len(normalized) < 12 or len(normalized) > 260:
            skipped["length"] += 1
            continue
        if normalized in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(normalized)
        if is_contaminated(text, cedr_index):
            skipped["cedr_overlap"] += 1
            continue
        base_group = GROUP_MAP.get(str(row["p_group"]).strip().lower())
        if not base_group:
            skipped["unmapped_group"] += 1
            continue
        if has_first_person_experiencer(row, text):
            label = base_group
        elif is_reported_or_nonself(row, text):
            label = "neutral"
        else:
            skipped["unclear_perspective"] += 1
            continue
        pools[label].append(
            {
                "text": text,
                "label": label,
                "predicate_group": str(row["p_group"]),
                "predicate_word": str(row["p_word"]),
                "index": index,
            }
        )

    labels = sorted(pools)
    if "neutral" not in pools or len(labels) < 2:
        raise RuntimeError(f"Not enough usable label pools: { {k: len(v) for k, v in pools.items()} }")

    records = []
    selected = sample_with_replacement([item for label in labels for item in pools[label]], args.count, rng)
    for item in selected:
        label = item["label"]
        same_pool = [candidate for candidate in pools[label] if candidate["text"] != item["text"]]
        if not same_pool:
            same_pool = pools[label]
        positive = rng.choice(same_pool)
        negatives = []
        for other_label in labels:
            if other_label == label:
                continue
            other_pool = pools[other_label]
            if other_pool:
                negatives.append(CEDR_PREFIX + rng.choice(other_pool)["text"])
        if label == "neutral":
            metadata = {"group": "neutral", "trigger_group": item["predicate_group"], "perspective": "reported_or_nonself"}
        else:
            metadata = {"group": label, "trigger_group": item["predicate_group"], "perspective": "first_person"}
        metadata.update({"source_index": item["index"], "predicate_word": item["predicate_word"]})
        records.append(
            {
                "source": "dl-ru/srl-emotion-predicates:cedr_perspective_boundary",
                "objective": "contrastive",
                "query": CEDR_PREFIX + item["text"],
                "positive": CEDR_PREFIX + positive["text"],
                "negatives": negatives,
                "metadata": metadata,
            }
        )

    rng.shuffle(records)
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, records)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "source": "dl-ru/srl-emotion-predicates",
            "records": len(records),
            "seed": args.seed,
            "available_by_label": {key: len(value) for key, value in pools.items()},
            "selected_by_label": dict(Counter(row["metadata"]["group"] for row in records)),
            "selected_by_perspective": dict(Counter(row["metadata"]["perspective"] for row in records)),
            "skipped": dict(skipped),
            "construction": "first-person experiencer rows are emotion examples; reported/non-self experiencer rows are neutral boundary examples",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR rows used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
