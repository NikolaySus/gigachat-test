from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
CEDR_PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
DEFAULT_SOURCE_DIR = (
    ROOT
    / "results/official_repro_cache/hf_home/datasets--anonymous-dialogs-org--dialogs-ru-emotional-conversations"
    / "snapshots/6ef2310d9da659276bc38d708c2f0bab1239ae92"
)

LABEL_MAP = {
    "neutral": "neutral",
    "happy": "joy",
    "laughing": "joy",
    "sad": "sadness",
    "angry": "anger",
    "surprise": "surprise",
    "fear": "fear",
}

EMOTION_LEXEMES = re.compile(
    r"\b("
    r"рад|счаст|любим|обож|весел|груст|печал|тоск|жаль|"
    r"злит|зл|бесит|гнев|ярост|ужас|страш|страх|бою|опас|"
    r"удив|шок|неожидан|потряса|волн|тревож"
    r")",
    re.IGNORECASE,
)
BAD_PATTERNS = [
    re.compile(r"\bплюс\s+семь\b", re.IGNORECASE),
    re.compile(r"\bвосемь\s+девять\b", re.IGNORECASE),
    re.compile(r"\bжидкостями и чувствами\b", re.IGNORECASE),
    re.compile(r"\bикском\b", re.IGNORECASE),
]


def clean_text(value: Any) -> str:
    text = str(value or "").strip().strip('"')
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_bad(text: str) -> bool:
    if any(pattern.search(text) for pattern in BAD_PATTERNS):
        return True
    if len(re.findall(r"\d", text)) >= 4:
        return True
    if text.count('"') >= 4:
        return True
    return False


def read_rows(source_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = source_dir / f"{split}.csv"
        with path.open(encoding="utf-8", errors="replace") as file:
            for index, row in enumerate(csv.DictReader(file, delimiter="|")):
                row["split"] = split
                row["source_index"] = index
                rows.append(row)
    return rows


def build_records(pools: dict[str, list[dict[str, Any]]], *, count_per_group: int, rng: random.Random) -> list[dict[str, Any]]:
    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    for group, rows in pools.items():
        rows = rows[:]
        rng.shuffle(rows)
        selected_by_group[group] = rows[: min(count_per_group, len(rows))]

    groups = sorted(selected_by_group)
    records = []
    for group, rows in selected_by_group.items():
        if len(rows) < 2:
            continue
        for row in rows:
            positive = rng.choice([candidate for candidate in rows if candidate is not row])
            negatives = []
            for negative_group in groups:
                if negative_group == group or not selected_by_group[negative_group]:
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(selected_by_group[negative_group])["text"])
            records.append(
                {
                    "source": "anonymous-dialogs-org/dialogs-ru-emotional-conversations:cedr_boundary",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + row["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": group,
                        "emotion": row["emotion"],
                        "split": row["split"],
                        "source_index": row["source_index"],
                        "neutral_has_emotion_lexeme": bool(row.get("neutral_has_emotion_lexeme")),
                    },
                }
            )
    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a clean CEDR boundary component from Russian emotional dialogs.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--count-per-group", type=int, default=420)
    parser.add_argument("--seed", type=int, default=1229)
    parser.add_argument("--name", default="cedr_dialogs_emotion_boundary_2520")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()

    for row in read_rows(args.source_dir):
        emotion = str(row.get("emotion") or "").strip().lower()
        group = LABEL_MAP.get(emotion)
        if group is None:
            skipped["unmapped_label"] += 1
            continue
        text = clean_text(row.get("text"))
        normalized = normalize_text(text)
        if len(normalized) < 20 or len(normalized) > 260:
            skipped["length"] += 1
            continue
        if looks_bad(text):
            skipped["synthetic_artifact"] += 1
            continue
        if normalized in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(normalized)
        if is_contaminated(text, cedr_index):
            skipped["cedr_overlap"] += 1
            continue
        if group == "neutral":
            if not EMOTION_LEXEMES.search(text):
                skipped["neutral_without_boundary_lexeme"] += 1
                continue
            row["neutral_has_emotion_lexeme"] = True
        row["text"] = text
        row["group"] = group
        pools[group].append(row)

    records = build_records(pools, count_per_group=args.count_per_group, rng=rng)
    path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(path, records)
    write_json(
        path.with_name(path.stem + "_summary.json"),
        {
            "name": args.name,
            "source": "anonymous-dialogs-org/dialogs-ru-emotional-conversations",
            "label_map": LABEL_MAP,
            "count_per_group": args.count_per_group,
            "available": {group: len(rows) for group, rows in pools.items()},
            "selected": dict(Counter(record["metadata"]["group"] for record in records)),
            "records": len(records),
            "skipped": dict(skipped),
            "contamination_policy": "exact and near CEDR overlap removed",
            "neutral_policy": "keep only neutral rows containing emotion-bearing lexemes",
        },
    )
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
