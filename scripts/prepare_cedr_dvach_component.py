from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

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
CEDR_PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
DEFAULT_LABEL_MAP = {
    "positive": "joy",
    "aggression": "anger",
    "anxiety": "fear",
    "neutral": "neutral",
}


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean CEDR-shaped Dvach emotion component.")
    parser.add_argument("--count-per-group", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument("--name", default="cedr_dvach_clean4_4800")
    parser.add_argument(
        "--sarcasm-as-surprise",
        action="store_true",
        help="Map Dvach sarcasm to CEDR surprise instead of excluding it.",
    )
    parser.add_argument(
        "--sarcasm-as-neutral",
        action="store_true",
        help="Map Dvach sarcasm to CEDR neutral instead of excluding it.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    dataset = load_dataset("Kostya165/ru_emotion_dvach", cache_dir=str(CACHE_DIR))
    if args.sarcasm_as_surprise and args.sarcasm_as_neutral:
        raise ValueError("Choose at most one sarcasm mapping.")
    label_map = dict(DEFAULT_LABEL_MAP)
    if args.sarcasm_as_surprise:
        label_map["sarcasm"] = "surprise"
    if args.sarcasm_as_neutral:
        label_map["sarcasm"] = "neutral"
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    skipped = Counter()
    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            group = label_map.get(str(row["label"]).strip().lower())
            if group is None:
                skipped["unmapped_label"] += 1
                continue
            text = clean_text(row["text"])
            normalized = normalize_text(text)
            if len(normalized) < 12 or len(normalized) > 320:
                skipped["length"] += 1
                continue
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            pools[group].append({"split": split, "index": index, "text": text, "group": group})

    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    for group in sorted(set(label_map.values())):
        pool = pools[group][:]
        rng.shuffle(pool)
        if len(pool) < args.count_per_group:
            raise ValueError(f"Need {args.count_per_group} rows for {group}, got {len(pool)}")
        selected_by_group[group] = pool[: args.count_per_group]

    records = []
    groups = sorted(selected_by_group)
    for group, rows in selected_by_group.items():
        for row in rows:
            positives = [candidate for candidate in rows if candidate is not row]
            positive = rng.choice(positives)
            negatives = [
                CEDR_PREFIX + rng.choice(selected_by_group[negative_group])["text"]
                for negative_group in groups
                if negative_group != group
            ]
            records.append(
                {
                    "source": "Kostya165/ru_emotion_dvach:cedr_clean4",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + row["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": group,
                        "split": row["split"],
                        "index": row["index"],
                    },
                }
            )
    rng.shuffle(records)

    path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(path, records)
    write_json(
        path.with_name(path.stem + "_summary.json"),
        {
            "name": args.name,
            "source": "Kostya165/ru_emotion_dvach",
            "label_map": label_map,
            "excluded": [] if (args.sarcasm_as_surprise or args.sarcasm_as_neutral) else ["sarcasm"],
            "count_per_group": args.count_per_group,
            "kept": len(records),
            "available": {group: len(rows) for group, rows in pools.items()},
            "selected": dict(Counter(record["metadata"]["group"] for record in records)),
            "skipped": dict(skipped),
            "contamination_policy": "exact and near CEDR overlap removed",
        },
    )
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
