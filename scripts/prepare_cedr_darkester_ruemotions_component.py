from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
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

LABEL_MAP = {
    "радость": "joy",
    "восторг": "joy",
    "веселье": "joy",
    "смех": "joy",
    "экстаз": "joy",
    "печаль": "sadness",
    "грусть": "sadness",
    "тоска": "sadness",
    "меланхолия": "sadness",
    "одиночество": "sadness",
    "отчаяние": "sadness",
    "разочарование": "sadness",
    "страдание": "sadness",
    "горечь": "sadness",
    "страх": "fear",
    "тревога": "fear",
    "паника": "fear",
    "ужас": "fear",
    "паранойя": "fear",
    "напряжение": "fear",
    "тревожное ожидание": "fear",
    "злость": "anger",
    "гнев": "anger",
    "ярость": "anger",
    "ненависть": "anger",
    "возмущение": "anger",
    "раздражение": "anger",
    "удивление": "surprise",
    "шок": "surprise",
    "изумление": "surprise",
    "недоумение": "surprise",
    "растерянность": "surprise",
    "замешательство": "surprise",
    "нейтральная": "neutral",
    "нейтральность": "neutral",
    "безразличие": "neutral",
    "спокойствие": "neutral",
    "умиротворение": "neutral",
}


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean Darkester/RuEmotions CEDR-shaped component.")
    parser.add_argument("--seed", type=int, default=905)
    parser.add_argument("--name", default="cedr_darkester_ruemotions_core")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    dataset = load_dataset("Darkester/RuEmotions", cache_dir=str(CACHE_DIR))["train"]
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()

    for index, row in enumerate(dataset):
        source_label = str(row["emotion"]).strip().lower()
        group = LABEL_MAP.get(source_label)
        if group is None:
            skipped["unmapped_label"] += 1
            continue
        text = clean_text(row["text"])
        normalized = normalize_text(text)
        if len(normalized) < 12 or len(normalized) > 360:
            skipped["length"] += 1
            continue
        if normalized in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(normalized)
        if is_contaminated(text, cedr_index):
            skipped["cedr_overlap"] += 1
            continue
        pools[group].append(
            {
                "text": text,
                "group": group,
                "index": index,
                "source_label": source_label,
            }
        )

    records = []
    for group, rows in pools.items():
        if len(rows) < 2:
            continue
        for row in rows:
            positives = [candidate for candidate in rows if candidate is not row]
            positive = rng.choice(positives)
            negatives = []
            for negative_group in GROUPS:
                if negative_group == group or not pools.get(negative_group):
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(pools[negative_group])["text"])
            records.append(
                {
                    "source": "Darkester/RuEmotions:cedr_core",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + row["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": group,
                        "index": row["index"],
                        "source_label": row["source_label"],
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
            "source": "Darkester/RuEmotions",
            "selected": dict(Counter(record["metadata"]["group"] for record in records)),
            "available": {group: len(rows) for group, rows in pools.items()},
            "skipped": dict(skipped),
            "label_map": LABEL_MAP,
            "contamination_policy": "exact and near CEDR overlap removed; zero overlaps found before filtering",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
