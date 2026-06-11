from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Audio, load_dataset

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import (
    CACHE_DIR,
    DATA_DIR,
    is_contaminated,
    load_cedr_index,
    normalize_text,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = {
    "neutral": "neutral",
    "happiness": "joy",
    "enthusiasm": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
}


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean CEDR-shaped RESD emotion component.")
    parser.add_argument("--seed", type=int, default=846)
    parser.add_argument("--name", default="cedr_resd_clean_emotion_component")
    args = parser.parse_args()

    cedr_index = load_cedr_index()
    dataset = load_dataset("Aniemore/resd_annotated", cache_dir=str(CACHE_DIR))
    dataset = dataset.cast_column("speech", Audio(decode=False))
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()
    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            group = LABEL_MAP.get(str(row["emotion"]))
            if group is None:
                skipped["unmapped_label"] += 1
                continue
            text = clean_text(row["text"])
            normalized = normalize_text(text)
            if len(normalized) < 8 or len(normalized) > 360:
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

    rng = random.Random(args.seed)
    groups = sorted(pools)
    records = []
    for group, items in pools.items():
        if len(items) < 2:
            continue
        for item in items:
            positives = [candidate for candidate in items if candidate is not item]
            positive = rng.choice(positives)
            negatives = []
            for negative_group in groups:
                if negative_group == group:
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(pools[negative_group])["text"])
            records.append(
                {
                    "source": "Aniemore/resd_annotated:cedr_clean",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives,
                    "metadata": {"group": group, "split": item["split"], "index": item["index"]},
                }
            )
    rng.shuffle(records)
    path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(path, records)
    path.with_name(path.stem + "_summary.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "source": "Aniemore/resd_annotated",
                "label_map": LABEL_MAP,
                "kept": len(records),
                "selected": dict(Counter(record["metadata"]["group"] for record in records)),
                "skipped": dict(skipped),
                "contamination_policy": "exact and near CEDR overlap removed",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
