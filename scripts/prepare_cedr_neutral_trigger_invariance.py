from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX, GROUPS
from prepare_cedr_neutral_lexical_distractors import (
    DEFAULT_SOURCES,
    LEXEMES,
    clean_text,
    collect_candidates,
    read_jsonl,
)
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, load_cedr_index, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def mask_trigger(text: str, group: str) -> str:
    pattern = LEXEMES[group]
    masked = pattern.sub("[эмоциональное слово]", text)
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked if masked != text else f"Нейтральное сообщение без выраженной эмоции: {text}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine neutral CEDR examples where emotion-trigger words should be invariant."
    )
    parser.add_argument("--source", type=Path, action="append", default=None)
    parser.add_argument(
        "--go-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    parser.add_argument("--count", type=int, default=3200)
    parser.add_argument("--negatives-per-row", type=int, default=3)
    parser.add_argument("--seed", type=int, default=903)
    parser.add_argument("--name", default="cedr_neutral_trigger_invariance_3200")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    paths = args.source or DEFAULT_SOURCES
    pools = collect_candidates(paths, cedr_index=cedr_index)
    skipped = pools.pop("_skipped")[0]["counter"] if "_skipped" in pools else {}

    go_rows = read_jsonl(args.go_path)
    emotion_pools: dict[str, list[str]] = {group: [] for group in GROUPS if group != "neutral"}
    for row in go_rows:
        group = row.get("metadata", {}).get("group")
        if group in emotion_pools:
            emotion_pools[group].append(row["query"])

    groups = [group for group in ["fear", "surprise", "joy", "sadness", "anger"] if pools.get(group)]
    selected: list[dict[str, Any]] = []
    per_group = args.count // max(1, len(groups))
    remainder = args.count % max(1, len(groups))
    selected_by_group: dict[str, int] = {}
    for index, group in enumerate(groups):
        target = per_group + (1 if index < remainder else 0)
        rows = pools[group][:]
        rng.shuffle(rows)
        rows = rows[: min(target, len(rows))]
        selected_by_group[group] = len(rows)
        selected.extend(rows)

    records = []
    for item in selected:
        group = item["trigger_group"]
        negatives = []
        hard_pool = emotion_pools.get(group) or []
        if hard_pool:
            negatives.extend(rng.sample(hard_pool, k=min(args.negatives_per_row, len(hard_pool))))
        for negative_group in ["joy", "sadness", "surprise", "fear", "anger"]:
            if negative_group == group or len(negatives) >= args.negatives_per_row + 2:
                continue
            pool = emotion_pools.get(negative_group) or []
            if pool:
                negatives.append(rng.choice(pool))
        records.append(
            {
                "source": f"neutral_trigger_invariance:{item['source']}",
                "objective": "contrastive",
                "query": CEDR_PREFIX + item["text"],
                "positive": CEDR_PREFIX + mask_trigger(item["text"], group),
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
            "selected_by_trigger_group": selected_by_group,
            "skipped": skipped,
            "go_path": str(args.go_path),
            "construction": "neutral query paired with same text after trigger masking; same-trigger emotion negatives",
            "contamination_policy": "exact and near CEDR overlap removed; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
