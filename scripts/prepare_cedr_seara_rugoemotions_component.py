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
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["neutral", "joy", "sadness", "surprise", "fear", "anger"]
CEDR_PRIOR = {
    "neutral": 3043,
    "joy": 1569,
    "sadness": 1417,
    "surprise": 607,
    "fear": 589,
    "anger": 411,
}


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^>+", "", text).strip()
    return text


def quality_ok(text: str, *, min_chars: int, max_chars: int) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < min_chars or len(normalized) > max_chars:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.35 * max(1, len(text)):
        return False
    return True


def target_counts(total: int) -> dict[str, int]:
    prior_sum = sum(CEDR_PRIOR.values())
    counts = {group: int(total * CEDR_PRIOR[group] / prior_sum) for group in GROUPS}
    remainder = total - sum(counts.values())
    order = sorted(
        GROUPS,
        key=lambda group: total * CEDR_PRIOR[group] / prior_sum - counts[group],
        reverse=True,
    )
    for group in order[:remainder]:
        counts[group] += 1
    return counts


def single_group(row: dict[str, Any]) -> str | None:
    active = [group for group in GROUPS if int(row.get(group, 0) or 0) == 1]
    if len(active) == 1:
        return active[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean seara/ru_go_emotions CEDR-shaped component.")
    parser.add_argument("--name", default="cedr_seara_rugoemotions_strict_prior9000")
    parser.add_argument("--count", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=1181)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=360)
    parser.add_argument("--dataset", default="seara/ru_go_emotions")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cedr_index = load_cedr_index()
    dataset = load_dataset(args.dataset, "raw", split="train", cache_dir=str(CACHE_DIR))

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen: set[str] = set()
    for index, row in enumerate(dataset):
        if bool(row.get("example_very_unclear", False)):
            skipped["unclear"] += 1
            continue
        group = single_group(row)
        if group is None:
            skipped["multilabel_or_unmapped"] += 1
            continue
        text = clean_text(row.get("ru_text"))
        normalized = normalize_text(text)
        if normalized in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(normalized)
        if not quality_ok(text, min_chars=args.min_chars, max_chars=args.max_chars):
            skipped["quality"] += 1
            continue
        if is_contaminated(text, cedr_index):
            skipped["cedr_overlap"] += 1
            continue
        pools[group].append(
            {
                "text": text,
                "group": group,
                "index": index,
                "id": row.get("id"),
                "subreddit": row.get("subreddit"),
            }
        )

    targets = target_counts(args.count)
    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    for group in GROUPS:
        pool = pools[group][:]
        rng.shuffle(pool)
        if len(pool) < targets[group]:
            raise RuntimeError(f"Not enough rows for {group}: need {targets[group]}, got {len(pool)}")
        selected_by_group[group] = pool[: targets[group]]

    records = []
    for group, items in selected_by_group.items():
        negative_groups = [candidate for candidate in GROUPS if candidate != group]
        for item in items:
            positives = [candidate for candidate in items if candidate is not item]
            negatives = [
                CEDR_PREFIX + rng.choice(selected_by_group[negative_group])["text"]
                for negative_group in negative_groups
            ]
            records.append(
                {
                    "source": f"{args.dataset}:strict_single_cedr_prior",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positives)["text"],
                    "negatives": negatives,
                    "metadata": {
                        "group": group,
                        "index": item["index"],
                        "id": item["id"],
                        "subreddit": item["subreddit"],
                    },
                }
            )

    rng.shuffle(records)
    output_path = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(output_path, records)
    output_path.with_name(output_path.stem + "_summary.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "source": f"{args.dataset}/raw",
                "records": len(records),
                "requested": args.count,
                "targets": targets,
                "available": {group: len(pools[group]) for group in GROUPS},
                "selected": dict(Counter(record["metadata"]["group"] for record in records)),
                "skipped": dict(skipped),
                "seed": args.seed,
                "construction": "strict single CEDR-group labels only; CEDR prior class balance; same-label positives and cross-label negatives",
                "contamination_policy": "exact and near CEDR overlap removed; CEDR records are not used",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"prepared {output_path.relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
