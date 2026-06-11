from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_cedr_goemotions_ru_component import CEDR_PREFIX
from prepare_cedr_neutral_lexical_distractors import read_jsonl
from prepare_open_ru_1r_nc_cedr_sentiment_ablations import DATA_DIR, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def strip_prefix(text: str) -> str:
    return text.removeprefix(CEDR_PREFIX).strip()


def load_grouped_texts(path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(path):
        group = row.get("metadata", {}).get("group")
        query = row.get("query")
        if isinstance(group, str) and isinstance(query, str) and query.strip():
            grouped[group].append(strip_prefix(query))
    return grouped


def load_factual_novelty(path: Path) -> list[str]:
    texts = []
    seen = set()
    for row in read_jsonl(path):
        query = row.get("query")
        if not isinstance(query, str):
            continue
        text = strip_prefix(query)
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return texts


def make_records(
    *,
    factual_texts: list[str],
    emotion_groups: dict[str, list[str]],
    count: int,
    seed: int,
    positives_per_record: int,
    negatives_per_group: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if len(factual_texts) < 2:
        raise RuntimeError("Need at least two factual novelty texts")

    rows = []
    repeated = list(factual_texts)
    while len(repeated) < count:
        repeated.extend(factual_texts)
    rng.shuffle(repeated)

    hard_groups = ["surprise", "anger", "fear", "sadness", "joy"]
    available = {group: values for group, values in emotion_groups.items() if values}
    missing = [group for group in hard_groups if group not in available]
    if missing:
        raise RuntimeError(f"Missing hard-negative groups: {missing}")

    for index, text in enumerate(repeated[:count]):
        positive_pool = [candidate for candidate in factual_texts if candidate != text]
        positives = rng.sample(positive_pool, k=min(positives_per_record + 1, len(positive_pool)))
        positive = positives[0]
        extra_positives = positives[1:]
        negatives = []
        for group in hard_groups:
            pool = available[group]
            k = min(negatives_per_group, len(pool))
            negatives.extend(rng.sample(pool, k=k))
        rng.shuffle(negatives)
        rows.append(
            {
                "source": "data-silence/lenta.ru_2-extended:cedr_factual_novelty_neutral",
                "objective": "contrastive",
                "query": CEDR_PREFIX + text,
                "positive": CEDR_PREFIX + positive,
                "positives": [CEDR_PREFIX + value for value in extra_positives],
                "negatives": negatives,
                "metadata": {
                    "group": "neutral",
                    "boundary": "factual novelty / reported surprise should stay neutral",
                    "index": index,
                },
            }
        )
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build clean CEDR boundary data where factual novelty is treated as neutral."
    )
    parser.add_argument("--name", default="cedr_lenta_factual_novelty_neutral_2400")
    parser.add_argument("--count", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=951)
    parser.add_argument("--positives-per-record", type=int, default=2)
    parser.add_argument("--negatives-per-group", type=int, default=1)
    parser.add_argument(
        "--factual-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_lenta_factual_surprise_strict_reported_1200.jsonl",
    )
    parser.add_argument(
        "--emotion-path",
        type=Path,
        default=DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    )
    args = parser.parse_args()

    factual_texts = load_factual_novelty(args.factual_path)
    emotion_groups = load_grouped_texts(args.emotion_path)
    rows = make_records(
        factual_texts=factual_texts,
        emotion_groups=emotion_groups,
        count=args.count,
        seed=args.seed,
        positives_per_record=args.positives_per_record,
        negatives_per_group=args.negatives_per_group,
    )
    out = DATA_DIR / f"open_ru_1r_nc_{args.name}.jsonl"
    write_jsonl(out, rows)
    write_json(
        out.with_name(out.stem + "_summary.json"),
        {
            "name": args.name,
            "records": len(rows),
            "source_factual_rows": len(factual_texts),
            "factual_path": str(args.factual_path),
            "emotion_path": str(args.emotion_path),
            "emotion_counts": {group: len(values) for group, values in sorted(emotion_groups.items())},
            "record_groups": dict(Counter(row["metadata"]["group"] for row in rows)),
            "construction": (
                "Factual Lenta rows containing surprise/novelty lexemes are neutral positives; "
                "GoEmotions-RU surprise and other emotions are hard negatives."
            ),
            "contamination_policy": "inherits factual-path CEDR exact/near-overlap filtering; no CEDR records used",
        },
    )
    print(f"prepared {out.relative_to(ROOT)}: {len(rows)} rows")


if __name__ == "__main__":
    main()
