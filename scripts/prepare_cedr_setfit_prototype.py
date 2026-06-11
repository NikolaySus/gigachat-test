from __future__ import annotations

import argparse
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
LABELS = ["joy", "sadness", "anger", "surprise", "fear"]
LABEL_RU = {
    "joy": "радость",
    "sadness": "грусть",
    "anger": "злость",
    "surprise": "удивление",
    "fear": "страх",
}
DEFAULT_SOURCES = [
    DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_dvach_clean4_4800.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_skywater_ruemotions_contrastive_strict_7200.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_brighter_rus_train_dev_clean.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_semeval2025_rus_tracka_train_dev_clean.jsonl",
]


def strip_prefix(text: str) -> str:
    text = str(text or "").strip()
    for prefix in [
        CEDR_PREFIX,
        "Определи эмоциональную окраску и смысл комментария: положительный, отрицательный или нейтральный\nкомментарий: ",
    ]:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    marker = "комментарий:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text


def clean_text(text: Any) -> str:
    text = strip_prefix(str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_pools(paths: list[Path], *, seed: int, per_label_cap: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cedr_index = load_cedr_index()
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()
    for path in paths:
        if not path.exists():
            skipped[f"missing:{path.name}"] += 1
            continue
        for index, record in enumerate(read_jsonl(path)):
            label = str(record.get("metadata", {}).get("group", ""))
            if label not in LABELS:
                skipped[f"label:{label or 'none'}"] += 1
                continue
            text = clean_text(record.get("query", ""))
            normalized = normalize_text(text)
            if len(normalized) < 8 or len(normalized) > 420:
                skipped["length"] += 1
                continue
            if normalized in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(normalized)
            if is_contaminated(text, cedr_index):
                skipped["cedr_overlap"] += 1
                continue
            pools[label].append(
                {
                    "label": label,
                    "text": text,
                    "source_file": path.name,
                    "source": record.get("source", ""),
                    "index": index,
                }
            )
    capped = {}
    for label in LABELS:
        rows = pools[label]
        rng.shuffle(rows)
        capped[label] = rows[:per_label_cap]
    return capped, {
        "source_files": [str(path.relative_to(ROOT)) for path in paths],
        "available": {label: len(pools[label]) for label in LABELS},
        "selected": {label: len(capped[label]) for label in LABELS},
        "skipped": dict(skipped),
    }


def lexical_overlap(left: str, right: str) -> float:
    a = set(normalize_text(left).split())
    b = set(normalize_text(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pick_hard_negatives(
    item: dict[str, Any],
    pools: dict[str, list[dict[str, Any]]],
    *,
    rng: random.Random,
    per_label: int,
    candidate_scan: int,
) -> list[str]:
    negatives = []
    for label in LABELS:
        if label == item["label"]:
            continue
        candidates = rng.sample(pools[label], k=min(candidate_scan, len(pools[label])))
        ranked = sorted(candidates, key=lambda row: lexical_overlap(item["text"], row["text"]), reverse=True)
        negatives.extend(CEDR_PREFIX + row["text"] for row in ranked[:per_label])
    return negatives


def make_contrastive(
    pools: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    per_label: int,
    negatives_per_label: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for label in LABELS:
        rows = pools[label][:]
        rng.shuffle(rows)
        rows = rows[:per_label]
        for item in rows:
            positive = rng.choice([row for row in pools[label] if row["text"] != item["text"]])
            records.append(
                {
                    "source": "cedr_setfit_hardneg_v1",
                    "objective": "contrastive",
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": pick_hard_negatives(
                        item,
                        pools,
                        rng=rng,
                        per_label=negatives_per_label,
                        candidate_scan=256,
                    ),
                    "metadata": {
                        "group": label,
                        "label_ru": LABEL_RU[label],
                        "source_file": item["source_file"],
                    },
                }
            )
    rng.shuffle(records)
    return records


def make_episode_records(
    pools: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    per_label: int,
    supports_per_label: int,
    objective: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for label in LABELS:
        queries = pools[label][:]
        rng.shuffle(queries)
        for item in queries[:per_label]:
            supports = {}
            for support_label in LABELS:
                candidates = [row for row in pools[support_label] if row["text"] != item["text"]]
                sampled = rng.sample(candidates, k=min(supports_per_label, len(candidates)))
                supports[support_label] = [CEDR_PREFIX + row["text"] for row in sampled]
            records.append(
                {
                    "source": f"cedr_setfit_{objective}_v1",
                    "objective": objective,
                    "query": CEDR_PREFIX + item["text"],
                    "label": label,
                    "prototypes" if objective == "prototype_classification" else "supports": supports,
                    "metadata": {
                        "group": label,
                        "label_ru": LABEL_RU[label],
                        "source_file": item["source_file"],
                    },
                }
            )
    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2921)
    parser.add_argument("--per-label-cap", type=int, default=2200)
    parser.add_argument("--contrastive-per-label", type=int, default=480)
    parser.add_argument("--episode-per-label", type=int, default=120)
    parser.add_argument("--supports-per-label", type=int, default=4)
    parser.add_argument("--negatives-per-label", type=int, default=2)
    parser.add_argument("--name", default="cedr_setfit_proto_v1")
    parser.add_argument("--sources", nargs="*", type=Path, default=DEFAULT_SOURCES)
    args = parser.parse_args()

    pools, meta = collect_pools(args.sources, seed=args.seed, per_label_cap=args.per_label_cap)
    missing_labels = [label for label in LABELS if len(pools[label]) < max(args.contrastive_per_label, args.episode_per_label)]
    if missing_labels:
        raise SystemExit(f"Not enough examples for labels: {missing_labels}; selected={meta['selected']}")

    contrastive = make_contrastive(
        pools,
        seed=args.seed + 1,
        per_label=args.contrastive_per_label,
        negatives_per_label=args.negatives_per_label,
    )
    prototype = make_episode_records(
        pools,
        seed=args.seed + 2,
        per_label=args.episode_per_label,
        supports_per_label=args.supports_per_label,
        objective="prototype_classification",
    )
    knn = make_episode_records(
        pools,
        seed=args.seed + 3,
        per_label=args.episode_per_label,
        supports_per_label=args.supports_per_label,
        objective="knn_classification",
    )

    outputs = {
        f"open_ru_1r_nc_{args.name}_contrastive.jsonl": contrastive,
        f"open_ru_1r_nc_{args.name}_prototype.jsonl": prototype,
        f"open_ru_1r_nc_{args.name}_knn.jsonl": knn,
        f"open_ru_1r_nc_{args.name}_mixed.jsonl": contrastive + prototype + knn,
    }
    rng = random.Random(args.seed + 4)
    for filename, records in outputs.items():
        records = records[:]
        rng.shuffle(records)
        write_jsonl(DATA_DIR / filename, records)
    write_json(
        DATA_DIR / f"open_ru_1r_nc_{args.name}_summary.json",
        {
            "name": args.name,
            "labels": LABELS,
            "counts": {name: len(records) for name, records in outputs.items()},
            "pool_meta": meta,
            "contamination_policy": "exact and near CEDR overlap removed again during pooling",
            "construction": {
                "contrastive": "same-label positive, lexically hardest sampled cross-label negatives",
                "prototype_classification": "query classified against per-record label prototypes",
                "knn_classification": "query classified against per-record label supports via logsumexp kNN-style loss",
            },
        },
    )
    for filename, records in outputs.items():
        print(f"prepared {(DATA_DIR / filename).relative_to(ROOT)}: {len(records)} rows")


if __name__ == "__main__":
    main()
