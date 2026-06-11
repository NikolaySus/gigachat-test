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
    ROOT,
    is_contaminated,
    load_cedr_index,
    normalize_text,
    write_json,
    write_jsonl,
)


PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
EMOTION_LABELS = ["joy", "sadness", "surprise", "fear", "anger"]
ALL_LABELS = ["neutral", *EMOTION_LABELS]
DEFAULT_SOURCES = [
    DATA_DIR / "open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_dvach_clean4_4800.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_skywater_ruemotions_contrastive_strict_7200.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_seara_rugoemotions_strict_prior9000.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_brighter_rus_train_dev_clean.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_semeval2025_rus_tracka_train_dev_clean.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_3200.jsonl",
    DATA_DIR / "open_ru_1r_nc_cedr_blog_neutral_broad_1600.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strip_prefix(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith(PREFIX):
        return text[len(PREFIX) :].strip()
    marker = "комментарий:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text


def clean_text(text: Any) -> str:
    text = strip_prefix(str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def labels_from_group(value: Any) -> list[str] | None:
    group = str(value or "").strip()
    if group in {"", "none", "no_emotion", "neutral"}:
        return []
    labels = [label for label in re.split(r"[+,|;/\s]+", group) if label in EMOTION_LABELS]
    if labels:
        return sorted(set(labels), key=EMOTION_LABELS.index)
    return None


def iter_labeled_texts(record: dict[str, Any]) -> list[tuple[str, list[str]]]:
    metadata = record.get("metadata") or {}
    labels = labels_from_group(metadata.get("group") or metadata.get("raw_label") or metadata.get("label"))
    if labels is None:
        return []
    texts = []
    objective = record.get("objective")
    if objective == "contrastive":
        texts.extend([record.get("query", ""), record.get("positive", "")])
        texts.extend(record.get("positives", []))
    elif objective == "pair_score":
        return []
    else:
        texts.append(record.get("query", "") or record.get("text", ""))
    return [(clean_text(text), labels) for text in texts if clean_text(text)]


def collect_pools(paths: list[Path], *, seed: int, per_label_cap: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rng = random.Random(seed)
    cedr_index = load_cedr_index()
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    skipped = Counter()
    for path in paths:
        if not path.exists():
            skipped[f"missing:{path.name}"] += 1
            continue
        for index, record in enumerate(read_jsonl(path)):
            for text, labels in iter_labeled_texts(record):
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
                pool_label = "neutral" if not labels else labels[0] if len(labels) == 1 else "+".join(labels)
                pools[pool_label].append(
                    {
                        "text": text,
                        "labels": labels,
                        "source_file": path.name,
                        "source": record.get("source", ""),
                        "index": index,
                    }
                )
    capped: dict[str, list[dict[str, Any]]] = {}
    for label, rows in pools.items():
        rows = rows[:]
        rng.shuffle(rows)
        capped[label] = rows[:per_label_cap]
    return capped, {
        "source_files": [str(path.relative_to(ROOT)) for path in paths],
        "available": {label: len(rows) for label, rows in sorted(pools.items())},
        "selected": {label: len(rows) for label, rows in sorted(capped.items())},
        "skipped": dict(skipped),
    }


def lexical_overlap(left: str, right: str) -> float:
    a = set(normalize_text(left).split())
    b = set(normalize_text(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def make_support_records(
    pools: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    neutral_records: int,
    emotion_records_per_label: int,
    supports_per_label: int,
    hard_scan: int,
    positive_weight: float,
    similarity_threshold: float,
    support_pooling: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    support_pools = {label: pools.get(label, []) for label in EMOTION_LABELS}
    missing = [label for label, rows in support_pools.items() if len(rows) < supports_per_label + 1]
    if missing:
        raise RuntimeError(f"Missing support labels: {missing}")

    queries: list[dict[str, Any]] = []
    queries.extend(pools.get("neutral", [])[:neutral_records])
    for label in EMOTION_LABELS:
        queries.extend(pools.get(label, [])[:emotion_records_per_label])
    rng.shuffle(queries)

    records = []
    for item in queries:
        supports: dict[str, list[str]] = {}
        for label in EMOTION_LABELS:
            candidates = [row for row in support_pools[label] if row["text"] != item["text"]]
            scanned = rng.sample(candidates, k=min(hard_scan, len(candidates)))
            if label in item["labels"]:
                rng.shuffle(scanned)
                picked = scanned[:supports_per_label]
            else:
                picked = sorted(scanned, key=lambda row: lexical_overlap(item["text"], row["text"]), reverse=True)[
                    :supports_per_label
                ]
            supports[label] = [PREFIX + row["text"] for row in picked]
        records.append(
            {
                "source": "cedr_multilabel_support_v1",
                "objective": "multilabel_support_classification",
                "query": PREFIX + item["text"],
                "labels": item["labels"],
                "supports": supports,
                "positive_weight": positive_weight,
                "similarity_threshold": similarity_threshold,
                "support_pooling": support_pooling,
                "metadata": {
                    "groups": item["labels"] or ["neutral"],
                    "source_file": item["source_file"],
                    "supports_per_label": supports_per_label,
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="cedr_multilabel_support_v1")
    parser.add_argument("--seed", type=int, default=5317)
    parser.add_argument("--per-label-cap", type=int, default=2600)
    parser.add_argument("--neutral-records", type=int, default=2600)
    parser.add_argument("--emotion-records-per-label", type=int, default=900)
    parser.add_argument("--supports-per-label", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--hard-scan", type=int, default=256)
    parser.add_argument("--positive-weight", type=float, default=2.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.45)
    parser.add_argument("--support-pooling", default="max", choices=["max", "mean_top2", "logsumexp"])
    parser.add_argument("--sources", nargs="*", type=Path, default=DEFAULT_SOURCES)
    args = parser.parse_args()

    pools, meta = collect_pools(args.sources, seed=args.seed, per_label_cap=args.per_label_cap)
    summary = {
        "name": args.name,
        "labels": EMOTION_LABELS,
        "pool_meta": meta,
        "outputs": {},
        "contamination_policy": "Exact and near CEDR train/test overlap removed during pooling.",
        "objective": "multilabel_support_classification; BCE over five emotion support sets, including empty-label neutral queries.",
    }
    for supports_per_label in args.supports_per_label:
        records = make_support_records(
            pools,
            seed=args.seed + supports_per_label,
            neutral_records=args.neutral_records,
            emotion_records_per_label=args.emotion_records_per_label,
            supports_per_label=supports_per_label,
            hard_scan=args.hard_scan,
            positive_weight=args.positive_weight,
            similarity_threshold=args.similarity_threshold,
            support_pooling=args.support_pooling,
        )
        rng = random.Random(args.seed + 100 + supports_per_label)
        rng.shuffle(records)
        path = DATA_DIR / f"open_ru_1r_nc_{args.name}_p{supports_per_label}_{len(records)}.jsonl"
        write_jsonl(path, records)
        summary["outputs"][f"p{supports_per_label}"] = {
            "path": str(path.relative_to(ROOT)),
            "records": len(records),
            "query_label_counts": dict(Counter("+".join(row["labels"]) or "neutral" for row in records)),
            "supports_per_label": supports_per_label,
        }
        print(f"prepared {path.relative_to(ROOT)}: {len(records)} rows")

    summary_path = DATA_DIR / f"open_ru_1r_nc_{args.name}_summary.json"
    write_json(summary_path, summary)
    print(summary_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
