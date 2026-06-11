from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


TOKEN_RE = re.compile(r"[\w]+", re.U)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("ё", "е").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize_text(value)))


def jaccard(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def load_eval_texts(cache_dir: Path) -> set[str]:
    texts: set[str] = set()
    for dataset_name in ("mteb/RuSTSBenchmarkSTS", "mteb/CEDRClassification"):
        dataset = load_dataset(dataset_name, cache_dir=str(cache_dir))
        for split in dataset:
            for row in dataset[split]:
                for key in ("sentence1", "sentence2", "text"):
                    if key in row:
                        value = normalize_text(row[key])
                        if value:
                            texts.add(value)
    return texts


def score_nli_pair(row: dict[str, Any]) -> float | None:
    label = str(row.get("label") or "").strip().lower()
    try:
        reverse = float(row.get("reverse_entailment_score") or 0.0)
    except (TypeError, ValueError):
        reverse = 0.0
    reverse = max(0.0, min(1.0, reverse))
    if label == "entailment":
        return 0.85 + 0.15 * reverse
    if label in {"not_entailment", "neutral", "contradiction"}:
        return min(0.80, reverse)
    return None


def bucket_name(score: float) -> str | None:
    if score <= 0.10:
        return "low"
    if 0.20 <= score < 0.70:
        return "mid"
    if 0.85 <= score < 0.95:
        return "high"
    if score >= 0.95:
        return "very_high"
    return None


def is_good_text_pair(left: str, right: str, min_chars: int, max_chars: int) -> bool:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if left_norm == right_norm:
        return False
    if not (min_chars <= len(left_norm) <= max_chars):
        return False
    if not (min_chars <= len(right_norm) <= max_chars):
        return False
    left_tokens = tokens(left_norm)
    right_tokens = tokens(right_norm)
    if len(left_tokens) < 4 or len(right_tokens) < 4:
        return False
    overlap = jaccard(left_norm, right_norm)
    return overlap <= 0.92


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fair RuSTS repair set from Russian NLI pairs, with exact "
            "RuSTS/CEDR text overlap removed and ordered score-balanced batches."
        )
    )
    parser.add_argument("--dataset", default="cointegrated/nli-rus-translated-v2021")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/fair_rusts_nli_pairscore_balanced_3200_seed2051.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/fair_rusts_nli_pairscore_balanced_3200_seed2051_summary.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--max-records", type=int, default=3200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2051)
    parser.add_argument("--scan-limit", type=int, default=300000)
    parser.add_argument("--min-chars", type=int, default=24)
    parser.add_argument("--max-chars", type=int, default=320)
    args = parser.parse_args()

    if args.batch_size % 4 != 0:
        raise ValueError("--batch-size must be divisible by 4")

    random.seed(args.seed)
    eval_texts = load_eval_texts(args.cache_dir)
    stream = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
        cache_dir=str(args.cache_dir),
    )

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    skipped = Counter()
    source_counts = Counter()
    label_counts = Counter()
    scanned = 0
    target_per_bucket = args.max_records // 4
    reservoir_limit = max(target_per_bucket * 3, 1200)

    for row in stream:
        scanned += 1
        if scanned > args.scan_limit:
            break
        source_counts[str(row.get("source") or "unknown")] += 1
        label_counts[str(row.get("label") or "unknown")] += 1
        left = str(row.get("premise_ru") or "")
        right = str(row.get("hypothesis_ru") or "")
        left_norm = normalize_text(left)
        right_norm = normalize_text(right)
        if left_norm in eval_texts or right_norm in eval_texts:
            skipped["eval_text_exact_overlap"] += 1
            continue
        pair_key = tuple(sorted((left_norm, right_norm)))
        if pair_key in seen_pairs:
            skipped["duplicate_pair"] += 1
            continue
        if not is_good_text_pair(left, right, args.min_chars, args.max_chars):
            skipped["quality"] += 1
            continue
        score = score_nli_pair(row)
        if score is None:
            skipped["unknown_label"] += 1
            continue
        bucket = bucket_name(score)
        if bucket is None:
            skipped["score_gap"] += 1
            continue
        seen_pairs.add(pair_key)
        record = {
            "objective": "pair_score",
            "sentence1": left_norm,
            "sentence2": right_norm,
            "score": round(score, 4),
            "source": f"{args.dataset}:{args.split}:{row.get('source')}",
            "metadata": {
                "label": row.get("label"),
                "reverse_entailment_score": row.get("reverse_entailment_score"),
                "score_bucket": bucket,
                "contamination_policy": "Exact normalized text overlap with mteb/RuSTSBenchmarkSTS and mteb/CEDRClassification removed.",
            },
        }
        if len(buckets[bucket]) < reservoir_limit:
            buckets[bucket].append(record)
        else:
            skipped["bucket_reservoir_full"] += 1
        if all(len(buckets[name]) >= reservoir_limit for name in ("low", "mid", "high", "very_high")):
            break

    for values in buckets.values():
        random.shuffle(values)

    per_bucket_per_batch = args.batch_size // 4
    used = {name: 0 for name in ("low", "mid", "high", "very_high")}
    records: list[dict[str, Any]] = []
    while len(records) + args.batch_size <= args.max_records:
        batch: list[dict[str, Any]] = []
        for name in ("low", "mid", "high", "very_high"):
            start = used[name]
            end = start + per_bucket_per_batch
            if end > len(buckets[name]):
                batch = []
                break
            batch.extend(buckets[name][start:end])
            used[name] = end
        if not batch:
            break
        random.shuffle(batch)
        records.extend(batch)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "output": str(args.out),
        "records": len(records),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "scanned": scanned,
        "scan_limit": args.scan_limit,
        "eval_texts": len(eval_texts),
        "skipped": dict(skipped),
        "source_counts_scanned_top20": source_counts.most_common(20),
        "label_counts_scanned": dict(label_counts),
        "bucket_available": {name: len(buckets[name]) for name in ("low", "mid", "high", "very_high")},
        "bucket_used": used,
        "construction": (
            "Ordered batches contain equal low/mid/high/very_high NLI-derived pair-score records. "
            "Entailment is treated as high similarity; non-entailment uses reverse entailment as a graded proxy."
        ),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
