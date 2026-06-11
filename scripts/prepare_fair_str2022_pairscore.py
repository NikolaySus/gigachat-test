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


def bucket_name(score: float) -> str:
    if score <= 0.20:
        return "low"
    if score < 0.50:
        return "mid_low"
    if score < 0.80:
        return "mid_high"
    return "high"


def split_pair(text: str) -> tuple[str, str] | None:
    parts = [part.strip() for part in str(text or "").split("\n") if part.strip()]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def is_good_text(text: str, min_chars: int, max_chars: int) -> bool:
    norm = normalize_text(text)
    if not (min_chars <= len(norm) <= max_chars):
        return False
    return len(tokens(norm)) >= 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fair graded relatedness pair-score set from STR-2022, excluding STS source rows."
    )
    parser.add_argument("--dataset", default="vkpriya/str-2022")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/fair_rusts_str2022_nonsts_pairscore_balanced_seed2271.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/fair_rusts_str2022_nonsts_pairscore_balanced_seed2271_summary.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--max-records", type=int, default=1968)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2271)
    parser.add_argument("--min-chars", type=int, default=18)
    parser.add_argument("--max-chars", type=int, default=360)
    args = parser.parse_args()

    if args.batch_size % 4 != 0:
        raise ValueError("--batch-size must be divisible by 4")

    rng = random.Random(args.seed)
    eval_texts = load_eval_texts(args.cache_dir)
    dataset = load_dataset(args.dataset, split=args.split, cache_dir=str(args.cache_dir))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    source_counts = Counter()
    seen_pairs: set[tuple[str, str]] = set()

    for row in dataset:
        source_id = str(row.get("SourceID") or "")
        source_counts[source_id] += 1
        if source_id == "STS":
            skipped["source_sts_excluded"] += 1
            continue
        pair = split_pair(str(row.get("Text") or ""))
        if pair is None:
            skipped["bad_pair_format"] += 1
            continue
        left, right = pair
        left_norm = normalize_text(left)
        right_norm = normalize_text(right)
        if not left_norm or not right_norm or left_norm == right_norm:
            skipped["empty_or_same"] += 1
            continue
        if not is_good_text(left_norm, args.min_chars, args.max_chars) or not is_good_text(right_norm, args.min_chars, args.max_chars):
            skipped["quality"] += 1
            continue
        if left_norm in eval_texts or right_norm in eval_texts:
            skipped["eval_text_exact_overlap"] += 1
            continue
        pair_key = tuple(sorted((left_norm, right_norm)))
        if pair_key in seen_pairs:
            skipped["duplicate_pair"] += 1
            continue
        seen_pairs.add(pair_key)

        score = max(0.0, min(1.0, float(row["Score"])))
        bucket = bucket_name(score)
        buckets[bucket].append(
            {
                "objective": "pair_score",
                "sentence1": left_norm,
                "sentence2": right_norm,
                "score": round(score, 4),
                "source": f"{args.dataset}:{args.split}:{source_id}",
                "metadata": {
                    "index": row.get("Index"),
                    "source_id": source_id,
                    "subset_id": row.get("SubsetID"),
                    "pair_id": row.get("PairID"),
                    "score_bucket": bucket,
                    "contamination_policy": "SourceID=STS excluded. Exact normalized text overlap with mteb/RuSTSBenchmarkSTS and mteb/CEDRClassification removed.",
                },
            }
        )

    for values in buckets.values():
        rng.shuffle(values)

    bucket_order = ("low", "mid_low", "mid_high", "high")
    per_bucket_per_batch = args.batch_size // len(bucket_order)
    requested_batches = args.max_records // args.batch_size
    max_batches = min(
        requested_batches,
        *(len(buckets[name]) // per_bucket_per_batch for name in bucket_order),
    )
    records: list[dict[str, Any]] = []
    used = {name: 0 for name in bucket_order}
    for batch_index in range(max_batches):
        batch: list[dict[str, Any]] = []
        for name in bucket_order:
            start = batch_index * per_bucket_per_batch
            end = start + per_bucket_per_batch
            batch.extend(buckets[name][start:end])
            used[name] += per_bucket_per_batch
        rng.shuffle(batch)
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
        "eval_texts": len(eval_texts),
        "source_counts": dict(source_counts),
        "skipped": dict(skipped),
        "bucket_available": {name: len(buckets[name]) for name in bucket_order},
        "bucket_used": used,
        "construction": "Balanced batches contain equal low/mid-low/mid-high/high STR-2022 relatedness rows after excluding SourceID=STS.",
        "fairness": "No STS source rows are used; exact RuSTS/CEDR text overlaps are removed. This is English relatedness data, not STS-B/RuSTS target data.",
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
