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
        dataset = load_dataset(dataset_name, cache_dir=str(cache_dir), trust_remote_code=True)
        for split in dataset:
            for row in dataset[split]:
                for key in ("sentence1", "sentence2", "text"):
                    if key in row:
                        value = normalize_text(row[key])
                        if value:
                            texts.add(value)
    return texts


def is_good_pair(left: str, right: str, *, min_chars: int, max_chars: int) -> bool:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if left_norm == right_norm:
        return False
    if not (min_chars <= len(left_norm) <= max_chars):
        return False
    if not (min_chars <= len(right_norm) <= max_chars):
        return False
    if len(tokens(left_norm)) < 4 or len(tokens(right_norm)) < 4:
        return False
    alpha_left = sum(char.isalpha() for char in left_norm) / max(len(left_norm), 1)
    alpha_right = sum(char.isalpha() for char in right_norm) / max(len(right_norm), 1)
    if min(alpha_left, alpha_right) < 0.45:
        return False
    return jaccard(left_norm, right_norm) <= 0.94


def add_record(
    buckets: dict[str, list[dict[str, Any]]],
    seen_pairs: set[tuple[str, str]],
    skipped: Counter[str],
    *,
    left: str,
    right: str,
    score: float,
    bucket: str,
    source: str,
    eval_texts: set[str],
    min_chars: int,
    max_chars: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if left_norm in eval_texts or right_norm in eval_texts:
        skipped["eval_text_exact_overlap"] += 1
        return
    pair_key = tuple(sorted((left_norm, right_norm)))
    if pair_key in seen_pairs:
        skipped["duplicate_pair"] += 1
        return
    if not is_good_pair(left_norm, right_norm, min_chars=min_chars, max_chars=max_chars):
        skipped["quality"] += 1
        return
    seen_pairs.add(pair_key)
    buckets[bucket].append(
        {
            "objective": "pair_score",
            "sentence1": left_norm,
            "sentence2": right_norm,
            "score": round(max(0.0, min(1.0, score)), 4),
            "source": source,
            "metadata": {
                "score_bucket": bucket,
                "jaccard": round(jaccard(left_norm, right_norm), 4),
                "contamination_policy": "Exact normalized text overlap with mteb/RuSTSBenchmarkSTS and mteb/CEDRClassification removed.",
                **(metadata or {}),
            },
        }
    )


def collect_merionum(
    buckets: dict[str, list[dict[str, Any]]],
    seen_pairs: set[tuple[str, str]],
    skipped: Counter[str],
    eval_texts: set[str],
    cache_dir: Path,
    min_chars: int,
    max_chars: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    dataset = load_dataset("merionum/ru_paraphraser", cache_dir=str(cache_dir))
    for split in dataset:
        for row in dataset[split]:
            label = str(row.get("class"))
            counts[f"{split}:{label}"] += 1
            if label == "1":
                score, bucket = 1.0, "very_high"
            elif label == "0":
                score, bucket = 0.65, "mid"
            elif label == "-1":
                score, bucket = 0.0, "low"
            else:
                skipped["merionum_unknown_label"] += 1
                continue
            add_record(
                buckets,
                seen_pairs,
                skipped,
                left=row.get("text_1", ""),
                right=row.get("text_2", ""),
                score=score,
                bucket=bucket,
                source=f"merionum/ru_paraphraser:{split}",
                eval_texts=eval_texts,
                min_chars=min_chars,
                max_chars=max_chars,
                metadata={"label": label},
            )
    return counts


def collect_paws(
    buckets: dict[str, list[dict[str, Any]]],
    seen_pairs: set[tuple[str, str]],
    skipped: Counter[str],
    eval_texts: set[str],
    cache_dir: Path,
    min_chars: int,
    max_chars: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    dataset = load_dataset("MilyaShams/paws-ru_10k", cache_dir=str(cache_dir))
    for split in dataset:
        for row in dataset[split]:
            label = int(row.get("label"))
            counts[f"{split}:{label}"] += 1
            if label == 1:
                score, bucket = 0.9, "high"
            else:
                score, bucket = 0.0, "low"
            add_record(
                buckets,
                seen_pairs,
                skipped,
                left=row.get("sentence1", ""),
                right=row.get("sentence2", ""),
                score=score,
                bucket=bucket,
                source=f"MilyaShams/paws-ru_10k:{split}",
                eval_texts=eval_texts,
                min_chars=min_chars,
                max_chars=max_chars,
                metadata={"label": label},
            )
    return counts


def collect_andidu(
    buckets: dict[str, list[dict[str, Any]]],
    seen_pairs: set[tuple[str, str]],
    skipped: Counter[str],
    eval_texts: set[str],
    cache_dir: Path,
    min_chars: int,
    max_chars: int,
    rng: random.Random,
    scan_limit: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    positives: list[tuple[str, str, str]] = []
    for dataset_name, source_tag in (
        ("andidu/paraphrase-ru-reviews", "reviews"),
        ("andidu/paraphrase-ru-it", "it"),
    ):
        dataset = load_dataset(dataset_name, split="train", cache_dir=str(cache_dir))
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        for index in indices[:scan_limit]:
            row = dataset[index]
            left = row.get("O", "")
            right = row.get("P", "")
            counts[f"{source_tag}:scanned"] += 1
            overlap = jaccard(left, right)
            if not (0.08 <= overlap <= 0.86):
                skipped[f"{source_tag}_jaccard"] += 1
                continue
            add_record(
                buckets,
                seen_pairs,
                skipped,
                left=left,
                right=right,
                score=0.9,
                bucket="high",
                source=f"{dataset_name}:train",
                eval_texts=eval_texts,
                min_chars=min_chars,
                max_chars=max_chars,
                metadata={"dataset_row_id": row.get("id")},
            )
            positives.append((normalize_text(left), normalize_text(right), dataset_name))

    if not positives:
        return counts

    # Random cross-pair negatives are intentionally mixed across domains. They
    # give the correlation/rank loss low-similarity anchors without using any
    # benchmark text or released-model teacher.
    left_pool = [left for left, _, _ in positives]
    right_pool = [right for _, right, _ in positives]
    target_cross = min(len(positives), 4000)
    for _ in range(target_cross * 4):
        if counts["andidu_cross_added"] >= target_cross:
            break
        left = rng.choice(left_pool)
        right = rng.choice(right_pool)
        overlap = jaccard(left, right)
        if overlap > 0.18:
            skipped["andidu_cross_too_similar"] += 1
            continue
        add_record(
            buckets,
            seen_pairs,
            skipped,
            left=left,
            right=right,
            score=0.0,
            bucket="low",
            source="andidu/paraphrase-cross-negative",
            eval_texts=eval_texts,
            min_chars=min_chars,
            max_chars=max_chars,
            metadata={"negative_construction": "random_cross_pair"},
        )
        counts["andidu_cross_added"] += 1
    return counts


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare fair ordered pair-score batches from Russian paraphrase datasets."
    )
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/fair_rusts_paraphrase_pairscore_balanced_6400_seed2071.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/fair_rusts_paraphrase_pairscore_balanced_6400_seed2071_summary.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--max-records", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2071)
    parser.add_argument("--andidu-scan-limit", type=int, default=12000)
    parser.add_argument("--min-chars", type=int, default=18)
    parser.add_argument("--max-chars", type=int, default=420)
    args = parser.parse_args()

    if args.batch_size % 4 != 0:
        raise ValueError("--batch-size must be divisible by 4")

    rng = random.Random(args.seed)
    eval_texts = load_eval_texts(args.cache_dir)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    skipped: Counter[str] = Counter()
    source_counts = {
        "merionum": collect_merionum(buckets, seen_pairs, skipped, eval_texts, args.cache_dir, args.min_chars, args.max_chars),
        "paws": collect_paws(buckets, seen_pairs, skipped, eval_texts, args.cache_dir, args.min_chars, args.max_chars),
        "andidu": collect_andidu(
            buckets,
            seen_pairs,
            skipped,
            eval_texts,
            args.cache_dir,
            args.min_chars,
            args.max_chars,
            rng,
            args.andidu_scan_limit,
        ),
    }

    for values in buckets.values():
        rng.shuffle(values)

    bucket_order = ("low", "mid", "high", "very_high")
    per_bucket = args.batch_size // len(bucket_order)
    used = {name: 0 for name in bucket_order}
    records: list[dict[str, Any]] = []
    while len(records) + args.batch_size <= args.max_records:
        batch: list[dict[str, Any]] = []
        for name in bucket_order:
            start = used[name]
            end = start + per_bucket
            if end > len(buckets[name]):
                batch = []
                break
            batch.extend(buckets[name][start:end])
            used[name] = end
        if not batch:
            break
        rng.shuffle(batch)
        records.extend(batch)

    write_jsonl(args.out, records)
    summary = {
        "output": str(args.out),
        "records": len(records),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "eval_texts": len(eval_texts),
        "bucket_available": {name: len(buckets[name]) for name in bucket_order},
        "bucket_used": used,
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
        "skipped": dict(skipped),
        "construction": (
            "Ordered batches contain low/mid/high/very_high pair-score records from "
            "merionum/ru_paraphraser, MilyaShams/paws-ru_10k, and andidu paraphrase "
            "datasets. Exact RuSTS/CEDR eval text overlaps are removed."
        ),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
