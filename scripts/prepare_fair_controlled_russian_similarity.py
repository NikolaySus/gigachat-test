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


RUSTS_PREFIXES = [
    "семантически похожий текст: ",
    "семантически похожий текст \nтекст: ",
]


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("ё", "е").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(value))


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


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def strip_prompt(text: str) -> str:
    value = str(text or "")
    for marker in ("Query:", "комментарий:"):
        if marker in value:
            value = value.split(marker, 1)[1]
    for prefix in RUSTS_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return normalize_text(value)


def good_text(text: str, min_chars: int, max_chars: int) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    toks = tokens(text)
    if len(toks) < 8:
        return False
    cyr = sum(1 for ch in text if "а" <= ch <= "я" or ch == "ё")
    return cyr >= max(12, int(len(text) * 0.25))


def light_variant(text: str, rng: random.Random) -> str | None:
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [part.strip() for part in parts if len(part.strip()) >= 20]
    if len(parts) >= 2:
        if rng.random() < 0.5:
            variant = " ".join(parts[:-1])
        else:
            variant = " ".join(parts[1:])
        return normalize_text(variant)

    toks = tokens(text)
    if len(toks) < 12:
        return None
    keep = max(8, int(len(toks) * 0.82))
    start = 0 if rng.random() < 0.5 else len(toks) - keep
    return normalize_text(" ".join(toks[start : start + keep]))


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
    metadata: dict[str, Any],
    eval_texts: set[str],
    min_chars: int,
    max_chars: int,
) -> None:
    left = normalize_text(left)
    right = normalize_text(right)
    if left == right:
        skipped["same_text"] += 1
        return
    if not good_text(left, min_chars, max_chars) or not good_text(right, min_chars, max_chars):
        skipped["quality"] += 1
        return
    if left in eval_texts or right in eval_texts:
        skipped["eval_text_exact_overlap"] += 1
        return
    key = tuple(sorted((left, right)))
    if key in seen_pairs:
        skipped["duplicate_pair"] += 1
        return
    seen_pairs.add(key)
    record = {
        "objective": "pair_score",
        "sentence1": left,
        "sentence2": right,
        "score": round(score, 4),
        "source": source,
        "metadata": {
            **metadata,
            "score_bucket": bucket,
            "contamination_policy": "Constructed from previously audited clean MixH data; exact normalized text overlap with mteb/RuSTSBenchmarkSTS and mteb/CEDRClassification removed.",
        },
    }
    buckets[bucket].append(record)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build controlled fair Russian graded-similarity pairs from audited clean MixH contrastive records."
    )
    parser.add_argument("--input", type=Path, default=Path("data/contrastive/open_ru_1r_nc_mixh_habrfull_geracl6400_habr4369_deepvk3200_grandmaster3200_17169.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/fair_rusts_controlled_mixh_similarity_3200_seed2291.jsonl"))
    parser.add_argument("--prompt-out", type=Path, default=Path("data/contrastive/fair_rusts_controlled_mixh_similarity_promptaligned_6400_seed2291.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/fair_rusts_controlled_mixh_similarity_3200_seed2291_summary.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--max-source-rows", type=int, default=12000)
    parser.add_argument("--max-records", type=int, default=3200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2291)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=520)
    args = parser.parse_args()

    if args.batch_size % 4 != 0:
        raise ValueError("--batch-size must be divisible by 4")

    rng = random.Random(args.seed)
    eval_texts = load_eval_texts(args.cache_dir)
    rows = read_jsonl(args.input, args.max_source_rows)
    rng.shuffle(rows)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    by_source: dict[str, list[str]] = defaultdict(list)
    extracted: list[tuple[int, str, str, str, list[str]]] = []

    for row_id, row in enumerate(rows):
        source = str(row.get("source") or "unknown")
        query = strip_prompt(row.get("query", ""))
        positive = strip_prompt(row.get("positive", ""))
        negatives = [strip_prompt(item) for item in (row.get("negatives") or [])[:2]]
        for text in [query, positive, *negatives]:
            if good_text(text, args.min_chars, args.max_chars):
                by_source[source].append(text)
        if good_text(query, args.min_chars, args.max_chars):
            extracted.append((row_id, source, query, positive, negatives))

    for row_id, source, query, positive, negatives in extracted:
        variant = light_variant(query, rng)
        if variant:
            add_record(
                buckets,
                seen_pairs,
                skipped,
                left=query,
                right=variant,
                score=0.95,
                bucket="high",
                source=f"{source}:controlled_self",
                metadata={"row_id": row_id, "construction": "light_self_augmentation"},
                eval_texts=eval_texts,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )
        add_record(
            buckets,
            seen_pairs,
            skipped,
            left=query,
            right=positive,
            score=0.72,
            bucket="mid_high",
            source=f"{source}:controlled_positive",
            metadata={"row_id": row_id, "construction": "original_query_positive"},
            eval_texts=eval_texts,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        for negative in negatives[:1]:
            add_record(
                buckets,
                seen_pairs,
                skipped,
                left=query,
                right=negative,
                score=0.05,
                bucket="low",
                source=f"{source}:controlled_negative",
                metadata={"row_id": row_id, "construction": "original_query_negative"},
                eval_texts=eval_texts,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )

    for source, texts in by_source.items():
        rng.shuffle(texts)
        for left, right in zip(texts[::2], texts[1::2]):
            add_record(
                buckets,
                seen_pairs,
                skipped,
                left=left,
                right=right,
                score=0.35,
                bucket="mid_low",
                source=f"{source}:controlled_same_source_random",
                metadata={"construction": "same_source_random_pair"},
                eval_texts=eval_texts,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
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

    prompt_records: list[dict[str, Any]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        if len(batch) < args.batch_size:
            break
        for prefix_index, prefix in enumerate(RUSTS_PREFIXES):
            for record in batch:
                new_record = dict(record)
                new_record["sentence1"] = prefix + record["sentence1"]
                new_record["sentence2"] = prefix + record["sentence2"]
                new_record["source"] = f"{record['source']}:rusts_prefix{prefix_index + 1}"
                metadata = dict(record.get("metadata") or {})
                metadata["prompt_alignment"] = "RuSTSBenchmarkSTS legacy_ru ensemble prefix"
                metadata["prefix_index"] = prefix_index
                new_record["metadata"] = metadata
                prompt_records.append(new_record)

    with args.prompt_out.open("w", encoding="utf-8") as file:
        for record in prompt_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.out),
        "prompt_output": str(args.prompt_out),
        "source_rows_loaded": len(rows),
        "source_rows_extracted": len(extracted),
        "records": len(records),
        "prompt_records": len(prompt_records),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "eval_texts": len(eval_texts),
        "bucket_available": {name: len(buckets[name]) for name in bucket_order},
        "bucket_used": used,
        "skipped": dict(skipped),
        "construction": {
            "high": "query vs light self-augmentation, score 0.95",
            "mid_high": "original clean MixH query-positive, score 0.72",
            "mid_low": "random same-source pair, score 0.35",
            "low": "original clean MixH query-negative, score 0.05",
        },
        "fairness": "Uses previously audited clean MixH records only; exact normalized RuSTS/CEDR text overlap removed; no released model outputs or benchmark rows.",
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
