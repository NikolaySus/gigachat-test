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


RUSTS_PREFIXES = [
    "семантически похожий текст: ",
    "семантически похожий текст \nтекст: ",
]
SENT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[\w]+", re.U)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("ё", "е").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    return normalize_text(value)


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def is_ru_sentence(text: str, min_chars: int, max_chars: int) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    if token_count(text) < 7:
        return False
    cyr = sum(1 for ch in text if "а" <= ch <= "я" or ch == "ё")
    return cyr >= max(10, int(len(text) * 0.35))


def sentences(text: str, min_chars: int, max_chars: int) -> list[str]:
    text = strip_prompt(text)
    parts = [normalize_text(part) for part in SENT_RE.split(text)]
    output: list[str] = []
    for part in parts:
        part = part.strip(" .!?;:,-")
        if is_ru_sentence(part, min_chars, max_chars):
            output.append(part)
    if not output and is_ru_sentence(text, min_chars, max_chars):
        output.append(text)
    return output


def crop_variant(sentence: str) -> str | None:
    toks = TOKEN_RE.findall(sentence)
    if len(toks) < 12:
        return None
    if len(toks) >= 18:
        keep = max(8, int(len(toks) * 0.72))
    else:
        keep = len(toks) - 3
    return normalize_text(" ".join(toks[:keep]))


def add_record(
    buckets: dict[str, list[dict[str, Any]]],
    seen: set[tuple[str, str]],
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
    if not is_ru_sentence(left, min_chars, max_chars) or not is_ru_sentence(right, min_chars, max_chars):
        skipped["quality"] += 1
        return
    if left in eval_texts or right in eval_texts:
        skipped["eval_text_exact_overlap"] += 1
        return
    key = tuple(sorted((left, right)))
    if key in seen:
        skipped["duplicate_pair"] += 1
        return
    seen.add(key)
    buckets[bucket].append(
        {
            "objective": "pair_score",
            "sentence1": left,
            "sentence2": right,
            "score": round(score, 4),
            "source": source,
            "metadata": {
                **metadata,
                "score_bucket": bucket,
                "contamination_policy": "Constructed from previously audited clean open-data records; exact normalized text overlap with mteb/RuSTSBenchmarkSTS and mteb/CEDRClassification removed.",
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build short Russian sentence-level graded similarity pairs.")
    parser.add_argument("--lenta", type=Path, default=Path("data/contrastive/open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_3200.jsonl"))
    parser.add_argument("--mixh", type=Path, default=Path("data/contrastive/open_ru_1r_nc_mixh_habrfull_geracl6400_habr4369_deepvk3200_grandmaster3200_17169.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/fair_rusts_sentence_similarity_2400_seed2301.jsonl"))
    parser.add_argument("--prompt-out", type=Path, default=Path("data/contrastive/fair_rusts_sentence_similarity_promptaligned_4800_seed2301.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/fair_rusts_sentence_similarity_2400_seed2301_summary.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--max-records", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--min-chars", type=int, default=45)
    parser.add_argument("--max-chars", type=int, default=240)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_texts = load_eval_texts(args.cache_dir)
    rows = read_jsonl(args.lenta) + read_jsonl(args.mixh, 6000)
    rng.shuffle(rows)

    seen: set[tuple[str, str]] = set()
    skipped: Counter[str] = Counter()
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[str]] = defaultdict(list)
    all_sentences: list[tuple[str, str]] = []

    for row_id, row in enumerate(rows):
        source = str(row.get("source") or "unknown")
        group = str((row.get("metadata") or {}).get("topic") or (row.get("metadata") or {}).get("group") or source)
        fields = [row.get("query", ""), row.get("positive", ""), *(row.get("negatives") or [])[:2]]
        local_sentences: list[str] = []
        for field in fields:
            for sent in sentences(str(field), args.min_chars, args.max_chars):
                local_sentences.append(sent)
                by_group[group].append(sent)
                all_sentences.append((group, sent))

        if len(local_sentences) >= 1:
            base = local_sentences[0]
            variant = crop_variant(base)
            if variant:
                add_record(
                    buckets,
                    seen,
                    skipped,
                    left=base,
                    right=variant,
                    score=0.92,
                    bucket="high",
                    source=f"{source}:sentence_crop",
                    metadata={"row_id": row_id, "construction": "sentence_crop"},
                    eval_texts=eval_texts,
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                )
        if len(local_sentences) >= 2:
            add_record(
                buckets,
                seen,
                skipped,
                left=local_sentences[0],
                right=local_sentences[1],
                score=0.68,
                bucket="mid_high",
                source=f"{source}:nearby_sentence",
                metadata={"row_id": row_id, "construction": "nearby_same_record_sentence"},
                eval_texts=eval_texts,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )

    for group, values in by_group.items():
        rng.shuffle(values)
        for left, right in zip(values[::2], values[1::2]):
            add_record(
                buckets,
                seen,
                skipped,
                left=left,
                right=right,
                score=0.38,
                bucket="mid_low",
                source=f"{group}:same_group_sentence",
                metadata={"construction": "same_group_random_sentence"},
                eval_texts=eval_texts,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )

    rng.shuffle(all_sentences)
    for i in range(0, len(all_sentences) - 1, 2):
        left_group, left = all_sentences[i]
        right_group, right = all_sentences[i + 1]
        if left_group == right_group:
            continue
        add_record(
            buckets,
            seen,
            skipped,
            left=left,
            right=right,
            score=0.08,
            bucket="low",
            source="cross_group_sentence",
            metadata={"construction": "cross_group_random_sentence", "left_group": left_group, "right_group": right_group},
            eval_texts=eval_texts,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )

    for values in buckets.values():
        rng.shuffle(values)

    bucket_order = ("low", "mid_low", "mid_high", "high")
    per_bucket_per_batch = args.batch_size // len(bucket_order)
    max_batches = min(args.max_records // args.batch_size, *(len(buckets[name]) // per_bucket_per_batch for name in bucket_order))
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
        "output": str(args.out),
        "prompt_output": str(args.prompt_out),
        "records": len(records),
        "prompt_records": len(prompt_records),
        "bucket_available": {name: len(buckets[name]) for name in bucket_order},
        "bucket_used": used,
        "skipped": dict(skipped),
        "eval_texts": len(eval_texts),
        "fairness": "Uses clean open-data source records only; exact RuSTS/CEDR text overlap removed; no released model outputs or benchmark rows.",
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
