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
NUMBER_RE = re.compile(r"\b\d+(?:[,.]\d+)?\b")
RUSTS_PREFIXES = [
    "семантически похожий текст: ",
    "семантически похожий текст \nтекст: ",
]
NEGATION_PATTERNS = [
    (re.compile(r"\bявляется\b"), "не является"),
    (re.compile(r"\bбыл\b"), "не был"),
    (re.compile(r"\bбыла\b"), "не была"),
    (re.compile(r"\bбыли\b"), "не были"),
    (re.compile(r"\bнаходится\b"), "не находится"),
    (re.compile(r"\bполучил\b"), "не получил"),
    (re.compile(r"\bполучила\b"), "не получила"),
    (re.compile(r"\bимеет\b"), "не имеет"),
    (re.compile(r"\bможет\b"), "не может"),
]


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalized_key(value: Any) -> str:
    return normalize_text(value).lower()


def token_count(value: str) -> int:
    return len(TOKEN_RE.findall(value))


def is_good_sentence(value: str) -> bool:
    value = normalize_text(value)
    if not (35 <= len(value) <= 260):
        return False
    if token_count(value) < 6:
        return False
    cyr = sum(1 for char in value.lower() if "а" <= char <= "я" or char == "е")
    return cyr >= max(12, int(len(value) * 0.35))


def load_eval_texts(cache_dir: Path) -> set[str]:
    texts: set[str] = set()
    for dataset_name in ("mteb/RuSTSBenchmarkSTS", "mteb/CEDRClassification"):
        dataset = load_dataset(dataset_name, cache_dir=str(cache_dir), trust_remote_code=True)
        for split in dataset:
            for row in dataset[split]:
                for key in ("sentence1", "sentence2", "text"):
                    if key in row:
                        text = normalized_key(row[key])
                        if text:
                            texts.add(text)
    return texts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def number_variant(text: str, rng: random.Random) -> str | None:
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    old = rng.choice(matches)
    try:
        old_float = float(old.replace(",", "."))
    except ValueError:
        return None
    candidates = [
        str(max(1, int(old_float) + delta))
        for delta in (-7, -3, -1, 1, 2, 5, 10)
        if int(old_float) + delta > 0
    ]
    if not candidates:
        return None
    new = rng.choice(candidates)
    return NUMBER_RE.sub(new, text, count=1)


def negation_variant(text: str) -> str | None:
    lowered = text.lower()
    if " не " in f" {lowered} ":
        return re.sub(r"\bне\s+", "", text, count=1, flags=re.I)
    for pattern, replacement in NEGATION_PATTERNS:
        if pattern.search(lowered):
            return pattern.sub(replacement, text, count=1)
    return None


def crop_variant(text: str) -> str | None:
    tokens = TOKEN_RE.findall(text)
    if len(tokens) < 12:
        return None
    keep = max(7, int(len(tokens) * 0.68))
    cropped = " ".join(tokens[:keep])
    if cropped == text or len(cropped) < 35:
        return None
    return cropped


def add_pair(
    buckets: dict[str, list[dict[str, Any]]],
    seen: set[tuple[str, str]],
    skipped: Counter[str],
    *,
    left: str,
    right: str,
    score: float,
    bucket: str,
    source: str,
    eval_texts: set[str],
    metadata: dict[str, Any],
) -> None:
    left = normalize_text(left)
    right = normalize_text(right)
    left_key = normalized_key(left)
    right_key = normalized_key(right)
    if left_key == right_key:
        skipped["same"] += 1
        return
    if left_key in eval_texts or right_key in eval_texts:
        skipped["eval_overlap"] += 1
        return
    if not is_good_sentence(left) or not is_good_sentence(right):
        skipped["quality"] += 1
        return
    key = tuple(sorted((left_key, right_key)))
    if key in seen:
        skipped["duplicate"] += 1
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
                "contamination_policy": (
                    "Constructed from existing clean project pair sources; exact normalized text overlap "
                    "with mteb/RuSTSBenchmarkSTS and mteb/CEDRClassification removed."
                ),
            },
        }
    )


def prompt_align(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    for record in records:
        for prefix_index, prefix in enumerate(RUSTS_PREFIXES):
            item = dict(record)
            item["sentence1"] = prefix + str(record["sentence1"])
            item["sentence2"] = prefix + str(record["sentence2"])
            metadata = dict(record.get("metadata") or {})
            metadata["prompt_alignment"] = "RuSTSBenchmarkSTS legacy_ru ensemble prefix"
            metadata["prefix_index"] = prefix_index
            item["metadata"] = metadata
            aligned.append(item)
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean RuSTS trap-style graded pair-score records.")
    parser.add_argument("--paraphrase", type=Path, default=Path("data/contrastive/fair_rusts_paraphrase_pairscore_balanced_6400_seed2071.jsonl"))
    parser.add_argument("--deepvk", type=Path, default=Path("data/contrastive/fair_rusts_deepvk_hnp_plain_pairscore_9000.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/fair_rusts_trap_pairs_3200_seed2351.jsonl"))
    parser.add_argument("--prompt-out", type=Path, default=Path("data/contrastive/fair_rusts_trap_pairs_promptaligned_6400_seed2351.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/fair_rusts_trap_pairs_3200_seed2351_summary.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--records-per-bucket", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2351)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    eval_texts = load_eval_texts(args.cache_dir)
    rows = read_jsonl(args.paraphrase) + read_jsonl(args.deepvk)
    rng.shuffle(rows)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()

    for row in rows:
        if row.get("objective") != "pair_score":
            continue
        left = normalize_text(row.get("sentence1", ""))
        right = normalize_text(row.get("sentence2", ""))
        score = float(row.get("score", 0.0))
        source = str(row.get("source", "unknown"))
        if score >= 0.9:
            add_pair(
                buckets,
                seen,
                skipped,
                left=left,
                right=right,
                score=0.96,
                bucket="high",
                source=f"{source}:original_high",
                eval_texts=eval_texts,
                metadata={"construction": "original_high_pair", "original_score": score},
            )
            cropped = crop_variant(left)
            if cropped:
                add_pair(
                    buckets,
                    seen,
                    skipped,
                    left=left,
                    right=cropped,
                    score=0.72,
                    bucket="mid_high",
                    source=f"{source}:crop",
                    eval_texts=eval_texts,
                    metadata={"construction": "same_sentence_crop"},
                )
        elif 0.45 <= score <= 0.75:
            add_pair(
                buckets,
                seen,
                skipped,
                left=left,
                right=right,
                score=0.58,
                bucket="mid_high",
                source=f"{source}:original_mid",
                eval_texts=eval_texts,
                metadata={"construction": "original_mid_pair", "original_score": score},
            )
        elif score <= 0.05:
            add_pair(
                buckets,
                seen,
                skipped,
                left=left,
                right=right,
                score=0.04,
                bucket="low",
                source=f"{source}:original_low",
                eval_texts=eval_texts,
                metadata={"construction": "original_low_pair", "original_score": score},
            )

        for maker, bucket, trap_score, name in (
            (number_variant, "mid_low", 0.42, "number_swap"),
            (lambda text, _rng: negation_variant(text), "low", 0.12, "negation_flip"),
        ):
            variant = maker(left, rng)
            if variant:
                add_pair(
                    buckets,
                    seen,
                    skipped,
                    left=left,
                    right=variant,
                    score=trap_score,
                    bucket=bucket,
                    source=f"{source}:{name}",
                    eval_texts=eval_texts,
                    metadata={"construction": name},
                )

    for values in buckets.values():
        rng.shuffle(values)

    bucket_order = ("low", "mid_low", "mid_high", "high")
    selected: list[dict[str, Any]] = []
    bucket_used: dict[str, int] = {}
    for bucket in bucket_order:
        take = min(args.records_per_bucket, len(buckets[bucket]))
        selected.extend(buckets[bucket][:take])
        bucket_used[bucket] = take
    rng.shuffle(selected)

    write_jsonl(args.out, selected)
    prompt_records = prompt_align(selected)
    write_jsonl(args.prompt_out, prompt_records)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(
            {
                "output": str(args.out),
                "prompt_output": str(args.prompt_out),
                "records": len(selected),
                "prompt_records": len(prompt_records),
                "bucket_available": {bucket: len(buckets[bucket]) for bucket in bucket_order},
                "bucket_used": bucket_used,
                "skipped": dict(skipped),
                "seed": args.seed,
                "fairness": "No benchmark rows or released-model scores; exact RuSTS/CEDR text overlap removed.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
