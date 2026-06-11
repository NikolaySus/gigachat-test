from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def strip_prompt(text: str) -> str:
    marker = "\nQuery: "
    if marker in text:
        return text.split(marker, 1)[1]
    return text


def score_pairs(
    pairs: list[dict[str, Any]],
    *,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    scored: list[dict[str, Any]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [item["sentence1"] for item in batch],
            [item["sentence2"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
            probabilities = F.softmax(logits, dim=-1)
            paraphrase_scores = probabilities[:, 1].detach().cpu().tolist()
        for item, score in zip(batch, paraphrase_scores, strict=True):
            scored.append({**item, "open_teacher_score": float(score)})
    return scored


def bucket_name(score: float) -> str | None:
    if score <= 0.12:
        return "low"
    if 0.25 <= score < 0.70:
        return "mid"
    if 0.70 <= score < 0.90:
        return "high"
    if score >= 0.90:
        return "very_high"
    return None


def ordered_balanced(
    records: list[dict[str, Any]],
    *,
    max_records: int,
    batch_size: int,
    seed: int,
    imbalanced_fill: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if batch_size % 4 != 0:
        raise ValueError("batch size must be divisible by 4")
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for record in records:
        bucket = bucket_name(record["open_teacher_score"])
        if bucket is None:
            skipped["score_gap"] += 1
            continue
        normalized = {
            "objective": "pair_score",
            "sentence1": record["sentence1"],
            "sentence2": record["sentence2"],
            "score": round(record["open_teacher_score"], 6),
            "source": "open_teacher_paraphrase_scored",
            "metadata": {
                **record.get("metadata", {}),
                "source": record.get("source"),
                "original_score": record.get("score"),
                "score_bucket": bucket,
                "open_teacher": record.get("open_teacher"),
                "open_teacher_score": round(record["open_teacher_score"], 6),
                "fairness_note": "Open-source paraphrase classifier teacher; not released Giga model and not benchmark rows.",
            },
        }
        buckets[bucket].append(normalized)

    for values in buckets.values():
        rng.shuffle(values)

    order = ("low", "mid", "high", "very_high")
    per_bucket = batch_size // 4
    used = {name: 0 for name in order}
    output: list[dict[str, Any]] = []
    while len(output) + batch_size <= max_records:
        batch: list[dict[str, Any]] = []
        for name in order:
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
        output.extend(batch)
    if imbalanced_fill and len(output) + batch_size <= max_records:
        while len(output) + batch_size <= max_records:
            batch = []
            for name, count in (("low", batch_size // 2), ("very_high", batch_size - batch_size // 2)):
                start = used[name]
                end = start + count
                if end > len(buckets[name]):
                    batch = []
                    break
                batch.extend(buckets[name][start:end])
                used[name] = end
            if not batch:
                break
            rng.shuffle(batch)
            output.extend(batch)
    return output, {
        "bucket_available": {name: len(buckets[name]) for name in order},
        "bucket_used": used,
        "skipped": dict(skipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore fair paraphrase pair-score records with an open-source Russian paraphrase classifier."
    )
    parser.add_argument("--input", type=Path, default=Path("data/contrastive/fair_rusts_paraphrase_pairscore_balanced_6400_seed2071.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/fair_rusts_open_teacher_paraphrase_balanced_3200_seed2081.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/fair_rusts_open_teacher_paraphrase_balanced_3200_seed2081_summary.json"))
    parser.add_argument("--model-name", default="s-nlp/ruRoberta-large-paraphrase-v1")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--score-batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=3200)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--imbalanced-fill", action="store_true")
    parser.add_argument("--seed", type=int, default=2081)
    args = parser.parse_args()

    raw_records = read_jsonl(args.input)
    pairs = []
    for record in raw_records:
        if record.get("objective") != "pair_score":
            continue
        pairs.append(
            {
                "sentence1": strip_prompt(str(record["sentence1"])),
                "sentence2": strip_prompt(str(record["sentence2"])),
                "score": record.get("score"),
                "source": record.get("source"),
                "metadata": record.get("metadata", {}),
                "open_teacher": args.model_name,
            }
        )
    scored = score_pairs(
        pairs,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.score_batch_size,
        max_length=args.max_length,
    )
    output, balance_summary = ordered_balanced(
        scored,
        max_records=args.max_records,
        batch_size=args.train_batch_size,
        seed=args.seed,
        imbalanced_fill=args.imbalanced_fill,
    )
    write_jsonl(args.out, output)
    scores = [item["open_teacher_score"] for item in scored]
    selected_scores = [item["score"] for item in output]
    summary = {
        "input": str(args.input),
        "output": str(args.out),
        "model_name": args.model_name,
        "raw_records": len(raw_records),
        "scored_pairs": len(scored),
        "written_records": len(output),
        "score_min_all": min(scores) if scores else None,
        "score_mean_all": sum(scores) / len(scores) if scores else None,
        "score_max_all": max(scores) if scores else None,
        "score_min_selected": min(selected_scores) if selected_scores else None,
        "score_mean_selected": sum(selected_scores) / len(selected_scores) if selected_scores else None,
        "score_max_selected": max(selected_scores) if selected_scores else None,
        **balance_summary,
        "fairness": (
            "The teacher is an open-source Russian paraphrase classifier. The released Giga embedding model "
            "is not used for scoring, distillation, or regularization. Input records were already exact-overlap "
            "filtered against mteb/RuSTSBenchmarkSTS and mteb/CEDRClassification."
        ),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
