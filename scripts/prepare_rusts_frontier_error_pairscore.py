from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.giga_model_utils import ModelLoadConfig, encode_texts, load_giga_embeddings


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_bucket(score: float) -> str:
    if score < 0.20:
        return "low"
    if score < 0.45:
        return "mid_low"
    if score < 0.70:
        return "mid"
    if score < 0.88:
        return "high"
    return "very_high"


def strip_prompt(text: str) -> str:
    marker = "\nQuery: "
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mine clean pair_score records where the current fair frontier "
            "disagrees with open-source graded STS labels."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--latent-checkpoint", type=Path, required=True)
    parser.add_argument("--max-input-records", type=int, default=32000)
    parser.add_argument("--records-per-bucket", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2191)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--strip-retrieval-prefix",
        action="store_true",
        help="Remove Instruct/Query prefix before scoring and writing pair_score records.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_rows = [row for row in read_jsonl(args.input) if row.get("objective") == "pair_score"]
    rng.shuffle(all_rows)
    rows = all_rows[: args.max_input_records]

    pairs: list[tuple[str, str]] = []
    labels: list[float] = []
    raw_rows: list[dict[str, Any]] = []
    for row in rows:
        left = str(row["sentence1"])
        right = str(row["sentence2"])
        if args.strip_retrieval_prefix:
            left = strip_prompt(left)
            right = strip_prompt(right)
        score = float(row["score"])
        if not (0.0 <= score <= 1.0):
            continue
        pairs.append((left, right))
        labels.append(score)
        raw_rows.append(row)

    tokenizer, model = load_giga_embeddings(
        ModelLoadConfig(
            max_length=args.max_length,
            batch_size=args.batch_size,
            attn_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
            latent_checkpoint=args.latent_checkpoint,
        )
    )

    unique_texts: list[str] = []
    text_to_index: dict[str, int] = {}
    for left, right in pairs:
        for text in (left, right):
            if text not in text_to_index:
                text_to_index[text] = len(unique_texts)
                unique_texts.append(text)

    embeddings = encode_texts(
        tqdm(unique_texts, desc="encoding texts"),
        tokenizer,
        model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    candidates_by_bucket: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    score_counter: Counter[str] = Counter()
    similarities: list[float] = []
    errors: list[float] = []
    for (left, right), label, source_row in zip(pairs, labels, raw_rows, strict=True):
        left_vec = embeddings[text_to_index[left]]
        right_vec = embeddings[text_to_index[right]]
        similarity = float(np.dot(left_vec, right_vec))
        normalized_similarity = (similarity + 1.0) / 2.0
        error = abs(normalized_similarity - label)
        bucket = score_bucket(label)
        score_counter[bucket] += 1
        similarities.append(normalized_similarity)
        errors.append(error)
        metadata = dict(source_row.get("metadata") or {})
        metadata.update(
            {
                "frontier_normalized_similarity": round(normalized_similarity, 6),
                "frontier_abs_error": round(error, 6),
                "score_bucket": bucket,
                "mining": "current_fair_frontier_label_disagreement",
                "base_source": source_row.get("source"),
            }
        )
        candidates_by_bucket[bucket].append(
            (
                error,
                {
                    "objective": "pair_score",
                    "source": f"{source_row.get('source', 'unknown')}:frontier_error_mined",
                    "sentence1": left,
                    "sentence2": right,
                    "score": round(label, 6),
                    "metadata": metadata,
                },
            )
        )

    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for bucket in ("low", "mid_low", "mid", "high", "very_high"):
        bucket_candidates = sorted(candidates_by_bucket[bucket], key=lambda item: item[0], reverse=True)
        chosen = [row for _error, row in bucket_candidates[: args.records_per_bucket]]
        selected_counts[bucket] = len(chosen)
        selected.extend(chosen)

    # Preserve bucket ladders inside adjacent batches for the rank/correlation losses.
    grouped = {bucket: [row for row in selected if row["metadata"]["score_bucket"] == bucket] for bucket in selected_counts}
    max_rounds = max((len(rows) for rows in grouped.values()), default=0)
    ordered: list[dict[str, Any]] = []
    for index in range(max_rounds):
        for bucket in ("low", "mid_low", "mid", "high", "very_high"):
            bucket_rows = grouped.get(bucket, [])
            if index < len(bucket_rows):
                ordered.append(bucket_rows[index])

    write_jsonl(args.output, ordered)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "latent_checkpoint": str(args.latent_checkpoint),
        "input_pair_score_records": len(all_rows),
        "scored_records": len(raw_rows),
        "output_records": len(ordered),
        "records_per_bucket": args.records_per_bucket,
        "available_by_bucket": dict(score_counter),
        "selected_by_bucket": dict(selected_counts),
        "frontier_similarity_mean": float(np.mean(similarities)) if similarities else None,
        "frontier_abs_error_mean": float(np.mean(errors)) if errors else None,
        "frontier_abs_error_p90": float(np.quantile(errors, 0.9)) if errors else None,
        "strip_retrieval_prefix": args.strip_retrieval_prefix,
        "fairness": (
            "Uses only open-source pair labels and the current fair frontier to "
            "mine hard examples; no released latent checkpoint or benchmark rows "
            "are used as training labels."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
