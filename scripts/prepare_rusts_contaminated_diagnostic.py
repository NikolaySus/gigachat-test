from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


LEGACY_RU_PREFIXES = [
    "семантически похожий текст: ",
    "семантически похожий текст \nтекст: ",
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def bucket(score: float) -> str:
    if score < 0.2:
        return "low"
    if score < 0.4:
        return "mid_low"
    if score < 0.6:
        return "mid"
    if score < 0.8:
        return "mid_high"
    return "high"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deliberately contaminated RuSTS pair-score diagnostic data."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--prompt-aligned-output", type=Path)
    parser.add_argument("--dataset", default="mteb/RuSTSBenchmarkSTS")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        for index, item in enumerate(dataset[split]):
            score = float(item["score"]) / 5.0
            record = {
                "objective": "pair_score",
                "source": f"{args.dataset}:{split}:CONTAMINATED_DIAGNOSTIC_ONLY",
                "sentence1": str(item["sentence1"]).strip(),
                "sentence2": str(item["sentence2"]).strip(),
                "score": round(score, 6),
                "metadata": {
                    "split": split,
                    "row_index": index,
                    "raw_score": float(item["score"]),
                    "score_bucket": bucket(score),
                    "contamination": (
                        "Direct mteb/RuSTSBenchmarkSTS row. Diagnostic only; "
                        "forbidden for fair training."
                    ),
                },
            }
            rows.append(record)
            split_counts[split] += 1
            bucket_counts[bucket(score)] += 1

    write_jsonl(args.output, rows)

    prompt_rows: list[dict[str, Any]] = []
    if args.prompt_aligned_output:
        for row in rows:
            for prefix in LEGACY_RU_PREFIXES:
                prompt_row = dict(row)
                prompt_row["sentence1"] = prefix + row["sentence1"].replace("ё", "е").replace("Ё", "Е")
                prompt_row["sentence2"] = prefix + row["sentence2"].replace("ё", "е").replace("Ё", "Е")
                metadata = dict(row["metadata"])
                metadata["prompt_prefix"] = prefix
                prompt_row["metadata"] = metadata
                prompt_rows.append(prompt_row)
        write_jsonl(args.prompt_aligned_output, prompt_rows)

    summary = {
        "dataset": args.dataset,
        "output": str(args.output),
        "records": len(rows),
        "split_counts": dict(split_counts),
        "bucket_counts": dict(bucket_counts),
        "prompt_aligned_output": str(args.prompt_aligned_output) if args.prompt_aligned_output else None,
        "prompt_aligned_records": len(prompt_rows),
        "fairness": (
            "CONTAMINATED_DIAGNOSTIC_ONLY. Uses exact RuSTS benchmark rows, "
            "including test rows. It can diagnose objective/data mismatch but "
            "cannot be used for fair model selection."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
