from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


RETRIEVAL_PROMPT = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
SOURCE = "Vikhrmodels/habr_qa_sbs:filtered"

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def clean_text(value) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalized(text: str) -> str:
    return clean_text(text).lower()


def lexical_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalized(left), normalized(right)).ratio()


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def rejection_reason(
    question: str,
    best: str,
    bad: str,
    *,
    min_question_chars: int,
    min_best_chars: int,
    min_bad_chars: int,
    min_best_words: int,
    min_bad_words: int,
    max_answer_chars: int,
    min_best_bad_similarity: float,
    max_best_bad_similarity: float,
) -> str | None:
    if not question or not best or not bad:
        return "empty_field"
    if len(question) < min_question_chars:
        return "short_question"
    if len(best) < min_best_chars:
        return "short_best"
    if len(bad) < min_bad_chars:
        return "short_bad"
    if word_count(best) < min_best_words:
        return "few_best_words"
    if word_count(bad) < min_bad_words:
        return "few_bad_words"
    if len(best) > max_answer_chars or len(bad) > max_answer_chars:
        return "long_answer"
    if normalized(best) == normalized(bad):
        return "same_best_bad"
    similarity = lexical_similarity(best, bad)
    if similarity < min_best_bad_similarity:
        return "easy_best_bad"
    if similarity >= max_best_bad_similarity:
        return "similar_best_bad"
    return None


def build_records(
    *,
    limit: int,
    seed: int,
    min_question_chars: int,
    min_best_chars: int,
    min_bad_chars: int,
    min_best_words: int,
    min_bad_words: int,
    max_answer_chars: int,
    min_best_bad_similarity: float,
    max_best_bad_similarity: float,
) -> tuple[list[dict], dict]:
    dataset = load_dataset("Vikhrmodels/habr_qa_sbs", split="train", streaming=True)
    records = []
    rejection_counts: dict[str, int] = {}
    seen = set()
    raw_rows = 0

    for row in dataset:
        raw_rows += 1
        question = clean_text(row.get("question"))
        best = clean_text(row.get("best"))
        bad = clean_text(row.get("bad"))
        reason = rejection_reason(
            question,
            best,
            bad,
            min_question_chars=min_question_chars,
            min_best_chars=min_best_chars,
            min_bad_chars=min_bad_chars,
            min_best_words=min_best_words,
            min_bad_words=min_bad_words,
            max_answer_chars=max_answer_chars,
            min_best_bad_similarity=min_best_bad_similarity,
            max_best_bad_similarity=max_best_bad_similarity,
        )
        if reason is not None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        key = (normalized(question), normalized(best), normalized(bad))
        if key in seen:
            rejection_counts["duplicate_triplet"] = rejection_counts.get("duplicate_triplet", 0) + 1
            continue
        seen.add(key)

        records.append(
            {
                "source": SOURCE,
                "query": RETRIEVAL_PROMPT + question,
                "positive": best[:max_answer_chars],
                "negatives": [bad[:max_answer_chars]],
                "metadata": {
                    "best_bad_similarity": round(lexical_similarity(best, bad), 6),
                    "question_chars": len(question),
                    "best_chars": len(best),
                    "bad_chars": len(bad),
                },
            }
        )

    rng = random.Random(seed)
    rng.shuffle(records)
    records = records[:limit]
    summary = {
        "source": SOURCE,
        "raw_rows": raw_rows,
        "records": len(records),
        "rejection_counts": rejection_counts,
        "filters": {
            "min_question_chars": min_question_chars,
            "min_best_chars": min_best_chars,
            "min_bad_chars": min_bad_chars,
            "min_best_words": min_best_words,
            "min_bad_words": min_bad_words,
            "max_answer_chars": max_answer_chars,
            "min_best_bad_similarity": min_best_bad_similarity,
            "max_best_bad_similarity": max_best_bad_similarity,
        },
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare STS-v15 recovery data from filtered Habr QA SBS.")
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/open_ru_sts_v15_habr_qa_sbs.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/open_ru_sts_v15_habr_qa_sbs_summary.json"))
    parser.add_argument("--limit", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-question-chars", type=int, default=20)
    parser.add_argument("--min-best-chars", type=int, default=80)
    parser.add_argument("--min-bad-chars", type=int, default=40)
    parser.add_argument("--min-best-words", type=int, default=8)
    parser.add_argument("--min-bad-words", type=int, default=5)
    parser.add_argument("--max-answer-chars", type=int, default=3500)
    parser.add_argument("--min-best-bad-similarity", type=float, default=0.0)
    parser.add_argument("--max-best-bad-similarity", type=float, default=0.88)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    records, summary = build_records(
        limit=args.limit,
        seed=args.seed,
        min_question_chars=args.min_question_chars,
        min_best_chars=args.min_best_chars,
        min_bad_chars=args.min_bad_chars,
        min_best_words=args.min_best_words,
        min_bad_words=args.min_bad_words,
        max_answer_chars=args.max_answer_chars,
        min_best_bad_similarity=args.min_best_bad_similarity,
        max_best_bad_similarity=args.max_best_bad_similarity,
    )
    write_jsonl(args.out, records)
    summary["output_path"] = str(args.out)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} records to {args.out}")
    print(f"Wrote summary to {args.summary_out}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
