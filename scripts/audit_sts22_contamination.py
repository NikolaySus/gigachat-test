from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from giga_model_utils import ModelLoadConfig, encode_texts, load_giga_embeddings, write_json


PROMPT_RE = re.compile(r"^Instruct:.*?\nQuery:\s*", re.S)
TOKEN_RE = re.compile(r"[\w]+", re.U)


def normalize_text(value: Any) -> str:
    text = PROMPT_RE.sub("", str(value))
    text = text.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def token_set(value: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize_text(value)))


def snippet(value: str, limit: int = 220) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def load_train_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source = str(record.get("source", "unknown"))
            objective = str(record.get("objective", "contrastive"))
            source_counts[source] += 1

            if objective == "pair_score":
                fields = [("sentence1", record["sentence1"]), ("sentence2", record["sentence2"])]
                pair_texts = [normalize_text(record["sentence1"]), normalize_text(record["sentence2"])]
            elif objective == "contrastive":
                fields = [("query", record["query"]), ("positive", record["positive"])]
                fields.extend((f"negative_{idx}", text) for idx, text in enumerate(record.get("negatives", [])))
                pair_texts = [normalize_text(record["query"]), normalize_text(record["positive"])]
            else:
                continue

            for field, raw_text in fields:
                text = normalize_text(raw_text)
                if text:
                    records.append(
                        {
                            "line_no": line_no,
                            "source": source,
                            "objective": objective,
                            "field": field,
                            "text": text,
                        }
                    )

            if all(pair_texts):
                pairs.append(
                    {
                        "line_no": line_no,
                        "source": source,
                        "objective": objective,
                        "pair_key": tuple(sorted(pair_texts)),
                    }
                )

    return records, pairs, source_counts


def load_sts22_ru_test(cache_dir: Path) -> list[dict[str, Any]]:
    dataset = load_dataset("mteb/sts22-crosslingual-sts", cache_dir=str(cache_dir))
    return [
        {
            "row_index": idx,
            "id": str(row["id"]),
            "score": float(row["score"]),
            "sentence1": normalize_text(row["sentence1"]),
            "sentence2": normalize_text(row["sentence2"]),
        }
        for idx, row in enumerate(dataset["test"])
        if row.get("lang") == "ru"
    ]


def exact_and_containment_matches(
    train_texts: list[dict[str, Any]],
    train_pairs: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in train_texts:
        train_by_text[item["text"]].append(item)

    train_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in train_pairs:
        train_by_pair[item["pair_key"]].append(item)

    exact_text: list[dict[str, Any]] = []
    exact_pair: list[dict[str, Any]] = []
    containment: list[dict[str, Any]] = []

    long_train = [item for item in train_texts if len(item["text"]) >= 80]
    for row in eval_rows:
        eval_fields = [("sentence1", row["sentence1"]), ("sentence2", row["sentence2"])]
        pair_key = tuple(sorted([row["sentence1"], row["sentence2"]]))
        for match in train_by_pair.get(pair_key, []):
            exact_pair.append(
                {
                    "sts22_id": row["id"],
                    "sts22_row_index": row["row_index"],
                    "score": row["score"],
                    "train_line_no": match["line_no"],
                    "train_source": match["source"],
                    "train_objective": match["objective"],
                }
            )

        for eval_field, eval_text in eval_fields:
            for match in train_by_text.get(eval_text, []):
                exact_text.append(
                    {
                        "sts22_id": row["id"],
                        "sts22_row_index": row["row_index"],
                        "sts22_field": eval_field,
                        "train_line_no": match["line_no"],
                        "train_source": match["source"],
                        "train_objective": match["objective"],
                        "train_field": match["field"],
                        "text": snippet(eval_text),
                    }
                )

            if len(eval_text) < 80:
                continue
            for train_item in long_train:
                train_text = train_item["text"]
                if eval_text == train_text:
                    continue
                if eval_text in train_text or train_text in eval_text:
                    containment.append(
                        {
                            "sts22_id": row["id"],
                            "sts22_row_index": row["row_index"],
                            "sts22_field": eval_field,
                            "train_line_no": train_item["line_no"],
                            "train_source": train_item["source"],
                            "train_field": train_item["field"],
                            "eval_length": len(eval_text),
                            "train_length": len(train_text),
                            "eval_text": snippet(eval_text),
                            "train_text": snippet(train_text),
                        }
                    )
                    break

    return exact_text, exact_pair, containment


def lexical_near_matches(
    train_texts: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    thresholds: list[float],
    max_matches: int,
) -> list[dict[str, Any]]:
    min_threshold = min(thresholds)
    indexed: list[tuple[dict[str, Any], set[str]]] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    for item in train_texts:
        tokens = token_set(item["text"])
        if len(tokens) < 5:
            continue
        indexed.append((item, tokens))
        index = len(indexed) - 1
        for token in tokens:
            if len(token) >= 5:
                inverted[token].append(index)

    matches: list[dict[str, Any]] = []
    for row in eval_rows:
        for eval_field in ("sentence1", "sentence2"):
            eval_text = row[eval_field]
            eval_tokens = token_set(eval_text)
            if len(eval_tokens) < 5:
                continue

            candidate_counts: dict[int, int] = defaultdict(int)
            for token in eval_tokens:
                if len(token) < 5:
                    continue
                for index in inverted.get(token, []):
                    candidate_counts[index] += 1

            for index, _ in sorted(candidate_counts.items(), key=lambda item: -item[1])[:500]:
                train_item, train_tokens = indexed[index]
                intersection = len(eval_tokens & train_tokens)
                union = len(eval_tokens | train_tokens)
                jaccard = intersection / union if union else 0.0
                if jaccard < min_threshold:
                    continue
                matches.append(
                    {
                        "jaccard": round(jaccard, 6),
                        "threshold_bucket": max(threshold for threshold in thresholds if jaccard >= threshold),
                        "sts22_id": row["id"],
                        "sts22_row_index": row["row_index"],
                        "sts22_field": eval_field,
                        "train_line_no": train_item["line_no"],
                        "train_source": train_item["source"],
                        "train_objective": train_item["objective"],
                        "train_field": train_item["field"],
                        "eval_text": snippet(eval_text),
                        "train_text": snippet(train_item["text"]),
                    }
                )

    matches.sort(key=lambda item: item["jaccard"], reverse=True)
    return matches[:max_matches]


def semantic_near_matches(
    train_texts: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_length: int,
    threshold: float,
    top_k: int,
    local_files_only: bool,
    attn_implementation: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique_train: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in train_texts:
        unique_train[item["text"]].append(item)
    train_values = list(unique_train)

    eval_items: list[dict[str, Any]] = []
    for row in eval_rows:
        eval_items.append({"row": row, "field": "sentence1", "text": row["sentence1"]})
        eval_items.append({"row": row, "field": "sentence2", "text": row["sentence2"]})

    tokenizer, model = load_giga_embeddings(
        ModelLoadConfig(
            batch_size=batch_size,
            max_length=max_length,
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
        )
    )
    eval_embeddings = encode_texts(
        [item["text"] for item in eval_items],
        tokenizer,
        model,
        batch_size=batch_size,
        max_length=max_length,
    )
    train_embeddings = encode_texts(
        train_values,
        tokenizer,
        model,
        batch_size=batch_size,
        max_length=max_length,
    )

    matches: list[dict[str, Any]] = []
    top_candidates_by_eval: list[list[dict[str, Any]]] = [[] for _ in eval_items]
    chunk_size = 2048
    for start in range(0, len(train_values), chunk_size):
        end = min(start + chunk_size, len(train_values))
        scores = eval_embeddings @ train_embeddings[start:end].T
        for eval_index in range(scores.shape[0]):
            row_scores = scores[eval_index]
            if top_k < len(row_scores):
                candidate_indexes = np.argpartition(row_scores, -top_k)[-top_k:]
            else:
                candidate_indexes = np.arange(len(row_scores))
            for local_index in candidate_indexes:
                score = float(row_scores[local_index])
                train_text = train_values[start + int(local_index)]
                train_item = unique_train[train_text][0]
                eval_item = eval_items[eval_index]
                candidate = {
                    "cosine": round(score, 6),
                    "sts22_id": eval_item["row"]["id"],
                    "sts22_row_index": eval_item["row"]["row_index"],
                    "sts22_field": eval_item["field"],
                    "train_line_no": train_item["line_no"],
                    "train_source": train_item["source"],
                    "train_objective": train_item["objective"],
                    "train_field": train_item["field"],
                    "eval_text": snippet(eval_item["text"]),
                    "train_text": snippet(train_text),
                }
                top_candidates_by_eval[eval_index].append(candidate)
                top_candidates_by_eval[eval_index].sort(key=lambda item: item["cosine"], reverse=True)
                del top_candidates_by_eval[eval_index][top_k:]
                if score >= threshold:
                    matches.append(candidate)

    matches.sort(key=lambda item: item["cosine"], reverse=True)
    top_candidates = [
        candidate
        for candidates in top_candidates_by_eval
        for candidate in candidates[:top_k]
    ]
    top_candidates.sort(key=lambda item: item["cosine"], reverse=True)
    return matches[: max(top_k * len(eval_items), top_k)], top_candidates[: max(top_k * len(eval_items), top_k)]


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# STS22 Contamination Audit",
        "",
        f"Training data: `{summary['train_path']}`",
        f"STS22 split: `{summary['sts22_split']}`",
        f"STS22 language: `{summary['sts22_lang']}`",
        "",
        "## Counts",
        "",
        f"- Train records: `{summary['train_record_count']}`",
        f"- Train unique texts: `{summary['train_unique_text_count']}`",
        f"- Train pairs: `{summary['train_pair_count']}`",
        f"- STS22 eval pairs: `{summary['sts22_pair_count']}`",
        "",
        "## Match Counts",
        "",
        f"- Exact pair matches: `{summary['match_counts']['exact_pair']}`",
        f"- Exact text matches: `{summary['match_counts']['exact_text']}`",
        f"- Containment matches: `{summary['match_counts']['containment']}`",
        f"- Lexical near matches: `{summary['match_counts']['lexical_near']}`",
        f"- Semantic near matches: `{summary['match_counts']['semantic_near']}`",
        "",
        "## Source Counts",
        "",
    ]
    for source, count in summary["train_sources"].items():
        lines.append(f"- `{source}`: `{count}`")

    lines.extend(["", "## Conclusion", "", summary["conclusion"], ""])

    for key, title, score_key in [
        ("lexical_near", "Top Lexical Matches", "jaccard"),
        ("semantic_near", "Top Semantic Matches Above Threshold", "cosine"),
        ("semantic_top_candidates", "Nearest Semantic Candidates", "cosine"),
    ]:
        matches = summary["matches"][key][:10]
        if not matches:
            continue
        lines.extend(["", f"## {title}", ""])
        for match in matches:
            lines.extend(
                [
                    f"- `{score_key}={match[score_key]}` STS22 `{match['sts22_id']}` "
                    f"{match['sts22_field']} vs train line `{match['train_line_no']}` "
                    f"source `{match['train_source']}`",
                    f"  - STS22: {match['eval_text']}",
                    f"  - Train: {match['train_text']}",
                ]
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit STS training data against STS22 Russian test.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/mteb_cache/datasets"))
    parser.add_argument("--lexical-thresholds", type=float, nargs="+", default=[0.70, 0.80, 0.90])
    parser.add_argument("--max-lexical-matches", type=int, default=200)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--semantic-threshold", type=float, default=0.92)
    parser.add_argument("--semantic-top-k", type=int, default=5)
    parser.add_argument("--semantic-max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str((Path("results/mteb_cache") / "hf_home").resolve()))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.cache_dir.resolve()))

    train_texts, train_pairs, source_counts = load_train_records(args.train_jsonl)
    eval_rows = load_sts22_ru_test(args.cache_dir)

    exact_text, exact_pair, containment = exact_and_containment_matches(train_texts, train_pairs, eval_rows)
    lexical = lexical_near_matches(
        train_texts,
        eval_rows,
        sorted(args.lexical_thresholds),
        args.max_lexical_matches,
    )
    semantic: list[dict[str, Any]] = []
    semantic_top_candidates: list[dict[str, Any]] = []
    if args.semantic:
        semantic, semantic_top_candidates = semantic_near_matches(
            train_texts,
            eval_rows,
            batch_size=args.batch_size,
            max_length=args.semantic_max_length,
            threshold=args.semantic_threshold,
            top_k=args.semantic_top_k,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )

    strong_matches = len(exact_pair) + len(exact_text) + len(containment)
    if strong_matches:
        conclusion = "Strong direct overlap was found. Treat STS22 as contaminated for this training mix."
    elif args.semantic and semantic:
        conclusion = (
            "No exact, containment, or lexical near-duplicate overlap was found. "
            "Review semantic-near matches manually; if they are topical only, STS22 remains probably clean."
        )
    elif args.semantic:
        conclusion = (
            "No exact, containment, lexical near-duplicate, or semantic-near overlap above threshold was found. "
            "STS22 is probably clean for this training mix; validate the small score gain with another clean task."
        )
    else:
        conclusion = (
            "No exact, containment, or lexical near-duplicate overlap was found. "
            "Run with --semantic for the stronger paraphrase-level check."
        )

    unique_text_count = len({item["text"] for item in train_texts})
    summary = {
        "train_path": str(args.train_jsonl),
        "sts22_split": "test",
        "sts22_lang": "ru",
        "train_record_count": int(sum(source_counts.values())),
        "train_unique_text_count": unique_text_count,
        "train_pair_count": len(train_pairs),
        "train_unique_pair_count": len({item["pair_key"] for item in train_pairs}),
        "sts22_pair_count": len(eval_rows),
        "train_sources": dict(sorted(source_counts.items())),
        "settings": {
            "lexical_thresholds": sorted(args.lexical_thresholds),
            "semantic": bool(args.semantic),
            "semantic_threshold": args.semantic_threshold,
            "semantic_top_k": args.semantic_top_k,
            "semantic_max_length": args.semantic_max_length,
        },
        "match_counts": {
            "exact_pair": len(exact_pair),
            "exact_text": len(exact_text),
            "containment": len(containment),
            "lexical_near": len(lexical),
            "semantic_near": len(semantic),
        },
        "matches": {
            "exact_pair": exact_pair,
            "exact_text": exact_text,
            "containment": containment,
            "lexical_near": lexical,
            "semantic_near": semantic,
            "semantic_top_candidates": semantic_top_candidates,
        },
        "conclusion": conclusion,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
