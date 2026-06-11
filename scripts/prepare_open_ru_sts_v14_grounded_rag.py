from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


RETRIEVAL_PROMPT = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
SOURCE = "Vikhrmodels/Grounded-RAG-RU-v2:good"


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(value) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def doc_text(doc: dict, *, max_chars: int) -> str:
    title = clean_text(doc.get("title"))
    content = clean_text(doc.get("content"))
    text = f"{title}. {content}" if title else content
    return text[:max_chars].strip()


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_conversation(row: dict) -> tuple[list[dict], str, list[int]] | None:
    by_role = {message.get("role"): message.get("content", "") for message in row["conversation"]}
    docs_raw = by_role.get("documents")
    question = clean_text(by_role.get("user"))
    if not docs_raw or not question:
        return None
    try:
        docs = json.loads(docs_raw)
    except json.JSONDecodeError:
        return None
    assistant_json = None
    for message in row["conversation"]:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", "")).strip()
        if content.startswith("{") and "relevant_doc_ids" in content:
            assistant_json = content
            break
    if not assistant_json:
        return None
    try:
        relevant_doc_ids = json.loads(assistant_json).get("relevant_doc_ids") or []
    except json.JSONDecodeError:
        return None
    relevant_doc_ids = [int(doc_id) for doc_id in relevant_doc_ids if isinstance(doc_id, int) or str(doc_id).isdigit()]
    if not docs or not relevant_doc_ids:
        return None
    return docs, question, relevant_doc_ids


def build_records(
    *,
    limit: int,
    negatives_per_record: int,
    random_negatives_per_record: int,
    doc_max_chars: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    dataset = load_dataset("Vikhrmodels/Grounded-RAG-RU-v2", split="train")
    parsed_rows = []
    random_doc_pool: list[str] = []
    skipped = 0

    for row in dataset:
        if row.get("type") != "good":
            continue
        parsed = parse_conversation(row)
        if parsed is None:
            skipped += 1
            continue
        docs, question, relevant_doc_ids = parsed
        docs_by_id = {
            int(doc["doc_id"]): doc_text(doc, max_chars=doc_max_chars)
            for doc in docs
            if str(doc.get("doc_id", "")).isdigit()
        }
        docs_by_id = {doc_id: text for doc_id, text in docs_by_id.items() if text}
        if not docs_by_id:
            skipped += 1
            continue
        random_doc_pool.extend(docs_by_id.values())
        parsed_rows.append(
            {
                "cluster": int(row["cluster"]),
                "id": str(row.get("id")),
                "question": question,
                "docs_by_id": docs_by_id,
                "relevant_doc_ids": [doc_id for doc_id in relevant_doc_ids if doc_id in docs_by_id],
            }
        )

    rng.shuffle(parsed_rows)
    records = []
    for row in parsed_rows:
        if len(records) >= limit:
            break
        relevant_ids = row["relevant_doc_ids"]
        if not relevant_ids:
            continue
        cluster_negatives = [
            text
            for doc_id, text in row["docs_by_id"].items()
            if doc_id not in set(relevant_ids)
        ]
        for positive_doc_id in relevant_ids:
            if len(records) >= limit:
                break
            positive = row["docs_by_id"].get(positive_doc_id)
            if not positive:
                continue
            negatives = []
            rng.shuffle(cluster_negatives)
            negatives.extend(cluster_negatives[:negatives_per_record])
            random_pool = [text for text in rng.sample(random_doc_pool, k=min(len(random_doc_pool), 32)) if text != positive]
            negatives.extend(random_pool[:random_negatives_per_record])
            deduped_negatives = []
            seen = {positive}
            for negative in negatives:
                if negative in seen:
                    continue
                seen.add(negative)
                deduped_negatives.append(negative)
            if not deduped_negatives:
                continue
            records.append(
                {
                    "source": SOURCE,
                    "query": RETRIEVAL_PROMPT + row["question"],
                    "positive": positive,
                    "negatives": deduped_negatives,
                    "metadata": {
                        "dataset_id": row["id"],
                        "cluster": row["cluster"],
                        "positive_doc_id": positive_doc_id,
                        "relevant_doc_ids": relevant_ids,
                    },
                }
            )

    summary = {
        "source": SOURCE,
        "records": len(records),
        "parsed_good_rows": len(parsed_rows),
        "skipped_rows": skipped,
        "unique_doc_pool_size": len(set(random_doc_pool)),
        "doc_max_chars": doc_max_chars,
        "negatives_per_record": negatives_per_record,
        "random_negatives_per_record": random_negatives_per_record,
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare STS-v14 recovery data from Vikhr Grounded-RAG-RU-v2.")
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/open_ru_sts_v14_grounded_rag.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/open_ru_sts_v14_grounded_rag_summary.json"))
    parser.add_argument("--limit", type=int, default=8000)
    parser.add_argument("--negatives-per-record", type=int, default=2)
    parser.add_argument("--random-negatives-per-record", type=int, default=1)
    parser.add_argument("--doc-max-chars", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    records, summary = build_records(
        limit=args.limit,
        negatives_per_record=args.negatives_per_record,
        random_negatives_per_record=args.random_negatives_per_record,
        doc_max_chars=args.doc_max_chars,
        seed=args.seed,
    )
    rng = random.Random(args.seed)
    rng.shuffle(records)
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
