from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


SEMANTIC_PROMPT = "Instruct: Given a text, retrieve semantically similar text\nQuery: "
RETRIEVAL_PROMPT = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
CLASSIFICATION_PROMPT = "Instruct: Classify the category of the given scientific paper\nQuery: "


def clean_text(value) -> str:
    return " ".join(str(value).split())


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_other(items: list[str], forbidden: str, rng: random.Random, count: int) -> list[str]:
    pool = [item for item in items if item and item != forbidden]
    if not pool:
        return []
    return rng.sample(pool, k=min(count, len(pool)))


def paraphraser_records(limit: int, negatives_per_record: int, rng: random.Random) -> list[dict]:
    dataset = load_dataset("merionum/ru_paraphraser")["train"]
    positives = [
        (clean_text(row["text_1"]), clean_text(row["text_2"]))
        for row in dataset
        if str(row["class"]) == "1"
    ]
    all_texts = [text for pair in positives for text in pair]
    rng.shuffle(positives)
    records = []
    for query, positive in positives[:limit]:
        records.append(
            {
                "source": "merionum/ru_paraphraser",
                "query": SEMANTIC_PROMPT + query,
                "positive": positive,
                "negatives": sample_other(all_texts, positive, rng, negatives_per_record),
            }
        )
    return records


def sts_records(limit: int, negatives_per_record: int, rng: random.Random) -> list[dict]:
    dataset = load_dataset("ai-forever/ru-stsbenchmark-sts", "sts")["train"]
    positives = [
        (clean_text(row["sentence1"]), clean_text(row["sentence2"]))
        for row in dataset
        if float(row["score"]) >= 4.0
    ]
    negative_pool = [
        clean_text(row["sentence2"])
        for row in dataset
        if float(row["score"]) <= 1.0
    ]
    rng.shuffle(positives)
    records = []
    for query, positive in positives[:limit]:
        records.append(
            {
                "source": "ai-forever/ru-stsbenchmark-sts",
                "query": SEMANTIC_PROMPT + query,
                "positive": positive,
                "negatives": sample_other(negative_pool, positive, rng, negatives_per_record),
            }
        )
    return records


def scibench_records(dataset_name: str, limit: int, negatives_per_record: int, rng: random.Random) -> list[dict]:
    dataset = load_dataset(dataset_name)["train"]
    by_label: dict[str, list[str]] = {}
    for row in dataset:
        by_label.setdefault(str(row["label"]), []).append(clean_text(row["text"]))
    labels = list(by_label)
    examples = list(dataset)
    rng.shuffle(examples)
    records = []
    for row in examples:
        if len(records) >= limit:
            break
        label = str(row["label"])
        same_label = [text for text in by_label[label] if text != clean_text(row["text"])]
        if not same_label:
            continue
        positive = rng.choice(same_label)
        negative_labels = [other for other in labels if other != label]
        negatives = []
        for negative_label in rng.sample(negative_labels, k=min(negatives_per_record, len(negative_labels))):
            negatives.append(rng.choice(by_label[negative_label]))
        records.append(
            {
                "source": dataset_name,
                "query": CLASSIFICATION_PROMPT + clean_text(row["text"]),
                "positive": positive,
                "negatives": negatives,
            }
        )
    return records


def rubq_records(limit: int, negatives_per_record: int, rng: random.Random) -> list[dict]:
    queries = load_dataset("ai-forever/rubq-retrieval", "queries")["queries"]
    corpus = load_dataset("ai-forever/rubq-retrieval", "corpus")["corpus"]
    qrels = load_dataset("ai-forever/rubq-retrieval", "default")["test"]
    query_by_id = {str(row["_id"]): clean_text(row["text"]) for row in queries}
    doc_by_id = {
        str(row["_id"]): clean_text((row.get("title") or "") + " " + row["text"])
        for row in corpus
    }
    all_doc_ids = list(doc_by_id)
    qrels_list = list(qrels)
    rng.shuffle(qrels_list)
    records = []
    for row in qrels_list:
        if len(records) >= limit:
            break
        query = query_by_id.get(str(row["query-id"]))
        positive = doc_by_id.get(str(row["corpus-id"]))
        if not query or not positive:
            continue
        negative_ids = [doc_id for doc_id in rng.sample(all_doc_ids, k=min(len(all_doc_ids), negatives_per_record * 4)) if doc_id != str(row["corpus-id"])]
        negatives = [doc_by_id[doc_id] for doc_id in negative_ids[:negatives_per_record]]
        records.append(
            {
                "source": "ai-forever/rubq-retrieval",
                "query": RETRIEVAL_PROMPT + query,
                "positive": positive,
                "negatives": negatives,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a first open-Russian contrastive JSONL mix.")
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/open_ru_train.jsonl"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--per-source-limit", type=int, default=512)
    parser.add_argument("--negatives-per-record", type=int, default=2)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    rng = random.Random(args.seed)
    records = []
    records.extend(paraphraser_records(args.per_source_limit, args.negatives_per_record, rng))
    records.extend(sts_records(args.per_source_limit, args.negatives_per_record, rng))
    records.extend(scibench_records("ai-forever/ru-scibench-grnti-classification", args.per_source_limit, args.negatives_per_record, rng))
    records.extend(scibench_records("ai-forever/ru-scibench-oecd-classification", args.per_source_limit, args.negatives_per_record, rng))
    records.extend(rubq_records(args.per_source_limit, args.negatives_per_record, rng))
    rng.shuffle(records)
    write_jsonl(args.out, records)
    counts = {}
    for record in records:
        counts[record["source"]] = counts.get(record["source"], 0) + 1
    print(f"Wrote {len(records)} records to {args.out}")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
