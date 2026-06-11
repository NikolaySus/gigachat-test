from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset


SEMANTIC_PROMPT = "Instruct: Given a text, retrieve semantically similar text\nQuery: "
RETRIEVAL_PROMPT = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
CLASSIFICATION_PROMPT = "Instruct: Classify the category of the given scientific paper\nQuery: "


def clean_text(value) -> str:
    return " ".join(str(value).split())


def is_good_text(value: str, *, min_chars: int = 20, max_chars: int = 1800) -> bool:
    text = clean_text(value)
    if not (min_chars <= len(text) <= max_chars):
        return False
    alpha = sum(char.isalpha() for char in text)
    return alpha / max(len(text), 1) >= 0.45


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


def pair_score_record(source: str, sentence1: str, sentence2: str, score: float) -> dict:
    return {
        "objective": "pair_score",
        "source": source,
        "sentence1": SEMANTIC_PROMPT + clean_text(sentence1),
        "sentence2": clean_text(sentence2),
        "score": max(0.0, min(1.0, float(score))),
    }


def contrastive_record(source: str, query: str, positive: str, negatives: list[str]) -> dict:
    return {
        "objective": "contrastive",
        "source": source,
        "query": SEMANTIC_PROMPT + clean_text(query),
        "positive": clean_text(positive),
        "negatives": [clean_text(negative) for negative in negatives if is_good_text(negative)],
    }


def paraphraser_records(limit: int, negatives_per_record: int, rng: random.Random) -> list[dict]:
    dataset = load_dataset("merionum/ru_paraphraser")["train"]
    rows = [
        {
            "text_1": clean_text(row["text_1"]),
            "text_2": clean_text(row["text_2"]),
            "class": str(row["class"]),
        }
        for row in dataset
        if is_good_text(row["text_1"]) and is_good_text(row["text_2"])
    ]
    positives = [(row["text_1"], row["text_2"]) for row in rows if row["class"] == "1"]
    hard_negatives = [row["text_2"] for row in rows if row["class"] == "-1"]
    rng.shuffle(rows)
    records = []
    for row in rows:
        if len(records) >= limit:
            break
        score = {"1": 1.0, "0": 0.65, "-1": 0.0}.get(row["class"])
        if score is None:
            continue
        records.append(pair_score_record("merionum/ru_paraphraser", row["text_1"], row["text_2"], score))
    rng.shuffle(positives)
    for query, positive in positives[:limit]:
        records.append(
            contrastive_record(
                "merionum/ru_paraphraser",
                query,
                positive,
                sample_other(hard_negatives, positive, rng, negatives_per_record),
            )
        )
    return records


def ru_stsbenchmark_records(limit: int, rng: random.Random) -> list[dict]:
    dataset = load_dataset("ai-forever/ru-stsbenchmark-sts", "sts")["train"]
    rows = list(dataset)
    rng.shuffle(rows)
    records = []
    for row in rows:
        if len(records) >= limit:
            break
        sentence1 = clean_text(row["sentence1"])
        sentence2 = clean_text(row["sentence2"])
        if not is_good_text(sentence1) or not is_good_text(sentence2):
            continue
        records.append(
            pair_score_record(
                "ai-forever/ru-stsbenchmark-sts",
                sentence1,
                sentence2,
                float(row["score"]) / 5.0,
            )
        )
    return records


def ru_hnp_records(
    limit: int,
    negatives_per_record: int,
    positives_per_query: int,
    rng: random.Random,
) -> list[dict]:
    dataset = load_dataset("deepvk/ru-HNP")["train"]
    rows = list(dataset)
    rng.shuffle(rows)
    records = []
    selected = 0
    for row in rows:
        if selected >= limit:
            break
        query = clean_text(row["query"])
        positives = [clean_text(value) for value in row["pos"] if is_good_text(value)]
        negatives = [clean_text(value) for value in row["neg"] if is_good_text(value)]
        if not is_good_text(query) or not positives or not negatives:
            continue
        rng.shuffle(positives)
        rng.shuffle(negatives)
        for positive in positives[:positives_per_query]:
            records.append(
                contrastive_record(
                    "deepvk/ru-HNP",
                    query,
                    positive,
                    negatives[:negatives_per_record],
                )
            )
            records.append(pair_score_record("deepvk/ru-HNP", query, positive, 1.0))
        for negative in negatives[:positives_per_query]:
            records.append(pair_score_record("deepvk/ru-HNP", query, negative, 0.0))
        selected += 1
    return records


def scibench_records(dataset_name: str, limit: int, negatives_per_record: int, rng: random.Random) -> list[dict]:
    dataset = load_dataset(dataset_name)["train"]
    by_label: dict[str, list[str]] = {}
    for row in dataset:
        text = clean_text(row["text"])
        if is_good_text(text, max_chars=2400):
            by_label.setdefault(str(row["label"]), []).append(text)
    labels = list(by_label)
    examples = list(dataset)
    rng.shuffle(examples)
    records = []
    for row in examples:
        if len(records) >= limit:
            break
        text = clean_text(row["text"])
        if not is_good_text(text, max_chars=2400):
            continue
        label = str(row["label"])
        same_label = [candidate for candidate in by_label[label] if candidate != text]
        if not same_label:
            continue
        positive = rng.choice(same_label)
        negative_labels = [other for other in labels if other != label]
        negatives = []
        for negative_label in rng.sample(negative_labels, k=min(negatives_per_record, len(negative_labels))):
            negatives.append(rng.choice(by_label[negative_label]))
        records.append(
            {
                "objective": "contrastive",
                "source": dataset_name,
                "query": CLASSIFICATION_PROMPT + text,
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
        negative_ids = [
            doc_id
            for doc_id in rng.sample(all_doc_ids, k=min(len(all_doc_ids), negatives_per_record * 4))
            if doc_id != str(row["corpus-id"])
        ]
        negatives = [doc_by_id[doc_id] for doc_id in negative_ids[:negatives_per_record]]
        records.append(
            {
                "objective": "contrastive",
                "source": "ai-forever/rubq-retrieval",
                "query": RETRIEVAL_PROMPT + query,
                "positive": positive,
                "negatives": negatives,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare STS-focused open Russian training mix v2.")
    parser.add_argument("--out", type=Path, default=Path("data/contrastive/open_ru_sts_v2_train.jsonl"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--paraphraser-limit", type=int, default=2000)
    parser.add_argument("--ru-stsbenchmark-limit", type=int, default=2000)
    parser.add_argument("--ru-hnp-limit", type=int, default=4000)
    parser.add_argument("--scibench-limit", type=int, default=512)
    parser.add_argument("--rubq-limit", type=int, default=512)
    parser.add_argument("--negatives-per-record", type=int, default=5)
    parser.add_argument("--ru-hnp-positives-per-query", type=int, default=2)
    args = parser.parse_args()

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    rng = random.Random(args.seed)
    records = []
    records.extend(paraphraser_records(args.paraphraser_limit, args.negatives_per_record, rng))
    records.extend(ru_stsbenchmark_records(args.ru_stsbenchmark_limit, rng))
    records.extend(
        ru_hnp_records(
            args.ru_hnp_limit,
            args.negatives_per_record,
            args.ru_hnp_positives_per_query,
            rng,
        )
    )
    records.extend(
        scibench_records(
            "ai-forever/ru-scibench-grnti-classification",
            args.scibench_limit,
            min(args.negatives_per_record, 2),
            rng,
        )
    )
    records.extend(
        scibench_records(
            "ai-forever/ru-scibench-oecd-classification",
            args.scibench_limit,
            min(args.negatives_per_record, 2),
            rng,
        )
    )
    records.extend(rubq_records(args.rubq_limit, min(args.negatives_per_record, 2), rng))

    rng.shuffle(records)
    write_jsonl(args.out, records)
    counts: dict[str, int] = {}
    objective_counts: dict[str, int] = {}
    for record in records:
        counts[record["source"]] = counts.get(record["source"], 0) + 1
        objective_counts[record["objective"]] = objective_counts.get(record["objective"], 0) + 1
    print(f"Wrote {len(records)} records to {args.out}")
    print(json.dumps({"sources": counts, "objectives": objective_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
