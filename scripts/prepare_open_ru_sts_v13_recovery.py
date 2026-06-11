from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


RETRIEVAL_PROMPT = "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
CLASSIFICATION_PROMPT = "Instruct: Classify the category of the given scientific paper\nQuery: "

GERACL_CLASSES_SOURCE = "deepvk/GeRaCl_synthethic_dataset:synthetic_classes_train"
GERACL_POSITIVES_SOURCE = "deepvk/GeRaCl_synthethic_dataset:synthetic_positives"
PROMPTRIEVER_SOURCE = "Vladimirlv/ru-promptriever-dataset:standard"

TARGETED_V12_SOURCES = [
    "ai-forever/rubq-retrieval",
    "ai-forever/ru-scibench-grnti-classification",
    "ai-forever/ru-scibench-oecd-classification",
]


def clean_text(value) -> str:
    return " ".join(str(value or "").split())


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def text_hashes_from_records(records: Iterable[dict]) -> set[str]:
    hashes = set()
    for record in records:
        candidates = []
        if "query" in record:
            candidates.append(record["query"])
        if "positive" in record:
            candidates.append(record["positive"])
        candidates.extend(record.get("negatives") or [])
        if "sentence1" in record:
            candidates.append(record["sentence1"])
        if "sentence2" in record:
            candidates.append(record["sentence2"])
        for text in candidates:
            normalized = clean_text(text).lower()
            if normalized:
                hashes.add(normalized)
    return hashes


def geracl_synthetic_class_records(limit: int, negatives_per_record: int, rng: random.Random) -> list[dict]:
    dataset = load_dataset(
        "deepvk/GeRaCl_synthethic_dataset",
        "synthetic_classes_train",
        split="train",
    )
    by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_scenario: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for row in dataset:
        text = clean_text(row["text"])
        if not text:
            continue
        scenarios = row.get("scenarios") or []
        for index, scenario in enumerate(scenarios):
            classes = row.get(f"classes_{index}") or []
            if not classes:
                continue
            positive_class = clean_text(classes[0])
            scenario = clean_text(scenario)
            if not scenario or not positive_class:
                continue
            key = (scenario, positive_class)
            by_key[key].append(text)

    for key, texts in by_key.items():
        if len(texts) < 2:
            continue
        by_scenario[key[0]].append(key)

    keys = [key for key, texts in by_key.items() if len(texts) >= 2 and len(by_scenario[key[0]]) >= 2]
    rng.shuffle(keys)

    records = []
    for key in keys:
        if len(records) >= limit:
            break
        scenario, positive_class = key
        query, positive = rng.sample(by_key[key], k=2)
        negative_keys = [other for other in by_scenario[scenario] if other != key]
        rng.shuffle(negative_keys)
        negatives = []
        for negative_key in negative_keys:
            negatives.append(rng.choice(by_key[negative_key]))
            if len(negatives) >= negatives_per_record:
                break
        if not negatives:
            continue
        records.append(
            {
                "source": GERACL_CLASSES_SOURCE,
                "query": CLASSIFICATION_PROMPT + query,
                "positive": positive,
                "negatives": negatives,
                "metadata": {
                    "scenario": scenario,
                    "positive_class": positive_class,
                },
            }
        )
    return records


def geracl_synthetic_positive_records(limit: int, negatives_per_record: int, rng: random.Random) -> list[dict]:
    dataset = load_dataset(
        "deepvk/GeRaCl_synthethic_dataset",
        "synthetic_positives",
        split="train",
    )
    rows = []
    by_class: dict[str, list[int]] = defaultdict(list)
    for row in dataset:
        text = clean_text(row["text"])
        classes = sorted({clean_text(item) for item in row.get("classes", []) if clean_text(item)})
        if not text or not classes:
            continue
        index = len(rows)
        rows.append({"text": text, "classes": set(classes)})
        for class_name in classes:
            by_class[class_name].append(index)

    indices = list(range(len(rows)))
    rng.shuffle(indices)
    records = []
    for index in indices:
        if len(records) >= limit:
            break
        row = rows[index]
        positive_pool = set()
        for class_name in row["classes"]:
            positive_pool.update(by_class[class_name])
        positive_pool.discard(index)
        if not positive_pool:
            continue
        positive_index = rng.choice(list(positive_pool))
        negatives = []
        negative_candidates = indices[:]
        rng.shuffle(negative_candidates)
        for negative_index in negative_candidates:
            if negative_index == index:
                continue
            if row["classes"].isdisjoint(rows[negative_index]["classes"]):
                negatives.append(rows[negative_index]["text"])
            if len(negatives) >= negatives_per_record:
                break
        if len(negatives) < negatives_per_record:
            continue
        records.append(
            {
                "source": GERACL_POSITIVES_SOURCE,
                "query": CLASSIFICATION_PROMPT + row["text"],
                "positive": rows[positive_index]["text"],
                "negatives": negatives,
            }
        )
    return records


def passage_text(passage: dict) -> str:
    title = clean_text(passage.get("title"))
    text = clean_text(passage.get("text"))
    return clean_text(f"{title} {text}" if title else text)


def promptriever_records(limit: int, negatives_per_record: int) -> list[dict]:
    dataset = load_dataset(
        "Vladimirlv/ru-promptriever-dataset",
        split="train",
        streaming=True,
    )
    records = []
    for row in dataset:
        if len(records) >= limit:
            break
        if row.get("is_repeated") or row.get("has_instruction"):
            continue
        positives = row.get("positive_passages") or []
        negatives_source = row.get("negative_passages") or []
        if not positives or not negatives_source:
            continue
        query = clean_text(row.get("only_query") or row.get("query"))
        positive = passage_text(positives[0])
        negatives = [passage_text(item) for item in negatives_source[:negatives_per_record]]
        negatives = [item for item in negatives if item and item != positive]
        if not query or not positive or len(negatives) < 1:
            continue
        records.append(
            {
                "source": PROMPTRIEVER_SOURCE,
                "query": RETRIEVAL_PROMPT + query,
                "positive": positive,
                "negatives": negatives,
            }
        )
    return records


def existing_targeted_records(path: Path, limit_per_source: int, rng: random.Random) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in read_jsonl(path):
        source = record.get("source")
        if source in TARGETED_V12_SOURCES:
            by_source[source].append(record)
    records = []
    for source in TARGETED_V12_SOURCES:
        source_records = by_source[source]
        rng.shuffle(source_records)
        records.extend(source_records[:limit_per_source])
    return records


def write_summary(path: Path, records: list[dict], extra: dict) -> None:
    counts = {}
    for record in records:
        counts[record["source"]] = counts.get(record["source"], 0) + 1
    summary = {
        "total_records": len(records),
        "counts": counts,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean-first STS-v13 clustering/retrieval recovery data.")
    parser.add_argument("--candidates-out", type=Path, default=Path("data/contrastive/open_ru_sts_v13_recovery_candidates.jsonl"))
    parser.add_argument("--balanced-out", type=Path, default=Path("data/contrastive/open_ru_sts_v13_balanced_recovery.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/open_ru_sts_v13_recovery_summary.json"))
    parser.add_argument("--v12-data", type=Path, default=Path("data/contrastive/open_ru_recovery_v12_2134_per_source.jsonl"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--negatives-per-record", type=int, default=2)
    parser.add_argument("--geracl-classes-limit", type=int, default=2000)
    parser.add_argument("--geracl-positives-limit", type=int, default=1000)
    parser.add_argument("--promptriever-limit", type=int, default=2000)
    parser.add_argument("--v12-limit-per-source", type=int, default=2134)
    parser.add_argument("--balanced-geracl-classes", type=int, default=1250)
    parser.add_argument("--balanced-geracl-positives", type=int, default=750)
    parser.add_argument("--balanced-promptriever", type=int, default=1500)
    parser.add_argument("--balanced-v12-per-source", type=int, default=500)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    rng = random.Random(args.seed)
    geracl_classes = geracl_synthetic_class_records(args.geracl_classes_limit, args.negatives_per_record, rng)
    geracl_positives = geracl_synthetic_positive_records(args.geracl_positives_limit, args.negatives_per_record, rng)
    promptriever = promptriever_records(args.promptriever_limit, args.negatives_per_record)
    v12_targeted = existing_targeted_records(args.v12_data, args.v12_limit_per_source, rng)

    candidates = geracl_classes + geracl_positives + promptriever + v12_targeted
    rng.shuffle(candidates)
    write_jsonl(args.candidates_out, candidates)

    v12_balanced = existing_targeted_records(args.v12_data, args.balanced_v12_per_source, rng)
    balanced = (
        geracl_classes[: args.balanced_geracl_classes]
        + geracl_positives[: args.balanced_geracl_positives]
        + promptriever[: args.balanced_promptriever]
        + v12_balanced
    )
    rng.shuffle(balanced)
    write_jsonl(args.balanced_out, balanced)

    candidate_hashes = text_hashes_from_records(candidates)
    balanced_hashes = text_hashes_from_records(balanced)
    write_summary(
        args.summary_out,
        candidates,
        {
            "candidate_path": str(args.candidates_out),
            "balanced_path": str(args.balanced_out),
            "balanced_records": len(balanced),
            "unique_candidate_texts": len(candidate_hashes),
            "unique_balanced_texts": len(balanced_hashes),
            "balanced_counts": {
                source: sum(1 for record in balanced if record["source"] == source)
                for source in sorted({record["source"] for record in balanced})
            },
            "excluded_geracl_configs": ["ru_mteb_classes", "ru_mteb_extended_classes"],
            "notes": [
                "GeRaCl synthetic_classes_train uses classes_i[0] as the positive class for each synthetic scenario.",
                "RuPromptriever keeps non-repeated, non-instruction rows only.",
            ],
        },
    )
    print(f"Wrote {len(candidates)} candidate records to {args.candidates_out}")
    print(f"Wrote {len(balanced)} balanced records to {args.balanced_out}")
    print(f"Wrote summary to {args.summary_out}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
