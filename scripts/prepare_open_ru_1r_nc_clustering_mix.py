from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"Need {count} records, got {len(records)}")
    records = records[:]
    random.Random(seed).shuffle(records)
    return records[:count]


def filter_grounded_rag(
    records: list[dict[str, Any]],
    *,
    min_query_chars: int,
    min_positive_chars: int,
    min_negative_chars: int,
    min_negatives: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    clusters = set()
    for record in records:
        negatives = [
            negative
            for negative in record.get("negatives", [])
            if len(str(negative)) >= min_negative_chars
        ]
        positive = str(record["positive"])
        if len(str(record["query"])) < min_query_chars:
            continue
        if len(positive) < min_positive_chars:
            continue
        if len(negatives) < min_negatives:
            continue
        deduped = []
        seen = {positive}
        for negative in negatives:
            if negative in seen:
                continue
            seen.add(negative)
            deduped.append(negative)
        if len(deduped) < min_negatives:
            continue
        kept = dict(record)
        kept["negatives"] = deduped[:min_negatives]
        filtered.append(kept)
        clusters.add(record.get("metadata", {}).get("cluster"))
    return filtered, {
        "min_query_chars": min_query_chars,
        "min_positive_chars": min_positive_chars,
        "min_negative_chars": min_negative_chars,
        "min_negatives": min_negatives,
        "source_total": len(records),
        "kept": len(filtered),
        "clusters_kept": len(clusters),
    }


def extra_habr_records(path: Path, *, count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = read_jsonl(path)
    selected = sample(records, count=count, seed=seed)
    return selected, {"source_total": len(records), "kept": len(selected), "source_path": str(path)}


def extract_conversation_text(conversation: Any) -> str:
    if not isinstance(conversation, list):
        return ""
    parts = []
    for message in conversation:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or message.get("value") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def grandmaster_records(*, count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    dataset = load_dataset("Vikhrmodels/GrandMaster-PRO-MAX", split="train", streaming=True)
    by_cluster: dict[int, list[str]] = defaultdict(list)
    scanned = 0
    for row in dataset:
        scanned += 1
        if row.get("prompt_lang") != "ru" or row.get("answer_lang") != "ru":
            continue
        text = normalize_space(extract_conversation_text(row.get("conversation")))
        if not (350 <= len(text) <= 3500):
            continue
        cluster = int(row.get("cluster", -1))
        if cluster < 0:
            continue
        by_cluster[cluster].append(text)
        if scanned >= 220_000:
            break

    usable = {cluster: texts for cluster, texts in by_cluster.items() if len(texts) >= 2}
    clusters = list(usable)
    if len(clusters) < 2:
        raise RuntimeError("Not enough usable GrandMaster clusters")
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count and attempts < count * 20:
        attempts += 1
        cluster = rng.choice(clusters)
        query, positive = rng.sample(usable[cluster], 2)
        negative_cluster = rng.choice([item for item in clusters if item != cluster])
        negative = rng.choice(usable[negative_cluster])
        rows.append(
            {
                "source": "Vikhrmodels/GrandMaster-PRO-MAX:clustered",
                "query": "Instruct: Given a text, retrieve another text from the same semantic topic cluster\nQuery: " + query,
                "positive": positive,
                "negatives": [negative],
                "metadata": {"cluster": cluster, "negative_cluster": negative_cluster},
                "objective": "contrastive",
            }
        )
    return rows, {
        "scanned": scanned,
        "usable_clusters": len(usable),
        "cluster_size_histogram": dict(Counter(min(len(texts), 10) for texts in usable.values())),
        "kept": len(rows),
    }


def veles_records(*, count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    dataset = load_dataset("Vikhrmodels/Veles-2.5", split="train", streaming=True)
    by_topic: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    for row in dataset:
        scanned += 1
        topic = normalize_space(str(row.get("topic") or row.get("category") or ""))
        if not topic or topic.lower() == "none":
            continue
        text = normalize_space(extract_conversation_text(row.get("conversations")))
        if not (250 <= len(text) <= 3000):
            continue
        by_topic[topic].append(text)
        if scanned >= 120_000:
            break

    usable = {topic: texts for topic, texts in by_topic.items() if len(texts) >= 2}
    topics = list(usable)
    if len(topics) < 2:
        raise RuntimeError("Not enough usable Veles topics")
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count and attempts < count * 20:
        attempts += 1
        topic = rng.choice(topics)
        query, positive = rng.sample(usable[topic], 2)
        negative_topic = rng.choice([item for item in topics if item != topic])
        negative = rng.choice(usable[negative_topic])
        rows.append(
            {
                "source": "Vikhrmodels/Veles-2.5:topic",
                "query": "Instruct: Given a text, retrieve another text from the same semantic topic\nQuery: " + query,
                "positive": positive,
                "negatives": [negative],
                "metadata": {"topic": topic, "negative_topic": negative_topic},
                "objective": "contrastive",
            }
        )
    return rows, {
        "scanned": scanned,
        "usable_topics": len(usable),
        "topic_size_histogram": dict(Counter(min(len(texts), 10) for texts in usable.values())),
        "kept": len(rows),
    }


def replacement_records(args) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    if args.replacement == "grounded_strict":
        source = read_jsonl(args.grounded_path)
        records, summary = filter_grounded_rag(
            source,
            min_query_chars=args.grounded_min_query_chars,
            min_positive_chars=args.grounded_min_positive_chars,
            min_negative_chars=args.grounded_min_negative_chars,
            min_negatives=args.grounded_min_negatives,
        )
        return sample(records, count=args.replacement_count, seed=args.seed + 4), summary, "grounded_strict"
    if args.replacement == "habr_extra":
        return (*extra_habr_records(args.habr_extra_path, count=args.replacement_count, seed=args.seed + 4), "habr_extra")
    if args.replacement == "grandmaster":
        return (*grandmaster_records(count=args.replacement_count, seed=args.seed + 4), "grandmaster")
    if args.replacement == "veles":
        return (*veles_records(count=args.replacement_count, seed=args.seed + 4), "veles")
    raise ValueError(f"Unknown replacement: {args.replacement}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 1R-NC clustering-focused replacement mix.")
    parser.add_argument("--replacement", choices=("grounded_strict", "habr_extra", "grandmaster", "veles"), required=True)
    parser.add_argument("--geracl-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_geracl.jsonl"))
    parser.add_argument("--habr-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"))
    parser.add_argument("--habr-extra-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_habr_qa_sbs_harder_sim019_len.jsonl"))
    parser.add_argument("--deepvk-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl"))
    parser.add_argument("--grounded-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_grounded_rag_v2.jsonl"))
    parser.add_argument("--replacement-out", type=Path, required=True)
    parser.add_argument("--mix-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--replacement-count", type=int, default=3200)
    parser.add_argument("--grounded-min-query-chars", type=int, default=180)
    parser.add_argument("--grounded-min-positive-chars", type=int, default=1200)
    parser.add_argument("--grounded-min-negative-chars", type=int, default=1200)
    parser.add_argument("--grounded-min-negatives", type=int, default=2)
    args = parser.parse_args()

    replacement, replacement_summary, replacement_name = replacement_records(args)
    write_jsonl(args.replacement_out, replacement)

    geracl = read_jsonl(args.geracl_path)
    habr = read_jsonl(args.habr_path)
    deepvk = read_jsonl(args.deepvk_path)
    selected = {
        "geracl": sample(geracl, count=6400, seed=args.seed),
        "habr_harder": sample(habr, count=3200, seed=args.seed + 1),
        "deepvk_filtered": sample(deepvk, count=3200, seed=args.seed + 2),
        replacement_name: replacement,
    }
    mixed: list[dict[str, Any]] = []
    for records in selected.values():
        mixed.extend(records)
    random.Random(args.seed + 3).shuffle(mixed)
    write_jsonl(args.mix_out, mixed)

    summary_path = args.summary_out or args.mix_out.with_name(args.mix_out.stem + "_summary.json")
    write_json(
        summary_path,
        {
            "output": str(args.mix_out),
            "replacement_output": str(args.replacement_out),
            "seed": args.seed,
            "ratio": {
                "geracl": 2,
                "habr_harder": 1,
                "deepvk_ru_hnp_filtered": 1,
                replacement_name: 1,
            },
            "counts": {
                "geracl_source": len(geracl),
                "geracl_used": len(selected["geracl"]),
                "habr_harder_source": len(habr),
                "habr_harder_used": len(selected["habr_harder"]),
                "deepvk_filtered_source": len(deepvk),
                "deepvk_used": len(selected["deepvk_filtered"]),
                f"{replacement_name}_used": len(replacement),
                "total": len(mixed),
            },
            "batch_size": 4,
            "max_steps_1x": len(mixed) // 4,
            "source_paths": {
                "geracl": str(args.geracl_path),
                "habr_harder": str(args.habr_path),
                "deepvk_filtered": str(args.deepvk_path),
            },
            "replacement": replacement_name,
            "replacement_summary": replacement_summary,
        },
    )
    print(f"Wrote {args.replacement_out}")
    print(f"Wrote {args.mix_out}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
