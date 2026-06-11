#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[\w]+", re.U)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\ufeff", " ").replace("ё", "е").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_key(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^0-9a-zа-яе]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cyrillic_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cyr = [ch for ch in letters if "а" <= ch.lower() <= "я" or ch.lower() == "е"]
    return len(cyr) / len(letters)


def compact(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    return text[:max_chars].strip()


def useful(text: str, *, min_chars: int, max_chars: int) -> bool:
    text = normalize_text(text)
    if not (min_chars <= len(text) <= max_chars):
        return False
    if len(TOKEN_RE.findall(text)) < 8:
        return False
    return cyrillic_ratio(text) >= 0.55


def benchmark_needles(cache_dir: Path) -> dict[str, set[str]]:
    needles: dict[str, set[str]] = {"exact": set(), "prefix": set(), "title": set()}
    for dataset_name in (
        "ai-forever/ru-scibench-oecd-classification",
        "ai-forever/ru-scibench-grnti-classification",
        "ai-forever/ru-scibench-oecd-clustering-p2p",
        "ai-forever/ru-scibench-grnti-clustering-p2p",
    ):
        try:
            dataset = load_dataset(dataset_name, cache_dir=str(cache_dir))
        except Exception:
            continue
        for split in dataset.values():
            text_column = "text" if "text" in split.column_names else "sentences"
            for row in split:
                text = normalize_key(row.get(text_column, ""))
                if len(text) >= 80:
                    needles["exact"].add(text)
                    needles["prefix"].add(text[:240])
                title = text.split(".", 1)[0]
                if len(title) >= 30:
                    needles["title"].add(title)
    return needles


def contaminated(text: str, needles: dict[str, set[str]]) -> bool:
    key = normalize_key(text)
    if key in needles["exact"]:
        return True
    if len(key) >= 240 and key[:240] in needles["prefix"]:
        return True
    title = key.split(".", 1)[0]
    return len(title) >= 30 and title in needles["title"]


def jaccard(left: str, right: str) -> float:
    lt = set(TOKEN_RE.findall(normalize_text(left)))
    rt = set(TOKEN_RE.findall(normalize_text(right)))
    if not lt or not rt:
        return 0.0
    return len(lt & rt) / len(lt | rt)


def add_pair(
    records: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    skipped: Counter[str],
    *,
    left: str,
    right: str,
    score: float,
    source: str,
    metadata: dict[str, Any],
    min_chars: int,
    max_chars: int,
) -> None:
    left = compact(left, max_chars)
    right = compact(right, max_chars)
    if not useful(left, min_chars=min_chars, max_chars=max_chars):
        skipped["left_quality"] += 1
        return
    if not useful(right, min_chars=min_chars, max_chars=max_chars):
        skipped["right_quality"] += 1
        return
    if normalize_key(left) == normalize_key(right):
        skipped["same_text"] += 1
        return
    key = tuple(sorted((normalize_key(left), normalize_key(right))))
    if key in seen:
        skipped["duplicate_pair"] += 1
        return
    seen.add(key)
    records.append(
        {
            "objective": "pair_score",
            "sentence1": left,
            "sentence2": right,
            "score": round(max(0.0, min(1.0, score)), 4),
            "source": source,
            "metadata": {
                **metadata,
                "jaccard": round(jaccard(left, right), 4),
                "contamination_policy": (
                    "Derived from reginafeles/humarticles train split. Exact text, prefix, "
                    "and title overlaps with RuSciBench GRNTI/OECD benchmark datasets are removed. "
                    "No target benchmark rows, released outputs, released teacher, or released latent "
                    "weights are used."
                ),
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean humarticles scientific pair-score data.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/contrastive/fair_humarticles_science_pairscore_2400_seed3161.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/hf_cache")
    parser.add_argument("--max-records", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=3161)
    parser.add_argument("--min-chars", type=int, default=45)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--article-lead-chars", type=int, default=2200)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    needles = benchmark_needles(args.cache_dir)
    dataset = load_dataset("reginafeles/humarticles", split="train", cache_dir=str(args.cache_dir))

    articles: list[dict[str, str]] = []
    skipped = Counter()
    for index, row in enumerate(dataset):
        title = compact(row.get("title", ""), 420)
        abstract = compact(row.get("abstract", ""), args.max_chars)
        summary = compact(row.get("sum", ""), args.max_chars)
        article = compact(row.get("article", ""), args.article_lead_chars)
        joined = " ".join(part for part in (title, abstract, summary, article) if part)
        if contaminated(joined, needles):
            skipped["benchmark_overlap"] += 1
            continue
        if not useful(abstract, min_chars=args.min_chars, max_chars=args.max_chars):
            skipped["bad_abstract"] += 1
            continue
        if not useful(summary, min_chars=args.min_chars, max_chars=args.max_chars):
            skipped["bad_summary"] += 1
            continue
        if not useful(article, min_chars=args.min_chars, max_chars=args.article_lead_chars):
            skipped["bad_article"] += 1
            continue
        articles.append(
            {
                "id": str(row.get("id") or index),
                "title": title,
                "abstract": abstract,
                "summary": summary,
                "article": article,
                "year": str(row.get("year") or ""),
                "author": str(row.get("author") or ""),
            }
        )

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for article in articles:
        base_meta = {
            "humarticles_id": article["id"],
            "year": article["year"],
            "author": article["author"][:120],
        }
        add_pair(
            records,
            seen,
            skipped,
            left=article["abstract"],
            right=article["summary"],
            score=0.9,
            source="reginafeles/humarticles:abstract_summary",
            metadata={**base_meta, "construction": "abstract_summary"},
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        add_pair(
            records,
            seen,
            skipped,
            left=article["article"],
            right=article["abstract"],
            score=0.82,
            source="reginafeles/humarticles:article_abstract",
            metadata={**base_meta, "construction": "article_lead_abstract"},
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        add_pair(
            records,
            seen,
            skipped,
            left=article["article"],
            right=article["summary"],
            score=0.78,
            source="reginafeles/humarticles:article_summary",
            metadata={**base_meta, "construction": "article_lead_summary"},
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        if useful(article["title"], min_chars=12, max_chars=420):
            add_pair(
                records,
                seen,
                skipped,
                left=article["title"],
                right=article["abstract"],
                score=0.68,
                source="reginafeles/humarticles:title_abstract",
                metadata={**base_meta, "construction": "title_abstract"},
                min_chars=12,
                max_chars=args.max_chars,
            )

    shuffled = articles[:]
    rng.shuffle(shuffled)
    for left, right in zip(shuffled, shuffled[1:] + shuffled[:1]):
        if left["id"] == right["id"]:
            continue
        add_pair(
            records,
            seen,
            skipped,
            left=left["abstract"],
            right=right["summary"],
            score=0.08,
            source="reginafeles/humarticles:cross_article_negative",
            metadata={
                "left_id": left["id"],
                "right_id": right["id"],
                "left_year": left["year"],
                "right_year": right["year"],
                "construction": "cross_article_abstract_summary_negative",
            },
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )

    rng.shuffle(records)
    if len(records) > args.max_records:
        # Keep the low-score tail represented; ranking/correlation losses need broad score support.
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            score = float(record["score"])
            bucket = "low" if score < 0.2 else "mid" if score < 0.8 else "high"
            by_bucket[bucket].append(record)
        target = {"low": args.max_records // 4, "mid": args.max_records // 4, "high": args.max_records - args.max_records // 2}
        selected: list[dict[str, Any]] = []
        for bucket, count in target.items():
            rng.shuffle(by_bucket[bucket])
            selected.extend(by_bucket[bucket][:count])
        rng.shuffle(selected)
        records = selected[: args.max_records]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output),
        "dataset": "reginafeles/humarticles",
        "source_rows": len(dataset),
        "usable_articles": len(articles),
        "records": len(records),
        "score_counts": Counter(str(record["score"]) for record in records),
        "skipped": dict(skipped),
        "benchmark_needles": {key: len(value) for key, value in needles.items()},
        "fairness": (
            "Exact/prefix/title overlaps with RuSciBench GRNTI/OECD benchmark datasets removed; "
            "target benchmark rows and released-model signals are not used."
        ),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
