from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "hf_datasets"
DATA_DIR = ROOT / "data" / "contrastive"
REPORT_DIR = ROOT / "results" / "contamination" / "rusts_external"

SEMANTIC_PROMPT = "Instruct: Given a text, retrieve semantically similar text\nQuery: "


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def strip_prompt(value: str) -> str:
    text = clean_text(value)
    if "\nQuery: " in text:
        return text.split("\nQuery: ", 1)[1]
    return text


def normalize(value: str) -> str:
    text = strip_prompt(value).lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", " ", text).strip()


def is_good_text(value: str, *, min_chars: int = 18, max_chars: int = 700) -> bool:
    text = clean_text(value)
    if not (min_chars <= len(text) <= max_chars):
        return False
    alpha = sum(char.isalpha() for char in text)
    return alpha / max(len(text), 1) >= 0.45


def lexical_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize(left).split())
    right_tokens = set(normalize(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def pair_score_record(source: str, sentence1: str, sentence2: str, score: float, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "objective": "pair_score",
        "source": source,
        "sentence1": SEMANTIC_PROMPT + clean_text(sentence1),
        "sentence2": clean_text(sentence2),
        "score": max(0.0, min(1.0, float(score))),
        "metadata": metadata or {},
    }


def contrastive_record(source: str, query: str, positive: str, negatives: list[str], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "objective": "contrastive",
        "source": source,
        "query": SEMANTIC_PROMPT + clean_text(query),
        "positive": clean_text(positive),
        "negatives": [clean_text(negative) for negative in negatives if is_good_text(negative)],
        "metadata": metadata or {},
    }


def iter_texts(record: dict[str, Any]) -> Iterable[str]:
    for key in ("query", "positive", "sentence1", "sentence2"):
        if record.get(key):
            yield str(record[key])
    for value in record.get("negatives") or []:
        yield str(value)


def load_eval_texts() -> dict[str, set[str]]:
    eval_sets: dict[str, set[str]] = {"RuSTSBenchmarkSTS": set(), "CEDRClassification": set()}

    rusts = load_dataset("mteb/RuSTSBenchmarkSTS", cache_dir=str(CACHE_DIR))
    for split in rusts:
        for row in rusts[split]:
            for key in ("sentence1", "sentence2"):
                text = normalize(row.get(key, ""))
                if text:
                    eval_sets["RuSTSBenchmarkSTS"].add(text)

    cedr = load_dataset("mteb/CEDRClassification", cache_dir=str(CACHE_DIR), trust_remote_code=True)
    for split in cedr:
        for row in cedr[split]:
            for key in ("text", "sentence", "query"):
                text = normalize(row.get(key, ""))
                if text:
                    eval_sets["CEDRClassification"].add(text)
    return eval_sets


def audit_records(records: list[dict[str, Any]], eval_sets: dict[str, set[str]], *, sample_near: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = []
    skipped = {"RuSTSBenchmarkSTS_exact": 0, "CEDRClassification_exact": 0}
    near_examples: list[dict[str, Any]] = []
    rusts_texts = list(eval_sets["RuSTSBenchmarkSTS"])
    for record in records:
        texts = [normalize(text) for text in iter_texts(record)]
        texts = [text for text in texts if text]
        rusts_hit = any(text in eval_sets["RuSTSBenchmarkSTS"] for text in texts)
        cedr_hit = any(text in eval_sets["CEDRClassification"] for text in texts)
        if rusts_hit:
            skipped["RuSTSBenchmarkSTS_exact"] += 1
            continue
        if cedr_hit:
            skipped["CEDRClassification_exact"] += 1
            continue
        if sample_near and len(near_examples) < 20:
            for text in texts[:2]:
                if not text:
                    continue
                candidates = random.sample(rusts_texts, k=min(100, len(rusts_texts)))
                best = max((SequenceMatcher(None, text, other).ratio(), other) for other in candidates)
                if best[0] >= 0.92:
                    near_examples.append({"source_text": text, "rusts_text": best[1], "ratio": best[0]})
                    break
        kept.append(record)
    return kept, {"input_records": len(records), "kept_records": len(kept), "skipped": skipped, "sampled_near_rusts": near_examples}


def leipzig_records(*, dataset_name: str, source_prefix: str, limit: int, seed: int, variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    capped = limit > 0
    positives: list[dict[str, Any]] = []
    weak_pairs: list[dict[str, Any]] = []
    dataset = load_dataset(dataset_name, split="train", streaming=True, cache_dir=str(CACHE_DIR))

    for row in dataset:
        original = clean_text(row["original"])
        paraphrase = clean_text(row["ru"])
        if not is_good_text(original) or not is_good_text(paraphrase):
            continue
        jaccard = lexical_jaccard(original, paraphrase)
        p_good = float(row.get("p_good") or 0.0)
        labse = float(row.get("labse_sim") or 0.0)
        forward = float(row.get("forward_entailment") or 0.0)
        backward = float(row.get("backward_entailment") or 0.0)
        chrf = float(row.get("chrf_sim") or 0.0)
        item = {
            "original": original,
            "paraphrase": paraphrase,
            "p_good": p_good,
            "labse": labse,
            "forward": forward,
            "backward": backward,
            "chrf": chrf,
            "jaccard": jaccard,
        }
        if variant == "strict":
            is_positive = p_good >= 0.90 and labse >= 0.88 and forward >= 0.60 and backward >= 0.60 and 0.12 <= jaccard <= 0.72
        elif variant == "diverse":
            is_positive = p_good >= 0.82 and labse >= 0.84 and forward >= 0.45 and backward >= 0.45 and 0.04 <= jaccard <= 0.42
        else:
            raise ValueError(f"Unknown variant: {variant}")
        if is_positive:
            positives.append(item)
        elif p_good <= 0.28 and labse <= 0.82 and jaccard <= 0.50:
            weak_pairs.append(item)
        if capped and len(positives) >= limit * 3 and len(weak_pairs) >= limit:
            break

    rng.shuffle(positives)
    rng.shuffle(weak_pairs)
    positive_limit = min(limit, len(positives)) if capped else len(positives)
    weak_limit = min(limit // 2, len(weak_pairs)) if capped else min(len(weak_pairs), positive_limit // 2)
    negative_pool_limit = max(limit, 100) if capped else len(weak_pairs)
    negative_pool = [item["paraphrase"] for item in weak_pairs[:negative_pool_limit]]
    random_negative_pool = (
        [item["paraphrase"] for item in positives[positive_limit : positive_limit * 3]]
        if capped
        else [item["paraphrase"] for item in positives]
    )
    records: list[dict[str, Any]] = []
    for item in positives[:positive_limit]:
        records.append(
            pair_score_record(
                f"{source_prefix}:{variant}",
                item["original"],
                item["paraphrase"],
                min(1.0, max(0.75, item["p_good"])),
                {key: item[key] for key in ("p_good", "labse", "forward", "backward", "chrf", "jaccard")},
            )
        )
        negatives = []
        if negative_pool:
            negatives.extend(rng.sample(negative_pool, k=min(2, len(negative_pool))))
        if random_negative_pool:
            negatives.extend(rng.sample(random_negative_pool, k=min(2, len(random_negative_pool))))
        if negatives:
            records.append(
                contrastive_record(
                    f"{source_prefix}:{variant}",
                    item["original"],
                    item["paraphrase"],
                    negatives[:4],
                    {"p_good": item["p_good"], "jaccard": item["jaccard"]},
                )
            )
    for item in weak_pairs[:weak_limit]:
        records.append(
            pair_score_record(
                f"{source_prefix}:{variant}:weak_negative",
                item["original"],
                item["paraphrase"],
                0.0,
                {key: item[key] for key in ("p_good", "labse", "forward", "backward", "chrf", "jaccard")},
            )
        )
    rng.shuffle(records)
    if capped:
        records = records[: limit * 2]
    return records, {
        "dataset_name": dataset_name,
        "positive_candidates": len(positives),
        "weak_pair_candidates": len(weak_pairs),
        "positive_used": positive_limit,
        "weak_pair_used": weak_limit,
        "limit": limit,
        "variant": variant,
    }


def cointegrated_records(*, limit: int, seed: int, variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return leipzig_records(
        dataset_name="cointegrated/ru-paraphrase-NMT-Leipzig",
        source_prefix="cointegrated/ru-paraphrase-NMT-Leipzig",
        limit=limit,
        seed=seed,
        variant=variant,
    )


def cleaned_leipzig_records(*, limit: int, seed: int, variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return leipzig_records(
        dataset_name="fyaronskiy/ru-paraphrase-NMT-Leipzig-cleaned",
        source_prefix="fyaronskiy/ru-paraphrase-NMT-Leipzig-cleaned",
        limit=limit,
        seed=seed,
        variant=variant,
    )


def merionum_records(*, limit: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    dataset = load_dataset("merionum/ru_paraphraser", split="train", cache_dir=str(CACHE_DIR))
    rows = [
        {
            "text_1": clean_text(row["text_1"]),
            "text_2": clean_text(row["text_2"]),
            "class": str(row["class"]),
            "jaccard": lexical_jaccard(row["text_1"], row["text_2"]),
        }
        for row in dataset
        if is_good_text(row["text_1"], max_chars=500) and is_good_text(row["text_2"], max_chars=500)
    ]
    positives = [row for row in rows if row["class"] == "1" and row["jaccard"] <= 0.82]
    related = [row for row in rows if row["class"] == "0"]
    negatives = [row for row in rows if row["class"] == "-1"]
    rng.shuffle(positives)
    rng.shuffle(related)
    rng.shuffle(negatives)
    negative_pool = [row["text_2"] for row in negatives]
    records: list[dict[str, Any]] = []
    for row in positives[:limit]:
        records.append(pair_score_record("merionum/ru_paraphraser:positive", row["text_1"], row["text_2"], 1.0, {"jaccard": row["jaccard"]}))
        records.append(
            contrastive_record(
                "merionum/ru_paraphraser:positive",
                row["text_1"],
                row["text_2"],
                rng.sample(negative_pool, k=min(4, len(negative_pool))),
                {"jaccard": row["jaccard"]},
            )
        )
    for row in related[: limit // 3]:
        records.append(pair_score_record("merionum/ru_paraphraser:related", row["text_1"], row["text_2"], 0.65, {"jaccard": row["jaccard"]}))
    for row in negatives[: limit // 3]:
        records.append(pair_score_record("merionum/ru_paraphraser:negative", row["text_1"], row["text_2"], 0.0, {"jaccard": row["jaccard"]}))
    rng.shuffle(records)
    return records[: limit * 2], {
        "positive_candidates": len(positives),
        "related_candidates": len(related),
        "negative_candidates": len(negatives),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1600, help="Positive-pair cap. Use 0 for all matching records.")
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "cointegrated_strict",
            "cointegrated_diverse",
            "cleaned_leipzig_diverse",
            "merionum_ru_paraphraser",
        ],
        choices=[
            "cointegrated_strict",
            "cointegrated_diverse",
            "cleaned_leipzig_diverse",
            "merionum_ru_paraphraser",
        ],
    )
    args = parser.parse_args()

    eval_sets = load_eval_texts()
    candidate_builders = {
        "cointegrated_strict": lambda: cointegrated_records(limit=args.limit, seed=args.seed, variant="strict"),
        "cointegrated_diverse": lambda: cointegrated_records(limit=args.limit, seed=args.seed + 1, variant="diverse"),
        "cleaned_leipzig_diverse": lambda: cleaned_leipzig_records(limit=args.limit, seed=args.seed + 3, variant="diverse"),
        "merionum_ru_paraphraser": lambda: merionum_records(limit=args.limit, seed=args.seed + 2),
    }

    index = {}
    for name in args.candidates:
        records, build_summary = candidate_builders[name]()
        audited, audit_summary = audit_records(records, eval_sets)
        output = DATA_DIR / f"rusts_external_{name}_{len(audited)}.jsonl"
        summary = REPORT_DIR / f"{output.stem}_summary.json"
        write_jsonl(output, audited)
        write_summary(
            summary,
            {
                "dataset": name,
                "output": str(output.relative_to(ROOT)),
                "records_before_audit": len(records),
                "records_after_audit": len(audited),
                "build_summary": build_summary,
                "audit": audit_summary,
                "contamination_policy": "Exact normalized text overlap with mteb/RuSTSBenchmarkSTS and mteb/CEDRClassification removed. ai-forever/ru-stsbenchmark-sts and mteb/stsb_multi_mt are deliberately excluded.",
            },
        )
        index[name] = {"data_path": str(output.relative_to(ROOT)), "summary_path": str(summary.relative_to(ROOT)), "records": len(audited)}

    write_summary(REPORT_DIR / "rusts_external_ablation_index.json", index)
    for name, item in index.items():
        print(name, item["records"], item["data_path"])


if __name__ == "__main__":
    main()
