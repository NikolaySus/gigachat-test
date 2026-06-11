from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[\wа-яА-ЯёЁ]+", re.UNICODE)


@dataclass(frozen=True)
class Variant:
    name: str
    min_chars: int
    max_chars: int
    max_repetition_ratio: float
    max_bad_char_ratio: float
    min_alpha_ratio: float
    dedupe_texts: bool
    prune_false_negatives: bool
    drop_if_no_negative: bool
    max_negative_to_positive_ratio: float
    max_negative_to_query_ratio: float
    min_query_positive_ratio: float
    strict_sensitive: bool


VARIANTS: dict[str, Variant] = {
    "basic": Variant(
        name="basic",
        min_chars=24,
        max_chars=4096 * 5,
        max_repetition_ratio=0.42,
        max_bad_char_ratio=0.08,
        min_alpha_ratio=0.18,
        dedupe_texts=True,
        prune_false_negatives=False,
        drop_if_no_negative=False,
        max_negative_to_positive_ratio=1.0,
        max_negative_to_query_ratio=1.0,
        min_query_positive_ratio=0.0,
        strict_sensitive=False,
    ),
    "sim_guard": Variant(
        name="sim_guard",
        min_chars=24,
        max_chars=4096 * 5,
        max_repetition_ratio=0.40,
        max_bad_char_ratio=0.07,
        min_alpha_ratio=0.20,
        dedupe_texts=True,
        prune_false_negatives=True,
        drop_if_no_negative=True,
        max_negative_to_positive_ratio=0.92,
        max_negative_to_query_ratio=0.96,
        min_query_positive_ratio=0.0,
        strict_sensitive=False,
    ),
    "strict": Variant(
        name="strict",
        min_chars=32,
        max_chars=4096 * 4,
        max_repetition_ratio=0.34,
        max_bad_char_ratio=0.05,
        min_alpha_ratio=0.24,
        dedupe_texts=True,
        prune_false_negatives=True,
        drop_if_no_negative=True,
        max_negative_to_positive_ratio=0.88,
        max_negative_to_query_ratio=0.94,
        min_query_positive_ratio=0.015,
        strict_sensitive=True,
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalized_key(text: str) -> str:
    return normalize_text(text).casefold()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalized_key(text).encode("utf-8")).hexdigest()


def text_fields(record: dict[str, Any]) -> list[str]:
    if record.get("objective", "contrastive") == "contrastive":
        return [record.get("query", ""), record.get("positive", ""), *record.get("negatives", [])]
    return [record.get("sentence1", ""), record.get("sentence2", "")]


def repetition_ratio(text: str) -> float:
    words = [w.casefold() for w in WORD_RE.findall(text)]
    if len(words) < 8:
        return 0.0
    counts = Counter(words)
    return counts.most_common(1)[0][1] / len(words)


def bad_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = 0
    for char in text:
        category = unicodedata.category(char)
        if category[0] == "C" and char not in "\n\t":
            bad += 1
        elif char == "\ufffd":
            bad += 1
    return bad / len(text)


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if char.isalpha()) / len(text)


def char_ngrams(text: str, n: int = 4) -> set[str]:
    text = normalized_key(text)
    if len(text) <= n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def lexical_similarity(left: str, right: str) -> float:
    left_key = normalized_key(left)
    right_key = normalized_key(right)
    if not left_key or not right_key:
        return 0.0
    left_grams = char_ngrams(left_key)
    right_grams = char_ngrams(right_key)
    if not left_grams or not right_grams:
        return SequenceMatcher(None, left_key, right_key).ratio()
    overlap = len(left_grams & right_grams)
    union = len(left_grams | right_grams)
    jaccard = overlap / union if union else 0.0
    if jaccard < 0.25:
        return jaccard
    return max(jaccard, SequenceMatcher(None, left_key, right_key).ratio())


def text_quality_ok(text: str, variant: Variant) -> tuple[bool, str | None]:
    text = normalize_text(text)
    if len(text) < variant.min_chars:
        return False, "too_short"
    if len(text) > variant.max_chars:
        return False, "too_long"
    if bad_char_ratio(text) > variant.max_bad_char_ratio:
        return False, "bad_chars"
    if alpha_ratio(text) < variant.min_alpha_ratio:
        return False, "low_alpha"
    if repetition_ratio(text) > variant.max_repetition_ratio:
        return False, "repetition"
    return True, None


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    record = dict(record)
    objective = record.get("objective", "contrastive")
    record["objective"] = objective
    if objective == "contrastive":
        record["query"] = normalize_text(record.get("query", ""))
        record["positive"] = normalize_text(record.get("positive", ""))
        record["negatives"] = [normalize_text(x) for x in record.get("negatives", [])]
    else:
        record["sentence1"] = normalize_text(record.get("sentence1", ""))
        record["sentence2"] = normalize_text(record.get("sentence2", ""))
    return record


def record_signature(record: dict[str, Any]) -> str:
    fields = [text_hash(text) for text in text_fields(record)]
    return "|".join([record.get("source", ""), record.get("objective", "contrastive"), *fields])


def filter_records(
    records: list[dict[str, Any]],
    *,
    variant: Variant,
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    pruned_negatives = 0
    seen_records: set[str] = set()
    seen_text_pairs: set[tuple[str, str]] = set()

    for raw in records:
        record = normalize_record(raw)
        signature = record_signature(record)
        if signature in seen_records:
            rejected["duplicate_record"] += 1
            continue
        seen_records.add(signature)

        objective = record.get("objective", "contrastive")
        fields = text_fields(record)
        failed_reason = None
        for text in fields[:2]:
            ok, reason = text_quality_ok(text, variant)
            if not ok:
                failed_reason = f"main_{reason}"
                break
        if failed_reason:
            rejected[failed_reason] += 1
            continue

        if objective == "contrastive":
            query = record["query"]
            positive = record["positive"]
            pair_key = (text_hash(query), text_hash(positive))
            reverse_pair_key = (pair_key[1], pair_key[0])
            if variant.dedupe_texts and (pair_key in seen_text_pairs or reverse_pair_key in seen_text_pairs):
                rejected["duplicate_pair"] += 1
                continue
            seen_text_pairs.add(pair_key)

            qp_similarity = lexical_similarity(query, positive)
            if qp_similarity < variant.min_query_positive_ratio:
                rejected["low_query_positive_similarity"] += 1
                continue

            filtered_negatives = []
            for negative in record.get("negatives", []):
                ok, reason = text_quality_ok(negative, variant)
                if not ok:
                    rejected[f"negative_{reason}"] += 1
                    pruned_negatives += 1
                    continue
                if variant.prune_false_negatives:
                    neg_pos = lexical_similarity(negative, positive)
                    neg_query = lexical_similarity(negative, query)
                    if neg_pos >= variant.max_negative_to_positive_ratio or neg_query >= variant.max_negative_to_query_ratio:
                        rejected["suspected_false_negative"] += 1
                        pruned_negatives += 1
                        continue
                filtered_negatives.append(negative)
            record["negatives"] = filtered_negatives
            if variant.drop_if_no_negative and not filtered_negatives:
                rejected["no_negative_after_filter"] += 1
                continue
            if variant.strict_sensitive and "sensitive" in source_name:
                # Keep sensitive-topic records where the positive is not almost a paraphrase of the query.
                # The task signal should be fine-grained topic discrimination, not exact conversational echo.
                if qp_similarity > 0.98:
                    rejected["sensitive_near_duplicate_positive"] += 1
                    continue
        kept.append(record)

    summary = {
        "source": source_name,
        "input_records": len(records),
        "kept_records": len(kept),
        "rejected_records": len(records) - len(kept),
        "pruned_negatives": pruned_negatives,
        "rejection_reasons": dict(sorted(rejected.items())),
    }
    return kept, summary


def shuffled(records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    records = records[:]
    random.Random(seed).shuffle(records)
    return records


def build_variant(args: argparse.Namespace, variant: Variant) -> dict[str, Any]:
    source_paths = {
        "geracl": args.geracl_path,
        "habr": args.habr_path,
        "deepvk": args.deepvk_path,
        "grounded": args.grounded_path,
        "sensitive": args.sensitive_path,
    }
    filtered: dict[str, list[dict[str, Any]]] = {}
    source_summaries = []
    for offset, (name, path) in enumerate(source_paths.items()):
        records = shuffled(read_jsonl(path), seed=args.seed + offset)
        kept, summary = filter_records(records, variant=variant, source_name=name)
        filtered[name] = kept
        source_summaries.append(summary)

    geracl = shuffled(filtered["geracl"], seed=args.seed + 20)
    geracl_stage1_count = min(args.geracl_stage1_count, len(geracl))
    geracl_stage1 = geracl[:geracl_stage1_count]
    geracl_remaining = geracl[geracl_stage1_count:]

    stage1 = []
    for name in ("geracl", "habr", "deepvk", "grounded", "sensitive"):
        if name == "geracl":
            stage1.extend(geracl_stage1)
        else:
            stage1.extend(filtered[name])
    stage1 = shuffled(stage1, seed=args.seed + 30)

    suffix = variant.name
    stage1_out = args.output_dir / f"open_ru_1r_nc_mixm_{suffix}_full_quality_stage1_{len(stage1)}.jsonl"
    remaining_out = args.output_dir / f"open_ru_1r_nc_mixm_{suffix}_geracl_remaining_{len(geracl_remaining)}.jsonl"
    summary_out = args.output_dir / f"open_ru_1r_nc_mixm_{suffix}_summary.json"
    write_jsonl(stage1_out, stage1)
    write_jsonl(remaining_out, geracl_remaining)

    stage1_steps = max(1, len(stage1) // args.batch_size_stage1)
    stage2_steps = max(1, len(geracl_remaining) // args.batch_size_stage2)
    config = {
        "name": f"exp01r_nc_mixm_{suffix}_full_quality_4096",
        "description": (
            "Fair 1R-NC Mix M quality-filter ablation. Stage 1 uses GeRaCl:Habr:DeepVK:"
            "Grounded:Sensitive as 2:1:1:1:1 base without arbitrary non-GeRaCl caps; "
            "Habr/DeepVK/Grounded/Sensitive are consumed fully after quality filtering, "
            "and stage 2 consumes the remaining GeRaCl rows."
        ),
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "flash_attention_2",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": None,
        "freeze_llm": True,
        "reinit_latent": True,
        "data_path": str(stage1_out),
        "output_dir": f"experiments/exp01_reinit_fair/checkpoints/open_ru_1r_nc_mixm_{suffix}_full_quality_4096",
        "max_length": 4096,
        "batch_size": args.batch_size_stage1,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "temperature": 0.02,
        "max_steps": stage1_steps + stage2_steps,
        "log_every": 50,
        "save_every": stage1_steps,
        "seed": args.seed,
        "stages": [
            {
                "name": f"mixm_{suffix}_stage1_full_quality",
                "data_path": str(stage1_out),
                "max_steps": stage1_steps,
                "batch_size": args.batch_size_stage1,
                "save_every": stage1_steps,
            },
            {
                "name": f"mixm_{suffix}_geracl_remaining",
                "data_path": str(remaining_out),
                "max_steps": stage2_steps,
                "batch_size": args.batch_size_stage2,
                "learning_rate": 3e-6,
                "save_every": stage2_steps,
            },
        ],
    }
    config_out = args.config_dir / f"exp01r_nc_mixm_{suffix}_full_quality_4096.json"
    write_json(config_out, config)

    summary = {
        "variant": variant.__dict__,
        "seed": args.seed,
        "stage1_output": str(stage1_out),
        "geracl_remaining_output": str(remaining_out),
        "config_output": str(config_out),
        "counts": {
            "geracl_stage1": len(geracl_stage1),
            "geracl_remaining": len(geracl_remaining),
            "habr": len(filtered["habr"]),
            "deepvk": len(filtered["deepvk"]),
            "grounded": len(filtered["grounded"]),
            "sensitive": len(filtered["sensitive"]),
            "stage1_total": len(stage1),
            "total_training_records": len(stage1) + len(geracl_remaining),
            "stage1_steps_1x": stage1_steps,
            "stage2_steps_1x": stage2_steps,
        },
        "source_summaries": source_summaries,
        "source_paths": {name: str(path) for name, path in source_paths.items()},
    }
    write_json(summary_out, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full-use Mix M quality-filter variants.")
    parser.add_argument("--geracl-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_geracl.jsonl"))
    parser.add_argument("--habr-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"))
    parser.add_argument("--deepvk-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl"))
    parser.add_argument("--grounded-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_grounded_rag_v2_q180_doc1200_neg2.jsonl"))
    parser.add_argument("--sensitive-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_sensitive_topic_mvrcii_uc_berkeley_3200.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/contrastive"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--variants", nargs="+", default=["basic", "sim_guard", "strict"], choices=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--geracl-stage1-count", type=int, default=6400)
    parser.add_argument("--batch-size-stage1", type=int, default=4)
    parser.add_argument("--batch-size-stage2", type=int, default=8)
    args = parser.parse_args()

    all_summaries = {}
    for name in args.variants:
        summary = build_variant(args, VARIANTS[name])
        all_summaries[name] = summary
        print(
            f"{name}: stage1={summary['counts']['stage1_total']} "
            f"remaining={summary['counts']['geracl_remaining']} "
            f"steps={summary['counts']['stage1_steps_1x']}+{summary['counts']['stage2_steps_1x']}"
        )
    write_json(args.output_dir / "open_ru_1r_nc_mixm_quality_variants_summary.json", all_summaries)


if __name__ == "__main__":
    main()
