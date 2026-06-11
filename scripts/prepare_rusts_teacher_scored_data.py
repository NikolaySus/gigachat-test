from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "official_repro"))

from run_official_rumteb import (  # noqa: E402
    DEFAULT_LEGACY_PREFIX_ENSEMBLES,
    DEFAULT_TASK_BATCH_SIZES,
    DEFAULT_TASK_PROMPT_MODES,
    DEFAULT_TASK_TEXT_NORMALIZATIONS,
    GigaOfficialMTEBWrapper,
)


CACHE_DIR = ROOT / ".cache" / "hf_datasets"
DATA_DIR = ROOT / "data" / "contrastive"
REPORT_DIR = ROOT / "results" / "contamination" / "rusts_external"

RUSTS_PREFIXES = DEFAULT_LEGACY_PREFIX_ENSEMBLES["RuSTSBenchmarkSTS"]


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def strip_prompt(value: str) -> str:
    text = clean_text(value)
    if "\nQuery: " in text:
        return text.split("\nQuery: ", 1)[1]
    for prefix in RUSTS_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def normalize(value: str) -> str:
    text = strip_prompt(value).lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", " ", text).strip()


def is_good_text(value: str, *, min_chars: int = 14, max_chars: int = 700) -> bool:
    text = clean_text(value)
    if not (min_chars <= len(text) <= max_chars):
        return False
    alpha = sum(char.isalpha() for char in text)
    return alpha / max(len(text), 1) >= 0.45


def load_eval_texts() -> set[str]:
    texts: set[str] = set()
    for dataset_name in ("mteb/RuSTSBenchmarkSTS", "mteb/CEDRClassification"):
        dataset = load_dataset(dataset_name, cache_dir=str(CACHE_DIR))
        for split in dataset:
            for row in dataset[split]:
                for key in ("sentence1", "sentence2", "text", "sentence", "query"):
                    text = normalize(row.get(key, ""))
                    if text:
                        texts.add(text)
    return texts


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_pairs(paths: list[Path]) -> list[dict[str, Any]]:
    pairs = []
    seen = set()
    for path in paths:
        for row in iter_jsonl(path):
            source = row.get("source", path.stem)
            if row.get("objective") == "pair_score":
                candidates = [(row.get("sentence1"), row.get("sentence2"), row.get("score"), "pair_score")]
            elif row.get("objective") == "contrastive":
                candidates = [(row.get("query"), row.get("positive"), 1.0, "positive")]
                candidates.extend((row.get("query"), negative, 0.0, "negative") for negative in row.get("negatives", []))
            else:
                continue
            for left, right, original_score, relation in candidates:
                left_text = strip_prompt(clean_text(left))
                right_text = strip_prompt(clean_text(right))
                if not is_good_text(left_text) or not is_good_text(right_text):
                    continue
                key = tuple(sorted((normalize(left_text), normalize(right_text))))
                if key[0] == key[1] or key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "left": left_text,
                        "right": right_text,
                        "source": source,
                        "relation": relation,
                        "original_score": float(original_score) if original_score is not None else None,
                    }
                )
    return pairs


def audit_pairs(pairs: list[dict[str, Any]], eval_texts: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = []
    skipped = 0
    for pair in pairs:
        if normalize(pair["left"]) in eval_texts or normalize(pair["right"]) in eval_texts:
            skipped += 1
            continue
        kept.append(pair)
    return kept, {"input_pairs": len(pairs), "kept_pairs": len(kept), "skipped_eval_exact": skipped}


def make_teacher() -> GigaOfficialMTEBWrapper:
    return GigaOfficialMTEBWrapper(
        batch_size=8,
        max_length=4096,
        model_revision="40b27667b9ad586d7812675df76e5062ccc80b0e",
        attn_implementation="eager",
        torch_dtype="bfloat16",
        local_files_only=True,
        latent_checkpoint=None,
        task_prompts={},
        prompt_mode="legacy_ru",
        symmetric_instruction="none",
        legacy_prefix_ensembles=DEFAULT_LEGACY_PREFIX_ENSEMBLES,
        task_prompt_modes=DEFAULT_TASK_PROMPT_MODES,
        task_text_normalizations=DEFAULT_TASK_TEXT_NORMALIZATIONS,
        task_batch_sizes=DEFAULT_TASK_BATCH_SIZES,
    )


def teacher_scores(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    teacher = make_teacher()
    unique_texts = []
    index = {}
    for pair in pairs:
        for key in ("left", "right"):
            text = pair[key]
            if text not in index:
                index[text] = len(unique_texts)
                unique_texts.append(text)
    embeddings = teacher.encode(unique_texts, task_name="RuSTSBenchmarkSTS")
    for pair in pairs:
        left = embeddings[index[pair["left"]]]
        right = embeddings[index[pair["right"]]]
        cosine = float(np.dot(left, right))
        pair["teacher_cosine"] = cosine
        pair["teacher_score"] = max(0.0, min(1.0, cosine))
    return pairs


def select_pairs(pairs: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positives = [pair for pair in pairs if pair["teacher_score"] >= 0.78]
    mid = [pair for pair in pairs if 0.48 <= pair["teacher_score"] < 0.78]
    negatives = [pair for pair in pairs if pair["teacher_score"] <= 0.28]
    positives.sort(key=lambda pair: pair["teacher_score"], reverse=True)
    mid.sort(key=lambda pair: abs(pair["teacher_score"] - 0.62))
    negatives.sort(key=lambda pair: pair["teacher_score"])

    selected = []
    quotas = [
        (positives, int(limit * 0.45)),
        (mid, int(limit * 0.30)),
        (negatives, limit),
    ]
    for pool, count in quotas:
        selected.extend(pool[:count])
    if len(selected) < limit:
        remaining = [pair for pair in pairs if pair not in selected]
        remaining.sort(key=lambda pair: abs(pair["teacher_score"] - 0.5), reverse=True)
        selected.extend(remaining[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


def write_jsonl(path: Path, pairs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for pair in pairs:
            for prefix in RUSTS_PREFIXES:
                record = {
                    "objective": "pair_score",
                    "source": "teacher_scored_rusts_external",
                    "sentence1": prefix + pair["left"],
                    "sentence2": prefix + pair["right"],
                    "score": pair["teacher_score"],
                    "metadata": {
                        "source": pair["source"],
                        "relation": pair["relation"],
                        "original_score": pair["original_score"],
                        "teacher_cosine": pair["teacher_cosine"],
                    },
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")


def score_bins(pairs: list[dict[str, Any]]) -> dict[str, int]:
    bins = {"0.00-0.28": 0, "0.28-0.48": 0, "0.48-0.78": 0, "0.78-1.00": 0}
    for pair in pairs:
        score = pair["teacher_score"]
        if score <= 0.28:
            bins["0.00-0.28"] += 1
        elif score < 0.48:
            bins["0.28-0.48"] += 1
        elif score < 0.78:
            bins["0.48-0.78"] += 1
        else:
            bins["0.78-1.00"] += 1
    return bins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=5401)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "rusts_external_teacher_scored_clean_12000.jsonl")
    parser.add_argument(
        "--sources",
        nargs="+",
        type=Path,
        default=[
            DATA_DIR / "rusts_external_cointegrated_diverse_3200.jsonl",
            DATA_DIR / "rusts_external_cleaned_leipzig_diverse_3200.jsonl",
            DATA_DIR / "rusts_external_merionum_ru_paraphraser_3198.jsonl",
        ],
    )
    args = parser.parse_args()

    eval_texts = load_eval_texts()
    pairs = collect_pairs(args.sources)
    audited, audit = audit_pairs(pairs, eval_texts)
    scored = teacher_scores(audited)
    selected = select_pairs(scored, limit=args.limit, seed=args.seed)
    write_jsonl(args.output, selected)
    summary = {
        "output": str(args.output.relative_to(ROOT)),
        "source_files": [str(path.relative_to(ROOT)) for path in args.sources],
        "audit": audit,
        "scored_pairs": len(scored),
        "selected_pairs": len(selected),
        "written_records": len(selected) * len(RUSTS_PREFIXES),
        "score_bins_all": score_bins(scored),
        "score_bins_selected": score_bins(selected),
        "score_min_selected": min(pair["teacher_score"] for pair in selected),
        "score_mean_selected": sum(pair["teacher_score"] for pair in selected) / len(selected),
        "score_max_selected": max(pair["teacher_score"] for pair in selected),
        "contamination_policy": "Exact normalized text overlap with mteb/RuSTSBenchmarkSTS and mteb/CEDRClassification removed before teacher scoring. Direct STS benchmark datasets are not used.",
    }
    summary_path = REPORT_DIR / f"{args.output.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
