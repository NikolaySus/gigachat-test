from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

from run_official_rumteb import GigaOfficialMTEBWrapper, patch_datasets_trust_remote_code


MODEL_REV_0AD = "0ad5b29bfecd806cecc9d66b927d828a736594dc"
BASELINE_FROZEN_LEGACY_RU_0AD = 0.675025
TARGET_SCORE = 0.700000


VARIANTS: dict[str, tuple[str, str]] = {
    "none": ("", ""),
    "legacy_ru_same": ("семантически похожий текст: ", "семантически похожий текст: "),
    "nli_same": (
        "Дана предпосылка, найди гипотезу, которая из нее следует\nтекст: ",
        "Дана предпосылка, найди гипотезу, которая из нее следует\nтекст: ",
    ),
    "premise_hypothesis": ("предпосылка: ", "гипотеза: "),
    "ru_premise_hypothesis": ("Дана предпосылка: ", "Проверь гипотезу: "),
    "entails_direction": (
        "Текст, из которого может следовать утверждение: ",
        "Утверждение, которое может следовать из текста: ",
    ),
    "condition_conclusion": ("условие: ", "вывод: "),
    "nli_role_long": (
        "Предпосылка для задачи логического следования: ",
        "Гипотеза для задачи логического следования: ",
    ),
    "reverse_premise_hypothesis": ("гипотеза: ", "предпосылка: "),
    "reverse_entails_direction": (
        "Утверждение, которое может следовать из текста: ",
        "Текст, из которого может следовать утверждение: ",
    ),
}

PAIR_VARIANTS: dict[str, tuple[str, str, str]] = {
    "pair_ru_entailment_yes_no": (
        "Предпосылка: {sentence1}\nГипотеза: {sentence2}\nОпредели, следует ли гипотеза из предпосылки.",
        "Да, гипотеза следует из предпосылки.",
        "Нет, гипотеза не следует из предпосылки.",
    ),
    "pair_ru_entailment_short": (
        "Предпосылка: {sentence1}\nГипотеза: {sentence2}",
        "следует",
        "не следует",
    ),
    "pair_ru_semantic_nli": (
        "Текст 1: {sentence1}\nТекст 2: {sentence2}\nСвязь между текстами:",
        "второй текст логически следует из первого",
        "второй текст не следует из первого",
    ),
    "pair_ru_true_false": (
        "Если верно: {sentence1}\nМожно заключить: {sentence2}",
        "верно",
        "неверно",
    ),
    "pair_en_entailment": (
        "Premise: {sentence1}\nHypothesis: {sentence2}\nDoes the hypothesis follow from the premise?",
        "The hypothesis follows from the premise.",
        "The hypothesis does not follow from the premise.",
    ),
    "pair_ru_contradiction_aware": (
        "Предпосылка: {sentence1}\nГипотеза: {sentence2}\nНужно отличить логическое следование от противоречия или неизвестности.",
        "логическое следование",
        "противоречие или неизвестность",
    ),
}


def configure_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MTEB_CACHE"] = str(cache_dir.resolve())
    os.environ["HF_HOME"] = str((cache_dir / "hf_home").resolve())
    os.environ["HF_DATASETS_CACHE"] = str((cache_dir / "datasets").resolve())
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    hf_modules = str((cache_dir / "hf_home" / "modules").resolve())
    if hf_modules not in sys.path:
        sys.path.insert(0, hf_modules)
    patch_datasets_trust_remote_code()


def load_terra() -> tuple[list[str], list[str], np.ndarray]:
    import mteb

    benchmark = mteb.get_benchmark("MTEB(rus, v1)")
    task = next(task for task in benchmark.tasks if task.metadata.name == "TERRa")
    task.load_data()
    dataset = task.dataset["dev"]
    if isinstance(dataset, list):
        dataset = dataset[0]
    return list(dataset["sentence1"]), list(dataset["sentence2"]), np.asarray(dataset["labels"], dtype=np.int64)


def ap(scores: np.ndarray, labels: np.ndarray, *, reverse: bool = False) -> float:
    return float(average_precision_score(labels, -scores if reverse else scores))


def compute_scores(emb1: np.ndarray, emb2: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
    manhattan_distances = paired_manhattan_distances(emb1, emb2)
    euclidean_distances = paired_euclidean_distances(emb1, emb2)
    dot_scores = np.sum(emb1 * emb2, axis=1)
    scores = {
        "cosine_ap": ap(cosine_scores, labels),
        "dot_ap": ap(dot_scores, labels),
        "manhattan_ap": ap(manhattan_distances, labels, reverse=True),
        "euclidean_ap": ap(euclidean_distances, labels, reverse=True),
    }
    scores["main_score"] = max(scores.values())
    return scores


def compute_pair_anchor_scores(pair_emb: np.ndarray, positive_emb: np.ndarray, negative_emb: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    positive_scores = pair_emb @ positive_emb
    negative_scores = pair_emb @ negative_emb
    margin_scores = positive_scores - negative_scores
    scores = {
        "positive_ap": ap(positive_scores, labels),
        "negative_ap": ap(negative_scores, labels, reverse=True),
        "margin_ap": ap(margin_scores, labels),
    }
    scores["main_score"] = max(scores.values())
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe TERRa-specific pair-aware wrapper prompts.")
    parser.add_argument("--output", type=Path, default=Path("results/official_repro/terra_pair_wrapper_probe_0ad.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/official_repro_cache"))
    parser.add_argument("--model-revision", default=MODEL_REV_0AD)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32", "auto"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--pair-variants", nargs="*", default=list(PAIR_VARIANTS))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sentence1, sentence2, labels = load_terra()
    model = GigaOfficialMTEBWrapper(
        batch_size=args.batch_size,
        max_length=args.max_length,
        model_revision=args.model_revision,
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
        latent_checkpoint=None,
        task_prompts={"TERRa": ("PairClassification", None)},
        prompt_mode="none",
        symmetric_instruction="none",
    )

    results = {
        "model_revision": args.model_revision,
        "seed": args.seed,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "num_pairs": len(labels),
        "baseline_frozen_legacy_ru_0ad": BASELINE_FROZEN_LEGACY_RU_0AD,
        "target_score": TARGET_SCORE,
        "variants": {},
        "pair_variants": {},
    }
    for name in args.variants:
        if name not in VARIANTS:
            raise SystemExit(f"Unknown variant {name!r}. Available: {', '.join(VARIANTS)}")
        prefix1, prefix2 = VARIANTS[name]
        texts1 = [prefix1 + text for text in sentence1]
        texts2 = [prefix2 + text for text in sentence2]
        emb1 = model._encode_texts(texts1, batch_size=args.batch_size, instruction=None, prompt_mode="none")
        emb2 = model._encode_texts(texts2, batch_size=args.batch_size, instruction=None, prompt_mode="none")
        variant_scores = compute_scores(emb1, emb2, labels)
        variant_scores["prefix1"] = prefix1
        variant_scores["prefix2"] = prefix2
        results["variants"][name] = variant_scores
        delta = variant_scores["main_score"] - BASELINE_FROZEN_LEGACY_RU_0AD
        print(f"{name}: {variant_scores['main_score']:.6f} (delta {delta:+.6f})", flush=True)

    for name in args.pair_variants:
        if name not in PAIR_VARIANTS:
            raise SystemExit(f"Unknown pair variant {name!r}. Available: {', '.join(PAIR_VARIANTS)}")
        template, positive_anchor, negative_anchor = PAIR_VARIANTS[name]
        pair_texts = [
            template.format(sentence1=left, sentence2=right)
            for left, right in zip(sentence1, sentence2)
        ]
        pair_emb = model._encode_texts(pair_texts, batch_size=args.batch_size, instruction=None, prompt_mode="none")
        anchors = model._encode_texts(
            [positive_anchor, negative_anchor],
            batch_size=2,
            instruction=None,
            prompt_mode="none",
        )
        variant_scores = compute_pair_anchor_scores(pair_emb, anchors[0], anchors[1], labels)
        variant_scores["template"] = template
        variant_scores["positive_anchor"] = positive_anchor
        variant_scores["negative_anchor"] = negative_anchor
        results["pair_variants"][name] = variant_scores
        delta = variant_scores["main_score"] - BASELINE_FROZEN_LEGACY_RU_0AD
        target_gap = variant_scores["main_score"] - TARGET_SCORE
        print(f"{name}: {variant_scores['main_score']:.6f} (delta {delta:+.6f}, target {target_gap:+.6f})", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
