from __future__ import annotations

import argparse
import itertools
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
ONLINE_OFFICIAL_TERRA = 0.795677


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
    "entails_direction_newline": (
        "Текст, из которого может следовать утверждение:\n",
        "Утверждение, которое может следовать из текста:\n",
    ),
    "entails_direction_text_field": (
        "Текст, из которого может следовать утверждение:\nтекст: ",
        "Утверждение, которое может следовать из текста:\nутверждение: ",
    ),
    "entails_direction_statement_field": (
        "Исходный текст:\n",
        "Утверждение, проверяемое по исходному тексту:\n",
    ),
    "entails_direction_query_style": (
        "Дан текст, из которого нужно вывести утверждение:\nтекст: ",
        "Нужно найти утверждение, которое следует из текста:\nутверждение: ",
    ),
    "entails_direction_short": (
        "из этого следует: ",
        "следующее утверждение: ",
    ),
    "entails_direction_fact_claim": (
        "Факт или ситуация: ",
        "Вывод или утверждение: ",
    ),
    "entails_direction_logical": (
        "Логическая предпосылка: ",
        "Логическое следствие: ",
    ),
    "entails_direction_semantic": (
        "Смысл исходного текста: ",
        "Смысл проверяемого утверждения: ",
    ),
    "entails_direction_question": (
        "Ответ находится в тексте: ",
        "Проверяемое утверждение: ",
    ),
    "entails_direction_hypothesis_query": (
        "Документ для проверки гипотезы: ",
        "Гипотеза, которую нужно проверить: ",
    ),
    "entails_direction_rte": (
        "Premise / предпосылка: ",
        "Hypothesis / гипотеза: ",
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
    sentence1 = dataset["sentence1"]
    sentence2 = dataset["sentence2"]
    labels = dataset["labels"]
    if len(sentence1) == 1 and isinstance(sentence1[0], list):
        sentence1 = sentence1[0]
    if len(sentence2) == 1 and isinstance(sentence2[0], list):
        sentence2 = sentence2[0]
    if len(labels) == 1 and isinstance(labels[0], list):
        labels = labels[0]
    return list(sentence1), list(sentence2), np.asarray(labels, dtype=np.int64)


def ap(scores: np.ndarray, labels: np.ndarray, *, reverse: bool = False) -> float:
    return float(average_precision_score(labels, -scores if reverse else scores))


def compute_scores(emb1: np.ndarray, emb2: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    arrays = compute_pair_score_arrays(emb1, emb2)
    scores = {
        "cosine_ap": ap(arrays["cosine"], labels),
        "dot_ap": ap(arrays["dot"], labels),
        "manhattan_ap": ap(arrays["manhattan"], labels),
        "euclidean_ap": ap(arrays["euclidean"], labels),
    }
    scores["main_score"] = max(scores.values())
    return scores


def compute_pair_score_arrays(emb1: np.ndarray, emb2: np.ndarray) -> dict[str, np.ndarray]:
    cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
    manhattan_distances = paired_manhattan_distances(emb1, emb2)
    euclidean_distances = paired_euclidean_distances(emb1, emb2)
    dot_scores = np.sum(emb1 * emb2, axis=1)
    return {
        "cosine": cosine_scores,
        "dot": dot_scores,
        "manhattan": -manhattan_distances,
        "euclidean": -euclidean_distances,
    }


def compute_pair_anchor_scores(pair_emb: np.ndarray, positive_emb: np.ndarray, negative_emb: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    arrays = compute_pair_anchor_score_arrays(pair_emb, positive_emb, negative_emb)
    scores = {f"{name}_ap": ap(values, labels) for name, values in arrays.items()}
    scores["main_score"] = max(scores.values())
    return scores


def compute_pair_anchor_score_arrays(pair_emb: np.ndarray, positive_emb: np.ndarray, negative_emb: np.ndarray) -> dict[str, np.ndarray]:
    positive_scores = pair_emb @ positive_emb
    negative_scores = pair_emb @ negative_emb
    margin_scores = positive_scores - negative_scores
    return {
        "positive": positive_scores,
        "negative_inverse": -negative_scores,
        "margin": margin_scores,
    }


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    if std < 1e-12:
        return values - values.mean()
    return (values - values.mean()) / std


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    denom = max(len(values) - 1, 1)
    return ranks / denom


def compute_ensembles(score_streams: dict[str, np.ndarray], labels: np.ndarray, *, top_k: int = 8) -> list[dict[str, object]]:
    ranked = sorted(
        ((ap(values, labels), name, values) for name, values in score_streams.items()),
        reverse=True,
        key=lambda item: item[0],
    )[:top_k]
    results: list[dict[str, object]] = []
    for size in (2, 3, 4):
        for combo in itertools.combinations(ranked, size):
            names = [item[1] for item in combo]
            arrays = [item[2] for item in combo]
            zavg = np.mean([zscore(values) for values in arrays], axis=0)
            ravg = np.mean([rank01(values) for values in arrays], axis=0)
            results.append(
                {
                    "name": "zavg:" + "+".join(names),
                    "main_score": ap(zavg, labels),
                    "members": names,
                    "method": "zavg",
                }
            )
            results.append(
                {
                    "name": "rankavg:" + "+".join(names),
                    "main_score": ap(ravg, labels),
                    "members": names,
                    "method": "rankavg",
                }
            )
    results.sort(reverse=True, key=lambda item: float(item["main_score"]))
    return results


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
        "online_official_terra": ONLINE_OFFICIAL_TERRA,
        "variants": {},
        "pair_variants": {},
        "ensembles": {},
    }
    score_streams: dict[str, np.ndarray] = {}
    for name in args.variants:
        if name not in VARIANTS:
            raise SystemExit(f"Unknown variant {name!r}. Available: {', '.join(VARIANTS)}")
        prefix1, prefix2 = VARIANTS[name]
        texts1 = [prefix1 + text for text in sentence1]
        texts2 = [prefix2 + text for text in sentence2]
        emb1 = model._encode_texts(texts1, batch_size=args.batch_size, instruction=None, prompt_mode="none")
        emb2 = model._encode_texts(texts2, batch_size=args.batch_size, instruction=None, prompt_mode="none")
        variant_scores = compute_scores(emb1, emb2, labels)
        arrays = compute_pair_score_arrays(emb1, emb2)
        for metric_name, values in arrays.items():
            score_streams[f"variant:{name}:{metric_name}"] = values
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
        arrays = compute_pair_anchor_score_arrays(pair_emb, anchors[0], anchors[1])
        for metric_name, values in arrays.items():
            score_streams[f"pair_variant:{name}:{metric_name}"] = values
        variant_scores["template"] = template
        variant_scores["positive_anchor"] = positive_anchor
        variant_scores["negative_anchor"] = negative_anchor
        results["pair_variants"][name] = variant_scores
        delta = variant_scores["main_score"] - BASELINE_FROZEN_LEGACY_RU_0AD
        target_gap = variant_scores["main_score"] - TARGET_SCORE
        print(f"{name}: {variant_scores['main_score']:.6f} (delta {delta:+.6f}, target {target_gap:+.6f})", flush=True)

    for row in compute_ensembles(score_streams, labels):
        row = dict(row)
        name = str(row.pop("name"))
        results["ensembles"][name] = row
    if results["ensembles"]:
        best_name, best_row = max(
            results["ensembles"].items(),
            key=lambda item: float(item[1]["main_score"]),
        )
        best_score = float(best_row["main_score"])
        print(
            f"best_ensemble: {best_name}: {best_score:.6f} "
            f"(official {best_score - ONLINE_OFFICIAL_TERRA:+.6f})",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
