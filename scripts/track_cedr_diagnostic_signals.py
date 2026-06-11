from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPRO = ROOT / "official_repro"
if str(OFFICIAL_REPRO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPRO))

from run_official_rumteb import (  # noqa: E402
    DEFAULT_LEGACY_PREFIX_ENSEMBLES,
    DEFAULT_MTEB_PROMPT_OVERRIDES,
    DEFAULT_TASK_BATCH_SIZES,
    DEFAULT_TASK_PROMPT_MODES,
    DEFAULT_TASK_SEEDS,
    DEFAULT_TASK_TEXT_NORMALIZATIONS,
    GigaOfficialMTEBWrapper,
    MODEL_REVISION,
)


LABEL_NAMES = {
    0: "joy",
    1: "sadness",
    2: "surprise",
    3: "fear",
    4: "anger",
}
NEUTRAL = "neutral"

DEFAULT_MODELS = {
    "base_mixh_habrfull": ROOT
    / "experiments/exp01_reinit_fair/checkpoints/open_ru_1r_nc_mixh_habrfull_plus_geracl_remaining_4096_eager_frozenrepro/latest.pt",
    "retained_best": ROOT
    / "experiments/exp01_reinit_fair/checkpoints/cedr_correction_lenta_reported_replay_go9000_from_retainedbest_lr1e7_anchor10_reh1_300_4096_eager_frozenrepro/step-150.pt",
    "bad_no_geracl": ROOT
    / "experiments/exp01_reinit_fair/checkpoints/mixh_habrfull_leave1out_no_geracl_4096_eager_frozenrepro/latest.pt",
    "a050_extrapolated": ROOT
    / "experiments/exp01_reinit_fair/checkpoints/cedr_goal_best_minus_nogeracl_a050/latent.pt",
    "contam_knn_episode": ROOT
    / "experiments/exp01_reinit_fair/checkpoints/CONTAMINATED_cedr_all_knn_episode_4096_eager_frozenrepro/step-3200.pt",
}

SLANG_PATTERNS = [
    r"\b(лол|кек|имхо|хз|омг|жесть|капец|пипец|блин|черт|вау|офиг|афиг)\b",
    r"\b(ахах+|хаха+|ха-ха|ору|ржу)\b",
    r"(?::|=|;)-?[)(DPpOo/\\]",
    r"[)(]{2,}",
]

FEATURE_PATTERNS = {
    "emotion_joy_lexeme": [
        r"\b(рад|рада|рады|радость|счаст|улыб|весел|класс|ура|кайф|люблю)\w*\b",
    ],
    "emotion_sadness_lexeme": [
        r"\b(груст|печал|тоск|плак|слез|жалко|скорб|уныл)\w*\b",
    ],
    "emotion_surprise_lexeme": [
        r"\b(удив|неожидан|шок|офиг|афиг|ого|ничего себе|внезап)\w*\b",
    ],
    "emotion_fear_lexeme": [
        r"\b(страш|боюсь|боят|ужас|паник|тревог|опасн|кошмар)\w*\b",
    ],
    "emotion_anger_lexeme": [
        r"\b(зл|гнев|бесит|ненави|ярост|раздраж|бесил|возмущ)\w*\b",
    ],
    "reported_speech": [
        r"\b(сказал|сказала|сообщил|заявил|отметил|рассказал|пишет|говорит|по словам)\w*\b",
    ],
    "negative_topic": [
        r"\b(смерт|убит|войн|теракт|авари|катастроф|пожар|болезн|рак|насил|преступ)\w*\b",
    ],
    "first_person": [
        r"\b(я|мне|меня|мой|моя|моё|мои|мы|нам|нас|наш|наша)\b",
    ],
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    return value


def label_name(labels: list[int] | tuple[int, ...]) -> str:
    if not labels:
        return NEUTRAL
    return "+".join(LABEL_NAMES.get(int(label), str(label)) for label in labels)


def prediction_name(row: np.ndarray, classes: np.ndarray) -> str:
    labels = [int(classes[index]) for index, value in enumerate(row) if value]
    return label_name(labels)


def checkpoint_hash(path: Path | None) -> str:
    if path is None:
        return "released"
    resolved = str(path.resolve())
    stat = path.stat()
    return hashlib.sha1(f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:12]


def compile_any(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, flags=re.IGNORECASE | re.UNICODE) for pattern in patterns]


COMPILED_SLANG = compile_any(SLANG_PATTERNS)
COMPILED_FEATURES = {key: compile_any(patterns) for key, patterns in FEATURE_PATTERNS.items()}


def has_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def text_features(text: str, labels: list[int] | tuple[int, ...]) -> dict[str, Any]:
    stripped = " ".join(str(text).split())
    lower = stripped.lower()
    chars = len(stripped)
    words = re.findall(r"[\wёЁ]+", stripped, flags=re.UNICODE)
    uppercase = sum(1 for char in stripped if char.isupper())
    letters = sum(1 for char in stripped if char.isalpha())
    features: dict[str, Any] = {
        "chars": chars,
        "words": len(words),
        "label_cardinality": len(labels),
        "true_label": label_name(labels),
        "is_neutral": len(labels) == 0,
        "has_slang_or_emoticon": has_any(lower, COMPILED_SLANG),
        "has_exclamation": "!" in stripped,
        "has_question": "?" in stripped,
        "uppercase_ratio": uppercase / max(1, letters),
    }
    for name, patterns in COMPILED_FEATURES.items():
        features[f"has_{name}"] = has_any(lower, patterns)
    features["emotion_lexeme_count"] = sum(
        int(features[f"has_emotion_{name}_lexeme"])
        for name in ["joy", "sadness", "surprise", "fear", "anger"]
    )
    features["neutral_with_emotion_lexeme"] = bool(features["is_neutral"] and features["emotion_lexeme_count"] > 0)
    features["reported_negative_topic"] = bool(features["has_reported_speech"] and features["has_negative_topic"])
    return features


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([1.0 if row.get(key) else 0.0 for row in rows]))


def feature_enrichment(group: list[dict[str, Any]], reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bool_keys = [
        "is_neutral",
        "has_slang_or_emoticon",
        "has_exclamation",
        "has_question",
        "has_reported_speech",
        "has_negative_topic",
        "has_first_person",
        "neutral_with_emotion_lexeme",
        "reported_negative_topic",
        "has_emotion_joy_lexeme",
        "has_emotion_sadness_lexeme",
        "has_emotion_surprise_lexeme",
        "has_emotion_fear_lexeme",
        "has_emotion_anger_lexeme",
    ]
    rows = []
    for key in bool_keys:
        group_rate = mean_bool(group, key)
        ref_rate = mean_bool(reference, key)
        rows.append({"feature": key, "group_rate": group_rate, "all_rate": ref_rate, "delta": group_rate - ref_rate})
    rows.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return rows


def setup_mteb(seed: int) -> tuple[Any, Any, Any, list[list[int]], list[int], dict[str, tuple[str, str | None]]]:
    cache_dir = ROOT / "results" / "official_repro_cache"
    os.environ.setdefault("MTEB_CACHE", str(cache_dir.resolve()))
    os.environ.setdefault("HF_HOME", str((cache_dir / "hf_home").resolve()))
    os.environ.setdefault("HF_DATASETS_CACHE", str((cache_dir / "datasets").resolve()))
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    hf_modules = str((cache_dir / "hf_home" / "modules").resolve())
    if hf_modules not in sys.path:
        sys.path.insert(0, hf_modules)

    import mteb

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    benchmark = mteb.get_benchmark("MTEB(rus, v1)")
    task_by_name = {task.metadata.name: task for task in benchmark.tasks}
    task = task_by_name["CEDRClassification"]
    task.seed = DEFAULT_TASK_SEEDS.get(task.metadata.name, seed)
    task.load_data()

    task_prompts = {
        item.metadata.name: (str(item.metadata.type), getattr(item.metadata, "prompt", None))
        for item in benchmark.tasks
    }
    for task_name, prompt in DEFAULT_MTEB_PROMPT_OVERRIDES.items():
        task_type, _ = task_prompts.get(task_name, ("", None))
        task_prompts[task_name] = (task_type, prompt)

    # Keep the RNG path aligned with official_repro/run_official_rumteb.py.
    _official_evaluator_rng_alignment = mteb.MTEB(tasks=[task])

    ds = task.dataset["default"] if "default" in task.dataset else task.dataset
    train_split = ds["train"]
    test_split = ds["test"]

    train_samples: list[list[int]] = []
    for _ in range(task.n_experiments):
        sample_indices, _ = task._undersample_data_indices(
            train_split["label"], task.samples_per_label, None
        )
        train_samples.append(sample_indices)

    unique_train_indices = list(set(itertools.chain.from_iterable(train_samples)))
    return task, train_split, test_split, train_samples, unique_train_indices, task_prompts


def encode_checkpoint(
    *,
    label: str,
    latent_checkpoint: Path | None,
    train_texts: list[str],
    test_texts: list[str],
    task_prompts: dict[str, tuple[str, str | None]],
    batch_size: int,
    max_length: int,
    attn_implementation: str,
    torch_dtype: str,
    local_files_only: bool,
    cache_dir: Path,
    reuse_cache: bool,
    encode_task_name: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    task_cache_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", encode_task_name or "none")
    cache_path = cache_dir / f"{label}_{task_cache_key}_{checkpoint_hash(latent_checkpoint)}.npz"
    if reuse_cache and cache_path.exists():
        cached = np.load(cache_path)
        return cached["train"], cached["test"]

    model = GigaOfficialMTEBWrapper(
        batch_size=batch_size,
        max_length=max_length,
        model_revision=MODEL_REVISION,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        latent_checkpoint=latent_checkpoint,
        task_prompts=task_prompts,
        prompt_mode="legacy_ru",
        symmetric_instruction="mteb",
        legacy_prefix_ensembles=dict(DEFAULT_LEGACY_PREFIX_ENSEMBLES),
        task_prompt_modes=dict(DEFAULT_TASK_PROMPT_MODES),
        task_text_normalizations=DEFAULT_TASK_TEXT_NORMALIZATIONS,
        task_batch_sizes=DEFAULT_TASK_BATCH_SIZES,
    )
    encode_kwargs = {"task_name": encode_task_name} if encode_task_name else {}
    train_embeddings = model.encode(train_texts, **encode_kwargs)
    test_embeddings = model.encode(test_texts, **encode_kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, train=train_embeddings, test=test_embeddings)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train_embeddings, test_embeddings


def evaluate_embeddings(
    *,
    task: Any,
    train_split: Any,
    test_split: Any,
    train_samples: list[list[int]],
    unique_train_indices: list[int],
    train_embeddings_raw: np.ndarray,
    test_embeddings: np.ndarray,
    binarizer: MultiLabelBinarizer,
    classes: np.ndarray,
) -> dict[str, Any]:
    train_embeddings = {
        index: train_embeddings_raw[position]
        for position, index in enumerate(unique_train_indices)
    }
    y_test = binarizer.transform(test_split["label"])
    per_test = [
        {"correct": 0, "predictions": [], "neighbor_labels": []}
        for _ in range(len(test_split))
    ]
    accuracies = []
    f1s = []
    confusions: Counter[str] = Counter()
    wrong_by_true: Counter[str] = Counter()
    pred_counter: Counter[str] = Counter()
    true_names = [label_name(labels) for labels in test_split["label"]]

    for experiment_index, sample_indices in enumerate(train_samples, start=1):
        classifier = clone(task.evaluator_model)
        x_train = np.stack([train_embeddings[index] for index in sample_indices])
        y_train = binarizer.transform(train_split.select(sample_indices)["label"])
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(test_embeddings)
        exact = (y_pred == y_test).all(axis=1)
        accuracies.append(float(classifier.score(test_embeddings, y_test)))
        f1s.append(float(f1_score(y_test, y_pred, average="macro", zero_division=0)))
        distances, positions = classifier.kneighbors(test_embeddings, n_neighbors=int(task.evaluator_model.get_params().get("n_neighbors", 5)))

        for index, ok in enumerate(exact):
            pred_name = prediction_name(y_pred[index], classes)
            pred_counter[pred_name] += 1
            neighbor_labels = [
                label_name(train_split[int(sample_indices[int(position)])]["label"])
                for position in positions[index]
            ]
            per_test[index]["predictions"].append(pred_name)
            per_test[index]["neighbor_labels"].append(neighbor_labels)
            if ok:
                per_test[index]["correct"] += 1
            else:
                wrong_by_true[true_names[index]] += 1
                confusions[f"{true_names[index]} -> {pred_name}"] += 1

    for row in per_test:
        row["correct_rate"] = row["correct"] / max(1, len(train_samples))
        row["main_prediction"] = Counter(row["predictions"]).most_common(1)[0][0]
        row["main_neighbor_signature"] = Counter(
            "|".join(labels) for labels in row["neighbor_labels"]
        ).most_common(1)[0][0]

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_f1": float(np.mean(f1s)),
        "wrong_by_true": dict(wrong_by_true),
        "top_confusions": confusions.most_common(30),
        "predicted_label_counts": dict(pred_counter),
        "per_test": per_test,
    }


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# CEDR Diagnostic Signal Tracking",
        "",
        "This is intentionally allowed to look at CEDR rows. Use it only as a diagnostic source for hypotheses and for constructing clean analog datasets.",
        "",
        "Protocol note: this script manually reconstructs the CEDR 5-NN multilabel evaluator under the frozen wrapper. Treat row-level flips as the primary signal; absolute scores can differ slightly from historical full-MTEB JSONs when the local MTEB package changes.",
        "",
        "## Checkpoint Scores",
        "",
        "| Model | Mean accuracy | Mean macro-F1 | Wrong rows per 10 experiments |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in summary["models"].items():
        wrong_total = sum(metrics["wrong_by_true"].values())
        lines.append(f"| {name} | {metrics['mean_accuracy']:.6f} | {metrics['mean_f1']:.6f} | {wrong_total} |")

    lines.extend(["", "## Flip Sets", "", "| Set | Count | Meaning |", "|---|---:|---|"])
    meanings = {
        "a050_fixed_base": "a050 is correct more often than base",
        "a050_hurts_base": "a050 is worse than base",
        "bad_hurts_base": "bad/no-GeRaCl is worse than base",
        "retained_fixed_base": "retained_best is better than base",
        "contam_fixed_base": "contaminated KNN-episode model is better than base",
        "a050_only_strong": "a050 is mostly correct while base, bad, and retained are mostly wrong",
    }
    for key, rows in summary["flip_sets"].items():
        lines.append(f"| {key} | {len(rows)} | {meanings.get(key, '')} |")

    lines.extend(["", "## a050 Fixed Base: Top True Labels", "", "| Label | Count |", "|---|---:|"])
    for key, count in summary["diagnostics"]["a050_fixed_true_labels"]:
        lines.append(f"| {key} | {count} |")

    lines.extend(["", "## a050 Fixed Base: Top Base Confusions", "", "| Confusion | Count |", "|---|---:|"])
    for key, count in summary["diagnostics"]["a050_fixed_base_confusions"]:
        lines.append(f"| {key} | {count} |")

    lines.extend(["", "## Feature Enrichment in a050-Fixed Rows", "", "| Feature | Fixed rows | All rows | Delta |", "|---|---:|---:|---:|"])
    for item in summary["diagnostics"]["a050_fixed_feature_enrichment"][:20]:
        lines.append(
            f"| {item['feature']} | {item['group_rate']:.3f} | {item['all_rate']:.3f} | {item['delta']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Dataset Construction Hints",
            "",
            "- Contaminated diagnostic subset: rows in `a050_fixed_base` are the highest-value CEDR rows for understanding the target boundary; rows in `bad_hurts_base` describe what the no-GeRaCl direction breaks.",
            "- Fair analog construction: reproduce enriched features without CEDR text. If neutral rows with emotion lexemes are enriched, mine reported/quoted/news neutral text with emotion words. If slang/emoticons are enriched, preserve social-media rows instead of filtering them away.",
            "- Objective hint: train pairwise preferences where a clean proxy sample should move toward the a050-neighbor label pattern and away from the bad/no-GeRaCl neighbor pattern, rather than only matching class labels.",
            "- Evaluation hint: keep tracking the same flip sets after each fair ablation; CEDR may improve only when the a050-fixed subset improves without increasing a050-hurts-like errors.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Track CEDR row-level signals across checkpoints.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/official_repro/cedr_diagnostic_tracking")
    parser.add_argument("--models", nargs="+", default=["base_mixh_habrfull", "retained_best", "bad_no_geracl", "a050_extrapolated", "contam_knn_episode"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--encode-task-name",
        default="CEDRClassification",
        help="Task name passed to wrapper.encode. Default uses the CEDR frozen-wrapper prefix.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task, train_split, test_split, train_samples, unique_train_indices, task_prompts = setup_mteb(args.seed)
    train_texts = train_split.select(unique_train_indices)["text"]
    test_texts = test_split["text"]

    binarizer = MultiLabelBinarizer()
    binarizer.fit(test_split["label"])
    classes = np.asarray(binarizer.classes_)
    all_features = [text_features(test_split[index]["text"], test_split[index]["label"]) for index in range(len(test_split))]

    model_metrics: dict[str, Any] = {}
    for name in args.models:
        if name == "released":
            checkpoint = None
        else:
            checkpoint = DEFAULT_MODELS.get(name)
            if checkpoint is None:
                raise SystemExit(f"Unknown model label {name!r}. Known: {', '.join(sorted(DEFAULT_MODELS))}, released")
            if not checkpoint.exists():
                raise SystemExit(f"Checkpoint for {name!r} does not exist: {checkpoint}")
        print(f"encoding/evaluating {name}: {checkpoint if checkpoint else 'released'}", flush=True)
        train_embeddings, test_embeddings = encode_checkpoint(
            label=name,
            latent_checkpoint=checkpoint,
            train_texts=train_texts,
            test_texts=test_texts,
            task_prompts=task_prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            attn_implementation=args.attn_implementation,
            torch_dtype=args.torch_dtype,
            local_files_only=args.local_files_only,
            cache_dir=args.output_dir / "embedding_cache",
            reuse_cache=args.reuse_cache,
            encode_task_name=args.encode_task_name,
        )
        model_metrics[name] = evaluate_embeddings(
            task=task,
            train_split=train_split,
            test_split=test_split,
            train_samples=train_samples,
            unique_train_indices=unique_train_indices,
            train_embeddings_raw=train_embeddings,
            test_embeddings=test_embeddings,
            binarizer=binarizer,
            classes=classes,
        )

    def rate(name: str, index: int) -> float:
        if name not in model_metrics:
            return 0.0
        return float(model_metrics[name]["per_test"][index]["correct_rate"])

    flip_sets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_row_records = []
    for index in range(len(test_split)):
        model_rows = {
            name: {
                "correct_rate": metrics["per_test"][index]["correct_rate"],
                "main_prediction": metrics["per_test"][index]["main_prediction"],
                "main_neighbor_signature": metrics["per_test"][index]["main_neighbor_signature"],
            }
            for name, metrics in model_metrics.items()
        }
        record = {
            "test_index": index,
            "text": test_split[index]["text"],
            "true_label": label_name(test_split[index]["label"]),
            "features": all_features[index],
            "models": model_rows,
        }
        per_row_records.append(record)

        base = rate("base_mixh_habrfull", index)
        a050 = rate("a050_extrapolated", index)
        bad = rate("bad_no_geracl", index)
        retained = rate("retained_best", index)
        contam = rate("contam_knn_episode", index)
        if a050 - base >= 0.3:
            flip_sets["a050_fixed_base"].append(record)
        if base - a050 >= 0.3:
            flip_sets["a050_hurts_base"].append(record)
        if base - bad >= 0.3:
            flip_sets["bad_hurts_base"].append(record)
        if retained - base >= 0.3:
            flip_sets["retained_fixed_base"].append(record)
        if "contam_knn_episode" in model_metrics and contam - base >= 0.3:
            flip_sets["contam_fixed_base"].append(record)
        if a050 >= 0.7 and max(base, bad, retained) <= 0.3:
            flip_sets["a050_only_strong"].append(record)

    a050_fixed = flip_sets.get("a050_fixed_base", [])
    a050_fixed_true = Counter(row["true_label"] for row in a050_fixed)
    a050_fixed_confusions: Counter[str] = Counter()
    for row in a050_fixed:
        base_pred = row["models"]["base_mixh_habrfull"]["main_prediction"]
        a050_fixed_confusions[f"{row['true_label']} -> {base_pred}"] += 1

    summary = {
        "task": "CEDRClassification",
        "seed": args.seed,
        "models": {
            name: {
                "mean_accuracy": metrics["mean_accuracy"],
                "mean_f1": metrics["mean_f1"],
                "wrong_by_true": metrics["wrong_by_true"],
                "top_confusions": metrics["top_confusions"],
                "predicted_label_counts": metrics["predicted_label_counts"],
            }
            for name, metrics in model_metrics.items()
        },
        "flip_sets": {key: rows for key, rows in flip_sets.items()},
        "diagnostics": {
            "a050_fixed_true_labels": a050_fixed_true.most_common(),
            "a050_fixed_base_confusions": a050_fixed_confusions.most_common(30),
            "a050_fixed_feature_enrichment": feature_enrichment(
                [row["features"] for row in a050_fixed],
                all_features,
            ),
        },
    }

    (args.output_dir / "cedr_diagnostic_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "cedr_row_tracking.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_row_records:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    for key, rows in flip_sets.items():
        with (args.output_dir / f"{key}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    write_markdown(summary, args.output_dir / "CEDR_DIAGNOSTIC_TRACKING.md")

    print(f"wrote {args.output_dir / 'CEDR_DIAGNOSTIC_TRACKING.md'}")
    for name, metrics in summary["models"].items():
        print(f"{name}: accuracy={metrics['mean_accuracy']:.6f} f1={metrics['mean_f1']:.6f}")
    for key, rows in summary["flip_sets"].items():
        print(f"{key}: {len(rows)}")


if __name__ == "__main__":
    main()
