from __future__ import annotations

import argparse
import json
import os
import random
import sys
import itertools
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


def label_name(labels: list[int] | tuple[int, ...]) -> str:
    if not labels:
        return NEUTRAL
    return "+".join(LABEL_NAMES.get(int(label), str(label)) for label in labels)


def prediction_name(row: np.ndarray, classes: np.ndarray) -> str:
    labels = [int(classes[index]) for index, value in enumerate(row) if value]
    return label_name(labels)


def top_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


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


def neighbor_label_summary(
    classifier: Any,
    sample_indices: list[int],
    train_split: Any,
    test_embeddings: np.ndarray,
    test_index: int,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    distances, positions = classifier.kneighbors(test_embeddings[test_index : test_index + 1], n_neighbors=limit)
    rows = []
    for distance, position in zip(distances[0], positions[0], strict=True):
        train_index = int(sample_indices[int(position)])
        rows.append(
            {
                "train_index": train_index,
                "distance": float(distance),
                "label": label_name(train_split[train_index]["label"]),
                "text": train_split[train_index]["text"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CEDRClassification 5-NN failures.")
    parser.add_argument("--latent-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hard-examples", type=int, default=50)
    parser.add_argument("--neighbor-examples", type=int, default=20)
    parser.add_argument(
        "--encode-task-name",
        default=None,
        help=(
            "Optional task_name passed directly to wrapper.encode. Leave unset to mirror "
            "current MTEB/frozen-wrapper behavior, where task_metadata is passed and the "
            "wrapper falls back to the generic legacy prefix."
        ),
    )
    args = parser.parse_args()

    cache_dir = ROOT / "results" / "official_repro_cache"
    os.environ.setdefault("MTEB_CACHE", str(cache_dir.resolve()))
    os.environ.setdefault("HF_HOME", str((cache_dir / "hf_home").resolve()))
    os.environ.setdefault("HF_DATASETS_CACHE", str((cache_dir / "datasets").resolve()))
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    hf_modules = str((cache_dir / "hf_home" / "modules").resolve())
    if hf_modules not in sys.path:
        sys.path.insert(0, hf_modules)

    import mteb

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    benchmark = mteb.get_benchmark("MTEB(rus, v1)")
    task_by_name = {task.metadata.name: task for task in benchmark.tasks}
    task = task_by_name["CEDRClassification"]
    task.seed = DEFAULT_TASK_SEEDS.get(task.metadata.name, args.seed)
    task.load_data()

    task_prompts = {
        item.metadata.name: (str(item.metadata.type), getattr(item.metadata, "prompt", None))
        for item in benchmark.tasks
    }
    for task_name, prompt in DEFAULT_MTEB_PROMPT_OVERRIDES.items():
        task_type, _ = task_prompts.get(task_name, ("", None))
        task_prompts[task_name] = (task_type, prompt)

    model = GigaOfficialMTEBWrapper(
        batch_size=args.batch_size,
        max_length=args.max_length,
        model_revision=MODEL_REVISION,
        attn_implementation=args.attn_implementation,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
        latent_checkpoint=args.latent_checkpoint,
        task_prompts=task_prompts,
        prompt_mode="legacy_ru",
        symmetric_instruction="mteb",
        legacy_prefix_ensembles=dict(DEFAULT_LEGACY_PREFIX_ENSEMBLES),
        task_prompt_modes=dict(DEFAULT_TASK_PROMPT_MODES),
        task_text_normalizations=DEFAULT_TASK_TEXT_NORMALIZATIONS,
        task_batch_sizes=DEFAULT_TASK_BATCH_SIZES,
    )

    # Match official_repro/run_official_rumteb.py: model construction can touch
    # RNG state, so reset immediately before task evaluation/sampling.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # MTEB.run constructs an evaluator after resetting seeds and before task
    # sampling. MTEB initialization consumes RNG in this environment, so doing
    # the same here is required to reproduce official scores_per_experiment.
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

    # Mirror mteb.abstasks.AbsTaskMultilabelClassification exactly. The set
    # iteration order affects encode batching; with bf16 this can move a few
    # borderline 5-NN decisions.
    unique_train_indices = list(set(itertools.chain.from_iterable(train_samples)))
    train_texts = train_split.select(unique_train_indices)["text"]
    test_texts = test_split["text"]

    encode_kwargs: dict[str, Any] = {}
    if args.encode_task_name:
        encode_kwargs["task_name"] = args.encode_task_name
    train_embeddings_raw = model.encode(train_texts, **encode_kwargs)
    test_embeddings = model.encode(test_texts, **encode_kwargs)
    train_embeddings = {
        index: train_embeddings_raw[position]
        for position, index in enumerate(unique_train_indices)
    }

    binarizer = MultiLabelBinarizer()
    y_test = binarizer.fit_transform(test_split["label"])
    classes = np.asarray(binarizer.classes_)

    summary: dict[str, Any] = {
        "task": task.metadata.name,
        "latent_checkpoint": str(args.latent_checkpoint) if args.latent_checkpoint is not None else None,
        "seed": args.seed,
        "samples_per_label": task.samples_per_label,
        "n_experiments": task.n_experiments,
        "classifier": type(task.evaluator_model).__name__,
        "classifier_params": task.evaluator_model.get_params(),
        "encode_task_name": args.encode_task_name,
        "train_size": len(train_split),
        "test_size": len(test_split),
        "unique_train_encoded": len(unique_train_indices),
        "test_label_counts": Counter(label_name(labels) for labels in test_split["label"]),
        "experiments": [],
    }

    wrong_counter: Counter[int] = Counter()
    wrong_by_true: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    predicted_counter: Counter[str] = Counter()
    true_counter: Counter[str] = Counter(label_name(labels) for labels in test_split["label"])
    per_test_predictions: list[dict[str, Any]] = [
        {
            "test_index": index,
            "label": label_name(test_split[index]["label"]),
            "text": test_split[index]["text"],
            "predictions": [],
        }
        for index in range(len(test_split))
    ]

    for experiment_index, sample_indices in enumerate(train_samples, start=1):
        classifier = clone(task.evaluator_model)
        x_train = np.stack([train_embeddings[index] for index in sample_indices])
        y_train = binarizer.transform(train_split.select(sample_indices)["label"])
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(test_embeddings)

        exact = (y_pred == y_test).all(axis=1)
        true_names = [label_name(labels) for labels in test_split["label"]]
        pred_names = [prediction_name(row, classes) for row in y_pred]
        predicted_counter.update(pred_names)
        for index, ok in enumerate(exact):
            per_test_predictions[index]["predictions"].append(
                {
                    "experiment": experiment_index,
                    "prediction": pred_names[index],
                    "correct": bool(ok),
                }
            )
            if ok:
                continue
            wrong_counter[index] += 1
            wrong_by_true[true_names[index]] += 1
            confusion[f"{true_names[index]} -> {pred_names[index]}"] += 1

        per_true = {}
        for true_name in sorted(set(true_names)):
            indices = [i for i, name in enumerate(true_names) if name == true_name]
            per_true[true_name] = {
                "count": len(indices),
                "exact_accuracy": float(exact[indices].mean()) if indices else None,
            }

        summary["experiments"].append(
            {
                "experiment": experiment_index,
                "train_indices": sample_indices,
                "accuracy": float(classifier.score(test_embeddings, y_test)),
                "f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "per_true": per_true,
                "wrong_by_true": dict(wrong_by_true),
                "top_confusions": top_counter(confusion, 20),
            }
        )

    hard_examples = []
    for index, count in wrong_counter.most_common(args.hard_examples):
        hard_examples.append(
            {
                "test_index": index,
                "wrong_count": count,
                "label": label_name(test_split[index]["label"]),
                "text": test_split[index]["text"],
            }
        )

    neighbor_examples = []
    for index, _ in wrong_counter.most_common(args.neighbor_examples):
        per_experiment_neighbors = []
        for experiment_index, sample_indices in enumerate(train_samples, start=1):
            classifier = clone(task.evaluator_model)
            x_train = np.stack([train_embeddings[item] for item in sample_indices])
            y_train = binarizer.transform(train_split.select(sample_indices)["label"])
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(test_embeddings[index : index + 1])
            pred_name = prediction_name(y_pred[0], classes)
            per_experiment_neighbors.append(
                {
                    "experiment": experiment_index,
                    "prediction": pred_name,
                    "neighbors": neighbor_label_summary(
                        classifier,
                        sample_indices,
                        train_split,
                        test_embeddings,
                        index,
                        limit=int(task.evaluator_model.get_params().get("n_neighbors", 5)),
                    ),
                }
            )
        neighbor_examples.append(
            {
                "test_index": index,
                "wrong_count": int(wrong_counter[index]),
                "label": label_name(test_split[index]["label"]),
                "text": test_split[index]["text"],
                "experiments": per_experiment_neighbors,
            }
        )

    summary["aggregate"] = {
        "mean_accuracy": float(np.mean([item["accuracy"] for item in summary["experiments"]])),
        "mean_f1": float(np.mean([item["f1"] for item in summary["experiments"]])),
        "true_label_counts": dict(true_counter),
        "predicted_label_counts": dict(predicted_counter),
        "wrong_by_true": dict(wrong_by_true),
        "top_confusions": top_counter(confusion, 50),
        "hard_examples": hard_examples,
        "neighbor_examples": neighbor_examples,
        "per_test_predictions": per_test_predictions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {args.output}")
    print("wrong_by_true", dict(wrong_by_true))
    print("top_confusions")
    for item in top_counter(confusion, 15):
        print(f"  {item['key']}: {item['count']}")


if __name__ == "__main__":
    main()
