from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RUMTEB_TASK_DATASETS: dict[str, str] = {
    "GeoreviewClassification": "mteb/GeoreviewClassification",
    "HeadlineClassification": "ai-forever/headline-classification",
    "InappropriatenessClassification": "ai-forever/inappropriateness-classification",
    "KinopoiskClassification": "ai-forever/kinopoisk-sentiment-classification",
    "MassiveIntentClassification": "mteb/amazon_massive_intent",
    "MassiveScenarioClassification": "mteb/amazon_massive_scenario",
    "RuReviewsClassification": "ai-forever/ru-reviews-classification",
    "RuSciBenchGRNTIClassification": "ai-forever/ru-scibench-grnti-classification",
    "RuSciBenchOECDClassification": "ai-forever/ru-scibench-oecd-classification",
    "GeoreviewClusteringP2P": "ai-forever/georeview-clustering-p2p",
    "RuSciBenchGRNTIClusteringP2P": "ai-forever/ru-scibench-grnti-classification",
    "RuSciBenchOECDClusteringP2P": "ai-forever/ru-scibench-oecd-classification",
    "CEDRClassification": "mteb/CEDRClassification",
    "SensitiveTopicsClassification": "ai-forever/sensitive-topics-classification",
    "TERRa": "mteb/TERRa",
    "MIRACLReranking": "mteb/MIRACLReranking",
    "RuBQReranking": "mteb/RuBQReranking",
    "MIRACLRetrievalHardNegatives.v2": "mteb/MIRACLRetrievalHardNegatives",
    "RiaNewsRetrievalHardNegatives.v2": "mteb/RiaNewsRetrieval_test_top_250_only_w_correct-v2",
    "RuBQRetrieval": "ai-forever/rubq-retrieval",
    "RUParaPhraserSTS": "merionum/ru_paraphraser",
    "STS22": "mteb/sts22-crosslingual-sts",
    "RuSTSBenchmarkSTS": "ai-forever/ru-stsbenchmark-sts",
}


def load_training_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    validate_training_manifest(manifest, path)
    return manifest


def validate_training_manifest(manifest: dict[str, Any], path: Path | None = None) -> None:
    label = str(path) if path is not None else "training manifest"
    if not isinstance(manifest.get("name"), str) or not manifest["name"]:
        raise ValueError(f"{label}: missing non-empty `name`")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"{label}: `datasets` must be a list")
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"{label}: dataset #{index} must be an object")
        for key in ("id", "usage", "coverage_tags", "contaminates_rumteb_tasks"):
            if key not in dataset:
                raise ValueError(f"{label}: dataset #{index} missing `{key}`")
        if not isinstance(dataset["id"], str) or not dataset["id"]:
            raise ValueError(f"{label}: dataset #{index} has invalid `id`")
        for key in ("usage", "coverage_tags", "contaminates_rumteb_tasks"):
            if not isinstance(dataset[key], list):
                raise ValueError(f"{label}: dataset #{index} `{key}` must be a list")


def contaminated_tasks_from_manifest(manifest: dict[str, Any] | None) -> dict[str, list[str]]:
    if manifest is None:
        return {}

    contaminated: dict[str, list[str]] = defaultdict(list)
    dataset_to_tasks: dict[str, set[str]] = defaultdict(set)
    for task_name, dataset_id in RUMTEB_TASK_DATASETS.items():
        dataset_to_tasks[dataset_id].add(task_name)

    for dataset in manifest.get("datasets", []):
        dataset_id = dataset["id"]
        usage = set(dataset.get("usage", []))
        if "diagnostic_only" in usage and len(usage) == 1:
            continue

        for task_name in sorted(dataset_to_tasks.get(dataset_id, ())):
            contaminated[task_name].append(dataset_id)

        for task_name in dataset.get("contaminates_rumteb_tasks", []):
            contaminated[task_name].append(dataset_id)

    return {task: sorted(set(reasons)) for task, reasons in sorted(contaminated.items())}


def filter_tasks_by_scope(
    task_names: Iterable[str],
    contaminated_tasks: dict[str, list[str]],
    eval_scope: str,
) -> list[str]:
    task_names = list(task_names)
    if eval_scope == "all":
        return task_names
    if eval_scope == "clean":
        return [name for name in task_names if name not in contaminated_tasks]
    if eval_scope == "contaminated":
        return [name for name in task_names if name in contaminated_tasks]
    raise ValueError(f"Unsupported eval scope: {eval_scope}")


def task_category(task_name: str) -> str:
    if "Clustering" in task_name:
        return "Clustering"
    if "Retrieval" in task_name:
        return "Retrieval"
    if "Reranking" in task_name:
        return "Reranking"
    if "STS" in task_name or task_name in {"TERRa", "RUParaPhraserSTS"}:
        return "STS/NLI"
    return "Classification"


def extract_main_score(result: dict[str, Any]) -> float | None:
    values: list[float] = []
    for entries in result.get("scores", {}).values():
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and "main_score" in entry:
                values.append(float(entry["main_score"]))
    if not values:
        return None
    return sum(values) / len(values)
