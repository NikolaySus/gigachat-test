from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
CONFIG_DIR = ROOT / "configs" / "experiments"
CACHE_DIR = ROOT / "results" / "mteb_cache"

CEDR_PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
GROUPS = ["neutral", "joy", "sadness", "anger", "surprise", "fear"]
EMOTION_GROUPS = ["joy", "sadness", "anger", "surprise", "fear"]
CEDR_TRAIN_UNITS = {
    "neutral": 3043,
    "joy": 1569,
    "sadness": 1417,
    "surprise": 607,
    "fear": 589,
    "anger": 411,
}


def normalize_text(value: Any) -> str:
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
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
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count]


def load_flagged(name: str) -> set[tuple[str, int]]:
    path = ROOT / "results" / "contamination" / "cedr_candidates" / f"{name}_flagged_rows.json"
    if not path.exists():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(str(row["split"]), int(row["index"])) for row in rows}


def target_counts(count: int, groups: list[str]) -> dict[str, int]:
    total = sum(CEDR_TRAIN_UNITS[group] for group in groups)
    targets = {group: math.floor(count * CEDR_TRAIN_UNITS[group] / total) for group in groups}
    remaining = count - sum(targets.values())
    order = sorted(groups, key=lambda group: (count * CEDR_TRAIN_UNITS[group] / total) - targets[group], reverse=True)
    for group in order[:remaining]:
        targets[group] += 1
    return targets


def cap_and_redistribute(targets: dict[str, int], pools: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    targets = dict(targets)
    total = sum(targets.values())
    for group in list(targets):
        targets[group] = min(targets[group], len(pools[group]))
    while sum(targets.values()) < total:
        progressed = False
        for group in sorted(targets, key=lambda name: len(pools[name]) - targets[name], reverse=True):
            if targets[group] < len(pools[group]):
                targets[group] += 1
                progressed = True
                if sum(targets.values()) == total:
                    break
        if not progressed:
            break
    return targets


def build_records_from_pools(
    pools: dict[str, list[dict[str, Any]]],
    *,
    count: int,
    seed: int,
    source: str,
    groups: list[str] = GROUPS,
    neutral_as_positive_class: bool = True,
    extra_negative_pools: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    targets = cap_and_redistribute(target_counts(count, groups), pools)
    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    globally_selected: set[tuple[str, int, str]] = set()
    for group, target in sorted(targets.items(), key=lambda item: len(pools[item[0]])):
        pool = pools[group][:]
        rng.shuffle(pool)
        selected = []
        for item in pool:
            key = (str(item["split"]), int(item["index"]), item["text"])
            if key not in globally_selected:
                selected.append(item)
                globally_selected.add(key)
            if len(selected) == target:
                break
        if len(selected) < target:
            raise ValueError(f"Not enough rows for {source}:{group}: need {target}, got {len(pool)}")
        selected_by_group[group] = selected

    records = []
    all_negative_pools = {group: selected_by_group[group] for group in groups}
    if extra_negative_pools:
        all_negative_pools.update(extra_negative_pools)

    for group, items in selected_by_group.items():
        if group == "neutral" and not neutral_as_positive_class:
            continue
        for item in items:
            positive_pool = [candidate for candidate in items if candidate["text"] != item["text"]]
            if not positive_pool:
                continue
            negatives = []
            for negative_group, negative_pool in all_negative_pools.items():
                if negative_group == group or not negative_pool:
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(negative_pool)["text"])
            records.append(
                {
                    "source": source,
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positive_pool)["text"],
                    "negatives": negatives[:8],
                    "metadata": {"group": group, "labels": item.get("labels", []), "split": item["split"], "index": item["index"]},
                    "objective": "contrastive",
                }
            )

    rng.shuffle(records)
    return records, {
        "source": source,
        "requested": count,
        "kept": len(records),
        "groups": groups,
        "target_by_group": targets,
        "selected_by_group": dict(Counter(record["metadata"]["group"] for record in records)),
        "neutral_as_positive_class": neutral_as_positive_class,
    }


def load_go_ekman_pools() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    flagged = load_flagged("go_ekman")
    dataset = load_dataset("SkyWater21/ru_go_emotions_ekman", "simplified_ekman", cache_dir=str(CACHE_DIR))
    names = dataset["train"].features["labels_ekman"].feature.names
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()
    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            if (split, index) in flagged:
                skipped["flagged_overlap"] += 1
                continue
            text = normalize_text(row["ru_text"])
            if len(text) < 8 or len(text) > 420:
                skipped["length"] += 1
                continue
            if text in seen:
                skipped["duplicate"] += 1
                continue
            labels = [names[label] for label in row["labels_ekman"]]
            labels = [label for label in labels if label != "disgust"]
            if not labels:
                skipped["disgust_only"] += 1
                continue
            seen.add(text)
            groups = labels if labels != ["neutral"] else ["neutral"]
            for group in groups:
                if group in GROUPS:
                    pools[group].append({"split": split, "index": index, "text": row["ru_text"], "labels": labels})
    return pools, {"skipped": dict(skipped), "available_by_group": {key: len(value) for key, value in pools.items()}}


def load_ru_sentiment_pools() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    flagged = load_flagged("ru_sentiment_social")
    dataset = load_dataset("DmitrySharonov/ru_sentiment_neg_pos_neutral", cache_dir=str(CACHE_DIR))
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    map_label = {"positive": "joy", "negative": "anger", "neutral": "neutral"}
    skipped = Counter()
    seen = set()
    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            if (split, index) in flagged:
                skipped["flagged_overlap"] += 1
                continue
            text = normalize_text(row["text"])
            if len(text) < 12 or len(text) > 240:
                skipped["length"] += 1
                continue
            if text in seen:
                skipped["duplicate"] += 1
                continue
            group = map_label.get(row["label"])
            if group is None:
                skipped["unknown_label"] += 1
                continue
            seen.add(text)
            pools[group].append({"split": split, "index": index, "text": row["text"], "labels": [row["label"]]})
    return pools, {"skipped": dict(skipped), "available_by_group": {key: len(value) for key, value in pools.items()}}


def make_mix(
    *,
    name: str,
    cedr_records: list[dict[str, Any]],
    cedr_summary: dict[str, Any],
    seed: int,
) -> None:
    geracl = read_jsonl(DATA_DIR / "open_ru_1r_nc_geracl.jsonl")
    habr = read_jsonl(DATA_DIR / "open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl")
    deepvk = read_jsonl(DATA_DIR / "open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl")
    grandmaster = read_jsonl(DATA_DIR / "open_ru_1r_nc_grandmaster_clustered_3200.jsonl")
    selected = {
        "geracl": sample(geracl, count=6400, seed=seed),
        "habr_harder_full": habr,
        "deepvk_filtered": sample(deepvk, count=3200, seed=seed + 2),
        "grandmaster": grandmaster,
        "cedr_candidate": cedr_records,
    }
    mixed = []
    for records in selected.values():
        mixed.extend(records)
    random.Random(seed + 13).shuffle(mixed)

    data_path = DATA_DIR / f"open_ru_1r_nc_{name}.jsonl"
    write_jsonl(data_path, mixed)
    write_json(
        data_path.with_name(data_path.stem + "_summary.json"),
        {
            "output": str(data_path.relative_to(ROOT)),
            "seed": seed,
            "base_run": "Mix H HabrFull stage 1",
            "change": f"Replace/define CEDR candidate component: {name}",
            "component_counts": {key: len(value) for key, value in selected.items()},
            "total_records": len(mixed),
            "max_steps_1x_batch3": math.ceil(len(mixed) / 3),
            "cedr_component": cedr_summary,
        },
    )

    config = {
        "name": f"exp01r_nc_{name}_4096_eager_frozenrepro",
        "description": f"CEDR ablation, stage 1 only: {name}",
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": None,
        "freeze_llm": True,
        "reinit_latent": True,
        "data_path": str(data_path.relative_to(ROOT)),
        "output_dir": f"experiments/exp01_reinit_fair/checkpoints/{name}_4096_eager_frozenrepro",
        "max_length": 4096,
        "batch_size": 3,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "temperature": 0.02,
        "max_steps": math.ceil(len(mixed) / 3),
        "log_every": 50,
        "save_every": math.ceil(len(mixed) / 3),
        "seed": seed,
    }
    write_json(CONFIG_DIR / f"exp01r_nc_{name}_4096_eager_frozenrepro.json", config)


def main() -> None:
    seed = 77
    go_pools, go_meta = load_go_ekman_pools()
    sentiment_pools, sentiment_meta = load_ru_sentiment_pools()

    go_prior, go_prior_summary = build_records_from_pools(
        go_pools,
        count=3200,
        seed=seed + 20,
        source="SkyWater21/ru_go_emotions_ekman:cedr_prior_no_disgust",
        groups=GROUPS,
        neutral_as_positive_class=True,
    )
    go_prior_summary["loader"] = go_meta
    make_mix(name="cedr_goekman_prior", cedr_records=go_prior, cedr_summary=go_prior_summary, seed=seed)

    go_emotion_only, go_emotion_only_summary = build_records_from_pools(
        go_pools,
        count=3200,
        seed=seed + 30,
        source="SkyWater21/ru_go_emotions_ekman:emotion_only_no_disgust",
        groups=EMOTION_GROUPS,
        neutral_as_positive_class=True,
        extra_negative_pools={"neutral": sample(go_pools["neutral"], count=1600, seed=seed + 31)},
    )
    go_emotion_only_summary["loader"] = go_meta
    make_mix(name="cedr_goekman_emotion_only", cedr_records=go_emotion_only, cedr_summary=go_emotion_only_summary, seed=seed)

    sentiment_prior, sentiment_prior_summary = build_records_from_pools(
        sentiment_pools,
        count=3200,
        seed=seed + 40,
        source="DmitrySharonov/ru_sentiment_neg_pos_neutral:cedr_polarity_prior",
        groups=["neutral", "joy", "anger"],
        neutral_as_positive_class=True,
    )
    sentiment_prior_summary["loader"] = sentiment_meta
    make_mix(name="cedr_rusentiment_polarity", cedr_records=sentiment_prior, cedr_summary=sentiment_prior_summary, seed=seed)

    go_half, go_half_summary = build_records_from_pools(
        go_pools,
        count=2200,
        seed=seed + 50,
        source="SkyWater21/ru_go_emotions_ekman:cedr_prior_no_disgust",
        groups=GROUPS,
        neutral_as_positive_class=True,
    )
    sentiment_neutral = sample(sentiment_pools["neutral"], count=1000, seed=seed + 51)
    go_emotion_records = [record for record in go_half if record["metadata"]["group"] != "neutral"]
    rng = random.Random(seed + 52)
    for item in sentiment_neutral:
        positives = [candidate for candidate in sentiment_neutral if candidate["text"] != item["text"]]
        if not positives:
            continue
        negatives = []
        for group in EMOTION_GROUPS:
            negatives.append(CEDR_PREFIX + rng.choice(go_pools[group])["text"])
        go_emotion_records.append(
            {
                "source": "DmitrySharonov/ru_sentiment_neg_pos_neutral:neutral_rejection_with_go_emotion_negatives",
                "query": CEDR_PREFIX + item["text"],
                "positive": CEDR_PREFIX + rng.choice(positives)["text"],
                "negatives": negatives,
                "metadata": {"group": "neutral", "labels": item["labels"], "split": item["split"], "index": item["index"]},
                "objective": "contrastive",
            }
        )
    rng.shuffle(go_emotion_records)
    make_mix(
        name="cedr_goekman_rusent_neutral",
        cedr_records=go_emotion_records,
        cedr_summary={
            "source": "go_ekman_2200_plus_rusentiment_neutral_1000",
            "go_component": go_half_summary,
            "sentiment_loader": sentiment_meta,
            "kept": len(go_emotion_records),
            "selected_by_group": dict(Counter(record["metadata"]["group"] for record in go_emotion_records)),
        },
        seed=seed,
    )


if __name__ == "__main__":
    main()
