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
EMOTIONS = ["joy", "sadness", "anger", "surprise", "fear"]
CEDR_TRAIN_UNITS = {
    "joy": 1569,
    "sadness": 1417,
    "surprise": 607,
    "fear": 589,
    "anger": 411,
}
RUEMOTIONS_MAP = {
    "радость": "joy",
    "восторг": "joy",
    "восхищение": "joy",
    "веселье": "joy",
    "смех": "joy",
    "экстаз": "joy",
    "благодарность": "joy",
    "любовь": "joy",
    "нежность": "joy",
    "грусть": "sadness",
    "печаль": "sadness",
    "тоска": "sadness",
    "меланхолия": "sadness",
    "одиночество": "sadness",
    "страдание": "sadness",
    "отчаяние": "sadness",
    "горечь": "sadness",
    "злость": "anger",
    "гнев": "anger",
    "ярость": "anger",
    "ненависть": "anger",
    "обида": "anger",
    "раздражение": "anger",
    "возмущение": "anger",
    "фрустрация": "anger",
    "удивление": "surprise",
    "изумление": "surprise",
    "шок": "surprise",
    "растерянность": "surprise",
    "недоумение": "surprise",
    "замешательство": "surprise",
    "страх": "fear",
    "тревога": "fear",
    "ужас": "fear",
    "паника": "fear",
    "паранойя": "fear",
    "неуверенность": "fear",
}


def normalize_text(value: Any) -> str:
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(value: str) -> set[str]:
    return set(re.findall(r"[\w]+", normalize_text(value), flags=re.U))


def jaccard(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def target_counts(count: int) -> dict[str, int]:
    total = sum(CEDR_TRAIN_UNITS.values())
    targets = {group: math.floor(count * CEDR_TRAIN_UNITS[group] / total) for group in EMOTIONS}
    remaining = count - sum(targets.values())
    order = sorted(EMOTIONS, key=lambda group: (count * CEDR_TRAIN_UNITS[group] / total) - targets[group], reverse=True)
    for group in order[:remaining]:
        targets[group] += 1
    return targets


def load_twitter_pools() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    flagged = load_flagged("twitter_emotions_ekman")
    dataset = load_dataset("AiLab-IMCS-UL/twitter_emotions-ru", "simplified_ekman", cache_dir=str(CACHE_DIR))
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()
    for split, ds in dataset.items():
        label_feature = ds.features["labels_ekman"]
        for index, row in enumerate(ds):
            if (split, index) in flagged:
                skipped["flagged_overlap"] += 1
                continue
            label = label_feature.int2str(row["labels_ekman"])
            if label not in EMOTIONS:
                skipped["unsupported_label"] += 1
                continue
            text = normalize_text(row["ru_text"])
            if len(text) < 16 or len(text) > 360:
                skipped["length"] += 1
                continue
            if text in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(text)
            pools[label].append({"split": split, "index": index, "text": row["ru_text"], "label": label})
    return pools, {"skipped": dict(skipped), "available_by_group": {key: len(value) for key, value in pools.items()}}


def load_ruemotions_pools() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    flagged = load_flagged("ruemotions")
    dataset = load_dataset("Darkester/RuEmotions", cache_dir=str(CACHE_DIR))
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen = set()
    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            if (split, index) in flagged:
                skipped["flagged_overlap"] += 1
                continue
            label = RUEMOTIONS_MAP.get(str(row["emotion"]).strip().lower())
            if label is None:
                skipped["unmapped_label"] += 1
                continue
            text = normalize_text(row["text"])
            if len(text) < 12 or len(text) > 360:
                skipped["length"] += 1
                continue
            if text in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(text)
            pools[label].append({"split": split, "index": index, "text": row["text"], "label": label, "raw_label": row["emotion"]})
    return pools, {"skipped": dict(skipped), "available_by_group": {key: len(value) for key, value in pools.items()}}


def select_balanced(pools: dict[str, list[dict[str, Any]]], *, count: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    selected = {}
    for group, target in target_counts(count).items():
        pool = pools[group][:]
        rng.shuffle(pool)
        if len(pool) < target:
            raise ValueError(f"Not enough records for {group}: need {target}, got {len(pool)}")
        selected[group] = pool[:target]
    return selected


def make_same_label_records(selected: dict[str, list[dict[str, Any]]], pools: dict[str, list[dict[str, Any]]], *, seed: int, source: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for group, items in selected.items():
        for item in items:
            positive_pool = [candidate for candidate in items if candidate["text"] != item["text"]]
            negatives = []
            for negative_group in EMOTIONS:
                if negative_group == group:
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(pools[negative_group])["text"])
            records.append(
                {
                    "source": source,
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + rng.choice(positive_pool)["text"],
                    "negatives": negatives,
                    "metadata": {"group": group, "split": item["split"], "index": item["index"]},
                    "objective": "contrastive",
                }
            )
    rng.shuffle(records)
    return records


def make_topic_preserving_records(selected: dict[str, list[dict[str, Any]]], pools: dict[str, list[dict[str, Any]]], *, seed: int, source: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for group, items in selected.items():
        for item in items:
            candidates = [candidate for candidate in items if candidate["text"] != item["text"]]
            nearby = sorted(rng.sample(candidates, k=min(64, len(candidates))), key=lambda cand: jaccard(item["text"], cand["text"]), reverse=True)
            positive = nearby[0]
            far_same = [candidate for candidate in rng.sample(candidates, k=min(128, len(candidates))) if jaccard(item["text"], candidate["text"]) <= 0.02]
            negatives = [CEDR_PREFIX + candidate["text"] for candidate in far_same[:2]]
            for negative_group in EMOTIONS:
                if negative_group == group:
                    continue
                negatives.append(CEDR_PREFIX + rng.choice(pools[negative_group])["text"])
            records.append(
                {
                    "source": source,
                    "query": CEDR_PREFIX + item["text"],
                    "positive": CEDR_PREFIX + positive["text"],
                    "negatives": negatives[:8],
                    "metadata": {"group": group, "split": item["split"], "index": item["index"], "positive_jaccard": jaccard(item["text"], positive["text"])},
                    "objective": "contrastive",
                }
            )
    rng.shuffle(records)
    return records


def merge_pools(*pool_sets: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pools in pool_sets:
        for group, rows in pools.items():
            merged[group].extend(rows)
    return merged


def make_mix(*, name: str, cedr_records: list[dict[str, Any]], cedr_summary: dict[str, Any], seed: int) -> None:
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
    max_steps = math.ceil(len(mixed) / 3)
    write_json(
        data_path.with_name(data_path.stem + "_summary.json"),
        {
            "output": str(data_path.relative_to(ROOT)),
            "seed": seed,
            "base_run": "Mix H HabrFull stage 1",
            "component_counts": {key: len(value) for key, value in selected.items()},
            "total_records": len(mixed),
            "max_steps_1x_batch3": max_steps,
            "cedr_component": cedr_summary,
        },
    )

    write_json(
        CONFIG_DIR / f"exp01r_nc_{name}_4096_eager_frozenrepro.json",
        {
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
            "max_steps": max_steps,
            "log_every": 50,
            "save_every": max_steps,
            "seed": seed,
        },
    )


def main() -> None:
    seed = 91
    twitter_pools, twitter_meta = load_twitter_pools()
    ruemotions_pools, ruemotions_meta = load_ruemotions_pools()

    twitter_selected = select_balanced(twitter_pools, count=3200, seed=seed + 10)
    twitter_same = make_same_label_records(
        twitter_selected,
        twitter_pools,
        seed=seed + 11,
        source="AiLab-IMCS-UL/twitter_emotions-ru:ekman_same_label_balanced",
    )
    make_mix(
        name="cedr_twitter_ekman_same_label",
        cedr_records=twitter_same,
        cedr_summary={
            "source": "AiLab-IMCS-UL/twitter_emotions-ru",
            "construction": "same-label positive, other-label negatives",
            "loader": twitter_meta,
            "selected_by_group": dict(Counter(record["metadata"]["group"] for record in twitter_same)),
        },
        seed=seed,
    )

    twitter_topic = make_topic_preserving_records(
        twitter_selected,
        twitter_pools,
        seed=seed + 12,
        source="AiLab-IMCS-UL/twitter_emotions-ru:ekman_topic_preserving_same_emotion_hardneg",
    )
    make_mix(
        name="cedr_twitter_ekman_topic_hardneg",
        cedr_records=twitter_topic,
        cedr_summary={
            "source": "AiLab-IMCS-UL/twitter_emotions-ru",
            "construction": "same-label lexical-near positive, same-label lexical-far hard negatives plus other-label negatives",
            "loader": twitter_meta,
            "selected_by_group": dict(Counter(record["metadata"]["group"] for record in twitter_topic)),
        },
        seed=seed,
    )

    merged = merge_pools(twitter_pools, ruemotions_pools)
    mixed_selected = select_balanced(merged, count=3200, seed=seed + 20)
    twitter_ruemotion_topic = make_topic_preserving_records(
        mixed_selected,
        merged,
        seed=seed + 21,
        source="twitter_emotions_ru_plus_ruemotions:topic_preserving_same_emotion_hardneg",
    )
    make_mix(
        name="cedr_twitter_ruemotions_topic_hardneg",
        cedr_records=twitter_ruemotion_topic,
        cedr_summary={
            "source": "AiLab-IMCS-UL/twitter_emotions-ru + Darkester/RuEmotions",
            "construction": "same-label lexical-near positive, same-label lexical-far hard negatives plus other-label negatives",
            "twitter_loader": twitter_meta,
            "ruemotions_loader": ruemotions_meta,
            "selected_by_group": dict(Counter(record["metadata"]["group"] for record in twitter_ruemotion_topic)),
        },
        seed=seed,
    )


if __name__ == "__main__":
    main()
