from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
CONFIG_DIR = ROOT / "configs" / "experiments"
CACHE_DIR = ROOT / "results" / "mteb_cache"
FLAG_DIR = ROOT / "results" / "contamination" / "cedr_candidates"

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
    text = str(value).replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def quality_ok(text: str, *, strict: bool) -> bool:
    text = normalize_text(text)
    if len(text) < (24 if strict else 16) or len(text) > (260 if strict else 360):
        return False
    if re.search(r"https?://|www\.|@\w+", text, flags=re.I):
        return False
    cyr = len(re.findall(r"[А-Яа-я]", text))
    alpha = len(re.findall(r"[A-Za-zА-Яа-я]", text))
    if alpha and cyr / alpha < (0.75 if strict else 0.55):
        return False
    if len(re.findall(r"#\w+", text)) > 1:
        return False
    if re.search(r"(.)\1{5,}", text):
        return False
    if strict and len(re.findall(r"\w+", text, flags=re.U)) < 5:
        return False
    return True


def load_flagged(name: str) -> set[tuple[str, int]]:
    path = FLAG_DIR / f"{name}_flagged_rows.json"
    if not path.exists():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(str(row["split"]), int(row["index"])) for row in rows}


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


def target_counts(count: int, *, prior: str) -> dict[str, int]:
    if prior == "uniform":
        base = {emotion: count // len(EMOTIONS) for emotion in EMOTIONS}
        for emotion in EMOTIONS[: count - sum(base.values())]:
            base[emotion] += 1
        return base
    total = sum(CEDR_TRAIN_UNITS.values())
    targets = {emotion: math.floor(count * CEDR_TRAIN_UNITS[emotion] / total) for emotion in EMOTIONS}
    remainder = count - sum(targets.values())
    order = sorted(
        EMOTIONS,
        key=lambda emotion: (count * CEDR_TRAIN_UNITS[emotion] / total) - targets[emotion],
        reverse=True,
    )
    for emotion in order[:remainder]:
        targets[emotion] += 1
    return targets


def add_row(
    pools: dict[str, list[dict[str, Any]]],
    seen: set[str],
    *,
    source: str,
    split: str,
    index: int,
    text: str,
    label: str,
    strict: bool,
    metadata: dict[str, Any] | None = None,
) -> None:
    text = normalize_text(text)
    key = re.sub(r"\W+", " ", text.lower(), flags=re.U).strip()
    if label not in EMOTIONS or not quality_ok(text, strict=strict) or key in seen:
        return
    seen.add(key)
    pools[label].append(
        {
            "source_dataset": source,
            "split": split,
            "index": index,
            "text": text,
            "label": label,
            "metadata": metadata or {},
        }
    )


def load_source_pools(*, sources: list[str], strict: bool) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    skipped = Counter()

    if "twitter" in sources:
        flagged = load_flagged("twitter_emotions_ekman")
        dataset = load_dataset("AiLab-IMCS-UL/twitter_emotions-ru", "simplified_ekman", cache_dir=str(CACHE_DIR))
        for split, ds in dataset.items():
            label_feature = ds.features["labels_ekman"]
            for index, row in enumerate(ds):
                if (split, index) in flagged:
                    skipped["twitter_flagged"] += 1
                    continue
                add_row(
                    pools,
                    seen,
                    source="AiLab-IMCS-UL/twitter_emotions-ru",
                    split=split,
                    index=index,
                    text=row["ru_text"],
                    label=label_feature.int2str(row["labels_ekman"]),
                    strict=strict,
                )

    if "go_ekman" in sources:
        flagged = load_flagged("go_ekman")
        dataset = load_dataset("SkyWater21/ru_go_emotions_ekman", "simplified_ekman", cache_dir=str(CACHE_DIR))
        label_names = dataset["train"].features["labels_ekman"].feature.names
        for split, ds in dataset.items():
            for index, row in enumerate(ds):
                if (split, index) in flagged:
                    skipped["go_ekman_flagged"] += 1
                    continue
                labels = [label_names[label] for label in row["labels_ekman"] if label_names[label] in EMOTIONS]
                if len(labels) != 1:
                    skipped["go_ekman_multilabel_or_unmapped"] += 1
                    continue
                add_row(
                    pools,
                    seen,
                    source="SkyWater21/ru_go_emotions_ekman",
                    split=split,
                    index=index,
                    text=row["ru_text"],
                    label=labels[0],
                    strict=strict,
                )

    if "brighter" in sources:
        flagged = load_flagged("brighter")
        dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "rus", cache_dir=str(CACHE_DIR))
        label_names = ["anger", "fear", "joy", "sadness", "surprise"]
        for split, ds in dataset.items():
            for index, row in enumerate(ds):
                if (split, index) in flagged:
                    skipped["brighter_flagged"] += 1
                    continue
                labels = [label for label in label_names if int(row.get(label, 0)) == 1]
                if len(labels) != 1:
                    skipped["brighter_multilabel_or_no_label"] += 1
                    continue
                add_row(
                    pools,
                    seen,
                    source="brighter-dataset/BRIGHTER-emotion-categories:rus",
                    split=split,
                    index=index,
                    text=row["text"],
                    label=labels[0],
                    strict=strict,
                )

    if "ruemotions" in sources:
        flagged = load_flagged("ruemotions")
        dataset = load_dataset("Darkester/RuEmotions", cache_dir=str(CACHE_DIR))
        for split, ds in dataset.items():
            for index, row in enumerate(ds):
                if (split, index) in flagged:
                    skipped["ruemotions_flagged"] += 1
                    continue
                label = RUEMOTIONS_MAP.get(str(row["emotion"]).strip().lower())
                if label is None:
                    skipped["ruemotions_unmapped"] += 1
                    continue
                add_row(
                    pools,
                    seen,
                    source="Darkester/RuEmotions",
                    split=split,
                    index=index,
                    text=row["text"],
                    label=label,
                    strict=strict,
                    metadata={"raw_label": row["emotion"]},
                )

    meta = {
        "sources": sources,
        "strict": strict,
        "skipped": dict(skipped),
        "available_by_group": {emotion: len(pools[emotion]) for emotion in EMOTIONS},
        "source_counts": dict(Counter(row["source_dataset"] for rows in pools.values() for row in rows)),
    }
    return pools, meta


def select_rows(
    pools: dict[str, list[dict[str, Any]]],
    *,
    count: int,
    prior: str,
    seed: int,
    pool_limit_per_label: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected = []
    for emotion, target in target_counts(count, prior=prior).items():
        rows = pools[emotion][:]
        rng.shuffle(rows)
        rows = rows[:pool_limit_per_label]
        if len(rows) < target:
            raise ValueError(f"Not enough {emotion} rows: need {target}, got {len(rows)}")
        selected.extend(rows[:target])
    rng.shuffle(selected)
    return selected


def row_id(row: dict[str, Any]) -> str:
    return f"{row['source_dataset']}::{row['split']}::{row['index']}"


def mine_records(
    selected: list[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    *,
    source_name: str,
    seed: int,
    negatives_per_record: int,
    pool_limit_per_label: int,
    positive_min_sim: float,
    positive_max_sim: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    selected_ids = {row_id(row) for row in selected}
    pool_rows = selected[:]
    pool_ids = set(selected_ids)
    for emotion in EMOTIONS:
        rows = pools[emotion][:]
        rng.shuffle(rows)
        for row in rows:
            identifier = row_id(row)
            if identifier in pool_ids:
                continue
            pool_rows.append(row)
            pool_ids.add(identifier)
            if sum(1 for existing in pool_rows if existing["label"] == emotion) >= pool_limit_per_label:
                break

    by_id = {row_id(row): row for row in pool_rows}
    selected = [row for row in selected if row_id(row) in by_id]
    texts = [row["text"] for row in pool_rows]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    index_by_id = {row_id(row): idx for idx, row in enumerate(pool_rows)}
    labels = [row["label"] for row in pool_rows]
    same_label_indices: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        same_label_indices[label].append(idx)
    selected_indices = [index_by_id[row_id(row)] for row in selected]
    neighbor_count = min(len(pool_rows), 220)
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine", algorithm="brute", n_jobs=-1)
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix[selected_indices], return_distance=True)

    records = []
    stats = Counter()
    positive_sims = []
    hard_negative_sims = []
    for item, item_index, item_distances, item_neighbors in zip(selected, selected_indices, distances, indices, strict=True):
        same = [
            (int(idx), float(1.0 - distance))
            for distance, idx in zip(item_distances, item_neighbors, strict=True)
            if idx != item_index
            and labels[idx] == item["label"]
            and positive_min_sim <= 1.0 - distance <= positive_max_sim
            and pool_rows[int(idx)]["text"] != item["text"]
        ]
        if not same:
            same = [
                (int(idx), float(1.0 - distance))
                for distance, idx in zip(item_distances, item_neighbors, strict=True)
                if idx != item_index and labels[idx] == item["label"] and pool_rows[int(idx)]["text"] != item["text"]
            ]
            stats["positive_fallback"] += 1
        if not same:
            fallback_same = [idx for idx in same_label_indices[item["label"]] if idx != item_index]
            if fallback_same:
                idx = rng.choice(fallback_same)
                same = [(idx, 0.0)]
                stats["positive_random_fallback"] += 1
        if not same:
            stats["no_positive"] += 1
            continue
        same.sort(key=lambda pair: pair[1], reverse=True)
        positive_index, positive_sim = same[min(len(same) - 1, rng.randrange(min(5, len(same))))]

        hard_negatives = [
            (int(idx), float(1.0 - distance))
            for distance, idx in zip(item_distances, item_neighbors, strict=True)
            if labels[idx] != item["label"] and pool_rows[int(idx)]["text"] != item["text"]
        ]
        hard_negatives.sort(key=lambda pair: pair[1], reverse=True)
        selected_negatives = []
        used_negative_labels = set()
        for idx, sim in hard_negatives:
            label = labels[idx]
            if label in used_negative_labels and len(used_negative_labels) < len(EMOTIONS) - 1:
                continue
            selected_negatives.append((idx, sim))
            used_negative_labels.add(label)
            if len(selected_negatives) >= negatives_per_record:
                break
        if len(selected_negatives) < negatives_per_record:
            for idx, sim in hard_negatives:
                if idx not in {existing_idx for existing_idx, _ in selected_negatives}:
                    selected_negatives.append((idx, sim))
                    if len(selected_negatives) >= negatives_per_record:
                        break
        if not selected_negatives:
            stats["no_negative"] += 1
            continue

        positive_sims.append(float(positive_sim))
        hard_negative_sims.extend(float(sim) for _, sim in selected_negatives)
        records.append(
            {
                "source": source_name,
                "query": CEDR_PREFIX + item["text"],
                "positive": CEDR_PREFIX + pool_rows[positive_index]["text"],
                "negatives": [CEDR_PREFIX + pool_rows[idx]["text"] for idx, _ in selected_negatives],
                "metadata": {
                    "group": item["label"],
                    "split": item["split"],
                    "index": item["index"],
                    "source_dataset": item["source_dataset"],
                    "positive_source_dataset": pool_rows[positive_index]["source_dataset"],
                    "positive_similarity": float(positive_sim),
                    "negative_labels": [labels[idx] for idx, _ in selected_negatives],
                    "negative_similarities": [float(sim) for _, sim in selected_negatives],
                },
                "objective": "contrastive",
            }
        )

    rng.shuffle(records)
    summary = {
        "source": source_name,
        "records": len(records),
        "selected_by_group": dict(Counter(record["metadata"]["group"] for record in records)),
        "selected_by_source": dict(Counter(record["metadata"]["source_dataset"] for record in records)),
        "positive_similarity_mean": sum(positive_sims) / len(positive_sims) if positive_sims else None,
        "positive_similarity_min": min(positive_sims) if positive_sims else None,
        "positive_similarity_max": max(positive_sims) if positive_sims else None,
        "negative_similarity_mean": sum(hard_negative_sims) / len(hard_negative_sims) if hard_negative_sims else None,
        "negative_similarity_max": max(hard_negative_sims) if hard_negative_sims else None,
        "stats": dict(stats),
    }
    return records, summary


def sample(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"Need {count} records, got {len(records)}")
    rows = records[:]
    random.Random(seed).shuffle(rows)
    return rows[:count]


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
        "cedr_mined": cedr_records,
    }
    mixed = []
    for records in selected.values():
        mixed.extend(records)
    random.Random(seed + 13).shuffle(mixed)

    data_path = DATA_DIR / f"open_ru_1r_nc_{name}.jsonl"
    write_jsonl(data_path, mixed)
    max_steps = min(6000, math.ceil(len(mixed) / 3))
    write_json(
        data_path.with_name(data_path.stem + "_summary.json"),
        {
            "output": str(data_path.relative_to(ROOT)),
            "seed": seed,
            "base_run": "Mix H HabrFull stage 1",
            "component_counts": {key: len(value) for key, value in selected.items()},
            "total_records": len(mixed),
            "max_steps_batch3": max_steps,
            "max_steps_1x_batch3": math.ceil(len(mixed) / 3),
            "cedr_component": cedr_summary,
        },
    )

    write_json(
        CONFIG_DIR / f"exp01r_nc_{name}_4096_eager_frozenrepro.json",
        {
            "name": f"exp01r_nc_{name}_4096_eager_frozenrepro",
            "description": f"CEDR mined v1 ablation, stage 1 only: {name}",
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
            "save_every": 3000,
            "seed": seed,
        },
    )


def build_variant(
    *,
    name: str,
    sources: list[str],
    strict: bool,
    prior: str,
    count: int,
    seed: int,
    pool_limit_per_label: int,
    positive_min_sim: float,
    positive_max_sim: float,
) -> None:
    pools, loader_meta = load_source_pools(sources=sources, strict=strict)
    selected = select_rows(
        pools,
        count=count,
        prior=prior,
        seed=seed + 10,
        pool_limit_per_label=pool_limit_per_label,
    )
    records, mined_summary = mine_records(
        selected,
        pools,
        source_name=f"cedr_mined_v1:{name}",
        seed=seed + 20,
        negatives_per_record=4,
        pool_limit_per_label=pool_limit_per_label,
        positive_min_sim=positive_min_sim,
        positive_max_sim=positive_max_sim,
    )
    if len(records) < count * 0.95:
        raise ValueError(f"{name}: only mined {len(records)} records from requested {count}")
    make_mix(
        name=name,
        cedr_records=records[:count],
        cedr_summary={
            "construction": "TF-IDF char ngram mined same-label positive plus nearest different-label hard negatives",
            "sources": sources,
            "strict_quality": strict,
            "prior": prior,
            "requested_count": count,
            "pool_limit_per_label": pool_limit_per_label,
            "positive_similarity_window": [positive_min_sim, positive_max_sim],
            "loader": loader_meta,
            "mined": mined_summary,
        },
        seed=seed,
    )


def main() -> None:
    build_variant(
        name="cedr_mined_v1_twitter_prior",
        sources=["twitter"],
        strict=True,
        prior="cedr",
        count=3200,
        seed=211,
        pool_limit_per_label=3000,
        positive_min_sim=0.08,
        positive_max_sim=0.72,
    )
    build_variant(
        name="cedr_mined_v1_multi_prior",
        sources=["twitter", "go_ekman", "brighter", "ruemotions"],
        strict=True,
        prior="cedr",
        count=3200,
        seed=223,
        pool_limit_per_label=3000,
        positive_min_sim=0.08,
        positive_max_sim=0.72,
    )
    build_variant(
        name="cedr_mined_v1_multi_uniform",
        sources=["twitter", "go_ekman", "brighter", "ruemotions"],
        strict=True,
        prior="uniform",
        count=3200,
        seed=227,
        pool_limit_per_label=3000,
        positive_min_sim=0.08,
        positive_max_sim=0.72,
    )


if __name__ == "__main__":
    main()
