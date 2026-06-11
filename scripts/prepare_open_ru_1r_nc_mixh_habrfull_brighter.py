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
CHECKPOINT_DIR = ROOT / "experiments" / "exp01_reinit_fair" / "checkpoints"
CACHE_DIR = ROOT / "results" / "mteb_cache"

CEDR_PREFIX = "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: "
GROUPS = ["neutral", "joy", "sadness", "anger", "surprise", "fear"]
CEDR_LABELS = ["joy", "sadness", "anger", "surprise", "fear"]


def normalize_text(value: Any) -> str:
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
    selected = records[:]
    random.Random(seed).shuffle(selected)
    return selected[:count]


def load_flagged_rows(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(str(row["split"]), int(row["index"])) for row in rows}


def build_brighter_records(*, count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flagged = load_flagged_rows(ROOT / "results" / "contamination" / "brighter_cedr" / "flagged_brighter_rows.json")
    dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "rus", cache_dir=str(CACHE_DIR))

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    available_by_group: Counter[str] = Counter()
    skipped_flagged = 0
    skipped_short = 0
    skipped_disgust_only = 0
    kept_with_disgust_label_dropped = 0
    multilabel_rows = 0
    skipped_duplicate = 0
    seen_texts: set[str] = set()

    for split, ds in dataset.items():
        for index, row in enumerate(ds):
            if (split, index) in flagged:
                skipped_flagged += 1
                continue
            text = normalize_text(row["text"])
            if len(text) < 8:
                skipped_short += 1
                continue
            if text in seen_texts:
                skipped_duplicate += 1
                continue
            labels = [label for label in CEDR_LABELS if int(row.get(label, 0)) == 1]
            has_disgust = int(row.get("disgust", 0)) == 1
            if has_disgust and not labels:
                skipped_disgust_only += 1
                continue
            if has_disgust:
                kept_with_disgust_label_dropped += 1
            if len(labels) > 1:
                multilabel_rows += 1
            groups = labels if labels else ["neutral"]
            seen_texts.add(text)
            for group in groups:
                item = {
                    "split": split,
                    "index": index,
                    "id": row.get("id"),
                    "text": row["text"],
                    "group": group,
                    "labels": labels if labels else [],
                }
                pools[group].append(item)
                available_by_group[group] += 1

    rng = random.Random(seed)
    # Match CEDR train priors: empty labels are common, while fear/anger are rarer.
    cedr_train_units = {
        "neutral": 3043,
        "joy": 1569,
        "sadness": 1417,
        "surprise": 607,
        "fear": 589,
        "anger": 411,
    }
    total_units = sum(cedr_train_units.values())
    target_by_group = {
        group: math.floor(count * cedr_train_units[group] / total_units)
        for group in GROUPS
    }
    remainder = count - sum(target_by_group.values())
    fractional_order = sorted(
        GROUPS,
        key=lambda group: (count * cedr_train_units[group] / total_units) - target_by_group[group],
        reverse=True,
    )
    for group in fractional_order[:remainder]:
        target_by_group[group] += 1
    capped_groups: dict[str, int] = {}
    for group, target in list(target_by_group.items()):
        cap = len(pools[group])
        if target > cap:
            capped_groups[group] = cap
            target_by_group[group] = cap
    while sum(target_by_group.values()) < count:
        progressed = False
        for group in sorted(GROUPS, key=lambda name: len(pools[name]) - target_by_group[name], reverse=True):
            if target_by_group[group] < len(pools[group]):
                target_by_group[group] += 1
                progressed = True
                if sum(target_by_group.values()) == count:
                    break
        if not progressed:
            break

    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    globally_selected: set[tuple[str, int]] = set()
    for group, target in sorted(target_by_group.items(), key=lambda item: len(pools[item[0]])):
        pool = pools[group][:]
        rng.shuffle(pool)
        selected = [
            item for item in pool
            if (item["split"], item["index"]) not in globally_selected
        ][:target]
        if len(selected) < target:
            selected_keys = {(item["split"], item["index"]) for item in selected}
            selected.extend(
                item for item in pool
                if (item["split"], item["index"]) not in selected_keys
            )
            selected = selected[:target]
        if len(selected) < target:
            raise ValueError(f"Not enough BRIGHTER rows for {group}: need {target}, got {len(pool)}")
        selected_by_group[group] = selected
        globally_selected.update((item["split"], item["index"]) for item in selected)

    selected = [item for group_items in selected_by_group.values() for item in group_items]
    rng.shuffle(selected)

    records: list[dict[str, Any]] = []
    for item in selected:
        group = item["group"]
        positive_pool = [candidate for candidate in selected_by_group[group] if candidate is not item]
        if not positive_pool:
            positive_pool = [candidate for candidate in pools[group] if candidate["text"] != item["text"]]
        positive = rng.choice(positive_pool)
        negatives = []
        for negative_group in GROUPS:
            if negative_group == group:
                continue
            negative_pool = selected_by_group[negative_group] or pools[negative_group]
            negatives.append(CEDR_PREFIX + rng.choice(negative_pool)["text"])
        records.append(
            {
                "source": "brighter-dataset/BRIGHTER-emotion-categories:rus:cedr_compatible_clean_no_disgust",
                "query": CEDR_PREFIX + item["text"],
                "positive": CEDR_PREFIX + positive["text"],
                "negatives": negatives,
                "metadata": {
                    "group": group,
                    "labels": item["labels"],
                    "brighter_split": item["split"],
                    "brighter_index": item["index"],
                    "brighter_id": item.get("id"),
                    "positive_group": positive["group"],
                    "positive_brighter_split": positive["split"],
                    "positive_brighter_index": positive["index"],
                    "contamination_filter": "results/contamination/brighter_cedr/flagged_brighter_rows.json",
                    "dropped_disgust": True,
                    "multilabel_allowed": True,
                },
                "objective": "contrastive",
            }
        )

    summary = {
        "requested": count,
        "kept": len(records),
        "groups": GROUPS,
        "target_by_group": target_by_group,
        "target_distribution": "CEDR train empty/label prior",
        "cedr_train_units": cedr_train_units,
        "capped_groups": capped_groups,
        "available_by_group": dict(available_by_group),
        "selected_by_group": dict(Counter(record["metadata"]["group"] for record in records)),
        "skipped_flagged_overlap": skipped_flagged,
        "skipped_short": skipped_short,
        "skipped_duplicate": skipped_duplicate,
        "skipped_disgust_only": skipped_disgust_only,
        "kept_with_disgust_label_dropped": kept_with_disgust_label_dropped,
        "multilabel_rows_after_filters": multilabel_rows,
        "prefix": CEDR_PREFIX,
    }
    return records, summary


def main() -> None:
    seed = 77
    brighter_count = 3200
    geracl_path = DATA_DIR / "open_ru_1r_nc_geracl.jsonl"
    habr_path = DATA_DIR / "open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"
    deepvk_path = DATA_DIR / "open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl"
    grandmaster_path = DATA_DIR / "open_ru_1r_nc_grandmaster_clustered_3200.jsonl"
    remaining_geracl_path = DATA_DIR / "open_ru_1r_nc_mixb_geracl_remaining_9600.jsonl"

    geracl = read_jsonl(geracl_path)
    habr = read_jsonl(habr_path)
    deepvk = read_jsonl(deepvk_path)
    grandmaster = read_jsonl(grandmaster_path)
    brighter, brighter_summary = build_brighter_records(count=brighter_count, seed=seed + 10)

    brighter_path = DATA_DIR / "open_ru_1r_nc_brighter_cedr_clean_no_disgust_contrastive_3200.jsonl"
    write_jsonl(brighter_path, brighter)
    write_json(brighter_path.with_name(brighter_path.stem + "_summary.json"), brighter_summary)

    selected = {
        "geracl": sample(geracl, count=6400, seed=seed),
        "habr_harder_full": habr,
        "deepvk_filtered": sample(deepvk, count=3200, seed=seed + 2),
        "grandmaster": grandmaster,
        "brighter_cedr_clean": brighter,
    }

    mixed: list[dict[str, Any]] = []
    for records in selected.values():
        mixed.extend(records)
    random.Random(seed + 13).shuffle(mixed)

    mix_path = DATA_DIR / "open_ru_1r_nc_mixh_habrfull_brighter_geracl6400_habr4369_deepvk3200_grandmaster3200_brighter3200_20369.jsonl"
    summary_path = mix_path.with_name(mix_path.stem + "_summary.json")
    write_jsonl(mix_path, mixed)
    write_json(
        summary_path,
        {
            "output": str(mix_path.relative_to(ROOT)),
            "seed": seed,
            "base_run": "Mix H HabrFull",
            "change": "Replace RuIzard with BRIGHTER Russian CEDR-compatible clean no-disgust component at weight 1.",
            "ratio": {
                "geracl": 2,
                "habr_harder_full": "full_4369",
                "deepvk_ru_hnp_filtered": 1,
                "grandmaster": 1,
                "brighter_cedr_clean_no_disgust": 1,
            },
            "counts": {
                "geracl_source": len(geracl),
                "geracl_used": len(selected["geracl"]),
                "habr_harder_source": len(habr),
                "habr_harder_used": len(selected["habr_harder_full"]),
                "deepvk_filtered_source": len(deepvk),
                "deepvk_used": len(selected["deepvk_filtered"]),
                "grandmaster_source": len(grandmaster),
                "grandmaster_used": len(selected["grandmaster"]),
                "brighter_used": len(selected["brighter_cedr_clean"]),
                "total": len(mixed),
            },
            "batch_size": 3,
            "max_steps_1x": math.ceil(len(mixed) / 3),
            "source_paths": {
                "geracl": str(geracl_path.relative_to(ROOT)),
                "habr_harder": str(habr_path.relative_to(ROOT)),
                "deepvk_filtered": str(deepvk_path.relative_to(ROOT)),
                "grandmaster": str(grandmaster_path.relative_to(ROOT)),
                "brighter_cedr_clean_no_disgust": str(brighter_path.relative_to(ROOT)),
            },
            "brighter_summary": brighter_summary,
            "stage2_remaining_geracl": str(remaining_geracl_path.relative_to(ROOT)),
        },
    )

    stage1_name = "exp01r_nc_mixh_habrfull_brighter_4096_eager_frozenrepro"
    stage1_output = CHECKPOINT_DIR / "open_ru_1r_nc_mixh_habrfull_brighter_4096_eager_frozenrepro"
    stage1_steps = math.ceil(len(mixed) / 3)
    stage1_config = {
        "name": stage1_name,
        "description": "Eager frozen-wrapper reproduction: Mix H Habr-full plus clean BRIGHTER CEDR-compatible no-disgust emotion component.",
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": None,
        "freeze_llm": True,
        "reinit_latent": True,
        "data_path": str(mix_path.relative_to(ROOT)),
        "output_dir": str(stage1_output.relative_to(ROOT)),
        "max_length": 4096,
        "batch_size": 3,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "temperature": 0.02,
        "max_steps": stage1_steps,
        "log_every": 50,
        "save_every": stage1_steps,
        "seed": seed,
    }

    stage2_name = "exp01r_nc_mixh_habrfull_brighter_plus_geracl_remaining_4096_eager_frozenrepro"
    stage2_output = CHECKPOINT_DIR / "open_ru_1r_nc_mixh_habrfull_brighter_plus_geracl_remaining_4096_eager_frozenrepro"
    stage2_config = {
        "name": stage2_name,
        "description": "Stage 2 eager continuation: start from Mix H Habr-full + BRIGHTER and train one pass over remaining GeRaCl.",
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": str((stage1_output / "latest.pt").relative_to(ROOT)),
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": str(remaining_geracl_path.relative_to(ROOT)),
        "output_dir": str(stage2_output.relative_to(ROOT)),
        "max_length": 4096,
        "batch_size": 4,
        "learning_rate": 3e-6,
        "weight_decay": 0.01,
        "temperature": 0.02,
        "max_steps": 2400,
        "log_every": 50,
        "save_every": 2400,
        "seed": 75,
    }

    write_json(CONFIG_DIR / f"{stage1_name}.json", stage1_config)
    write_json(CONFIG_DIR / f"{stage2_name}.json", stage2_config)

    print(f"Wrote {brighter_path.relative_to(ROOT)}")
    print(f"Wrote {mix_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote configs/experiments/{stage1_name}.json")
    print(f"Wrote configs/experiments/{stage2_name}.json")


if __name__ == "__main__":
    main()
