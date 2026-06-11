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


def build_ruizard_records(*, count: int, seed: int, ignore_overlap: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flagged = set() if ignore_overlap else load_flagged_rows(
        ROOT / "results" / "contamination" / "ruizard_cedr" / "flagged_ruizard_rows.json"
    )
    dataset = load_dataset("Djacon/ru-izard-emotions", cache_dir=str(CACHE_DIR))
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_counts: Counter[str] = Counter()
    skipped_flagged = 0
    skipped_short = 0
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
            dedupe_key = text
            if dedupe_key in seen_texts:
                continue
            seen_texts.add(dedupe_key)
            labels = [label for label in CEDR_LABELS if int(row.get(label, 0)) == 1]
            is_neutral = int(row.get("neutral", 0)) == 1 and not labels
            groups = labels or (["neutral"] if is_neutral else [])
            for group in groups:
                item = {
                    "split": split,
                    "index": index,
                    "text": row["text"],
                    "group": group,
                    "labels": labels if labels else ["neutral"],
                }
                pools[group].append(item)
                all_counts[group] += 1

    rng = random.Random(seed)
    target_by_group = {group: count // len(GROUPS) for group in GROUPS}
    for group in GROUPS[: count % len(GROUPS)]:
        target_by_group[group] += 1

    selected: list[dict[str, Any]] = []
    for group, target in target_by_group.items():
        pool = pools[group][:]
        rng.shuffle(pool)
        if len(pool) < target:
            raise ValueError(f"Not enough RuIzard rows for {group}: need {target}, got {len(pool)}")
        selected.extend(pool[:target])
    rng.shuffle(selected)

    records: list[dict[str, Any]] = []
    selected_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        selected_by_group[item["group"]].append(item)

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
                "source": "Djacon/ru-izard-emotions:cedr_compatible_emotion_contrastive",
                "query": CEDR_PREFIX + item["text"],
                "positive": CEDR_PREFIX + positive["text"],
                "negatives": negatives,
                "metadata": {
                    "group": group,
                    "labels": item["labels"],
                    "ruizard_split": item["split"],
                    "ruizard_index": item["index"],
                    "positive_group": positive["group"],
                    "positive_ruizard_split": positive["split"],
                    "positive_ruizard_index": positive["index"],
                    "ignore_cedr_overlap": ignore_overlap,
                },
                "objective": "contrastive",
            }
        )

    summary = {
        "requested": count,
        "kept": len(records),
        "groups": GROUPS,
        "target_by_group": target_by_group,
        "available_by_group": dict(all_counts),
        "selected_by_group": dict(Counter(record["metadata"]["group"] for record in records)),
        "skipped_flagged_overlap": skipped_flagged,
        "skipped_short": skipped_short,
        "ignore_cedr_overlap": ignore_overlap,
        "prefix": CEDR_PREFIX,
    }
    return records, summary


def main() -> None:
    seed = 71
    ignore_overlap = True
    ruizard_count = 3200
    geracl_path = DATA_DIR / "open_ru_1r_nc_geracl.jsonl"
    habr_path = DATA_DIR / "open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"
    deepvk_path = DATA_DIR / "open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl"
    grandmaster_path = DATA_DIR / "open_ru_1r_nc_grandmaster_clustered_3200.jsonl"
    remaining_geracl_path = DATA_DIR / "open_ru_1r_nc_mixb_geracl_remaining_9600.jsonl"

    geracl = read_jsonl(geracl_path)
    habr = read_jsonl(habr_path)
    deepvk = read_jsonl(deepvk_path)
    grandmaster = read_jsonl(grandmaster_path)
    ruizard, ruizard_summary = build_ruizard_records(count=ruizard_count, seed=seed + 10, ignore_overlap=ignore_overlap)

    ruizard_path = DATA_DIR / "open_ru_1r_nc_ruizard_emotion_contrastive_3200.jsonl"
    write_jsonl(ruizard_path, ruizard)
    write_json(ruizard_path.with_name(ruizard_path.stem + "_summary.json"), ruizard_summary)

    selected = {
        "geracl": sample(geracl, count=6400, seed=seed),
        "habr_harder_full": habr,
        "deepvk_filtered": sample(deepvk, count=3200, seed=seed + 2),
        "grandmaster": grandmaster,
        "ruizard_emotion": ruizard,
    }

    mixed: list[dict[str, Any]] = []
    for records in selected.values():
        mixed.extend(records)
    random.Random(seed + 13).shuffle(mixed)

    mix_path = DATA_DIR / "open_ru_1r_nc_mixh_habrfull_ruizard_geracl6400_habr4369_deepvk3200_grandmaster3200_ruizard3200_20369.jsonl"
    summary_path = mix_path.with_name(mix_path.stem + "_summary.json")
    write_jsonl(mix_path, mixed)
    write_json(
        summary_path,
        {
            "output": str(mix_path.relative_to(ROOT)),
            "seed": seed,
            "base_run": "Mix H HabrFull",
            "change": "Add RuIzard CEDR-compatible emotion contrastive component at weight 1.",
            "ratio": {
                "geracl": 2,
                "habr_harder_full": "full_4369",
                "deepvk_ru_hnp_filtered": 1,
                "grandmaster": 1,
                "ruizard_emotion": 1,
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
                "ruizard_used": len(selected["ruizard_emotion"]),
                "total": len(mixed),
            },
            "batch_size": 4,
            "max_steps_1x": math.ceil(len(mixed) / 4),
            "source_paths": {
                "geracl": str(geracl_path.relative_to(ROOT)),
                "habr_harder": str(habr_path.relative_to(ROOT)),
                "deepvk_filtered": str(deepvk_path.relative_to(ROOT)),
                "grandmaster": str(grandmaster_path.relative_to(ROOT)),
                "ruizard_emotion": str(ruizard_path.relative_to(ROOT)),
            },
            "ruizard_summary": ruizard_summary,
            "stage2_remaining_geracl": str(remaining_geracl_path.relative_to(ROOT)),
        },
    )

    stage1_name = "exp01r_nc_mixh_habrfull_ruizard_4096_eager_frozenrepro"
    stage1_output = CHECKPOINT_DIR / "open_ru_1r_nc_mixh_habrfull_ruizard_4096_eager_frozenrepro"
    stage1_config = {
        "name": stage1_name,
        "description": "Eager frozen-wrapper reproduction: Mix H Habr-full plus RuIzard emotion contrastive stage 1.",
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
        "batch_size": 4,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "temperature": 0.02,
        "max_steps": math.ceil(len(mixed) / 4),
        "log_every": 50,
        "save_every": math.ceil(len(mixed) / 4),
        "seed": seed,
    }

    stage2_name = "exp01r_nc_mixh_habrfull_ruizard_plus_geracl_remaining_4096_eager_frozenrepro"
    stage2_output = CHECKPOINT_DIR / "open_ru_1r_nc_mixh_habrfull_ruizard_plus_geracl_remaining_4096_eager_frozenrepro"
    stage2_config = {
        "name": stage2_name,
        "description": "Stage 2 eager continuation: start from Mix H Habr-full + RuIzard and train one pass over remaining GeRaCl.",
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

    print(f"Wrote {ruizard_path.relative_to(ROOT)}")
    print(f"Wrote {mix_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote configs/experiments/{stage1_name}.json")
    print(f"Wrote configs/experiments/{stage2_name}.json")


if __name__ == "__main__":
    main()
