from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
CONFIG_DIR = ROOT / "configs" / "experiments"
CHECKPOINT_DIR = ROOT / "experiments" / "exp01_reinit_fair" / "checkpoints"


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


def main() -> None:
    seed = 71
    geracl_path = DATA_DIR / "open_ru_1r_nc_geracl.jsonl"
    habr_path = DATA_DIR / "open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"
    deepvk_path = DATA_DIR / "open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl"
    grandmaster_path = DATA_DIR / "open_ru_1r_nc_grandmaster_clustered_3200.jsonl"
    remaining_geracl_path = DATA_DIR / "open_ru_1r_nc_mixb_geracl_remaining_9600.jsonl"

    geracl = read_jsonl(geracl_path)
    habr = read_jsonl(habr_path)
    deepvk = read_jsonl(deepvk_path)
    grandmaster = read_jsonl(grandmaster_path)

    selected = {
        "geracl": sample(geracl, count=6400, seed=seed),
        "habr_harder_full": habr,
        "deepvk_filtered": sample(deepvk, count=3200, seed=seed + 2),
        "grandmaster": grandmaster,
    }

    mixed: list[dict[str, Any]] = []
    for records in selected.values():
        mixed.extend(records)
    random.Random(seed + 3).shuffle(mixed)

    mix_path = DATA_DIR / "open_ru_1r_nc_mixh_habrfull_geracl6400_habr4369_deepvk3200_grandmaster3200_17169.jsonl"
    summary_path = mix_path.with_name(mix_path.stem + "_summary.json")
    write_jsonl(mix_path, mixed)
    write_json(
        summary_path,
        {
            "output": str(mix_path.relative_to(ROOT)),
            "seed": seed,
            "base_run": "Mix H",
            "change": "Use all Habr-harder sim021_len records in stage 1 instead of sampling 3200.",
            "counts": {
                "geracl_source": len(geracl),
                "geracl_used": len(selected["geracl"]),
                "habr_harder_source": len(habr),
                "habr_harder_used": len(selected["habr_harder_full"]),
                "deepvk_filtered_source": len(deepvk),
                "deepvk_used": len(selected["deepvk_filtered"]),
                "grandmaster_source": len(grandmaster),
                "grandmaster_used": len(selected["grandmaster"]),
                "total": len(mixed),
            },
            "batch_size": 4,
            "max_steps_1x": math.ceil(len(mixed) / 4),
            "source_paths": {
                "geracl": str(geracl_path.relative_to(ROOT)),
                "habr_harder": str(habr_path.relative_to(ROOT)),
                "deepvk_filtered": str(deepvk_path.relative_to(ROOT)),
                "grandmaster": str(grandmaster_path.relative_to(ROOT)),
            },
            "stage2_remaining_geracl": str(remaining_geracl_path.relative_to(ROOT)),
        },
    )

    stage1_name = "exp01r_nc_mixh_habrfull_4096"
    stage1_output = CHECKPOINT_DIR / "open_ru_1r_nc_mixh_habrfull_4096"
    stage1_config = {
        "name": stage1_name,
        "description": "Fair 1R-NC Mix H variant: reinitialize original latent-attention block and train stage 1 with full Habr-harder sim021_len in the Mix H recipe.",
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "flash_attention_2",
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

    stage2_name = "exp01r_nc_mixh_habrfull_plus_geracl_remaining_4096"
    stage2_output = CHECKPOINT_DIR / "open_ru_1r_nc_mixh_habrfull_plus_geracl_remaining_4096"
    stage2_config = {
        "name": stage2_name,
        "description": "Continuation ablation: start from Mix H Habr-full checkpoint and train one pass over the standard remaining GeRaCl recovery split.",
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "flash_attention_2",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": str((stage1_output / "latest.pt").relative_to(ROOT)),
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": str(remaining_geracl_path.relative_to(ROOT)),
        "output_dir": str(stage2_output.relative_to(ROOT)),
        "max_length": 4096,
        "batch_size": 8,
        "learning_rate": 3e-6,
        "weight_decay": 0.01,
        "temperature": 0.02,
        "max_steps": 1200,
        "log_every": 50,
        "save_every": 1200,
        "seed": 75,
    }

    write_json(CONFIG_DIR / f"{stage1_name}.json", stage1_config)
    write_json(CONFIG_DIR / f"{stage2_name}.json", stage2_config)

    print(f"Wrote {mix_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote configs/experiments/{stage1_name}.json")
    print(f"Wrote configs/experiments/{stage2_name}.json")


if __name__ == "__main__":
    main()
