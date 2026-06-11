from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


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


def shuffled(records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    records = records[:]
    random.Random(seed).shuffle(records)
    return records


def take(records: list[dict[str, Any]], count: int, *, name: str) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"{name}: need {count} records, got {len(records)}")
    return records[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mix N three-stage correction data and config.")
    parser.add_argument("--seed", type=int, default=191)
    parser.add_argument("--geracl-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_geracl.jsonl"))
    parser.add_argument("--habr-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"))
    parser.add_argument("--mvrcii-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_sensitive_topic_mvrcii_3200.jsonl"))
    parser.add_argument("--uc-berkeley-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_sensitive_topic_uc_berkeley_3200.jsonl"))
    parser.add_argument("--correction-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_mixn_correction_geracl3200_habr800_mvrcii400_ucb400_seed191.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_mixn_three_stage_summary.json"))
    parser.add_argument("--config-out", type=Path, default=Path("configs/experiments/exp01r_nc_mixn_three_stage_correction_4096.json"))
    args = parser.parse_args()

    selected = {
        "geracl_rehearsal": take(shuffled(read_jsonl(args.geracl_path), seed=args.seed), 3200, name="geracl"),
        "habr_harder": take(shuffled(read_jsonl(args.habr_path), seed=args.seed + 1), 800, name="habr"),
        "sensitive_mvrcii": take(shuffled(read_jsonl(args.mvrcii_path), seed=args.seed + 2), 400, name="mvrcii"),
        "sensitive_uc_berkeley": take(shuffled(read_jsonl(args.uc_berkeley_path), seed=args.seed + 3), 400, name="uc_berkeley"),
    }
    correction: list[dict[str, Any]] = []
    for records in selected.values():
        correction.extend(records)
    correction = shuffled(correction, seed=args.seed + 4)
    write_jsonl(args.correction_out, correction)

    config = {
        "name": "exp01r_nc_mixn_three_stage_correction_4096",
        "description": (
            "Fair 1R-NC Mix N: Mix F broad stage, one pass over Mix F GeRaCl remainder, "
            "then low-LR correction with GeRaCl rehearsal:Habr:MVR-CII:UC Berkeley = 4:1:0.5:0.5."
        ),
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "flash_attention_2",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": None,
        "freeze_llm": True,
        "reinit_latent": True,
        "data_path": "data/contrastive/open_ru_1r_nc_mixf_geracl2_habr1_deepvk1_groundedstrict1_16000.jsonl",
        "output_dir": "experiments/exp01_reinit_fair/checkpoints/open_ru_1r_nc_mixn_three_stage_correction_4096",
        "max_length": 4096,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "temperature": 0.02,
        "max_steps": 6400,
        "log_every": 50,
        "save_every": 4000,
        "seed": args.seed,
        "stages": [
            {
                "name": "mixf_broad_stage",
                "data_path": "data/contrastive/open_ru_1r_nc_mixf_geracl2_habr1_deepvk1_groundedstrict1_16000.jsonl",
                "max_steps": 4000,
                "batch_size": 4,
                "learning_rate": 1e-5,
                "save_every": 4000,
            },
            {
                "name": "mixf_geracl_remaining_seed53",
                "data_path": "data/contrastive/open_ru_1r_nc_mixf_geracl_remaining_seed53_9600.jsonl",
                "max_steps": 1200,
                "batch_size": 8,
                "learning_rate": 3e-6,
                "save_every": 1200,
            },
            {
                "name": "low_lr_sensitive_terra_correction",
                "data_path": str(args.correction_out),
                "max_steps": 1200,
                "batch_size": 4,
                "learning_rate": 2e-6,
                "save_every": 1200,
            },
        ],
    }
    write_json(args.config_out, config)

    summary = {
        "seed": args.seed,
        "correction_output": str(args.correction_out),
        "config_output": str(args.config_out),
        "correction_counts": {name: len(records) for name, records in selected.items()},
        "correction_total": len(correction),
        "correction_steps_batch4": len(correction) // 4,
        "stage_plan": [
            {"stage": 1, "data": config["stages"][0]["data_path"], "steps": 4000, "batch_size": 4, "lr": 1e-5},
            {"stage": 2, "data": config["stages"][1]["data_path"], "steps": 1200, "batch_size": 8, "lr": 3e-6},
            {"stage": 3, "data": config["stages"][2]["data_path"], "steps": 1200, "batch_size": 4, "lr": 2e-6},
        ],
        "source_paths": {
            "geracl": str(args.geracl_path),
            "habr": str(args.habr_path),
            "mvrcii": str(args.mvrcii_path),
            "uc_berkeley": str(args.uc_berkeley_path),
        },
    }
    write_json(args.summary_out, summary)
    print(f"Wrote {args.correction_out}")
    print(f"Wrote {args.config_out}")
    print(f"Wrote {args.summary_out}")


if __name__ == "__main__":
    main()
