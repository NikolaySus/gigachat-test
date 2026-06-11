from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


BASE_CHECKPOINT = (
    "experiments/exp01_reinit_fair/checkpoints/"
    "open_ru_1r_nc_mixf_plus_geracl_remaining_seed53_4096/latest.pt"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
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


def mixed_dataset(
    *,
    pools: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    actual_counts: dict[str, int] = {}
    for offset, (name, count) in enumerate(counts.items()):
        records = take(shuffled(pools[name], seed=seed + offset), count, name=name)
        selected.extend(records)
        actual_counts[name] = len(records)
    return shuffled(selected, seed=seed + 100), actual_counts


def base_config(
    *,
    name: str,
    data_path: Path,
    output_dir: Path,
    seed: int,
    max_steps: int,
    learning_rate: float,
    description: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "flash_attention_2",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": BASE_CHECKPOINT,
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "max_length": 4096,
        "batch_size": 4,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "temperature": 0.02,
        "max_steps": max_steps,
        "log_every": 50,
        "save_every": max_steps,
        "seed": seed,
    }


def staged_config(
    *,
    name: str,
    stages: list[dict[str, Any]],
    output_dir: Path,
    seed: int,
    description: str,
) -> dict[str, Any]:
    first_stage = stages[0]
    return {
        "name": name,
        "description": description,
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "flash_attention_2",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": BASE_CHECKPOINT,
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": first_stage["data_path"],
        "output_dir": str(output_dir),
        "max_length": 4096,
        "batch_size": 4,
        "learning_rate": float(first_stage["learning_rate"]),
        "weight_decay": 0.01,
        "temperature": 0.02,
        "max_steps": sum(int(stage["max_steps"]) for stage in stages),
        "log_every": 50,
        "save_every": int(first_stage["max_steps"]),
        "seed": seed,
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mix O five-task continuation probes.")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--geracl-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_geracl.jsonl"))
    parser.add_argument("--habr-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_habr_qa_sbs_harder_sim021_len.jsonl"))
    parser.add_argument("--deepvk-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl"))
    parser.add_argument("--grounded-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_grounded_rag_v2_q180_doc1200_neg2.jsonl"))
    parser.add_argument("--mvrcii-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_sensitive_topic_mvrcii_3200.jsonl"))
    parser.add_argument("--uc-berkeley-path", type=Path, default=Path("data/contrastive/open_ru_1r_nc_sensitive_topic_uc_berkeley_3200.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/contrastive"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--summary-out", type=Path, default=Path("data/contrastive/open_ru_1r_nc_mixo_five_task_probes_summary.json"))
    args = parser.parse_args()

    pools = {
        "geracl": read_jsonl(args.geracl_path),
        "habr": read_jsonl(args.habr_path),
        "deepvk": read_jsonl(args.deepvk_path),
        "grounded": read_jsonl(args.grounded_path),
        "mvrcii": read_jsonl(args.mvrcii_path),
        "uc_berkeley": read_jsonl(args.uc_berkeley_path),
    }

    written: dict[str, Any] = {
        "seed": args.seed,
        "base_checkpoint": BASE_CHECKPOINT,
        "source_paths": {
            "geracl": str(args.geracl_path),
            "habr": str(args.habr_path),
            "deepvk": str(args.deepvk_path),
            "grounded": str(args.grounded_path),
            "mvrcii": str(args.mvrcii_path),
            "uc_berkeley": str(args.uc_berkeley_path),
        },
        "probes": {},
    }

    # Pair calibration: GeRaCl:Habr:UC Berkeley:MVR-CII = 8:2:1:1.
    pair_records, pair_counts = mixed_dataset(
        pools=pools,
        counts={"geracl": 2400, "habr": 600, "uc_berkeley": 300, "mvrcii": 300},
        seed=args.seed,
    )
    pair_path = args.data_dir / "open_ru_1r_nc_mixo_pair_calibration_g8_h2_uc1_mvr1_3600.jsonl"
    write_jsonl(pair_path, pair_records)
    for steps in (300, 600, 900):
        name = f"exp01r_nc_mixo_pair_calibration_{steps}_4096"
        config_path = args.config_dir / f"{name}.json"
        output_dir = Path("experiments/exp01_reinit_fair/checkpoints") / name.removeprefix("exp01r_nc_")
        config = base_config(
            name=name,
            data_path=pair_path,
            output_dir=output_dir,
            seed=args.seed + steps,
            max_steps=steps,
            learning_rate=1e-6,
            description=(
                "Mix O continuation probe from Mix F seed53. Pair calibration data uses "
                "GeRaCl:Habr:UC Berkeley:MVR-CII = 8:2:1:1."
            ),
        )
        write_json(config_path, config)
        written["probes"][name] = {
            "family": "pair_calibration",
            "config": str(config_path),
            "data": str(pair_path),
            "counts": pair_counts,
            "steps": steps,
            "learning_rate": 1e-6,
        }

    # OECD/taxonomy repair: GeRaCl:DeepVK:GroundedStrict = 6:2:1.
    taxonomy_records, taxonomy_counts = mixed_dataset(
        pools=pools,
        counts={"geracl": 2400, "deepvk": 800, "grounded": 400},
        seed=args.seed + 20,
    )
    taxonomy_path = args.data_dir / "open_ru_1r_nc_mixo_taxonomy_repair_g6_deepvk2_grounded1_3600.jsonl"
    write_jsonl(taxonomy_path, taxonomy_records)
    for steps in (300, 600):
        name = f"exp01r_nc_mixo_taxonomy_repair_{steps}_4096"
        config_path = args.config_dir / f"{name}.json"
        output_dir = Path("experiments/exp01_reinit_fair/checkpoints") / name.removeprefix("exp01r_nc_")
        config = base_config(
            name=name,
            data_path=taxonomy_path,
            output_dir=output_dir,
            seed=args.seed + 100 + steps,
            max_steps=steps,
            learning_rate=1e-6,
            description=(
                "Mix O continuation probe from Mix F seed53. Taxonomy repair data uses "
                "GeRaCl:DeepVK:GroundedStrict = 6:2:1."
            ),
        )
        write_json(config_path, config)
        written["probes"][name] = {
            "family": "taxonomy_repair",
            "config": str(config_path),
            "data": str(taxonomy_path),
            "counts": taxonomy_counts,
            "steps": steps,
            "learning_rate": 1e-6,
        }

    # Alternating two-stage probe.
    alt_taxonomy_records, alt_taxonomy_counts = mixed_dataset(
        pools=pools,
        counts={"geracl": 1200, "deepvk": 400, "grounded": 200},
        seed=args.seed + 40,
    )
    alt_pair_records, alt_pair_counts = mixed_dataset(
        pools=pools,
        counts={"geracl": 2800, "habr": 560, "uc_berkeley": 280, "mvrcii": 280},
        seed=args.seed + 60,
    )
    alt_taxonomy_path = args.data_dir / "open_ru_1r_nc_mixo_alt_taxonomy_g6_deepvk2_grounded1_1800.jsonl"
    alt_pair_path = args.data_dir / "open_ru_1r_nc_mixo_alt_pair_g10_h2_uc1_mvr1_3920.jsonl"
    write_jsonl(alt_taxonomy_path, alt_taxonomy_records)
    write_jsonl(alt_pair_path, alt_pair_records)
    alt_name = "exp01r_nc_mixo_alternating_tax400_pair400_4096"
    alt_config_path = args.config_dir / f"{alt_name}.json"
    alt_output_dir = Path("experiments/exp01_reinit_fair/checkpoints") / alt_name.removeprefix("exp01r_nc_")
    stages = [
        {
            "name": "taxonomy_repair",
            "data_path": str(alt_taxonomy_path),
            "max_steps": 400,
            "batch_size": 4,
            "learning_rate": 1e-6,
            "save_every": 400,
        },
        {
            "name": "pair_calibration",
            "data_path": str(alt_pair_path),
            "max_steps": 400,
            "batch_size": 4,
            "learning_rate": 5e-7,
            "save_every": 400,
        },
    ]
    alt_config = staged_config(
        name=alt_name,
        stages=stages,
        output_dir=alt_output_dir,
        seed=args.seed + 200,
        description=(
            "Mix O continuation probe from Mix F seed53. Stage A repairs taxonomy with "
            "GeRaCl:DeepVK:GroundedStrict = 6:2:1, then Stage B calibrates pair tasks with "
            "GeRaCl:Habr:UC Berkeley:MVR-CII = 10:2:1:1."
        ),
    )
    write_json(alt_config_path, alt_config)
    written["probes"][alt_name] = {
        "family": "alternating",
        "config": str(alt_config_path),
        "data": [str(alt_taxonomy_path), str(alt_pair_path)],
        "counts": {"taxonomy": alt_taxonomy_counts, "pair": alt_pair_counts},
        "steps": [400, 400],
        "learning_rate": [1e-6, 5e-7],
    }

    write_json(args.summary_out, written)
    print(f"Wrote {args.summary_out}")
    for name, spec in written["probes"].items():
        print(f"{name}: {spec['config']}")


if __name__ == "__main__":
    main()
