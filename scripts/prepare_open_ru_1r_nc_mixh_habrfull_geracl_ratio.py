from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "contrastive"
CONFIG_DIR = ROOT / "configs" / "experiments"

BASE_DATA = DATA_DIR / "open_ru_1r_nc_mixh_habrfull_geracl6400_habr4369_deepvk3200_grandmaster3200_17169.jsonl"
GERACL_REMAINING = DATA_DIR / "open_ru_1r_nc_mixb_geracl_remaining_9600.jsonl"
BASE_CONFIG = CONFIG_DIR / "exp01r_nc_mixh_habrfull_4096_eager_frozenrepro.json"

GERACL_SOURCE = "deepvk/GeRaCl_synthethic_dataset"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_config(
    base_config: dict[str, Any],
    *,
    name: str,
    description: str,
    data_path: Path,
    rows: int,
    batch_size: int = 2,
) -> dict[str, Any]:
    max_steps = math.ceil(rows / batch_size)
    config = dict(base_config)
    config.update(
        {
            "name": f"exp01r_nc_{name}_4096_eager_frozenrepro",
            "description": description,
            "data_path": str(data_path.relative_to(ROOT)),
            "output_dir": f"experiments/exp01_reinit_fair/checkpoints/{name}_4096_eager_frozenrepro",
            "batch_size": batch_size,
            "max_steps": max_steps,
            "save_every": max_steps,
        }
    )
    return config


def main() -> None:
    base_rows = read_jsonl(BASE_DATA)
    remaining_geracl = read_jsonl(GERACL_REMAINING)
    base_config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    seed = int(base_config.get("seed", 71))
    rng = random.Random(seed + 2500)

    geracl_base = [row for row in base_rows if row.get("source") == GERACL_SOURCE]
    non_geracl = [row for row in base_rows if row.get("source") != GERACL_SOURCE]
    all_geracl = geracl_base + remaining_geracl
    rng.shuffle(all_geracl)

    variants: list[tuple[str, list[dict[str, Any]], str]] = []
    variants.append(
        (
            "mixh_habrfull_b2_control",
            list(base_rows),
            "Batch-size control for Mix H HabrFull: same rows as baseline, batch 2, one pass.",
        )
    )
    variants.append(
        (
            "mixh_habrfull_geracl1600",
            non_geracl + all_geracl[:1600],
            "Reduced-GeRaCl Mix H HabrFull stage-1 ablation: GeRaCl 1600 + fixed Habr/DeepVK/Grandmaster.",
        )
    )
    variants.append(
        (
            "mixh_habrfull_geracl3200",
            non_geracl + all_geracl[:3200],
            "Reduced-GeRaCl Mix H HabrFull stage-1 ablation: GeRaCl 3200 + fixed Habr/DeepVK/Grandmaster.",
        )
    )
    variants.append(
        (
            "mixh_habrfull_allgeracl_stage1",
            non_geracl + all_geracl,
            "All-GeRaCl-in-stage-1 Mix H HabrFull ablation: GeRaCl 16000 mixed with Habr/DeepVK/Grandmaster, no stage 2.",
        )
    )

    for offset, (name, rows, description) in enumerate(variants):
        shuffled = list(rows)
        random.Random(seed + 2600 + offset).shuffle(shuffled)
        counts = Counter(row.get("source", "") for row in shuffled)
        data_path = DATA_DIR / f"open_ru_1r_nc_{name}_{len(shuffled)}.jsonl"
        write_jsonl(data_path, shuffled)
        write_json(
            data_path.with_name(data_path.stem + "_summary.json"),
            {
                "output": str(data_path.relative_to(ROOT)),
                "base_data": str(BASE_DATA.relative_to(ROOT)),
                "geracl_remaining": str(GERACL_REMAINING.relative_to(ROOT)),
                "counts": dict(counts),
                "total_records": len(shuffled),
                "batch_size": 2,
                "max_steps_1x": math.ceil(len(shuffled) / 2),
                "seed": seed,
                "description": description,
            },
        )
        config = build_config(
            base_config,
            name=name,
            description=description,
            data_path=data_path,
            rows=len(shuffled),
            batch_size=2,
        )
        write_json(CONFIG_DIR / f"exp01r_nc_{name}_4096_eager_frozenrepro.json", config)


if __name__ == "__main__":
    main()
