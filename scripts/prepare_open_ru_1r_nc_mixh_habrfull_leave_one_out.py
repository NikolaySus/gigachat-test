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
BASE_CONFIG = CONFIG_DIR / "exp01r_nc_mixh_habrfull_4096_eager_frozenrepro.json"

COMPONENTS = {
    "no_geracl": "deepvk/GeRaCl_synthethic_dataset",
    "no_habr": "Vikhrmodels/habr_qa_sbs:filtered:hard",
    "no_deepvk": "deepvk/ru-HNP",
    "no_grandmaster": "Vikhrmodels/GrandMaster-PRO-MAX:clustered",
}


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


def main() -> None:
    rows = read_jsonl(BASE_DATA)
    base_config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    base_counts = Counter(row.get("source", "") for row in rows)
    seed = int(base_config.get("seed", 71))
    batch_size = int(base_config.get("batch_size", 4))

    for variant, removed_source in COMPONENTS.items():
        kept = [row for row in rows if row.get("source", "") != removed_source]
        random.Random(seed + 1000 + len(variant)).shuffle(kept)
        counts = Counter(row.get("source", "") for row in kept)
        max_steps = math.ceil(len(kept) / batch_size)
        name = f"mixh_habrfull_leave1out_{variant}"
        data_path = DATA_DIR / f"open_ru_1r_nc_{name}_{len(kept)}.jsonl"
        write_jsonl(data_path, kept)
        write_json(
            data_path.with_name(data_path.stem + "_summary.json"),
            {
                "output": str(data_path.relative_to(ROOT)),
                "base_data": str(BASE_DATA.relative_to(ROOT)),
                "base_counts": dict(base_counts),
                "removed_variant": variant,
                "removed_source": removed_source,
                "kept_counts": dict(counts),
                "total_records": len(kept),
                "batch_size": batch_size,
                "max_steps_1x": max_steps,
                "seed": seed,
            },
        )

        config = dict(base_config)
        config.update(
            {
                "name": f"exp01r_nc_{name}_4096_eager_frozenrepro",
                "description": (
                    "Mix H HabrFull stage-1 leave-one-out CEDR ablation. "
                    f"Removed source: {removed_source}"
                ),
                "data_path": str(data_path.relative_to(ROOT)),
                "output_dir": f"experiments/exp01_reinit_fair/checkpoints/{name}_4096_eager_frozenrepro",
                "max_steps": max_steps,
                "save_every": max_steps,
                "seed": seed,
            }
        )
        write_json(CONFIG_DIR / f"exp01r_nc_{name}_4096_eager_frozenrepro.json", config)


if __name__ == "__main__":
    main()
