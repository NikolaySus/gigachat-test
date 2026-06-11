from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
CHECKPOINT_ROOT = ROOT / "experiments" / "exp01_reinit_fair" / "checkpoints"
RESULTS_ROOT = ROOT / "results" / "official_repro"
SCREEN_JSON = RESULTS_ROOT / "loss_gap_screen" / "loss_gap_screen.json"

CLEAN_BASE = (
    "experiments/exp01_reinit_fair/checkpoints/"
    "open_ru_1r_nc_mixh_habrfull_plus_geracl_remaining_4096_eager_frozenrepro/latest.pt"
)
REHEARSAL_DATA = "data/contrastive/open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl"

GATE5_TASKS = [
    "CEDRClassification",
    "GeoreviewClassification",
    "RuSciBenchOECDClassification",
    "RuSTSBenchmarkSTS",
    "GeoreviewClusteringP2P",
]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def slug(value: str) -> str:
    value = value.removesuffix(".jsonl")
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        else:
            allowed.append("_")
    return "_".join("".join(allowed).split("_"))[:90]


def load_candidates(path: Path, *, top_k: int, min_batches: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["rows"]
        if row["loss_gap"] > 0.0
        and int(row.get("valid_batches", 0)) >= min_batches
        and row["loss"] == "contrastive"
    ]
    rows.sort(key=lambda row: row["loss_gap"], reverse=True)
    return rows[:top_k]


def write_config(
    candidate: dict[str, Any],
    *,
    index: int,
    steps: int,
    seed: int,
    batch_size: int,
) -> tuple[str, Path]:
    run_name = f"lossgap_top{index:02d}_{slug(candidate['dataset'])}_4096_eager_frozenrepro"
    config = {
        "name": run_name,
        "description": (
            "Loss-gap candidate ablation. Released latent weights were used only for offline "
            "screening, not as labels or training targets."
        ),
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": CLEAN_BASE,
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": f"data/contrastive/{candidate['dataset']}",
        "output_dir": str((CHECKPOINT_ROOT / run_name).relative_to(ROOT)),
        "max_length": 4096,
        "batch_size": batch_size,
        "learning_rate": 2e-6,
        "weight_decay": 0.001,
        "temperature": float(candidate["temperature"]),
        "max_steps": steps,
        "log_every": 50,
        "save_every": steps,
        "seed": seed + index,
        "retention": {
            "rehearsal_data_path": REHEARSAL_DATA,
            "rehearsal_batch_size": 3,
            "rehearsal_loss_weight": 0.0,
        },
        "stages": [
            {
                "name": "lossgap_candidate",
                "data_path": f"data/contrastive/{candidate['dataset']}",
                "max_steps": steps,
                "batch_size": batch_size,
                "learning_rate": 2e-6,
                "weight_decay": 0.001,
                "temperature": float(candidate["temperature"]),
                "pair_score_loss_weight": 1.0,
                "parameter_anchor_weight": 0.5,
                "rehearsal_loss_weight": 0.25,
                "save_every": steps,
                "shuffle": True,
            }
        ],
        "screening": {
            "dataset": candidate["dataset"],
            "loss": candidate["loss"],
            "temperature": candidate["temperature"],
            "our_loss": candidate["our_loss"],
            "released_loss": candidate["released_loss"],
            "loss_gap": candidate["loss_gap"],
            "valid_batches": candidate["valid_batches"],
        },
    }
    path = CONFIG_DIR / f"{run_name}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_name, path


def score_table(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Local avg:"):
            scores["avg"] = float(line.split(":", 1)[1].strip())
        if not line.startswith("| ") or line.startswith("| Task ") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 4:
            scores[cells[0]] = float(cells[1])
            scores[f"{cells[0]}.delta"] = float(cells[3])
    return scores


def eval_checkpoint(checkpoint: Path, *, run_name: str) -> tuple[Path, dict[str, float]]:
    result_dir = RESULTS_ROOT / f"{run_name}_gate5"
    comparison_path = RESULTS_ROOT / f"{run_name}_gate5_comparison.md"
    if not comparison_path.exists():
        run(
            [
                "official_repro/.venv/bin/python",
                "official_repro/run_official_rumteb.py",
                "--output-folder",
                str(result_dir.relative_to(ROOT)),
                "--tasks",
                *GATE5_TASKS,
                "--prompt-mode",
                "legacy_ru",
                "--batch-size",
                "8",
                "--seed",
                "8",
                "--attn-implementation",
                "eager",
                "--latent-checkpoint",
                str(checkpoint.relative_to(ROOT)),
                "--overwrite-results",
                "--reset-seed-per-task",
            ]
        )
        run(
            [
                "official_repro/.venv/bin/python",
                "official_repro/compare_official_rumteb.py",
                str(result_dir.relative_to(ROOT)),
                "--write-md",
                str(comparison_path.relative_to(ROOT)),
            ]
        )
    return comparison_path, score_table(comparison_path)


def cleanup_output_dir(output_dir: Path, keep: Path) -> None:
    for path in output_dir.glob("*.pt"):
        if path != keep:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-json", default=str(SCREEN_JSON.relative_to(ROOT)))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-batches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1831)
    args = parser.parse_args()

    candidates = load_candidates(ROOT / args.screen_json, top_k=args.top_k, min_batches=args.min_batches)
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        run_name, config_path = write_config(
            candidate,
            index=index,
            steps=args.steps,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        output_dir = CHECKPOINT_ROOT / run_name
        checkpoint = output_dir / f"step-{args.steps}.pt"
        if not checkpoint.exists():
            if output_dir.exists() and not any(output_dir.glob("*.pt")):
                shutil.rmtree(output_dir)
            run(["uv", "run", "python", "scripts/train_exp01b_latent_memory.py", "--config", str(config_path)])
        comparison_path, scores = eval_checkpoint(checkpoint, run_name=run_name)
        cleanup_output_dir(output_dir, checkpoint)
        rows.append(
            {
                "rank": index,
                "run_name": run_name,
                "dataset": candidate["dataset"],
                "loss": candidate["loss"],
                "temperature": candidate["temperature"],
                "gap": candidate["loss_gap"],
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "comparison": str(comparison_path.relative_to(ROOT)),
                **scores,
            }
        )

    summary = RESULTS_ROOT / "loss_gap_ablation_gate5_summary.md"
    lines = [
        "# Loss-Gap Candidate 5-Task Ablations",
        "",
        f"Steps per candidate: {args.steps}",
        "Start checkpoint: clean fair no-manip base.",
        "Released latent weights were used only for offline loss-gap screening.",
        "",
        "| Rank | Dataset | Temp | Gap | Avg | CEDR | GeoCls | OECDCls | RuSTS | GeoCluster | Comparison |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | `{row['dataset']}` | {row['temperature']:.2f} | {row['gap']:+.6f} | "
            f"{row.get('avg', float('nan')):.6f} | {row.get('CEDRClassification', float('nan')):.6f} | "
            f"{row.get('GeoreviewClassification', float('nan')):.6f} | "
            f"{row.get('RuSciBenchOECDClassification', float('nan')):.6f} | "
            f"{row.get('RuSTSBenchmarkSTS', float('nan')):.6f} | "
            f"{row.get('GeoreviewClusteringP2P', float('nan')):.6f} | `{row['comparison']}` |"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.relative_to(ROOT))


if __name__ == "__main__":
    main()
