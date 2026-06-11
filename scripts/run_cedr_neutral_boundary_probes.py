from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
CHECKPOINT_ROOT = ROOT / "experiments" / "exp01_reinit_fair" / "checkpoints"
RESULTS_ROOT = ROOT / "results" / "official_repro"

CLEAN_BASE = (
    "experiments/exp01_reinit_fair/checkpoints/"
    "open_ru_1r_nc_mixh_habrfull_plus_geracl_remaining_4096_eager_frozenrepro/latest.pt"
)
GO9000 = "data/contrastive/open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl"
LENTA_REPORTED = "data/contrastive/open_ru_1r_nc_cedr_lenta_news_neutral_distractors_reported_3200.jsonl"
NEUTRAL_MIX = "data/contrastive/open_ru_1r_nc_cedr_neutral_boundary_reported_mix_7000.jsonl"
HABR_DEEPVK = "data/contrastive/cedr_a050_recovery_habr1x_deepvk1x_11977.jsonl"
RUSTS_32K = "data/contrastive/rusts_external_cointegrated_diverse_32000.jsonl"

CEDR_TASK = ["CEDRClassification"]
GATE5_TASKS = [
    "CEDRClassification",
    "GeoreviewClassification",
    "RuSciBenchOECDClassification",
    "RuSTSBenchmarkSTS",
    "GeoreviewClusteringP2P",
]

OFFICIAL = {
    "avg": 0.657981,
    "CEDRClassification": 0.685069,
    "GeoreviewClassification": 0.547510,
    "RuSciBenchOECDClassification": 0.545068,
    "RuSTSBenchmarkSTS": 0.835900,
    "GeoreviewClusteringP2P": 0.676357,
}

BASELINE_CEDR = 0.641817


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    stages: list[dict[str, Any]]


def stage(
    name: str,
    data_path: str,
    *,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    save_every: int,
    anchor: float = 10.0,
    rehearsal: float = 1.0,
    temperature: float = 0.02,
    weight_decay: float = 0.01,
    pair_score_weight: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "data_path": data_path,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "parameter_anchor_weight": anchor,
        "rehearsal_loss_weight": rehearsal,
        "save_every": save_every,
    }
    if pair_score_weight is not None:
        row["pair_score_loss_weight"] = pair_score_weight
    return row


VARIANTS = {
    "neutral_reported_lr5e8": Variant(
        name="neutral_reported_lr5e8",
        description="Only strict reported Lenta neutral-boundary rows, very low LR.",
        stages=[
            stage("reported_neutral", LENTA_REPORTED, max_steps=150, batch_size=3, learning_rate=5e-8, save_every=50),
        ],
    ),
    "neutral_reported_lr1e7": Variant(
        name="neutral_reported_lr1e7",
        description="Only strict reported Lenta neutral-boundary rows, low LR.",
        stages=[
            stage("reported_neutral", LENTA_REPORTED, max_steps=150, batch_size=3, learning_rate=1e-7, save_every=50),
        ],
    ),
    "neutral_mix_300": Variant(
        name="neutral_mix_300",
        description="3:2:1:1 neutral-boundary mix, CEDR-only checkpoints at 100/200/300.",
        stages=[
            stage("neutral_boundary_mix", NEUTRAL_MIX, max_steps=300, batch_size=3, learning_rate=1e-7, save_every=100),
        ],
    ),
    "neutral_mix_repair_habrdeepvk400": Variant(
        name="neutral_mix_repair_habrdeepvk400",
        description="Neutral-boundary mix followed by Habr+DeepVK recovery.",
        stages=[
            stage("neutral_boundary_mix", NEUTRAL_MIX, max_steps=200, batch_size=3, learning_rate=1e-7, save_every=100),
            stage(
                "habr_deepvk_repair",
                HABR_DEEPVK,
                max_steps=400,
                batch_size=2,
                learning_rate=2e-6,
                save_every=200,
                anchor=1.0,
                rehearsal=0.5,
                weight_decay=0.001,
            ),
        ],
    ),
    "neutral_mix_repair_sts400": Variant(
        name="neutral_mix_repair_sts400",
        description="Neutral-boundary mix followed by short RuSTS repair.",
        stages=[
            stage("neutral_boundary_mix", NEUTRAL_MIX, max_steps=200, batch_size=3, learning_rate=1e-7, save_every=100),
            stage(
                "rusts_repair",
                RUSTS_32K,
                max_steps=400,
                batch_size=4,
                learning_rate=5e-7,
                save_every=200,
                anchor=5.0,
                rehearsal=0.75,
                weight_decay=0.001,
                pair_score_weight=5.0,
            ),
        ],
    ),
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_neutral_mix() -> None:
    if not (ROOT / NEUTRAL_MIX).exists():
        run(["uv", "run", "python", "scripts/prepare_cedr_neutral_boundary_mix.py"])


def write_config(variant: Variant, *, seed: int) -> Path:
    run_name = f"nomanip_{variant.name}_4096_eager_frozenrepro"
    config = {
        "name": run_name,
        "description": f"No direct weight manipulation. {variant.description}",
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": CLEAN_BASE,
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": variant.stages[0]["data_path"],
        "output_dir": str((CHECKPOINT_ROOT / run_name).relative_to(ROOT)),
        "max_length": 4096,
        "batch_size": variant.stages[0]["batch_size"],
        "learning_rate": variant.stages[0]["learning_rate"],
        "weight_decay": variant.stages[0].get("weight_decay", 0.01),
        "temperature": 0.02,
        "max_steps": sum(int(item["max_steps"]) for item in variant.stages),
        "log_every": 50,
        "save_every": 10_000_000,
        "seed": seed,
        "retention": {
            "parameter_anchor_weight": 0.0,
            "rehearsal_data_path": GO9000,
            "rehearsal_batch_size": 3,
            "rehearsal_loss_weight": 0.0,
        },
        "stages": variant.stages,
    }
    path = CONFIG_DIR / f"{run_name}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def checkpoint_steps(output_dir: Path) -> list[Path]:
    checkpoints = []
    for path in output_dir.glob("step-*.pt"):
        try:
            int(path.stem.split("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append(path)
    return sorted(checkpoints, key=lambda item: int(item.stem.split("-", 1)[1]))


def score_table(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Local avg:"):
            scores["avg"] = float(line.split(":", 1)[1].strip())
        if not line.startswith("| ") or line.startswith("| Task ") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4:
            continue
        task, local, _official, delta = cells
        scores[task] = float(local)
        scores[f"{task}.delta"] = float(delta)
    return scores


def eval_checkpoint(checkpoint: Path, *, run_name: str, tag: str, tasks: list[str]) -> tuple[Path, dict[str, float]]:
    task_tag = "cedr" if tasks == CEDR_TASK else "gate5"
    result_dir = RESULTS_ROOT / f"{run_name}_{tag}_{task_tag}"
    comparison_path = RESULTS_ROOT / f"{run_name}_{tag}_{task_tag}_comparison.md"
    run(
        [
            "official_repro/.venv/bin/python",
            "official_repro/run_official_rumteb.py",
            "--output-folder",
            str(result_dir.relative_to(ROOT)),
            "--tasks",
            *tasks,
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


def gate_promoted(row: dict[str, Any], *, cedr_threshold: float) -> bool:
    return float(row["CEDRClassification"]) >= cedr_threshold


def cleanup_checkpoints(output_dir: Path, keep: set[Path]) -> None:
    for path in output_dir.glob("*.pt"):
        if path not in keep:
            path.unlink()
    latest = output_dir / "latest.pt"
    if latest.exists() and latest not in keep:
        latest.unlink()


def write_summary(rows: list[dict[str, Any]]) -> Path:
    summary_path = RESULTS_ROOT / "cedr_neutral_boundary_probe_summary.md"
    lines = [
        "# CEDR Neutral-Boundary Probe Summary",
        "",
        f"Clean base: `{CLEAN_BASE}`.",
        "Training uses no direct weight arithmetic. CEDR-only scoring is used before 5-task promotion.",
        "",
        "| Variant | Checkpoint | CEDR | Gate avg | GeoCls | OECDCls | RuSTS | GeoCluster | Comparison |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: (item.get("gate_avg", -1.0), item["CEDRClassification"]), reverse=True):
        lines.append(
            f"| `{row['variant']}` | `{row['checkpoint']}` | "
            f"{row['CEDRClassification']:.6f} | "
            f"{row.get('gate_avg', float('nan')):.6f} | "
            f"{row.get('GeoreviewClassification', float('nan')):.6f} | "
            f"{row.get('RuSciBenchOECDClassification', float('nan')):.6f} | "
            f"{row.get('RuSTSBenchmarkSTS', float('nan')):.6f} | "
            f"{row.get('GeoreviewClusteringP2P', float('nan')):.6f} | "
            f"`{row['comparison']}` |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=["neutral_reported_lr5e8", "neutral_reported_lr1e7", "neutral_mix_300"])
    parser.add_argument("--seed", type=int, default=3119)
    parser.add_argument("--cedr-promote-threshold", type=float, default=BASELINE_CEDR + 0.005)
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    args = parser.parse_args()

    ensure_neutral_mix()
    rows: list[dict[str, Any]] = []
    for index, variant_name in enumerate(args.variants):
        if variant_name not in VARIANTS:
            raise SystemExit(f"Unknown variant {variant_name}. Available: {', '.join(VARIANTS)}")
        variant = VARIANTS[variant_name]
        config_path = write_config(variant, seed=args.seed + index)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_name = config["name"]
        output_dir = CHECKPOINT_ROOT / run_name

        if not checkpoint_steps(output_dir):
            run(["uv", "run", "python", "scripts/train_exp01b_latent_memory.py", "--config", str(config_path)])

        variant_rows = []
        for checkpoint in checkpoint_steps(output_dir):
            tag = checkpoint.stem.replace("-", "")
            comparison_path, scores = eval_checkpoint(checkpoint, run_name=run_name, tag=tag, tasks=CEDR_TASK)
            row: dict[str, Any] = {
                "variant": variant_name,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "comparison": str(comparison_path.relative_to(ROOT)),
                **scores,
            }
            if gate_promoted(row, cedr_threshold=args.cedr_promote_threshold):
                gate_comparison, gate_scores = eval_checkpoint(checkpoint, run_name=run_name, tag=tag, tasks=GATE5_TASKS)
                row.update(gate_scores)
                row["gate_avg"] = gate_scores["avg"]
                row["comparison"] = str(gate_comparison.relative_to(ROOT))
            variant_rows.append(row)
            rows.append(row)

        if not args.keep_all_checkpoints:
            promoted = [row for row in variant_rows if "gate_avg" in row]
            best = max(promoted or variant_rows, key=lambda item: (item.get("gate_avg", -1.0), item["CEDRClassification"]))
            cleanup_checkpoints(output_dir, {ROOT / best["checkpoint"]})

    summary_path = write_summary(rows)
    print(summary_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
