from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
CHECKPOINT_ROOT = ROOT / "experiments" / "exp01_reinit_fair" / "checkpoints"
RESULTS_ROOT = ROOT / "results" / "official_repro"

BASE_CHECKPOINT = (
    "experiments/exp01_reinit_fair/checkpoints/"
    "cedr_a050_recover_habr1x_deepvk1x_lr2e6_anchor1_rehgo05_5989_4096_eager_frozenrepro/"
    "step-5989.pt"
)
REHEARSAL_DATA = "data/contrastive/open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl"

GATE5_TASKS = [
    "CEDRClassification",
    "GeoreviewClassification",
    "RuSciBenchOECDClassification",
    "RuSTSBenchmarkSTS",
    "GeoreviewClusteringP2P",
]

DATASETS = {
    "teacher_only": "data/contrastive/rusts_external_teacher_scored_clean_12000.jsonl",
    "mixed_recovery_teacher": "data/contrastive/cedr_a050_recovery_habr1x_deepvk1x_plus_teacher_rusts_23977.jsonl",
}

BASE_SCORES = {
    "avg": 0.641726,
    "CEDRClassification": 0.688151,
    "GeoreviewClassification": 0.544727,
    "RuSciBenchOECDClassification": 0.527881,
    "RuSTSBenchmarkSTS": 0.796255,
    "GeoreviewClusteringP2P": 0.651616,
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def compact_float(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e").replace("e+", "e")


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


def write_config(args: argparse.Namespace) -> tuple[str, Path]:
    suffix = (
        f"{args.dataset}_steps{args.steps}_lr{compact_float(args.learning_rate)}_"
        f"anchor{args.anchor_weight:g}_reh{args.rehearsal_loss_weight:g}_pairw{args.pair_score_loss_weight:g}"
    ).replace(".", "p")
    name = f"cedr_a050_habr1x_deepvk1x_teacher_rusts_{suffix}_4096_eager_frozenrepro"
    config = {
        "name": name,
        "description": "Teacher-scored clean RuSTS-style ablation from Habr 1x + DeepVK 1x CEDR-recovery checkpoint.",
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": BASE_CHECKPOINT,
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": DATASETS[args.dataset],
        "output_dir": str((CHECKPOINT_ROOT / name).relative_to(ROOT)),
        "max_length": 4096,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "temperature": args.temperature,
        "max_steps": args.steps,
        "pair_score_loss_weight": args.pair_score_loss_weight,
        "log_every": 50,
        "save_every": args.steps,
        "seed": args.seed,
        "retention": {
            "parameter_anchor_weight": args.anchor_weight,
            "rehearsal_data_path": REHEARSAL_DATA,
            "rehearsal_batch_size": args.batch_size,
            "rehearsal_loss_weight": args.rehearsal_loss_weight,
        },
    }
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--anchor-weight", type=float, default=2.0)
    parser.add_argument("--rehearsal-loss-weight", type=float, default=0.5)
    parser.add_argument("--pair-score-loss-weight", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=5501)
    parser.add_argument("--keep-checkpoint", action="store_true")
    args = parser.parse_args()

    name, config_path = write_config(args)
    checkpoint = CHECKPOINT_ROOT / name / f"step-{args.steps}.pt"
    latest = CHECKPOINT_ROOT / name / "latest.pt"
    result_dir = RESULTS_ROOT / f"{name}_gate5"
    comparison_path = RESULTS_ROOT / f"{name}_gate5_comparison.md"

    if not checkpoint.exists():
        run(["uv", "run", "python", "scripts/train_exp01b_latent_memory.py", "--config", str(config_path)])
    if latest.exists():
        latest.unlink()
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
    scores = score_table(comparison_path)
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    if not args.keep_checkpoint and checkpoint.exists():
        checkpoint.unlink()


if __name__ == "__main__":
    main()
