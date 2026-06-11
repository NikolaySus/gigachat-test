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
REPORT = RESULTS_ROOT / "cedr_diagnostic_contaminated_training_report.md"

DATA_PATH = "data/contrastive/CONTAMINATED_cedr_diagnostic_a050_fixed_neutral.jsonl"
ALL_LABELED = "data/contrastive/CONTAMINATED_cedr_all_labeled_supcon.jsonl"
GO9000 = "data/contrastive/open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl"
BASE = "experiments/exp01_reinit_fair/checkpoints/open_ru_1r_nc_mixh_habrfull_plus_geracl_remaining_4096_eager_frozenrepro/latest.pt"
RETAINED = "experiments/exp01_reinit_fair/checkpoints/cedr_correction_lenta_reported_replay_go9000_from_retainedbest_lr1e7_anchor10_reh1_300_4096_eager_frozenrepro/step-150.pt"
BAD = "experiments/exp01_reinit_fair/checkpoints/mixh_habrfull_leave1out_no_geracl_4096_eager_frozenrepro/latest.pt"
A050 = "experiments/exp01_reinit_fair/checkpoints/cedr_goal_best_minus_nogeracl_a050/latent.pt"

CEDR_TASK = ["CEDRClassification"]
RELEASED_CEDR = 0.685069


@dataclass(frozen=True)
class Variant:
    name: str
    initial_checkpoint: str
    max_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    temperature: float
    anchor: float
    rehearsal: float
    negative_direction: float = 0.0
    direction_checkpoint: str | None = None
    data_path: str = DATA_PATH
    task_loss_weight: float = 1.0
    teacher_checkpoint: str | None = None
    distillation: float = 0.0
    distillation_max_texts: int = 8
    save_every: int = 50


VARIANTS = {
    "diag_base_lr1e6": Variant(
        name="diag_base_lr1e6",
        initial_checkpoint=BASE,
        max_steps=300,
        batch_size=1,
        learning_rate=1e-6,
        weight_decay=0.0,
        temperature=0.03,
        anchor=1.0,
        rehearsal=0.25,
    ),
    "diag_retained_lr2e7": Variant(
        name="diag_retained_lr2e7",
        initial_checkpoint=RETAINED,
        max_steps=300,
        batch_size=1,
        learning_rate=2e-7,
        weight_decay=0.0,
        temperature=0.03,
        anchor=5.0,
        rehearsal=0.5,
    ),
    "diag_retained_lr5e7": Variant(
        name="diag_retained_lr5e7",
        initial_checkpoint=RETAINED,
        max_steps=300,
        batch_size=1,
        learning_rate=5e-7,
        weight_decay=0.0,
        temperature=0.04,
        anchor=2.0,
        rehearsal=0.25,
    ),
    "diag_retained_antibad": Variant(
        name="diag_retained_antibad",
        initial_checkpoint=RETAINED,
        max_steps=300,
        batch_size=1,
        learning_rate=2e-7,
        weight_decay=0.0,
        temperature=0.03,
        anchor=5.0,
        rehearsal=0.5,
        negative_direction=0.05,
    ),
    "distill_a050_allcedr": Variant(
        name="distill_a050_allcedr",
        initial_checkpoint=RETAINED,
        data_path=ALL_LABELED,
        max_steps=300,
        batch_size=4,
        learning_rate=5e-7,
        weight_decay=0.0,
        temperature=0.05,
        anchor=1.0,
        rehearsal=0.0,
        task_loss_weight=0.0,
        teacher_checkpoint=A050,
        distillation=1.0,
        distillation_max_texts=4,
    ),
    "direction_a050_retained_w005": Variant(
        name="direction_a050_retained_w005",
        initial_checkpoint=RETAINED,
        data_path=DATA_PATH,
        max_steps=120,
        batch_size=1,
        learning_rate=5e-7,
        weight_decay=0.0,
        temperature=0.04,
        anchor=0.0,
        rehearsal=0.0,
        negative_direction=-0.05,
        direction_checkpoint=A050,
        teacher_checkpoint=RETAINED,
        task_loss_weight=0.0,
        save_every=10,
    ),
    "direction_a050_retained_w02": Variant(
        name="direction_a050_retained_w02",
        initial_checkpoint=RETAINED,
        data_path=DATA_PATH,
        max_steps=120,
        batch_size=1,
        learning_rate=5e-7,
        weight_decay=0.0,
        temperature=0.04,
        anchor=0.0,
        rehearsal=0.0,
        negative_direction=-0.2,
        direction_checkpoint=A050,
        teacher_checkpoint=RETAINED,
        task_loss_weight=0.0,
        save_every=10,
    ),
    "direction_a050_retained_w02_anchor01": Variant(
        name="direction_a050_retained_w02_anchor01",
        initial_checkpoint=RETAINED,
        data_path=DATA_PATH,
        max_steps=120,
        batch_size=1,
        learning_rate=5e-7,
        weight_decay=0.0,
        temperature=0.04,
        anchor=0.1,
        rehearsal=0.0,
        negative_direction=-0.2,
        direction_checkpoint=A050,
        teacher_checkpoint=RETAINED,
        task_loss_weight=0.0,
        save_every=10,
    ),
    "anchor_a050_retained_lr5e6": Variant(
        name="anchor_a050_retained_lr5e6",
        initial_checkpoint=RETAINED,
        data_path=DATA_PATH,
        max_steps=80,
        batch_size=1,
        learning_rate=5e-6,
        weight_decay=0.0,
        temperature=0.04,
        anchor=1.0,
        rehearsal=0.0,
        teacher_checkpoint=A050,
        task_loss_weight=0.0,
        save_every=20,
    ),
    "anchor_a050_retained_lr2e5": Variant(
        name="anchor_a050_retained_lr2e5",
        initial_checkpoint=RETAINED,
        data_path=DATA_PATH,
        max_steps=80,
        batch_size=1,
        learning_rate=2e-5,
        weight_decay=0.0,
        temperature=0.04,
        anchor=1.0,
        rehearsal=0.0,
        teacher_checkpoint=A050,
        task_loss_weight=0.0,
        save_every=20,
    ),
    "anchor_a050_scaled_lr2e5": Variant(
        name="anchor_a050_scaled_lr2e5",
        initial_checkpoint=RETAINED,
        data_path=DATA_PATH,
        max_steps=80,
        batch_size=1,
        learning_rate=2e-5,
        weight_decay=0.0,
        temperature=0.04,
        anchor=1_000_000.0,
        rehearsal=0.0,
        teacher_checkpoint=A050,
        task_loss_weight=0.0,
        save_every=20,
    ),
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_data() -> None:
    if not (ROOT / DATA_PATH).exists():
        run(["uv", "run", "python", "scripts/prepare_cedr_diagnostic_contaminated_dataset.py"])
    if not (ROOT / ALL_LABELED).exists():
        run(["uv", "run", "python", "scripts/prepare_cedr_contaminated_ablation.py"])


def write_config(variant: Variant, *, seed: int) -> tuple[str, Path]:
    run_name = f"CONTAMINATED_cedr_diagnostic_{variant.name}_4096_eager_frozenrepro"
    retention: dict[str, Any] = {
        "parameter_anchor_weight": 0.0,
        "rehearsal_data_path": GO9000,
        "rehearsal_batch_size": 3,
        "rehearsal_loss_weight": 0.0,
    }
    if variant.negative_direction:
        retention["negative_direction_checkpoint"] = variant.direction_checkpoint or BAD
    if variant.teacher_checkpoint:
        retention["teacher_latent_checkpoint"] = variant.teacher_checkpoint
        retention["distillation_max_texts"] = variant.distillation_max_texts
    stage: dict[str, Any] = {
        "name": variant.name,
        "data_path": variant.data_path,
        "max_steps": variant.max_steps,
        "batch_size": variant.batch_size,
        "learning_rate": variant.learning_rate,
        "weight_decay": variant.weight_decay,
        "temperature": variant.temperature,
        "parameter_anchor_weight": variant.anchor,
        "rehearsal_loss_weight": variant.rehearsal,
        "negative_direction_weight": variant.negative_direction,
        "task_loss_weight": variant.task_loss_weight,
        "distillation_loss_weight": variant.distillation,
        "save_every": variant.save_every,
    }
    config = {
        "name": run_name,
        "description": "INTENTIONALLY CONTAMINATED CEDR diagnostic training from row-level flip set. Do not report as fair result.",
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": variant.initial_checkpoint,
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": variant.data_path,
        "output_dir": str((CHECKPOINT_ROOT / run_name).relative_to(ROOT)),
        "max_length": 4096,
        "batch_size": variant.batch_size,
        "learning_rate": variant.learning_rate,
        "weight_decay": variant.weight_decay,
        "temperature": variant.temperature,
        "max_steps": variant.max_steps,
        "log_every": 25,
        "save_every": 10_000_000,
        "seed": seed,
        "retention": retention,
        "stages": [stage],
        "contamination": {
            "status": "YES",
            "reason": "Uses CEDR benchmark rows, labels, and diagnostic flip sets.",
            "do_not_use_for_fair_results": True,
        },
    }
    path = CONFIG_DIR / f"{run_name}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_name, path


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
        if not line.startswith("| ") or line.startswith("| Task ") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 4:
            continue
        task, local, _official, delta = cells
        scores[task] = float(local)
        scores[f"{task}.delta"] = float(delta)
    return scores


def eval_checkpoint(checkpoint: Path, *, run_name: str, tag: str) -> tuple[Path, dict[str, float]]:
    result_dir = RESULTS_ROOT / f"{run_name}_{tag}_cedr"
    comparison_path = RESULTS_ROOT / f"{run_name}_{tag}_cedr_comparison.md"
    run(
        [
            "official_repro/.venv/bin/python",
            "official_repro/run_official_rumteb.py",
            "--output-folder",
            str(result_dir.relative_to(ROOT)),
            "--tasks",
            *CEDR_TASK,
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


def cleanup(output_dir: Path, keep: Path) -> None:
    for path in checkpoint_steps(output_dir):
        if path == keep:
            continue
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
    latest = output_dir / "latest.pt"
    if latest.exists() and latest.resolve() != keep.resolve():
        latest.unlink(missing_ok=True)
        (output_dir / "latest.json").unlink(missing_ok=True)


def write_report(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# CEDR Diagnostic Contaminated Training",
        "",
        "This report is intentionally contaminated and is only for diagnosing which CEDR signal can be learned. Do not use these scores as fair results.",
        "",
        f"Released CEDR target: `{RELEASED_CEDR:.6f}`.",
        "",
        "| Variant | Initial checkpoint | Best step | CEDR | Delta vs released | Comparison |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | `{row['initial_checkpoint']}` | {row['step']} | "
            f"{row['CEDRClassification']:.6f} | {row['CEDRClassification'] - RELEASED_CEDR:+.6f} | "
            f"`{row['comparison'].relative_to(ROOT)}` |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run contaminated CEDR diagnostic training variants.")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=9551)
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    args = parser.parse_args()

    ensure_data()
    rows: list[dict[str, Any]] = []
    for key in args.variants:
        variant = VARIANTS[key]
        run_name, config_path = write_config(variant, seed=args.seed)
        output_dir = CHECKPOINT_ROOT / run_name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        run(["uv", "run", "python", "scripts/train_exp01b_latent_memory.py", "--config", str(config_path.relative_to(ROOT))])
        scored = []
        for checkpoint in checkpoint_steps(output_dir):
            comparison, scores = eval_checkpoint(checkpoint, run_name=run_name, tag=checkpoint.stem)
            step = int(checkpoint.stem.split("-", 1)[1])
            scored.append({"checkpoint": checkpoint, "comparison": comparison, "step": step, **scores})
        best = max(scored, key=lambda item: float(item.get("CEDRClassification", -1.0)))
        if not args.keep_all_checkpoints:
            cleanup(output_dir, keep=best["checkpoint"])
        rows.append(
            {
                "variant": key,
                "initial_checkpoint": variant.initial_checkpoint,
                **best,
            }
        )
        write_report(rows)


if __name__ == "__main__":
    main()
