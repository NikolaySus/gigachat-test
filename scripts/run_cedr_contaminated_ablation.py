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
REPORT = RESULTS_ROOT / "cedr_contaminated_ablation_report.md"

CLEAN_BASE = (
    "experiments/exp01_reinit_fair/checkpoints/"
    "open_ru_1r_nc_mixh_habrfull_plus_geracl_remaining_4096_eager_frozenrepro/latest.pt"
)
GO9000 = "data/contrastive/open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl"

CEDR_TASK = ["CEDRClassification"]
GATE5_TASKS = [
    "CEDRClassification",
    "GeoreviewClassification",
    "RuSciBenchOECDClassification",
    "RuSTSBenchmarkSTS",
    "GeoreviewClusteringP2P",
]

OFFICIAL_CEDR = 0.685069
BASELINE_CEDR = 0.641817


@dataclass(frozen=True)
class Variant:
    name: str
    data_path: str
    description: str
    max_steps: int
    batch_size: int = 2
    learning_rate: float = 1e-7
    weight_decay: float = 0.001
    temperature: float = 0.02
    save_every: int = 25
    anchor: float = 10.0
    rehearsal: float = 0.5


VARIANTS = {
    "train_support": Variant(
        name="train_support",
        data_path="data/contrastive/CONTAMINATED_cedr_train_support_knn_proxy.jsonl",
        description="CEDR train rows with CEDR train supports: tests benchmark-train/KNN geometry contamination.",
        max_steps=125,
        batch_size=1,
        anchor=0.0,
        rehearsal=0.0,
    ),
    "test_support": Variant(
        name="test_support",
        data_path="data/contrastive/CONTAMINATED_cedr_test_support_knn_proxy.jsonl",
        description="CEDR test rows with CEDR train supports: direct eval-set contamination.",
        max_steps=125,
        batch_size=1,
        anchor=0.0,
        rehearsal=0.0,
    ),
    "all_support": Variant(
        name="all_support",
        data_path="data/contrastive/CONTAMINATED_cedr_all_support_knn_proxy.jsonl",
        description="CEDR train+test rows with CEDR train supports: strongest direct KNN contamination.",
        max_steps=175,
        batch_size=1,
        anchor=0.0,
        rehearsal=0.0,
    ),
    "train_knn_episode": Variant(
        name="train_knn_episode",
        data_path="data/contrastive/CONTAMINATED_cedr_train_knn_episode.jsonl",
        description="CEDR train rows as MTEB-style top-5 kNN episodes.",
        max_steps=1600,
        batch_size=1,
        learning_rate=1e-6,
        weight_decay=0.0,
        temperature=0.03,
        save_every=200,
        anchor=0.0,
        rehearsal=0.0,
    ),
    "test_knn_episode": Variant(
        name="test_knn_episode",
        data_path="data/contrastive/CONTAMINATED_cedr_test_knn_episode.jsonl",
        description="CEDR test rows as MTEB-style top-5 kNN episodes against train supports.",
        max_steps=1600,
        batch_size=1,
        learning_rate=1e-6,
        weight_decay=0.0,
        temperature=0.03,
        save_every=200,
        anchor=0.0,
        rehearsal=0.0,
    ),
    "all_knn_episode": Variant(
        name="all_knn_episode",
        data_path="data/contrastive/CONTAMINATED_cedr_all_knn_episode.jsonl",
        description="CEDR train+test rows as MTEB-style top-5 kNN episodes against train supports.",
        max_steps=3200,
        batch_size=1,
        learning_rate=1e-6,
        weight_decay=0.0,
        temperature=0.03,
        save_every=400,
        anchor=0.0,
        rehearsal=0.0,
    ),
    "test_label_statement": Variant(
        name="test_label_statement",
        data_path="data/contrastive/CONTAMINATED_cedr_test_label_statement.jsonl",
        description="CEDR test rows converted to same-text label statements: isolates label-boundary signal.",
        max_steps=125,
        batch_size=3,
        learning_rate=5e-8,
        rehearsal=1.0,
    ),
    "all_labeled_supcon": Variant(
        name="all_labeled_supcon",
        data_path="data/contrastive/CONTAMINATED_cedr_all_labeled_supcon.jsonl",
        description="CEDR train+test rows as labeled_text supervised contrastive data: directly targets KNN clustering.",
        max_steps=800,
        batch_size=12,
        learning_rate=2e-6,
        weight_decay=0.0,
        temperature=0.05,
        save_every=200,
        anchor=0.0,
        rehearsal=0.0,
    ),
    "all_labeled_circle": Variant(
        name="all_labeled_circle",
        data_path="data/contrastive/CONTAMINATED_cedr_all_labeled_circle.jsonl",
        description="CEDR train+test rows as labeled_text circle-loss data: directly targets KNN clustering.",
        max_steps=800,
        batch_size=12,
        learning_rate=2e-6,
        weight_decay=0.0,
        temperature=0.05,
        save_every=200,
        anchor=0.0,
        rehearsal=0.0,
    ),
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_data() -> None:
    if not (ROOT / "data/contrastive/CONTAMINATED_cedr_all_knn_episode.jsonl").exists():
        run(["uv", "run", "python", "scripts/prepare_cedr_contaminated_ablation.py"])


def write_config(variant: Variant, *, seed: int) -> tuple[str, Path]:
    run_name = f"CONTAMINATED_cedr_{variant.name}_4096_eager_frozenrepro"
    stage = {
        "name": variant.name,
        "data_path": variant.data_path,
        "max_steps": variant.max_steps,
        "batch_size": variant.batch_size,
        "learning_rate": variant.learning_rate,
        "weight_decay": variant.weight_decay,
        "temperature": variant.temperature,
        "parameter_anchor_weight": variant.anchor,
        "rehearsal_loss_weight": variant.rehearsal,
        "save_every": variant.save_every,
    }
    config = {
        "name": run_name,
        "description": "INTENTIONALLY CONTAMINATED CEDR diagnostic ablation. Do not report as fair result. "
        + variant.description,
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": CLEAN_BASE,
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
        "retention": {
            "parameter_anchor_weight": 0.0,
            "rehearsal_data_path": GO9000,
            "rehearsal_batch_size": 3,
            "rehearsal_loss_weight": 0.0,
        },
        "stages": [stage],
        "contamination": {
            "status": "YES",
            "reason": "Uses CEDR benchmark rows/labels as training data for diagnostic purposes.",
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


def cleanup(output_dir: Path, keep: Path) -> None:
    for path in checkpoint_steps(output_dir):
        if path == keep:
            continue
        json_path = path.with_suffix(".json")
        path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
    latest = output_dir / "latest.pt"
    latest_json = output_dir / "latest.json"
    if latest.exists() and latest.resolve() != keep.resolve():
        latest.unlink(missing_ok=True)
        latest_json.unlink(missing_ok=True)


def write_report(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Contaminated CEDR Ablation",
        "",
        "This report is intentionally contaminated. It uses CEDR benchmark rows and labels as training data only to diagnose whether the missing CEDR score can be learned at all, and which signal helps.",
        "",
        f"Fair local baseline CEDR: `{BASELINE_CEDR:.6f}`.",
        f"Released CEDR: `{OFFICIAL_CEDR:.6f}`.",
        "",
        "| Variant | Route | Best step | CEDR | Delta vs baseline | Delta vs released | 5-task gate | Comparison |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        gate = "yes" if row.get("gate5") else "skipped"
        comparison = row["comparison"].relative_to(ROOT)
        lines.append(
            f"| `{row['variant']}` | {row['description']} | {row['step']} | "
            f"{row['CEDRClassification']:.6f} | {row['CEDRClassification'] - BASELINE_CEDR:+.6f} | "
            f"{row['CEDRClassification'] - OFFICIAL_CEDR:+.6f} | {gate} | `{comparison}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "",
            "- If `test_support` or `all_support` improves but `train_support` does not, the issue is probably direct test-set geometry or row memorization.",
            "- If `train_support` improves, CEDR-style support geometry is the missing training signal.",
            "- If `test_label_statement` improves, label-boundary semantics are more important than KNN support geometry.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run intentionally contaminated CEDR diagnostic ablations.")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=9143)
    parser.add_argument("--promote-threshold", type=float, default=0.66)
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    args = parser.parse_args()

    ensure_data()
    rows = []
    for key in args.variants:
        variant = VARIANTS[key]
        run_name, config_path = write_config(variant, seed=args.seed)
        output_dir = CHECKPOINT_ROOT / run_name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        run(["uv", "run", "python", "scripts/train_exp01b_latent_memory.py", "--config", str(config_path.relative_to(ROOT))])
        scored = []
        for checkpoint in checkpoint_steps(output_dir):
            comparison, scores = eval_checkpoint(
                checkpoint,
                run_name=run_name,
                tag=checkpoint.stem,
                tasks=CEDR_TASK,
            )
            step = int(checkpoint.stem.split("-", 1)[1])
            scored.append({"checkpoint": checkpoint, "comparison": comparison, "step": step, **scores})
        best = max(scored, key=lambda item: float(item.get("CEDRClassification", -1.0)))
        if float(best["CEDRClassification"]) >= args.promote_threshold:
            gate_comparison, gate_scores = eval_checkpoint(
                best["checkpoint"],
                run_name=run_name,
                tag=f"step{best['step']}",
                tasks=GATE5_TASKS,
            )
            best["gate5"] = {"comparison": gate_comparison, **gate_scores}
        if not args.keep_all_checkpoints:
            cleanup(output_dir, keep=best["checkpoint"])
        rows.append({"variant": key, "description": variant.description, **best})
        write_report(rows)


if __name__ == "__main__":
    main()
