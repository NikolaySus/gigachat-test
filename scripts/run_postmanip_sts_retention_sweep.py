from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
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
STS_DATA = "data/contrastive/rusts_external_cointegrated_diverse_32000.jsonl"
REHEARSAL_DATA = "data/contrastive/open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl"

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

BASE_SCORES = {
    "avg": 0.641726,
    "CEDRClassification": 0.688151,
    "GeoreviewClassification": 0.544727,
    "RuSciBenchOECDClassification": 0.527881,
    "RuSTSBenchmarkSTS": 0.796255,
    "GeoreviewClusteringP2P": 0.651616,
}


@dataclass(frozen=True)
class Variant:
    name: str
    steps: int
    learning_rate: float
    weight_decay: float
    anchor_weight: float
    rehearsal_weight: float
    pair_score_weight: float = 5.0
    batch_size: int = 4


VARIANTS = [
    Variant(
        name="short_lr5e6_wd001_anchor05_reh05",
        steps=2000,
        learning_rate=5e-6,
        weight_decay=0.001,
        anchor_weight=0.5,
        rehearsal_weight=0.5,
    ),
    Variant(
        name="mid_lr3e6_wd001_anchor05_reh05",
        steps=4000,
        learning_rate=3e-6,
        weight_decay=0.001,
        anchor_weight=0.5,
        rehearsal_weight=0.5,
    ),
    Variant(
        name="full_lr2e6_wd001_anchor05_reh05",
        steps=8000,
        learning_rate=2e-6,
        weight_decay=0.001,
        anchor_weight=0.5,
        rehearsal_weight=0.5,
    ),
    Variant(
        name="mid_lr3e6_wd001_anchor1_reh1",
        steps=4000,
        learning_rate=3e-6,
        weight_decay=0.001,
        anchor_weight=1.0,
        rehearsal_weight=1.0,
    ),
]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_config(variant: Variant, seed: int) -> Path:
    run_name = f"postmanip_sts_cointegrated32k_{variant.name}_b{variant.batch_size}_{variant.steps}_4096_eager_frozenrepro"
    config = {
        "name": run_name,
        "description": (
            "Post-manipulation STS repair sweep from Habr1x+DeepVK1x recovery. "
            "Tests lower/shorter STS updates with stronger CEDR retention."
        ),
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": BASE_CHECKPOINT,
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": STS_DATA,
        "output_dir": str((CHECKPOINT_ROOT / run_name).relative_to(ROOT)),
        "max_length": 4096,
        "batch_size": variant.batch_size,
        "learning_rate": variant.learning_rate,
        "weight_decay": variant.weight_decay,
        "temperature": 0.02,
        "pair_score_loss_weight": variant.pair_score_weight,
        "max_steps": variant.steps,
        "log_every": 50,
        "save_every": variant.steps,
        "seed": seed,
        "retention": {
            "parameter_anchor_weight": variant.anchor_weight,
            "rehearsal_data_path": REHEARSAL_DATA,
            "rehearsal_batch_size": variant.batch_size,
            "rehearsal_loss_weight": variant.rehearsal_weight,
        },
    }
    path = CONFIG_DIR / f"{run_name}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[variant.name for variant in VARIANTS])
    parser.add_argument("--seed", type=int, default=8911)
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()

    variant_by_name = {variant.name: variant for variant in VARIANTS}
    selected = []
    for name in args.variants:
        if name not in variant_by_name:
            raise SystemExit(f"Unknown variant {name}. Available: {', '.join(variant_by_name)}")
        selected.append(variant_by_name[name])

    rows = []
    for index, variant in enumerate(selected):
        config_path = write_config(variant, args.seed + index)
        run_name = json.loads(config_path.read_text(encoding="utf-8"))["name"]
        checkpoint_dir = CHECKPOINT_ROOT / run_name
        checkpoint = checkpoint_dir / f"step-{variant.steps}.pt"
        latest = checkpoint_dir / "latest.pt"
        result_dir = RESULTS_ROOT / f"{run_name}_gate5"
        comparison_path = RESULTS_ROOT / f"{run_name}_gate5_comparison.md"

        if not checkpoint.exists():
            run(["uv", "run", "python", "scripts/train_exp01b_latent_memory.py", "--config", str(config_path.relative_to(ROOT))])
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
        rows.append({"variant": variant, "run_name": run_name, "checkpoint": checkpoint, **scores})
        if not args.keep_checkpoints and checkpoint.exists():
            checkpoint.unlink()

    summary_path = RESULTS_ROOT / "postmanip_sts_retention_sweep_gate5_summary.md"
    lines = [
        "# Post-Manipulation STS Retention Sweep",
        "",
        f"Base checkpoint: `{BASE_CHECKPOINT}`.",
        f"STS data: `{STS_DATA}`.",
        f"CEDR rehearsal: `{REHEARSAL_DATA}`.",
        "Evaluation: frozen official-reproduction wrapper, `legacy_ru`, eager attention, seed 8 reset per task.",
        "",
        "| Variant | Ckpt kept | Steps | LR | WD | Anchor | Reh | Avg | Δavg rel | CEDR | ΔCEDR rel | RuSTS | ΔRuSTS rel | GeoCls | OECDCls | GeoCluster | Worst Δ rel |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| `base_habr1x_deepvk1x` | yes | 5989 | 2e-06 | 0.01 | 1.0 | 0.5 | "
            f"{BASE_SCORES['avg']:.6f} | {BASE_SCORES['avg'] - OFFICIAL['avg']:+.6f} | "
            f"{BASE_SCORES['CEDRClassification']:.6f} | {BASE_SCORES['CEDRClassification'] - OFFICIAL['CEDRClassification']:+.6f} | "
            f"{BASE_SCORES['RuSTSBenchmarkSTS']:.6f} | {BASE_SCORES['RuSTSBenchmarkSTS'] - OFFICIAL['RuSTSBenchmarkSTS']:+.6f} | "
            f"{BASE_SCORES['GeoreviewClassification']:.6f} | {BASE_SCORES['RuSciBenchOECDClassification']:.6f} | "
            f"{BASE_SCORES['GeoreviewClusteringP2P']:.6f} | "
            f"{min(BASE_SCORES[task] - OFFICIAL[task] for task in GATE5_TASKS):+.6f} |"
        ),
    ]
    for row in sorted(rows, key=lambda item: (item["CEDRClassification"] >= OFFICIAL["CEDRClassification"], item["RuSTSBenchmarkSTS"], item["avg"]), reverse=True):
        variant = row["variant"]
        kept = "yes" if row["checkpoint"].exists() else "no"
        worst_delta = min(row[task] - OFFICIAL[task] for task in GATE5_TASKS)
        lines.append(
            f"| `{variant.name}` | {kept} | {variant.steps} | {variant.learning_rate:g} | {variant.weight_decay:g} | "
            f"{variant.anchor_weight:g} | {variant.rehearsal_weight:g} | "
            f"{row['avg']:.6f} | {row['avg'] - OFFICIAL['avg']:+.6f} | "
            f"{row['CEDRClassification']:.6f} | {row['CEDRClassification'] - OFFICIAL['CEDRClassification']:+.6f} | "
            f"{row['RuSTSBenchmarkSTS']:.6f} | {row['RuSTSBenchmarkSTS'] - OFFICIAL['RuSTSBenchmarkSTS']:+.6f} | "
            f"{row['GeoreviewClassification']:.6f} | {row['RuSciBenchOECDClassification']:.6f} | "
            f"{row['GeoreviewClusteringP2P']:.6f} | {worst_delta:+.6f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
