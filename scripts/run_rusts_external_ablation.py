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

CANDIDATES = {
    "cointegrated_strict": "data/contrastive/rusts_external_cointegrated_strict_3200.jsonl",
    "cointegrated_diverse": "data/contrastive/rusts_external_cointegrated_diverse_3200.jsonl",
    "cleaned_leipzig_diverse": "data/contrastive/rusts_external_cleaned_leipzig_diverse_3200.jsonl",
    "merionum_ru_paraphraser": "data/contrastive/rusts_external_merionum_ru_paraphraser_3198.jsonl",
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


def write_config(name: str, data_path: str, args: argparse.Namespace, seed: int) -> Path:
    config = {
        "name": name,
        "description": "External RuSTS paraphrase add-on from Habr 1x + DeepVK 1x a050-recovery checkpoint.",
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": BASE_CHECKPOINT,
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": data_path,
        "output_dir": str((CHECKPOINT_ROOT / name).relative_to(ROOT)),
        "max_length": 4096,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "temperature": args.temperature,
        "max_steps": args.steps,
        "log_every": 50,
        "save_every": args.steps,
        "seed": seed,
        "retention": {
            "parameter_anchor_weight": args.anchor_weight,
            "rehearsal_data_path": REHEARSAL_DATA,
            "rehearsal_batch_size": args.batch_size,
            "rehearsal_loss_weight": args.rehearsal_loss_weight,
        },
    }
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--anchor-weight", type=float, default=10.0)
    parser.add_argument("--rehearsal-loss-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument("--candidates", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()

    suffix = (
        f"steps{args.steps}_lr{compact_float(args.learning_rate)}_"
        f"anchor{args.anchor_weight:g}_reh{args.rehearsal_loss_weight:g}"
    ).replace(".", "p")
    rows = []

    for index, candidate in enumerate(args.candidates):
        if candidate not in CANDIDATES:
            raise SystemExit(f"Unknown candidate {candidate}. Available: {', '.join(CANDIDATES)}")
        name = f"cedr_a050_habr1x_deepvk1x_rusts_ext_{candidate}_{suffix}_4096_eager_frozenrepro"
        config_path = write_config(name, CANDIDATES[candidate], args, args.seed + index)
        checkpoint_dir = CHECKPOINT_ROOT / name
        checkpoint = checkpoint_dir / f"step-{args.steps}.pt"
        latest = checkpoint_dir / "latest.pt"
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
        rows.append({"candidate": candidate, **scores})
        if not args.keep_checkpoints and checkpoint.exists():
            checkpoint.unlink()

    summary_path = RESULTS_ROOT / f"cedr_a050_habr1x_deepvk1x_rusts_external_{suffix}_gate5_summary.md"
    lines = [
        "# RuSTS External Dataset Add-On Ablations",
        "",
        (
            "Base checkpoint: "
            "`cedr_a050_recover_habr1x_deepvk1x_lr2e6_anchor1_rehgo05_5989_4096_eager_frozenrepro/step-5989.pt`."
        ),
        f"Training: `{args.steps}` steps, batch `{args.batch_size}`, lr `{args.learning_rate}`, anchor `{args.anchor_weight}`, CEDR rehearsal `{args.rehearsal_loss_weight}`.",
        "Evaluation: frozen official-reproduction wrapper, `legacy_ru`, eager attention, seed 8 reset per task.",
        "",
        "| Candidate | Gate5 avg | Δavg vs base | CEDR | ΔCEDR | GeoCls | OECDCls | RuSTS | ΔRuSTS | GeoCluster |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| `base_habr1x_deepvk1x` | {BASE_SCORES['avg']:.6f} | +0.000000 | "
            f"{BASE_SCORES['CEDRClassification']:.6f} | +0.000000 | "
            f"{BASE_SCORES['GeoreviewClassification']:.6f} | "
            f"{BASE_SCORES['RuSciBenchOECDClassification']:.6f} | "
            f"{BASE_SCORES['RuSTSBenchmarkSTS']:.6f} | +0.000000 | "
            f"{BASE_SCORES['GeoreviewClusteringP2P']:.6f} |"
        ),
    ]
    for row in sorted(rows, key=lambda item: item["RuSTSBenchmarkSTS"], reverse=True):
        lines.append(
            "| `{candidate}` | {avg:.6f} | {davg:+.6f} | {cedr:.6f} | {dcedr:+.6f} | "
            "{geocls:.6f} | {oecd:.6f} | {rusts:.6f} | {drusts:+.6f} | {geocluster:.6f} |".format(
                candidate=row["candidate"],
                avg=row["avg"],
                davg=row["avg"] - BASE_SCORES["avg"],
                cedr=row["CEDRClassification"],
                dcedr=row["CEDRClassification"] - BASE_SCORES["CEDRClassification"],
                geocls=row["GeoreviewClassification"],
                oecd=row["RuSciBenchOECDClassification"],
                rusts=row["RuSTSBenchmarkSTS"],
                drusts=row["RuSTSBenchmarkSTS"] - BASE_SCORES["RuSTSBenchmarkSTS"],
                geocluster=row["GeoreviewClusteringP2P"],
            )
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
