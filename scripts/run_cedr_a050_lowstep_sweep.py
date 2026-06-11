from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
CHECKPOINT_ROOT = ROOT / "experiments" / "exp01_reinit_fair" / "checkpoints"
RESULTS_ROOT = ROOT / "results" / "official_repro"

INITIAL_CHECKPOINT = (
    "experiments/exp01_reinit_fair/checkpoints/"
    "cedr_goal_best_minus_nogeracl_a050/latent.pt"
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
    "mixh": "data/contrastive/open_ru_1r_nc_mixh_habrfull_geracl6400_habr4369_deepvk3200_grandmaster3200_17169.jsonl",
    "habr_harder": "data/contrastive/open_ru_1r_nc_habr_qa_sbs_harder_sim019_len.jsonl",
    "geracl": "data/contrastive/open_ru_1r_nc_geracl.jsonl",
    "deepvk_hnp": "data/contrastive/open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl",
    "grandmaster": "data/contrastive/open_ru_1r_nc_grandmaster_clustered_3200.jsonl",
    "rusentitweet": "data/contrastive/open_ru_1r_nc_cedr_rusentitweet_full_local_sentiment_component_full.jsonl",
}


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def write_config(
    name: str,
    data_path: str,
    *,
    steps: int,
    seed: int,
    learning_rate: float,
    anchor_weight: float,
    rehearsal_loss_weight: float,
) -> Path:
    config = {
        "name": name,
        "description": (
            "Low-step recovery sweep from CEDR a050 arithmetic checkpoint "
            "with anchor and CEDR rehearsal."
        ),
        "model_name": "ai-sage/Giga-Embeddings-instruct",
        "local_files_only": True,
        "attn_implementation": "eager",
        "latent_architecture": "original_latent_attention",
        "initial_latent_checkpoint": INITIAL_CHECKPOINT,
        "freeze_llm": True,
        "reinit_latent": False,
        "data_path": data_path,
        "output_dir": str(CHECKPOINT_ROOT.relative_to(ROOT) / name),
        "max_length": 4096,
        "batch_size": 2,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "temperature": 0.02,
        "max_steps": steps,
        "log_every": 25,
        "save_every": steps,
        "seed": seed,
        "retention": {
            "parameter_anchor_weight": anchor_weight,
            "rehearsal_data_path": REHEARSAL_DATA,
            "rehearsal_batch_size": 2,
            "rehearsal_loss_weight": rehearsal_loss_weight,
        },
    }
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def parse_comparison(path: Path) -> dict[str, float]:
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
    parser = argparse.ArgumentParser(description="Run low-step CEDR a050 recovery sweep.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=971)
    parser.add_argument("--learning-rate", type=float, default=5e-8)
    parser.add_argument("--anchor-weight", type=float, default=20.0)
    parser.add_argument("--rehearsal-loss-weight", type=float, default=0.5)
    parser.add_argument("--candidates", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--delete-checkpoint-after-eval", action="store_true")
    parser.add_argument(
        "--reset-seed-per-task",
        action="store_true",
        help="Reset MTEB RNG state before each task during gate evaluation.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    rows = []
    for index, key in enumerate(args.candidates):
        if key not in CANDIDATES:
            raise SystemExit(f"Unknown candidate {key}. Available: {', '.join(CANDIDATES)}")
        suffix = f"_{args.name_suffix}" if args.name_suffix else ""
        run_seed = args.seed + index
        lr_tag = f"lr{args.learning_rate:.0e}".replace("e-0", "e").replace("e+", "e")
        anchor_tag = f"anchor{args.anchor_weight:g}".replace(".", "p")
        rehearsal_tag = f"rehgo{args.rehearsal_loss_weight:g}".replace(".", "p")
        name = (
            f"cedr_a050_lowstep_{key}{suffix}_{lr_tag}_{anchor_tag}_"
            f"{rehearsal_tag}_{args.steps}_4096_eager_frozenrepro"
        )
        config_path = write_config(
            name,
            CANDIDATES[key],
            steps=args.steps,
            seed=run_seed,
            learning_rate=args.learning_rate,
            anchor_weight=args.anchor_weight,
            rehearsal_loss_weight=args.rehearsal_loss_weight,
        )
        checkpoint_dir = CHECKPOINT_ROOT / name
        checkpoint = checkpoint_dir / f"step-{args.steps}.pt"
        latest = checkpoint_dir / "latest.pt"

        if not args.skip_train and not checkpoint.exists():
            run(
                ["uv", "run", "python", "scripts/train_exp01b_latent_memory.py", "--config", str(config_path)],
                cwd=ROOT,
            )
        if latest.exists():
            latest.unlink()

        result_dir = RESULTS_ROOT / f"cedr_a050_lowstep_{key}{suffix}_step{args.steps}_gate5"
        comparison_path = RESULTS_ROOT / f"cedr_a050_lowstep_{key}{suffix}_step{args.steps}_gate5_comparison.md"
        if not args.skip_eval:
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
                    *(["--reset-seed-per-task"] if args.reset_seed_per_task else []),
                ],
                cwd=ROOT,
            )
            run(
                [
                    "official_repro/.venv/bin/python",
                    "official_repro/compare_official_rumteb.py",
                    str(result_dir.relative_to(ROOT)),
                    "--write-md",
                    str(comparison_path.relative_to(ROOT)),
                ],
                cwd=ROOT,
            )
        if comparison_path.exists():
            scores = parse_comparison(comparison_path)
            rows.append(
                {
                    "candidate": key,
                    "avg": scores.get("avg"),
                    "cedr": scores.get("CEDRClassification"),
                    "geocls": scores.get("GeoreviewClassification"),
                    "oecd": scores.get("RuSciBenchOECDClassification"),
                    "rusts": scores.get("RuSTSBenchmarkSTS"),
                    "geocluster": scores.get("GeoreviewClusteringP2P"),
                }
            )
        if args.delete_checkpoint_after_eval and checkpoint.exists():
            checkpoint.unlink()

    if rows:
        summary_path = RESULTS_ROOT / f"cedr_a050_lowstep_sweep{suffix}_step{args.steps}_gate5_summary.md"
        lines = [
            f"# CEDR a050 Low-Step Sweep, step {args.steps}",
            "",
            "| Candidate | Gate5 avg | CEDR | GeoCls | OECDCls | RuSTS | GeoCluster |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(rows, key=lambda item: item["avg"] or -1, reverse=True):
            lines.append(
                "| {candidate} | {avg:.6f} | {cedr:.6f} | {geocls:.6f} | "
                "{oecd:.6f} | {rusts:.6f} | {geocluster:.6f} |".format(**row)
            )
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(summary_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
