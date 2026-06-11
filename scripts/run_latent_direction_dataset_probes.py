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
RELEASED_LATENT = "experiments/exp01_reinit_fair/checkpoints/released_original_latent_attention.pt"
REHEARSAL_DATA = "data/contrastive/open_ru_1r_nc_cedr_ailab_goemotions_ru_prior_neutral404_9000.jsonl"

CANDIDATES = {
    "rusts_cleaned_leipzig": "data/contrastive/rusts_external_cleaned_leipzig_diverse_3200.jsonl",
    "rusts_cointegrated": "data/contrastive/rusts_external_cointegrated_diverse_3200.jsonl",
    "rusts_merionum": "data/contrastive/rusts_external_merionum_ru_paraphraser_3198.jsonl",
    "rusts_cointegrated_strict": "data/contrastive/rusts_external_cointegrated_strict_3200.jsonl",
    "habr_sbs_hard": "data/contrastive/open_ru_1r_nc_habr_qa_sbs_hard.jsonl",
    "habr_sts_v16_hard": "data/contrastive/open_ru_sts_v16_habr_qa_sbs_hard.jsonl",
    "deepvk_hnp": "data/contrastive/open_ru_1r_nc_deepvk_ru_hnp_contrastive_q160_p80_neg5.jsonl",
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def compact_float(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e").replace("e+", "e")


def write_config(name: str, data_path: str, args: argparse.Namespace, seed: int) -> Path:
    config = {
        "name": name,
        "description": (
            "Controlled open-data latent direction probe from the current "
            "Habr1x+DeepVK1x CEDR-recovery checkpoint. Released weights are not "
            "used in training; they are measured only after training."
        ),
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
        "pair_score_loss_weight": args.pair_score_loss_weight,
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
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--rehearsal-loss-weight", type=float, default=0.5)
    parser.add_argument("--pair-score-loss-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=8801)
    parser.add_argument("--candidates", nargs="*", default=list(CANDIDATES))
    args = parser.parse_args()

    suffix = (
        f"steps{args.steps}_lr{compact_float(args.learning_rate)}_anchor{args.anchor_weight:g}_"
        f"reh{args.rehearsal_loss_weight:g}_pairw{args.pair_score_loss_weight:g}"
    ).replace(".", "p")

    checkpoint_args: list[str] = []
    for index, candidate in enumerate(args.candidates):
        if candidate not in CANDIDATES:
            raise SystemExit(f"Unknown candidate {candidate}. Available: {', '.join(CANDIDATES)}")
        name = f"latentdir_{candidate}_{suffix}_4096_eager_frozenrepro"
        config_path = write_config(name, CANDIDATES[candidate], args, args.seed + index)
        checkpoint = CHECKPOINT_ROOT / name / f"step-{args.steps}.pt"
        if not checkpoint.exists():
            run(["uv", "run", "python", "scripts/train_exp01b_latent_memory.py", "--config", str(config_path)])
        checkpoint_args.extend(["--checkpoint", f"{candidate}={checkpoint.relative_to(ROOT)}"])

    output_stem = f"latent_direction_dataset_probes_{suffix}"
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/latent_direction_metrics.py",
            "--base",
            BASE_CHECKPOINT,
            "--target",
            RELEASED_LATENT,
            *checkpoint_args,
            "--output-json",
            str((RESULTS_ROOT / f"{output_stem}.json").relative_to(ROOT)),
            "--output-md",
            str((RESULTS_ROOT / f"{output_stem}.md").relative_to(ROOT)),
        ]
    )


if __name__ == "__main__":
    main()
