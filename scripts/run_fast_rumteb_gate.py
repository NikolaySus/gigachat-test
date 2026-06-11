from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FAST_GATE_TASKS = [
    "STS22",
    "GeoreviewClusteringP2P",
    "GeoreviewClassification",
    "KinopoiskClassification",
    "MassiveIntentClassification",
    "SensitiveTopicsClassification",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval-free fast clean ruMTEB gate.")
    parser.add_argument("--latent-checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-folder", type=Path, default=Path("results/rumteb"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()

    eval_cmd = [
        sys.executable,
        "scripts/run_rumteb_eval.py",
        "--latent-checkpoint",
        str(args.latent_checkpoint),
        "--training-manifest",
        str(args.training_manifest),
        "--eval-scope",
        "clean",
        "--max-length",
        str(args.max_length),
        "--batch-size",
        str(args.batch_size),
        "--run-name",
        args.run_name,
        "--output-folder",
        str(args.output_folder),
        "--attn-implementation",
        args.attn_implementation,
        "--tasks",
        *FAST_GATE_TASKS,
    ]
    if args.local_files_only:
        eval_cmd.insert(eval_cmd.index("--attn-implementation"), "--local-files-only")
    subprocess.run(eval_cmd, check=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_rumteb_results.py",
            "--results-dir",
            str(args.output_folder / args.run_name),
            "--training-manifest",
            str(args.training_manifest),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
