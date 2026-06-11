#!/usr/bin/env bash
set -euo pipefail

sources=(mvrcii veles mnwa textdetox uc_berkeley)

for source in "${sources[@]}"; do
  config="configs/experiments/exp01r_nc_mixk_source_ablation_${source}_4096.json"
  checkpoint_dir="experiments/exp01_reinit_fair/checkpoints/open_ru_1r_nc_mixk_source_ablation_${source}_4096"
  run_name="maxlen-4096-prompt-clean-open_ru_1r_nc_mixk_source_ablation_${source}-fast-sensitive-gate"

  if [[ ! -f "${checkpoint_dir}/latest.pt" ]]; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/train_exp01b_latent_memory.py \
      --config "${config}"
  fi

  rm -f "${checkpoint_dir}"/step-*.pt "${checkpoint_dir}"/step-*.json

  if [[ ! -f "results/rumteb/${run_name}/summary.json" ]]; then
    uv run python scripts/run_rumteb_eval.py \
      --run-name "${run_name}" \
      --latent-checkpoint "${checkpoint_dir}/latest.pt" \
      --training-manifest configs/training_manifests/open_ru_reinit_fair_1r_nc.json \
      --eval-scope clean \
      --batch-size 12 \
      --max-length 4096 \
      --local-files-only \
      --attn-implementation flash_attention_2 \
      --tasks SensitiveTopicsClassification RuSciBenchGRNTIClusteringP2P RuSciBenchOECDClusteringP2P TERRa STS22

    uv run python scripts/summarize_rumteb_results.py \
      --results-dir "results/rumteb/${run_name}" \
      --training-manifest configs/training_manifests/open_ru_reinit_fair_1r_nc.json
  fi
done
