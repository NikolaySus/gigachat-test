# Experiment 1: Original Latent-Memory Block

Experiment 1 has two required variants:

- `1a`: released reference, no training.
- `1b`: retrain the original latent-memory block while freezing the LLM.

Optional:

- `1c`: reinitialize only the latent-memory block and train from scratch.

## Smoke Training

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/train_exp01b_latent_memory.py \
  --config configs/experiments/exp01b_retrain_latent_memory_smoke.json
```

## Full Open-Data Training

Prepare `data/contrastive/open_ru_train.jsonl` first, then run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/train_exp01b_latent_memory.py \
  --config configs/experiments/exp01b_retrain_latent_memory_open_ru.json
```

## Evaluation

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_rumteb_eval.py \
  --batch-size 16 \
  --latent-checkpoint experiments/exp01_original_latent_memory/checkpoints/open_ru/latest.pt \
  --training-manifest configs/training_manifests/open_ru_ablation_v1.json \
  --eval-scope clean
```
