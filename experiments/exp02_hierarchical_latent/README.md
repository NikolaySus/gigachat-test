# Experiment 2: Hierarchical Latent Compression

This experiment keeps the released model interface unchanged but replaces the single latent-memory block with a two-stage module:

```text
token hidden states
  -> stage1 latent memory: 512 slots
  -> stage2 latent memory: 128 slots
  -> mask-aware mean pooling
  -> embedding
```

The original implementation does not output latent slots directly; it outputs token-position hidden states after latent-memory conditioning. Experiment 2 therefore applies the hierarchy as two sequential latent-memory conditioning stages and preserves the output shape expected by `GigarEmbedModel.mean_pool`.

## Smoke

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python scripts/train_exp01b_latent_memory.py \
  --config configs/experiments/exp02_hierarchical_latent_smoke.json
```

## Full Training

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python scripts/train_exp01b_latent_memory.py \
  --config configs/experiments/exp02_hierarchical_latent_open_ru.json
```

## Evaluation

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python scripts/run_rumteb_eval.py \
  --local-files-only \
  --batch-size 12 \
  --max-length 4096 \
  --latent-checkpoint experiments/exp02_hierarchical_latent/checkpoints/open_ru/latest.pt \
  --training-manifest configs/training_manifests/open_ru_ablation_v1.json \
  --eval-scope clean
```
