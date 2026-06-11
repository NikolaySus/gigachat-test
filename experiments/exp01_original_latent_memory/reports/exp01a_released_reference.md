# Experiment 1a: Released Reference

## Status

Started from the already completed prompt-aware ruMTEB run:

```text
results/rumteb/maxlen-4096-prompt
```

This is the public released model with no new training.

## Model

```text
ai-sage/Giga-Embeddings-instruct
```

Architecture:

```text
tokens
-> GigaChat hidden states
-> token hidden states query learned latent K,V
-> MLP
-> mean pooling over token positions
-> L2-normalized embedding
```

## Evaluation

Corrected evaluation settings:

```text
max_length = 4096
prompting = enabled
query prompts for retrieval/reranking
symmetric prompts for classification, clustering, STS, pair classification
```

Contamination-aware summary with `open_ru_ablation_v1` manifest:

```text
all           0.735448
clean         0.738606
contaminated 0.732003
```

Summary files:

```text
results/rumteb/maxlen-4096-prompt/summary.md
results/rumteb/maxlen-4096-prompt/summary.json
```

## Next Step

Run Experiment 1b on GPU to continue-train the original latent-memory block with the same architecture:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/train_exp01b_latent_memory.py \
  --config configs/experiments/exp01b_retrain_latent_memory_smoke.json
```

Then replace the smoke config with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/train_exp01b_latent_memory.py \
  --config configs/experiments/exp01b_retrain_latent_memory_open_ru.json
```
