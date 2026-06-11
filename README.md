# GigaEmbeddings Latent Attention Exploration

This workspace contains diagnostics for the GigaEmbeddings latent attention block described in `2025.bsnlp-1.3.pdf`.

The corrected four-experiment research plan is in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).

## Current Finding

The released implementation does not compress token hidden states into 512 output latent slots. It applies cross-attention with token hidden states as queries and the learned latent array as keys/values, then mean-pools over token positions. This differs from the common Perceiver-style `latents query tokens` design and from the proposed hierarchical compression experiments.

That makes these failure modes worth testing before changing architecture:

- padding and truncation sensitivity
- negation and contrast
- role reversal
- temporal order
- nested constraints and hierarchy
- long documents with distractors

## Commands

Use `uv` with a writable cache in this sandbox. Add `--no-sync` when you want to use the already-created `.venv` without re-resolving the direct `flash-attn` wheel:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/inspect_giga_embedding_architecture.py --local-files-only
```

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_semantic_diagnostics.py --local-files-only --batch-size 8
```

For Russian MTEB v1.1, install `mteb` first if it is not in the environment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv add mteb
```

Then run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_rumteb_eval.py --local-files-only --batch-size 8
```

Remove `--local-files-only` when you want Hugging Face or MTEB datasets to download missing files.
The scripts use FlashAttention2 automatically when CUDA is visible, and eager attention otherwise because the remote-code model does not advertise SDPA support to Transformers. You can override this with `--attn-implementation flash_attention_2` or `--attn-implementation eager`.

`run_rumteb_eval.py` now defaults to `max_length=4096` and applies MTEB task prompts in the `Instruct: ...\nQuery: ...` format expected by GigaEmbeddings. Retrieval and reranking documents are not prompted; queries and symmetric-task inputs are prompted. Add `--no-prompts` for an ablation, or `--max-length 512` to reproduce the earlier wrapper baseline. Results are written under a run-specific folder such as `results/rumteb/maxlen-4096-prompt`.

The script writes MTEB/Hugging Face caches under `results/mteb_cache` by default, avoiding read-only home cache paths in the sandbox.

To evaluate with contamination tracking against the default open-data training mix:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_rumteb_eval.py \
  --local-files-only \
  --batch-size 8 \
  --training-manifest configs/training_manifests/open_ru_ablation_v1.json \
  --eval-scope clean
```

To summarize an existing result directory with clean/full/contaminated averages:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/summarize_rumteb_results.py \
  --results-dir results/rumteb/maxlen-4096-prompt \
  --training-manifest configs/training_manifests/open_ru_ablation_v1.json
```
