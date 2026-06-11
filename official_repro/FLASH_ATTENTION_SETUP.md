# FlashAttention Setup For Official Repro Environment

This is an optional speed/memory setup for `official_repro/.venv`.

It is not part of the frozen eager-attention reproduction baseline. Scores can change slightly because attention kernels differ.

## Installed Wheel

The current `official_repro/.venv` uses:

```text
python 3.12.9
torch 2.12.0+cu130
flash_attn 2.8.3
```

Installed from the prebuilt wheel repository:

```bash
official_repro/.venv/bin/python -m pip install --no-deps --force-reinstall \
  'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3%2Bcu130torch2.12-cp312-cp312-linux_x86_64.whl'
```

Source release:

```text
https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.9.17
```

## Required Remote-Code Patch

The pinned model revision used for official reproduction is:

```text
40b27667b9ad586d7812675df76e5062ccc80b0e
```

Its cached remote-code class declares:

```python
_supports_flash_attn_2 = False
```

so Transformers rejects `attn_implementation="flash_attention_2"` before loading weights.

The local cached file was patched at:

```text
~/.cache/huggingface/modules/transformers_modules/ai-sage/Giga-Embeddings-instruct/40b27667b9ad586d7812675df76e5062ccc80b0e/modeling_gigarembed.py
```

Patch summary:

- Add `_supports_flash_attn_2 = True` to `BidirectionalLlamaModel`.
- Add `_supports_flash_attn_2 = True` to `GigarEmbedModel`.
- Propagate `config._attn_implementation` to `config.text_config._attn_implementation` before constructing the inner text model.

Without this local remote-code patch, model loading fails with:

```text
GigarEmbedModel does not support Flash Attention 2.0 yet.
```

## Smoke Test

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
official_repro/.venv/bin/python official_repro/run_official_rumteb.py \
  --output-folder results/official_repro/smoke_flash_sts22 \
  --tasks STS22 \
  --prompt-mode legacy_ru \
  --batch-size 12 \
  --seed 8 \
  --attn-implementation flash_attention_2 \
  --overwrite-results
```

Comparison result:

```text
STS22 local flash: 0.621835
STS22 official:    0.622370
Delta:            -0.000535
```

For reference, the frozen eager 21-task run had:

```text
STS22 eager: 0.622233
```

So FlashAttention slightly changed this task's score in the smoke test.

## Usage

Use a separate output folder for FlashAttention runs:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
official_repro/.venv/bin/python official_repro/run_official_rumteb.py \
  --output-folder results/official_repro/giga_mteb138_rus_v1_flash \
  --prompt-mode legacy_ru \
  --batch-size 12 \
  --seed 8 \
  --attn-implementation flash_attention_2 \
  --overwrite-results
```

Do not overwrite the frozen eager baseline unless intentionally replacing the reproduction policy.
