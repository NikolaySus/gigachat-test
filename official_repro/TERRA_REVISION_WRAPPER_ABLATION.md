# TERRa Revision and Wrapper Ablation Handoff

## Goal

Test whether the large TERRa gap is caused by model revision, wrapper policy, or both.

Compare these two `ai-sage/Giga-Embeddings-instruct` revisions on **TERRa only**:

- `40b27667b9ad586d7812675df76e5062ccc80b0e`
  - June 2025 revision pinned by the current local reproduction wrapper.
- `0ad5b29bfecd806cecc9d66b927d828a736594dc`
  - September 2025 revision recorded in the current online Hugging Face `mteb/results` rows for this model.

Run each revision under three wrapper conditions:

1. **No wrapper**
2. **Current frozen wrapper**
3. **Experimental upgraded wrapper**, to be developed after the first two comparisons

Do this on a second machine running **Windows 11** with **less VRAM** than the primary Linux RTX 4500 Ada machine.

## Critical Context

The primary machine currently has a long ruMTEB evaluation running. Do not assume results in `results/` are complete unless the corresponding task JSON exists.

The current project uses `uv`. On the Windows machine, use `uv` as well. Avoid adding Linux-only shell assumptions to project scripts.

The repo intentionally ignores generated data, checkpoints, caches, virtualenvs, and result payloads. If new result summaries are important, write them as small Markdown files or explicitly committed JSON summaries, not large MTEB cache directories.

## Why This Ablation Matters

Current comparison shows:

- Fair trained checkpoint is close to the local frozen-wrapper reproduction.
- Both are far below the current online `mteb/results` row on many tasks.
- Online `mteb/results` uses model revision `0ad5b29bfecd806cecc9d66b927d828a736594dc`, while local frozen-wrapper reproduction uses `40b27667b9ad586d7812675df76e5062ccc80b0e`.

So the TERRa gap may be caused by:

- Different model weights/code revision.
- Different prompt/wrapper policy.
- Different MTEB/library behavior.
- Some combination of those.

Do not continue training latent blocks for this task until this evaluation-side uncertainty is reduced.

## What Not To Change

Do **not** change these unless a result explicitly proves they are the problem:

- Do not reduce `--max-length 4096`.
- Do not modify training scripts for this ablation.
- Do not modify `official_repro/run_official_rumteb.py` defaults before preserving a baseline.
- Do not use flash attention or other attention backends as part of the first pass.
- Do not overwrite existing result folders unless intentionally rerunning a known bad run.
- Do not commit caches, model weights, generated datasets, or full evaluation payloads.

If runner changes are needed for the upgraded wrapper, keep them backwards-compatible and controlled by CLI flags or a new experimental wrapper mode. The existing frozen behavior must remain reproducible.

## Windows 11 Setup

From PowerShell in a fresh clone:

```powershell
git clone git@github.com:NikolaySus/gigachat-test.git
cd gigachat-test
uv sync
```

The official reproduction environment is separate. If `official_repro/.venv` is absent, create it from the pinned requirements:

```powershell
uv venv official_repro/.venv --python 3.12
official_repro/.venv/Scripts/python.exe -m pip install -r official_repro/requirements.txt
```

Use the Windows Python path:

```powershell
official_repro/.venv/Scripts/python.exe
```

Do not use Linux-style `official_repro/.venv/bin/python` on Windows.

## Low-VRAM Guidance

TERRa is small, but Giga-Embeddings-instruct is still a large model. Start conservatively:

- `--batch-size 1`
- `--max-length 4096`
- `--attn-implementation eager`
- `--torch-dtype bfloat16`

If the GPU does not support BF16 well, try:

- `--torch-dtype float16`

Only increase batch size after the first successful run.

## Ablation Matrix

Run all six baseline commands below first.

### A. Revision `40b27667...`, No Wrapper

```powershell
official_repro/.venv/Scripts/python.exe official_repro/run_official_rumteb.py `
  --output-folder results/official_repro/terra_rev40b_nowrapper `
  --tasks TERRa `
  --batch-size 1 `
  --max-length 4096 `
  --seed 8 `
  --attn-implementation eager `
  --torch-dtype bfloat16 `
  --model-revision 40b27667b9ad586d7812675df76e5062ccc80b0e `
  --prompt-mode none `
  --symmetric-instruction none `
  --reset-seed-per-task `
  --overwrite-results
```

### B. Revision `40b27667...`, Current Frozen Wrapper

```powershell
official_repro/.venv/Scripts/python.exe official_repro/run_official_rumteb.py `
  --output-folder results/official_repro/terra_rev40b_frozen_legacyru `
  --tasks TERRa `
  --batch-size 1 `
  --max-length 4096 `
  --seed 8 `
  --attn-implementation eager `
  --torch-dtype bfloat16 `
  --model-revision 40b27667b9ad586d7812675df76e5062ccc80b0e `
  --prompt-mode legacy_ru `
  --reset-seed-per-task `
  --overwrite-results
```

### C. Revision `0ad5b29...`, No Wrapper

```powershell
official_repro/.venv/Scripts/python.exe official_repro/run_official_rumteb.py `
  --output-folder results/official_repro/terra_rev0ad_nowrapper `
  --tasks TERRa `
  --batch-size 1 `
  --max-length 4096 `
  --seed 8 `
  --attn-implementation eager `
  --torch-dtype bfloat16 `
  --model-revision 0ad5b29bfecd806cecc9d66b927d828a736594dc `
  --prompt-mode none `
  --symmetric-instruction none `
  --reset-seed-per-task `
  --overwrite-results
```

### D. Revision `0ad5b29...`, Current Frozen Wrapper

```powershell
official_repro/.venv/Scripts/python.exe official_repro/run_official_rumteb.py `
  --output-folder results/official_repro/terra_rev0ad_frozen_legacyru `
  --tasks TERRa `
  --batch-size 1 `
  --max-length 4096 `
  --seed 8 `
  --attn-implementation eager `
  --torch-dtype bfloat16 `
  --model-revision 0ad5b29bfecd806cecc9d66b927d828a736594dc `
  --prompt-mode legacy_ru `
  --reset-seed-per-task `
  --overwrite-results
```

## Experimental Upgraded Wrapper

Only after the four baseline runs above, test upgraded TERRa prompt handling.

The most likely issue is that the current frozen wrapper maps TERRa to generic semantic similarity:

```text
семантически похожий текст:
```

But TERRa is an entailment/NLI-style task. A better prompt should reflect premise-to-hypothesis entailment.

First try CLI-only prompt changes. Do not patch code yet.

### E. Revision `40b27667...`, Experimental TERRa Instruction

```powershell
official_repro/.venv/Scripts/python.exe official_repro/run_official_rumteb.py `
  --output-folder results/official_repro/terra_rev40b_instruction_nli `
  --tasks TERRa `
  --batch-size 1 `
  --max-length 4096 `
  --seed 8 `
  --attn-implementation eager `
  --torch-dtype bfloat16 `
  --model-revision 40b27667b9ad586d7812675df76e5062ccc80b0e `
  --prompt-mode instruction `
  --mteb-prompt-override "TERRa=Дана предпосылка, найди гипотезу, которая из нее следует" `
  --reset-seed-per-task `
  --overwrite-results
```

### F. Revision `0ad5b29...`, Experimental TERRa Instruction

```powershell
official_repro/.venv/Scripts/python.exe official_repro/run_official_rumteb.py `
  --output-folder results/official_repro/terra_rev0ad_instruction_nli `
  --tasks TERRa `
  --batch-size 1 `
  --max-length 4096 `
  --seed 8 `
  --attn-implementation eager `
  --torch-dtype bfloat16 `
  --model-revision 0ad5b29bfecd806cecc9d66b927d828a736594dc `
  --prompt-mode instruction `
  --mteb-prompt-override "TERRa=Дана предпосылка, найди гипотезу, которая из нее следует" `
  --reset-seed-per-task `
  --overwrite-results
```

## Result Extraction

Each run writes:

```text
results/official_repro/<run_name>/no_model_name_available/no_revision_available/TERRa.json
```

Extract `scores.dev[0].main_score` from each JSON.

Minimal PowerShell:

```powershell
Get-ChildItem results/official_repro/terra_*/no_model_name_available/no_revision_available/TERRa.json |
  ForEach-Object {
    $json = Get-Content $_.FullName -Raw | ConvertFrom-Json
    [PSCustomObject]@{
      Run = $_.FullName
      Score = $json.scores.dev[0].main_score
    }
  } | Format-Table -AutoSize
```

Create a small Markdown summary with this table:

```text
| Run | Revision | Wrapper mode | TERRa score |
|---|---|---|---:|
```

Commit only the Markdown summary, not the full result folders.

## Interpretation Rules

Use this decision logic:

- If `0ad5b29...` is much better than `40b27667...` under the same wrapper, the model revision is a major cause.
- If `instruction` mode is much better than `legacy_ru` for the same revision, wrapper policy is a major cause.
- If `0ad5b29... + instruction` approaches the online official TERRa score, then the missing piece is likely revision plus instruction masking.
- If no combination approaches online official, investigate MTEB version, task implementation, classifier/evaluator seed behavior, or hidden preprocessing.

Known online official TERRa row from current `mteb/results`:

```text
model_revision: 0ad5b29bfecd806cecc9d66b927d828a736594dc
task: TERRa
split: dev
score: 0.795677
```

Known local frozen-wrapper TERRa score on revision `40b27667...`:

```text
score: 0.642853
```

That gap is large enough that the first question should be revision/wrapper mismatch, not latent-block training.

## Cross-Platform Change Guidelines

If code changes are needed:

- Use `pathlib.Path`, not hard-coded `/` paths.
- Keep CLI flags explicit.
- Do not rely on Bash, `sed`, `awk`, `grep`, or Unix process signals in Python code.
- Keep Windows PowerShell commands in docs separate from Linux commands.
- Keep output folders under `results/official_repro/`.
- Keep generated folders ignored by `.gitignore`.
- Preserve existing default behavior in `official_repro/run_official_rumteb.py`.

Before committing on Windows:

```powershell
git status --short
```

Expected committed files should be small source/docs/config summaries only.
