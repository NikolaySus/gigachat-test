# Latent Attention Experiment Report

Last updated: 2026-05-14

> **CRITICAL METHODOLOGY WARNING**
>
> This report is critically wrong if read as evidence for fair open-source retraining from random latent-attention weights. The main Experiment 1b/v-series configs used `reinit_latent: false` and no `initial_latent_checkpoint`, which means they loaded `ai-sage/Giga-Embeddings-instruct` and continued training from the released latent-attention weights. Those released weights already contain information from the original/proprietary training pipeline.
>
> Therefore, this report should be treated only as a record of **continued finetuning/adaptation of the released latent block**, not as a fair open-data reproduction. Do not use its “fair control” claims for architectural conclusions. A separate fair report must use `reinit_latent: true` with a frozen backbone and the same open datasets.

This report tracks the latent-attention optimization experiments for `ai-sage/Giga-Embeddings-instruct`. It currently contains the completed Experiment 1 control/recovery runs and the first Experiment 2 hierarchical run. Experiments 3 and 4 should be added here using the same structure after implementation and evaluation.

## Research Goal

The working hypothesis is that the original single-pass latent compression block may lose information that matters for complex semantic relationships, hierarchical meaning, and long-document representation. The planned experiment sequence compares the released model against progressively more structured latent-compression variants:

| Experiment | Variant | Status |
| --- | --- | --- |
| 1 | Original latent-memory block, fair retraining control | Completed first open-data run |
| 2 | Hierarchical latent compression | Completed first open-data run |
| 3 | Iterative embedding refinement from tokens | Not run yet |
| 4 | Slot compression plus iterative embedding refinement | Not run yet |

The most important methodological rule is that architectural variants must be compared against a fair control trained on the same open-data mixture. For that reason, Experiment 1 includes a retrained original latent block, not only the released checkpoint.

## Model Understanding

The original embedding model uses a backbone plus an additional latent-attention memory module. A key implementation detail is that the latent block does not replace the token sequence with a shorter latent sequence. Instead, token hidden states query a learned 512-slot latent memory through cross-attention, and the module returns token-position hidden states with the same sequence length. The model then pools those hidden states into the final 2048-dimensional embedding.

For the baseline block used in these experiments:

| Component | Value |
| --- | --- |
| Base model | `ai-sage/Giga-Embeddings-instruct` |
| Latent slots | 512 |
| Cross-attention heads | 8 |
| Latent dimension | 2048 |
| MLP expansion | 4x |
| Pooling | Existing model pooling over transformed token-position hidden states |
| Final embedding dimension | 2048 |

Experiment 1b keeps this architecture unchanged and trains only the latent-attention module. This makes it a control for future architecture changes: if Experiments 2-4 improve, they should improve over Experiment 1b, not only over the released model.

## Evaluation Protocol

Primary benchmark:

| Item | Setting |
| --- | --- |
| Benchmark | ruMTEB v1.1 task set through `mteb` |
| Evaluation script | `scripts/run_rumteb_eval.py` |
| Summary script | `scripts/summarize_rumteb_results.py` |
| Batch size | 12 |
| Evaluation max length | 4096 |
| GPU | NVIDIA RTX 4500 Ada, 24 GB VRAM |
| Contamination manifest | `configs/training_manifests/open_ru_ablation_v1.json` |

The benchmark is split into:

- `clean`: tasks not known to overlap with the training manifest.
- `contaminated`: tasks whose source datasets are known or expected to overlap with training data.
- `all`: full task set, useful for completeness but not for deciding whether training data is fair.

Clean score is the main decision metric. If a retrained model beats the released model on clean ruMTEB by a meaningful margin, the result should be treated as suspicious until data leakage is checked again.

## Training Data

The first open-data training mix was generated into:

`data/contrastive/open_ru_train.jsonl`

Generation script:

```bash
HF_HOME=results/mteb_cache/hf_home \
HF_DATASETS_CACHE=results/mteb_cache/datasets \
HF_DATASETS_OFFLINE=1 \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python scripts/prepare_open_ru_contrastive.py \
  --offline \
  --per-source-limit 512 \
  --negatives-per-record 2
```

Generated size:

| Source | Records | Main use |
| --- | ---: | --- |
| `merionum/ru_paraphraser` | 512 | STS/paraphrase positives |
| `ai-forever/ru-stsbenchmark-sts` | 512 | STS positives and low-score negatives |
| `ai-forever/ru-scibench-grnti-classification` | 512 | Same-label/different-label scientific classification pairs |
| `ai-forever/ru-scibench-oecd-classification` | 512 | Same-label/different-label scientific classification pairs |
| `ai-forever/rubq-retrieval` | 512 | Query-document retrieval pairs |
| Total | 2560 | Mixed contrastive training |

Known contamination:

Several training sources are exact or near-exact ruMTEB task sources. This is intentional for the first control run only if contaminated tasks are excluded from clean reporting. The manifest marks these overlaps so clean evaluation excludes the affected tasks.

## Experiment 1: Original Latent-Memory Control

### Purpose

Experiment 1 establishes two baselines:

| Variant | Description |
| --- | --- |
| 1a | Released model, no retraining |
| 1b | Same architecture, retrained latent-attention block on the open-data mix |

The important comparison is 1a vs 1b on clean ruMTEB. Future architecture variants should be compared mainly against 1b because it controls for the training data and training script.

### Training Configuration

Config:

`configs/experiments/exp01b_retrain_latent_memory_open_ru.json`

Key settings:

| Setting | Value |
| --- | --- |
| Model | `ai-sage/Giga-Embeddings-instruct` |
| Trainable parameters | `model.latent_attention_model` only |
| LLM backbone | Frozen |
| Latent block | Continued from released weights |
| Reinitialized latent block | No |
| Training max length | 512 |
| Batch size | 12 |
| Learning rate | `1e-5` |
| Weight decay | `0.01` |
| Contrastive temperature | `0.02` |
| Max steps | 1000 |
| Save interval | 100 steps |
| Seed | 13 |

Training command:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python scripts/train_exp01b_latent_memory.py \
  --config configs/experiments/exp01b_retrain_latent_memory_open_ru.json
```

Training outputs:

| Artifact | Path |
| --- | --- |
| Latest checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru/latest.pt` |
| Latest metadata | `experiments/exp01_original_latent_memory/checkpoints/open_ru/latest.json` |
| Step checkpoints | `experiments/exp01_original_latent_memory/checkpoints/open_ru/step-*.pt` |

Training runtime:

| Metric | Value |
| --- | ---: |
| Steps | 1000 |
| Elapsed time | 41.25 minutes |
| Steps per second | 0.404 |
| Mean training loss | 0.834551 |
| Last training loss | 0.284354 |

Observed GPU behavior:

- Batch size 12 was stable.
- Training used about 19.7 GB VRAM plus desktop overhead at peak observation.
- GPU utilization was near 100% during training.
- Batch size 16 is not recommended for this model/evaluation setup on 24 GB VRAM, especially at `max_length=4096`.

### Evaluation

Clean ruMTEB evaluation command:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python scripts/run_rumteb_eval.py \
  --local-files-only \
  --batch-size 12 \
  --max-length 4096 \
  --latent-checkpoint experiments/exp01_original_latent_memory/checkpoints/open_ru/latest.pt \
  --training-manifest configs/training_manifests/open_ru_ablation_v1.json \
  --eval-scope clean
```

Summary command:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python scripts/summarize_rumteb_results.py \
  --results-dir results/rumteb/maxlen-4096-prompt-clean-open_ru_ablation_v1 \
  --training-manifest configs/training_manifests/open_ru_ablation_v1.json
```

Evaluation outputs:

| Artifact | Path |
| --- | --- |
| Clean ruMTEB result directory | `results/rumteb/maxlen-4096-prompt-clean-open_ru_ablation_v1` |
| Clean summary markdown | `results/rumteb/maxlen-4096-prompt-clean-open_ru_ablation_v1/summary.md` |
| Clean summary JSON | `results/rumteb/maxlen-4096-prompt-clean-open_ru_ablation_v1/summary.json` |

Runtime note:

The clean evaluation was not stuck, but `RiaNewsRetrievalHardNegatives.v2` dominated wall time at `max_length=4096`. The process stayed in running state, kept GPU utilization around 93-100%, and finally wrote:

- `RiaNewsRetrievalHardNegatives.v2.json` at 2026-05-09 23:02:07.
- `STS22.json` at 2026-05-09 23:03:11.

For future runs, RIA retrieval should be expected to behave like an overnight-style task when evaluating full context length. It should stay in the benchmark, but it should not be used as a quick smoke check.

### Clean ruMTEB Results

Aggregate comparison:

| Model | Clean average |
| --- | ---: |
| Released reference, prompt-aware, maxlen 4096 | 0.738606 |
| Experiment 1b retrained latent block | 0.737332 |
| Delta | -0.001274 |

Category comparison:

| Category | Released reference | Experiment 1b | Delta |
| --- | ---: | ---: | ---: |
| Classification | 0.732829 | 0.732364 | -0.000465 |
| Clustering | 0.727023 | 0.729729 | +0.002706 |
| Retrieval | 0.889040 | 0.887210 | -0.001830 |
| STS/NLI | 0.651749 | 0.639772 | -0.011977 |

Per-task comparison:

| Task | Released reference | Experiment 1b | Delta |
| --- | ---: | ---: | ---: |
| CEDRClassification | 0.652922 | 0.652232 | -0.000690 |
| GeoreviewClassification | 0.578418 | 0.578320 | -0.000098 |
| GeoreviewClusteringP2P | 0.727023 | 0.729729 | +0.002706 |
| HeadlineClassification | 0.894336 | 0.894434 | +0.000098 |
| InappropriatenessClassification | 0.787012 | 0.786768 | -0.000244 |
| KinopoiskClassification | 0.718333 | 0.717733 | -0.000600 |
| MassiveIntentClassification | 0.845744 | 0.844585 | -0.001159 |
| MassiveScenarioClassification | 0.910495 | 0.909930 | -0.000565 |
| RiaNewsRetrievalHardNegatives.v2 | 0.889040 | 0.887210 | -0.001830 |
| RuReviewsClassification | 0.767139 | 0.766650 | -0.000489 |
| STS22 | 0.651749 | 0.639772 | -0.011977 |
| SensitiveTopicsClassification | 0.441064 | 0.440625 | -0.000439 |

### Interpretation

Experiment 1b is slightly worse than the released model on clean ruMTEB:

```text
clean delta = -0.001274
```

This is a useful control result. It suggests that the open-data retraining setup does not create suspicious clean-benchmark gains. The result therefore does not currently indicate benchmark contamination in the clean split.

Most tasks changed only minimally. The main negative movement is `STS22`, with a drop of about 0.012. That suggests the mixed objective may be slightly damaging cross-lingual STS sensitivity, or that training the latent module alone can perturb semantic similarity geometry even when most classification tasks remain stable.

The only positive movement above noise is `GeoreviewClusteringP2P`, but the gain is small and does not compensate for STS and retrieval loss.

### Conclusion for Experiment 1

Experiment 1b is a valid fair-control checkpoint, but it is not an improvement over the released model. It should be used as the trained baseline for Experiments 2-4.

Current conclusion:

- Training only the released latent-attention block on the current open-data mix is stable.
- The training pipeline works with batch size 12 on 24 GB VRAM.
- Clean ruMTEB does not improve, so there is no immediate clean-split contamination alarm.
- The current data/objective mix is not enough to improve embeddings by itself.
- Future architecture changes must beat `0.737332` clean average under the same evaluation protocol to be considered promising.

## Implications for Experiments 2-4

Future experiments should keep the same contamination discipline:

- Train on the same or explicitly versioned training mix.
- Evaluate with the same `max_length=4096`, batch size 12, and prompt-aware wrapper.
- Report clean and contaminated scopes separately.
- Treat clean-score improvement over the released model as suspicious until leakage checks are repeated.
- Compare architecture gains primarily against Experiment 1b, not only against the released model.

Updated next steps after Experiment 2:

1. Keep Experiment 2 as a valid candidate, because it beats the fair Experiment 1b control.
2. Implement Experiment 3 and Experiment 4 with the same contamination discipline.
3. Add a retrieval profiling pass before or during Experiment 4, because full-context RIA retrieval is now the dominant evaluation cost.
4. Improve the training mix/objective for STS-style similarity before drawing strong conclusions from small architecture deltas.

## Hypotheses and Corrections for Experiments 3-4

Experiment 2 changed the interpretation of the original architecture. The original latent-attention block is better described as learned latent memory used by token-position hidden states, not as a module that emits 512 latent-slot embeddings. This correction matters for Experiments 3 and 4: future variants should preserve the model's expected token-hidden-state interface unless the pooling path is intentionally redesigned and evaluated as a larger architectural change.

Updated hypothesis for Experiment 3:

The useful part of iterative refinement may be repeated access to token-level evidence after an initial embedding estimate is formed. However, direct refinement from all token hidden states at `max_length=4096` is likely to be expensive. Experiment 3 should therefore be treated as a quality-first experiment, not a compute-efficient one. It should test whether iterative embedding updates improve STS, retrieval, and hard semantic relationships enough to justify the added cost. The implementation should refine the pooled embedding using backbone/token hidden states or transformed token hidden states, with a small residual scale, pre-update normalization, and shared weights across iterations. If it only matches Experiment 2 while costing more, it should not be promoted.

Updated hypothesis for Experiment 4:

Experiment 4 is now the most important quality/compute tradeoff candidate. Because the original model does not naturally expose compressed latent-slot outputs, "slot compression plus refinement" should be implemented carefully. The refinement context should be a bounded representation produced by the latent module, such as a learned summary/slot projection added explicitly for the experiment, or a compact pooled set derived from token-position hidden states. It should not assume that the original 512 learned latent vectors are already document-specific semantic slots; by themselves they are learned memory parameters, not input-dependent compressed document components. The goal is to create input-dependent compact state first, then refine the embedding from that compact state.

Practical corrections before implementing Experiments 3 and 4:

- Do not describe the original block as `tokens -> 512 latent slots -> embedding` in implementation docs; use `tokens -> latent-memory-conditioned token states -> pooling -> embedding`.
- For Experiment 3, record wall time separately for short clean tasks and RIA retrieval, because full-token iterative refinement may be prohibitively slow.
- For Experiment 4, make the compressed refinement state explicit and input-dependent; otherwise the experiment will not test the intended compression-and-reasoning hypothesis.
- Keep Experiment 1b as the fair training baseline and Experiment 2 as the first architectural baseline. Future experiments should beat `0.738266` clean average to improve over the best experimental checkpoint so far, and should beat `0.738606` before claiming improvement over the released model.
- STS remains the main quality weakness. A small average gain that comes only from retrieval/classification but leaves `STS22` far below the released model should be interpreted cautiously.

## Experiment 2: Hierarchical Latent Compression

Status: completed first open-data run.

### Purpose

Experiment 2 tests whether a two-stage latent-memory block can improve semantic compression while staying close to the original model contract. The initial conceptual plan was `tokens -> latent_512 -> latent_128 -> embedding`, but the implementation was adjusted after inspecting the actual model: the original latent module preserves token positions. Therefore, the implemented hierarchical variant applies two latent-memory stages over token-position hidden states, then uses the model's existing pooling path.

Implemented architecture:

```text
tokens
  -> stage1 learned latent memory, 512 slots
  -> token-position hidden states
  -> stage2 learned latent memory, 128 slots
  -> token-position hidden states
  -> existing pooling path
  -> embedding
```

This is not a literal sequence-length compression to 128 hidden states. It is a hierarchical latent-memory transformation that preserves the original model interface.

### Implementation

Implemented files:

| File | Purpose |
| --- | --- |
| `scripts/latent_experiment_modules.py` | Adds `HierarchicalLatentAttentionModel`, installer, and checkpoint loader support. |
| `scripts/train_exp01b_latent_memory.py` | Adds `latent_architecture` config support and saves architecture metadata. |
| `scripts/giga_model_utils.py` | Loads both original and hierarchical latent checkpoints. |
| `scripts/run_rumteb_eval.py` | Adds `--run-name` so experiment results do not overwrite each other. |

Configuration files:

| Config | Purpose |
| --- | --- |
| `configs/experiments/exp02_hierarchical_latent_smoke.json` | Two-step smoke training run. |
| `configs/experiments/exp02_hierarchical_latent_open_ru.json` | Full open-data training run. |

Smoke checks passed:

- Two-step training completed.
- Hierarchical checkpoint reloaded.
- Forward pass produced a `(1, 2048)` embedding.

### Training Configuration

Config:

`configs/experiments/exp02_hierarchical_latent_open_ru.json`

Key settings:

| Setting | Value |
| --- | --- |
| Model | `ai-sage/Giga-Embeddings-instruct` |
| Trainable parameters | `model.latent_attention_model` only |
| LLM backbone | Frozen |
| Stage 1 latent memory | 512 slots, initialized from released latent block |
| Stage 2 latent memory | 128 slots, newly initialized |
| Training max length | 512 |
| Batch size | 8 |
| Learning rate | `1e-5` |
| Weight decay | `0.01` |
| Contrastive temperature | `0.02` |
| Max steps | 1000 |
| Save interval | 100 steps |
| Seed | 13 |

Training command:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python scripts/train_exp01b_latent_memory.py \
  --config configs/experiments/exp02_hierarchical_latent_open_ru.json
```

Training outputs:

| Artifact | Path |
| --- | --- |
| Latest checkpoint | `experiments/exp02_hierarchical_latent/checkpoints/open_ru/latest.pt` |
| Latest metadata | `experiments/exp02_hierarchical_latent/checkpoints/open_ru/latest.json` |
| Step checkpoints | `experiments/exp02_hierarchical_latent/checkpoints/open_ru/step-*.pt` |

Training runtime:

| Metric | Value |
| --- | ---: |
| Steps | 1000 |
| Elapsed time | 28.45 minutes |
| Steps per second | 0.586 |
| Mean training loss | 0.718319 |
| Last training loss | 0.192240 |

Observed GPU behavior:

- Batch size 8 was stable.
- Training used about 18.5 GB VRAM plus desktop overhead at peak observation.
- Batch size 12 was not used for training because Experiment 2 adds an extra attention/FFN stage and the margin over the 24 GB card would be too small.
- Evaluation with batch size 12 was stable at `max_length=4096`, using about 15 GB VRAM during RIA retrieval.

### Evaluation

Clean ruMTEB evaluation command:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python scripts/run_rumteb_eval.py \
  --local-files-only \
  --batch-size 12 \
  --max-length 4096 \
  --latent-checkpoint experiments/exp02_hierarchical_latent/checkpoints/open_ru/latest.pt \
  --training-manifest configs/training_manifests/open_ru_ablation_v1.json \
  --eval-scope clean \
  --run-name maxlen-4096-prompt-clean-open_ru_ablation_v1-exp02_hierarchical_latent
```

Summary command:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-sync python scripts/summarize_rumteb_results.py \
  --results-dir results/rumteb/maxlen-4096-prompt-clean-open_ru_ablation_v1-exp02_hierarchical_latent \
  --training-manifest configs/training_manifests/open_ru_ablation_v1.json
```

Evaluation outputs:

| Artifact | Path |
| --- | --- |
| Clean ruMTEB result directory | `results/rumteb/maxlen-4096-prompt-clean-open_ru_ablation_v1-exp02_hierarchical_latent` |
| Clean summary markdown | `results/rumteb/maxlen-4096-prompt-clean-open_ru_ablation_v1-exp02_hierarchical_latent/summary.md` |
| Clean summary JSON | `results/rumteb/maxlen-4096-prompt-clean-open_ru_ablation_v1-exp02_hierarchical_latent/summary.json` |

Runtime note:

Most non-retrieval tasks completed quickly between 03:24 and 03:38 MSK on 2026-05-10. `RiaNewsRetrievalHardNegatives.v2` dominated the run and wrote its result at 06:28:46. `STS22` then completed at 06:29:55.

The RIA retrieval stage kept GPU utilization around 93-100% and used about 15 GB VRAM. It was not stuck, but Experiment 2 made the full-context retrieval wall time substantially worse than Experiment 1.

### Clean ruMTEB Results

Aggregate comparison:

| Model | Clean average | Delta vs Exp 1b | Delta vs released |
| --- | ---: | ---: | ---: |
| Released reference, prompt-aware, maxlen 4096 | 0.738606 | +0.001274 | 0.000000 |
| Experiment 1b retrained original latent block | 0.737332 | 0.000000 | -0.001274 |
| Experiment 2 hierarchical latent block | 0.738266 | +0.000934 | -0.000340 |

Category comparison:

| Category | Released reference | Experiment 1b | Experiment 2 | Exp 2 vs Exp 1b | Exp 2 vs released |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classification | 0.732829 | 0.732364 | 0.733123 | +0.000759 | +0.000294 |
| Clustering | 0.727023 | 0.729729 | 0.730907 | +0.001178 | +0.003884 |
| Retrieval | 0.889040 | 0.887210 | 0.889220 | +0.002010 | +0.000180 |
| STS/NLI | 0.651749 | 0.639772 | 0.640964 | +0.001192 | -0.010785 |

Per-task comparison:

| Task | Released reference | Experiment 1b | Experiment 2 | Exp 2 vs Exp 1b |
| --- | ---: | ---: | ---: | ---: |
| CEDRClassification | 0.652922 | 0.652232 | 0.653135 | +0.000903 |
| GeoreviewClassification | 0.578418 | 0.578320 | 0.579297 | +0.000977 |
| GeoreviewClusteringP2P | 0.727023 | 0.729729 | 0.730907 | +0.001178 |
| HeadlineClassification | 0.894336 | 0.894434 | 0.894043 | -0.000391 |
| InappropriatenessClassification | 0.787012 | 0.786768 | 0.786914 | +0.000146 |
| KinopoiskClassification | 0.718333 | 0.717733 | 0.719000 | +0.001267 |
| MassiveIntentClassification | 0.845744 | 0.844585 | 0.846044 | +0.001459 |
| MassiveScenarioClassification | 0.910495 | 0.909930 | 0.910246 | +0.000316 |
| RiaNewsRetrievalHardNegatives.v2 | 0.889040 | 0.887210 | 0.889220 | +0.002010 |
| RuReviewsClassification | 0.767139 | 0.766650 | 0.768457 | +0.001807 |
| STS22 | 0.651749 | 0.639772 | 0.640964 | +0.001192 |
| SensitiveTopicsClassification | 0.441064 | 0.440625 | 0.440967 | +0.000342 |

### Interpretation

Experiment 2 improves over the fair retrained baseline:

```text
Exp 2 clean delta vs Exp 1b = +0.000934
```

This is directionally positive across almost every clean task, with the largest improvements on retrieval, clustering, and several classification tasks. The result supports the idea that an extra latent-memory stage can recover some of the quality lost by retraining the original block on the current open-data mix.

However, Experiment 2 is still slightly below the released model:

```text
Exp 2 clean delta vs released = -0.000340
```

The remaining gap is mainly `STS22`. Exp 2 improves STS over Exp 1b, but it does not recover the released model's STS score. This suggests that the architecture is not obviously harmful, but the current training mix/objective still under-serves STS-style similarity geometry.

There is no clean-split contamination alarm. Exp 2 does not outperform the released model on the clean average, and the known contaminated datasets remain excluded by the manifest.

### Conclusion for Experiment 2

Experiment 2 is a promising but not decisive architectural result:

- It beats the fair Experiment 1b control by about `+0.0009`.
- It nearly matches the released model clean average, but does not beat it.
- It improves retrieval and clustering clean scores versus both Exp 1b and the released reference.
- It carries a major full-context evaluation-time cost, especially on RIA retrieval.

The practical conclusion is to keep Experiment 2 as a valid candidate, but not to treat it as the final design. The next experiments should focus on getting a larger quality gain per added compute. Experiment 4 is still especially relevant because it may provide iterative refinement over a bounded latent representation rather than adding expensive full-token transformations.

## Experiment 1b STS-Focused Rerun

Status: completed.

This rerun keeps the original latent-attention architecture and retrains only the latent block, but changes the training mix/objective to address the STS weakness observed in Experiment 1b and Experiment 2.

Changes from the first Experiment 1b run:

- Added mixed-objective training records:
  - contrastive query/positive/negative records;
  - graded `pair_score` records trained with an MSE loss over normalized embedding similarity.
- Added `deepvk/ru-HNP` as a high-quality Russian hard-negative paraphrase source.
- Preserved open-source contamination controls through `configs/training_manifests/open_ru_sts_v2.json`.
- Kept the model context cap at `max_length = 4096`.

Training configuration:

| Item | Value |
| --- | --- |
| Config | `configs/experiments/exp01b_retrain_latent_memory_sts_v2.json` |
| Training data | `data/contrastive/open_ru_sts_v2_train.jsonl` |
| Records | 31,220 |
| Batch size | 12 |
| Max length | 4096 |
| Max steps | 1000 |
| Trainable parameters | original latent block only |
| Checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v2/latest.pt` |
| Training time | about 13.2 minutes |

Training data composition:

| Source | Records |
| --- | ---: |
| `deepvk/ru-HNP` | 23,998 |
| `merionum/ru_paraphraser` | 3,686 |
| `ai-forever/ru-stsbenchmark-sts` | 2,000 |
| `ai-forever/ru-scibench-grnti-classification` | 512 |
| `ai-forever/ru-scibench-oecd-classification` | 512 |
| `ai-forever/rubq-retrieval` | 512 |

Objective composition:

| Objective | Records |
| --- | ---: |
| `pair_score` | 19,999 |
| `contrastive` | 11,221 |

Evaluation:

| Item | Value |
| --- | --- |
| Results directory | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v2-exp01b` |
| Summary | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v2-exp01b/summary.md` |
| Evaluation scope | clean tasks only |
| Batch size | 12 |
| Max length | 4096 |
| Completed task count | 16 |

Full clean summary for this run:

| Metric | Score |
| --- | ---: |
| Clean average over 16 clean tasks | 0.740057 |
| Classification average | 0.732611 |
| Clustering average | 0.726697 |
| Reranking average | 0.738005 |
| Retrieval average | 0.810895 |
| STS/NLI average | 0.711458 |
| STS22 | 0.631344 |

The 16-task average is not directly comparable to the earlier 12-task Exp 1b/Exp 2 summaries because this run also completed `MIRACLReranking`, `MIRACLRetrievalHardNegatives.v2`, `RuBQReranking`, and `TERRa`.

Fair 12-task overlap comparison:

| Model | Overlap clean average | STS22 | RiaNews retrieval |
| --- | ---: | ---: | ---: |
| Experiment 1b, original open-RU mix | 0.737332 | 0.639772 | 0.887210 |
| Experiment 2, hierarchical open-RU mix | 0.738266 | 0.640964 | 0.889220 |
| Experiment 1b STS-focused rerun | 0.736289 | 0.631344 | 0.883920 |

Delta of STS-focused rerun versus original Experiment 1b on the shared 12 tasks:

| Task | Delta |
| --- | ---: |
| CEDRClassification | +0.000531 |
| GeoreviewClassification | -0.000146 |
| GeoreviewClusteringP2P | -0.003032 |
| HeadlineClassification | +0.000048 |
| InappropriatenessClassification | +0.000390 |
| KinopoiskClassification | +0.000734 |
| MassiveIntentClassification | +0.000905 |
| MassiveScenarioClassification | +0.000497 |
| RiaNewsRetrievalHardNegatives.v2 | -0.003290 |
| RuReviewsClassification | -0.000293 |
| STS22 | -0.008428 |
| SensitiveTopicsClassification | -0.000439 |

### Interpretation of STS-Focused Rerun

The STS-focused rerun did not fix the STS gap. Despite adding `ru-HNP`, RuSTSBenchmark, and ParaPhraser-style pair supervision, `STS22` dropped from `0.639772` to `0.631344` relative to the first fair Experiment 1b run. The 12-task overlap average also dropped from `0.737332` to `0.736289`.

This is evidence against the current STS-v2 recipe, not against the general idea of STS-focused fine-tuning. The likely issues are:

- The simple MSE pair-score objective may be miscalibrated for the model's embedding geometry.
- The training mix is dominated by synthetic ru-HNP pairs, which may improve paraphrase discrimination without improving cross-lingual STS22-style graded similarity.
- ParaPhraser class labels are too coarse for calibrated STS geometry, especially the neutral class mapping.
- Training only the latent block may not be enough to relearn the released model's proprietary STS alignment.
- The STS-v2 run may trade off retrieval/clustering structure for pair-level similarity calibration without actually improving STS22.

Conclusion: do not use this STS-v2 training recipe as the default baseline for Experiments 3 and 4. Keep the first Experiment 1b open-RU training mix as the fair baseline unless a better STS objective is designed. If we revisit STS tuning, prefer a smaller controlled ablation:

- lower `deepvk/ru-HNP` weight;
- separate contrastive and pair-score schedules rather than summing them blindly in mixed batches;
- use ranking/listwise STS loss or cosine-margin losses instead of raw MSE;
- validate on held-out non-ruMTEB STS/paraphrase diagnostics before running full ruMTEB;
- keep known contaminated STS datasets out of clean evaluation and report contaminated scores separately only as sanity checks.

## Experiment 1b STS-v3 Controlled Rerun

Status: STS22-only evaluation completed. Full clean ruMTEB not run yet.

Purpose: test whether the STS-v2 failure was caused by the training recipe rather than the data sources themselves.

Changes from STS-v2:

- Reduced `deepvk/ru-HNP` from 23,998 records to 3,000 records.
- Reduced negatives per record from 5 to 3.
- Used explicit staged training:
  - stage 1: soft/graded `pair_score` records only;
  - stage 2: hard contrastive records only.
- Kept `max_length = 4096`.
- Kept the original latent-attention architecture.

Important correction: STS-v2 did not train soft negatives before hard negatives. It shuffled all objectives together. STS-v3 is the first run with an explicit soft-then-hard schedule.

Training data:

| Source | Records |
| --- | ---: |
| `deepvk/ru-HNP` | 3,000 |
| `merionum/ru_paraphraser` | 3,000 |
| `ai-forever/ru-stsbenchmark-sts` | 1,500 |
| `ai-forever/ru-scibench-grnti-classification` | 384 |
| `ai-forever/ru-scibench-oecd-classification` | 384 |
| `ai-forever/rubq-retrieval` | 384 |

Objective composition:

| Objective | Records |
| --- | ---: |
| `pair_score` | 5,000 |
| `contrastive` | 3,652 |

Training path:

| Phase | Config | Notes |
| --- | --- | --- |
| Initial staged run | `configs/experiments/exp01b_retrain_latent_memory_sts_v3.json` | OOM during hard stage at batch 12; checkpoint saved at step 400. |
| Resume hard stage | `configs/experiments/exp01b_retrain_latent_memory_sts_v3_resume_hard.json` | Continued from step 400 with batch 8; rare long-batch OOM after checkpoint step 300. |
| Final hard-stage finish | `configs/experiments/exp01b_retrain_latent_memory_sts_v3_resume_hard_b4.json` | Continued with batch 4 and completed. |

Final checkpoint:

```text
experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v3_final/latest.pt
```

The batch reductions were only for the hard-negative contrastive phase. Context length was not reduced.

Evaluation:

| Item | Value |
| --- | --- |
| Results directory | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v3-exp01b-sts22` |
| Evaluation scope | clean |
| Task | `STS22` |
| Score | 0.649605 |

STS22 comparison:

| Model / run | STS22 |
| --- | ---: |
| Released reference | 0.651749 |
| Experiment 1b original open-RU mix | 0.639772 |
| Experiment 2 hierarchical open-RU mix | 0.640964 |
| Experiment 1b STS-v2 | 0.631344 |
| Experiment 1b STS-v3 | 0.649605 |

### Interpretation of STS-v3

STS-v3 is a positive training result. It recovers most of the released model's STS22 score while keeping clean-evaluation contamination controls. The key difference is not simply adding STS/paraphrase data, because STS-v2 already did that and got worse. The likely useful changes are:

- much lower ru-HNP dominance;
- soft/graded similarity training before hard-negative contrastive training;
- less aggressive hard-negative fanout;
- avoiding mixed batches where MSE pair calibration and hard contrastive pressure compete in the same step.

This does not yet prove STS-v3 is better overall. We only evaluated STS22. It may still trade off retrieval or classification quality, so the next validation step should be a 12-task clean overlap run before adopting STS-v3 as the baseline for later architecture experiments.

## Experiment 1b STS-v9/v10 Retention Ablations

Status: STS22-only evaluation completed for both runs. A 12-task clean overlap evaluation was also completed for STS-v9.

Purpose: isolate which retention technique is worth keeping after the STS-v6 recipe. Earlier retention experiments mixed several mechanisms and regressed STS22, so these ablations split them:

| Run | Recipe | Regularization | Rehearsal | Distillation |
| --- | --- | --- | --- | --- |
| STS-v6 | DeepVK upweighted 3x baseline | No | No | No |
| STS-v7 | Combined retention | Yes | Yes | Yes |
| STS-v8 | No distillation | Yes | Yes | No |
| STS-v9 | Anchor-only | Yes | No | No |
| STS-v10 | Rehearsal-only | No | Yes | No |

Training configuration:

| Run | Config | Checkpoint | Elapsed time |
| --- | --- | --- | ---: |
| STS-v9 | `configs/experiments/exp01b_retrain_latent_memory_sts_v9_anchor_only.json` | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v9_anchor_only/latest.pt` | 20.72 min |
| STS-v10 | `configs/experiments/exp01b_retrain_latent_memory_sts_v10_rehearsal_only.json` | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v10_rehearsal_only/latest.pt` | 42.20 min |

Both runs used the STS-v6 training data and staged schedule:

- stage 1: 450 soft `pair_score` steps, batch size 12;
- stage 2: 1800 hard contrastive steps, batch size 4;
- `max_length = 4096`;
- frozen LLM backbone;
- trainable original latent-attention block only.

STS-v9 added only latent-parameter anchoring with `parameter_anchor_weight = 10.0`. STS-v10 added only light rehearsal from `data/contrastive/open_ru_train.jsonl` with `rehearsal_batch_size = 4` and `rehearsal_loss_weight = 0.15`.

Evaluation:

| Run | Results directory | STS22 |
| --- | --- | ---: |
| Released reference | `results/rumteb/maxlen-4096-prompt` | 0.651749 |
| STS-v5, DeepVK 1.5x | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v5-deepvk-upweighted-1p5x-sts22` | 0.652481 |
| STS-v6, DeepVK 3x | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v6-deepvk-upweighted-3x-sts22` | 0.660403 |
| STS-v7, distillation + rehearsal + anchor | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v7-retention-sts22` | 0.647608 |
| STS-v8, rehearsal + anchor | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v8-rehearsal-anchor-sts22` | 0.636332 |
| STS-v9, anchor-only | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v9-anchor-only-sts22` | 0.660412 |
| STS-v10, rehearsal-only | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v10-rehearsal-only-sts22` | 0.637000 |

STS-v9 12-task clean overlap evaluation:

| Item | Value |
| --- | --- |
| Results directory | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v9-anchor-only-12task-overlap` |
| Summary JSON | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v9-anchor-only-12task-overlap/summary.json` |
| Summary markdown | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v9-anchor-only-12task-overlap/summary.md` |
| Task count | 12 |
| Batch size | 12 |
| Max length | 4096 |
| Runtime note | RIA retrieval dominated the run and took about 3 hours of silent GPU-bound computation. |

12-task clean overlap comparison:

| Model | Clean average | Classification | Clustering | Retrieval | STS/NLI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Released reference | 0.738606 | 0.732829 | 0.727023 | 0.889040 | 0.651749 |
| Experiment 1b original open-RU mix | 0.737332 | 0.732364 | 0.729729 | 0.887210 | 0.639772 |
| Experiment 2 hierarchical open-RU mix | 0.738266 | 0.733123 | 0.730907 | 0.889220 | 0.640964 |
| STS-v9 anchor-only | 0.734551 | 0.730556 | 0.705301 | 0.873900 | 0.660412 |

Per-task STS-v9 comparison against Experiment 2:

| Task | Experiment 2 | STS-v9 | Delta |
| --- | ---: | ---: | ---: |
| CEDRClassification | 0.653135 | 0.647503 | -0.005632 |
| GeoreviewClassification | 0.579297 | 0.570117 | -0.009180 |
| GeoreviewClusteringP2P | 0.730907 | 0.705301 | -0.025606 |
| HeadlineClassification | 0.894043 | 0.892139 | -0.001904 |
| InappropriatenessClassification | 0.786914 | 0.778369 | -0.008545 |
| KinopoiskClassification | 0.719000 | 0.719933 | +0.000933 |
| MassiveIntentClassification | 0.846045 | 0.851336 | +0.005292 |
| MassiveScenarioClassification | 0.910246 | 0.911847 | +0.001601 |
| RiaNewsRetrievalHardNegatives.v2 | 0.889220 | 0.873900 | -0.015320 |
| RuReviewsClassification | 0.768457 | 0.766602 | -0.001855 |
| STS22 | 0.640964 | 0.660412 | +0.019448 |
| SensitiveTopicsClassification | 0.440967 | 0.437158 | -0.003809 |

Semantic diagnostics:

| Run | Passed cases | Mean margin |
| --- | ---: | ---: |
| Released reference | 4/8 | -0.016135 |
| STS-v5 | 4/8 | 0.009252 |
| STS-v6 | 4/8 | 0.013085 |
| STS-v7 | 4/8 | -0.000333 |
| STS-v8 | 4/8 | 0.005682 |
| STS-v9 | 4/8 | 0.012832 |
| STS-v10 | 4/8 | 0.005178 |

### Interpretation of STS-v9/v10

STS-v9 is effectively tied with STS-v6 on STS22:

```text
STS-v9 - STS-v6 = +0.000009
```

This means parameter anchoring did not harm the useful STS-v6 adaptation and may be worth keeping as a low-risk regularizer. The gain is far too small to call an improvement, but the main result is that anchor-only regularization does not reproduce the STS-v7/v8 regression.

STS-v10 regressed strongly:

```text
STS-v10 - STS-v6 = -0.023403
```

Light rehearsal alone is therefore not a good retention strategy in the current form. It roughly doubles training time and appears to pull the latent block back toward the earlier mixed open-RU objective, undoing much of the STS-v6 gain. The same pattern explains why STS-v8 was weak: rehearsal is likely the harmful component, while anchoring alone is not.

The 12-task overlap changes the conclusion. STS-v9 is useful as an STS-focused checkpoint, but it is not a general replacement baseline. It improves `STS22` substantially over Experiment 2, but the gain is paid for by large losses on clustering and retrieval, especially `GeoreviewClusteringP2P` and `RiaNewsRetrievalHardNegatives.v2`.

Conclusion: keep STS-v6/STS-v9 as STS-focused ablations, not as the default architecture baseline. For general latent-attention architecture experiments, Experiment 2 remains the stronger open-data checkpoint on the 12-task clean overlap. Anchor regularization is still acceptable for stability, but the STS-v6/v9 data recipe is too STS-skewed unless we add a balancing schedule or stronger retrieval/clustering preservation. Do not use the current rehearsal-only recipe.

## Experiment 1b STS-v11 Balanced Curriculum

Status: training completed; quick diagnostics and STS22 evaluation completed. Full 12-task clean overlap was not run because STS22 regressed.

Purpose: test whether a stage-level curriculum can combine the broad Experiment 1b data with the STS-v6/v9 data without the per-step rehearsal loss that hurt STS-v10.

Training configuration:

| Item | Value |
| --- | --- |
| Config | `configs/experiments/exp01b_retrain_latent_memory_sts_v11_balanced_curriculum.json` |
| Manifest | `configs/training_manifests/open_ru_sts_v11.json` |
| Checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v11_balanced_curriculum/latest.pt` |
| Max length | 4096 |
| Trainable parameters | original latent-attention block only |
| LLM backbone | frozen |
| Parameter anchor | `parameter_anchor_weight = 10.0` |
| Distillation | no |
| Per-step rehearsal | no |
| Training time | 29.47 min |

Final schedule:

| Stage | Data | Objective | Steps | Batch |
| --- | --- | --- | ---: | ---: |
| broad warmup | `data/contrastive/open_ru_train.jsonl` | contrastive | 500 | 4 |
| soft STS | `data/contrastive/open_ru_sts_v6_train.jsonl` | pair_score | 450 | 12 |
| hard STS | `data/contrastive/open_ru_sts_v6_train.jsonl` | contrastive | 1500 | 4 |
| broad recovery | `data/contrastive/open_ru_train.jsonl` | contrastive | 600 | 4 |

The initial attempt used batch 12 for the broad stages and failed with CUDA OOM. The broad records contain multiple long candidates per record, so their effective encoded batch is much larger than the nominal batch size. The successful run capped both broad stages to batch 4 while keeping the 4096-token context.

Quick evaluation:

| Run | STS22 | Diagnostics passed | Diagnostic mean margin |
| --- | ---: | ---: | ---: |
| Released reference | 0.651749 | 4/8 | -0.016135 |
| Experiment 2 | 0.640964 | not run in this table | not run in this table |
| STS-v9 anchor-only | 0.660412 | 4/8 | 0.012832 |
| STS-v11 balanced curriculum | 0.636188 | 4/8 | -0.014163 |

### Interpretation of STS-v11

The stage-level broad warmup/recovery idea did not work in this form. It likely over-corrected toward the broad contrastive geometry and erased the STS-v6/v9 gain. This is especially visible in `STS22`, where v11 is below the released model, Experiment 2, and STS-v9.

The broad recovery stage may still be conceptually useful, but it needs to be much weaker or more targeted. The next correction should not repeat this exact schedule. Better options:

- start from the STS-v9 checkpoint and apply a short low-LR broad recovery stage only;
- reduce broad recovery to 100-200 steps instead of 600;
- use retrieval/classification batches only for the recovery stage, excluding coarse paraphrase/STS overlap;
- keep a small held-out STS22-like validation proxy or quick STS22 check after candidate recovery lengths;
- consider freezing more of the latent block during recovery if we need to preserve STS geometry.

Conclusion: do not promote STS-v11. Keep the checkpoint only as evidence that naive stage-level broad recovery can cause catastrophic STS forgetting.

## Experiment 1b STS-v12 Targeted Recovery

Status: completed. The 100-step, 200-step, 800-step, 1600-step targeted one-pass, and all-1b-source one-pass recovery variants were trained from STS-v9. Quick STS22 was run for all variants; the 200-step, 800-step, 1600-step targeted one-pass, and all-1b-source one-pass variants were also evaluated on the 12-task clean overlap.

Purpose: test a weaker recovery strategy after STS-v11 failed. Instead of training broad stages before and after STS from the released model, STS-v12 starts from the strong STS-v9 checkpoint and applies only a short, low-LR recovery stage on retrieval/classification/clustering sources.

Training configuration:

| Item | Value |
| --- | --- |
| Initial checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v9_anchor_only/latest.pt` |
| Recovery data | `data/contrastive/open_ru_train.jsonl` |
| Recovery sources | `ai-forever/rubq-retrieval`, `ai-forever/ru-scibench-grnti-classification`, `ai-forever/ru-scibench-oecd-classification` |
| Excluded from recovery | `merionum/ru_paraphraser`, `ai-forever/ru-stsbenchmark-sts` |
| Objective | contrastive |
| Max length | 4096 |
| Batch size | 4 |
| Learning rate | `2e-6` |
| Parameter anchor | `parameter_anchor_weight = 10.0` |
| LLM backbone | frozen |

Artifacts:

| Variant | Config | Checkpoint | Training time |
| --- | --- | --- | ---: |
| STS-v12 100-step | `configs/experiments/exp01b_retrain_latent_memory_sts_v12_v9_targeted_recovery_100.json` | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v12_v9_targeted_recovery_100/latest.pt` | 1.35 min |
| STS-v12 200-step | `configs/experiments/exp01b_retrain_latent_memory_sts_v12_v9_targeted_recovery_200.json` | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v12_v9_targeted_recovery_200/latest.pt` | 2.61 min |
| STS-v12 800-step | `configs/experiments/exp01b_retrain_latent_memory_sts_v12_v9_targeted_recovery_800.json` | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v12_v9_targeted_recovery_800/latest.pt` | 10.23 min |
| STS-v12 1600-step one-pass | `configs/experiments/exp01b_retrain_latent_memory_sts_v12_v9_targeted_recovery_1600_1pass.json` | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v12_v9_targeted_recovery_1600_1pass/latest.pt` | 20.41 min |
| STS-v12 all-1b-source one-pass | `configs/experiments/exp01b_retrain_latent_memory_sts_v12_v9_all_1b_1pass.json` | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v12_v9_all_1b_1pass/latest.pt` | 26.52 min |

Quick checks:

| Run | STS22 | Diagnostics passed | Diagnostic mean margin |
| --- | ---: | ---: | ---: |
| STS-v9 | 0.660412 | 4/8 | 0.012832 |
| STS-v12 100-step | 0.660701 | 4/8 | 0.012616 |
| STS-v12 200-step | 0.660455 | 4/8 | 0.012499 |
| STS-v12 800-step | 0.659498 | 4/8 | 0.011574 |
| STS-v12 1600-step one-pass | 0.660815 | 4/8 | 0.011114 |
| STS-v12 all-1b-source one-pass | 0.660338 | 4/8 | 0.011186 |

The 200-step variant was initially selected for the 12-task overlap because it had slightly more recovery pressure while preserving STS22. The later 800-step variant was run to test whether additional targeted recovery can regain more broad-task quality without fully erasing the STS gain.

The 1600-step one-pass variant uses a regenerated recovery JSONL with `2134` examples per source. The selected recovery sources contain `6402` records total, so `1600` steps at batch size `4` consume `6400` examples, approximately one data pass. This avoids the repeated exposure of the earlier 512-per-source recovery set.

The all-1b-source one-pass variant uses the same regenerated file, but includes all five sources: `ai-forever/rubq-retrieval`, `ai-forever/ru-scibench-grnti-classification`, `ai-forever/ru-scibench-oecd-classification`, `merionum/ru_paraphraser`, and `ai-forever/ru-stsbenchmark-sts`. The file contains `9353` records total, so the run used `2339` steps at batch size `4`, approximately one full pass over all available 1b-style recovery data.

12-task clean overlap comparison:

| Model | Clean average | Classification | Clustering | Retrieval | STS/NLI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Released reference | 0.738606 | 0.732829 | 0.727023 | 0.889040 | 0.651749 |
| Experiment 1b original open-RU mix | 0.737332 | 0.732364 | 0.729729 | 0.887210 | 0.639772 |
| Experiment 2 hierarchical open-RU mix | 0.738266 | 0.733123 | 0.730907 | 0.889220 | 0.640964 |
| STS-v9 anchor-only | 0.734551 | 0.730556 | 0.705301 | 0.873900 | 0.660412 |
| STS-v12 200-step targeted recovery | 0.734806 | 0.730564 | 0.707503 | 0.874640 | 0.660455 |
| STS-v12 800-step targeted recovery | 0.734917 | 0.730901 | 0.706923 | 0.874470 | 0.659498 |
| STS-v12 1600-step one-pass targeted recovery | 0.735696 | 0.731307 | 0.711032 | 0.874750 | 0.660815 |
| STS-v12 all-1b-source one-pass | 0.735488 | 0.731272 | 0.709667 | 0.874400 | 0.660338 |

Per-task STS-v12 200-step comparison:

| Task | STS-v9 | STS-v12 200-step | Delta vs STS-v9 | Delta vs Exp 2 |
| --- | ---: | ---: | ---: | ---: |
| CEDRClassification | 0.647503 | 0.647396 | -0.000107 | -0.005739 |
| GeoreviewClassification | 0.570117 | 0.569824 | -0.000293 | -0.009473 |
| GeoreviewClusteringP2P | 0.705301 | 0.707503 | +0.002202 | -0.023404 |
| HeadlineClassification | 0.892139 | 0.892139 | 0.000000 | -0.001904 |
| InappropriatenessClassification | 0.778369 | 0.778369 | 0.000000 | -0.008545 |
| KinopoiskClassification | 0.719933 | 0.719933 | 0.000000 | +0.000933 |
| MassiveIntentClassification | 0.851336 | 0.851427 | +0.000090 | +0.005382 |
| MassiveScenarioClassification | 0.911847 | 0.911888 | +0.000041 | +0.001642 |
| RiaNewsRetrievalHardNegatives.v2 | 0.873900 | 0.874640 | +0.000740 | -0.014580 |
| RuReviewsClassification | 0.766602 | 0.766797 | +0.000195 | -0.001660 |
| STS22 | 0.660412 | 0.660455 | +0.000043 | +0.019491 |
| SensitiveTopicsClassification | 0.437158 | 0.437305 | +0.000147 | -0.003662 |

Per-task STS-v12 800-step comparison:

| Task | STS-v12 200-step | STS-v12 800-step | Delta vs 200-step | Delta vs Exp 2 |
| --- | ---: | ---: | ---: | ---: |
| CEDRClassification | 0.647396 | 0.648831 | +0.001435 | -0.004304 |
| GeoreviewClassification | 0.569824 | 0.569971 | +0.000147 | -0.009326 |
| GeoreviewClusteringP2P | 0.707503 | 0.706923 | -0.000580 | -0.023984 |
| HeadlineClassification | 0.892139 | 0.892139 | +0.000000 | -0.001904 |
| InappropriatenessClassification | 0.778369 | 0.778906 | +0.000537 | -0.008008 |
| KinopoiskClassification | 0.719933 | 0.720267 | +0.000334 | +0.001267 |
| MassiveIntentClassification | 0.851427 | 0.851508 | +0.000081 | +0.005463 |
| MassiveScenarioClassification | 0.911888 | 0.911798 | -0.000089 | +0.001553 |
| RiaNewsRetrievalHardNegatives.v2 | 0.874640 | 0.874470 | -0.000170 | -0.014750 |
| RuReviewsClassification | 0.766797 | 0.767188 | +0.000391 | -0.001269 |
| STS22 | 0.660455 | 0.659498 | -0.000957 | +0.018534 |
| SensitiveTopicsClassification | 0.437305 | 0.437500 | +0.000195 | -0.003467 |

Per-task STS-v12 1600-step one-pass comparison:

| Task | STS-v12 800-step | STS-v12 1600-step one-pass | Delta vs 800-step | Delta vs Exp 2 |
| --- | ---: | ---: | ---: | ---: |
| CEDRClassification | 0.648831 | 0.650000 | +0.001169 | -0.003135 |
| GeoreviewClassification | 0.569971 | 0.570996 | +0.001025 | -0.008301 |
| GeoreviewClusteringP2P | 0.706923 | 0.711032 | +0.004109 | -0.019875 |
| HeadlineClassification | 0.892139 | 0.892236 | +0.000097 | -0.001807 |
| InappropriatenessClassification | 0.778906 | 0.778467 | -0.000439 | -0.008447 |
| KinopoiskClassification | 0.720267 | 0.721000 | +0.000733 | +0.002000 |
| MassiveIntentClassification | 0.851508 | 0.851382 | -0.000126 | +0.005338 |
| MassiveScenarioClassification | 0.911798 | 0.911673 | -0.000126 | +0.001427 |
| RiaNewsRetrievalHardNegatives.v2 | 0.874470 | 0.874750 | +0.000280 | -0.014470 |
| RuReviewsClassification | 0.767188 | 0.767334 | +0.000146 | -0.001123 |
| STS22 | 0.659498 | 0.660815 | +0.001317 | +0.019851 |
| SensitiveTopicsClassification | 0.437500 | 0.438672 | +0.001172 | -0.002295 |

Per-task STS-v12 all-1b-source one-pass comparison:

| Task | STS-v12 targeted 1600 one-pass | STS-v12 all-1b-source one-pass | Delta vs targeted 1600 | Delta vs Exp 2 |
| --- | ---: | ---: | ---: | ---: |
| CEDRClassification | 0.650000 | 0.649734 | -0.000266 | -0.003401 |
| GeoreviewClassification | 0.570996 | 0.570898 | -0.000098 | -0.008399 |
| GeoreviewClusteringP2P | 0.711032 | 0.709667 | -0.001365 | -0.021240 |
| HeadlineClassification | 0.892236 | 0.892188 | -0.000048 | -0.001855 |
| InappropriatenessClassification | 0.778467 | 0.778467 | +0.000000 | -0.008447 |
| KinopoiskClassification | 0.721000 | 0.720933 | -0.000067 | +0.001933 |
| MassiveIntentClassification | 0.851382 | 0.851400 | +0.000018 | +0.005355 |
| MassiveScenarioClassification | 0.911673 | 0.911673 | +0.000000 | +0.001427 |
| RiaNewsRetrievalHardNegatives.v2 | 0.874750 | 0.874400 | -0.000350 | -0.014820 |
| RuReviewsClassification | 0.767334 | 0.767676 | +0.000342 | -0.000781 |
| STS22 | 0.660815 | 0.660338 | -0.000477 | +0.019374 |
| SensitiveTopicsClassification | 0.438672 | 0.438477 | -0.000195 | -0.002490 |

### Interpretation of STS-v12

STS-v12 is better than STS-v9, but only marginally:

```text
STS-v12 200-step clean delta vs STS-v9 = +0.000255
STS-v12 800-step clean delta vs STS-v9 = +0.000365
STS-v12 1600-step one-pass clean delta vs STS-v9 = +0.001145
STS-v12 all-1b-source one-pass clean delta vs STS-v9 = +0.000937
```

The recovery stage moved the intended metrics in the right direction: `GeoreviewClusteringP2P` improved by `+0.002202`, and `RiaNewsRetrievalHardNegatives.v2` improved by `+0.000740`, while STS22 stayed essentially unchanged. This is evidence that targeted low-LR recovery is safer than the broad v11 curriculum.

The 800-step variant slightly improves the 12-task average over 200 steps, mostly through small classification gains, but it gives back some STS quality and does not improve the main broad-task gaps. Relative to 200 steps, `STS22` drops by `-0.000957`, `GeoreviewClusteringP2P` drops by `-0.000580`, and `RiaNewsRetrievalHardNegatives.v2` drops by `-0.000170`.

The 1600-step one-pass variant is the strongest STS-v12 result. Compared with the 800-step variant, it improves the 12-task clean average by `+0.000780`, improves `STS22` by `+0.001317`, improves `GeoreviewClusteringP2P` by `+0.004109`, and slightly improves RIA retrieval by `+0.000280`. This suggests the earlier 512-record recovery set was too repetitive, and broader one-pass recovery is a better training regime.

The all-1b-source one-pass variant did not improve over the targeted one-pass recovery. Compared with targeted 1600 one-pass, it reduces the clean average by `-0.000209`, `STS22` by `-0.000477`, `GeoreviewClusteringP2P` by `-0.001365`, and RIA retrieval by `-0.000350`. The only visible gain is a small `RuReviewsClassification` increase of `+0.000342`. This suggests that reintroducing the paraphrase and RuSTS sources during recovery is not helpful after STS-v9; the useful signal is still concentrated in the targeted retrieval/classification/clustering recovery sources.

The improvement is far too small to solve the broad-task regression:

```text
STS-v12 200-step clean delta vs Experiment 2 = -0.003460
STS-v12 800-step clean delta vs Experiment 2 = -0.003350
STS-v12 1600-step one-pass clean delta vs Experiment 2 = -0.002570
STS-v12 all-1b-source one-pass clean delta vs Experiment 2 = -0.002778
```

The main remaining gaps are still clustering and retrieval. STS-v12 should therefore be treated as a stronger STS-focused checkpoint, not yet a general baseline. For architecture experiments, Experiment 2 remains the stronger general checkpoint, but the 1600-step targeted one-pass variant is the best STS-v12 checkpoint so far and should replace the 100/200/800-step and all-1b-source variants for future STS-preserving recovery comparisons.

Conclusion: targeted recovery works directionally, but needs a stronger or better-targeted mechanism before it can compete with Experiment 2. A future follow-up could test lower-temperature retrieval-only recovery, a validation-controlled recovery sweep, source-balanced batches that avoid zero/near-zero loss batches, or task-specific rehearsal that explicitly protects clustering/retrieval while preserving STS.

## Experiment 1b STS-v13 Clean Clustering/Retrieval Recovery

Status: completed as an exploratory recovery attempt. Three v13 variants were trained from the STS-v12 1600-step targeted one-pass checkpoint.

Purpose: test whether clean additional clustering/retrieval data can recover the main v12_1600 gaps without sacrificing the STS22 gain.

Data:

| Source | Role | Records used |
| --- | --- | ---: |
| `deepvk/GeRaCl_synthethic_dataset:synthetic_classes_train` | synthetic clustering/classification | 2000 candidates / 1250 balanced |
| `deepvk/GeRaCl_synthethic_dataset:synthetic_positives` | synthetic positive class overlap | 1000 candidates / 750 balanced |
| `Vladimirlv/ru-promptriever-dataset:standard` | retrieval with hard negatives | 2000 candidates / 1500 balanced |
| Existing v12 targeted recovery sources | retrieval/classification/clustering anchor | 6402 candidates / 1500 balanced |

Clean-first policy: GeRaCl `ru_mteb_classes` and `ru_mteb_extended_classes` were excluded because they reuse RU-MTEB class lists. RuPromptriever was marked research-only because its license is `CC-BY-NC-4.0`.

Artifacts:

| Item | Path |
| --- | --- |
| Data builder | `scripts/prepare_open_ru_sts_v13_recovery.py` |
| Candidate data | `data/contrastive/open_ru_sts_v13_recovery_candidates.jsonl` |
| Balanced data | `data/contrastive/open_ru_sts_v13_balanced_recovery.jsonl` |
| Data summary | `data/contrastive/open_ru_sts_v13_recovery_summary.json` |
| Manifest | `configs/training_manifests/open_ru_sts_v13.json` |

Training:

| Variant | Config | Checkpoint | Training time |
| --- | --- | --- | ---: |
| v13a GeRaCl clustering-only | `configs/experiments/exp01b_retrain_latent_memory_sts_v13a_geracl_clustering_only.json` | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v13a_geracl_clustering_only/latest.pt` | 7.16 min |
| v13b RuPromptriever retrieval-only | `configs/experiments/exp01b_retrain_latent_memory_sts_v13b_promptriever_retrieval_only.json` | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v13b_promptriever_retrieval_only/latest.pt` | 3.84 min |
| v13c balanced recovery | `configs/experiments/exp01b_retrain_latent_memory_sts_v13c_balanced_recovery.json` | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v13c_balanced_recovery/latest.pt` | 14.99 min |

Common settings: `max_length = 4096`, `batch_size = 4`, `learning_rate = 1e-6`, `parameter_anchor_weight = 20.0`, frozen LLM backbone, original latent block only.

Quick gate results:

| Model | GeoreviewClusteringP2P | STS22 | Diagnostics |
| --- | ---: | ---: | ---: |
| STS-v12 1600-step targeted one-pass | 0.711032 | 0.660815 | 4/8 |
| v13a GeRaCl clustering-only | 0.710475 | 0.660799 | 4/8 |
| v13b RuPromptriever retrieval-only | 0.709769 | 0.660219 | 4/8 |
| v13c balanced recovery | 0.705883 | 0.660379 | 4/8 |

RIA retrieval check:

| Model | RiaNewsRetrievalHardNegatives.v2 | Delta vs v12_1600 |
| --- | ---: | ---: |
| STS-v12 1600-step targeted one-pass | 0.874750 | 0.000000 |
| v13b RuPromptriever retrieval-only | 0.875200 | +0.000450 |

### Interpretation of STS-v13

The clean GeRaCl synthetic-class signal did not solve the clustering gap. v13a preserved STS almost exactly but slightly reduced `GeoreviewClusteringP2P` by `-0.000557` versus v12_1600. The balanced v13c mix was worse, reducing clustering by `-0.005149` and STS22 by `-0.000436`.

RuPromptriever produced a tiny RIA gain, `+0.000450`, but also reduced `GeoreviewClusteringP2P` by `-0.001263` and `STS22` by `-0.000596`. This is not enough to justify promotion, especially since v12_1600 is already far behind 1b and Experiment 2 on RIA retrieval.

Conclusion: do not promote v13. The current added datasets are useful as negative evidence: clean synthetic class labels and generic Russian retrieval hard negatives do not recover the lost broad geometry after STS-focused training. The next recovery attempt should change the method, not just add more of these data sources. Better candidates are source-balanced multi-task batches during the original STS stage, stronger rehearsal from the 1b/Exp2 embedding geometry, or a dedicated retrieval/clustering validation loss with early stopping.

## Experiment 1b STS-v14 Grounded-RAG Retrieval Recovery

Status: completed as a targeted clean retrieval recovery probe. STS-v14 was trained from the STS-v12 1600-step targeted one-pass checkpoint.

Purpose: test whether `Vikhrmodels/Grounded-RAG-RU-v2` can improve broad retrieval/clustering quality without erasing the STS-v12 gains. This dataset was selected because it provides Russian query/document relevance structure with in-context non-relevant documents, and does not directly reuse known ruMTEB task sources in the current contamination manifest.

Data:

| Source | Role | Records |
| --- | --- | ---: |
| `Vikhrmodels/Grounded-RAG-RU-v2:good` | query to cited relevant document with same-cluster and random negatives | 8000 |

Data construction details:

| Item | Value |
| --- | --- |
| Builder | `scripts/prepare_open_ru_sts_v14_grounded_rag.py` |
| Output JSONL | `data/contrastive/open_ru_sts_v14_grounded_rag.jsonl` |
| Summary | `data/contrastive/open_ru_sts_v14_grounded_rag_summary.json` |
| Manifest | `configs/training_manifests/open_ru_sts_v14.json` |
| Parsed good rows | 34574 |
| Unique document pool | 17560 |
| Positive text max chars | 3500 |
| Negatives per record | 2 same-cluster plus 1 random |

Training:

| Item | Value |
| --- | --- |
| Config | `configs/experiments/exp01b_retrain_latent_memory_sts_v14_grounded_rag_recovery.json` |
| Initial checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v12_v9_targeted_recovery_1600_1pass/latest.pt` |
| Output checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v14_grounded_rag_recovery/latest.pt` |
| Steps | 1500 |
| Batch size | 4 |
| Max length | 4096 |
| Learning rate | `1e-6` |
| Parameter anchor | `parameter_anchor_weight = 20.0` |
| LLM backbone | frozen |
| Training time | 55.88 min |
| Mean loss | 0.127264 |
| Last loss | 0.479907 |

Quick diagnostics:

| Model | Diagnostics passed | Notes |
| --- | ---: | --- |
| STS-v12 1600-step targeted one-pass | 4/8 | previous best STS-focused checkpoint |
| STS-v14 Grounded-RAG recovery | 4/8 | same pass count; failures remained role reversal, policy contrast, numeric threshold, and distractor-long |

Clean gate results:

| Model | GeoreviewClusteringP2P | STS22 | RiaNewsRetrievalHardNegatives.v2 |
| --- | ---: | ---: | ---: |
| STS-v12 1600-step targeted one-pass | 0.711032 | 0.660815 | 0.874750 |
| STS-v14 Grounded-RAG recovery | 0.710619 | 0.660672 | 0.874750 |
| Delta | -0.000413 | -0.000143 | 0.000000 |

Evaluation artifacts:

| Item | Path |
| --- | --- |
| Semantic diagnostics | `results/semantic_diagnostics/v14_grounded_rag_recovery` |
| Georeview/STS gate | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v14-grounded-rag-gate` |
| RIA gate | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v14-grounded-rag-ria` |

Runtime note: the two-task Georeview/STS gate completed quickly, but full-context `RiaNewsRetrievalHardNegatives.v2` took about 2h51m before writing the result JSON. It remained GPU-bound throughout, so it was not stuck. This confirms that RIA at `max_length = 4096` is too expensive for every inner-loop dataset probe.

### Interpretation of STS-v14

STS-v14 is not an improvement over STS-v12 1600-step targeted one-pass. It preserved STS and RIA almost exactly, but did not recover the retrieval gap:

```text
RIA delta vs STS-v12 1600 = 0.000000
GeoreviewClusteringP2P delta vs STS-v12 1600 = -0.000413
STS22 delta vs STS-v12 1600 = -0.000143
```

The likely reason is that the generated Grounded-RAG contrastive records are often too easy for the current model. Training loss was frequently near zero, with only intermittent hard batches. Same-cluster negatives helped, but not enough to produce a measurable ruMTEB retrieval gain.

Conclusion: do not promote v14. `Grounded-RAG-RU-v2` is still a useful candidate, but the current conversion is too weak. A better follow-up would use harder candidate construction: all documents from the same conversation as in-batch candidates, multi-positive scoring when several cited docs are relevant, shorter passage windows to reduce lexical shortcuts, and source-balanced mixing with the successful STS-v12 targeted recovery data rather than a standalone recovery stage.

## Experiment 1b STS-v14b Grounded-RAG Higher-LR Ablation

Status: completed as a controlled learning-rate ablation of STS-v14.

Purpose: test whether the neutral STS-v14 result was caused by underfitting the Grounded-RAG recovery data. STS-v14b uses the same data, initial checkpoint, context length, batch size, step count, and anchor regularization as STS-v14. The only intended change is increasing the latent-block learning rate from `1e-6` to `3e-6`.

Training:

| Item | Value |
| --- | --- |
| Config | `configs/experiments/exp01b_retrain_latent_memory_sts_v14b_grounded_rag_lr3e6.json` |
| Initial checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v12_v9_targeted_recovery_1600_1pass/latest.pt` |
| Output checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v14b_grounded_rag_lr3e6/latest.pt` |
| Data | `data/contrastive/open_ru_sts_v14_grounded_rag.jsonl` |
| Steps | 1500 |
| Batch size | 4 |
| Max length | 4096 |
| Learning rate | `3e-6` |
| Parameter anchor | `parameter_anchor_weight = 20.0` |
| LLM backbone | frozen |
| Training time | 55.54 min |
| Mean loss | 0.126568 |
| Last loss | 0.433753 |

Quick diagnostics:

| Model | Diagnostics passed | Notes |
| --- | ---: | --- |
| STS-v14 Grounded-RAG recovery | 4/8 | failures: role reversal, policy contrast, numeric threshold, distractor-long |
| STS-v14b Grounded-RAG LR `3e-6` | 4/8 | same failure pattern |

Clean gate results:

| Model | GeoreviewClusteringP2P | STS22 | RiaNewsRetrievalHardNegatives.v2 |
| --- | ---: | ---: | ---: |
| STS-v12 1600-step targeted one-pass | 0.711032 | 0.660815 | 0.874750 |
| STS-v14 Grounded-RAG recovery | 0.710619 | 0.660672 | 0.874750 |
| STS-v14b Grounded-RAG LR `3e-6` | 0.708657 | 0.660909 | not run |
| v14b delta vs STS-v12 1600 | -0.002375 | +0.000094 | n/a |
| v14b delta vs STS-v14 | -0.001962 | +0.000237 | n/a |

Evaluation artifacts:

| Item | Path |
| --- | --- |
| Semantic diagnostics | `results/semantic_diagnostics/v14b_grounded_rag_lr3e6` |
| Georeview/STS gate | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v14b-grounded-rag-lr3e6-gate` |

### Interpretation of STS-v14b

Increasing the learning rate did not solve the STS-v14 weakness. The STS22 score moved up only by `+0.000094` versus STS-v12 1600, which is too small to treat as a meaningful gain, while `GeoreviewClusteringP2P` dropped by `-0.002375`.

RIA was intentionally skipped for this ablation. The previous v14 RIA run took about 2h51m at full context and was exactly neutral versus STS-v12 1600. Since the cheap clean gate already regressed on clustering, another expensive RIA run would not change the promotion decision.

Conclusion: do not promote v14b. The Grounded-RAG conversion is still likely under-hard rather than under-trained. Future use of this dataset should focus on stronger hard-negative construction and balanced mixing, not simply a higher learning rate.

## Experiment 1b STS-v15 Habr QA SBS Higher-LR Recovery

Status: completed as a targeted clean QA/retrieval recovery probe.

Purpose: test whether real Russian technical QA preference data can recover broad clustering/retrieval quality after STS-focused v12 training. STS-v15 uses filtered `Vikhrmodels/habr_qa_sbs` rows as `question -> best answer` positives with `bad answer` negatives. Per request, the learning rate was set to `3e-6` so the effect would be more visible.

Data preparation:

| Item | Value |
| --- | --- |
| Builder | `scripts/prepare_open_ru_sts_v15_habr_qa_sbs.py` |
| Output JSONL | `data/contrastive/open_ru_sts_v15_habr_qa_sbs.jsonl` |
| Summary | `data/contrastive/open_ru_sts_v15_habr_qa_sbs_summary.json` |
| Manifest | `configs/training_manifests/open_ru_sts_v15.json` |
| Raw rows scanned | 102558 |
| Records kept | 8000 |
| Positive | `best` answer |
| Negative | `bad` answer |
| Main filters | min question 20 chars, min best 80 chars, min bad 40 chars, max answer 3500 chars, remove duplicate/same/similar best-bad pairs |

Main rejection counts:

| Reason | Count |
| --- | ---: |
| short best | 20134 |
| short bad | 6504 |
| same best/bad | 2298 |
| short question | 2142 |
| long answer | 406 |

Training:

| Item | Value |
| --- | --- |
| Config | `configs/experiments/exp01b_retrain_latent_memory_sts_v15_habr_qa_sbs_lr3e6.json` |
| Initial checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v12_v9_targeted_recovery_1600_1pass/latest.pt` |
| Output checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v15_habr_qa_sbs_lr3e6/latest.pt` |
| Steps | 1500 |
| Batch size | 4 |
| Max length | 4096 |
| Learning rate | `3e-6` |
| Parameter anchor | `parameter_anchor_weight = 20.0` |
| LLM backbone | frozen |
| Training time | 8.57 min |
| Mean loss | 1.922128 |
| Last loss | 1.124140 |

Quick diagnostics:

| Model | Diagnostics passed | Notes |
| --- | ---: | --- |
| STS-v12 1600-step targeted one-pass | 4/8 | previous best STS-focused checkpoint |
| STS-v15 Habr QA SBS LR `3e-6` | 4/8 | same failure pattern: role reversal, policy contrast, numeric threshold, distractor-long |

Clean gate results:

| Model | GeoreviewClusteringP2P | STS22 | RiaNewsRetrievalHardNegatives.v2 |
| --- | ---: | ---: | ---: |
| STS-v12 1600-step targeted one-pass | 0.711032 | 0.660815 | 0.874750 |
| STS-v14 Grounded-RAG recovery | 0.710619 | 0.660672 | 0.874750 |
| STS-v14b Grounded-RAG LR `3e-6` | 0.708657 | 0.660909 | not run |
| STS-v15 Habr QA SBS LR `3e-6` | 0.714725 | 0.660644 | 0.877540 |
| v15 delta vs STS-v12 1600 | +0.003693 | -0.000171 | +0.002790 |
| v15 delta vs STS-v14 | +0.004106 | -0.000028 | +0.002790 |

Evaluation artifacts:

| Item | Path |
| --- | --- |
| Semantic diagnostics | `results/semantic_diagnostics/v15_habr_qa_sbs_lr3e6` |
| Georeview/STS gate | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v15-habr-qa-sbs-lr3e6-gate` |
| RIA gate | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v15-habr-qa-sbs-lr3e6-ria` |

### Interpretation of STS-v15

STS-v15 is the first recovery probe after v12_1600 that improves both the clean clustering gate and the clean RIA retrieval gate:

```text
GeoreviewClusteringP2P delta vs STS-v12 1600 = +0.003693
RiaNewsRetrievalHardNegatives.v2 delta vs STS-v12 1600 = +0.002790
STS22 delta vs STS-v12 1600 = -0.000171
```

The STS22 movement is a small regression, but the clustering and retrieval gains are larger and directionally consistent. This suggests that filtered real QA preference data is a better recovery signal than the current Grounded-RAG conversion. The likely reason is that Habr QA pairs contain real user intent and answer-quality distinctions, while v14 often produced easy document negatives.

Conclusion: promote v15 as the current best recovery checkpoint for broad clean retrieval/clustering among the v13-v15 probes. The next useful ablation is not another standalone higher-LR QA run; it should test balanced mixing of v15 Habr QA with the v12 targeted one-pass data, or a lower-LR/longer-step variant to see whether the small STS22 regression can be reduced while preserving the Georeview/RIA gains.

## Experiment 1b STS-v16 Habr QA SBS Full One-Pass

Status: completed as an uncapped Habr QA SBS scale ablation.

Purpose: test whether the positive v15 signal improves further when the cap is removed and the model sees the full filtered `Vikhrmodels/habr_qa_sbs` set approximately once. This keeps the same starting checkpoint and learning rate as v15, but increases training from 1500 capped steps to 17747 steps over all filtered records.

Data preparation:

| Item | Value |
| --- | --- |
| Builder | `scripts/prepare_open_ru_sts_v15_habr_qa_sbs.py` |
| Output JSONL | `data/contrastive/open_ru_sts_v16_habr_qa_sbs_full.jsonl` |
| Summary | `data/contrastive/open_ru_sts_v16_habr_qa_sbs_full_summary.json` |
| Manifest | `configs/training_manifests/open_ru_sts_v16.json` |
| Raw rows scanned | 102558 |
| Records kept | 70986 |
| Data size | 107 MB |
| Positive | `best` answer |
| Negative | `bad` answer |
| Cap | none |

Main rejection counts:

| Reason | Count |
| --- | ---: |
| short best | 20134 |
| short bad | 6504 |
| same best/bad | 2298 |
| short question | 2142 |
| long answer | 406 |

Training:

| Item | Value |
| --- | --- |
| Config | `configs/experiments/exp01b_retrain_latent_memory_sts_v16_habr_qa_sbs_full_1pass_lr3e6.json` |
| Initial checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v12_v9_targeted_recovery_1600_1pass/latest.pt` |
| Output checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v16_habr_qa_sbs_full_1pass_lr3e6/latest.pt` |
| Steps | 17747 |
| Batch size | 4 |
| Max length | 4096 |
| Learning rate | `3e-6` |
| Parameter anchor | `parameter_anchor_weight = 20.0` |
| LLM backbone | frozen |
| Training time | 101.07 min |
| Mean loss | 1.712509 |
| Last loss | 0.165074 |

Quick diagnostics:

| Model | Diagnostics passed | Notes |
| --- | ---: | --- |
| STS-v15 Habr QA SBS LR `3e-6` | 4/8 | failures: role reversal, policy contrast, numeric threshold, distractor-long |
| STS-v16 Habr QA SBS full one-pass LR `3e-6` | 4/8 | same failure pattern |

Clean gate results:

| Model | GeoreviewClusteringP2P | STS22 | RiaNewsRetrievalHardNegatives.v2 |
| --- | ---: | ---: | ---: |
| Experiment 1b original open-RU mix | 0.729729 | 0.639772 | 0.887210 |
| STS-v12 1600-step targeted one-pass | 0.711032 | 0.660815 | 0.874750 |
| STS-v15 Habr QA SBS LR `3e-6` | 0.714725 | 0.660644 | 0.877540 |
| STS-v16 Habr QA SBS full one-pass LR `3e-6` | 0.712800 | 0.658658 | not run |
| v16 delta vs STS-v12 1600 | +0.001768 | -0.002157 | n/a |
| v16 delta vs STS-v15 | -0.001925 | -0.001986 | n/a |
| v16 delta vs original 1b | -0.016929 | +0.018886 | n/a |

Evaluation artifacts:

| Item | Path |
| --- | --- |
| Semantic diagnostics | `results/semantic_diagnostics/v16_habr_qa_sbs_full_1pass_lr3e6` |
| Georeview/STS gate | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v16-habr-qa-sbs-full-1pass-lr3e6-gate` |
| RIA gate | skipped because the cheap clean gate regressed versus v15 |

### Interpretation of STS-v16

Removing the cap did not improve over v15. Although v16 still remains above STS-v12 on `GeoreviewClusteringP2P`, it loses most of the v15 recovery and also reduces `STS22`:

```text
GeoreviewClusteringP2P delta vs STS-v15 = -0.001925
STS22 delta vs STS-v15 = -0.001986
```

This suggests that the useful Habr QA SBS signal is not simply proportional to more examples or more steps. The 1500-step capped v15 run likely acted as a targeted recovery regularizer, while the full one-pass v16 run shifted the latent block too far toward technical QA preference matching and away from the broader embedding geometry. RIA was skipped because the cheap gate already regressed on both checked tasks and previous full-context RIA runs are expensive.

Conclusion: do not promote v16. Keep v15 as the best Habr QA recovery checkpoint so far. The next Habr-based experiment should use controlled mixing or sampling rather than a full standalone pass: examples include source-balanced rehearsal with v12/v1b data, lower LR with early clean-gate checkpoints, or hard-example filtering instead of using all accepted Habr QA SBS rows.

## Experiment 1b STS-v16-Hard Habr QA SBS Hard-Filtered Recovery

Status: completed as the hard-filtered follow-up to the failed full one-pass v16 run.

Purpose: test whether the useful v15 Habr QA signal can be strengthened by filtering for harder `best` vs `bad` answer distinctions instead of training on all accepted rows. This run keeps the same v12_1600 initialization, frozen backbone, context length, batch size, learning rate, and parameter anchor as v15/v16.

Data preparation:

| Item | Value |
| --- | --- |
| Builder | `scripts/prepare_open_ru_sts_v15_habr_qa_sbs.py` |
| Output JSONL | `data/contrastive/open_ru_sts_v16_habr_qa_sbs_hard.jsonl` |
| Summary | `data/contrastive/open_ru_sts_v16_habr_qa_sbs_hard_summary.json` |
| Manifest | `configs/training_manifests/open_ru_sts_v16_hard.json` |
| Raw rows scanned | 102558 |
| Records kept | 10352 |
| Data size | 11 MB |
| Positive | `best` answer |
| Negative | `bad` answer |
| Hardness filter | `0.18 <= lexical_similarity(best, bad) < 0.86` |
| Additional filters | min question 25 chars, min best 120 chars, min bad 80 chars, min best 12 words, min bad 8 words, max answer 3000 chars |

Main rejection counts:

| Reason | Count |
| --- | ---: |
| easy best/bad | 39143 |
| short best | 30169 |
| short bad | 14402 |
| short question | 6058 |
| same best/bad | 1871 |
| long answer | 528 |

Training:

| Item | Value |
| --- | --- |
| Config | `configs/experiments/exp01b_retrain_latent_memory_sts_v16_habr_qa_sbs_hard_lr3e6.json` |
| Initial checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v12_v9_targeted_recovery_1600_1pass/latest.pt` |
| Output checkpoint | `experiments/exp01_original_latent_memory/checkpoints/open_ru_sts_v16_habr_qa_sbs_hard_lr3e6/latest.pt` |
| Steps | 2588 |
| Batch size | 4 |
| Max length | 4096 |
| Learning rate | `3e-6` |
| Parameter anchor | `parameter_anchor_weight = 20.0` |
| LLM backbone | frozen |
| Training time | 9.76 min |
| Mean loss | 1.635858 |
| Last loss | 0.707560 |

Quick diagnostics:

| Model | Diagnostics passed | Notes |
| --- | ---: | --- |
| STS-v15 Habr QA SBS LR `3e-6` | 4/8 | failures: role reversal, policy contrast, numeric threshold, distractor-long |
| STS-v16 full Habr one-pass LR `3e-6` | 4/8 | same failure pattern |
| STS-v16-hard Habr QA SBS LR `3e-6` | 4/8 | same failure pattern |

Clean gate results:

| Model | GeoreviewClusteringP2P | STS22 | RiaNewsRetrievalHardNegatives.v2 |
| --- | ---: | ---: | ---: |
| Experiment 1b original open-RU mix | 0.729729 | 0.639772 | 0.887210 |
| STS-v12 1600-step targeted one-pass | 0.711032 | 0.660815 | 0.874750 |
| STS-v15 Habr QA SBS LR `3e-6` | 0.714725 | 0.660644 | 0.877540 |
| STS-v16 full Habr one-pass LR `3e-6` | 0.712800 | 0.658658 | not run |
| STS-v16-hard Habr QA SBS LR `3e-6` | 0.718168 | 0.659734 | 0.878090 |
| v16-hard delta vs STS-v12 1600 | +0.007136 | -0.001081 | +0.003340 |
| v16-hard delta vs STS-v15 | +0.003443 | -0.000910 | +0.000550 |
| v16-hard delta vs original 1b | -0.011561 | +0.019962 | -0.009120 |

Evaluation artifacts:

| Item | Path |
| --- | --- |
| Semantic diagnostics | `results/semantic_diagnostics/v16_habr_qa_sbs_hard_lr3e6` |
| Georeview/STS gate | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v16-habr-qa-sbs-hard-lr3e6-gate` |
| RIA gate | `results/rumteb/maxlen-4096-prompt-clean-open_ru_sts_v16-habr-qa-sbs-hard-lr3e6-ria` |

### Interpretation of STS-v16-Hard

Hard filtering fixed the main failure mode of the full v16 run. It recovered and exceeded v15 on both checked broad tasks:

```text
GeoreviewClusteringP2P delta vs STS-v15 = +0.003443
RiaNewsRetrievalHardNegatives.v2 delta vs STS-v15 = +0.000550
STS22 delta vs STS-v15 = -0.000910
```

This supports the hypothesis that Habr QA SBS is useful when treated as hard preference data, not as a large generic QA pass. The min-similarity filter removed easy best/bad mismatches, so the latent block had to learn finer answer relevance distinctions. The cost is a small STS22 regression versus v15 and v12_1600, so this checkpoint is not a pure STS improvement.

Conclusion: promote v16-hard as the current best Habr-based clustering/retrieval recovery checkpoint. It should replace v15 only when broad retrieval/clustering is more important than preserving the last `0.0009` STS22. For the next run, use v16-hard as evidence that hard-example selection matters; do not return to uncapped full-pass training. The natural follow-up is a mixed hard-Habr plus STS rehearsal run to keep the Georeview/RIA gains while recovering STS22.

## Experiment 3: Iterative Embedding Refinement

Status: not run yet.

Planned architecture:

```text
embedding_0 = attention_pool(tokens)

for t in 1..T:
    delta = cross_attention(Q=embedding_t, K=tokens, V=tokens)
    embedding_{t+1} = embedding_t + alpha * MLP(delta)
```

Report items to fill after the run:

- Implementation notes.
- Iteration count and refinement hidden size.
- Stability settings such as normalization and residual scale.
- Training config.
- Checkpoint paths.
- Runtime and GPU memory.
- Clean ruMTEB result directory.
- Clean vs Experiment 1b deltas.
- Contamination/leakage interpretation.

## Experiment 4: Slot Compression Plus Embedding Refinement

Status: not run yet.

Planned architecture:

```text
tokens
  -> latent slots
  -> embedding_0

for t in 1..T:
    delta = cross_attention(Q=embedding_t, K=latent_slots, V=latent_slots)
    embedding_{t+1} = embedding_t + alpha * MLP(delta)
```

Report items to fill after the run:

- Implementation notes.
- Iteration count and slot count.
- Stability settings such as normalization and residual scale.
- Training config.
- Checkpoint paths.
- Runtime and GPU memory.
- Clean ruMTEB result directory.
- Clean vs Experiment 1b deltas.
- Contamination/leakage interpretation.
