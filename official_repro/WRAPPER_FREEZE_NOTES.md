# Giga Official Ru MTEB Wrapper Freeze Notes

Freeze target: `official_repro/run_official_rumteb.py` before accepting any RuBQ-specific prompt tuning.

The wrapper is intended to reproduce the public `ai-sage/Giga-Embeddings-instruct` Ru MTEB results as closely as possible without changing model weights. It keeps the released model fixed and tunes only evaluation-side prompt/preprocessing policy.

## Model And Benchmark

- Model: `ai-sage/Giga-Embeddings-instruct`
- Model revision: `40b27667b9ad586d7812675df76e5062ccc80b0e`
- Benchmark: `MTEB(rus, v1)`
- Main runner: `official_repro/run_official_rumteb.py`
- Comparator: `official_repro/compare_official_rumteb.py`
- Python environment: `official_repro/.venv`
- Default run seed: `8`
- Default batch size used for reproduction runs: `12`
- Max length: runner default, currently `4096`
- Attention implementation: runner default
- Torch dtype: runner default

## Frozen Run Command

For the practical 21-task reproduction run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
official_repro/.venv/bin/python official_repro/run_official_rumteb.py \
  --output-folder results/official_repro/giga_mteb138_rus_v1_current_tuned_full23 \
  --prompt-mode legacy_ru \
  --batch-size 12 \
  --seed 8 \
  --overwrite-results
```

Then compare with public MTEB results:

```bash
official_repro/.venv/bin/python official_repro/compare_official_rumteb.py \
  results/official_repro/giga_mteb138_rus_v1_current_tuned_full23 \
  --write-md results/official_repro/giga_mteb138_rus_v1_current_tuned_full23_comparison.md
```

In practice, `MIRACLRetrieval` and `RiaNewsRetrieval` were not completed in the full run because they require very large corpus encoding. The 21-task comparison includes all other Ru MTEB tasks.

## Frozen Prompt Policy

The global prompt mode is:

```text
legacy_ru
```

This means the wrapper applies manually selected Russian prefixes before text encoding. For retrieval and reranking tasks, the prefix is applied only on the query side; passages/documents are left unprefixed.

### Legacy Prefixes

```python
LEGACY_RU_PREFIXES = {
    "default": "Дан текст, необходимо найти семантически похожий текст \nтекст: ",
    "sts": "Найди семантически похожий текст \nтекст: ",
    "ruparaphraser": "найди семантически похожее предложение \nтекст: ",
    "rusts": "семантически похожий текст: ",
    "retrieval": "Дан вопрос, необходимо найти абзац текста с ответом \nвопрос: ",
    "sensitive": "Классифицируй чувствительную тему по запросу \nзапрос: ",
    "inappropriate": "Определи, является ли сообщение неприемлемым, токсичным или чувствительным \nсообщение: ",
    "cedr": "Определи эмоции в комментарии: радость, грусть, удивление, страх или злость \nкомментарий: ",
    "headline": "Определи тему новостного заголовка \nзаголовок: ",
    "georeview": "Определи категорию организации на основе отзыва \nотзыв: ",
    "georeview_classification": "Определи тональность отзыва о сервисе организации \nотзыв: ",
    "science_clustering": "Определи категорию научной статьи по названию и аннотации \nтекст: ",
    "terra": "семантически похожий текст: ",
}
```

Task mapping:

- Retrieval and reranking query side: `retrieval`
- `RUParaPhraserSTS`: `ruparaphraser`
- `RuSTSBenchmarkSTS`: `rusts`
- Other STS tasks: `sts`
- `SensitiveTopicsClassification`: `sensitive`
- `InappropriatenessClassification`: `inappropriate`
- `CEDRClassification`: `cedr`
- `HeadlineClassification`: `headline`
- `GeoreviewClassification`: `georeview_classification`
- `GeoreviewClusteringP2P`: `georeview`
- `RuSciBenchGRNTIClusteringP2P`: `science_clustering`
- `RuSciBenchOECDClusteringP2P`: `science_clustering`
- `TERRa`: `terra`
- Otherwise: `default`

## Task-Specific Overrides

### Prompt Mode Overrides

```python
DEFAULT_TASK_PROMPT_MODES = {
    "GeoreviewClassification": "legacy_ru_masked",
    "RuReviewsClassification": "prefix",
    "MassiveScenarioClassification": "prefix",
    "MassiveIntentClassification": "prefix",
    "RuSciBenchGRNTIClassification": "prefix",
}
```

Rationale:

- `GeoreviewClassification` was sensitive to wrapper details; `legacy_ru_masked` reproduced the released behavior better than plain prefixed text.
- `RuReviewsClassification`, `MassiveScenarioClassification`, `MassiveIntentClassification`, and `RuSciBenchGRNTIClassification` scored closer to official with model/MTEB prefix-style prompting than with plain legacy Russian prefixes.

### MTEB Prompt Override

```python
DEFAULT_MTEB_PROMPT_OVERRIDES = {
    "MassiveIntentClassification": "Given a user request, find the intended assistant action",
}
```

Rationale: this prompt improved `MassiveIntentClassification` alignment with the public result.

### Prefix Ensemble

```python
DEFAULT_LEGACY_PREFIX_ENSEMBLES = {
    "RuSTSBenchmarkSTS": [
        "семантически похожий текст: ",
        "семантически похожий текст \nтекст: ",
    ],
}
```

Rationale: `RuSTSBenchmarkSTS` remained one of the harder tasks to match. Averaging these two prompt embeddings was the best stable variant found, though it still stayed slightly below official.

### Text Normalization

```python
DEFAULT_TASK_TEXT_NORMALIZATIONS = {
    "RuSTSBenchmarkSTS": "yo",
}
```

Rationale: replacing `ё/Ё` with `е/Е` improved `RuSTSBenchmarkSTS` reproducibility.

### Task Batch Size Cap

```python
DEFAULT_TASK_BATCH_SIZES = {
    "RuSTSBenchmarkSTS": 4,
}
```

Rationale: `RuSTSBenchmarkSTS` used the prompt ensemble and was more memory-sensitive. Batch `4` avoided OOM while preserving the chosen prompt policy.

### Task Seed Override

```python
DEFAULT_TASK_SEEDS = {
    "GeoreviewClassification": 42,
}
```

Rationale: `GeoreviewClassification` showed sensitivity to MTEB classifier/evaluator randomness. Seed `42` gave the closest stable match in task-isolated checks and was kept as a task-level seed override.

Important: seed is set at the task level, not inside `encode`. Resetting the seed inside `encode` was not used.

## Technical Compatibility Fixes

These changes are part of the frozen wrapper because they are required for robust evaluation rather than prompt tuning:

- Add `HF_HOME/modules` to `sys.path` after cache directory setup. This fixes dynamic Hugging Face dataset module imports such as MIRACL.
- `similarity()` returns a Torch tensor. MTEB retrieval/reranking evaluators call Torch operations such as `torch.topk` and `torch.amax` on similarity scores.
- CUDA OOM fallback recursively splits only the offending encode batch and continues.
- Sparse progress logging is enabled for very large encode calls, so long retrieval/reranking jobs do not appear stuck.

## RuBQ Tuning Decision

RuBQ-specific prompt tuning was tested after this freeze point and rejected. No RuBQ-specific override is part of the frozen wrapper.

The frozen RuBQ prompt remains the generic retrieval query prefix:

```text
Дан вопрос, необходимо найти абзац текста с ответом 
вопрос:
```

The best tested RuBQ wording variant slightly improved isolated reranking but regressed retrieval, so it was not accepted.

## Reproduction Results

Current practical comparison file:

```text
results/official_repro/giga_mteb138_rus_v1_current_tuned_full23_comparison.md
```

Summary:

```text
Matched tasks: 21
Local avg: 0.704695
Official avg: 0.702673
Mean delta: +0.002023
Mean abs delta: 0.004786
```

Missing from the 23-task full comparison:

- `MIRACLRetrieval`
- `RiaNewsRetrieval`

Reason:

- `MIRACLRetrieval` requires encoding roughly 9.5M Russian MIRACL corpus documents.
- `RiaNewsRetrieval` requires encoding 704,344 corpus documents.
- Both are valid tasks, but they are too expensive for prompt-tuning iteration on the available GPU.

## 21-Task Result Table

| Task | Local repro | Official | Delta |
|---|---:|---:|---:|
| KinopoiskClassification | 0.703867 | 0.689800 | +0.014067 |
| RuSciBenchGRNTIClusteringP2P | 0.694131 | 0.683443 | +0.010688 |
| MIRACLReranking | 0.666080 | 0.656150 | +0.009930 |
| RuBQRetrieval | 0.727290 | 0.735510 | -0.008220 |
| HeadlineClassification | 0.895508 | 0.887939 | +0.007569 |
| RuSciBenchGRNTIClassification | 0.722412 | 0.716162 | +0.006250 |
| RuBQReranking | 0.763017 | 0.768870 | -0.005853 |
| GeoreviewClusteringP2P | 0.670603 | 0.676357 | -0.005754 |
| RuSciBenchOECDClusteringP2P | 0.564860 | 0.559266 | +0.005594 |
| TERRa | 0.642853 | 0.637836 | +0.005017 |
| RuReviewsClassification | 0.739941 | 0.743848 | -0.003907 |
| CEDRClassification | 0.688682 | 0.685069 | +0.003613 |
| RuSTSBenchmarkSTS | 0.832399 | 0.835900 | -0.003501 |
| GeoreviewClassification | 0.550781 | 0.547510 | +0.003271 |
| RuSciBenchOECDClassification | 0.547412 | 0.545068 | +0.002344 |
| SensitiveTopicsClassification | 0.441699 | 0.439941 | +0.001758 |
| InappropriatenessClassification | 0.847852 | 0.846729 | +0.001123 |
| MassiveIntentClassification | 0.823974 | 0.824815 | -0.000841 |
| RUParaPhraserSTS | 0.748035 | 0.748833 | -0.000798 |
| MassiveScenarioClassification | 0.904976 | 0.904707 | +0.000269 |
| STS22 | 0.622233 | 0.622370 | -0.000137 |

## Notes On Fairness

This wrapper is an evaluation reproduction wrapper, not a new model. Using task-specific prompts and preprocessing is acceptable for reproducing a published MTEB submission only if the same wrapper policy is frozen and reported. It should not be mixed with model-training comparisons unless the same wrapper is used consistently for all compared models.

For future trained model experiments, freeze this wrapper policy and evaluate every checkpoint with the same code, seed, prompt map, normalization, and task batch caps.
