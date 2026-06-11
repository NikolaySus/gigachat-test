# Corrected Research Plan: Latent Attention Experiments for GigaEmbeddings

## 1. Paper Re-Check and Baseline Interpretation

Re-reading `2025.bsnlp-1.3.pdf` confirms the important architecture detail from Figure 1: the released latent attention block is not Perceiver-style compression.

The paper describes:

```text
GigaChat LLM output = Q
trainable latent array = K,V
cross-attention output -> MLP -> mean pool -> embedding
```

The inspected Hugging Face implementation matches this. Token hidden states query a learned latent memory, then the model mean-pools over token positions. The 512 latent vectors are learned key/value memory slots, not output semantic slots.

This means the proposed `tokens -> latent slots -> embedding` experiments are real architecture changes, not small modifications of the released pooling block.

Paper-grounded training facts:

- backbone: GigaChat-3B adapted as a bidirectional encoder by removing causal masks;
- pruning: 9 of 36 deeper transformer blocks removed, producing a 2.5B backbone with nearly unchanged ruMTEB score;
- objective: InfoNCE with temperature `0.02`;
- pipeline: contrastive pretraining, retrieval fine-tuning with hard negatives, multitask generalization;
- paper hyperparameters: pretraining batch `16K`, fine-tuning/multitask batch `512`, max length `512`, hard negatives `7`;
- prompting: `Instruct: ...\nQuery: ...` outperforms prefix prompting.

Current corrected evaluation baseline:

```text
ruMTEB v1.1, max_length=4096, prompts enabled: 0.7354 average
```

Target diagnostic weaknesses:

- numeric polarity and thresholds;
- contrast clauses;
- role reversal;
- distractor-heavy long text.

## 2. Experiments

### Experiment 1: Original Latent-Memory Block

Architecture:

```text
tokens
-> GigaChat hidden states
-> token hidden states query learned latent K,V
-> MLP
-> mask-aware mean pooling over token positions
-> L2-normalized embedding
```

Purpose:

- Establish both the external released-model reference and the training-controlled architecture baseline.
- Preserve exact original architecture behavior while separating architecture effects from training-data effects.

Configuration:

```text
num_latents = 512
latent_dim = 2048
num_cross_heads = 8
MLP expansion = 4x
pooling = mean pooling over token outputs
evaluation max_length = 4096
prompting = enabled
```

Variants:

- **Experiment 1a: Released reference.** No new training; use `ai-sage/Giga-Embeddings-instruct` as-is.
- **Experiment 1b: Retrained original latent-memory control.** Freeze the LLM and train the original latent-memory block with the same open-data pipeline, optimizer, prompts, max length policy, and training budget used for Experiments 2-4.
- **Experiment 1c: Reinitialized original latent-memory control, optional.** Reinitialize only the latent-memory block and train it with the same setup as Experiment 1b. Use this only if time allows, because it may need more data to recover from scratch.

Primary comparison rule:

```text
Use Experiment 1a as the public-model reference.
Use Experiment 1b as the main architecture-control baseline.
Compare Experiments 2-4 primarily against Experiment 1b.
```

### Experiment 2: True Hierarchical Latent Compression

Architecture:

```text
tokens
-> GigaChat hidden states
-> latent_512 queries attend to token hidden states
-> latent_128 queries attend to latent_512
-> attention pooling over latent_128
-> L2-normalized embedding
```

Purpose:

- Test whether explicit semantic slot compression improves hierarchical and long-text representation.
- Correctly test the hypothesis `tokens -> semantic components -> embedding`.

Configuration:

```text
stage1_latents = 512
stage2_latents = 128
latent_dim = 2048
heads = 8
MLP expansion = 4x
pooling = attention pooling over latent_128
```

Training:

- Freeze LLM first.
- Train only compression and pooling modules.
- Use exactly the same open-data training pipeline and budget as Experiment 1b.
- Optionally unfreeze top 4-8 LLM layers after module-only results are measured.

### Experiment 3: Token-Based Iterative Embedding Refinement

Architecture:

```text
embedding_0 = attention_pool(token hidden states)

for t in 1..4:
    delta_t = cross_attention(Q=embedding_t, K=tokens, V=tokens)
    embedding_{t+1} = embedding_t + 0.5 * LN(MLP(delta_t))

final_embedding = L2-normalize(embedding_4)
```

Purpose:

- Test whether embedding compression benefits from iterative "look again and refine" reasoning.
- Directly target failures where small token-level details flip meaning.

Configuration:

```text
iterations = 4
heads = 8
embedding_dim = 2048
refinement_hidden = 4096
alpha = 0.5
weights = shared across iterations
LayerNorm = before residual update
```

Training:

- Freeze LLM initially.
- Train attention pooling and refinement block.
- Use exactly the same open-data training pipeline and budget as Experiment 1b.
- If unstable, use a learned residual gate initialized near zero.

### Experiment 4: Latent Slot Compression + Iterative Refinement

Architecture:

```text
tokens
-> latent_512 queries attend to token hidden states
-> embedding_0 = attention_pool(latent_512)

for t in 1..4:
    delta_t = cross_attention(Q=embedding_t, K=latent_512, V=latent_512)
    embedding_{t+1} = embedding_t + 0.5 * LN(MLP(delta_t))

final_embedding = L2-normalize(embedding_4)
```

Purpose:

- Combine true latent compression with iterative refinement.
- Keep refinement cost bounded by latent slots instead of full token length.

Configuration:

```text
latent_slots = 512
latent_dim = 2048
iterations = 4
heads = 8
refinement_hidden = 4096
alpha = 0.5
pooling = attention pooling over latent slots
```

Training:

- Freeze LLM first.
- Train latent compressor, pooling, and refinement modules.
- Use exactly the same open-data training pipeline and budget as Experiment 1b.
- Prioritize this experiment if only one new architecture can be trained after baseline.

## 3. Training Pipeline for Experiments

The original paper uses a full three-stage pipeline:

```text
1. large-batch contrastive pretraining
2. retrieval fine-tuning with hard negatives
3. multitask instruction tuning for retrieval, classification, clustering, STS-like tasks
```

For these architecture experiments, reproducing the full industrial pipeline is not required at first. Use a reduced open-dataset version:

```text
Stage A: module-only contrastive warmup
Stage B: retrieval-focused hard-negative fine-tuning
Stage C: compact multitask tuning
```

Open-source dataset candidates:

- MIRACL Russian retrieval data and corpus, Apache-2.0:
  - https://huggingface.co/datasets/miracl/miracl
  - https://huggingface.co/datasets/miracl/miracl-corpus
- RuBQ retrieval, CC-BY-SA-4.0:
  - https://huggingface.co/datasets/ai-forever/rubq-retrieval
- RuBQ reranking, CC-BY-SA-4.0:
  - https://huggingface.co/datasets/ai-forever/rubq-reranking
- RuSciBench GRNTI/OECD classification, MIT:
  - https://huggingface.co/datasets/ai-forever/ru-scibench-grnti-classification
  - https://huggingface.co/datasets/ai-forever/ru-scibench-oecd-classification
- Russian SuperGLUE / TERRa, MIT, for NLI-style entailment and role-sensitive pair supervision:
  - https://huggingface.co/datasets/RussianNLP/russian_super_glue
- ParaPhraser, MIT, for Russian paraphrase and semantic pair supervision:
  - https://huggingface.co/datasets/merionum/ru_paraphraser
- RuSTSBenchmarkSTS, CC-BY-SA-4.0, for STS-style supervision if excluded from clean STS evaluation:
  - https://huggingface.co/datasets/mteb/RuSTSBenchmarkSTS

If an evaluation dataset is reused for training, mark it as contaminated and exclude it from clean evaluation reporting.

Use `configs/training_manifests/open_ru_ablation_v1.json` as the first machine-readable contamination manifest. The evaluation wrapper accepts `--training-manifest` and `--eval-scope all|clean|contaminated`; the summarizer reports full, clean, and contaminated averages from any ruMTEB output directory.

Training weak spots to address before architecture conclusions:

- **Evaluation contamination.** Several useful open datasets are also ruMTEB tasks. Maintain two reports: clean ruMTEB excluding trained-on tasks, and full diagnostic ruMTEB with contaminated tasks clearly marked.
- **NLI and role reversal.** Retrieval data alone is weak supervision for entailment, argument roles, and contradiction. Include Russian SuperGLUE/TERRa-style pair data in the compact multitask stage.
- **Numeric and logical polarity.** Public embedding datasets underrepresent `до/от`, `не менее/не более`, date/order constraints, and legal/financial thresholds. Add a small controlled template set for diagnostics, and use it for training only as a separate ablation.
- **Contrast clauses.** Include paraphrase/NLI examples that preserve or invert `но`, `однако`, `кроме`, and conditional constraints.
- **Long-document compression.** A `512`-token-only training schedule will not test the proposed latent compression. Add a long-context phase at `2048` or `4096` using retrieval/document data with distractor-heavy hard negatives.
- **English preservation.** Either explicitly scope the study as Russian-only or include a small English preservation mix from open retrieval/STS/NLI datasets.
- **Hard-negative consistency.** Use exactly 7 hard negatives per query where possible. If a dataset has no hard negatives, mine them with the released baseline model and reuse the same negatives for Experiments 1b-4.

Training defaults:

```text
loss = InfoNCE
temperature = 0.02
hard negatives = 7 where available
module-only batch size = largest stable value on 24 GB VRAM
max_length = 512 for training warmup
max_length = 2048 or 4096 for long-context phase
max_length = 4096 for final evaluation
prompting = Instruct format
hard-negative mining model = released Experiment 1a baseline
```

Because the paper used private/synthetic large-scale data, results should be reported as architecture ablations under limited open-data training, not as a full reproduction of GigaEmbeddings training.

## 4. Evaluation Plan

All experiments use the same corrected evaluation harness.

### ruMTEB v1.1

Main evaluation:

```text
max_length = 4096
prompting = enabled
query prompts for retrieval/reranking
symmetric prompts for classification, clustering, STS, pair classification
```

Report:

- overall average;
- category averages;
- per-task scores;
- clean average excluding datasets used for training;
- contaminated-task average marked separately;
- delta against Experiment 1a;
- delta against Experiment 1b.

Acceptance rule:

```text
A new architecture is useful only if it improves targeted diagnostics and does not regress overall ruMTEB by more than 1 absolute point against Experiment 1b.
```

### Semantic Diagnostics

Run in both modes:

```text
raw text
prompted text
```

Required categories:

- negation;
- numeric thresholds;
- role reversal;
- contrast clauses;
- temporal order;
- hierarchy/nested constraints;
- multi-hop implication;
- long distractors.

Primary target failures from baseline:

```text
numeric_threshold
contrastive_policy
role_reversal_contract
distractor_long
```

Report:

- pass count;
- mean positive-vs-hard-negative margin;
- per-category margin;
- worst cases.

### Retrieval-Style Diagnostics

Use query/document format:

```text
query = prompted
document = unprompted
```

Include hard negatives for:

- numeric polarity;
- role reversal;
- contrast inversion;
- distractor-heavy document;
- long answer-bearing document.

Report:

- Recall@1;
- MRR@10;
- positive-vs-hardest-negative margin.

### Efficiency Evaluation

Measure on RTX 4500 Ada 24 GB:

```text
batch sizes = 8, 16, 24, 32
max_length = 512 and 4096
```

Report:

- peak VRAM;
- embeddings/sec;
- average latency;
- full ruMTEB runtime;
- maximum stable batch size.

Stop increasing batch size when peak VRAM exceeds about `22 GB` or CUDA OOM occurs.

## 5. Assumptions and Defaults

- Experiment 1a is the released model.
- Experiment 1b is the fair training-controlled baseline for architecture ablations.
- Experiments 2-4 are judged primarily against Experiment 1b, not only against the released model.
- All new experiments preserve embedding dimension `2048`.
- All final embeddings are L2-normalized.
- Prompt-aware `max_length=4096` ruMTEB is the main benchmark.
- `max_length=512` and `--no-prompts` are ablations only.
- Initial training freezes the LLM.
- Top-layer LLM unfreezing is optional and only after module-only results are measured.
- Use only datasets with public access and acceptable licenses for experiment training.
- Record dataset licenses in every run report.
