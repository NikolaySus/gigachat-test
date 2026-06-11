# Latent Attention Reinitialized-Weights Fair Experiment Report

Last updated: 2026-05-14, after Fair 1R and 1R-STS gate evaluations

This report is the corrected track for fair experiments where the latent-attention block is trained from reinitialized weights using open-source data. It exists because the previous report, `experiments/latent_attention_experiment_report.md`, used continuation training from the released latent block and is not a fair open-data reproduction.

## Core Correction

The old Experiment 1b/v-series configs used:

```json
"reinit_latent": false
```

and usually had no `initial_latent_checkpoint`. In the trainer this means:

```text
load ai-sage/Giga-Embeddings-instruct
keep released latent-attention weights
train latent block further
```

The fair setup must instead be:

```json
"initial_latent_checkpoint": null,
"reinit_latent": true,
"freeze_llm": true
```

This means:

```text
load frozen released LLM/backbone
randomly reinitialize latent-attention block
train only the latent-attention block on open-source data
```

The backbone still comes from the released model, so this is not full model pretraining from scratch. It is a fair test of whether the **latent compression module** can be trained from random weights using the open datasets available to us.

## Research Question

Can the original latent-attention block, initialized randomly and trained only on open-source data, recover enough quality to serve as a fair baseline for architectural experiments 2-4?

Secondary question:

Can the STS-v12/v16-hard curriculum still work when the latent block does not start from the released latent weights?

## Evaluation Rules

Primary benchmark:

| Item | Setting |
| --- | --- |
| Benchmark | ruMTEB v1.1 through `mteb` |
| Evaluation script | `scripts/run_rumteb_eval.py` |
| Summary script | `scripts/summarize_rumteb_results.py` |
| Batch size | 12 |
| Evaluation max length | 4096 |
| GPU | NVIDIA RTX 4500 Ada, 24 GB VRAM |
| Clean scope | required |
| Contaminated scope | tracked but not used for promotion |

Promotion decisions must use clean tasks only. If a reinitialized checkpoint beats the released model on clean ruMTEB, assume leakage or evaluation error until proven otherwise.

## Required Baselines

| Baseline | Purpose | Status |
| --- | --- | --- |
| Released model | Upper reference with original proprietary pipeline | already evaluated in old report |
| Old continuation 1b | Measures finetuning from released latent weights | recorded in old report, not fair |
| Reinit 1R | Fair original-architecture open-data control | trained and gate-evaluated |
| Reinit 1R-STS/v16-hard curriculum | Best current open-data curriculum, but from random latent weights | trained and gate-evaluated |

## Fair Experiment 1R: Original Latent Block Reinitialized

Status: trained and gate-evaluated.

Purpose: train the original latent-attention architecture from reinitialized latent weights on the same broad open-RU data used by the old Experiment 1b. This is the corrected fair control.

Configuration requirements:

| Item | Value |
| --- | --- |
| Architecture | original latent-attention block |
| Base model | `ai-sage/Giga-Embeddings-instruct` |
| LLM backbone | frozen |
| Latent block | reinitialized |
| `reinit_latent` | `true` |
| `initial_latent_checkpoint` | absent or `null` |
| Training data | `data/contrastive/open_ru_train.jsonl` |
| Max length | 4096 |
| Batch size | attempted 12, OOM at full 4096 context; rerun with 8 |
| Evaluation | clean ruMTEB, diagnostics |

Observed training:

| Item | Value |
| --- | --- |
| Config path | `configs/experiments/exp01r_reinit_latent_memory_open_ru_4096.json` |
| Output checkpoint | `experiments/exp01_reinit_fair/checkpoints/open_ru_1r_4096/latest.pt` |
| Initial latent checkpoint | `null` |
| `reinit_latent` | `true` |
| Data | `data/contrastive/open_ru_train.jsonl` |
| Max length | 4096 |
| Batch size | 8 |
| Steps | 1000 |
| LR | 1e-5 |
| Runtime | 26.14 min |
| Mean loss | 1.018038 |
| Last loss | 0.000083 |

Note: the first attempt used batch size 12 and failed with CUDA OOM. Context length was not reduced.

Clean gate evaluation:

| Item | Value |
| --- | --- |
| Results directory | `results/rumteb/maxlen-4096-prompt-clean-reinit-fair-1r-gate` |
| Evaluation scope | clean |
| Tasks | `GeoreviewClusteringP2P`, `STS22` |
| Batch size | 12 |
| Max length | 4096 |
| Average | 0.693878 |
| GeoreviewClusteringP2P | 0.732767 |
| STS22 | 0.654989 |

Full ruMTEB evaluation:

| Item | Value |
| --- | --- |
| Results directory | `results/rumteb/maxlen-4096-prompt-all-reinit-fair-1r-full-rumteb` |
| Evaluation scope | all, with clean/contaminated split from manifest |
| Task count | 23 |
| Batch size | 12 |
| Max length | 4096 |
| Runtime | approximately 6h 53m wall time |
| All-task average | 0.735943 |
| Clean average | 0.739934 |
| Contaminated average | 0.731588 |

Comparison with released model run `results/rumteb/maxlen-4096-prompt`:

| Scope | Reinit 1R | Released model | Delta |
| --- | ---: | ---: | ---: |
| all | 0.735943 | 0.735448 | +0.000494 |
| clean | 0.739934 | 0.738606 | +0.001328 |
| contaminated | 0.731588 | 0.732003 | -0.000416 |

Category comparison with released model:

| Category | Reinit 1R | Released model | Delta |
| --- | ---: | ---: | ---: |
| Classification | 0.721605 | 0.720526 | +0.001079 |
| Clustering | 0.672662 | 0.671674 | +0.000988 |
| Reranking | 0.739735 | 0.740430 | -0.000695 |
| Retrieval | 0.813027 | 0.814010 | -0.000983 |
| STS/NLI | 0.763123 | 0.762904 | +0.000219 |

Per-task comparison with released model:

| Task | Scope | Reinit 1R | Released model | Delta |
| --- | --- | ---: | ---: | ---: |
| `CEDRClassification` | clean | 0.653029 | 0.652922 | +0.000107 |
| `GeoreviewClassification` | clean | 0.581201 | 0.578418 | +0.002783 |
| `GeoreviewClusteringP2P` | clean | 0.732767 | 0.727023 | +0.005744 |
| `HeadlineClassification` | clean | 0.893701 | 0.894336 | -0.000635 |
| `InappropriatenessClassification` | clean | 0.785303 | 0.787012 | -0.001709 |
| `KinopoiskClassification` | clean | 0.721800 | 0.718333 | +0.003467 |
| `MIRACLReranking` | contaminated | 0.675700 | 0.675510 | +0.000190 |
| `MIRACLRetrievalHardNegatives.v2` | contaminated | 0.745300 | 0.746190 | -0.000890 |
| `MassiveIntentClassification` | clean | 0.848502 | 0.845744 | +0.002758 |
| `MassiveScenarioClassification` | clean | 0.911488 | 0.910495 | +0.000992 |
| `RUParaPhraserSTS` | contaminated | 0.780782 | 0.781025 | -0.000243 |
| `RiaNewsRetrievalHardNegatives.v2` | clean | 0.887450 | 0.889040 | -0.001590 |
| `RuBQReranking` | contaminated | 0.803770 | 0.805350 | -0.001580 |
| `RuBQRetrieval` | contaminated | 0.806330 | 0.806800 | -0.000470 |
| `RuReviewsClassification` | clean | 0.768701 | 0.767139 | +0.001562 |
| `RuSTSBenchmarkSTS` | contaminated | 0.838228 | 0.835041 | +0.003187 |
| `RuSciBenchGRNTIClassification` | contaminated | 0.747559 | 0.746777 | +0.000782 |
| `RuSciBenchGRNTIClusteringP2P` | contaminated | 0.703600 | 0.706869 | -0.003269 |
| `RuSciBenchOECDClassification` | contaminated | 0.586084 | 0.583545 | +0.002539 |
| `RuSciBenchOECDClusteringP2P` | contaminated | 0.581618 | 0.581129 | +0.000489 |
| `STS22` | clean | 0.654989 | 0.651749 | +0.003240 |
| `SensitiveTopicsClassification` | clean | 0.440283 | 0.441064 | -0.000781 |
| `TERRa` | contaminated | 0.778494 | 0.783801 | -0.005307 |

Semantic diagnostics:

| Case | Category | Margin | Passed |
| --- | --- | ---: | --- |
| `negation_medical` | negation | 0.032020 | yes |
| `role_reversal_contract` | argument roles | -0.043627 | no |
| `contrastive_policy` | contrast | -0.134497 | no |
| `hierarchy_tax` | hierarchy | 0.049394 | yes |
| `temporal_order` | temporal | 0.034912 | yes |
| `numeric_threshold` | numeric | -0.177023 | no |
| `multi_hop_science` | multi-hop | 0.132821 | yes |
| `distractor_long` | distractor | -0.035036 | no |

Diagnostics result: 4/8 passed.

Observed interpretation:

Reinit 1R is unexpectedly strong. On the full 23-task ruMTEB pass it is essentially tied with the released model and is slightly higher on the clean average. The deltas are small enough that this should be treated as parity, not a meaningful win. The important result is that a randomly reinitialized latent block trained on broad open-source data can recover released-model ruMTEB performance when the frozen backbone is retained.

## Fair Experiment 1R-STS: Reinit STS/Recovery Curriculum

Status: trained and gate-evaluated.

Purpose: test whether the best discovered curriculum can recover balanced quality without released latent initialization.

Proposed continuous curriculum:

```text
random latent block
  -> broad open-RU training
  -> STS/paraphrase/hard-negative recovery
  -> targeted v12 recovery pass
  -> hard-filtered Habr QA SBS recovery
```

Datasets/stages:

| Stage | Data | Role |
| --- | --- | --- |
| 1 | `data/contrastive/open_ru_train.jsonl` | broad retrieval/classification/clustering control |
| 2 | STS-v6/v9 style STS mix | STS, paraphrase, hard negatives |
| 3 | `data/contrastive/open_ru_recovery_v12_2134_per_source.jsonl` | targeted broad recovery |
| 4 | `data/contrastive/open_ru_sts_v16_habr_qa_sbs_hard.jsonl` | hard QA preference recovery |

Important: stages should be run from the reinitialized 1R checkpoint lineage only. Do not load any old continuation checkpoint such as `open_ru_sts_v12_v9_targeted_recovery_1600_1pass/latest.pt`.

Executed curriculum:

| Stage | Config | Initial checkpoint | Data | Steps | Batch | LR | Runtime | Mean loss |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 broad 1R | `configs/experiments/exp01r_reinit_latent_memory_open_ru_4096.json` | `null` | `data/contrastive/open_ru_train.jsonl` | 1000 | 8 | 1e-5 | 26.14 min | 1.018038 |
| 2 STS-v9 anchor | `configs/experiments/exp01r_reinit_latent_memory_sts_v9_anchor_open_ru_4096.json` | `experiments/exp01_reinit_fair/checkpoints/open_ru_1r_4096/latest.pt` | `data/contrastive/open_ru_sts_v6_train.jsonl` | 2250 | 12 soft / 4 hard | stage config | 20.57 min | 0.189983 |
| 3 v12 targeted recovery | `configs/experiments/exp01r_reinit_latent_memory_sts_v12_targeted_recovery_1600_4096.json` | `experiments/exp01_reinit_fair/checkpoints/open_ru_sts_v9_anchor_4096/latest.pt` | `data/contrastive/open_ru_recovery_v12_2134_per_source.jsonl` | 1600 | 4 | 2e-6 | 20.43 min | 1.046268 |
| 4 Habr QA SBS hard | `configs/experiments/exp01r_reinit_latent_memory_sts_v16_habr_qa_sbs_hard_4096.json` | `experiments/exp01_reinit_fair/checkpoints/open_ru_sts_v12_targeted_recovery_1600_4096/latest.pt` | `data/contrastive/open_ru_sts_v16_habr_qa_sbs_hard.jsonl` | 2588 | 4 | 3e-6 | 9.65 min | 1.502822 |

Final checkpoint:

`experiments/exp01_reinit_fair/checkpoints/open_ru_sts_v16_habr_qa_sbs_hard_4096/latest.pt`

Training manifest:

`configs/training_manifests/open_ru_reinit_fair_sts_v16_hard.json`

Clean gate evaluation:

| Item | Value |
| --- | --- |
| Results directory | `results/rumteb/maxlen-4096-prompt-clean-reinit-fair-sts-v16-habr-qa-sbs-hard-gate` |
| Evaluation scope | clean |
| Tasks | `GeoreviewClusteringP2P`, `STS22` |
| Batch size | 12 |
| Max length | 4096 |
| Average | 0.668339 |
| GeoreviewClusteringP2P | 0.694955 |
| STS22 | 0.641723 |

Semantic diagnostics:

| Case | Category | Margin | Passed |
| --- | --- | ---: | --- |
| `negation_medical` | negation | 0.107817 | yes |
| `role_reversal_contract` | argument roles | -0.020606 | no |
| `contrastive_policy` | contrast | -0.041171 | no |
| `hierarchy_tax` | hierarchy | 0.026668 | yes |
| `temporal_order` | temporal | 0.022419 | yes |
| `numeric_threshold` | numeric | -0.133292 | no |
| `multi_hop_science` | multi-hop | 0.140190 | yes |
| `distractor_long` | distractor | -0.019306 | no |

Diagnostics result: 4/8 passed.

RIA evaluation was deferred for this first gate pass. It is much more expensive, and the Georeview score already shows broad-task damage relative to old continuation baselines. Run RIA only if we need a full table for this checkpoint.

## Experiment 2R: Hierarchical Latent Block Reinitialized

Status: planned after fair 1R.

Purpose: compare hierarchical latent compression against a true fair original-architecture reinit baseline.

Configuration requirements:

| Item | Value |
| --- | --- |
| Architecture | hierarchical latent attention |
| Stage 1 latents | 512 |
| Stage 2 latents | 128 |
| LLM backbone | frozen |
| Latent modules | initialized without released latent inheritance unless explicitly testing transfer |
| Training data | same as fair 1R |
| Comparison target | fair 1R, not old Experiment 1b |

## Reporting Template

For each fair run, record:

| Item | Value |
| --- | --- |
| Config path | |
| Initial latent checkpoint | must be `null` for first fair run |
| `reinit_latent` | must be `true` for first fair run |
| Trainable parameters | latent block only |
| Dataset manifest | |
| Steps | |
| Batch size | |
| Max length | |
| LR | |
| Runtime | |
| Mean loss | |
| Diagnostics | |
| Clean ruMTEB results | |
| Contamination notes | |

Comparison table:

| Model | GeoreviewClusteringP2P | STS22 | RiaNewsRetrievalHardNegatives.v2 | Clean average |
| --- | ---: | ---: | ---: | ---: |
| Released model | 0.727023 | 0.651749 | 0.889040 | 0.735448 all / 0.738606 clean |
| Old continuation 1b | 0.729729 | 0.639772 | 0.887210 | TBD |
| Old continuation v16-hard | 0.718168 | 0.659734 | 0.878090 | TBD |
| Reinit 1R | 0.732767 | 0.654989 | 0.887450 | 0.735943 all / 0.739934 clean |
| Reinit 1R-STS/v16-hard curriculum | 0.694955 | 0.641723 | not run | 0.668339 over gate tasks |

## Current Conclusion

Fair 1R is currently the stronger corrected baseline. It reaches Georeview 0.732767, STS22 0.654989, and RIA 0.887450. On the full 23-task ruMTEB pass, it is at practical parity with the released model: 0.735943 vs 0.735448 all-task average, and 0.739934 vs 0.738606 clean average. The deltas are too small to claim improvement, but strong enough to validate Fair 1R as the baseline for architectural experiments.

This changes the interpretation of the previous 1R-STS result. The problem is not that random latent initialization cannot recover the gate tasks; it can. The problem is that the old STS-v9/v12/v16 sequence over-specializes or destabilizes a newly learned latent compressor. The likely mechanism is catastrophic forgetting or mismatch between broad contrastive training and later narrow hard-negative QA/STS pressure.

Before moving to architectural experiments 2R-4R, the minimum next checks are:

1. Treat Reinit 1R, not 1R-STS, as the current fair Experiment 1 baseline.
2. For 2R-4R, compare against the full Reinit 1R ruMTEB result, not only the two-task gate.
3. If we revisit STS tuning, use much lighter updates from 1R: lower LR, fewer steps, stronger broad-data rehearsal, and validation-based early stopping against clean Georeview/STS/RIA instead of a fixed late-stage curriculum.

## Mix K: Clean Sensitive-Topic Discrimination

Status: completed as a targeted fair ablation.

Purpose: reduce the largest negative delta on `SensitiveTopicsClassification` without using contaminated records from `NiGuLa/Russian_Sensitive_Topics` or the official `ai-forever/sensitive-topics-classification` / `mteb/SensitiveTopicsClassification` data.

Contamination audit outcome:

| Dataset | Decision | Reason |
| --- | --- | --- |
| `NiGuLa/Russian_Sensitive_Topics` | rejected | exact normalized overlap with RuMTEB SensitiveTopics train/test |
| `ai-forever/sensitive-topics-classification` | rejected | target benchmark dataset |
| `mteb/SensitiveTopicsClassification` | rejected | target benchmark dataset |
| `mvrcii/safety-moderation-benchmark` | accepted | no exact normalized overlap found |
| `Vikhrmodels/Veles-2.5` | accepted | lexical topic source, no exact normalized overlap found |
| `Mnwa/russian-toxic` | accepted after filtering | 54 exact normalized benchmark overlaps removed |
| `textdetox/multilingual_toxicity_dataset` `ru` | accepted | no exact normalized overlap found |
| `ucberkeley-dlab/measuring-hate-speech` | accepted | English auxiliary source, no exact normalized overlap found |

Training:

| Stage | Config | Data | Steps | Batch | Max length | LR | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mix K broad stage | `configs/experiments/exp01r_nc_mixk_geracl2_habr1_deepvk1_groundedstrict1_sensitive1_4096.json` | GeRaCl:Habr:DeepVK:GroundedStrict:Sensitive = 2:1:1:1:1 | 4800 | 4 | 4096 | 1e-5 | 103.20 min |
| GeRaCl remaining recovery | `configs/experiments/exp01r_nc_mixk_plus_geracl_remaining_seed113_4096.json` | GeRaCl records not used in Mix K seed 113 | 1200 | 8 | 4096 | 3e-6 | 26.61 min |

Generated data:

| File | Rows |
| --- | ---: |
| `data/contrastive/open_ru_1r_nc_sensitive_topic_discrimination_3200.jsonl` | 3200 |
| `data/contrastive/open_ru_1r_nc_mixk_geracl2_habr1_deepvk1_groundedstrict1_sensitive1_19200.jsonl` | 19200 |
| `data/contrastive/open_ru_1r_nc_mixk_geracl_remaining_seed113_9600.jsonl` | 9600 |

Fast gate result:

| Task | Mix K | Fair Mix F seed 53 | Official released | Delta vs Mix F | Delta vs official |
| --- | ---: | ---: | ---: | ---: | ---: |
| SensitiveTopicsClassification | 0.442139 | 0.436816 | 0.441064 | +0.005323 | +0.001075 |
| RuSciBenchGRNTIClusteringP2P | 0.700102 | 0.708029 | 0.706869 | -0.007927 | -0.006767 |
| RuSciBenchOECDClusteringP2P | 0.577761 | 0.578695 | 0.581129 | -0.000934 | -0.003368 |
| STS22 | 0.653935 | 0.663461 | 0.651749 | -0.009526 | +0.002186 |
| TERRa | 0.783054 | 0.773285 | 0.783801 | +0.009769 | -0.000747 |
| Gate average | 0.631398 | 0.632057 | 0.632922 | -0.000659 | -0.001524 |

Conclusion: Mix K does not pass the proposed gate. The clean sensitive-topic component works for `SensitiveTopicsClassification` and also improves `TERRa`, but it damages `RuSciBenchGRNTIClusteringP2P` and `STS22` too much. The next sensitive-topic attempt should keep the clean sensitive component, but move it to a lighter late-stage adapter-style pass or lower its first-stage weight while increasing scientific/topic clustering rehearsal.

## SensitiveTopicDiscrimination Source Probes

Status: completed as marginal source-isolation probes from the current best fair Mix F checkpoint.

Purpose: determine which underlying clean sensitive-topic sources are helpful or harmful before building another full mixed-stage experiment. Each probe used only one `SensitiveTopicDiscrimination` source, trained for 800 steps from `open_ru_1r_nc_mixf_plus_geracl_remaining_seed53_4096/latest.pt`, then evaluated on the five-task gate.

Detailed report: `results/rumteb/sensitive_source_probe_comparison.md`.

| Probe | Gate average | Delta vs Mix F | SensitiveTopicsClassification delta | GRNTI delta | OECD delta | TERRa delta | STS22 delta | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MVR-CII | 0.633162 | +0.001105 | +0.000098 | +0.002160 | +0.002006 | +0.001668 | -0.000409 | helpful |
| Veles | 0.631506 | -0.000551 | -0.001171 | -0.001793 | -0.000647 | -0.001436 | +0.002291 | harmful |
| MNWA | 0.631566 | -0.000491 | +0.001416 | -0.005420 | +0.000870 | +0.001761 | -0.001081 | harmful |
| TextDetox | 0.632065 | +0.000008 | +0.001221 | -0.003978 | +0.001585 | +0.001668 | -0.000455 | neutral |
| UC Berkeley | 0.633304 | +0.001246 | +0.000489 | +0.003639 | +0.000348 | +0.001775 | -0.000019 | helpful |
| MVR-CII + UC Berkeley | 0.633147 | +0.001090 | +0.000928 | +0.003725 | -0.000907 | +0.001476 | +0.000226 | helpful |

Conclusion: do not keep the full Mix K sensitive pool unchanged. The best single source is UC Berkeley by gate average. The combined MVR-CII + UC Berkeley probe is also helpful and gives the strongest `SensitiveTopicsClassification` and GRNTI values among the helpful probes, but it loses enough on OECD and TERRa that its average is slightly below UC Berkeley alone. TextDetox is usable only cautiously because it helps sensitive-topic classification but hurts GRNTI. Veles and MNWA should be excluded from the next mixed experiment unless used at very low weight with explicit clustering rehearsal.

## Mix L: TERRa + Clean Sensitive-Topic Variant

Status: completed as a five-task gate ablation.

Purpose: keep the useful clean sensitive-topic sources from the source probes (`mvrcii/safety-moderation-benchmark` + `ucberkeley-dlab/measuring-hate-speech`) while increasing the TERRa/Habr signal and preserving broad GeRaCl recovery.

Training:

| Stage | Config | Data | Steps | Batch | Max length | LR | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mix L broad stage | `configs/experiments/exp01r_nc_mixl_terra_sensitive_4096.json` | GeRaCl 12800 + Habr harder all 4369 + DeepVK all 3451 + GroundedStrict 3200 + Sensitive MVR-CII/UC Berkeley 3200 | 6755 | 4 | 4096 | 1e-5 | 115.6 min |
| GeRaCl remaining recovery | same staged config | remaining GeRaCl seed 131, 3200 rows | 400 | 8 | 4096 | 3e-6 | 8.7 min |

Generated data:

| File | Rows |
| --- | ---: |
| `data/contrastive/open_ru_1r_nc_mixl_geracl12800_habrall_deepvkall_grounded3200_sensitive_mvrcii_ucb_27020.jsonl` | 27020 |
| `data/contrastive/open_ru_1r_nc_mixl_geracl_remaining_seed131_3200.jsonl` | 3200 |

Five-task gate result:

| Task | Mix L | Fair Mix F seed 53 | Official released | Delta vs Mix F | Delta vs official |
| --- | ---: | ---: | ---: | ---: | ---: |
| SensitiveTopicsClassification | 0.438330 | 0.436816 | 0.441064 | +0.001514 | -0.002734 |
| RuSciBenchGRNTIClusteringP2P | 0.696573 | 0.708029 | 0.706869 | -0.011456 | -0.010296 |
| RuSciBenchOECDClusteringP2P | 0.571308 | 0.578695 | 0.581129 | -0.007387 | -0.009821 |
| TERRa | 0.780496 | 0.773285 | 0.783801 | +0.007211 | -0.003305 |
| STS22 | 0.654494 | 0.663461 | 0.651749 | -0.008967 | +0.002745 |
| Gate average | 0.628240 | 0.632057 | 0.632922 | -0.003817 | -0.004682 |

Conclusion: Mix L does not pass the gate. The added clean sensitive-topic pair and larger Habr/TERRa pressure improve `SensitiveTopicsClassification` and `TERRa` versus Mix F, but the loss on both RuSciBench clustering tasks and STS22 is too large. This suggests the current mix is over-weighting pairwise/topical discrimination at the expense of scientific taxonomy geometry. Do not run a full ruMTEB pass for Mix L unless a later question specifically needs it.

## Mix M: Full-Use Quality-Filtered Variants

Status: two variants completed; one strict variant prepared but not trained.

Purpose: test whether the Mix K/L pattern was caused by arbitrary caps and repeated passes. Mix M uses the requested `GeRaCl:Habr:DeepVK:Grounded:Sensitive = 2:1:1:1:1` base, consumes every available non-GeRaCl record after quality filtering, and uses the remaining GeRaCl rows in stage 2. This keeps the experiment close to a single-pass curriculum.

Filtering variants:

| Variant | Main idea | Stage 1 rows | Stage 2 GeRaCl rows | Steps | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `basic` | normalize, dedupe, remove malformed/short/repetitive text, keep all valid negatives | 20618 | 9540 | 5154 + 1192 | trained |
| `sim_guard` | `basic` plus lexical false-negative pruning for near-duplicate negatives | 20616 | 9528 | 5154 + 1191 | trained |
| `strict` | stronger text-quality and query-positive similarity filters | 18659 | 4049 | 4664 + 506 | not trained; likely over-filters GeRaCl |

Training:

| Run | Config | Runtime | Notes |
| --- | --- | ---: | --- |
| Mix M `basic` | `configs/experiments/exp01r_nc_mixm_basic_full_quality_4096.json` | 126.61 min | completed from reinitialized latent block |
| Mix M `sim_guard` | `configs/experiments/exp01r_nc_mixm_sim_guard_full_quality_4096.json` | 126.47 min | completed from reinitialized latent block |

Five-task gate:

| Task | Official released | Fair Mix F seed 53 | Mix M basic | Mix M sim_guard | Basic delta vs Mix F | Basic delta vs official |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SensitiveTopicsClassification | 0.441064 | 0.436816 | 0.441943 | 0.439941 | +0.005127 | +0.000879 |
| RuSciBenchGRNTIClusteringP2P | 0.706869 | 0.708029 | 0.699692 | 0.691438 | -0.008337 | -0.007177 |
| RuSciBenchOECDClusteringP2P | 0.581129 | 0.578695 | 0.575950 | 0.572812 | -0.002745 | -0.005179 |
| TERRa | 0.783801 | 0.773285 | 0.783280 | 0.781679 | +0.009995 | -0.000521 |
| STS22 | 0.651749 | 0.663461 | 0.653641 | 0.654350 | -0.009820 | +0.001892 |
| Gate average | 0.632922 | 0.632057 | 0.630901 | 0.628044 | -0.001156 | -0.002021 |

Conclusion: full-use quality filtering is better than Mix L and better than false-negative pruning, but still does not beat Mix F or the released gate average. The useful signal is clear: SensitiveTopics and TERRa improve strongly, but the full-use recipe still weakens RuSciBench clustering and STS. `sim_guard` made clustering worse than `basic`, so the lexical false-negative pruning rule is too crude for these sources. The prepared `strict` variant should not be prioritized because its query-positive lexical-overlap rule removes many GeRaCl examples whose labels are intentionally taxonomy-style rather than paraphrase-style.

Next correction: keep Mix F as the baseline. If we want the Sensitive/TERRa gains, add them as a smaller auxiliary stage with explicit clustering rehearsal or lower weight, not by fully expanding all pairwise/topical sources in the main stage. Better false-negative filtering should use model/cross-encoder scores or positive-aware mining rather than lexical overlap alone; this matches the direction used in NV-Retriever and Sentence Transformers hard-negative tooling.

## Mix N: Three-Stage Low-LR Correction

Status: completed as a five-task gate ablation.

Purpose: test whether the useful `SensitiveTopicsClassification` and `TERRa` gains can be added after the stronger Mix F curriculum without the larger clustering damage seen in Mix K/L/M. The recipe keeps Mix F broad training, then the GeRaCl remaining recovery stage, then adds a small low-LR correction stage with heavy GeRaCl rehearsal.

Detailed comparison: `results/rumteb/mixn_three_stage_gate_comparison.md`.

Training:

| Stage | Config | Data | Steps | Batch | Max length | LR | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mix F broad stage | `configs/experiments/exp01r_nc_mixn_three_stage_correction_4096.json` | GeRaCl:Habr:DeepVK:GroundedStrict = 2:1:1:1 | 4000 | 4 | 4096 | 1e-5 | included below |
| GeRaCl remaining recovery | same staged config | remaining GeRaCl seed 53 | 1200 | 8 | 4096 | 3e-6 | included below |
| Low-LR correction | same staged config | GeRaCl rehearsal:Habr:MVR-CII:UC Berkeley = 4:1:0.5:0.5 | 1200 | 4 | 4096 | 2e-6 | included below |
| Total | same staged config | three stages | 6400 | mixed | 4096 | mixed | 125.34 min |

Generated data:

| File | Rows |
| --- | ---: |
| `data/contrastive/open_ru_1r_nc_mixn_correction_geracl3200_habr800_mvrcii400_ucb400_seed191.jsonl` | 4800 |

Five-task gate:

| Task | Official released | Fair Mix F seed 53 | Mix M basic | Mix N | Delta vs Mix F | Delta vs official |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SensitiveTopicsClassification | 0.441064 | 0.436816 | 0.441943 | 0.438232 | +0.001416 | -0.002832 |
| RuSciBenchGRNTIClusteringP2P | 0.706869 | 0.708029 | 0.699692 | 0.703228 | -0.004801 | -0.003641 |
| RuSciBenchOECDClusteringP2P | 0.581129 | 0.578695 | 0.575950 | 0.574109 | -0.004586 | -0.007020 |
| TERRa | 0.783801 | 0.773285 | 0.783280 | 0.782282 | +0.008997 | -0.001519 |
| STS22 | 0.651749 | 0.663461 | 0.653641 | 0.654000 | -0.009461 | +0.002251 |
| Gate average | 0.632922 | 0.632057 | 0.630901 | 0.630370 | -0.001687 | -0.002552 |

Conclusion: Mix N does not pass the gate. The three-stage correction improves `TERRa` strongly versus Mix F and slightly improves `SensitiveTopicsClassification`, but it still damages both RuSciBench clustering tasks and gives up most of Mix F's STS gain. This means the late correction idea works directionally for pair-classification behavior, but the current correction mixture is too broad and still distorts taxonomy geometry.

Next correction: keep Mix F as the broad baseline. To improve the five-task gate, the next attempt should avoid another full mixed sensitive/TERRa correction. Prefer a task-targeted small correction with explicit scientific-taxonomy rehearsal, or split objectives by stage: one stage for taxonomy/clustering preservation, then a very small low-LR pair-classification calibration. OECD is currently the most fragile task, so any correction should be rejected early if OECD drops more than about 0.002 from Mix F.

## Mix O: Continuation Probes From Mix F

Status: completed as six five-task gate ablations.

Purpose: test whether small continuation stages from the strong Mix F checkpoint can recover the remaining five-task gaps against the released model without retraining a full curriculum.

Detailed comparison: `results/rumteb/mixo_continuation_probe_gate_comparison.md`.

Base checkpoint:

`experiments/exp01_reinit_fair/checkpoints/open_ru_1r_nc_mixf_plus_geracl_remaining_seed53_4096/latest.pt`

Generated helper:

`scripts/prepare_open_ru_1r_nc_mixo_five_task_probes.py`

Generated data summary:

`data/contrastive/open_ru_1r_nc_mixo_five_task_probes_summary.json`

Probe recipes:

| Probe | Config | Continuation data | Steps | Batch | LR |
| --- | --- | --- | ---: | ---: | ---: |
| `pair300` | `configs/experiments/exp01r_nc_mixo_pair_calibration_300_4096.json` | GeRaCl:Habr:UC Berkeley:MVR-CII = 8:2:1:1 | 300 | 4 | 1e-6 |
| `pair600` | `configs/experiments/exp01r_nc_mixo_pair_calibration_600_4096.json` | GeRaCl:Habr:UC Berkeley:MVR-CII = 8:2:1:1 | 600 | 4 | 1e-6 |
| `pair900` | `configs/experiments/exp01r_nc_mixo_pair_calibration_900_4096.json` | GeRaCl:Habr:UC Berkeley:MVR-CII = 8:2:1:1 | 900 | 4 | 1e-6 |
| `tax300` | `configs/experiments/exp01r_nc_mixo_taxonomy_repair_300_4096.json` | GeRaCl:DeepVK:Grounded = 6:2:1 | 300 | 4 | 1e-6 |
| `tax600` | `configs/experiments/exp01r_nc_mixo_taxonomy_repair_600_4096.json` | GeRaCl:DeepVK:Grounded = 6:2:1 | 600 | 4 | 1e-6 |
| `alt400+400` | `configs/experiments/exp01r_nc_mixo_alternating_tax400_pair400_4096.json` | taxonomy repair 400 steps, then pair calibration 400 steps | 800 | 4 | 1e-6 then 5e-7 |

Five-task gate:

| Run | GateAvg | Sensitive | GRNTI | OECD | TERRa | STS22 | Worst delta vs released | Pass all released? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Official released | 0.632922 | 0.441064 | 0.706869 | 0.581129 | 0.783801 | 0.651749 | +0.000000 | yes |
| Fair Mix F seed 53 | 0.632057 | 0.436816 | 0.708029 | 0.578695 | 0.773285 | 0.663461 | -0.010516 | no |
| `pair300` | 0.632098 | 0.436816 | 0.708225 | 0.578481 | 0.773310 | 0.663659 | -0.010491 | no |
| `pair600` | 0.631998 | 0.436816 | 0.707854 | 0.578790 | 0.772873 | 0.663659 | -0.010928 | no |
| `pair900` | 0.632172 | 0.436670 | 0.708935 | 0.578458 | 0.773236 | 0.663560 | -0.010565 | no |
| `tax300` | 0.631893 | 0.436865 | 0.707422 | 0.578681 | 0.772840 | 0.663659 | -0.010961 | no |
| `tax600` | 0.632309 | 0.436816 | 0.710124 | 0.578019 | 0.772971 | 0.663615 | -0.010830 | no |
| `alt400+400` | 0.632209 | 0.436816 | 0.709175 | 0.578611 | 0.772784 | 0.663659 | -0.011017 | no |

Conclusion: none of the continuation probes passed the released model on all five gate tasks. The best gate average was `tax600` at `0.632309`, above Mix F by `+0.000252` but still below the released checkpoint by `-0.000613`. The continuation probes can improve GRNTI, especially `tax600`, and preserve the STS22 advantage, but they do not fix the main gaps: `TERRa`, `SensitiveTopicsClassification`, and OECD remain below released.

Next correction: do not scale these continuation probes directly. Low-LR continuation from Mix F is too weak to repair pair-classification and sensitive-topic behavior once the Mix F geometry is established. The next productive direction should change the main curriculum proportions or add task-aligned sources earlier, while preserving explicit scientific-taxonomy rehearsal.
