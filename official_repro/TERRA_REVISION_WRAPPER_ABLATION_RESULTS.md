# TERRa Revision and Wrapper Ablation Results

Date: 2026-06-12

Evaluator:

- Script: `official_repro/run_official_rumteb.py`
- Task: `TERRa`
- Split: `dev`
- Seed: `8`
- Reset seed per task: yes
- Attention implementation: `eager`
- Torch dtype: `bfloat16`
- Max length: `4096`
- Batch size: `1`

Online reference from current `mteb/results`:

| Source | Revision | TERRa |
|---|---|---:|
| Online official row | `0ad5b29bfecd806cecc9d66b927d828a736594dc` | 0.795677 |

## Results

| Run | Revision | Wrapper mode | TERRa | Delta vs online official |
|---|---|---|---:|---:|
| `terra_rev40b_nowrapper` | `40b27667b9ad586d7812675df76e5062ccc80b0e` | none | 0.598308 | -0.197369 |
| `terra_rev40b_frozen_legacyru` | `40b27667b9ad586d7812675df76e5062ccc80b0e` | frozen `legacy_ru` | 0.642196 | -0.153481 |
| `terra_rev40b_instruction_nli` | `40b27667b9ad586d7812675df76e5062ccc80b0e` | instruction NLI prompt | 0.601480 | -0.194197 |
| `terra_rev0ad_nowrapper` | `0ad5b29bfecd806cecc9d66b927d828a736594dc` | none | 0.645565 | -0.150112 |
| `terra_rev0ad_frozen_legacyru` | `0ad5b29bfecd806cecc9d66b927d828a736594dc` | frozen `legacy_ru` | 0.675025 | -0.120652 |
| `terra_rev0ad_instruction_nli` | `0ad5b29bfecd806cecc9d66b927d828a736594dc` | instruction NLI prompt | 0.645565 | -0.150112 |

## Interpretation

The September revision `0ad5b29...` improves TERRa compared with the June revision `40b27667...`, but the improvement is not enough to reproduce the online official score.

The current frozen `legacy_ru` wrapper improves TERRa on both revisions:

- `40b27667...`: +0.043888 over no wrapper
- `0ad5b29...`: +0.029460 over no wrapper

The tested NLI instruction prompt does not explain the official score gap. It is worse than frozen `legacy_ru` for both revisions and equal to no-wrapper for `0ad5b29...`.

Current best local combination from this ablation:

| Revision | Wrapper mode | TERRa | Remaining gap |
|---|---|---:|---:|
| `0ad5b29bfecd806cecc9d66b927d828a736594dc` | frozen `legacy_ru` | 0.675025 | -0.120652 |

So the TERRa mismatch is not solved by switching to the online model revision or by the first simple NLI prompt. The remaining gap likely comes from another evaluation-side difference: task version, pair-classification evaluator behavior, hidden text preprocessing, different prompt application/masking, or another wrapper detail.

## Result Files

Full JSON result payloads are intentionally left under ignored `results/` paths:

- `results/official_repro/terra_rev40b_nowrapper/no_model_name_available/no_revision_available/TERRa.json`
- `results/official_repro/terra_rev40b_frozen_legacyru/no_model_name_available/no_revision_available/TERRa.json`
- `results/official_repro/terra_rev40b_instruction_nli/no_model_name_available/no_revision_available/TERRa.json`
- `results/official_repro/terra_rev0ad_nowrapper/no_model_name_available/no_revision_available/TERRa.json`
- `results/official_repro/terra_rev0ad_frozen_legacyru/no_model_name_available/no_revision_available/TERRa.json`
- `results/official_repro/terra_rev0ad_instruction_nli/no_model_name_available/no_revision_available/TERRa.json`
