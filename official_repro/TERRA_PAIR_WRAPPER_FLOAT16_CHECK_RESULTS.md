# TERRa Pair Wrapper Probe Summary

Input: `results/official_repro/terra_pair_wrapper_probe_0ad_best_float16.json`

Baseline frozen `legacy_ru` on `0ad5b29...`: `0.675025`
Target: `0.700000`
Online official TERRa: `0.795677`

| Rank | Family | Variant | TERRa | Delta vs baseline | Delta vs target | Delta vs official | Best metric |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `variants` | `logical_entails_proposition_fields` | 0.785562 | +0.110537 | +0.085562 | -0.010115 | `manhattan_ap` |
| 2 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:cosine` | 0.785044 | +0.110019 | +0.085044 | -0.010633 | `zavg` |
| 3 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:dot` | 0.785044 | +0.110019 | +0.085044 | -0.010633 | `zavg` |
| 4 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:euclidean` | 0.784790 | +0.109765 | +0.084790 | -0.010887 | `zavg` |
| 5 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:dot` | 0.784617 | +0.109592 | +0.084617 | -0.011060 | `zavg` |
| 6 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:cosine` | 0.784606 | +0.109581 | +0.084606 | -0.011071 | `rankavg` |
| 7 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:dot` | 0.784606 | +0.109581 | +0.084606 | -0.011071 | `rankavg` |
| 8 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:euclidean` | 0.784606 | +0.109581 | +0.084606 | -0.011071 | `rankavg` |
| 9 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:euclidean` | 0.784415 | +0.109390 | +0.084415 | -0.011262 | `zavg` |
| 10 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:dot+variant:logical_entails_proposition_fields:euclidean` | 0.784415 | +0.109390 | +0.084415 | -0.011262 | `zavg` |
| 11 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:dot+variant:logical_entails_proposition_fields:euclidean` | 0.784031 | +0.109006 | +0.084031 | -0.011646 | `zavg` |
| 12 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:dot+variant:logical_entails_proposition_fields:euclidean` | 0.783859 | +0.108834 | +0.083859 | -0.011818 | `rankavg` |
| 13 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:dot` | 0.783778 | +0.108753 | +0.083778 | -0.011899 | `rankavg` |
| 14 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:euclidean` | 0.783778 | +0.108753 | +0.083778 | -0.011899 | `rankavg` |
| 15 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:manhattan+variant:logical_entails_proposition_fields:dot+variant:logical_entails_proposition_fields:euclidean` | 0.783778 | +0.108753 | +0.083778 | -0.011899 | `rankavg` |
| 16 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:dot` | 0.783393 | +0.108368 | +0.083393 | -0.012284 | `zavg` |
| 17 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:dot` | 0.783393 | +0.108368 | +0.083393 | -0.012284 | `rankavg` |
| 18 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:euclidean` | 0.783393 | +0.108368 | +0.083393 | -0.012284 | `zavg` |
| 19 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:euclidean` | 0.783393 | +0.108368 | +0.083393 | -0.012284 | `rankavg` |
| 20 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:dot+variant:logical_entails_proposition_fields:euclidean` | 0.783393 | +0.108368 | +0.083393 | -0.012284 | `zavg` |
| 21 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:dot+variant:logical_entails_proposition_fields:euclidean` | 0.783393 | +0.108368 | +0.083393 | -0.012284 | `rankavg` |
| 22 | `ensembles` | `zavg:variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:dot+variant:logical_entails_proposition_fields:euclidean` | 0.783393 | +0.108368 | +0.083393 | -0.012284 | `zavg` |
| 23 | `ensembles` | `rankavg:variant:logical_entails_proposition_fields:cosine+variant:logical_entails_proposition_fields:dot+variant:logical_entails_proposition_fields:euclidean` | 0.783393 | +0.108368 | +0.083393 | -0.012284 | `rankavg` |

## Conclusion

Best variant: `variants/logical_entails_proposition_fields` with TERRa `0.785562`.
This meets the target.
