# TERRa Pair Wrapper Probe Summary

Input: `results/official_repro/terra_pair_wrapper_probe_0ad.json`

Baseline frozen `legacy_ru` on `0ad5b29...`: `0.675025`
Target: `0.700000`

| Rank | Family | Variant | TERRa | Delta vs baseline | Delta vs target | Best metric |
|---:|---|---|---:|---:|---:|---|
| 1 | `variants` | `entails_direction` | 0.746343 | +0.071318 | +0.046343 | `cosine_ap` |
| 2 | `variants` | `nli_same` | 0.743200 | +0.068175 | +0.043200 | `cosine_ap` |
| 3 | `variants` | `nli_role_long` | 0.740921 | +0.065896 | +0.040921 | `manhattan_ap` |
| 4 | `variants` | `reverse_entails_direction` | 0.738565 | +0.063540 | +0.038565 | `manhattan_ap` |
| 5 | `variants` | `ru_premise_hypothesis` | 0.700136 | +0.025111 | +0.000136 | `manhattan_ap` |
| 6 | `variants` | `legacy_ru_same` | 0.675025 | +0.000000 | -0.024975 | `cosine_ap` |
| 7 | `variants` | `condition_conclusion` | 0.663742 | -0.011283 | -0.036258 | `cosine_ap` |
| 8 | `variants` | `reverse_premise_hypothesis` | 0.649254 | -0.025771 | -0.050746 | `manhattan_ap` |
| 9 | `variants` | `premise_hypothesis` | 0.646823 | -0.028202 | -0.053177 | `manhattan_ap` |
| 10 | `variants` | `none` | 0.645565 | -0.029460 | -0.054435 | `cosine_ap` |
| 11 | `pair_variants` | `pair_ru_semantic_nli` | 0.639950 | -0.035075 | -0.060050 | `positive_ap` |
| 12 | `pair_variants` | `pair_en_entailment` | 0.615635 | -0.059390 | -0.084365 | `margin_ap` |
| 13 | `pair_variants` | `pair_ru_entailment_short` | 0.608659 | -0.066366 | -0.091341 | `positive_ap` |
| 14 | `pair_variants` | `pair_ru_true_false` | 0.599355 | -0.075670 | -0.100645 | `positive_ap` |
| 15 | `pair_variants` | `pair_ru_contradiction_aware` | 0.595317 | -0.079708 | -0.104683 | `positive_ap` |
| 16 | `pair_variants` | `pair_ru_entailment_yes_no` | 0.568809 | -0.106216 | -0.131191 | `margin_ap` |

## Conclusion

Best variant: `variants/entails_direction` with TERRa `0.746343`.
This meets the target.

Winning pair-aware prefixes:

| TERRa side | Prefix |
|---|---|
| `sentence1` / premise | `Текст, из которого может следовать утверждение: ` |
| `sentence2` / hypothesis | `Утверждение, которое может следовать из текста: ` |

This result is not expressible through standard MTEB `PairClassificationEvaluator.encode()` alone, because MTEB encodes a deduplicated union of `sentence1 + sentence2` and therefore loses side information before calling the wrapper. The probe evaluates the same TERRa pairs and AP-style metrics, but keeps side-aware prefixes.
