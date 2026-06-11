# ruMTEB Evaluation Coverage Table

Last updated: 2026-05-28

`avg / worst delta` means: average score on that eval subset, then the most negative per-task delta against the released model on the same tasks.

Rows marked `frozen-wrapper target 5 gate` use the current official-reproduction wrapper and the target gate:
`CEDRClassification`, `SensitiveTopicsClassification`, `TERRa`, `RuSciBenchGRNTIClusteringP2P`, `RuSciBenchOECDClusteringP2P`.

| Mix / model | Main training mix | Full 23 avg / worst delta | 18-task avg / worst delta | 6-task avg / worst delta | 5-task avg / worst delta | 2-task avg / worst delta | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Reinit 1R | old broad `open_ru_train` | 0.735943 / -0.005307 | - | - | - | 0.693878 / +0.003240 | fair reinit baseline |
| 1R-NC base | no-contamination broad start | - | - | 0.662878 / -0.000878 | - | - | fast gate |
| 1R-NC GeRaCl | GeRaCl only | 0.736547 / -0.010354 | - | 0.665485 / -0.002294 | - | - | full + fast gate |
| 1R-NC DeepVK | DeepVK only | - | - | 0.655280 / -0.012010 | - | - | fast gate |
| 1R-NC GroundedRAG | GroundedRAG only | - | - | 0.660779 / -0.001123 | - | - | fast gate |
| 1R-NC Habr hard | Habr hard only | - | - | 0.650517 / -0.020988 | - | - | fast gate |
| 1R-NC Promptriever | Promptriever only | - | - | 0.659739 / -0.003301 | - | - | fast gate |
| 1R-NC GeRaCl + Habr hard 2588 | GeRaCl, then Habr hard full pass | 0.735558 / -0.005565 | - | - | - | 0.740665 / -0.005565 | target2 also exists |
| Mix A | not found | - | - | - | - | - | no `mixa` result found |
| Mix B | GeRaCl:Habr = 2:1, then remaining GeRaCl | - | 0.721979 / -0.006518 | 0.700294 / -0.003279 | - | - | before remaining: 6-task 0.699124 |
| Mix C | GeRaCl:Habr:Grounded = 2:1:1, then remaining GeRaCl | - | 0.722128 / -0.008555 | - | - | - | short18 only |
| Mix D | GeRaCl:Habr:DeepVK = 2:1:1, then remaining GeRaCl | - | 0.722126 / -0.004930 | - | - | - | short18 only |
| Mix E | GeRaCl:Habr:DeepVK:physics = 2:1:1:1, then remaining GeRaCl | - | 0.720845 / -0.011605 | - | - | - | physics hurt average |
| Mix F | GeRaCl:Habr:DeepVK:GroundedStrict = 2:1:1:1, then remaining GeRaCl | 0.737025 / -0.010922 | - | - | - | 0.642431 / -0.003845 | original Mix F full run |
| Mix F seed53 | corrected Mix F with seed53 remaining GeRaCl | 0.737045 / -0.010516 | - | - | - | 0.643362 / -0.002434 | strongest full ruMTEB run |
| Mix G | GeRaCl:Habr:DeepVK:HabrExtra = 2:1:1:1, then remaining GeRaCl | - | - | - | - | 0.640578 / -0.004170 | target2 only |
| Mix H | GeRaCl:Habr:DeepVK:Grandmaster = 2:1:1:1, then remaining GeRaCl | 0.736712 / -0.004886 | - | - | 0.631782 / -0.004886 | 0.642194 / -0.002320 | full + target2 + five-task gate |
| Mix H HabrFull | Mix H variant with all 4,369 Habr-harder records in stage 1, then remaining GeRaCl | 0.736865 / -0.003759 | - | - | 0.632789 / -0.003759 | - | full + five-task gate |
| Mix H HabrFull no-GeRaCl | HabrFull + DeepVK + Grandmaster; GeRaCl removed | - | - | - | 0.593287 / -0.038151 | - | frozen-wrapper target 5 gate; CEDR/TERRa up, RuSciBench clustering down |
| CEDR goal a050 | checkpoint arithmetic: retained best + anti no-GeRaCl direction | - | 0.692005 / -0.038218 | - | 0.602034 / -0.005744 | - | frozen-wrapper target 5 gate + 18-task eval; improves CEDR/TERRa, broad STS and Georeview regressions on 18-task eval |
| Mix H HabrFull no-GeRaCl + Habr minimal | HabrFull + DeepVK + Grandmaster, then one full pass over minimally filtered Habr | - | - | - | 0.593912 / -0.037248 | - | frozen-wrapper target 5 gate; best CEDR in GeRaCl-removal block, clustering still weak |
| Mix H HabrFull batch2 control | same as Mix H HabrFull stage 1, batch 2 one pass | - | - | - | 0.595434 / -0.041073 | - | frozen-wrapper target 5 gate; CEDR/broad control |
| Mix H HabrFull all-GeRaCl stage1 | all 16,000 GeRaCl + HabrFull + DeepVK + Grandmaster in stage 1, no stage 2 | - | - | - | 0.595889 / -0.041233 | - | frozen-wrapper target 5 gate; best new candidate |
| Mix H HabrFull GeRaCl1600 | GeRaCl1600 + HabrFull + DeepVK + Grandmaster | - | - | - | 0.592923 / -0.042827 | - | frozen-wrapper target 5 gate; RuSciBench clustering regressed |
| Mix H HabrFull GeRaCl3200 | GeRaCl3200 + HabrFull + DeepVK + Grandmaster | - | - | - | 0.593632 / -0.041923 | - | frozen-wrapper target 5 gate; Sensitive/TERRa good, clustering regressed |
| Mix I | GeRaCl:Habr:DeepVK:Veles = 2:1:1:1, then remaining GeRaCl | - | - | - | - | 0.638801 / -0.006688 | target2 only |
| Mix J | GeRaCl:Habr:DeepVK:GroundedStrict:Grandmaster = 2:1:1:1:1, then remaining GeRaCl | - | - | - | - | 0.639657 / -0.004697 | target2 only |
| Mix K | GeRaCl:Habr:DeepVK:GroundedStrict:Sensitive = 2:1:1:1:1, then remaining GeRaCl | - | - | - | 0.631398 / -0.006767 | - | sensitive gate |
| Mix L | GeRaCl + all Habr + DeepVK + Grounded + MVR-CII/UC Berkeley | - | - | - | 0.628240 / -0.010296 | - | TERRa/sensitive variant |
| Mix M basic | full-use quality-filtered 2:1:1:1:1 | - | - | - | 0.630901 / -0.007177 | - | better Mix M variant |
| Mix M sim_guard | Mix M with lexical false-negative pruning | - | - | - | 0.628044 / -0.015431 | - | worse than basic |
| Mix N | Mix F, remaining GeRaCl, then low-LR correction | - | - | - | 0.630370 / -0.007020 | - | three-stage correction |
| Mix O pair300 | Mix F continuation pair calibration | - | - | - | 0.632098 / -0.010491 | - | probe |
| Mix O pair600 | Mix F continuation pair calibration | - | - | - | 0.631998 / -0.010928 | - | probe |
| Mix O pair900 | Mix F continuation pair calibration | - | - | - | 0.632172 / -0.010565 | - | probe |
| Mix O tax300 | Mix F continuation taxonomy repair | - | - | - | 0.631893 / -0.010961 | - | probe |
| Mix O tax600 | Mix F continuation taxonomy repair | - | - | - | 0.632309 / -0.010830 | - | best Mix O gate |
| Mix O alt400+400 | taxonomy repair, then pair calibration | - | - | - | 0.632209 / -0.011017 | - | probe |
