# AACPI V2 Phase 2B OOF Advantage Learnability

Dataset/pair: `mkg_w / mkgw_mhyper_adamf`

All primary metrics use genuinely outer-fold OOF predictions on non-reference DEV actions. The alpha0 rows are reported only as a supplemental scope.

| Scope | MAE | Smooth-L1 | Spearman | Positive AUPRC / prevalence | Harmful AUPRC / prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Non-reference | 0.022491 | 0.018647 | 0.142492 | 0.512219 / 0.421110 | 0.164070 / 0.161142 |
| All actions | 0.018559 | 0.014998 | 0.091440 | 0.382692 / 0.336888 | 0.143288 / 0.128913 |

| Predicted-U bucket | Predicted mean U | Actual mean U | P(U>0) | P(U<0) | P(U=0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| lowest_10pct | -0.011320 | -0.026386 | 0.233775 | 0.163126 | 0.603099 |
| 10_to_30pct | -0.005192 | -0.015324 | 0.322144 | 0.169062 | 0.508794 |
| 30_to_50pct | -0.002205 | -0.006097 | 0.401462 | 0.159172 | 0.439367 |
| 50_to_70pct | -0.000190 | -0.004754 | 0.478831 | 0.159123 | 0.362046 |
| 70_to_90pct | +0.002067 | -0.002292 | 0.508892 | 0.158636 | 0.332473 |
| highest_10pct | +0.006971 | +0.000888 | 0.554668 | 0.156305 | 0.289027 |

No kappa, lambda, tau, uncertainty penalty, fallback threshold, or policy evaluation is used in Phase 2B.
