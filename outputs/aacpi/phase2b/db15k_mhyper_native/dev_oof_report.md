# AACPI V2 Phase 2B OOF Advantage Learnability

Dataset/pair: `db15k / db15k_mhyper_native`

All primary metrics use genuinely outer-fold OOF predictions on non-reference DEV actions. The alpha0 rows are reported only as a supplemental scope.

| Scope | MAE | Smooth-L1 | Spearman | Positive AUPRC / prevalence | Harmful AUPRC / prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Non-reference | 0.020616 | 0.017723 | 0.039591 | 0.350185 / 0.330288 | 0.191387 / 0.190335 |
| All actions | 0.016798 | 0.014200 | 0.015016 | 0.259293 / 0.264230 | 0.168886 / 0.152268 |

| Predicted-U bucket | Predicted mean U | Actual mean U | P(U>0) | P(U<0) | P(U=0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| lowest_10pct | -0.006561 | -0.015897 | 0.236628 | 0.189397 | 0.573976 |
| 10_to_30pct | -0.002762 | -0.006406 | 0.299356 | 0.197028 | 0.503616 |
| 30_to_50pct | -0.001046 | -0.003269 | 0.345763 | 0.172777 | 0.481460 |
| 50_to_70pct | +0.000085 | -0.000884 | 0.354678 | 0.192474 | 0.452848 |
| 70_to_90pct | +0.001313 | -0.002144 | 0.360973 | 0.213123 | 0.425904 |
| highest_10pct | +0.003536 | -0.000859 | 0.344712 | 0.163152 | 0.492137 |

No kappa, lambda, tau, uncertainty penalty, fallback threshold, or policy evaluation is used in Phase 2B.
