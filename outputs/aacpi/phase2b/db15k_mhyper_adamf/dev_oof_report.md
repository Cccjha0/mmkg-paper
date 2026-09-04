# AACPI V2 Phase 2B OOF Advantage Learnability

Dataset/pair: `db15k / db15k_mhyper_adamf`

All primary metrics use genuinely outer-fold OOF predictions on non-reference DEV actions. The alpha0 rows are reported only as a supplemental scope.

| Scope | MAE | Smooth-L1 | Spearman | Positive AUPRC / prevalence | Harmful AUPRC / prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Non-reference | 0.031399 | 0.027755 | 0.083701 | 0.415269 / 0.346845 | 0.208171 / 0.211074 |
| All actions | 0.025435 | 0.022225 | 0.045897 | 0.300628 / 0.277476 | 0.187776 / 0.168859 |

| Predicted-U bucket | Predicted mean U | Actual mean U | P(U>0) | P(U<0) | P(U=0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| lowest_10pct | -0.012969 | -0.044143 | 0.196708 | 0.222427 | 0.580866 |
| 10_to_30pct | -0.004492 | -0.020036 | 0.262249 | 0.194898 | 0.542853 |
| 30_to_50pct | -0.001780 | -0.011776 | 0.349997 | 0.186977 | 0.463025 |
| 50_to_70pct | -0.000279 | -0.007855 | 0.393441 | 0.194656 | 0.411902 |
| 70_to_90pct | +0.001149 | -0.006512 | 0.408231 | 0.235634 | 0.356134 |
| highest_10pct | +0.003509 | -0.004243 | 0.443907 | 0.263977 | 0.292116 |

No kappa, lambda, tau, uncertainty penalty, fallback threshold, or policy evaluation is used in Phase 2B.
