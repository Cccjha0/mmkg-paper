# AACPI V2 Phase 2B OOF Advantage Learnability

Dataset/pair: `mkg_w / mkgw_mhyper_native`

All primary metrics use genuinely outer-fold OOF predictions on non-reference DEV actions. The alpha0 rows are reported only as a supplemental scope.

| Scope | MAE | Smooth-L1 | Spearman | Positive AUPRC / prevalence | Harmful AUPRC / prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Non-reference | 0.021778 | 0.019205 | 0.213460 | 0.366400 / 0.276276 | 0.388526 / 0.303516 |
| All actions | 0.019474 | 0.017077 | 0.206481 | 0.330478 / 0.245578 | 0.363040 / 0.269792 |

| Predicted-U bucket | Predicted mean U | Actual mean U | P(U>0) | P(U<0) | P(U=0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| lowest_10pct | -0.004707 | -0.012125 | 0.174470 | 0.442972 | 0.382558 |
| 10_to_30pct | -0.001742 | -0.002978 | 0.193744 | 0.373091 | 0.433165 |
| 30_to_50pct | -0.000472 | -0.000595 | 0.219367 | 0.334982 | 0.445652 |
| 50_to_70pct | +0.000443 | +0.000578 | 0.311279 | 0.267259 | 0.421462 |
| 70_to_90pct | +0.001604 | +0.001083 | 0.361982 | 0.222685 | 0.415333 |
| highest_10pct | +0.003804 | +0.005183 | 0.415542 | 0.196151 | 0.388307 |

No kappa, lambda, tau, uncertainty penalty, fallback threshold, or policy evaluation is used in Phase 2B.
