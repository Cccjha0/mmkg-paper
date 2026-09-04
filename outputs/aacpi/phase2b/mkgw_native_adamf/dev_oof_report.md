# AACPI V2 Phase 2B OOF Advantage Learnability

Dataset/pair: `mkg_w / mkgw_native_adamf`

All primary metrics use genuinely outer-fold OOF predictions on non-reference DEV actions. The alpha0 rows are reported only as a supplemental scope.

| Scope | MAE | Smooth-L1 | Spearman | Positive AUPRC / prevalence | Harmful AUPRC / prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Non-reference | 0.032526 | 0.029004 | 0.023478 | 0.296232 / 0.290458 | 0.308280 / 0.312738 |
| All actions | 0.027449 | 0.024202 | 0.022378 | 0.239022 / 0.242049 | 0.269241 / 0.260615 |

| Predicted-U bucket | Predicted mean U | Actual mean U | P(U>0) | P(U<0) | P(U=0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| lowest_10pct | -0.007081 | -0.017969 | 0.225366 | 0.290302 | 0.484331 |
| 10_to_30pct | -0.003349 | -0.008025 | 0.284729 | 0.314741 | 0.400530 |
| 30_to_50pct | -0.001411 | -0.004388 | 0.299618 | 0.320081 | 0.380301 |
| 50_to_70pct | +0.000038 | -0.004452 | 0.306907 | 0.325031 | 0.368062 |
| 70_to_90pct | +0.001596 | -0.005100 | 0.306010 | 0.323979 | 0.370011 |
| highest_10pct | +0.004617 | -0.003081 | 0.284690 | 0.269411 | 0.445900 |

No kappa, lambda, tau, uncertainty penalty, fallback threshold, or policy evaluation is used in Phase 2B.
