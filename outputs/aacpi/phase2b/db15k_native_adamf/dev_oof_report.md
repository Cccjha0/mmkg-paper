# AACPI V2 Phase 2B OOF Advantage Learnability

Dataset/pair: `db15k / db15k_native_adamf`

All primary metrics use genuinely outer-fold OOF predictions on non-reference DEV actions. The alpha0 rows are reported only as a supplemental scope.

| Scope | MAE | Smooth-L1 | Spearman | Positive AUPRC / prevalence | Harmful AUPRC / prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Non-reference | 0.034424 | 0.030659 | 0.106047 | 0.468237 / 0.401077 | 0.220465 / 0.210595 |
| All actions | 0.027867 | 0.024552 | 0.080223 | 0.370541 / 0.320862 | 0.190634 / 0.168476 |

| Predicted-U bucket | Predicted mean U | Actual mean U | P(U>0) | P(U<0) | P(U=0) |
| --- | ---: | ---: | ---: | ---: | ---: |
| lowest_10pct | -0.007477 | -0.027556 | 0.258613 | 0.228265 | 0.513123 |
| 10_to_30pct | -0.003009 | -0.009449 | 0.345773 | 0.214122 | 0.440105 |
| 30_to_50pct | -0.001097 | -0.005916 | 0.390706 | 0.218219 | 0.391075 |
| 50_to_70pct | +0.000238 | -0.003966 | 0.433204 | 0.210225 | 0.356572 |
| 70_to_90pct | +0.001768 | -0.003860 | 0.448941 | 0.204234 | 0.346824 |
| highest_10pct | +0.004816 | +0.002837 | 0.514911 | 0.184085 | 0.301005 |

No kappa, lambda, tau, uncertainty penalty, fallback threshold, or policy evaluation is used in Phase 2B.
