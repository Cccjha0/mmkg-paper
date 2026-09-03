# AACPI V2 Phase 2A DEV Diagnostics

This report is descriptive and DEV-only. No predictor or policy was trained.

## Definitions

- `winner_label = 1[RR_A > RR_B]`; endpoint ties map to zero exactly as specified.
- Winner direction is toward A for label 1 and toward B-or-tie for label 0.
- Best action maximizes actual local-action RR, then prefers smaller absolute deviation and then smaller alpha.
- A query is worth deviating when at least one frozen local action has strictly positive advantage.
- Because the binary winner label has no stay class, the deviation comparison uses `RR_A != RR_B` as its implied endpoint-preference signal. The exact 2x3 and action-class confusion matrices are exported separately.

## Winner label versus mixture action

| Dataset / pair | Endpoint ties | Worth deviating | Direction agreement on beneficial actions | Untied direction agreement | Opposite direction on untied beneficial actions | Endpoint preference vs deviation agreement | Stay despite endpoint preference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| db15k / M-Hyper + AdaMF-MAT | 18.11% | 40.81% | 66.86% | 66.06% | 33.94% | 57.00% | 51.33% |
| db15k / M-Hyper + NativE | 22.15% | 39.60% | 65.23% | 64.39% | 35.61% | 59.89% | 50.32% |
| db15k / NativE + AdaMF-MAT | 18.97% | 46.44% | 78.47% | 77.81% | 22.19% | 62.66% | 44.39% |
| mkg_w / M-Hyper + AdaMF-MAT | 19.85% | 47.99% | 70.91% | 70.54% | 29.46% | 66.65% | 40.87% |
| mkg_w / M-Hyper + NativE | 23.73% | 60.80% | 89.12% | 89.25% | 10.75% | 82.74% | 21.46% |
| mkg_w / NativE + AdaMF-MAT | 20.01% | 58.15% | 82.67% | 82.40% | 17.60% | 75.61% | 28.90% |

## Action-wise utility landscape

| Dataset / pair | Delta alpha | Positive U | Zero U | Negative U | Mean U | Mean positive U | Mean negative U | Best action | Max-radius boundary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| db15k / M-Hyper + AdaMF-MAT | -0.30 | 37.30% | 35.13% | 27.56% | -0.025326 | +0.036024 | -0.140629 | 29.72% | yes |
| db15k / M-Hyper + AdaMF-MAT | -0.20 | 36.38% | 39.91% | 23.70% | -0.018282 | +0.025244 | -0.115882 | 5.46% |  |
| db15k / M-Hyper + AdaMF-MAT | -0.10 | 34.02% | 47.62% | 18.36% | -0.008496 | +0.015321 | -0.074665 | 2.89% |  |
| db15k / M-Hyper + AdaMF-MAT | -0.05 | 31.03% | 54.17% | 14.80% | -0.004195 | +0.008475 | -0.046101 | 2.73% |  |
| db15k / M-Hyper + AdaMF-MAT | +0.00 | 0.00% | 100.00% | 0.00% | +0.000000 |  |  | 59.19% |  |
| db15k / M-Hyper + NativE | -0.30 | 36.18% | 39.61% | 24.21% | -0.008736 | +0.037608 | -0.092290 | 28.69% | yes |
| db15k / M-Hyper + NativE | -0.20 | 34.91% | 44.17% | 20.92% | -0.005113 | +0.025952 | -0.067738 | 5.44% |  |
| db15k / M-Hyper + NativE | -0.10 | 32.20% | 50.90% | 16.90% | -0.001978 | +0.014553 | -0.039434 | 2.79% |  |
| db15k / M-Hyper + NativE | -0.05 | 28.83% | 57.07% | 14.10% | -0.001037 | +0.008131 | -0.023977 | 2.69% |  |
| db15k / M-Hyper + NativE | +0.00 | 0.00% | 100.00% | 0.00% | +0.000000 |  |  | 60.40% |  |
| db15k / NativE + AdaMF-MAT | -0.30 | 42.39% | 31.19% | 26.42% | -0.010296 | +0.052985 | -0.123981 | 32.61% | yes |
| db15k / NativE + AdaMF-MAT | -0.20 | 41.58% | 35.05% | 23.37% | -0.009229 | +0.038433 | -0.107879 | 6.70% |  |
| db15k / NativE + AdaMF-MAT | -0.10 | 39.51% | 41.46% | 19.03% | -0.005781 | +0.021878 | -0.075801 | 3.61% |  |
| db15k / NativE + AdaMF-MAT | -0.05 | 36.95% | 47.63% | 15.42% | -0.003136 | +0.012345 | -0.049921 | 3.52% |  |
| db15k / NativE + AdaMF-MAT | +0.00 | 0.00% | 100.00% | 0.00% | +0.000000 |  |  | 53.56% |  |
| mkg_w / M-Hyper + AdaMF-MAT | -0.30 | 44.47% | 34.39% | 21.15% | -0.015889 | +0.021960 | -0.121325 | 37.29% | yes |
| mkg_w / M-Hyper + AdaMF-MAT | -0.20 | 43.74% | 38.19% | 18.07% | -0.010113 | +0.015949 | -0.094595 | 5.16% |  |
| mkg_w / M-Hyper + AdaMF-MAT | -0.10 | 41.46% | 44.52% | 14.03% | -0.004622 | +0.009303 | -0.060443 | 2.76% |  |
| mkg_w / M-Hyper + AdaMF-MAT | -0.05 | 38.78% | 50.00% | 11.22% | -0.002349 | +0.005373 | -0.039509 | 2.78% |  |
| mkg_w / M-Hyper + AdaMF-MAT | +0.00 | 0.00% | 100.00% | 0.00% | +0.000000 |  |  | 52.01% |  |
| mkg_w / M-Hyper + NativE | -0.30 | 43.54% | 35.59% | 20.86% | -0.001698 | +0.038529 | -0.088545 | 34.91% | yes |
| mkg_w / M-Hyper + NativE | -0.20 | 43.21% | 38.39% | 18.40% | -0.001108 | +0.029625 | -0.075585 | 5.46% |  |
| mkg_w / M-Hyper + NativE | -0.10 | 40.93% | 43.40% | 15.67% | -0.000832 | +0.021052 | -0.060299 | 2.65% |  |
| mkg_w / M-Hyper + NativE | -0.05 | 38.17% | 48.02% | 13.81% | -0.000168 | +0.017506 | -0.049598 | 3.25% |  |
| mkg_w / M-Hyper + NativE | +0.00 | 0.00% | 100.00% | 0.00% | +0.000000 |  |  | 39.20% |  |
| mkg_w / M-Hyper + NativE | +0.05 | 12.57% | 48.49% | 38.93% | -0.000830 | +0.047218 | -0.017381 | 2.15% |  |
| mkg_w / M-Hyper + NativE | +0.10 | 13.77% | 44.06% | 42.17% | -0.000746 | +0.055064 | -0.019753 | 1.77% |  |
| mkg_w / M-Hyper + NativE | +0.20 | 14.39% | 40.20% | 45.40% | -0.001133 | +0.064109 | -0.022820 | 2.72% |  |
| mkg_w / M-Hyper + NativE | +0.30 | 14.43% | 38.00% | 47.56% | -0.002099 | +0.077886 | -0.028048 | 7.89% | yes |
| mkg_w / NativE + AdaMF-MAT | -0.30 | 31.78% | 31.40% | 36.82% | -0.011367 | +0.061314 | -0.083801 | 23.32% | yes |
| mkg_w / NativE + AdaMF-MAT | -0.20 | 31.25% | 34.75% | 34.00% | -0.008435 | +0.049665 | -0.070454 | 5.32% |  |
| mkg_w / NativE + AdaMF-MAT | -0.10 | 29.83% | 40.57% | 29.61% | -0.004427 | +0.036266 | -0.051488 | 3.25% |  |
| mkg_w / NativE + AdaMF-MAT | -0.05 | 28.22% | 45.39% | 26.39% | -0.001849 | +0.031810 | -0.041025 | 4.51% |  |
| mkg_w / NativE + AdaMF-MAT | +0.00 | 0.00% | 100.00% | 0.00% | +0.000000 |  |  | 41.85% |  |
| mkg_w / NativE + AdaMF-MAT | +0.05 | 24.15% | 46.29% | 29.56% | -0.006412 | +0.024595 | -0.041795 | 21.74% |  |

## Interpretation boundary

High zero rates confirm a stepped RR surface, so later regression MAE cannot be the main scientific criterion. Boundary-best rates are descriptive evidence that some query utilities continue improving at the frozen radius; the Phase 2 action grid remains unchanged.

The NativE + AdaMF-MAT rows are retained as the primary falsification diagnostic. These results assess target alignment only and do not establish advantage learnability.
