# AACPI V2 Phase 2A Diagnostic Decision

**Date:** 2026-09-04
**Evidence:** DEV only
**Model training:** none
**Decision:** proceed to Phase 2B advantage learnability

## Winner supervision is not action supervision

For a beneficial local correction, the endpoint winner direction agrees with the best local action only 65.23%–89.12% of the time across the six dataset/pair settings. After excluding endpoint ties, 10.75%–35.61% of beneficial corrections point in the opposite direction from the endpoint winner label.

| Dataset / pair | Positive-opportunity queries | Winner-direction agreement on beneficial actions | Opposite direction after excluding endpoint ties | Endpoint preference but best action stays at anchor |
| --- | ---: | ---: | ---: | ---: |
| MKG-W / M-Hyper + NativE | 60.80% | 89.12% | 10.75% | 21.46% |
| MKG-W / M-Hyper + AdaMF-MAT | 47.99% | 70.91% | 29.46% | 40.87% |
| MKG-W / NativE + AdaMF-MAT | 58.15% | 82.67% | 17.60% | 28.90% |
| DB15K / M-Hyper + NativE | 39.60% | 65.23% | 35.61% | 50.32% |
| DB15K / M-Hyper + AdaMF-MAT | 40.81% | 66.86% | 33.94% | 51.33% |
| DB15K / NativE + AdaMF-MAT | 46.44% | 78.47% | 22.19% | 44.39% |

The two NativE + AdaMF-MAT falsification pairs retain substantial local opportunity and non-trivial target mismatch. This supports testing advantage learnability directly rather than increasing the complexity of the winner classifier.

The result supports the empirical motivation:

> Expert preference is not equivalent to mixture advantage.

It does not yet show that advantage is predictable from answer-agnostic geometry.

## Plateau and policy-aware evaluation

Across the six pairs, 38.83%–47.94% of non-reference actions have exactly zero advantage. Mean distinct RR values cover only 55.88%–62.27% of each action grid. The utility surface is therefore materially stepped and plateaued.

Phase 2B primary metrics exclude alpha0 reference rows and give AUPRC/calibration a central role. MAE, Smooth-L1, and Spearman remain supporting regression diagnostics. A small error around `U=0` is not enough to justify intervention.

## Frozen boundary concentration

The descriptive best action remains at the maximum negative radius, `delta_alpha=-0.30`, for:

- MKG-W / M-Hyper + NativE: 34.91%;
- MKG-W / M-Hyper + AdaMF-MAT: 37.29%;
- MKG-W / NativE + AdaMF-MAT: 23.32%;
- DB15K / M-Hyper + NativE: 28.69%;
- DB15K / M-Hyper + AdaMF-MAT: 29.72%;
- DB15K / NativE + AdaMF-MAT: 32.61%.

For MKG-W / M-Hyper + NativE, the positive `+0.30` boundary is also best for 7.89% of queries. The action grid remains frozen. These rates are recorded as a future radius-sensitivity result and limitation, not used to expand the Phase 2 search.

## Outputs

- `outputs/aacpi/phase2a_diagnostics/phase2a_summary.json`
- `outputs/aacpi/phase2a_diagnostics/winner_action_alignment.csv`
- `outputs/aacpi/phase2a_diagnostics/winner_action_confusion.csv`
- `outputs/aacpi/phase2a_diagnostics/action_utility_landscape.csv`
- `outputs/aacpi/phase2a_diagnostics/action_utility_sign_rates.svg`
- `outputs/aacpi/phase2a_diagnostics/action_mean_advantage.svg`
- `outputs/aacpi/phase2a_diagnostics/winner_best_direction_confusion.svg`
- per-pair compressed query diagnostics under `outputs/aacpi/phase2a_diagnostics/query_diagnostics/`

All outputs are descriptive DEV evidence. No AACPI predictor, greedy policy, conservative policy, or TEST evaluation was run.
