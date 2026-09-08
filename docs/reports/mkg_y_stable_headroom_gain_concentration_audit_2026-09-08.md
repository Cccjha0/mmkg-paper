# MKG-Y Y-E6 Stable Headroom and Gain Concentration Audit

Date: 2026-09-08
Split: DEV only

## Outcome

**Y_E6_CLOSURE_REPLICATION_REPORTED** (descriptive external replication; no progression gate or route selection).

## Stable headroom decomposition

| Pair | Raw Oracle | Consensus | 2-of-3 | 3-of-3 | LOSO | Frozen X4 OOF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-Y / M-Hyper + NativE | +0.036846 [+0.034337, +0.039465] | +0.033523 [+0.031146, +0.035994] | +0.019613 [+0.017615, +0.021685] | +0.009876 [+0.008355, +0.011443] | +0.009381 [+0.007907, +0.010906] | +0.003705 [+0.002253, +0.005167] |
| MKG-Y / M-Hyper + AdaMF-MAT | +0.044070 [+0.041258, +0.046975] | +0.040416 [+0.037685, +0.043169] | +0.024056 [+0.021790, +0.026443] | +0.013154 [+0.011397, +0.015006] | +0.011849 [+0.010073, +0.013676] | +0.006773 [+0.004831, +0.008698] |
| MKG-Y / NativE + AdaMF-MAT | +0.037844 [+0.035443, +0.040288] | +0.033461 [+0.031225, +0.035718] | +0.014547 [+0.012985, +0.016183] | +0.004988 [+0.004069, +0.005978] | +0.003892 [+0.002626, +0.005188] | -0.000441 [-0.001511, +0.000610] |

Consensus and two-/three-of-three quantities are all-seed ex-post upper diagnostics. LOSO is a held-out-seed diagnostic. Frozen X4 is the strict-OOF inference-time observable probe. They are not equivalent deployable policies.

## Recoveries and stable opportunities

| Pair | Consensus recovery | 2-of-3 recovery | 3-of-3 recovery | LOSO recovery | P(A2) | P(A3) | P(consensus != alpha0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-Y / M-Hyper + NativE | 91.0% | 53.2% | 26.8% | 25.5% | 61.4% | 34.4% | 71.9% |
| MKG-Y / M-Hyper + AdaMF-MAT | 91.7% | 54.6% | 29.8% | 26.9% | 62.5% | 36.2% | 73.9% |
| MKG-Y / NativE + AdaMF-MAT | 88.4% | 38.4% | 13.2% | 10.3% | 61.5% | 27.8% | 76.6% |

## Oracle gain concentration

| Pair | Top1 | Top5 | Top10 | Top20 | Q50 | Gini | Effective support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-Y / M-Hyper + NativE | 9.6% | 34.9% | 56.2% | 80.5% | 8.4% | 0.768 | 23.3% |
| MKG-Y / M-Hyper + AdaMF-MAT | 9.1% | 32.4% | 51.8% | 76.3% | 9.6% | 0.745 | 25.7% |
| MKG-Y / NativE + AdaMF-MAT | 8.3% | 31.5% | 52.9% | 77.2% | 9.3% | 0.749 | 25.8% |

The concentration curve orders non-negative original-triple Oracle gains from largest to smallest. Its diagonal denotes uniform contribution, not a random baseline.

## Descriptive closure evidence

- 2-of-3 headroom CI lower > 0: 3/3 pairs.
- Median 2-of-3 recovery: 53.2%.
- 3-of-3 headroom CI lower > 0: 3/3 pairs.
- Median 3-of-3 recovery: 26.8%.
- Top10 contributes at least 50%: 3/3 pairs.
- Q50 is at most 10%: 3/3 pairs.
- Frozen Y-E5 LOCAL_SIGNAL_PAIR count: 1/3.

These counts are external-replication diagnostics. The six-pair Experiment 6 classification thresholds are not reapplied or rescaled to three MKG-Y pairs.

## Evidence funnel

Raw Available → Seed-Stable / Transferable → Locally Observable → Empirically Deployable remains the interpretation order. Raw Oracle is not treated as headroom that merely awaits a better selector.

## Figures

1. figure1_stable_headroom_decomposition.svg
2. figure2_stable_recovery_forest.svg
3. figure3_oracle_gain_concentration_curves.svg
4. figure4_gain_concentration_summary.svg

## Integrity audit

- TEST access = 0
- checkpoint inference/retraining/reselection = 0
- new selector / feature / representation = 0
- policy tuning = 0
- action-grid modification = 0
- Y-E1–Y-E5 result modification = 0
- original-triple clustered bootstrap = 10000 replicates
- all direct source/output hashes recorded = yes
- MKG-Y TEST remains locked
- subsequent stage started = 0

Y_E6_CLOSURE_REPLICATION_REPORTED
