# MKG-Y Y-E3 Adaptive Resolution Audit

Date: 2026-09-08

## Outcome

**Y_E3_RESOLUTION_REPLICATION_REPORTED** (descriptive external replication; no gate or route selection).

## Main findings

Robust-positive counts across the three MKG-Y pairs are L1=0/3, L2=1/3, L3=1/3, L4=1/3.
The best group-level resolution matches or exceeds frozen X4 in 1/3 pairs; X4 significantly exceeds the best group in 1/3 pairs.
Finer-granularity train-up / OOF-flat-or-down behavior appears in 1/3 pairs.

The result is pair-dependent rather than a monotone granularity effect. Relation-level adaptation is robust only for M-Hyper + AdaMF-MAT; its OOF gain is +0.004920 (95% CI [+0.002682, +0.007142]) and recovers 11.2% of available headroom. M-Hyper + NativE has no robust L1-L4 resolution, while frozen X4 is positive and significant. NativE + AdaMF-MAT has no robust resolution at any level, including frozen X4.

Direction-only adaptation is negative for all three pairs. Moving from Relation to Relation×Direction or Context raises training gain without improving OOF gain for M-Hyper + AdaMF-MAT, and Context coverage falls to 48.3–79.3% across pairs. These results do not support a general claim that finer resolution improves identifiability.

## Pair × resolution metrics

| Pair | Level | Train gain | OOF gain | 95% clustered CI | Recovery | Negative transfer | Changed | Coverage | Group MAD | Robust |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MKG-Y / M-Hyper + NativE | L0 Global | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| MKG-Y / M-Hyper + NativE | L1 Direction | +0.000382 | -0.001199 | [-0.002794, +0.000367] | -3.3% | 22.7% | 50.1% | 100.0% | 0.075 | False |
| MKG-Y / M-Hyper + NativE | L2 Relation | +0.003827 | +0.000822 | [-0.001357, +0.003129] | 2.2% | 27.9% | 87.1% | 96.7% | 0.026 | False |
| MKG-Y / M-Hyper + NativE | L3 Relation×Direction | +0.005157 | +0.000399 | [-0.001788, +0.002542] | 1.1% | 34.9% | 93.6% | 96.7% | 0.064 | False |
| MKG-Y / M-Hyper + NativE | L4 Context | +0.006615 | +0.000591 | [-0.001168, +0.002348] | 1.6% | 27.1% | 64.7% | 66.5% | 0.038 | False |
| MKG-Y / M-Hyper + NativE | L5 Query | +0.014596 | +0.003705 | [+0.002268, +0.005210] | 10.1% | 17.7% | 56.1% | 100.0% | NA | True |
| MKG-Y / M-Hyper + AdaMF-MAT | L0 Global | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| MKG-Y / M-Hyper + AdaMF-MAT | L1 Direction | +0.000628 | -0.000722 | [-0.002317, +0.000859] | -1.6% | 28.7% | 70.1% | 100.0% | 0.050 | False |
| MKG-Y / M-Hyper + AdaMF-MAT | L2 Relation | +0.006332 | +0.004920 | [+0.002682, +0.007142] | 11.2% | 24.6% | 88.9% | 96.7% | 0.000 | True |
| MKG-Y / M-Hyper + AdaMF-MAT | L3 Relation×Direction | +0.007454 | +0.003837 | [+0.001546, +0.006170] | 8.7% | 26.9% | 89.4% | 96.7% | 0.006 | True |
| MKG-Y / M-Hyper + AdaMF-MAT | L4 Context | +0.009898 | +0.003356 | [+0.001285, +0.005424] | 7.6% | 23.9% | 74.6% | 79.3% | 0.023 | True |
| MKG-Y / M-Hyper + AdaMF-MAT | L5 Query | +0.015844 | +0.006773 | [+0.004903, +0.008726] | 15.4% | 14.4% | 53.2% | 100.0% | NA | True |
| MKG-Y / NativE + AdaMF-MAT | L0 Global | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| MKG-Y / NativE + AdaMF-MAT | L1 Direction | +0.000245 | -0.000726 | [-0.001869, +0.000441] | -1.9% | 19.6% | 60.0% | 100.0% | 0.050 | False |
| MKG-Y / NativE + AdaMF-MAT | L2 Relation | +0.002854 | +0.000788 | [-0.000737, +0.002319] | 2.1% | 24.3% | 75.9% | 96.7% | 0.008 | False |
| MKG-Y / NativE + AdaMF-MAT | L3 Relation×Direction | +0.003884 | -0.000383 | [-0.001972, +0.001204] | -1.0% | 27.0% | 85.8% | 96.7% | 0.034 | False |
| MKG-Y / NativE + AdaMF-MAT | L4 Context | +0.002696 | -0.000529 | [-0.001652, +0.000549] | -1.4% | 15.5% | 43.9% | 48.3% | 0.034 | False |
| MKG-Y / NativE + AdaMF-MAT | L5 Query | +0.004464 | -0.000441 | [-0.001501, +0.000607] | -1.2% | 18.7% | 57.3% | 100.0% | NA | False |

## Best group versus frozen X4

| Pair | Best group | Group gain | X4 gain | X4−group 95% CI | Overfit transitions |
| --- | --- | ---: | ---: | ---: | ---: |
| MKG-Y / M-Hyper + NativE | L2 | +0.000822 | +0.003705 | [+0.000864, +0.004919] | 1/3 |
| MKG-Y / M-Hyper + AdaMF-MAT | L2 | +0.004920 | +0.006773 | [-0.000376, +0.004060] | 2/3 |
| MKG-Y / NativE + AdaMF-MAT | L2 | +0.000788 | -0.000441 | [-0.002807, +0.000320] | 1/3 |

## Figures

1. [figure1_resolution_identifiability_curve](../../outputs/complementarity_identifiability/mkg_y_y_e3_resolution/figure1_resolution_identifiability_curve.svg)
2. [figure2_resolution_generalization_gap](../../outputs/complementarity_identifiability/mkg_y_y_e3_resolution/figure2_resolution_generalization_gap.svg)
3. [figure3_resolution_support_instability_negative_transfer](../../outputs/complementarity_identifiability/mkg_y_y_e3_resolution/figure3_resolution_support_instability_negative_transfer.svg)
4. [figure4_pair_specific_resolution_curves](../../outputs/complementarity_identifiability/mkg_y_y_e3_resolution/figure4_pair_specific_resolution_curves.svg)

## Operational audit

- TEST access = 0
- outer-triple leakage = 0
- L5 new training = 0
- new selector/policy development = 0
- hierarchical shrinkage/LCB/conservative policy = 0
- checkpoint modification = 0

MKG-Y TEST remains locked. Y-E4 was not run by this audit.
