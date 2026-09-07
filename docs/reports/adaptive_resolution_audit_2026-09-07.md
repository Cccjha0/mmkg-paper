# Adaptive Resolution Audit — Experiment 3

Date: 2026-09-07

## Outcome

Group-Level GO: **NO-GO**.
Experiment 2 Query-Level GO: **NO-GO**.
Frozen route decision: **ROUTE_C_LIMITS**.

## Main findings

No single L1–L4 resolution reaches the required four robust-positive pairs. The counts are L1=0/6, L2=1/6, L3=2/6, L4=2/6.
L4 has mean held-out support coverage 39.1% and mean support-weighted group-alpha foldwise MAD 0.022.
The best group-level point estimate is below the frozen X4 query-level probe in 6/6 pairs; X4 is significantly better by paired clustered bootstrap in 5/6.
The frozen X6 probe significantly exceeds the best group in 0/6 pairs.
Thus coarser adaptation does not recover the missing deployable headroom, while the richer candidate-level probe also fails. This supports a limits result rather than another selector or granularity extension.

## Pair × resolution metrics

| Pair | Level | Train gain | OOF gain | 95% clustered CI | Recovery | Neg. transfer | Changed | Coverage | Group MAD | Robust |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MKG-W / M-Hyper+NativE | L0 Global | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| MKG-W / M-Hyper+NativE | L1 Direction | +0.001014 | +0.000610 | [-0.000698, +0.001964] | 1.3% | 19.9% | 69.9% | 100.0% | 0.025 | False |
| MKG-W / M-Hyper+NativE | L2 Relation | +0.007404 | +0.004372 | [+0.002757, +0.006024] | 9.4% | 22.9% | 86.6% | 89.4% | 0.023 | True |
| MKG-W / M-Hyper+NativE | L3 Relation×Direction | +0.009554 | +0.004729 | [+0.003164, +0.006324] | 10.1% | 21.3% | 86.7% | 89.4% | 0.033 | True |
| MKG-W / M-Hyper+NativE | L4 Context | +0.012143 | +0.004583 | [+0.003116, +0.006073] | 9.8% | 20.1% | 77.7% | 80.0% | 0.023 | True |
| MKG-W / M-Hyper+NativE | L5 Query | +0.019684 | +0.007339 | [+0.005480, +0.009296] | 15.7% | 22.4% | 89.9% | 100.0% | NA | True |
| MKG-W / M-Hyper+AdaMF | L0 Global | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| MKG-W / M-Hyper+AdaMF | L1 Direction | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| MKG-W / M-Hyper+AdaMF | L2 Relation | +0.000762 | -0.000984 | [-0.001764, -0.000229] | -2.4% | 3.6% | 20.3% | 89.4% | 0.003 | False |
| MKG-W / M-Hyper+AdaMF | L3 Relation×Direction | +0.002606 | -0.001194 | [-0.002220, -0.000172] | -2.9% | 9.0% | 38.0% | 89.4% | 0.010 | False |
| MKG-W / M-Hyper+AdaMF | L4 Context | +0.000993 | +0.000647 | [+0.000204, +0.001096] | 1.6% | 3.9% | 17.0% | 25.6% | 0.028 | True |
| MKG-W / M-Hyper+AdaMF | L5 Query | +0.012424 | +0.005909 | [+0.004639, +0.007187] | 14.5% | 7.2% | 34.2% | 100.0% | NA | True |
| MKG-W / NativE+AdaMF | L0 Global | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| MKG-W / NativE+AdaMF | L1 Direction | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| MKG-W / NativE+AdaMF | L2 Relation | +0.002201 | +0.000273 | [-0.000825, +0.001346] | 0.6% | 10.5% | 37.0% | 89.4% | 0.011 | False |
| MKG-W / NativE+AdaMF | L3 Relation×Direction | +0.003274 | -0.000625 | [-0.001884, +0.000598] | -1.3% | 13.3% | 45.3% | 89.4% | 0.005 | False |
| MKG-W / NativE+AdaMF | L4 Context | +0.000685 | -0.000005 | [-0.000433, +0.000418] | -0.0% | 4.6% | 12.6% | 25.6% | 0.010 | False |
| MKG-W / NativE+AdaMF | L5 Query | +0.008234 | +0.001039 | [+0.000131, +0.002007] | 2.2% | 8.7% | 31.7% | 100.0% | NA | True |
| DB15K / M-Hyper+NativE | L0 Global | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| DB15K / M-Hyper+NativE | L1 Direction | +0.000007 | -0.000067 | [-0.000218, +0.000082] | -0.1% | 3.9% | 20.0% | 100.0% | 0.000 | False |
| DB15K / M-Hyper+NativE | L2 Relation | +0.002133 | +0.000231 | [-0.000582, +0.001049] | 0.5% | 9.2% | 37.9% | 85.1% | 0.019 | False |
| DB15K / M-Hyper+NativE | L3 Relation×Direction | +0.003634 | +0.001064 | [+0.000150, +0.001961] | 2.1% | 14.5% | 47.3% | 85.1% | 0.015 | True |
| DB15K / M-Hyper+NativE | L4 Context | +0.001793 | -0.000030 | [-0.000567, +0.000508] | -0.1% | 9.3% | 25.1% | 40.2% | 0.031 | False |
| DB15K / M-Hyper+NativE | L5 Query | +0.009172 | +0.002125 | [+0.001303, +0.002969] | 4.3% | 13.6% | 42.2% | 100.0% | NA | True |
| DB15K / M-Hyper+AdaMF | L0 Global | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| DB15K / M-Hyper+AdaMF | L1 Direction | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| DB15K / M-Hyper+AdaMF | L2 Relation | +0.000729 | -0.000020 | [-0.000513, +0.000473] | -0.0% | 2.7% | 10.7% | 85.1% | 0.001 | False |
| DB15K / M-Hyper+AdaMF | L3 Relation×Direction | +0.001551 | -0.000099 | [-0.000742, +0.000538] | -0.2% | 10.1% | 25.3% | 85.1% | 0.003 | False |
| DB15K / M-Hyper+AdaMF | L4 Context | +0.000605 | -0.000071 | [-0.000414, +0.000280] | -0.2% | 6.7% | 14.0% | 30.0% | 0.020 | False |
| DB15K / M-Hyper+AdaMF | L5 Query | +0.005040 | +0.000924 | [+0.000368, +0.001517] | 2.0% | 9.4% | 32.9% | 100.0% | NA | True |
| DB15K / NativE+AdaMF | L0 Global | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| DB15K / NativE+AdaMF | L1 Direction | +0.000000 | +0.000000 | [+0.000000, +0.000000] | 0.0% | 0.0% | 0.0% | 100.0% | 0.000 | False |
| DB15K / NativE+AdaMF | L2 Relation | +0.001969 | +0.000063 | [-0.000766, +0.000896] | 0.1% | 7.7% | 29.0% | 85.1% | 0.008 | False |
| DB15K / NativE+AdaMF | L3 Relation×Direction | +0.002766 | -0.000638 | [-0.001574, +0.000312] | -1.1% | 13.0% | 45.6% | 85.1% | 0.008 | False |
| DB15K / NativE+AdaMF | L4 Context | +0.001497 | -0.000240 | [-0.000825, +0.000328] | -0.4% | 7.6% | 23.1% | 33.3% | 0.020 | False |
| DB15K / NativE+AdaMF | L5 Query | +0.005386 | +0.001564 | [+0.001037, +0.002095] | 2.7% | 7.2% | 31.5% | 100.0% | NA | True |

## Group-Level GO

| Level | Robust-positive pairs | MKG-W represented | DB15K represented | ΔMRR < -0.001 pairs | Pass |
| --- | ---: | --- | --- | ---: | --- |
| L1 | 0/6 | False | False | 0 | False |
| L2 | 1/6 | True | False | 0 | False |
| L3 | 2/6 | True | True | 1 | False |
| L4 | 2/6 | True | False | 0 | False |

## Best group versus frozen query probes

| Pair | Best group | Group OOF gain | X4 query gain | X4−group 95% CI | X6−group 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| MKG-W / M-Hyper+NativE | L3 | +0.004729 | +0.007339 | [+0.000948, +0.004303] | [-0.002929, +0.000895] |
| MKG-W / M-Hyper+AdaMF | L4 | +0.000647 | +0.005909 | [+0.004035, +0.006506] | [-0.004151, -0.000200] |
| MKG-W / NativE+AdaMF | L2 | +0.000273 | +0.001039 | [-0.000309, +0.001843] | [-0.004388, -0.001492] |
| DB15K / M-Hyper+NativE | L3 | +0.001064 | +0.002125 | [+0.000122, +0.002002] | [-0.004824, -0.002466] |
| DB15K / M-Hyper+AdaMF | L1 | +0.000000 | +0.000924 | [+0.000362, +0.001492] | [-0.007273, -0.005102] |
| DB15K / NativE+AdaMF | L2 | +0.000063 | +0.001564 | [+0.000630, +0.002396] | [-0.004187, -0.001633] |

## Route evidence

- best group-level >= X4 query-level: 0/6 pairs
- X4 query-level significantly exceeds best group: 5/6 pairs
- finer-granularity train-up / OOF-flat-or-down pattern: 2/6 pairs
- X6 significantly exceeds best group: 0/6 pairs; required NativE+AdaMF inclusion = False

## Figures

1. [figure1_resolution_identifiability_curve](../../outputs/complementarity_identifiability/exp3_resolution/figure1_resolution_identifiability_curve.svg)
2. [figure2_resolution_generalization_gap](../../outputs/complementarity_identifiability/exp3_resolution/figure2_resolution_generalization_gap.svg)
3. [figure3_resolution_support_instability_negative_transfer](../../outputs/complementarity_identifiability/exp3_resolution/figure3_resolution_support_instability_negative_transfer.svg)
4. [figure4_highlighted_pair_curves](../../outputs/complementarity_identifiability/exp3_resolution/figure4_highlighted_pair_curves.svg)

## Operational audit

- TEST access = 0
- outer-triple leakage = 0
- new query selector / policy development = 0
- hierarchical shrinkage / LCB / conservative policy = 0
- checkpoint modification = 0
- all direct source and output hashes are in `audit_manifest.json`

ROUTE_C_LIMITS
