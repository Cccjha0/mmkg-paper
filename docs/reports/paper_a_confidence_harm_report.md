# Paper A Confidence-to-Harm Diagnostic Report

## Main answer

Current confidence provides limited, pair-sensitive harm discrimination rather than a reliably calibrated harmful-correction probability. Risk-score AUROC is above 0.5 in 4/4 pairs on grouped OOF DEV and 4/4 pairs on locked TEST, but AUPRC exceeds the pair's harm prevalence in only 2/4 DEV and 2/4 TEST cases. The highest-confidence quintile is consistently the safest, yet harm is generally concentrated in middle-confidence quintiles rather than increasing monotonically toward the lowest-confidence quintile. Confidence can therefore support selective retention of especially safe queries, but `1 - confidence` is not a strong standalone harm detector across all four pairs.

## Information boundary and counterfactual

This analysis never defines harm from the final fallback-filtered Anchored result. For every query it reconstructs the counterfactual raw bounded policy `clip(alpha0 + beta_locked * tanh(decision), 0, 1)`, applies only non-finite fallback, maps to the existing exact-ranking alpha grid, and reads the resulting reciprocal rank from the precomputed full-ranking grid. Harm is `rr_raw_bounded < rr_global`.

Confidence is `abs(2 * p_A - 1)` and harm risk is `1 - confidence`. Grouped OOF DEV predictions are evaluated separately from locked TEST. TEST results are diagnostic only and were not used to alter beta, the confidence threshold, the feature schema, or any operating point.

## Discrimination metrics

| Split | Dataset/Pair | Harm % | AUROC | AUPRC | AP lift | Spearman(confidence, delta RR) | Assessment |
|---|---|---:|---:|---:|---:|---:|---|
| DEV | MKG-W / M-Hyper + NativE | 19.78 | 0.5629 | 0.2023 | +0.0045 | -0.0308 | weak positive discrimination |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 8.77 | 0.5811 | 0.0926 | +0.0049 | -0.0247 | weak positive discrimination |
| DEV | DB15K / M-Hyper + NativE | 9.88 | 0.5194 | 0.0915 | -0.0073 | -0.0272 | weak AUROC; no precision lift |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 10.62 | 0.5340 | 0.1015 | -0.0047 | -0.0120 | weak AUROC; no precision lift |
| TEST | MKG-W / M-Hyper + NativE | 19.99 | 0.5583 | 0.2020 | +0.0021 | -0.0388 | weak positive discrimination |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 8.88 | 0.5686 | 0.0910 | +0.0022 | -0.0339 | weak positive discrimination |
| TEST | DB15K / M-Hyper + NativE | 9.74 | 0.5274 | 0.0916 | -0.0058 | -0.0276 | weak AUROC; no precision lift |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 10.88 | 0.5277 | 0.1026 | -0.0062 | -0.0254 | weak AUROC; no precision lift |

AUROC measures ranking discrimination of `1 - confidence` for harm. AUPRC must be read against harm prevalence; AP lift is AUPRC minus prevalence. Spearman tests whether higher confidence accompanies better raw correction outcomes, but it measures magnitude ordering rather than binary harm detection.

The coexistence of AUROC slightly above 0.5 and AUPRC below prevalence on DB15K is not contradictory: high-confidence queries contain very few harmful cases, while the risk ranking becomes non-monotonic through the middle and bottom of the distribution.

## Confidence quintiles

Quintiles are fixed-count rank buckets within each pair and split, sorted from highest to lowest confidence with `query_id` as deterministic tie-break. The bottom 20% is therefore the lowest-confidence bucket.

| Split | Dataset/Pair | Confidence bucket | n | Harm % | Mean delta RR | Mean harm | Mean confidence |
|---|---|---|---:|---:|---:|---:|---:|
| DEV | MKG-W / M-Hyper + NativE | top_20pct | 5132 | 4.23 | +0.015992 | 0.062503 | 0.7564 |
| DEV | MKG-W / M-Hyper + NativE | 20_to_40pct | 5131 | 26.04 | +0.008395 | 0.028958 | 0.3250 |
| DEV | MKG-W / M-Hyper + NativE | 40_to_60pct | 5131 | 26.35 | +0.004429 | 0.027322 | 0.2014 |
| DEV | MKG-W / M-Hyper + NativE | 60_to_80pct | 5131 | 24.89 | +0.002945 | 0.022988 | 0.1236 |
| DEV | MKG-W / M-Hyper + NativE | bottom_20pct | 5131 | 17.38 | -0.000949 | 0.023200 | 0.0430 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | top_20pct | 5132 | 0.16 | +0.002439 | 0.087872 | 0.7672 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 20_to_40pct | 5131 | 9.12 | +0.008418 | 0.022324 | 0.3710 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 40_to_60pct | 5131 | 14.68 | +0.002057 | 0.026540 | 0.2142 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 60_to_80pct | 5131 | 13.90 | -0.001151 | 0.027553 | 0.1236 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | bottom_20pct | 5131 | 6.02 | -0.000872 | 0.022609 | 0.0408 |
| DEV | DB15K / M-Hyper + NativE | top_20pct | 9507 | 1.36 | +0.000705 | 0.051066 | 0.4814 |
| DEV | DB15K / M-Hyper + NativE | 20_to_40pct | 9506 | 14.67 | +0.002770 | 0.038243 | 0.2653 |
| DEV | DB15K / M-Hyper + NativE | 40_to_60pct | 9507 | 16.06 | +0.002376 | 0.033610 | 0.1871 |
| DEV | DB15K / M-Hyper + NativE | 60_to_80pct | 9506 | 12.40 | +0.000582 | 0.038663 | 0.1147 |
| DEV | DB15K / M-Hyper + NativE | bottom_20pct | 9506 | 4.88 | +0.000263 | 0.022551 | 0.0388 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | top_20pct | 9507 | 1.69 | +0.000922 | 0.059227 | 0.5521 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 20_to_40pct | 9506 | 14.32 | +0.001331 | 0.045278 | 0.3206 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 40_to_60pct | 9507 | 17.38 | -0.000186 | 0.051213 | 0.2190 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 60_to_80pct | 9506 | 13.23 | -0.002893 | 0.067437 | 0.1336 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | bottom_20pct | 9506 | 6.46 | -0.001756 | 0.055832 | 0.0456 |
| TEST | MKG-W / M-Hyper + NativE | top_20pct | 5129 | 4.39 | +0.014014 | 0.078143 | 0.7565 |
| TEST | MKG-W / M-Hyper + NativE | 20_to_40pct | 5129 | 27.10 | +0.008919 | 0.033729 | 0.3196 |
| TEST | MKG-W / M-Hyper + NativE | 40_to_60pct | 5129 | 26.52 | +0.003521 | 0.032280 | 0.1990 |
| TEST | MKG-W / M-Hyper + NativE | 60_to_80pct | 5129 | 24.90 | +0.002613 | 0.029447 | 0.1221 |
| TEST | MKG-W / M-Hyper + NativE | bottom_20pct | 5128 | 17.04 | +0.000290 | 0.025713 | 0.0426 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | top_20pct | 5129 | 0.21 | +0.001684 | 0.043160 | 0.7692 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 20_to_40pct | 5129 | 9.81 | +0.006516 | 0.019420 | 0.3669 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 40_to_60pct | 5129 | 15.48 | +0.002130 | 0.025532 | 0.2133 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 60_to_80pct | 5129 | 13.26 | -0.000245 | 0.027073 | 0.1229 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | bottom_20pct | 5128 | 5.66 | -0.000381 | 0.034112 | 0.0407 |
| TEST | DB15K / M-Hyper + NativE | top_20pct | 11883 | 1.16 | +0.001187 | 0.032165 | 0.4858 |
| TEST | DB15K / M-Hyper + NativE | 20_to_40pct | 11882 | 13.84 | +0.002902 | 0.038439 | 0.2669 |
| TEST | DB15K / M-Hyper + NativE | 40_to_60pct | 11883 | 16.32 | +0.001819 | 0.035702 | 0.1864 |
| TEST | DB15K / M-Hyper + NativE | 60_to_80pct | 11882 | 12.39 | +0.000224 | 0.035421 | 0.1142 |
| TEST | DB15K / M-Hyper + NativE | bottom_20pct | 11882 | 4.97 | +0.000226 | 0.017071 | 0.0385 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | top_20pct | 11883 | 1.88 | +0.001947 | 0.037344 | 0.5561 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 20_to_40pct | 11882 | 15.03 | +0.002701 | 0.039055 | 0.3205 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 40_to_60pct | 11883 | 17.96 | -0.001838 | 0.053933 | 0.2203 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 60_to_80pct | 11882 | 13.28 | -0.002353 | 0.059166 | 0.1355 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | bottom_20pct | 11882 | 6.26 | -0.001952 | 0.061667 | 0.0465 |

## Source integrity

| Split | Dataset/Pair | Observations | Non-finite fallback | alpha0 | beta | Source rows SHA-256 | DEV lock SHA-256 |
|---|---|---:|---:|---:|---:|---|---|
| DEV | MKG-W / M-Hyper + NativE | 25656 | 0 | 0.60 | 0.50 | `e0a98d388819d535c15585165bb4fea18c47a7bb70e31ed0d21902c97dbf7d50` | `eee533f2d0b8b1a9a181d348c1df94b3db522b5f7c4a5a398a946b58d201c713` |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 25656 | 0 | 1.00 | 0.50 | `fbaa2a0faea02aa6191deff8110c6d10e78c1ac1092a4bd83b4926b06160a6b4` | `95109445388ea1ffeab17c11b5281880ca3928c827155d2f1e9486dbab93bb0f` |
| DEV | DB15K / M-Hyper + NativE | 47532 | 0 | 1.00 | 0.50 | `f49663bce80db958d4fbb3d576ed841dc5926387a9d67959533db484d8249dc3` | `a3ddf0b07697df57140c8a1dacdeb38ee96598272805b9db2a44bd59fa2b57ef` |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 47532 | 0 | 1.00 | 0.50 | `7b13db8cd4094ec4ce1651fa3a95a1d80f1c9e859d0d26381fe507ada372e066` | `b7ec3d6b524d16cf9c2128a3b84c8881cbe5e6b8ee7c2e5d592c58f756dac55d` |
| TEST | MKG-W / M-Hyper + NativE | 25644 | 0 | 0.60 | 0.50 | `469d2d427b61d3048de03ef485d804e2821c40a8af9a8e5e62d61e2ff4bbd624` | `eee533f2d0b8b1a9a181d348c1df94b3db522b5f7c4a5a398a946b58d201c713` |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 25644 | 0 | 1.00 | 0.50 | `0414041b7cd33267e79f0d7b6c12575f581f7e7ddaa22724c0a041d0cc087e60` | `95109445388ea1ffeab17c11b5281880ca3928c827155d2f1e9486dbab93bb0f` |
| TEST | DB15K / M-Hyper + NativE | 59412 | 0 | 1.00 | 0.50 | `2155f299165204d388ad45a509684148a3db3d3fa0dae9370896d2ed42271cfc` | `a3ddf0b07697df57140c8a1dacdeb38ee96598272805b9db2a44bd59fa2b57ef` |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 59412 | 0 | 1.00 | 0.50 | `77bb9f968099600ff110f110dea8cec197f5891396a055d35bd82542dce06e0c` | `b7ec3d6b524d16cf9c2128a3b84c8881cbe5e6b8ee7c2e5d592c58f756dac55d` |

The stored confidence and original fallback-filtered policy are reconstructed and verified before the counterfactual is evaluated. No model or combiner is trained by this script.

## Figure

- `outputs/paper_a_safe_correction/confidence_harm/confidence_vs_harm.pdf`

The four panels show harmful-correction rates from high- to low-confidence quintiles for grouped OOF DEV and locked TEST.
