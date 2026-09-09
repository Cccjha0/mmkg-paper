# Paper A Anchored Dynamic Correction-Magnitude Audit

## Main answer

- MKG-W / M-Hyper + NativE: 94.26% of observations change alpha; 27.34% of changed observations stay within |Delta alpha| <= 0.10. Mean and p95 |Delta alpha| are 0.2078 and 0.4000. Corrections above 0.20 cover 41.19% of all observations and contribute +0.004674 MRR (79.60% of the net Delta MRR).
- MKG-W / M-Hyper + AdaMF-MAT: 35.63% of observations change alpha; 13.23% of changed observations stay within |Delta alpha| <= 0.10. Mean and p95 |Delta alpha| are 0.0760 and 0.3000. Corrections above 0.20 cover 14.17% of all observations and contribute +0.001846 MRR (93.01% of the net Delta MRR).
- DB15K / M-Hyper + NativE: 44.42% of observations change alpha; 36.45% of changed observations stay within |Delta alpha| <= 0.10. Mean and p95 |Delta alpha| are 0.0693 and 0.2500. Corrections above 0.20 cover 8.53% of all observations and contribute +0.000746 MRR (58.69% of the net Delta MRR).
- DB15K / M-Hyper + AdaMF-MAT: 6.98% of observations change alpha; 0.00% of changed observations stay within |Delta alpha| <= 0.10. Mean and p95 |Delta alpha| are 0.0223 and 0.3000. Corrections above 0.20 cover 6.98% of all observations and contribute +0.000714 MRR (100.00% of the net Delta MRR).

The requested strong empirical claim is not supported across all four pairs. Anchor retention is pair-dependent: exact-anchor rates range from 5.74% to 93.02% on locked TEST. Moreover, the >0.20 bucket supplies a majority of the net TEST Delta MRR in every pair. Conservative behavior is therefore supported in the structural sense of a static anchor, confidence fallback, and a DEV-locked bounded trust region, but not as a universal observation that gains come mainly from rare, <=0.10 shifts. This distinction should be preserved in the paper.

## Definitions

`alpha_delta = alpha_final - alpha0`. Exact anchor uses `|alpha_delta| <= 1e-12`. Positive and negative shifts use strict signed comparisons outside that tolerance. Boundary alpha means final alpha is 0 or 1. Saturation is the stored policy event in which the pre-clipped bounded proposal reaches or crosses 0 or 1; it is distinct from final boundary alpha after fallback and exact-grid mapping.

For each magnitude bucket, Delta MRR contribution is `sum(rr_anchored - rr_global in bucket) / all observations in the group`. Bucket contributions therefore add exactly to the group's overall Delta MRR. Harm Rate is conditional on the bucket; Mean Harm and Mean Benefit are conditional on harmful and beneficial observations within that bucket.

## Pair-level magnitude summary

| Split | Dataset/Pair | Changed % | Exact anchor % | Mean abs Delta alpha | Median | p90 | p95 | Max | Positive % | Negative % | Boundary % | Saturation % | Delta MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV | MKG-W / M-Hyper + NativE | 83.55 | 16.45 | 0.0886 | 0.1000 | 0.2000 | 0.2000 | 0.2000 | 40.29 | 43.26 | 0.00 | 0.00 | +0.004209 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 36.82 | 63.18 | 0.0310 | 0.0000 | 0.1000 | 0.1000 | 0.2000 | 0.00 | 36.82 | 63.18 | 50.48 | +0.001261 |
| DEV | DB15K / M-Hyper + NativE | 39.68 | 60.32 | 0.0283 | 0.0000 | 0.1000 | 0.1000 | 0.1500 | 0.00 | 39.68 | 60.32 | 51.40 | +0.000606 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 21.20 | 78.80 | 0.0227 | 0.0000 | 0.1000 | 0.1000 | 0.2000 | 0.00 | 21.20 | 78.80 | 51.59 | +0.000267 |
| TEST | MKG-W / M-Hyper + NativE | 94.26 | 5.74 | 0.2078 | 0.2000 | 0.4000 | 0.4000 | 0.4000 | 44.68 | 49.57 | 21.00 | 19.36 | +0.005872 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 35.63 | 64.37 | 0.0760 | 0.0000 | 0.2500 | 0.3000 | 0.4500 | 0.00 | 35.63 | 64.37 | 50.36 | +0.001984 |
| TEST | DB15K / M-Hyper + NativE | 44.42 | 55.58 | 0.0693 | 0.0000 | 0.2000 | 0.2500 | 0.4500 | 0.00 | 44.42 | 55.58 | 52.20 | +0.001271 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 6.98 | 93.02 | 0.0223 | 0.0000 | 0.0000 | 0.3000 | 0.4500 | 0.00 | 6.98 | 93.02 | 52.11 | +0.000714 |

## Locked TEST bucket decomposition

| Dataset/Pair | Abs Delta alpha bucket | n (%) | Delta MRR contribution | Harm % | Mean Harm | Mean Benefit |
|---|---|---:|---:|---:|---:|---:|
| MKG-W / M-Hyper + NativE | 0 | 1473 (5.74) | +0.000000 | 0.00 | -- | -- |
| MKG-W / M-Hyper + NativE | (0,0.05] | 3099 (12.08) | +0.000093 | 23.27 | 0.024951 | 0.020417 |
| MKG-W / M-Hyper + NativE | (0.05,0.10] | 3510 (13.69) | +0.000111 | 25.81 | 0.031964 | 0.021316 |
| MKG-W / M-Hyper + NativE | (0.10,0.20] | 7000 (27.30) | +0.000994 | 25.77 | 0.030292 | 0.022058 |
| MKG-W / M-Hyper + NativE | >0.20 | 10562 (41.19) | +0.004674 | 16.05 | 0.039437 | 0.078051 |
| MKG-W / M-Hyper + AdaMF-MAT | 0 | 16508 (64.37) | +0.000000 | 0.00 | -- | -- |
| MKG-W / M-Hyper + AdaMF-MAT | (0,0.05] | 0 (0.00) | +0.000000 | -- | -- | -- |
| MKG-W / M-Hyper + AdaMF-MAT | (0.05,0.10] | 1209 (4.71) | +0.000013 | 18.28 | 0.017613 | 0.005910 |
| MKG-W / M-Hyper + AdaMF-MAT | (0.10,0.20] | 4292 (16.74) | +0.000125 | 22.06 | 0.027655 | 0.010448 |
| MKG-W / M-Hyper + AdaMF-MAT | >0.20 | 3635 (14.17) | +0.001846 | 19.39 | 0.021887 | 0.022689 |
| DB15K / M-Hyper + NativE | 0 | 33023 (55.58) | +0.000000 | 0.00 | -- | -- |
| DB15K / M-Hyper + NativE | (0,0.05] | 4563 (7.68) | +0.000043 | 12.47 | 0.017040 | 0.008056 |
| DB15K / M-Hyper + NativE | (0.05,0.10] | 5056 (8.51) | -0.000031 | 18.22 | 0.033732 | 0.013277 |
| DB15K / M-Hyper + NativE | (0.10,0.20] | 11701 (19.69) | +0.000514 | 24.72 | 0.035493 | 0.020896 |
| DB15K / M-Hyper + NativE | >0.20 | 5069 (8.53) | +0.000746 | 27.66 | 0.039717 | 0.031480 |
| DB15K / M-Hyper + AdaMF-MAT | 0 | 55268 (93.02) | +0.000000 | 0.00 | -- | -- |
| DB15K / M-Hyper + AdaMF-MAT | (0,0.05] | 0 (0.00) | +0.000000 | -- | -- | -- |
| DB15K / M-Hyper + AdaMF-MAT | (0.05,0.10] | 0 (0.00) | +0.000000 | -- | -- | -- |
| DB15K / M-Hyper + AdaMF-MAT | (0.10,0.20] | 0 (0.00) | +0.000000 | -- | -- | -- |
| DB15K / M-Hyper + AdaMF-MAT | >0.20 | 4144 (6.98) | +0.000714 | 29.46 | 0.040375 | 0.033439 |

## Reproducibility and information boundary

This is a read-only post-hoc audit. It trains no model, changes no anchor, beta, confidence threshold, fallback decision, or TEST result, and performs no selection from TEST. DEV and TEST are reported separately.

| Split | Dataset/Pair | Rows | Stored Delta checked | Fallback exact-anchor check | Source SHA-256 | DEV lock SHA-256 |
|---|---|---:|---:|---:|---|---|
| DEV | MKG-W / M-Hyper + NativE | 25656 | yes | yes | `e0a98d388819d535c15585165bb4fea18c47a7bb70e31ed0d21902c97dbf7d50` | `eee533f2d0b8b1a9a181d348c1df94b3db522b5f7c4a5a398a946b58d201c713` |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 25656 | yes | yes | `fbaa2a0faea02aa6191deff8110c6d10e78c1ac1092a4bd83b4926b06160a6b4` | `95109445388ea1ffeab17c11b5281880ca3928c827155d2f1e9486dbab93bb0f` |
| DEV | DB15K / M-Hyper + NativE | 47532 | yes | yes | `f49663bce80db958d4fbb3d576ed841dc5926387a9d67959533db484d8249dc3` | `a3ddf0b07697df57140c8a1dacdeb38ee96598272805b9db2a44bd59fa2b57ef` |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 47532 | yes | yes | `7b13db8cd4094ec4ce1651fa3a95a1d80f1c9e859d0d26381fe507ada372e066` | `b7ec3d6b524d16cf9c2128a3b84c8881cbe5e6b8ee7c2e5d592c58f756dac55d` |
| TEST | MKG-W / M-Hyper + NativE | 25644 | reconstructed | yes | `469d2d427b61d3048de03ef485d804e2821c40a8af9a8e5e62d61e2ff4bbd624` | `eee533f2d0b8b1a9a181d348c1df94b3db522b5f7c4a5a398a946b58d201c713` |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 25644 | reconstructed | yes | `0414041b7cd33267e79f0d7b6c12575f581f7e7ddaa22724c0a041d0cc087e60` | `95109445388ea1ffeab17c11b5281880ca3928c827155d2f1e9486dbab93bb0f` |
| TEST | DB15K / M-Hyper + NativE | 59412 | reconstructed | yes | `2155f299165204d388ad45a509684148a3db3d3fa0dae9370896d2ed42271cfc` | `a3ddf0b07697df57140c8a1dacdeb38ee96598272805b9db2a44bd59fa2b57ef` |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 59412 | reconstructed | yes | `77bb9f968099600ff110f110dea8cec197f5891396a055d35bd82542dce06e0c` | `b7ec3d6b524d16cf9c2128a3b84c8881cbe5e6b8ee7c2e5d592c58f756dac55d` |
