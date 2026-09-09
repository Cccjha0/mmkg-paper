# Paper A Anchored Dynamic Fallback Behavior Audit

## Main answer

Across grouped OOF DEV and locked TEST, the final confidence rule activates fallback in 3/4 main pairs. On locked TEST, 2/4 pairs have a zero threshold and therefore no finite-query fallback. Across the active pair/split cases, fallback has positive aggregate counterfactual utility in 5/5 cases: avoided raw RR loss exceeds sacrificed raw RR gain. This does not imply that every fallback decision is correct; avoided harm, missed benefit, and neutral outcomes are reported separately below.

## Counterfactual definition

For each query, the final stored policy is reconstructed from its actual alpha0, beta, decision, confidence threshold, and exact alpha grid. The counterfactual then removes only confidence fallback:

```text
alpha_raw = clip(alpha0 + beta * tanh(decision), 0, 1).
```

Non-finite queries still fall back. A final fallback query is `avoided_harm` if `rr_raw < rr_global`, `missed_benefit` if `rr_raw > rr_global`, and `neutral` under exact equality. Net fallback utility is total avoided RR loss minus total potential RR gain sacrificed, equivalently `sum(rr_global - rr_raw)` over fallback queries.

## Pair-level summary

| Split | Dataset/Pair | Fallback n (%) | Changed % | Effective adaptation % | Delta MRR | Avoided n (%) | Missed n (%) | Neutral n (%) | RR loss avoided | Gain sacrificed | Net fallback utility |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV | MKG-W / M-Hyper + NativE | 1220 (4.76) | 83.55 | 83.55 | +0.004209 | 115 (9.43) | 175 (14.34) | 930 (76.23) | 3.038544 | 2.644390 | +0.394155 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 4955 (19.31) | 36.82 | 36.82 | +0.001261 | 182 (3.67) | 611 (12.33) | 4162 (84.00) | 2.385715 | 1.775737 | +0.609978 |
| DEV | DB15K / M-Hyper + NativE | 0 (0.00) | 39.68 | 39.68 | +0.000606 | 0 (--) | 0 (--) | 0 (--) | 0.000000 | 0.000000 | +0.000000 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 21764 (45.79) | 21.20 | 21.20 | +0.000267 | 1522 (6.99) | 3885 (17.85) | 16357 (75.16) | 57.584430 | 35.409869 | +22.174560 |
| TEST | MKG-W / M-Hyper + NativE | 0 (0.00) | 94.26 | 94.26 | +0.005872 | 0 (--) | 0 (--) | 0 (--) | 0.000000 | 0.000000 | +0.000000 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 6211 (24.22) | 35.63 | 35.63 | +0.001984 | 405 (6.52) | 1365 (21.98) | 4441 (71.50) | 13.305775 | 12.193140 | +1.112635 |
| TEST | DB15K / M-Hyper + NativE | 0 (0.00) | 44.42 | 44.42 | +0.001271 | 0 (--) | 0 (--) | 0 (--) | 0.000000 | 0.000000 | +0.000000 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 39897 (67.15) | 6.98 | 6.98 | +0.000714 | 5244 (13.14) | 11555 (28.96) | 23098 (57.89) | 283.118993 | 222.942339 | +60.176654 |

Rates in the avoided/missed/neutral columns use final fallback queries as the denominator. Changed-from-anchor and effective-adaptation rates use all queries. They are equal here because every fallback query is verified to return exactly to alpha0.

## DB15K / M-Hyper + AdaMF-MAT stress case

- DEV: fallback covers 45.79% (21764/47532) of observations. Among fallback queries, 6.99% avoid harm, 17.85% miss a benefit, and 75.16% are neutral. Total RR loss avoided is 57.584430, potential gain sacrificed is 35.409869, and net fallback utility is +22.174560 (+0.000467 MRR contribution over all observations).
- TEST: fallback covers 67.15% (39897/59412) of observations. Among fallback queries, 13.14% avoid harm, 28.96% miss a benefit, and 57.89% are neutral. Total RR loss avoided is 283.118993, potential gain sacrificed is 222.942339, and net fallback utility is +60.176654 (+0.001013 MRR contribution over all observations).

This stress case quantifies the mechanism's tradeoff rather than using it to retune the threshold. TEST composition is post-hoc diagnostic evidence only.

## Figures

- `outputs/paper_a_safe_correction/fallback_audit/fallback_rate_vs_delta_mrr.pdf`
- `outputs/paper_a_safe_correction/fallback_audit/fallback_composition.pdf`

## Integrity and information boundary

| Split | Dataset/Pair | Rows | Fallback | Non-finite | Stored reconstruction | Source SHA-256 | DEV lock SHA-256 |
|---|---|---:|---:|---:|---:|---|---|
| DEV | MKG-W / M-Hyper + NativE | 25656 | 1220 | 0 | yes | `e0a98d388819d535c15585165bb4fea18c47a7bb70e31ed0d21902c97dbf7d50` | `eee533f2d0b8b1a9a181d348c1df94b3db522b5f7c4a5a398a946b58d201c713` |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 25656 | 4955 | 0 | yes | `fbaa2a0faea02aa6191deff8110c6d10e78c1ac1092a4bd83b4926b06160a6b4` | `95109445388ea1ffeab17c11b5281880ca3928c827155d2f1e9486dbab93bb0f` |
| DEV | DB15K / M-Hyper + NativE | 47532 | 0 | 0 | yes | `f49663bce80db958d4fbb3d576ed841dc5926387a9d67959533db484d8249dc3` | `a3ddf0b07697df57140c8a1dacdeb38ee96598272805b9db2a44bd59fa2b57ef` |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 47532 | 21764 | 0 | yes | `7b13db8cd4094ec4ce1651fa3a95a1d80f1c9e859d0d26381fe507ada372e066` | `b7ec3d6b524d16cf9c2128a3b84c8881cbe5e6b8ee7c2e5d592c58f756dac55d` |
| TEST | MKG-W / M-Hyper + NativE | 25644 | 0 | 0 | yes | `469d2d427b61d3048de03ef485d804e2821c40a8af9a8e5e62d61e2ff4bbd624` | `eee533f2d0b8b1a9a181d348c1df94b3db522b5f7c4a5a398a946b58d201c713` |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 25644 | 6211 | 0 | yes | `0414041b7cd33267e79f0d7b6c12575f581f7e7ddaa22724c0a041d0cc087e60` | `95109445388ea1ffeab17c11b5281880ca3928c827155d2f1e9486dbab93bb0f` |
| TEST | DB15K / M-Hyper + NativE | 59412 | 0 | 0 | yes | `2155f299165204d388ad45a509684148a3db3d3fa0dae9370896d2ed42271cfc` | `a3ddf0b07697df57140c8a1dacdeb38ee96598272805b9db2a44bd59fa2b57ef` |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 59412 | 39897 | 0 | yes | `77bb9f968099600ff110f110dea8cec197f5891396a055d35bd82542dce06e0c` | `b7ec3d6b524d16cf9c2128a3b84c8881cbe5e6b8ee7c2e5d592c58f756dac55d` |

This script trains no model, changes no threshold, and performs no TEST-based selection. Grouped OOF DEV and immutable locked TEST are reported separately. Every stored fallback flag, final alpha, final RR, and Global RR is reconstructed from exact-ranking assets before counterfactual classification.
