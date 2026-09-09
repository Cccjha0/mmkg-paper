# Paper A Selective Adaptation / Risk-Coverage Report

## Main answer

Allowing more queries to adapt initially increases net MRR, but negative-transfer risk accumulates throughout the curve and the marginal utility eventually diminishes. At the pooled level, both grouped OOF DEV and locked TEST reach their largest observed Delta MRR at 70% on the pre-fixed grid, then decline as coverage expands further. The behavior is pair-sensitive: three pairs remain net-positive at 100%, whereas DB15K / M-Hyper + AdaMF-MAT reverses below Global when every query is offered raw adaptation.

## Protocol boundary

This is a diagnostic analysis of existing query-level assets. It trains no base model or combiner, changes neither Anchored Dynamic nor beta, and performs no TEST-driven threshold or model selection. The fixed coverage grid is 0%, 10%, ..., 100%.

Confidence is `abs(2 * anchored_probability_a - 1)`. Within each pair and split, queries are sorted by decreasing confidence with `query_id` as a deterministic tie-break. The pooled curve is a micro-average of the four pair-wise selections. Selected finite queries use `clip(alpha0 + beta * tanh(decision), 0, 1)`, rounded with the existing nearest-grid rule; all other queries use alpha0. Reciprocal ranks are read from the precomputed exact filtered full-ranking alpha-grid columns.

Grouped OOF DEV uses held-out-triple decision/probability values but the final DEV-locked alpha0 and beta for each pair, as required by this diagnostic. Locked TEST uses the same final lock. The official confidence threshold is not applied inside this curve: coverage itself defines selection, while non-finite queries always fall back.

## Pair-level checkpoints

| Split | Dataset/Pair | Coverage | Delta MRR | Harm % | Benefit % | Changed-alpha % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DEV | MKG-W / M-Hyper + NativE | 10% | 0.001636 | 0.08 | 0.38 | 10.00 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 10% | 0.000000 | 0.00 | 0.00 | 0.00 |
| DEV | DB15K / M-Hyper + NativE | 10% | 0.000015 | 0.00 | 0.02 | 0.02 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 10% | 0.000082 | 0.04 | 0.13 | 0.17 |
| DEV | MKG-W / M-Hyper + NativE | 50% | 0.005123 | 8.77 | 14.53 | 50.00 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 50% | 0.002432 | 3.30 | 13.53 | 17.72 |
| DEV | DB15K / M-Hyper + NativE | 50% | 0.001027 | 4.89 | 10.61 | 18.09 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 50% | 0.000477 | 5.00 | 11.92 | 18.47 |
| DEV | MKG-W / M-Hyper + NativE | 100% | 0.006163 | 19.78 | 33.86 | 94.08 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 100% | 0.002178 | 8.77 | 30.36 | 46.29 |
| DEV | DB15K / M-Hyper + NativE | 100% | 0.001339 | 9.88 | 22.53 | 45.31 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 100% | -0.000516 | 10.62 | 24.32 | 45.72 |
| TEST | MKG-W / M-Hyper + NativE | 10% | 0.001378 | 0.09 | 0.32 | 10.00 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 10% | 0.000000 | 0.00 | 0.00 | 0.00 |
| TEST | DB15K / M-Hyper + NativE | 10% | 0.000024 | 0.01 | 0.06 | 0.08 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 10% | 0.000098 | 0.07 | 0.18 | 0.26 |
| TEST | MKG-W / M-Hyper + NativE | 50% | 0.005035 | 9.03 | 13.98 | 50.00 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 50% | 0.001903 | 3.51 | 13.26 | 17.69 |
| TEST | DB15K / M-Hyper + NativE | 50% | 0.001026 | 4.71 | 10.44 | 17.42 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 50% | 0.000737 | 5.27 | 11.33 | 17.98 |
| TEST | MKG-W / M-Hyper + NativE | 100% | 0.005872 | 19.99 | 33.20 | 94.26 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 100% | 0.001941 | 8.88 | 29.88 | 46.28 |
| TEST | DB15K / M-Hyper + NativE | 100% | 0.001271 | 9.74 | 22.34 | 44.42 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 100% | -0.000299 | 10.88 | 24.07 | 45.14 |

## Pooled curve

| Split | Dataset/Pair | Coverage | Delta MRR | Harm % | Benefit % | Changed-alpha % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DEV | Pooled / Pooled | 0% | 0.000000 | 0.00 | 0.00 | 0.00 |
| DEV | Pooled / Pooled | 10% | 0.000318 | 0.03 | 0.11 | 1.82 |
| DEV | Pooled / Pooled | 20% | 0.000752 | 0.35 | 0.76 | 4.33 |
| DEV | Pooled / Pooled | 30% | 0.001326 | 1.67 | 3.60 | 9.60 |
| DEV | Pooled / Pooled | 40% | 0.001608 | 3.47 | 7.82 | 16.42 |
| DEV | Pooled / Pooled | 50% | 0.001813 | 5.32 | 12.23 | 23.74 |
| DEV | Pooled / Pooled | 60% | 0.001977 | 7.08 | 16.38 | 31.03 |
| DEV | Pooled / Pooled | 70% | 0.001979 | 8.70 | 20.08 | 38.12 |
| DEV | Pooled / Pooled | 80% | 0.001890 | 10.10 | 23.23 | 45.00 |
| DEV | Pooled / Pooled | 90% | 0.001801 | 11.21 | 25.68 | 51.61 |
| DEV | Pooled / Pooled | 100% | 0.001729 | 11.66 | 26.47 | 54.16 |
| TEST | Pooled / Pooled | 0% | 0.000000 | 0.00 | 0.00 | 0.00 |
| TEST | Pooled / Pooled | 10% | 0.000250 | 0.04 | 0.13 | 1.62 |
| TEST | Pooled / Pooled | 20% | 0.000692 | 0.35 | 0.71 | 3.83 |
| TEST | Pooled / Pooled | 30% | 0.001165 | 1.62 | 3.41 | 8.76 |
| TEST | Pooled / Pooled | 40% | 0.001549 | 3.48 | 7.54 | 15.50 |
| TEST | Pooled / Pooled | 50% | 0.001662 | 5.38 | 11.71 | 22.57 |
| TEST | Pooled / Pooled | 60% | 0.001721 | 7.14 | 15.76 | 29.71 |
| TEST | Pooled / Pooled | 70% | 0.001740 | 8.71 | 19.35 | 36.65 |
| TEST | Pooled / Pooled | 80% | 0.001641 | 10.08 | 22.46 | 43.41 |
| TEST | Pooled / Pooled | 90% | 0.001549 | 11.15 | 24.87 | 49.90 |
| TEST | Pooled / Pooled | 100% | 0.001517 | 11.55 | 25.72 | 52.46 |

## Interpretation

- DEV: expanding to 100% requested coverage gives Delta MRR +0.001729, harm rate 11.66%, and changed-alpha rate 54.16%. The largest observed pooled Delta MRR on the fixed grid occurs at 70% coverage (+0.001979); this is retrospective description only, not a selected operating point.
- TEST: expanding to 100% requested coverage gives Delta MRR +0.001517, harm rate 11.55%, and changed-alpha rate 52.46%. The largest observed pooled Delta MRR on the fixed grid occurs at 70% coverage (+0.001740); this is retrospective description only, not a selected operating point.
- MKG-W / M-Hyper + NativE: the largest observed grid value is at 80% on DEV (+0.006353) and 90% on TEST (+0.005925). At 100% TEST coverage, Delta MRR is +0.005872 and harm rate is 19.99%.
- MKG-W / M-Hyper + AdaMF-MAT: the largest observed grid value is at 60% on DEV (+0.002583) and 60% on TEST (+0.002066). At 100% TEST coverage, Delta MRR is +0.001941 and harm rate is 8.88%.
- DB15K / M-Hyper + NativE: the largest observed grid value is at 100% on DEV (+0.001339) and 100% on TEST (+0.001271). At 100% TEST coverage, Delta MRR is +0.001271 and harm rate is 9.74%.
- DB15K / M-Hyper + AdaMF-MAT: the largest observed grid value is at 30% on DEV (+0.000517) and 40% on TEST (+0.000930). At 100% TEST coverage, Delta MRR is -0.000299 and harm rate is 10.88%.
- Harm rate is cumulative over all queries, so increasing coverage exposes additional queries to both beneficial and harmful corrections. Delta MRR shows the net magnitude balance; harm rate shows how widely negative transfer is distributed. Neither statistic substitutes for the other.
- Changed-alpha rate can be lower than adaptation coverage because a bounded raw correction may round back to alpha0 on the exact-ranking grid.
- This difference is especially visible when alpha0=1.0: the highest-confidence queries often support the primary expert, so a positive correction clips back to the anchor and changes no grid alpha. Requested coverage therefore should not be interpreted as effective intervention coverage.
- At 70% pooled TEST coverage, head queries have a higher harm frequency (12.59% vs. 4.83%), while harmful tail corrections have a larger conditional mean magnitude (0.064407 vs. 0.029011). Risk frequency and risk severity therefore remain distinct even after conditioning on coverage.

## Figures

- Coverage vs Delta MRR: `outputs/paper_a_safe_correction/risk_coverage/coverage_vs_delta_mrr.pdf`
- Coverage vs Harm Rate: `outputs/paper_a_safe_correction/risk_coverage/coverage_vs_harm_rate.pdf`

Each figure contains the four main model pairs and an additional pooled panel, with grouped OOF DEV and locked TEST shown separately. PDF, PNG, and SVG are generated without a hard-coded color palette.

## Source and lock audit

| Split | Dataset/Pair | alpha0 | beta | Final lock threshold | Stored-policy fallback % |
| --- | --- | ---: | ---: | ---: | ---: |
| DEV | MKG-W / M-Hyper + NativE | 0.60 | 0.50 | 0.00 | 4.76 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 1.00 | 0.50 | 0.10 | 19.31 |
| DEV | DB15K / M-Hyper + NativE | 1.00 | 0.50 | 0.00 | 0.00 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 1.00 | 0.50 | 0.30 | 45.79 |
| TEST | MKG-W / M-Hyper + NativE | 0.60 | 0.50 | 0.00 | 0.00 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 1.00 | 0.50 | 0.10 | 24.22 |
| TEST | DB15K / M-Hyper + NativE | 1.00 | 0.50 | 0.00 | 0.00 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 1.00 | 0.50 | 0.30 | 67.15 |

For DEV, the stored fallback column comes from the original fold-specific OOF policies and is shown only as an audit reference; the diagnostic curve uses the final DEV-locked alpha0/beta and replaces the confidence threshold with the fixed coverage rule. Directional tables summarize the same pair-wise selection and do not re-rank within head or tail.

Every source row file and lock was SHA-256 audited during execution. Stored confidence, fallback, applied alpha, and exact RR were reconstructed and checked against the original implementation before the diagnostic curve was computed.
