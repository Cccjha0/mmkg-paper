# Paper A Core Safety-Component Ablation

## Scope and matched design

This experiment changes only the requested safety component. It retrains no base MMKGC model and does not alter Anchored Dynamic. Grouped OOF DEV reuses the exact P3 folds and full-geometry logistic model. Full selects beta and confidence threshold on each outer training fold using the existing protocol; No Bound and No Fallback inherit that fold's Full winner. TEST uses the immutable final DEV lock and performs no tuning.

No Anchor is the current Query-soft logistic baseline. No Bound fixes beta=1.0 while retaining Full's alpha0, model, confidence mechanism, and threshold. No Fallback retains Full's alpha0, beta, and model but fixes the confidence threshold to zero; only non-finite features may fall back.

## TEST main table

| Dataset/Pair | Variant | Delta MRR | Harm % | Mean Harm | Changed % |
| --- | --- | ---: | ---: | ---: | ---: |
| MKG-W / M-Hyper + NativE | Full Anchored | +0.005872 | 19.99 | 0.032861 | 94.26 |
| MKG-W / M-Hyper + NativE | No Anchor / Query-soft | +0.006201 | 17.30 | 0.035085 | 94.45 |
| MKG-W / M-Hyper + NativE | No Bound | +0.007231 | 23.86 | 0.042776 | 97.04 |
| MKG-W / M-Hyper + NativE | No Fallback | +0.005872 | 19.99 | 0.032861 | 94.26 |
| MKG-W / M-Hyper + AdaMF-MAT | Full Anchored | +0.001984 | 7.30 | 0.024299 | 35.63 |
| MKG-W / M-Hyper + AdaMF-MAT | No Anchor / Query-soft | -0.007251 | 24.02 | 0.112009 | 98.03 |
| MKG-W / M-Hyper + AdaMF-MAT | No Bound | +0.003631 | 8.97 | 0.038983 | 35.63 |
| MKG-W / M-Hyper + AdaMF-MAT | No Fallback | +0.001941 | 8.88 | 0.025820 | 46.28 |
| DB15K / M-Hyper + NativE | Full Anchored | +0.001271 | 9.74 | 0.034421 | 44.42 |
| DB15K / M-Hyper + NativE | No Anchor / Query-soft | -0.008946 | 29.04 | 0.107805 | 100.00 |
| DB15K / M-Hyper + NativE | No Bound | +0.002521 | 12.09 | 0.051427 | 46.14 |
| DB15K / M-Hyper + NativE | No Fallback | +0.001271 | 9.74 | 0.034421 | 44.42 |
| DB15K / M-Hyper + AdaMF-MAT | Full Anchored | +0.000714 | 2.06 | 0.040375 | 6.98 |
| DB15K / M-Hyper + AdaMF-MAT | No Anchor / Query-soft | -0.026595 | 32.65 | 0.146308 | 100.00 |
| DB15K / M-Hyper + AdaMF-MAT | No Bound | +0.000971 | 2.53 | 0.055753 | 6.98 |
| DB15K / M-Hyper + AdaMF-MAT | No Fallback | -0.000299 | 10.88 | 0.051418 | 45.14 |

## Aggregate positivity

| Variant | Pairs positive | Pair-seeds positive | Pair-directions positive |
| --- | ---: | ---: | ---: |
| Full Anchored | 4/4 | 12/12 | 8/8 |
| No Anchor / Query-soft | 1/4 | 3/12 | 2/8 |
| No Bound | 4/4 | 12/12 | 8/8 |
| No Fallback | 3/4 | 10/12 | 6/8 |

## Answers to the core questions

- Anchor: Full is positive on 4/4 TEST pairs, 12/12 pair-seeds, and 8/8 pair-directions. Query-soft is positive on 1/4, 3/12, and 2/8, respectively. This shows that anchoring reduces pair-dependent negative transfer and improves sign consistency. It is not uniform dominance: Query-soft has a larger Delta MRR on 1/4 TEST pairs.
- Bound: fixing beta=1.0 reduces Delta MRR relative to Full on 0/4 TEST pairs and raises harm rate on 4/4; it also raises conditional Mean Harm on 4/4 and mean alpha deviation on 4/4. Thus the local bound is supported as a conservative risk-control component, even when beta=1.0 obtains greater average utility.
- Fallback stress case: on DB15K / M-Hyper + AdaMF-MAT, Full has Delta MRR +0.000714 and harm rate 2.06%; No Fallback has Delta MRR -0.000299 and harm rate 10.88%. The difference isolates the protection supplied by the locked confidence fallback.

## Statistical inference

Delta MRR, Harm Rate, and Mean Harm intervals use 10,000 percentile clustered-bootstrap samples with seed 20260910. The cluster is the original raw triple, so all seeds and both directions remain inside one resampling unit.

## Audit boundary

The reconstructed Full and Query-soft DEV results match the existing P3 assets. Their locked TEST alpha and reciprocal ranks match the existing TEST query rows. Source hashes and fold-wise Full selections are recorded in `outputs/paper_a_safe_correction/core_ablation/audit.json`.
