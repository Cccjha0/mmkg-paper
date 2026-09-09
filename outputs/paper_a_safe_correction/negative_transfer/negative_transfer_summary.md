# Paper A query-level negative-transfer analysis

## TEST main table

| Dataset/Pair | Method | Delta MRR | Harm % | Benefit % | Unchanged % | Mean Harm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DB15K / M-Hyper + AdaMF-MAT | Global | 0.000000 | 0.00 | 0.00 | 100.00 | -- |
| DB15K / M-Hyper + AdaMF-MAT | Query-soft | -0.026595 | 32.65 | 36.40 | 30.95 | 0.146308 |
| DB15K / M-Hyper + AdaMF-MAT | DynaSemble | -0.070642 | 46.88 | 31.77 | 21.34 | 0.210551 |
| DB15K / M-Hyper + AdaMF-MAT | Anchored Dynamic | 0.000714 | 2.06 | 4.62 | 93.33 | 0.040375 |
| DB15K / M-Hyper + NativE | Global | 0.000000 | 0.00 | 0.00 | 100.00 | -- |
| DB15K / M-Hyper + NativE | Query-soft | -0.008946 | 29.04 | 36.62 | 34.34 | 0.107805 |
| DB15K / M-Hyper + NativE | DynaSemble | -0.041145 | 41.31 | 31.33 | 27.36 | 0.168944 |
| DB15K / M-Hyper + NativE | Anchored Dynamic | 0.001271 | 9.74 | 22.34 | 67.93 | 0.034421 |
| MKG-W / M-Hyper + AdaMF-MAT | Global | 0.000000 | 0.00 | 0.00 | 100.00 | -- |
| MKG-W / M-Hyper + AdaMF-MAT | Query-soft | -0.007251 | 24.02 | 43.09 | 32.90 | 0.112009 |
| MKG-W / M-Hyper + AdaMF-MAT | DynaSemble | -0.049308 | 39.69 | 37.63 | 22.68 | 0.200402 |
| MKG-W / M-Hyper + AdaMF-MAT | Anchored Dynamic | 0.001984 | 7.30 | 24.56 | 68.14 | 0.024299 |
| MKG-W / M-Hyper + NativE | Global | 0.000000 | 0.00 | 0.00 | 100.00 | -- |
| MKG-W / M-Hyper + NativE | Query-soft | 0.006201 | 17.30 | 37.52 | 45.18 | 0.035085 |
| MKG-W / M-Hyper + NativE | DynaSemble | -0.016030 | 30.77 | 39.37 | 29.85 | 0.141934 |
| MKG-W / M-Hyper + NativE | Anchored Dynamic | 0.005872 | 19.99 | 33.20 | 46.81 | 0.032861 |

Delta MRR is the average of `rr_method - rr_global`. Mean Harm is the positive magnitude `mean(-delta_rr | delta_rr < 0)`. Outcomes with `abs(delta_rr) <= 1e-12` are unchanged.

Average performance improvement and query-level safety are different estimands: a method can improve MRR while still harming many queries, or improve more queries than it harms while losing MRR because harmful corrections are larger.

## Result interpretation

- Anchored Dynamic improves mean MRR on 4/4 pairs; all 4/4 clustered 95% CIs have a positive lower bound. Its harmful-query rate is not zero, ranging from 2.06% to 19.99%. Thus the supported claim is lower observed negative-transfer risk, not query-wise safety.
- Query-soft improves mean MRR on only 1/4 pairs even though beneficial queries outnumber harmful queries on 4/4 pairs. The sign reversal comes from loss magnitude: harmful corrections can be much larger than beneficial ones.
- DynaSemble improves mean MRR on 0/4 pairs. Its four Delta MRR estimates and materially different harm rates quantify pair-sensitive behavior under a strong primary.
- Across all 12 adaptive pair-method results, Delta MRR alone does not identify how often rankings are harmed or how large conditional losses are; both dimensions should be reported.

## Stress case: DB15K / M-Hyper + AdaMF-MAT

| Dataset/Pair | Method | Delta MRR | Harm % | Benefit % | Unchanged % | Mean Harm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DB15K / M-Hyper + AdaMF-MAT | Query-soft | -0.026595 | 32.65 | 36.40 | 30.95 | 0.146308 |
| DB15K / M-Hyper + AdaMF-MAT | DynaSemble | -0.070642 | 46.88 | 31.77 | 21.34 | 0.210551 |
| DB15K / M-Hyper + AdaMF-MAT | Anchored Dynamic | 0.000714 | 2.06 | 4.62 | 93.33 | 0.040375 |

In this stress case, Query-soft has Delta MRR -0.026595 with 32.65% harmful queries; DynaSemble has Delta MRR -0.070642 with 46.88% harmful queries; Anchored Dynamic has Delta MRR 0.000714 with 2.06% harmful queries. This is a joint comparison of average utility and observed harm frequency, not a formal worst-case guarantee.

## Statistical inference

Percentile 95% intervals use 10,000 clustered bootstrap samples with seed 20260909. The bootstrap unit is the original raw triple; all available seeds and both directions remain inside that unit. The six seed-direction observations are never treated as independent samples. Intervals for Delta MRR, Harm Rate, and Mean Harm are stored in `bootstrap_ci.json` and the summary CSV files.

## Information boundary

This is final TEST analysis of immutable existing query-level outputs. No model or combiner is trained, no threshold or method is selected, and no TEST statistic is fed back into Global, Query-soft, DynaSemble, or Anchored Dynamic.
