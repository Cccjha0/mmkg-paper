# Paper A DEV-only reliable-primary audit

## Decision table

| Dataset | Pair | Selected Primary | DEV Delta | 3/3 Seeds? | CI95 | Regime |
| --- | --- | --- | ---: | --- | --- | --- |
| MKG-W | M-Hyper vs NativE | M-Hyper | 0.024379 | Yes | [0.020243, 0.028682] | reliable-primary |
| MKG-W | M-Hyper vs AdaMF-MAT | M-Hyper | 0.063075 | Yes | [0.058193, 0.067976] | reliable-primary |
| MKG-W | NativE vs AdaMF-MAT | NativE | 0.038696 | Yes | [0.034434, 0.043062] | reliable-primary |
| DB15K | M-Hyper vs NativE | M-Hyper | 0.061565 | Yes | [0.058097, 0.065000] | reliable-primary |
| DB15K | M-Hyper vs AdaMF-MAT | M-Hyper | 0.090569 | Yes | [0.086837, 0.094249] | reliable-primary |
| DB15K | NativE vs AdaMF-MAT | NativE | 0.029004 | Yes | [0.026140, 0.031850] | reliable-primary |

## Automatic checks

- All four pre-existing M-Hyper main pairs are reliable-primary: **Yes**.
- MKG-W NativE + AdaMF-MAT is boundary under this rule: **No**.
- DB15K NativE + AdaMF-MAT is boundary under this rule: **No**.

Both NativE + AdaMF-MAT pairs satisfy the strict standalone DEV definition: NativE has higher pooled MRR, is higher in all three paired seeds, and has a positive clustered-CI lower bound. Therefore they are not reliable-primary boundary cases under this protocol. Any later failure of adaptive combination must be described as a combination-level stress case, not retroactively used to change the primary-selection rule.

## Per-seed deltas

| Dataset | Pair | Seed 1 | Seed 2 | Seed 3 |
| --- | --- | ---: | ---: | ---: |
| MKG-W | M-Hyper vs NativE | 0.023881 | 0.026500 | 0.022756 |
| MKG-W | M-Hyper vs AdaMF-MAT | 0.073270 | 0.055614 | 0.060340 |
| MKG-W | NativE vs AdaMF-MAT | 0.049389 | 0.029114 | 0.037584 |
| DB15K | M-Hyper vs NativE | 0.064398 | 0.056302 | 0.063994 |
| DB15K | M-Hyper vs AdaMF-MAT | 0.090901 | 0.098156 | 0.082651 |
| DB15K | NativE vs AdaMF-MAT | 0.026503 | 0.041853 | 0.018656 |

## Statistical and information boundary

The percentile intervals use 10,000 bootstrap resamples with seed 20260909. The original raw triple is the sampling unit; its three paired seeds and both prediction directions remain together. Selection uses only existing exact filtered DEV reciprocal ranks. No TEST file, outcome, threshold, or method selection is read, and no post-hoc MRR margin is introduced.
