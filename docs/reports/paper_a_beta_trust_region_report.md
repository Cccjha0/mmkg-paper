# Paper A Bounded-Correction / Beta Trust-Region Report

## Main answer

Beta has a direct operational meaning: it bounds how far the fused score of every candidate can move away from the DEV-selected Global anchor, scaled by the candidate's inter-expert score disagreement. Empirically, all 10 formal beta values remain net-positive in 8/8 pair/split curves, which is a wide sign-stable performance region. This stability does not imply constant risk: Harm Rate is monotone non-decreasing in 8/8 curves and mean alpha deviation in 8/8. Relative to the locked beta, `beta=1.0` raises Delta MRR in 8/8 cases, but also raises harm frequency in 8/8 and conditional Mean Harm in 8/8. The local bound therefore acts as a risk budget even when a wider correction range improves mean MRR.

## Theoretical derivation

For primary score `s_p(q,e)`, secondary score `s_s(q,e)`, and secondary weight `alpha`, define

```text
s_alpha(q,e) = (1 - alpha) s_p(q,e) + alpha s_s(q,e).
```

Subtracting the anchor score at `alpha0` gives the exact identity

```text
s_alpha(q,e) - s_alpha0(q,e)
  = (alpha - alpha0) [s_s(q,e) - s_p(q,e)].
```

The Anchored policy first proposes `alpha0 + beta*tanh(g(phi(q)))` and projects it to `[0,1]`. Because `alpha0` is itself in `[0,1]`, this projection cannot increase its distance from `alpha0`; hence `|alpha-alpha0| <= beta`. Therefore

```text
|s_alpha(q,e) - s_alpha0(q,e)|
  <= beta |s_s(q,e) - s_p(q,e)|.
```

Thus beta is a score-perturbation trust-region radius, not merely a generic hyperparameter. It limits candidate-score movement when the experts disagree, while producing little movement when their scores agree. For two candidates `e1,e2`, the anchor pairwise margin changes by at most `beta * |d(e1)-d(e2)|`, where `d(e)=s_s(q,e)-s_p(q,e)`. Consequently, an anchor top candidate is guaranteed unchanged whenever every anchor margin exceeds that pairwise perturbation bound.

A deterministic numerical check over 100,000 random score configurations passed at tolerance 1e-12; maximum identity error was 8.882e-16; maximum positive score-bound violation and trust-region violation were both 0.000e+00.

## Protocol boundary

The feature model, grouped folds, alpha0, confidence threshold, exact alpha grid, and all model outputs are frozen. Only beta is varied over the predeclared formal grid `0.05, 0.10, ..., 0.50`; `beta=1.0` is labeled no-local-bound diagnostic and is excluded from formal selection. The selected beta remains the existing DEV lock. TEST curves are descriptive only and cannot select beta.

## Full sensitivity table

| Split | Dataset/Pair | beta | Role | Locked? | Delta MRR | Harm % | Mean harm | Benefit % | Changed % | Fallback % | Mean abs(alpha-alpha0) | P95 | Saturation % |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV | MKG-W / M-Hyper + NativE | 0.05 | formal grid | no | +0.000520 | 4.42 | 0.070227 | 6.11 | 35.99 | 0.00 | 0.0180 | 0.0500 | 0.00 |
| DEV | MKG-W / M-Hyper + NativE | 0.10 | formal grid | no | +0.001626 | 11.32 | 0.036686 | 22.47 | 69.05 | 0.00 | 0.0452 | 0.1000 | 0.00 |
| DEV | MKG-W / M-Hyper + NativE | 0.15 | formal grid | no | +0.002515 | 13.89 | 0.030365 | 27.40 | 80.16 | 0.00 | 0.0673 | 0.1500 | 0.00 |
| DEV | MKG-W / M-Hyper + NativE | 0.20 | formal grid | no | +0.003665 | 15.24 | 0.025494 | 29.60 | 85.42 | 0.00 | 0.0896 | 0.2000 | 0.00 |
| DEV | MKG-W / M-Hyper + NativE | 0.25 | formal grid | no | +0.004220 | 16.38 | 0.025019 | 31.06 | 88.57 | 0.00 | 0.1125 | 0.2500 | 0.00 |
| DEV | MKG-W / M-Hyper + NativE | 0.30 | formal grid | no | +0.004610 | 17.33 | 0.026928 | 32.11 | 90.50 | 0.00 | 0.1347 | 0.3000 | 0.00 |
| DEV | MKG-W / M-Hyper + NativE | 0.35 | formal grid | no | +0.005092 | 17.95 | 0.026999 | 32.88 | 91.77 | 0.00 | 0.1572 | 0.3500 | 0.00 |
| DEV | MKG-W / M-Hyper + NativE | 0.40 | formal grid | no | +0.005537 | 18.57 | 0.027061 | 33.35 | 92.66 | 0.00 | 0.1796 | 0.4000 | 0.00 |
| DEV | MKG-W / M-Hyper + NativE | 0.45 | formal grid | no | +0.005807 | 19.25 | 0.027119 | 33.65 | 93.51 | 0.00 | 0.1956 | 0.4000 | 15.96 |
| DEV | MKG-W / M-Hyper + NativE | 0.50 | formal grid | yes | +0.006163 | 19.78 | 0.027442 | 33.86 | 94.08 | 0.00 | 0.2093 | 0.4000 | 19.57 |
| DEV | MKG-W / M-Hyper + NativE | 1.00 | no bound | no | +0.007100 | 23.81 | 0.036800 | 34.94 | 97.10 | 0.00 | 0.3162 | 0.5500 | 32.75 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 0.05 | formal grid | no | +0.000545 | 1.38 | 0.003596 | 8.00 | 10.70 | 24.18 | 0.0054 | 0.0500 | 50.48 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 0.10 | formal grid | no | +0.000786 | 4.62 | 0.009117 | 21.18 | 31.17 | 24.18 | 0.0165 | 0.0500 | 50.48 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 0.15 | formal grid | no | +0.000906 | 5.52 | 0.011482 | 23.79 | 35.68 | 24.18 | 0.0234 | 0.1000 | 50.48 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 0.20 | formal grid | no | +0.001301 | 5.75 | 0.013610 | 24.17 | 35.68 | 24.18 | 0.0304 | 0.1000 | 50.48 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 0.25 | formal grid | no | +0.001436 | 6.01 | 0.017662 | 24.49 | 35.68 | 24.18 | 0.0380 | 0.1500 | 50.48 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 0.30 | formal grid | no | +0.001812 | 6.23 | 0.019472 | 24.75 | 35.68 | 24.18 | 0.0461 | 0.2000 | 50.48 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 0.35 | formal grid | no | +0.001947 | 6.53 | 0.021989 | 24.99 | 35.68 | 24.18 | 0.0540 | 0.2000 | 50.48 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 0.40 | formal grid | no | +0.002244 | 6.73 | 0.022438 | 25.05 | 35.68 | 24.18 | 0.0617 | 0.2500 | 50.48 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 0.45 | formal grid | no | +0.002462 | 6.87 | 0.023137 | 25.09 | 35.68 | 24.18 | 0.0690 | 0.3000 | 50.48 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 0.50 | formal grid | yes | +0.002467 | 7.00 | 0.025368 | 25.11 | 35.68 | 24.18 | 0.0766 | 0.3000 | 50.48 |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 1.00 | no bound | no | +0.004532 | 8.77 | 0.035110 | 24.65 | 35.68 | 24.18 | 0.1536 | 0.6000 | 50.48 |
| DEV | DB15K / M-Hyper + NativE | 0.05 | formal grid | no | +0.000067 | 0.95 | 0.012323 | 2.53 | 4.78 | 0.00 | 0.0024 | 0.0000 | 51.40 |
| DEV | DB15K / M-Hyper + NativE | 0.10 | formal grid | no | +0.000225 | 5.24 | 0.014939 | 13.88 | 29.18 | 0.00 | 0.0146 | 0.0500 | 51.40 |
| DEV | DB15K / M-Hyper + NativE | 0.15 | formal grid | no | +0.000320 | 6.46 | 0.016942 | 16.91 | 36.42 | 0.00 | 0.0206 | 0.0500 | 51.40 |
| DEV | DB15K / M-Hyper + NativE | 0.20 | formal grid | no | +0.000606 | 7.22 | 0.020004 | 18.71 | 39.68 | 0.00 | 0.0283 | 0.1000 | 51.40 |
| DEV | DB15K / M-Hyper + NativE | 0.25 | formal grid | no | +0.000667 | 7.83 | 0.024197 | 19.78 | 41.51 | 0.00 | 0.0354 | 0.1000 | 51.40 |
| DEV | DB15K / M-Hyper + NativE | 0.30 | formal grid | no | +0.000716 | 8.42 | 0.027455 | 20.64 | 42.84 | 0.00 | 0.0426 | 0.1500 | 51.40 |
| DEV | DB15K / M-Hyper + NativE | 0.35 | formal grid | no | +0.000809 | 8.92 | 0.029772 | 21.27 | 43.75 | 0.00 | 0.0498 | 0.1500 | 51.40 |
| DEV | DB15K / M-Hyper + NativE | 0.40 | formal grid | no | +0.000994 | 9.30 | 0.032169 | 21.81 | 44.41 | 0.00 | 0.0571 | 0.2000 | 51.40 |
| DEV | DB15K / M-Hyper + NativE | 0.45 | formal grid | no | +0.001213 | 9.57 | 0.033808 | 22.20 | 44.88 | 0.00 | 0.0642 | 0.2000 | 51.40 |
| DEV | DB15K / M-Hyper + NativE | 0.50 | formal grid | yes | +0.001339 | 9.88 | 0.035643 | 22.53 | 45.31 | 0.00 | 0.0713 | 0.2500 | 51.40 |
| DEV | DB15K / M-Hyper + NativE | 1.00 | no bound | no | +0.002177 | 12.37 | 0.056988 | 24.03 | 47.00 | 0.00 | 0.1429 | 0.5000 | 51.40 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 0.05 | formal grid | no | +0.000038 | 1.39 | 0.016783 | 4.37 | 7.05 | 67.26 | 0.0035 | 0.0500 | 51.59 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 0.10 | formal grid | no | +0.000062 | 1.40 | 0.017106 | 4.37 | 7.05 | 67.26 | 0.0038 | 0.0500 | 51.59 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 0.15 | formal grid | no | +0.000249 | 1.61 | 0.021711 | 4.58 | 7.05 | 67.26 | 0.0071 | 0.1000 | 51.59 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 0.20 | formal grid | no | +0.000332 | 1.66 | 0.024419 | 4.64 | 7.05 | 67.26 | 0.0086 | 0.1000 | 51.59 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 0.25 | formal grid | no | +0.000316 | 1.75 | 0.031782 | 4.70 | 7.05 | 67.26 | 0.0111 | 0.1500 | 51.59 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 0.30 | formal grid | no | +0.000321 | 1.82 | 0.037195 | 4.70 | 7.05 | 67.26 | 0.0133 | 0.1500 | 51.59 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 0.35 | formal grid | no | +0.000392 | 1.86 | 0.038044 | 4.72 | 7.05 | 67.26 | 0.0154 | 0.2000 | 51.59 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 0.40 | formal grid | no | +0.000475 | 1.89 | 0.039881 | 4.73 | 7.05 | 67.26 | 0.0180 | 0.2500 | 51.59 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 0.45 | formal grid | no | +0.000455 | 1.93 | 0.043284 | 4.72 | 7.05 | 67.26 | 0.0199 | 0.2500 | 51.59 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 0.50 | formal grid | yes | +0.000509 | 1.99 | 0.046264 | 4.74 | 7.05 | 67.26 | 0.0225 | 0.3000 | 51.59 |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 1.00 | no bound | no | +0.000586 | 2.53 | 0.064816 | 4.33 | 7.05 | 67.26 | 0.0444 | 0.6000 | 51.59 |
| TEST | MKG-W / M-Hyper + NativE | 0.05 | formal grid | no | +0.000966 | 3.99 | 0.067486 | 5.96 | 35.57 | 0.00 | 0.0178 | 0.0500 | 0.00 |
| TEST | MKG-W / M-Hyper + NativE | 0.10 | formal grid | no | +0.001619 | 11.13 | 0.037768 | 21.40 | 68.48 | 0.00 | 0.0447 | 0.1000 | 0.00 |
| TEST | MKG-W / M-Hyper + NativE | 0.15 | formal grid | no | +0.002847 | 13.87 | 0.031328 | 26.27 | 80.16 | 0.00 | 0.0669 | 0.1500 | 0.00 |
| TEST | MKG-W / M-Hyper + NativE | 0.20 | formal grid | no | +0.002893 | 15.47 | 0.032003 | 28.52 | 85.22 | 0.00 | 0.0889 | 0.2000 | 0.00 |
| TEST | MKG-W / M-Hyper + NativE | 0.25 | formal grid | no | +0.003910 | 16.65 | 0.030443 | 29.89 | 88.19 | 0.00 | 0.1116 | 0.2500 | 0.00 |
| TEST | MKG-W / M-Hyper + NativE | 0.30 | formal grid | no | +0.004484 | 17.61 | 0.030273 | 30.94 | 90.29 | 0.00 | 0.1336 | 0.3000 | 0.00 |
| TEST | MKG-W / M-Hyper + NativE | 0.35 | formal grid | no | +0.005189 | 18.44 | 0.029641 | 31.82 | 91.86 | 0.00 | 0.1563 | 0.3500 | 0.00 |
| TEST | MKG-W / M-Hyper + NativE | 0.40 | formal grid | no | +0.005498 | 19.01 | 0.030468 | 32.44 | 92.90 | 0.00 | 0.1786 | 0.4000 | 0.00 |
| TEST | MKG-W / M-Hyper + NativE | 0.45 | formal grid | no | +0.005676 | 19.48 | 0.031618 | 32.81 | 93.69 | 0.00 | 0.1943 | 0.4000 | 16.03 |
| TEST | MKG-W / M-Hyper + NativE | 0.50 | formal grid | yes | +0.005872 | 19.99 | 0.032861 | 33.20 | 94.26 | 0.00 | 0.2078 | 0.4000 | 19.36 |
| TEST | MKG-W / M-Hyper + NativE | 1.00 | no bound | no | +0.007231 | 23.86 | 0.042776 | 34.21 | 97.04 | 0.00 | 0.3147 | 0.5500 | 32.33 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 0.05 | formal grid | no | +0.000497 | 1.45 | 0.005491 | 7.72 | 10.56 | 24.22 | 0.0053 | 0.0500 | 50.36 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 0.10 | formal grid | no | +0.000714 | 4.87 | 0.009994 | 20.38 | 30.91 | 24.22 | 0.0163 | 0.0500 | 50.36 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 0.15 | formal grid | no | +0.000860 | 5.71 | 0.011097 | 23.28 | 35.63 | 24.22 | 0.0232 | 0.1000 | 50.36 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 0.20 | formal grid | no | +0.001101 | 5.99 | 0.011895 | 23.57 | 35.63 | 24.22 | 0.0301 | 0.1000 | 50.36 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 0.25 | formal grid | no | +0.001281 | 6.25 | 0.014093 | 23.85 | 35.63 | 24.22 | 0.0377 | 0.1500 | 50.36 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 0.30 | formal grid | no | +0.001474 | 6.42 | 0.016703 | 24.27 | 35.63 | 24.22 | 0.0458 | 0.2000 | 50.36 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 0.35 | formal grid | no | +0.001819 | 6.64 | 0.017358 | 24.49 | 35.63 | 24.22 | 0.0536 | 0.2000 | 50.36 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 0.40 | formal grid | no | +0.001865 | 6.91 | 0.019605 | 24.59 | 35.63 | 24.22 | 0.0615 | 0.2500 | 50.36 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 0.45 | formal grid | no | +0.001821 | 7.11 | 0.022985 | 24.60 | 35.63 | 24.22 | 0.0684 | 0.2500 | 50.36 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 0.50 | formal grid | yes | +0.001984 | 7.30 | 0.024299 | 24.56 | 35.63 | 24.22 | 0.0760 | 0.3000 | 50.36 |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 1.00 | no bound | no | +0.003631 | 8.97 | 0.038983 | 24.30 | 35.63 | 24.22 | 0.1524 | 0.6000 | 50.36 |
| TEST | DB15K / M-Hyper + NativE | 0.05 | formal grid | no | +0.000094 | 0.90 | 0.014979 | 2.52 | 4.50 | 0.00 | 0.0023 | 0.0000 | 52.20 |
| TEST | DB15K / M-Hyper + NativE | 0.10 | formal grid | no | +0.000258 | 5.25 | 0.014961 | 13.77 | 28.23 | 0.00 | 0.0142 | 0.0500 | 52.20 |
| TEST | DB15K / M-Hyper + NativE | 0.15 | formal grid | no | +0.000345 | 6.50 | 0.016496 | 16.76 | 35.42 | 0.00 | 0.0200 | 0.0500 | 52.20 |
| TEST | DB15K / M-Hyper + NativE | 0.20 | formal grid | no | +0.000517 | 7.25 | 0.019206 | 18.50 | 38.63 | 0.00 | 0.0274 | 0.1000 | 52.20 |
| TEST | DB15K / M-Hyper + NativE | 0.25 | formal grid | no | +0.000619 | 7.83 | 0.022038 | 19.70 | 40.62 | 0.00 | 0.0344 | 0.1000 | 52.20 |
| TEST | DB15K / M-Hyper + NativE | 0.30 | formal grid | no | +0.000824 | 8.33 | 0.024073 | 20.49 | 41.93 | 0.00 | 0.0414 | 0.1500 | 52.20 |
| TEST | DB15K / M-Hyper + NativE | 0.35 | formal grid | no | +0.000932 | 8.76 | 0.026305 | 21.07 | 42.82 | 0.00 | 0.0483 | 0.1500 | 52.20 |
| TEST | DB15K / M-Hyper + NativE | 0.40 | formal grid | no | +0.001007 | 9.09 | 0.029627 | 21.59 | 43.49 | 0.00 | 0.0554 | 0.2000 | 52.20 |
| TEST | DB15K / M-Hyper + NativE | 0.45 | formal grid | no | +0.001139 | 9.39 | 0.032608 | 22.00 | 44.01 | 0.00 | 0.0623 | 0.2000 | 52.20 |
| TEST | DB15K / M-Hyper + NativE | 0.50 | formal grid | yes | +0.001271 | 9.74 | 0.034421 | 22.34 | 44.42 | 0.00 | 0.0693 | 0.2500 | 52.20 |
| TEST | DB15K / M-Hyper + NativE | 1.00 | no bound | no | +0.002521 | 12.09 | 0.051427 | 23.68 | 46.14 | 0.00 | 0.1386 | 0.5000 | 52.20 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 0.05 | formal grid | no | +0.000142 | 1.51 | 0.015457 | 4.35 | 6.98 | 67.15 | 0.0035 | 0.0500 | 52.11 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 0.10 | formal grid | no | +0.000141 | 1.52 | 0.016457 | 4.35 | 6.98 | 67.15 | 0.0038 | 0.0500 | 52.11 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 0.15 | formal grid | no | +0.000264 | 1.63 | 0.022310 | 4.59 | 6.98 | 67.15 | 0.0071 | 0.1000 | 52.11 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 0.20 | formal grid | no | +0.000330 | 1.69 | 0.024824 | 4.62 | 6.98 | 67.15 | 0.0084 | 0.1000 | 52.11 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 0.25 | formal grid | no | +0.000341 | 1.81 | 0.030569 | 4.64 | 6.98 | 67.15 | 0.0111 | 0.1500 | 52.11 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 0.30 | formal grid | no | +0.000413 | 1.86 | 0.033848 | 4.63 | 6.98 | 67.15 | 0.0132 | 0.1500 | 52.11 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 0.35 | formal grid | no | +0.000489 | 1.90 | 0.034970 | 4.65 | 6.98 | 67.15 | 0.0153 | 0.2000 | 52.11 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 0.40 | formal grid | no | +0.000574 | 1.96 | 0.036167 | 4.65 | 6.98 | 67.15 | 0.0178 | 0.2500 | 52.11 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 0.45 | formal grid | no | +0.000612 | 1.98 | 0.038407 | 4.65 | 6.98 | 67.15 | 0.0197 | 0.2500 | 52.11 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 0.50 | formal grid | yes | +0.000714 | 2.06 | 0.040375 | 4.62 | 6.98 | 67.15 | 0.0223 | 0.3000 | 52.11 |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 1.00 | no bound | no | +0.000971 | 2.53 | 0.055753 | 4.24 | 6.98 | 67.15 | 0.0440 | 0.6000 | 52.11 |

## Stability and risk interpretation

- DEV MKG-W / M-Hyper + NativE: Delta MRR spans +0.000520 to +0.006163 across the 10 formal beta values (10/10 positive). Harm rate is monotone non-decreasing and moves from 4.42% to 19.78%; mean alpha deviation is monotone non-decreasing from 0.0180 to 0.2093.
- DEV MKG-W / M-Hyper + AdaMF-MAT: Delta MRR spans +0.000545 to +0.002467 across the 10 formal beta values (10/10 positive). Harm rate is monotone non-decreasing and moves from 1.38% to 7.00%; mean alpha deviation is monotone non-decreasing from 0.0054 to 0.0766.
- DEV DB15K / M-Hyper + NativE: Delta MRR spans +0.000067 to +0.001339 across the 10 formal beta values (10/10 positive). Harm rate is monotone non-decreasing and moves from 0.95% to 9.88%; mean alpha deviation is monotone non-decreasing from 0.0024 to 0.0713.
- DEV DB15K / M-Hyper + AdaMF-MAT: Delta MRR spans +0.000038 to +0.000509 across the 10 formal beta values (10/10 positive). Harm rate is monotone non-decreasing and moves from 1.39% to 1.99%; mean alpha deviation is monotone non-decreasing from 0.0035 to 0.0225.
- TEST MKG-W / M-Hyper + NativE: Delta MRR spans +0.000966 to +0.005872 across the 10 formal beta values (10/10 positive). Harm rate is monotone non-decreasing and moves from 3.99% to 19.99%; mean alpha deviation is monotone non-decreasing from 0.0178 to 0.2078.
- TEST MKG-W / M-Hyper + AdaMF-MAT: Delta MRR spans +0.000497 to +0.001984 across the 10 formal beta values (10/10 positive). Harm rate is monotone non-decreasing and moves from 1.45% to 7.30%; mean alpha deviation is monotone non-decreasing from 0.0053 to 0.0760.
- TEST DB15K / M-Hyper + NativE: Delta MRR spans +0.000094 to +0.001271 across the 10 formal beta values (10/10 positive). Harm rate is monotone non-decreasing and moves from 0.90% to 9.74%; mean alpha deviation is monotone non-decreasing from 0.0023 to 0.0693.
- TEST DB15K / M-Hyper + AdaMF-MAT: Delta MRR spans +0.000141 to +0.000714 across the 10 formal beta values (10/10 positive). Harm rate is monotone non-decreasing and moves from 1.51% to 2.06%; mean alpha deviation is monotone non-decreasing from 0.0035 to 0.0223.

A wide sign-stable region means net MRR remains above Global over many beta values; it does not mean risk is unchanged. Harm frequency, conditional harm magnitude, and score-weight displacement must be read alongside Delta MRR. Grid rounding can create short plateaus or local non-monotonicities even though the continuous trust-region radius increases monotonically.

## Figures

- `outputs/paper_a_safe_correction/beta_sensitivity/beta_vs_delta_mrr.pdf`
- `outputs/paper_a_safe_correction/beta_sensitivity/beta_vs_harm_rate.pdf`
- `outputs/paper_a_safe_correction/beta_sensitivity/beta_vs_mean_alpha_deviation.pdf`

Each figure contains the four main pairs, grouped OOF DEV and locked TEST, the DEV-locked beta marker, and the separated `beta=1.0` no-local-bound endpoint.

## Source and lock audit

| Split | Dataset/Pair | alpha0 | Locked beta | Threshold | Observations | Source SHA-256 | Lock SHA-256 |
|---|---|---:|---:|---:|---:|---|---|
| DEV | MKG-W / M-Hyper + NativE | 0.60 | 0.50 | 0.00 | 25656 | `e0a98d388819d535c15585165bb4fea18c47a7bb70e31ed0d21902c97dbf7d50` | `eee533f2d0b8b1a9a181d348c1df94b3db522b5f7c4a5a398a946b58d201c713` |
| DEV | MKG-W / M-Hyper + AdaMF-MAT | 1.00 | 0.50 | 0.10 | 25656 | `fbaa2a0faea02aa6191deff8110c6d10e78c1ac1092a4bd83b4926b06160a6b4` | `95109445388ea1ffeab17c11b5281880ca3928c827155d2f1e9486dbab93bb0f` |
| DEV | DB15K / M-Hyper + NativE | 1.00 | 0.50 | 0.00 | 47532 | `f49663bce80db958d4fbb3d576ed841dc5926387a9d67959533db484d8249dc3` | `a3ddf0b07697df57140c8a1dacdeb38ee96598272805b9db2a44bd59fa2b57ef` |
| DEV | DB15K / M-Hyper + AdaMF-MAT | 1.00 | 0.50 | 0.30 | 47532 | `7b13db8cd4094ec4ce1651fa3a95a1d80f1c9e859d0d26381fe507ada372e066` | `b7ec3d6b524d16cf9c2128a3b84c8881cbe5e6b8ee7c2e5d592c58f756dac55d` |
| TEST | MKG-W / M-Hyper + NativE | 0.60 | 0.50 | 0.00 | 25644 | `469d2d427b61d3048de03ef485d804e2821c40a8af9a8e5e62d61e2ff4bbd624` | `eee533f2d0b8b1a9a181d348c1df94b3db522b5f7c4a5a398a946b58d201c713` |
| TEST | MKG-W / M-Hyper + AdaMF-MAT | 1.00 | 0.50 | 0.10 | 25644 | `0414041b7cd33267e79f0d7b6c12575f581f7e7ddaa22724c0a041d0cc087e60` | `95109445388ea1ffeab17c11b5281880ca3928c827155d2f1e9486dbab93bb0f` |
| TEST | DB15K / M-Hyper + NativE | 1.00 | 0.50 | 0.00 | 59412 | `2155f299165204d388ad45a509684148a3db3d3fa0dae9370896d2ed42271cfc` | `a3ddf0b07697df57140c8a1dacdeb38ee96598272805b9db2a44bd59fa2b57ef` |
| TEST | DB15K / M-Hyper + AdaMF-MAT | 1.00 | 0.50 | 0.30 | 59412 | `77bb9f968099600ff110f110dea8cec197f5891396a055d35bd82542dce06e0c` | `b7ec3d6b524d16cf9c2128a3b84c8881cbe5e6b8ee7c2e5d592c58f756dac55d` |

No base model or combiner was trained. Stored confidence and the original policy were reconstructed and checked before sensitivity evaluation. On locked TEST, the `beta=0.50` and `beta=1.00` rows exactly reproduce the existing Full and No Bound core-ablation outputs across 8 matched pair/configuration rows.
