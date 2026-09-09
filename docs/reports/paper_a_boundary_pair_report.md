# Paper A boundary / stress-pair report

## Framing and prerequisite audit

These two pairs were retained as pre-specified stress cases regardless of outcome. The frozen DEV-only reliable-primary rule selected NativE as primary for both pairs.

| Dataset | Selected primary | DEV primary delta | 3/3 seeds? | DEV CI95 | Regime |
| --- | --- | ---: | --- | --- | --- |
| MKG-W | NativE | +0.038696 | Yes | [0.034434, 0.043062] | reliable-primary |
| DB15K | NativE | +0.029004 | Yes | [0.026140, 0.031850] | reliable-primary |

## Immutable TEST results

| Dataset | Method | MRR | Delta vs Global [CI95] | Harm % | Mean Harm | Fallback % | Changed-alpha % | Mean abs(alpha-alpha0) | P95 abs(alpha-alpha0) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W | Strongest standalone | 0.338037 | -0.006329 [-0.007643, -0.005041] | 29.04 | 0.044708 | -- | 100.00 | 0.0500 | 0.0500 |
| MKG-W | Query-zscore 0.5 | 0.327106 | -0.017259 [-0.019973, -0.014591] | 40.18 | 0.105963 | -- | 100.00 | 0.4500 | 0.4500 |
| MKG-W | Global alpha | 0.344365 | +0.000000 [0.000000, 0.000000] | 0.00 | -- | -- | 0.00 | 0.0000 | 0.0000 |
| MKG-W | Query-soft | 0.327290 | -0.017075 [-0.019756, -0.014394] | 40.04 | 0.106345 | 0.00 | 100.00 | 0.4550 | 0.5500 |
| MKG-W | DynaSemble | 0.305904 | -0.038461 [-0.042293, -0.034692] | 48.48 | 0.146042 | 0.00 | 100.00 | 0.8747 | 0.9500 |
| MKG-W | Anchored Dynamic | 0.344442 | +0.000077 [-0.000249, 0.000420] | 0.39 | 0.132563 | 98.49 | 1.51 | 0.0042 | 0.0000 |
| DB15K | Strongest standalone | 0.312761 | +0.000000 [0.000000, 0.000000] | 0.00 | -- | -- | 0.00 | 0.0000 | 0.0000 |
| DB15K | Query-zscore 0.5 | 0.300244 | -0.012517 [-0.014387, -0.010674] | 32.03 | 0.142098 | -- | 100.00 | 0.5000 | 0.5000 |
| DB15K | Global alpha | 0.312761 | +0.000000 [0.000000, 0.000000] | 0.00 | -- | -- | 0.00 | 0.0000 | 0.0000 |
| DB15K | Query-soft | 0.302371 | -0.010390 [-0.012138, -0.008647] | 31.61 | 0.135120 | 0.00 | 100.00 | 0.4935 | 0.6000 |
| DB15K | DynaSemble | 0.282588 | -0.030173 [-0.032669, -0.027687] | 42.27 | 0.177385 | 0.00 | 100.00 | 1.0000 | 1.0000 |
| DB15K | Anchored Dynamic | 0.313287 | +0.000526 [0.000326, 0.000729] | 0.40 | 0.077580 | 88.09 | 2.25 | 0.0056 | 0.0000 |

## Stability

| Dataset | Method | Seed 1 delta | Seed 2 delta | Seed 3 delta | Head delta | Tail delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MKG-W | Query-soft | -0.018677 | -0.016241 | -0.016308 | -0.016743 | -0.017407 |
| MKG-W | DynaSemble | -0.055014 | -0.013164 | -0.047205 | -0.037276 | -0.039646 |
| MKG-W | Anchored Dynamic | -0.000067 | +0.000115 | +0.000185 | +0.000095 | +0.000059 |
| DB15K | Query-soft | -0.012163 | -0.013153 | -0.005855 | -0.011134 | -0.009647 |
| DB15K | DynaSemble | -0.029825 | -0.040348 | -0.020346 | -0.027843 | -0.032503 |
| DB15K | Anchored Dynamic | +0.000654 | +0.000380 | +0.000543 | +0.000350 | +0.000701 |

## Boundary-case interpretation

Anchored Dynamic is positive against Global on 2/2 stress pairs, with a positive clustered-CI lower bound on 1/2; it is positive on 5/6 pair-seeds, and 4/4 pair-directions. These observations characterize correction behavior; they do not license a claim that arbitrary model pairs universally benefit.

- MKG-W: Anchored changes alpha on only 1.51% of observations and falls back on 98.49%. Its Delta MRR is +0.000077 with CI [-0.000249, 0.000420]. The harmful-query rate is 0.39%, but conditional Mean Harm is 0.132563; safety comes from refusing most corrections, not from making every accepted correction harmless. Query-soft and DynaSemble have Delta MRR -0.017075 and -0.038461; DynaSemble zero-weight collapse occurs in seed(s) 1, 3.
- DB15K: Anchored changes alpha on only 2.25% of observations and falls back on 88.09%. Its Delta MRR is +0.000526 with CI [0.000326, 0.000729]. The harmful-query rate is 0.40%, but conditional Mean Harm is 0.077580; safety comes from refusing most corrections, not from making every accepted correction harmless. Query-soft and DynaSemble have Delta MRR -0.010390 and -0.030173; DynaSemble zero-weight collapse occurs in seed(s) 1, 2, 3.

The proposed explanatory condition—absence of a reliable primary—does not occur in either evaluated pair. Consequently these results cannot establish that missing primary reliability is associated with poorer or less stable correction. If a method fails here, the evidence instead points to a combination-level limitation despite a reliable standalone primary; if it succeeds, that still does not support universal pairwise benefit. A genuine non-reliable-primary pair is required to test the stated association directly.

## Information boundary

Base MMKGC models were not retrained. Global alpha, Anchored beta/threshold, Query-soft, and DynaSemble selectors were selected or fitted on DEV only. TEST was applied immutably and used only for the displayed final analysis. No boundary-pair outcome changes the main method or its hyperparameters.
