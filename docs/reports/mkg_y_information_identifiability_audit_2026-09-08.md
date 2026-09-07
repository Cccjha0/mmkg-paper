# MKG-Y Y-E2 Frozen X4 Information–Identifiability Audit

Date: 2026-09-08

## Outcome

**Y_E2_X4_REPLICATION_REPORTED** (descriptive replication; not a progression gate).

This is the frozen finite-model strict-OOF identifiability probe. It is reported as Empirical Identifiable Headroom, not theoretical identifiability and not a final deployable policy.

## Frozen protocol

- DEV only; exact filtered RR on alpha `0.00:0.05:1.00`; query-zscore score normalization.
- Five outer folds and three inner folds, grouped by original triple.
- Each held-out fold uses an alpha0 selected only from its outer-training groups.
- X4 is unchanged from Experiment 2: score geometry, cross-expert disagreement, TRAIN-only structural context, and TRAIN-only modality context.
- Four frozen learners/config grids are compared inside each outer fold; learner selection never sees that fold's held-out queries.
- X6 was not run because the pre-specified conditional justification was absent.

## Primary fold-wise nested-selected X4 probe

| Pair | OOF Global MRR | OOF MRR | OOF gain | 95% clustered CI | Available recovery | Positive gain | Negative transfer | Changed | Train gain | Train–OOF gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-Y / M-Hyper + NativE | 0.339323 | 0.343028 | +0.003705 | [+0.002253, +0.005167] | 10.1% | 13.6% | 17.7% | 56.1% | +0.014596 | +0.010891 |
| MKG-Y / M-Hyper + AdaMF-MAT | 0.336070 | 0.342843 | +0.006773 | [+0.004831, +0.008698] | 15.4% | 16.0% | 14.4% | 53.2% | +0.015844 | +0.009071 |
| MKG-Y / NativE + AdaMF-MAT | 0.327654 | 0.327213 | -0.000441 | [-0.001511, +0.000610] | -1.2% | 21.4% | 18.7% | 57.3% | +0.004464 | +0.004905 |

## Utility-identifiability diagnostics

| Pair | Spearman(pred U, U) | Positive AP lift | Harmful AP lift | Positive-vs-harmful AUROC |
| --- | ---: | ---: | ---: | ---: |
| MKG-Y / M-Hyper + NativE | +0.0493 | -0.0317 | +0.0155 | 0.4964 |
| MKG-Y / M-Hyper + AdaMF-MAT | +0.1218 | +0.0154 | +0.0204 | 0.5789 |
| MKG-Y / NativE + AdaMF-MAT | +0.0190 | +0.0310 | -0.0477 | 0.5158 |

## Descriptive replication summary

- Robust-positive pairs (OOF gain > 0 and clustered CI lower > 0): 2/3.
- Pairs recovering at least 10% of Y-E1 available headroom: 2/3.
- No GO/NO-GO gate is applied at Y-E2, and no selector or representation was developed.

## Figures

1. [figure_y_e2_x4_learner_comparison](../../outputs/complementarity_identifiability/mkg_y_y_e2_information/figure_y_e2_x4_learner_comparison.svg)
2. [figure_y_e2_available_vs_identifiable](../../outputs/complementarity_identifiability/mkg_y_y_e2_information/figure_y_e2_available_vs_identifiable.svg)

## Operational audit

- TEST access = 0
- outer-triple leakage = 0
- full-DEV alpha0 used for held-out fold = 0
- checkpoint retraining/reselection = 0
- new selector/representation = 0
- X6 run = 0
- final policy development = 0

All direct source and output hashes are recorded in `audit_manifest.json`.
