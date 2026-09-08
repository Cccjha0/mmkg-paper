# MKG-Y Y-E5 — X4 Local Identifiability Replication

Date: 2026-09-08
Split: DEV only
Frozen prior route: `ROUTE_C_LIMITS`
Experiment 4 status: frozen

## Representation availability

- X4: available as the exact 40-dimensional frozen Experiment 2 query representation.
- X6: `X6_FIXED_VECTOR_UNAVAILABLE`. Experiment 2 freezes a variable-size candidate set and set encoder, not a canonical target-independent fixed vector. No new encoder, embedding, PCA, or pooling representation was created.

## Frozen descriptive outcome

Recorded outcome: `Y_E5_LOCAL_IDENTIFIABILITY_REPLICATION_REPORTED`.
Final report outcome: **Y_E5_LOCAL_IDENTIFIABILITY_REPLICATION_REPORTED**.

## Primary k=10 results

| Pair | Direction agreement lift (95% CI) | kNN consensus utility (95% CI) | Utility lift vs matched (95% CI) | Matched coverage | Local-signal pair |
|---|---:|---:|---:|---:|---:|
| MKG-Y / M-Hyper + NativE | +0.07579 [+0.07089, +0.08073] | -0.00284 [-0.00473, -0.00098] | +0.00102 [-0.00044, +0.00245] | 98.0% | no (0/4 k) |
| MKG-Y / M-Hyper + AdaMF-MAT | +0.06966 [+0.06460, +0.07476] | +0.00038 [-0.00159, +0.00234] | +0.00377 [+0.00214, +0.00540] | 98.0% | yes (2/4 k) |
| MKG-Y / NativE + AdaMF-MAT | +0.05980 [+0.05545, +0.06404] | -0.00086 [-0.00264, +0.00094] | +0.00254 [+0.00120, +0.00390] | 98.0% | no (0/4 k) |

A `LOCAL_SIGNAL_PAIR` requires at least two frozen k values to have positive lower confidence bounds for direction-agreement lift, center consensus utility, and consensus-utility lift. Raw purity without matched-baseline advantage does not pass this gate.

## Diagnostic evidence

- LOCAL_SIGNAL_PAIR: 1/3.
- Supported datasets: mkg_y.
- NativE+AdaMF local-signal pairs: 0/1.
- X4 k=10 consensus CI includes zero: 2/3 pairs.
- X6 recovery gate: unavailable and therefore false; it is never imputed from set-encoder inputs.

## Protocol and diagnostics

The exact Experiment 2 outer-fold assignment is reused. Every held-out center searches only outer-training rows, and the audit asserts zero overlap in original-triple IDs. X4 scaling is fit separately in each outer-training pool. Matched random pools use exact seed × direction × relation matching with no fallback and 100 frozen without-replacement draws per fold stratum.

Beneficial-set Jaccard is missing when both sets are empty; `both_empty_rate` is reported separately. Neighbor-consensus actions use only outer-training neighbor utility curves. Center RR enters only after action selection, so center labels never choose the action.

## Outputs and figures

- `per_center_knn_metrics.csv.gz`: center-level X4 metrics at k={5,10,20,50}.
- `pair_k_summary.csv`, `matched_baseline_summary.csv`, `distance_purity_bins.csv`, and `bootstrap_ci.csv`.
- `neighbors/*.npz`: canonical query-ID table, top-50 row indices, distances, neighbor Oracle actions/directions, and beneficial-set masks.
- Figure 5.1: distance–purity curves with matched reference and explicit X6-unavailable panel.
- Figure 5.2: neighbor-consensus utility versus k.
- Figure 5.3: k=10 direction-purity lift forest.
- Figure 5.4: purity lift versus actual consensus utility.

## Integrity audit

- TEST access = 0
- checkpoint execution/retraining/reselection = 0
- outer-triple leakage = 0
- new representation / PCA / embedding = 0
- supervised metric learning = 0
- new selector or neighbor-method development = 0
- k tuning = 0
- distance-metric selection after results = 0
- original-triple bootstrap intact = yes (10000 percentile replicates, seed 20260908)
- all direct source/output hashes recorded = yes (`audit_manifest.json`; self-hash excluded)
- subsequent experiment started = 0

Y_E5_LOCAL_IDENTIFIABILITY_REPLICATION_REPORTED
