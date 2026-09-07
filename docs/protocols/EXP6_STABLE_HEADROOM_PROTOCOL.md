# Experiment 6 — Stable Headroom & Gain Concentration Audit

Status: frozen before the first systematic run on 2026-09-07.

This is the final DEV closure experiment. It decomposes the Experiment 1 Oracle headroom into all-seed consensus, two-of-three stable, three-of-three stable, held-out-seed LOSO, and frozen X4 inference-time observable gains. None of the first four diagnostics is a deployable policy.

## Canonical evidence

For each of the six frozen dataset/expert pairs, the audit follows the Experiment 1 source manifest to the canonical `full_ranking/dev_query_rows.csv`. It requires DEV, query-zscore normalization, seeds 1/2/3, both directions, the unchanged 21-action alpha grid, and the unchanged Experiment 1 Global alpha0. Source hashes are verified before any statistic is calculated.

The stable identity is `original_triple_id × direction`. Every identity must contain all three seeds, and every original triple must contain both directions. Raw Oracle Headroom must reproduce Experiment 1 within absolute tolerance `1e-12`.

## Stable upper diagnostics

Raw Oracle independently maximizes exact filtered RR for each seed. Consensus maximizes mean RR over all three seeds. The two-of-three and three-of-three sets exclude alpha0 and require strictly positive utility in at least two seeds or all three seeds, respectively. Empty sets fall back to alpha0. Eligible actions maximize all-seed mean RR with ties resolved by distance to alpha0 and then smaller alpha.

Recoveries divide pair-level headrooms; query-level ratios are never averaged and ratios are not clipped. Fragility is one minus the corresponding pair-level recovery. Stable opportunity rates distinguish action availability from action value.

## Gain concentration

The concentration unit is `original_triple_id`. Triple gain is the mean seed-specific Oracle gain over all three seeds and both directions. Its pair mean must reproduce Raw Oracle Headroom. Top-x support uses `ceil(xN)` triples after descending sort. Q50 is the smallest descending prefix reaching half of total gain. The curve is named `Cumulative Oracle-Gain Concentration`, not a Lorenz curve.

The supplementary Gini uses non-negative gains sorted ascending:

`sum_i (2*i-N-1)*x_(i) / (N*sum_i x_i)`, for `i=1..N`.

Normalized Effective Support is `((sum G)^2 / sum G^2) / N`.

## Uncertainty and closure

All Part-A intervals use 10,000 paired percentile bootstrap replicates with seed 20260907, resampling original triples while retaining both directions, all seeds, and all actions. Ratio intervals are computed inside each paired replicate and are not clipped.

Experiment 4 and 5 classifications are read from their frozen JSON outputs. `strong E4_INTERMEDIATE` is preregistered as: at least four LOSO CIs with lower bound above zero, both datasets represented, and median LOSO Recovery at least 10%. Joint interpretation follows the precedence in the machine-readable contract, with unmatched evidence defaulting to `FINAL_MIXED_LIMITS`.

The generated closure decision memo freezes the final RQs, claims, figures, TEST metrics, TEST commands, and forbidden post-TEST changes. Its generation does not unlock or access TEST. TEST remains prohibited until the memo itself is committed.
