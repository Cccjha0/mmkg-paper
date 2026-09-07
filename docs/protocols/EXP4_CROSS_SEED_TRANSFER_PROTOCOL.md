# Experiment 4 — Cross-Seed Transferable Complementarity Audit

Status: frozen before the first systematic run on 2026-09-07.

This DEV-only audit tests whether a query's ex-post best mixture action transfers across three independently trained seeds. It does not train, select, or propose an inference-time policy.

## Canonical evidence

For each of the six frozen dataset/expert pairs, the audit reads the Experiment 1 source manifest and follows its `source_query_rows.path` to canonical `full_ranking/dev_query_rows.csv`. The declared source hash must match. Paths are not inferred from dataset or model names.

Every `original_triple_id × direction` identity must contain seeds `{1,2,3}`. Every original triple must contain both `head` and `tail`, every RR action column for alpha `{0.00,0.05,...,1.00}` must exist and be finite, and the source must be DEV with `query_zscore`. Any violation fails the audit without row deletion.

## Frozen estimands

For query identity `q` and seed `s`, `alpha_star(q,s)` maximizes exact filtered RR. Ties are resolved by distance to the Experiment 1 Global alpha0, then by the smaller alpha. Raw Oracle Headroom is the mean seed-specific Oracle advantage.

For all six ordered source-target seed pairs, the source Oracle action is applied unchanged to the target seed. Negative utilities and zero-source-gain rows are retained. Transfer Recovery is the unbounded ratio of mean transfer gain to Raw Oracle Headroom.

Beneficial, zero, and harmful transfer rates use only source observations with strictly positive Oracle gain and a non-anchor Oracle action. Their denominator includes both target seeds.

LOSO chooses an action from mean RR over the two non-held-out seeds, using the same tie rule, then applies it once to the held-out seed. LOSO is a stability diagnostic, not a deployable or inference-time policy.

Exact-alpha and LEFT/ANCHOR/RIGHT agreement are evaluated across all three seeds per query identity. Non-anchor direction agreement excludes identities for which all seeds select ANCHOR.

## Uncertainty and frozen classification

All 95% intervals use 10,000 percentile bootstrap replicates with seed `20260907`, resampling original triples and preserving their seeds, directions, actions, and transfers. Recovery intervals use paired bootstrap ratios and are not clipped.

The frozen classification gates and prohibited operations are machine-readable in `EXP4_CROSS_SEED_TRANSFER_CONTRACT.json`. The report must end with exactly one of `E4_SUBSTANTIALLY_TRANSFERABLE`, `E4_RESIDUAL_DOMINATED`, or `E4_INTERMEDIATE`. Experiment 5 must not start automatically.
