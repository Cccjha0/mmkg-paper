# Experiment 5 — Local Identifiability / Action Ambiguity Audit

Status: frozen before the first systematic run on 2026-09-07.

This DEV-only diagnostic asks whether nearby queries in an already frozen inference-time observable space have consistent Oracle mixture preferences and whether neighbor consensus transfers positive utility to held-out centers. It is not a selector or deployable neighbor policy.

## Representation availability

X4 is the exact 40-dimensional representation frozen in Experiment 2. Its feature order is read from the audited Experiment 2 asset manifest and must equal the original X4 contract. Within each reused outer fold, mean and standard deviation are fit only on outer-training rows and applied to outer-held-out centers. Distance is Euclidean.

Experiment 2 X6 contains a variable-size union top-100 candidate set and a trained set encoder. It does not expose a frozen deterministic target-independent fixed-dimensional vector. Experiment 5 therefore freezes `X6_FIXED_VECTOR_UNAVAILABLE`. No encoder, pooling rule, PCA, or embedding may be created for this audit.

## Leakage-safe neighbors

The exact five-fold `outer_fold` vector is loaded from Experiment 2 X4 nested-selected OOF predictions and aligned by `query_id`. For fold `f`, centers have fold `f` and the reference pool has all other folds. The audit fails if any center original triple appears in the reference pool.

The exact k values are 5, 10, 20, and 50. Exact Euclidean top-50 neighbors are calculated once and prefixes define smaller k. Equal distances are resolved by smaller canonical reference-row index.

## Matched random baseline

The matched pool is restricted to the current outer-training rows with exactly the center's seed, direction, and relation. There is no fallback. A center is unsupported for k when the pool contains fewer than k rows.

For computational reproducibility, 100 without-replacement draws are generated once per outer-fold matched stratum with seed 20260907 and evaluated for every center in that stratum. This sharing does not use center labels and every supported center is still evaluated on all 100 frozen draws.

## Labels and utility

Oracle action ties use maximum RR, nearest alpha0, then smaller alpha. Direction is LEFT, ANCHOR, or RIGHT. A beneficial set contains exactly alphas with utility strictly above zero. Jaccard is missing when both sets are empty; that event is reported separately.

Neighbor consensus selects the alpha with maximum mean neighbor utility, then nearest alpha0, then smaller alpha. Selection uses reference labels only; center RR is used only after selection for evaluation.

## Uncertainty and classification

All intervals use 10,000 percentile bootstrap replicates with seed 20260905 and center `original_triple_id` as the cluster. The local-signal and Experiment 5 gates are evaluated exactly as encoded in `EXP5_LOCAL_IDENTIFIABILITY_CONTRACT.json`. The report ends with the frozen classification selected by the registered precedence rule.

No TEST data, checkpoint execution, new representation, metric learning, selector training, or subsequent experiment is permitted.
