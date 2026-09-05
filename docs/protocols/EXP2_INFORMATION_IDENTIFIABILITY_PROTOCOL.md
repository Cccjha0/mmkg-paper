# Experiment 2 — Information–Identifiability Audit Protocol

**Status:** frozen before the first systematic Experiment 2 run
**Date:** 2026-09-05
**Eligible:** Experiment 1 Available Complementarity Gate = GO (6/6 headroom, 6/6 prevalence)

## Scope

This experiment estimates **Empirical Identifiable Headroom** with strict nested OOF probes. It does not estimate theoretical `C_identifiable`, develop a final policy, resurrect AACPI, or access TEST.

The six frozen dataset-pairs and the 21-point alpha grid are unchanged from Experiment 1. Exact RR curves come only from the manifest-linked `full_ranking/dev_query_rows.csv` files. Phase 3A/4A assets contribute answer-agnostic inference-time features only.

## Information ladder

The exact machine-readable contract is `EXP2_INFORMATION_FEATURE_CONTRACT.json`. All levels include `alpha`, fold-relative `delta_alpha`, and `abs_delta_alpha`.

- **X1 Basic Score Geometry:** prediction direction and per-expert top-1 score, top-5 mean, top-1/top-2 margin, and score standard deviation.
- **X2 Cross-Expert Disagreement:** X1 plus the four frozen cross-expert differences and the ten frozen R1 top-k/ranking-disagreement fields.
- **X3 TRAIN-only Structural Context:** X2 plus the four R3 TRAIN frequency/diversity fields and seven C1 TRAIN structural fields.
- **X4 TRAIN-only Modality Context:** X3 plus the four frozen known-side/target-role R3 modality fields and two C2 TRAIN relation-support fields.
- **X5 Frozen Latent Query Context:** X4 plus separately projected expert latents. Each inner/outer projection is fit only on its current training original-triple groups, using the already frozen 16+16 PCA and difference/absolute-difference construction.
- **X6 Full Candidate Evidence:** X5 plus the union of expert A/B top-100 candidates, capped at 200 candidates. Per candidate: normalized expert scores, normalized full ranks, top-100 membership flags, signed/absolute score disagreement, and signed/absolute rank disagreement. Candidate identity and candidate embeddings are excluded. The permutation-invariant set encoder receives the X5 query context, this candidate set, and the three action descriptors.

Phase 3A R2 Global-to-action response fields are deliberately excluded: their old local grid and full-DEV Global anchor do not satisfy this experiment's fold-specific 21-action protocol.

## Learner ladder

- `linear_huber`: standardized SGD linear regressor with Huber loss.
- `hist_gbdt`: histogram gradient boosting regressor.
- `mlp_low`: Phase-2-style two-hidden-layer low-capacity MLP.
- `mlp_high`: controlled three-hidden-layer higher-capacity MLP.
- `set_encoder`: X6-only permutation-invariant candidate encoder with mean/max pooling.

The frozen hyperparameter candidates are in the machine-readable contract. Each learner family chooses its configuration within inner grouped CV. For X1–X5, the primary `nested_selected` probe also chooses among the four learner families using only inner-CV results. X6 has the one preregistered set-encoder family.

## Strict nested OOF

Original triples are assigned by the existing relation-stratified SHA256 grouped splitter.

For each of five outer folds:

1. Select `alpha0_fold` from the 21 actions using outer-train RR only. Ties prefer the alpha nearest 0.5, then the smaller alpha, matching the repository's frozen Global-selection rule.
2. Define every train and held-out target as `U_q(alpha)=RR_q(alpha)-RR_q(alpha0_fold)`.
3. Run three-fold grouped inner CV entirely within outer-train.
4. Select hyperparameters, and for the primary X1–X5 probe also learner family, by inner validation probe gain; ties use Spearman and then contract order.
5. Fit on all outer-train groups and predict every held-out query/action exactly once.

The fold Global action is always included. Its predicted advantage is overwritten with exact zero before action selection. Other predicted actions do not receive thresholds, fallback rules, uncertainty penalties, or post-hoc calibration. Ties prefer Global, then the nearest action, then smaller alpha.

## Metrics

For every valid dataset/pair/information/learner combination and the nested-selected X1–X5 probe, report:

- OOF MRR and delta against fold-specific OOF Global;
- original-triple clustered percentile-bootstrap 95% CI;
- headroom recovery against Experiment 1 Available Headroom;
- negative-transfer, positive-gain, and changed rates;
- non-Global action-row Spearman, positive AP lift, harmful AP lift, and positive-versus-harmful AUROC;
- outer-train probe gain, OOF gain, and train–OOF gap;
- seed and prediction-direction slices.

`Negative Transfer Rate` and `Positive Gain Rate` use all held-out queries. `Changed Rate` is the proportion whose chosen alpha differs from its fold Global. Sign metrics exclude the forced Global rows; positive-versus-harmful AUROC additionally excludes zero-utility rows.

## Preliminary query-level gate

A representation passes only if its primary nested-selected probe satisfies all of:

1. robust positive OOF gain in at least 4/6 pairs (`delta_mrr > 0` and clustered CI lower > 0);
2. positive OOF gain for both NativE + AdaMF-MAT pairs;
3. at least 3/6 pairs recover at least 10% of Experiment 1 Available Headroom;
4. MKG-W / NativE + AdaMF-MAT clustered CI lower > 0.

Failure is **Query-Level Preliminary NO-GO**, but Experiment 3 remains eligible. Passing is **Preliminary GO**, but query-selector development remains prohibited until Experiment 3's granularity comparison.

## Operational audit

- TEST access: 0
- checkpoint training/reselection: 0
- final-policy development: 0
- feature-family additions after results: prohibited
- candidate embeddings: excluded before the first systematic X6 run
