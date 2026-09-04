# AACPI Phase 3A Action-Response Representation Protocol

**Status:** frozen before the first systematic Phase 3A comparison
**Effective date:** 2026-09-04
**Evidence scope:** DEV-only representation diagnosis

## Research status and hypothesis

AACPI V2 remains a frozen **NO-GO** after Phase 2B. Phase 2C was not run, and no V2 definition, feature, action, alpha, result, or gate may be changed retrospectively. Phase 3A is a new representation hypothesis: static query summaries may omit information needed to distinguish beneficial from harmful action-induced rank movement, while an answer-agnostic description of the candidate landscape response from `alpha0` to `alpha` may recover that information.

Phase 3A evaluates identifiability only. It does not implement or evaluate Advantage-Greedy, a final AACPI policy, uncertainty, `kappa`, `lambda`, `tau`, LCB, a deviation penalty, a fallback rule, or TEST inference.

## Evidence and information boundary

- Every source query and utility row must have split exactly `dev`.
- Candidate landscapes may be reconstructed only from the already selected frozen expert checkpoints.
- The reconstruction process may read canonical TRAIN and DEV data, mappings, and modality tensors. It must not open or hash TEST rows.
- Candidate-landscape features use the full, unfiltered candidate set. This prevents the evaluation-only operation that retains the correct target while filtering other known answers from leaking target identity into a feature.
- Exact filtered DEV rank and RR remain supervision only through the frozen Phase 1 utility table. They are never inputs to a Phase 3A representation.
- Candidate score normalization reuses `router.score_combination.normalize_candidate_scores(..., "query_zscore")`, including its fixed epsilon. The candidate domain is unfiltered so normalization remains answer-agnostic.
- All features must be available before knowing the correct answer. Correct-target identity, score, rank, RR, availability, and any statistic derived from them are forbidden.
- MKG-W and DB15K TEST remain retrospective/secondary and ineligible for method selection.

## Frozen representation comparison

The four families R0, R1, R2, and R3 are defined in `AACPI_PHASE3A_FEATURE_CONTRACT.md` and are frozen together before the systematic run. No feature may be added, removed, or redefined after any Phase 3A OOF result is viewed. R0 is the unchanged V2 Base13 control. R1 adds static cross-expert ranking geometry. R2 adds action-response geometry to R0. R3 adds a frozen, lightweight TRAIN-only context to R2.

Fixed `k` values are `{5, 10, 20}`. Entering and leaving fractions are omitted because equal-size top-k sets make both fractions identical and a deterministic one-to-one transform of Jaccard. A separate concentration-change field is omitted because query-zscore mixture means are zero, making the proposed top-five concentration change identical to top-five mean-score change.

## Frozen estimator and split contract

Phase 3A reuses the Phase 2B two-hidden-layer MLP, hidden widths `{32,64,128}`, learning rates `{1e-4,3e-4,1e-3}`, negative-advantage weights `{1.0,1.5,2.0,3.0}`, Smooth-L1 loss and beta, optimizer settings, fixed epochs, inner selection rule, model seeds, and five-outer/three-inner nested CV. Only the first-layer input dimension changes with the representation.

The outer and inner split key remains the original triple `h=<head>|r=<relation>|t=<tail>`. All seeds, both directions, all actions, and all representations for a triple receive the same fold. Each reported prediction must be produced by a model for which its original-triple group was absent from training and hyperparameter selection.

## Frozen metrics

All primary metrics use non-reference action rows and true outer-fold OOF predictions:

- Spearman correlation between predicted and actual advantage;
- positive-action AUPRC lift over positive prevalence;
- harmful-action AUPRC lift over harmful prevalence, using `-predicted U` as the harmful score;
- positive-versus-harmful AUROC on nonzero-utility rows;
- highest predicted-advantage decile actual mean U, `P(U>0)`, and `P(U<0)`;
- action-, direction-, and seed-conditioned versions of the H1 metrics.

Zero-versus-nonzero activity discrimination is reported separately. It cannot substitute for positive-versus-harmful discrimination.

## Preregistered gates

### H1-style pair pass

A pair passes when all four Phase 2B criteria are strictly positive: non-reference Spearman, positive AUPRC lift, harmful AUPRC lift, and highest-decile actual mean U.

### Primary Gate

A representation is Phase 3A eligible only if at least four of six pairs pass, both NativE + AdaMF-MAT pairs pass, and MKG-W / NativE + AdaMF-MAT has positive top-decile actual U plus positive positive-action and harmful-action AUPRC lifts. A relation-specific model cannot be used.

### Representation Recovery Gate

The three prior FAIL pairs are MKG-W / NativE + AdaMF-MAT, DB15K / M-Hyper + NativE, and DB15K / M-Hyper + AdaMF-MAT. Relative to paired R0 OOF predictions, a prior FAIL pair has **clear sign-identifiability improvement** only when all of the following hold:

1. positive-versus-harmful AUROC increases by at least `0.02`;
2. both positive and harmful AUPRC lift increase, and the smaller increase is at least `0.01`;
3. highest-decile actual mean U increases by at least `0.001` and is positive;
4. non-reference Spearman does not decrease.

At least two of the three prior FAIL pairs must meet this definition, and MKG-W / NativE + AdaMF-MAT must be one of them.

The prior PASS set is stable only if at least two of its three pairs still satisfy the H1-style pass and none has both a nonpositive top-decile actual U and a nonpositive positive-versus-harmful AUROC lift (`AUROC - 0.5`).

### Decision rule

Phase 3A is GO only when one frozen candidate among R1, R2, and R3 satisfies both the Primary Gate and Representation Recovery Gate without violating prior-PASS stability. Otherwise Phase 3A is NO-GO. If none of R1/R2/R3 recovers MKG-W / NativE + AdaMF-MAT, its top-confidence actual U remains nonpositive, or harmful-action AUPRC lift remains nonpositive, score-only handcrafted representation expansion stops.

## Reproducibility and immutability

The action-response builder records hashes for every checkpoint, run configuration, TRAIN/DEV split, mapping, feature tensor, Phase 1 utility table, full-ranking summary, feature contract, and implementation file. It rejects non-DEV inputs and refuses overwrite unless explicitly requested. It does not alter expert checkpoints, Phase 1 utilities, Base13, action grids, or alpha0.

Any Phase 3A success remains development evidence. Confirmatory claims require a dataset whose TEST remains untouched until the representation and later policy are fully locked.
