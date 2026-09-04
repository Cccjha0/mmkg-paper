# AACPI V2 Phase 2D Advantage Identifiability Protocol

Status: frozen before Phase 2D systematic diagnosis
Effective date: 2026-09-04

## 1. Scientific status and scope

AACPI V2 Phase 2B is formally **NO-GO** under the frozen H1 gate. Three of six dataset/expert pairs passed all four H1 signals, below the required four, and MKG-W NativE + AdaMF-MAT failed although both NativE + AdaMF-MAT pairs were required to pass.

Phase 2D is an AACPI V2 failure diagnosis. It does not change, repair, or continue AACPI V2. Its sole question is why query-action advantage is identifiable from the frozen score geometry for some dataset/expert pairs but nearly unidentifiable for others. Any later AACPI V3 or new-representation hypothesis is a separate study that requires a new protocol before implementation.

## 2. Frozen inputs

Phase 2D may read only existing DEV artifacts:

- the six Phase 1 query-action utility tables and their source manifests;
- the frozen 13 answer-agnostic score-geometry fields;
- the six Phase 2B outer-fold OOF prediction tables, run audits, metrics, inner-search records, and outer-fold selections;
- existing exact per-alpha DEV ranking and endpoint-RR exports already represented in the Phase 2A winner/action diagnostics;
- original-triple, query, seed, direction, and relation metadata already stored in those DEV artifacts;
- historical Anchored Dynamic v1/P3 DEV cross-fit summaries, cited only as retrospective descriptive evidence.

The Phase 1 action grid, reference alpha, exact ranks, reciprocal ranks, and advantage targets remain immutable. The Phase 2B predictions must be the recorded outer-fold OOF predictions. In-sample predictions are inadmissible.

MKG-W and DB15K TEST remain retrospective/secondary evidence and are prohibited as Phase 2D inputs. No TEST rows may be read and no TEST evaluation command may be run.

## 3. Prohibited work

Phase 2D must not:

- add or modify a feature;
- expand or otherwise change the MLP, loss, search space, action grid, or reference alpha;
- retrain the Phase 2B predictor or train another model;
- introduce calibration fitting or thresholds;
- implement or evaluate Advantage-Greedy, kappa, lambda, tau, an LCB, a conservative policy, or any other new policy;
- rerun Phase 2B in response to a diagnostic result;
- access TEST;
- retroactively revise the Phase 2B H1 decision.

## 4. Common evaluation rules

All prediction diagnostics use non-reference action rows unless a table explicitly states otherwise. Advantage signs use the existing numerical tolerance of `1e-15`. Positive and harmful average-precision lifts are measured against the corresponding observed prevalence. Harmful-action ranking uses negative predicted advantage as its score.

Prediction calibration uses the fixed percentile buckets `0-10%`, `10-30%`, `30-50%`, `50-70%`, `70-90%`, and `90-100%`, ordered by OOF predicted advantage. Each bucket reports predicted mean, actual mean and median, sign prevalence, and count. No calibration transform is fitted.

Action diagnostics use the available signed `delta_alpha` values from the frozen local grid. Missing boundary actions are reported as unavailable and are never synthesized.

Feature diagnostics cover exactly the 13 fields in `router.constants.QUERY_GEOMETRY_FIELDS`. For each pair and feature they report sign-group means and medians, signed standardized mean differences for positive-versus-rest, harmful-versus-rest, and positive-versus-harmful rows; raw and orientation-free single-feature AUROC for each sign versus the remaining rows; positive-versus-harmful AUROC after excluding plateau rows; and Spearman correlation with continuous advantage. The positive-versus-harmful statistic distinguishes useful correction-sign information from a feature that merely separates nonzero utility from the `U=0` plateau. These are threshold-free descriptive statistics, not trained predictors.

## 5. Subgroup support and denominator rules

Direction and seed diagnostics use every non-reference DEV action row in their subgroup.

Relation diagnostics reuse the existing project minimum support standard of **60 distinct seed-direction DEV query instances**. Repeated action rows do not increase support. The table must also report distinct original triples, and interpretations must acknowledge that seeds and directions are repeated observations of those triples. Relations below the threshold are omitted from the primary relation table and counted in the audit.

Winner/action analysis uses the Phase 2A deterministic best action: maximum actual reciprocal rank, then minimum absolute alpha deviation, then smaller alpha. It reports separate denominators for all queries, endpoint-nontie queries, beneficial non-anchor queries, and their intersection. Endpoint ties, anchor-optimal queries, predicted-deviation/true-anchor cases, and opposite-direction cases are reported explicitly.

## 6. Frozen failure taxonomy rules

The taxonomy is descriptive and a pair may receive more than one label:

- **F1 Representation failure:** the pair fails H1; the largest orientation-free single-feature AUROC lift above 0.5 for both positive and harmful labels is below `0.05`, and the largest absolute feature/advantage Spearman is below `0.05`.
- **F2 Action heterogeneity / aliasing:** the pair fails H1; at least one available signed action passes the four H1-style signals while at least one other action fails, or the action-wise Spearman range is at least `0.10`.
- **F3 Regime heterogeneity:** the pair fails H1; head and tail disagree on H1-style pass, at least one seed passes while another fails, or supported relations include both positive and negative AP lifts with a lift range of at least `0.05`.
- **F4 Ranking without calibration:** pooled Spearman is greater than `0.05` but the highest predicted-advantage decile has non-positive actual mean advantage.
- **F5 Harm asymmetry:** positive AP lift is at least `0.02`, while harmful AP lift is non-positive or trails positive AP lift by at least `0.05`.
- **F6 Winner/action mismatch:** among endpoint-nontie beneficial-deviation queries, the best local direction is opposite the endpoint winner for at least `20%`, or at least `40%` of endpoint-nontie queries are truly anchor-optimal.

An H1-style subgroup pass requires positive Spearman, positive and harmful AP lift above prevalence, and positive top-decile actual mean advantage. These subgroup labels are diagnostic only.

## 7. Outcome decision

Phase 2D ends with one research-direction outcome and does not start the proposed next study:

- **Outcome A, representation bottleneck:** F1 is supported and stable subgroup structure does not account for the failure.
- **Outcome B, structured heterogeneity bottleneck:** action, direction, seed, or supported-relation regimes show materially different and internally useful signal, supporting F2 or F3.
- **Outcome C, advantage target weakly identifiable:** feature and subgroup diagnostics both lack stable signal, suggesting that available Oracle complementarity is largely latent under the frozen answer-agnostic score information.

When evidence supports more than one mechanism, the report must state the primary outcome and the secondary mechanisms with their numerical basis. It must not select a new feature set or model.

## 8. Reproducibility and integrity

One script must rebuild every Phase 2D table and figure. Its source manifest must record SHA-256 hashes for every Phase 1 utility table, Phase 2B OOF table and audit, Phase 2A winner/action table, frozen protocol/search-space file, and historical DEV Anchored summary it reads.

The final audit must confirm zero TEST rows, zero TEST evaluation commands, no Phase 1 utility modification, no Phase 2B retraining, no feature/action/reference-alpha change, outer-fold OOF provenance, original-triple fold isolation, and exact agreement with the frozen Phase 2B metrics.
