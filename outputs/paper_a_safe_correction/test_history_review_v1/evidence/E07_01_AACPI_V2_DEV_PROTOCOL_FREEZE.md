# AACPI V2 DEV Protocol Freeze

**Status:** frozen before AACPI model development
**Effective date:** 2026-09-04
**Scope:** AACPI V2 phase 1 and all later AACPI method development

## Purpose and evidence boundary

Advantage-Aware Conservative Policy Improvement (AACPI) treats the DEV-locked Global ensemble as a reference policy and learns whether a local action improves the true query-level ranking outcome relative to that reference. AACPI development is DEV-only. The existing MKG-W and DB15K TEST results have already been inspected in the preceding research program, so every future AACPI result on those TEST splits is retrospective, secondary evidence and is not eligible for AACPI method selection or confirmatory claims.

Confirmatory evidence must come from a new dataset whose TEST split remains unaccessed until the complete AACPI method, feature contract, action set, loss, hyperparameter search spaces, and decision rule have been frozen using DEV only.

## A. Frozen legacy experiment assets

The following assets and rules are frozen as of this protocol:

- the completed official M-Hyper, NativE, AdaMF-MAT, and other formal expert checkpoints used by the current experiments;
- the existing checkpoint-selection rules and selected seeds/checkpoints;
- the current MKG-W and DB15K full-ranking score and exact-ranking exports;
- the exact filtered-ranking protocol, including the current all-split true-fact filtering and rank tie semantics;
- the bidirectional head- and tail-prediction evaluation protocol;
- the paper-locked query-wise score-normalization rule;
- the existing Global alpha, Query-soft, Anchored Dynamic v1, Oracle, and related relation-alpha results, including their existing TEST outputs.

No expert may be retrained or replaced for AACPI, and no seed or checkpoint may be reselected because it gives a more favorable AACPI result. Historical result values and TEST outputs remain unchanged.

Existing TEST outcomes must not influence AACPI's method structure, feature set, action grid, loss, hyperparameter search space, decision rule, fallback behavior, or reporting thresholds. They may only be reported later as retrospective or secondary evaluation after an explicit AACPI method lock.

## B. Frozen AACPI formulation

### B.1 Reference policy

The reference policy is the shared Global weight `alpha0`. It is selected using DEV only and the already established Global-alpha selection protocol. It is then fixed for the dataset/expert pair.

### B.2 Advantage target

For query instance `q` and candidate action `alpha`, the supervised target is

```text
U_q(alpha) = RR_q(alpha) - RR_q(alpha0)
```

This is the true query-level filtered-ranking advantage relative to continuing with the Global reference policy. AACPI does not use an Expert-A-versus-Expert-B winner label as its primary target.

### B.3 Score normalization

AACPI reuses the current paper-locked query-wise normalization, `query_zscore`, exactly as implemented by the existing full-ranking ensemble evaluator. The rule, epsilon, filtering behavior, and endpoint handling are not changed for AACPI.

### B.4 Global-alpha search rule

AACPI reuses the current Global grid `0.00:0.05:1.00` and the existing exact-ranking selection rule. The rule maximizes pooled DEV MRR over seeds and both prediction directions. Existing deterministic ties are resolved by preferring the alpha closest to `0.5`, then the smaller alpha. AACPI does not alter this grid or tie rule.

### B.5 Local action rule

For each dataset/expert pair, the action table is formed from

```text
alpha0
alpha0 +/- {0.05, 0.10, 0.20, 0.30}
```

Each action is clipped to `[0, 1]`, then duplicate values are removed and the result is sorted. `alpha0` is always present as the reference action. The grid is fixed independently of observed utility outcomes.

### B.6 Grouping rule

The group key is the original triple:

```text
h=<head_id>|r=<relation_id>|t=<tail_id>
```

All seed 1/2/3 observations, both head and tail prediction directions, and every candidate-alpha action for the same original triple must remain in one group and one fold. Randomly splitting query-by-action rows across folds is prohibited.

The phase-1 utility-table builder records this key but does not assign folds or split rows. Later grouped cross-fit code must split unique original-triple keys first and propagate each assignment to all associated rows.

### B.7 Feature policy

The first AACPI implementation reuses the existing 13 answer-agnostic score-geometry features implemented by `router/query_geometry.py` and declared in `router/constants.py`. Phase 1 does not add handcrafted features. Target scores, correct-target ranks/RR, TEST-derived priors, and any other answer-aware information are forbidden as model inputs.

### B.8 TEST rule

AACPI construction, learnability analysis, estimator selection, and policy selection use grouped DEV cross-fit only. MKG-W and DB15K TEST are retrospective/secondary evidence and are not eligible for AACPI method selection. AACPI TEST evaluation must not be invoked before a later, explicit method lock.

The phase-1 utility-table builder enforces this boundary by accepting only source rows whose split is exactly `dev` and by producing DEV-only supervision.

## C. DEV-tunable items with controlled search spaces

The following values are not fixed to a particular optimum by this phase-1 protocol. They may be selected later using grouped DEV cross-fit:

- MLP hidden width;
- learning rate;
- negative-advantage weighting;
- Huber/Smooth-L1 parameters;
- `kappa`;
- `lambda`;
- `tau`;
- target clipping, only if DEV evidence shows it is necessary.

Before the first systematic AACPI model experiment, a fixed search space must be documented for every item that will be tuned. Search ranges must not be expanded after any TEST outcome is observed. Adding another tunable component also requires a DEV-only protocol amendment made before systematic experimentation.

## Phase-1 deliverables and exclusions

Phase 1 may freeze this protocol, record TEST exposure, and construct auditable DEV query-by-action utility tables. The utility summaries describe the observed action surface and do not select an AACPI policy.

Phase 1 excludes the AACPI MLP, model training, `kappa`/`lambda`/`tau` selection, the final lower-confidence-bound policy, AACPI TEST evaluation, expert retraining, checkpoint reselection, action-grid adaptation, new handcrafted features, and any modification of historical Global, Query-soft, Anchored Dynamic, Oracle, or relation-alpha results.

## Implementation anchors

- Exact full-ranking and alpha-grid export: `scripts/eval_heterogeneous_complementarity.py`
- Existing Global-alpha rule and original-triple key: `scripts/crossfit_heterogeneous_dev_policies.py`
- Answer-agnostic geometry implementation and field contract: `router/query_geometry.py`, `router/constants.py`
- AACPI DEV utility construction: `scripts/build_aacpi_utility_table.py`
- TEST exposure record: `docs/protocols/aacpi_test_exposure_manifest.csv`
