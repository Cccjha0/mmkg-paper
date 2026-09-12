# E03/E04: study scope and additional-pair boundary review

Recorded 2026-09-13 before running the new stratified analysis. This is a
retrospective, exploratory audit of information_boundary_v2, not a new holdout
or a prospective applicability test. All six pairs and both additional outcomes
remain visible. No scorer/selector fitting, policy search, TEST-based selection,
or replacement of earlier versioned artifacts is permitted.

## E03: local DEV contrasts without changing the policy

Keep each original five-fold partition and its fitted selector/settings. For
each holdout fold, pool the three seeds in the other four folds to estimate
mean endpoint RR(A)-RR(B) for each relation x prediction direction. Count unique
original training triples, not seed repetitions. Require at least 50 training
triples for an eligible cell; unsupported cells remain a separate reported bin.
Classify eligible cells using a fixed +/-0.005 MRR descriptive band (half an
MRR percentage point): secondary-favored below -0.005, close within the closed
band, primary-favored above +0.005. These are reporting thresholds chosen for
interpretability, not significance or optimized deployment thresholds.

Apply that map using only held-out relation and direction. Held-out endpoint
outcomes, gold IDs, ADC outcomes and TEST data do not enter the map application.
Report all four bins for all six pairs, their held-out counts, endpoint gap,
ADC-minus-fold-Global utility, changed rate and gain/loss decomposition. Preserve
training-cell ledgers and seed/direction results. This is the original
transductive grouped-OOF distribution (query keys may overlap folds), not an
unseen-query or new-pair test. Conditional intervals resample original triples
with all six seed/direction observations, 10,000 PCG64 draws; sorted cluster
order and dataset seeds match C12 (DEV 2026091214/15). Recompute each bin's ratio
denominator in each draw but hold the fold maps and fits fixed. Empty bins have
undefined intervals; record the number of valid draws. No multiplicity adjustment
or model-development uncertainty is supplied.

All six globally eligible pairs cannot establish necessity of the eligibility
condition. Local reversals are not globally ineligible pairs. Slightly negative
observed ADC gains also preclude treating the rule as a guarantee of improvement.

## E04: opportunity, preference discrimination, and action accounting

Replay original full and no-gate weights from saved margins and original
DEV-fold/final-DEV settings. Report final anchor/beta/tau and all five fold
settings. Separate four disjoint stages: gate-rejected nonzero raw proposal,
gate-rejected raw no-op, accepted raw no-op, and actually changed. Raw means
same radius/clip/grid with tau=0. Gate rejection must not be confused with all
unchanged weights. Invalid features must be counted separately (existing audit
certifies finite values for these saved observations).

For both DEV-OOF and TEST report all six pairs, with the two additional pairs
displayed together in the paper: endpoint B winner/tie rates, endpoint-oracle
minus best-single MRR, grid-oracle minus Global (answer-aware diagnostics only),
non-tied endpoint-label AUROC of the saved preference margin, and full/no-gate
utility. AUROC is a descriptive endpoint discrimination measure, not calibration,
harm prediction, or evidence that the signal can improve an interior anchor.
Give full benefit/unchanged/harm counts, unconditional gain/loss and their net,
actual changed count, plus previously unified full-minus-Global and direct
full-minus-no-gate clustered intervals. Preserve every seed/direction cell.

Positive oracle gaps refute an absence-of-opportunity explanation but do not
establish exploitable complementarity. Fixed-map gate comparisons measure the
effect of rejecting these proposals, not a causal explanation of training signal
quality or a proof that this gate is optimal. Compare against the existing
equal-intervention gate evidence and disclose its uncertainty. Historic 98.49%
fallback belongs to an older information contract and must not be presented as
the corrected run's metric.

Validation: source SHA-256 and row identities, exact original action replay,
stage exhaustiveness/disjointness, unchanged-row RR invariance, BG-HL identity,
prior point/interval reconciliation, no held-out outcomes in stratum assignment,
and meaningful synthetic tests for boundaries, unsupported cells and clustered
conditional denominators.
