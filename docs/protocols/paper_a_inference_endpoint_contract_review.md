# B08/B10: guarded inference and endpoint conventions

Protocol date: 2026-09-12. This is a retrospective code and numerical-contract audit. It does not introduce a new selector fit, choose a policy using TEST, or replace historical results.

## B08 execution contract

The supported guarded application entry point is `scripts/apply_anchored_safe.py`, using `router/anchored_inference.py`. It consumes the original trusted DEV model and lock. Only the 13 feature columns enter the selector; gold IDs, RR columns and filters do not enter the feature function or prediction. Output identifiers are copied solely for joining.

1. Validate the input schema, fitted median/scaler/logistic state and locked parameters. Malformed schemas or corrupt models remain explicit errors.
2. Initialize each proposal to the exact grid-aligned anchor. Reject rows containing NaN or either infinity before any preprocessing call.
3. Transform only remaining rows with the fitted imputer and scaler. Reject any row whose transformed coordinates are non-finite before classifier invocation.
4. Compute logits only for surviving rows. Reject non-finite logits before probability prediction. Reject non-finite or out-of-range returned probabilities before action computation.
5. On valid rows, use the original confidence gate, clipped tanh map and projection rule. Invalid rows stay at the anchor even when the confidence threshold is zero. Query-soft uses the same invalid-row guard.

Neutral diagnostic placeholders (logit 0, probability 0.5) on rejected rows are not predictions; the exported `predicted`, `invalid` and `invalid_reason` fields distinguish them. Raw score anomalies already hidden by the upstream geometry sanitizer are outside this guard's detection boundary.

The hash-bound historical experiment scripts are retained for reproduction and still have the earlier sanitize/predict/fallback order. They are not the guarded application entry point. Explicit conversion of remaining NaN/infinity coordinates to NaN enabled historical median imputation, but did not prevent later scaling overflow. A synthetic `[NaN, 1e308]` row demonstrates the old failure. No median-imputation guarantee for arbitrary infinities is claimed.

Local replay must verify all six original full-DEV locks on DEV and TEST: decisions/probabilities/continuous ADC weights within absolute 1e-12, grid ADC and Query-soft weights and fallback flags exactly, and cached RR within absolute 1e-12. New rejected rows require investigation before any result replacement. This replay does not regenerate raw candidate scores or fit selectors.

## Anchor and ranking ties

Global maximizes `(computed mean RR, -abs(alpha - 0.5), -alpha)` on the recorded grid. MRR ties are exact computed floating-point ties, without a tolerance: prefer proximity to 0.5, then the lower alpha. Apply the same rule to full-DEV and each OOF training portion. Policy ties prefer smaller beta, then larger threshold. Projection minimizes `(abs(alpha - proposal), abs(alpha - anchor), alpha)`.

Shared filtered ranking is `1 + count(candidate_score > separately_scored_gold)`, retaining the current gold and masking other known answers from the canonical TRAIN+DEV+TEST union. Equal scores do not increase rank. Head and tail tests exercise the actual shared evaluator as well as its dense and sparse filter helpers.

At an exact action endpoint, the deployed evaluator returns the corresponding raw standalone rank, ignoring the inactive expert. At an interior action it ranks the standardized mixture. This is an explicit operational convention. In exact arithmetic, finite active scores and the same finite positive affine normalization preserve strict comparisons and ties, including constant rows with epsilon denominator. Floating-point arithmetic can merge distinct scores: float32 `[0, 1, 1e8]` with gold at the first entry has raw rank 3 and normalized rank 2. Active non-finite scores can also invalidate equivalence. An inactive non-finite expert must not contaminate an endpoint through `0 * infinity`.

Cached endpoint columns have already been dispatched to standalone ranks. Their equality cannot prove equality with an unoverridden normalized endpoint. The B01 absence of non-finite raw scores likewise does not exclude floating-point tie changes.

## Fixed server audit; no local base-model scoring

Run `python scripts/audit_paper_a_endpoint_contract.py --plan-only` to inspect scope without loading models. Full execution is CUDA-only:

```powershell
git pull --ff-only
python scripts/audit_paper_a_endpoint_contract.py --device cuda
```

The audit uses all 18 previously audited checkpoints, both canonical DEV/TEST splits and both directions: 72 cells, 474,732 rows. It verifies checkpoint/config/data hashes, uses each expert's recorded batch and chunk sizes, fixes TRAIN+DEV+TEST filtering, and writes a source-bound immutable plan before scoring. It records the runtime and triple ordering. The experiment compares:

1. Raw export rank versus the shared evaluator's gold-preserving filter/count path.
2. Deployed alpha=0 and alpha=1 ranks versus the shared raw standalone rank, also with a deliberately invalid inactive expert.
3. Unoverridden normalized rank versus raw standalone rank, reporting every difference, per-cell counts, raw/normalized RR sums, examples and raw-score exceptional rows.

Completed cells can be resumed only under the same plan, checkpoint, coverage and difference-file hashes. Differences are retained for review; they do not trigger a change in anchor, weight, grid or TEST selection. This is a current reexecution audit, not proof that raw score bytes equal historical exports under arbitrary batching or runtimes.

Return `outputs/paper_a_safe_correction/endpoint_contract_audit_v1_return.zip`. Operational cells and the zip stay outside Git; a concise independently checked review will be committed after return. B10 remains pending that empirical comparison. If differences exist, quantify their effects and assess whether the endpoint convention requires reevaluation; do not silently substitute normalized endpoints or select a favorable TEST convention.
