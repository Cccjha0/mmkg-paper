# Paper A boundary / stress-pair protocol

## Scope

This protocol retains two pre-specified pairs irrespective of outcome:

- MKG-W / NativE + AdaMF-MAT;
- DB15K / NativE + AdaMF-MAT.

The experiment characterizes the applicability boundary of Anchored score correction. It does not optimize the pairs and cannot support a claim that arbitrary model pairs universally benefit.

## Reliable-primary prerequisite

Before TEST analysis, `scripts/audit_reliable_primary_regime.py` applies the frozen `paper_a_reliable_primary_protocol` rule using standalone exact filtered DEV rows only. The selected primary and regime are retained even if they contradict the initial boundary-case hypothesis. Pair orientation then follows that DEV-selected primary.

## Frozen inputs and methods

Existing exact full-ranking rows are reused:

- DEV: `outputs/<dataset>/anchored_dynamic/native_adamf_seed123/full_ranking/dev_query_rows.csv`;
- TEST: `outputs/complementarity_identifiability/closure_test/raw/<pair>/test_query_rows.csv`.

Frozen NativE and AdaMF-MAT checkpoints are used only where DynaSemble must reconstruct candidate-score statistics. Base MMKGC models are never retrained.

The reported methods are identical to the main-pair definitions:

1. strongest standalone expert selected by the DEV-only rule;
2. fixed Query-zscore 0.5;
3. DEV-selected Global alpha;
4. Query-soft logistic;
5. faithful released-code DynaSemble;
6. Anchored Dynamic.

Anchored Dynamic uses the same feature schema, beta grid, confidence-threshold grid, grouped DEV protocol, deterministic tie-breaking, and exact alpha grid as the main pairs. DynaSemble uses the released architecture and hyperparameters recorded in `paper_a_dynasemble_four_pair_protocol`; only expert identity changes: NativE is the learned-weight `expert_a`, and AdaMF-MAT has fixed weight 1.

## Information boundary

All fitting and hyperparameter selection occur on DEV. For every paired seed, DynaSemble trains a separate selector. Anchored alpha0, beta, confidence threshold, model, and DynaSemble selector hashes are locked before TEST. TEST only applies these immutable locks and reports results. No TEST result may change the boundary-pair methods or the main method.

Evaluation is exact filtered full ranking. Query-level harmful correction is defined relative to Global alpha as `rr_method - rr_global < 0`; Mean Harm is `mean(rr_global - rr_method | harmful)`.

## Output and execution

```powershell
powershell -File scripts/run_paper_a_boundary_pairs.ps1 -Stage dev -Python python -Device cuda
powershell -File scripts/run_paper_a_boundary_pairs.ps1 -Stage test -Python python -Device cuda
powershell -File scripts/run_paper_a_boundary_pairs.ps1 -Stage analyze -Python python
```

The split execution makes the DEV lock boundary explicit. `-Stage all` is available for a single controlled run. Outputs are written below `outputs/paper_a_safe_correction/boundary_pairs/`, and the final report is `docs/reports/paper_a_boundary_pair_report.md`.
