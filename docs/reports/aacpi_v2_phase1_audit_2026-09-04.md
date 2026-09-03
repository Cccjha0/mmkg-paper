# AACPI V2 Phase-1 Audit

**Date:** 2026-09-04
**Branch:** `m1/recent-mmkgc-baselines`
**Scope:** protocol freeze, TEST exposure boundary, and DEV utility-table construction only

## 1. Changed files

- `docs/protocols/AACPI_V2_DEV_PROTOCOL_FREEZE.md`: freezes the legacy assets, AACPI formulation, grouping/action/TEST rules, and the boundary around later DEV-tunable parameters.
- `docs/protocols/aacpi_test_exposure_manifest.csv`: records pair-level and dataset-level TEST exposure for MKG-W and DB15K.
- `scripts/build_aacpi_utility_table.py`: builds DEV query-by-local-action advantage supervision from existing exact-ranking alpha-grid exports.
- `router/constants.py`: holds the existing 13 score-geometry field names so offline table construction does not require importing PyTorch.
- `router/query_geometry.py`: imports and re-exports the same field tuple from `router/constants.py`; feature computation is unchanged.
- `ml/training/tests/test_aacpi_utility_table.py`: tests action clipping/deduplication, reference rows, original-triple propagation, output counts, and TEST rejection.
- `outputs/aacpi/utility_tables/*_dev_utility_table.csv.gz`: six compressed machine-readable DEV utility tables.
- `outputs/aacpi/utility_tables/*_dev_utility_summary.json`: descriptive utility-surface statistics and validation results.
- `outputs/aacpi/utility_tables/*_dev_source_manifest.json`: source paths/hashes, output hash, frozen configuration, feature contract, group key, and evidence role.

No historical result value, TEST output, expert checkpoint, or checkpoint selection was modified.

## 2. Frozen protocol

The freeze covers the formal M-Hyper, NativE, and AdaMF-MAT checkpoints; checkpoint and seed selection; MKG-W/DB15K exact full-ranking assets; all-split filtered-ranking and tie semantics; bidirectional head/tail evaluation; query-wise z-score normalization; the Global grid and exact-ranking selection rule; and all historical Global, Query-soft, Anchored Dynamic v1, Oracle, relation-alpha, and TEST outputs.

AACPI is frozen around the DEV-selected Global `alpha0`, the target `U_q(alpha) = RR_q(alpha) - RR_q(alpha0)`, and the clipped/deduplicated local action set `alpha0 +/- {0.05, 0.10, 0.20, 0.30}` including `alpha0`. The group key is the original triple `h=<head>|r=<relation>|t=<tail>` and must bind all seeds, both directions, and all actions to one future fold. Phase 1 reuses the existing 13 answer-agnostic geometry features.

MLP hidden width, learning rate, negative-advantage weighting, Huber/Smooth-L1 parameters, `kappa`, `lambda`, `tau`, and any necessary target clipping remain selectable using future grouped DEV cross-fit. Their fixed search spaces must be recorded before the first systematic experiment and may not be expanded after TEST access.

## 3. TEST exposure boundary

Pair-specific TEST full-ranking and Anchored Dynamic outputs exist for these four pairs:

- MKG-W / M-Hyper + NativE;
- MKG-W / M-Hyper + AdaMF-MAT;
- DB15K / M-Hyper + NativE;
- DB15K / M-Hyper + AdaMF-MAT.

They include the prior Global/relation/Oracle surface and Query-soft/Anchored Dynamic v1 evidence. They are retrospective, secondary, and not eligible for AACPI method selection.

No pair-specific TEST output was found for NativE + AdaMF-MAT on either dataset. Nevertheless, both dataset TEST splits and outcome landscapes were previously accessed, so these potential evaluations are also retrospective/secondary rather than untouched confirmatory evidence. The builder rejects every non-DEV source row; no TEST command was run in phase 1.

## 4. Utility-table implementation

The builder consumes `full_ranking/dev_query_rows.csv` and its DEV `selection.json`. Those rows were produced by `scripts/eval_heterogeneous_complementarity.py`, which already applies all-split exact filtering, head/tail evaluation, query-wise z-score normalization, protected expert endpoints, and strict-greater competition ranks before exporting exact RR at every alpha in `0.00:0.05:1.00`.

The builder reuses `alpha_column`, `best_alpha`, and `triple_key` from `scripts/crossfit_heterogeneous_dev_policies.py`. It does not recompute candidate scores or duplicate the filtered-ranking algorithm. Each local action is a member of the existing exact alpha grid, so `rr_action` is copied from the evaluator's exact stored column. Integer ranks are recovered from `1 / RR` only for output and reciprocal-consistency validation.

For every pair, the script recomputes the locked Global alpha and MRR using the existing selection function, checks the source `rr_global` against the alpha0 column, and samples 64 query instances across every local action for source-column/rank consistency. It verifies the expert endpoint rank when a local grid includes alpha 0 or 1.

Every output row carries `original_triple_id`. The builder checks that each group has the complete Cartesian product of the locked seeds and both directions, writes all actions with that same key, and performs no row or fold split. A later grouped cross-fit must split these unique keys before expanding to rows.

## 5. Generated DEV utility statistics

Rates below include the required reference action. “Non-ref zero” excludes the reference row. “Distinct RR” is the mean fraction of action-grid points having a distinct RR for a query; lower values indicate a more stepped/plateaued utility surface. Local headroom is the descriptive mean per-query best local-action RR minus Global MRR and is not a deployable policy result.

| Dataset | Pair | alpha0 | Triples | Query instances | Query-action rows | Positive-opportunity queries | Positive / zero / negative actions | Non-ref zero | Distinct RR | Global MRR | Local-oracle MRR | Local headroom |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| MKG-W | M-Hyper + NativE | 0.60 | 4,276 | 25,656 | 230,904 | 60.80% | 24.56% / 48.46% / 26.98% | 42.02% | 55.88% | 0.356356 | 0.383908 | 0.027552 |
| MKG-W | M-Hyper + AdaMF-MAT | 1.00 | 4,276 | 25,656 | 128,280 | 47.99% | 33.69% / 53.42% / 12.89% | 41.77% | 61.85% | 0.352822 | 0.363573 | 0.010751 |
| MKG-W | NativE + AdaMF-MAT | 0.95 | 4,276 | 25,656 | 153,936 | 58.15% | 24.20% / 49.73% / 26.06% | 39.68% | 60.80% | 0.334855 | 0.364476 | 0.029620 |
| DB15K | M-Hyper + NativE | 1.00 | 7,922 | 47,532 | 237,660 | 39.60% | 26.42% / 58.35% / 15.23% | 47.94% | 56.66% | 0.382587 | 0.396998 | 0.014411 |
| DB15K | M-Hyper + AdaMF-MAT | 1.00 | 7,922 | 47,532 | 237,660 | 40.81% | 27.75% / 55.37% / 16.89% | 44.21% | 58.59% | 0.382587 | 0.397264 | 0.014677 |
| DB15K | NativE + AdaMF-MAT | 1.00 | 7,922 | 47,532 | 237,660 | 46.44% | 32.09% / 51.07% / 16.85% | 38.83% | 62.27% | 0.321022 | 0.346136 | 0.025114 |

The deterministic descriptive best-action tie break is maximum RR, then minimum absolute delta, then smaller alpha. Best-delta distributions are:

- MKG-W / M-Hyper + NativE: `-0.30` 34.91%, `-0.20` 5.46%, `-0.10` 2.65%, `-0.05` 3.25%, `0.00` 39.20%, `+0.05` 2.15%, `+0.10` 1.77%, `+0.20` 2.72%, `+0.30` 7.89%.
- MKG-W / M-Hyper + AdaMF-MAT: `-0.30` 37.29%, `-0.20` 5.16%, `-0.10` 2.76%, `-0.05` 2.78%, `0.00` 52.01%.
- MKG-W / NativE + AdaMF-MAT: `-0.30` 23.32%, `-0.20` 5.32%, `-0.10` 3.25%, `-0.05` 4.51%, `0.00` 41.85%, `+0.05` 21.74%.
- DB15K / M-Hyper + NativE: `-0.30` 28.69%, `-0.20` 5.44%, `-0.10` 2.79%, `-0.05` 2.69%, `0.00` 60.40%.
- DB15K / M-Hyper + AdaMF-MAT: `-0.30` 29.72%, `-0.20` 5.46%, `-0.10` 2.89%, `-0.05` 2.73%, `0.00` 59.19%.
- DB15K / NativE + AdaMF-MAT: `-0.30` 32.61%, `-0.20` 6.70%, `-0.10` 3.61%, `-0.05` 3.52%, `0.00` 53.56%.

## 6. Problems and ambiguities

No blocker remains for phase-1 construction. The existing DEV export contains stable query IDs, original head/relation/tail IDs, seeds, directions, the frozen normalization label, the full exact alpha grid, and the locked Global selection.

The export stores exact per-alpha RR rather than the full candidate score matrices. This is sufficient and safer for phase 1 because every frozen local action lies on the exported grid. It also means the sampled validation compares against the evaluator's stored exact outputs instead of rerunning checkpoints. No expert or full-ranking evaluator was rerun.

The current runtime lacks `pytest`, so the new tests use standard-library `unittest`; all three tests pass. All six real DEV builds also passed their embedded validation suite.
