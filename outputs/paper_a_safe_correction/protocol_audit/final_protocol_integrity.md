DEV for all selection; TEST for immutable final evaluation only.

# Paper A final protocol-integrity audit

**Overall status: PASS**

This read-only audit evaluated 76 checks across 13 experiment groups. It changed no method, threshold, orientation, pair definition, or result asset.

## Experiment status

| Experiment | Status |
|---|---:|
| beta_sensitivity | PASS |
| boundary_static_query_soft_anchored | PASS |
| confidence_harm | PASS |
| core_ablation | PASS |
| dynasemble_boundary | PASS |
| dynasemble_main | PASS |
| fallback_audit | PASS |
| global_information_boundary | PASS |
| grouped_folds | PASS |
| main_static_query_soft_anchored | PASS |
| negative_transfer | PASS |
| primary_selection | PASS |
| risk_coverage | PASS |

## Leakage checks

| ID | Experiment | Dataset / Pair | Method | Status | Evidence-backed conclusion |
|---|---|---|---|---:|---|
| P01 | primary_selection | ALL / ALL | Primary selection | PASS | Six hashed inputs are DEV-only; TEST_NOT_USED=true; test_files_read is empty. |
| P02 | primary_selection | ALL / ALL | Primary selection | PASS | {"post_hoc_mrr_margin": null, "primary": "model with higher pooled DEV MRR", "reliable_primary_if_and_only_if": ["pooled DEV primary-minus-secondary delta > 0", "all 3 paired-seed deltas > 0", "original-triple clustered bootstrap 95% CI lower bound > 0"], "tie_policy": "boundary / non-reliable-primary"} |
| P03 | primary_selection | ALL / ALL | Primary selection | PASS | Decision CSV is present and hashed. |
| F01 | grouped_folds | MKG-W / M-Hyper + NativE | Query-soft / Anchored / core ablations | PASS | Validated 4,276 original-triple clusters and 25,656 observations; each cluster has 3 seeds x 2 directions in one fold. |
| G01 | main_static_query_soft_anchored | MKG-W / M-Hyper + NativE | Global alpha | PASS | selection_sha256=28f40889a625e53cf14c69111f2225f1d998019987bf6109a46c06126a43db6c; global_alpha=0.6; lock_alpha0=0.6; DEV source hash match=True. |
| A01 | main_static_query_soft_anchored | MKG-W / M-Hyper + NativE | Anchored Dynamic | PASS | selection_boundary='all fitting and hyperparameter selection use DEV only'; beta=0.5; threshold=0.0. |
| A02 | main_static_query_soft_anchored | MKG-W / M-Hyper + NativE | Query-soft / Anchored Dynamic | PASS | feature_schema_match=True; alpha_grid_match=True; policy_mirror_match=True; rows=25,644; row_metadata_problems=none. |
| A03 | main_static_query_soft_anchored | MKG-W / M-Hyper + NativE | Global / Query-soft / Anchored Dynamic | PASS | Matched 25,644 TEST observations over ranks, reciprocal ranks, Global alpha, and all 21 exact-ranking alpha-grid RR columns. |
| A04 | main_static_query_soft_anchored | MKG-W / M-Hyper + NativE | Anchored Dynamic | PASS | TEST summary embeds an exact copy of the DEV policy; locked model and selection hashes match. The immutable apply path cannot produce this summary without loading the lock. |
| D01 | dynasemble_main | MKG-W / M-Hyper + NativE | DynaSemble | PASS | training_split=dev_only; three seed-specific selector hashes match; TEST excluded from training/selection. |
| D02 | dynasemble_main | MKG-W / M-Hyper + NativE | DynaSemble | PASS | lock_sha256=4367219f681ac2bcfa373e7dd838a138dde9a0095f21bcb39f5f7efb82ef49c6; summary_lock_match=True; method_config_match=True; Global selection hash match=True; protocol hash match=True. |
| D03 | dynasemble_main | MKG-W / M-Hyper + NativE | DynaSemble | PASS | Validated 25,644 TEST rows; exact-reference rank mismatches=0; max RR error=0; reference hash match=True. |
| D04 | dynasemble_main | MKG-W / M-Hyper + NativE | DynaSemble | PASS | All seed-specific expert config and checkpoint hashes match the DEV lock. |
| D05 | dynasemble_main | MKG-W / M-Hyper + NativE | DynaSemble | PASS | provenance_rebind_count=0; each rebind marks test_outcomes_read=false and method_or_hyperparameter_changed=false and records the verified execution-critical hashes. |
| F01 | grouped_folds | MKG-W / M-Hyper + AdaMF-MAT | Query-soft / Anchored / core ablations | PASS | Validated 4,276 original-triple clusters and 25,656 observations; each cluster has 3 seeds x 2 directions in one fold. |
| G01 | main_static_query_soft_anchored | MKG-W / M-Hyper + AdaMF-MAT | Global alpha | PASS | selection_sha256=f1c688963e62b52470c30ee722713bb372c51df61421503396cfbc3e5e8ad198; global_alpha=1.0; lock_alpha0=1.0; DEV source hash match=True. |
| A01 | main_static_query_soft_anchored | MKG-W / M-Hyper + AdaMF-MAT | Anchored Dynamic | PASS | selection_boundary='all fitting and hyperparameter selection use DEV only'; beta=0.5; threshold=0.1. |
| A02 | main_static_query_soft_anchored | MKG-W / M-Hyper + AdaMF-MAT | Query-soft / Anchored Dynamic | PASS | feature_schema_match=True; alpha_grid_match=True; policy_mirror_match=True; rows=25,644; row_metadata_problems=none. |
| A03 | main_static_query_soft_anchored | MKG-W / M-Hyper + AdaMF-MAT | Global / Query-soft / Anchored Dynamic | PASS | Matched 25,644 TEST observations over ranks, reciprocal ranks, Global alpha, and all 21 exact-ranking alpha-grid RR columns. |
| A04 | main_static_query_soft_anchored | MKG-W / M-Hyper + AdaMF-MAT | Anchored Dynamic | PASS | TEST summary embeds an exact copy of the DEV policy; locked model and selection hashes match. The immutable apply path cannot produce this summary without loading the lock. |
| D01 | dynasemble_main | MKG-W / M-Hyper + AdaMF-MAT | DynaSemble | PASS | training_split=dev_only; three seed-specific selector hashes match; TEST excluded from training/selection. |
| D02 | dynasemble_main | MKG-W / M-Hyper + AdaMF-MAT | DynaSemble | PASS | lock_sha256=faa9ead15f78f0502cea5dab898b2e1abacec7a6002b6958cc5578d471012649; summary_lock_match=True; method_config_match=True; Global selection hash match=True; protocol hash match=True. |
| D03 | dynasemble_main | MKG-W / M-Hyper + AdaMF-MAT | DynaSemble | PASS | Validated 25,644 TEST rows; exact-reference rank mismatches=0; max RR error=0; reference hash match=True. |
| D04 | dynasemble_main | MKG-W / M-Hyper + AdaMF-MAT | DynaSemble | PASS | All seed-specific expert config and checkpoint hashes match the DEV lock. |
| D05 | dynasemble_main | MKG-W / M-Hyper + AdaMF-MAT | DynaSemble | PASS | provenance_rebind_count=0; each rebind marks test_outcomes_read=false and method_or_hyperparameter_changed=false and records the verified execution-critical hashes. |
| F01 | grouped_folds | DB15K / M-Hyper + NativE | Query-soft / Anchored / core ablations | PASS | Validated 7,922 original-triple clusters and 47,532 observations; each cluster has 3 seeds x 2 directions in one fold. |
| G01 | main_static_query_soft_anchored | DB15K / M-Hyper + NativE | Global alpha | PASS | selection_sha256=b0378a6bc3bc7f8ee50dc1f0aeee2d0073df801372d4f657dd35113ed07d2c04; global_alpha=1.0; lock_alpha0=1.0; DEV source hash match=True. |
| A01 | main_static_query_soft_anchored | DB15K / M-Hyper + NativE | Anchored Dynamic | PASS | selection_boundary='all fitting and hyperparameter selection use DEV only'; beta=0.5; threshold=0.0. |
| A02 | main_static_query_soft_anchored | DB15K / M-Hyper + NativE | Query-soft / Anchored Dynamic | PASS | feature_schema_match=True; alpha_grid_match=True; policy_mirror_match=True; rows=59,412; row_metadata_problems=none. |
| A03 | main_static_query_soft_anchored | DB15K / M-Hyper + NativE | Global / Query-soft / Anchored Dynamic | PASS | Matched 59,412 TEST observations over ranks, reciprocal ranks, Global alpha, and all 21 exact-ranking alpha-grid RR columns. |
| A04 | main_static_query_soft_anchored | DB15K / M-Hyper + NativE | Anchored Dynamic | PASS | TEST summary embeds an exact copy of the DEV policy; locked model and selection hashes match. The immutable apply path cannot produce this summary without loading the lock. |
| D01 | dynasemble_main | DB15K / M-Hyper + NativE | DynaSemble | PASS | training_split=dev_only; three seed-specific selector hashes match; TEST excluded from training/selection. |
| D02 | dynasemble_main | DB15K / M-Hyper + NativE | DynaSemble | PASS | lock_sha256=ad10678bdc1cb2b43e8105d1980202edd49ac7e06be3489844e976a2f75cace1; summary_lock_match=True; method_config_match=True; Global selection hash match=True; protocol hash match=True. |
| D03 | dynasemble_main | DB15K / M-Hyper + NativE | DynaSemble | PASS | Validated 59,412 TEST rows; exact-reference rank mismatches=0; max RR error=0; reference hash match=True. |
| D04 | dynasemble_main | DB15K / M-Hyper + NativE | DynaSemble | PASS | All seed-specific expert config and checkpoint hashes match the DEV lock. |
| D05 | dynasemble_main | DB15K / M-Hyper + NativE | DynaSemble | PASS | provenance_rebind_count=0; each rebind marks test_outcomes_read=false and method_or_hyperparameter_changed=false and records the verified execution-critical hashes. |
| F01 | grouped_folds | DB15K / M-Hyper + AdaMF-MAT | Query-soft / Anchored / core ablations | PASS | Validated 7,922 original-triple clusters and 47,532 observations; each cluster has 3 seeds x 2 directions in one fold. |
| G01 | main_static_query_soft_anchored | DB15K / M-Hyper + AdaMF-MAT | Global alpha | PASS | selection_sha256=4bbac94200d8d11821670e4a183f92975baba5c2531bf13f50aedf0fba64f282; global_alpha=1.0; lock_alpha0=1.0; DEV source hash match=True. |
| A01 | main_static_query_soft_anchored | DB15K / M-Hyper + AdaMF-MAT | Anchored Dynamic | PASS | selection_boundary='all fitting and hyperparameter selection use DEV only'; beta=0.5; threshold=0.3. |
| A02 | main_static_query_soft_anchored | DB15K / M-Hyper + AdaMF-MAT | Query-soft / Anchored Dynamic | PASS | feature_schema_match=True; alpha_grid_match=True; policy_mirror_match=True; rows=59,412; row_metadata_problems=none. |
| A03 | main_static_query_soft_anchored | DB15K / M-Hyper + AdaMF-MAT | Global / Query-soft / Anchored Dynamic | PASS | Matched 59,412 TEST observations over ranks, reciprocal ranks, Global alpha, and all 21 exact-ranking alpha-grid RR columns. |
| A04 | main_static_query_soft_anchored | DB15K / M-Hyper + AdaMF-MAT | Anchored Dynamic | PASS | TEST summary embeds an exact copy of the DEV policy; locked model and selection hashes match. The immutable apply path cannot produce this summary without loading the lock. |
| D01 | dynasemble_main | DB15K / M-Hyper + AdaMF-MAT | DynaSemble | PASS | training_split=dev_only; three seed-specific selector hashes match; TEST excluded from training/selection. |
| D02 | dynasemble_main | DB15K / M-Hyper + AdaMF-MAT | DynaSemble | PASS | lock_sha256=7ee6c3c608ace74d2afb94abb17cf6a02fedfc18bc33e7e110b56d62004d5a3d; summary_lock_match=True; method_config_match=True; Global selection hash match=True; protocol hash match=True. |
| D03 | dynasemble_main | DB15K / M-Hyper + AdaMF-MAT | DynaSemble | PASS | Validated 59,412 TEST rows; exact-reference rank mismatches=0; max RR error=0; reference hash match=True. |
| D04 | dynasemble_main | DB15K / M-Hyper + AdaMF-MAT | DynaSemble | PASS | All seed-specific expert config and checkpoint hashes match the DEV lock. |
| D05 | dynasemble_main | DB15K / M-Hyper + AdaMF-MAT | DynaSemble | PASS | provenance_rebind_count=0; each rebind marks test_outcomes_read=false and method_or_hyperparameter_changed=false and records the verified execution-critical hashes. |
| F01 | grouped_folds | MKG-W / NativE + AdaMF-MAT | Query-soft / Anchored / core ablations | PASS | Validated 4,276 original-triple clusters and 25,656 observations; each cluster has 3 seeds x 2 directions in one fold. |
| G01 | boundary_static_query_soft_anchored | MKG-W / NativE + AdaMF-MAT | Global alpha | PASS | selection_sha256=15bd47d17171a3e603c5a5fb2f27bc33875ef6fd7f78a95be3f12f67d3fdfeaf; global_alpha=0.95; lock_alpha0=0.95; DEV source hash match=True. |
| A01 | boundary_static_query_soft_anchored | MKG-W / NativE + AdaMF-MAT | Anchored Dynamic | PASS | selection_boundary='all fitting and hyperparameter selection use DEV only'; beta=0.5; threshold=0.3. |
| A02 | boundary_static_query_soft_anchored | MKG-W / NativE + AdaMF-MAT | Query-soft / Anchored Dynamic | PASS | feature_schema_match=True; alpha_grid_match=True; policy_mirror_match=True; rows=25,644; row_metadata_problems=none. |
| A03 | boundary_static_query_soft_anchored | MKG-W / NativE + AdaMF-MAT | Global / Query-soft / Anchored Dynamic | PASS | Matched 25,644 TEST observations over ranks, reciprocal ranks, Global alpha, and all 21 exact-ranking alpha-grid RR columns. |
| A04 | boundary_static_query_soft_anchored | MKG-W / NativE + AdaMF-MAT | Anchored Dynamic | PASS | TEST summary embeds an exact copy of the DEV policy; locked model and selection hashes match. The immutable apply path cannot produce this summary without loading the lock. |
| D01 | dynasemble_boundary | MKG-W / NativE + AdaMF-MAT | DynaSemble | PASS | training_split=dev_only; three seed-specific selector hashes match; TEST excluded from training/selection. |
| D02 | dynasemble_boundary | MKG-W / NativE + AdaMF-MAT | DynaSemble | PASS | lock_sha256=a1bc0539c15ec0a3e62ee6a238b7052b190814de3d6b5cf4dbf8133d33d1c494; summary_lock_match=True; method_config_match=True; Global selection hash match=True; protocol hash match=True. |
| D03 | dynasemble_boundary | MKG-W / NativE + AdaMF-MAT | DynaSemble | PASS | Validated 25,644 TEST rows; exact-reference rank mismatches=0; max RR error=0; reference hash match=True. |
| D04 | dynasemble_boundary | MKG-W / NativE + AdaMF-MAT | DynaSemble | PASS | All seed-specific expert config and checkpoint hashes match the DEV lock. |
| D05 | dynasemble_boundary | MKG-W / NativE + AdaMF-MAT | DynaSemble | PASS | provenance_rebind_count=1; each rebind marks test_outcomes_read=false and method_or_hyperparameter_changed=false and records the verified execution-critical hashes. |
| F01 | grouped_folds | DB15K / NativE + AdaMF-MAT | Query-soft / Anchored / core ablations | PASS | Validated 7,922 original-triple clusters and 47,532 observations; each cluster has 3 seeds x 2 directions in one fold. |
| G01 | boundary_static_query_soft_anchored | DB15K / NativE + AdaMF-MAT | Global alpha | PASS | selection_sha256=cce837ceb70ef765e6cc0ac6230b6a5241a7167e10621cb205339dbaa767c499; global_alpha=1.0; lock_alpha0=1.0; DEV source hash match=True. |
| A01 | boundary_static_query_soft_anchored | DB15K / NativE + AdaMF-MAT | Anchored Dynamic | PASS | selection_boundary='all fitting and hyperparameter selection use DEV only'; beta=0.5; threshold=0.2. |
| A02 | boundary_static_query_soft_anchored | DB15K / NativE + AdaMF-MAT | Query-soft / Anchored Dynamic | PASS | feature_schema_match=True; alpha_grid_match=True; policy_mirror_match=True; rows=59,412; row_metadata_problems=none. |
| A03 | boundary_static_query_soft_anchored | DB15K / NativE + AdaMF-MAT | Global / Query-soft / Anchored Dynamic | PASS | Matched 59,412 TEST observations over ranks, reciprocal ranks, Global alpha, and all 21 exact-ranking alpha-grid RR columns. |
| A04 | boundary_static_query_soft_anchored | DB15K / NativE + AdaMF-MAT | Anchored Dynamic | PASS | TEST summary embeds an exact copy of the DEV policy; locked model and selection hashes match. The immutable apply path cannot produce this summary without loading the lock. |
| D01 | dynasemble_boundary | DB15K / NativE + AdaMF-MAT | DynaSemble | PASS | training_split=dev_only; three seed-specific selector hashes match; TEST excluded from training/selection. |
| D02 | dynasemble_boundary | DB15K / NativE + AdaMF-MAT | DynaSemble | PASS | lock_sha256=3d4fa52b6646e2762415fb8be9fc26c72cef6c0df9f4cf88bfe0e6975c8ed7cd; summary_lock_match=True; method_config_match=True; Global selection hash match=True; protocol hash match=True. |
| D03 | dynasemble_boundary | DB15K / NativE + AdaMF-MAT | DynaSemble | PASS | Validated 59,412 TEST rows; exact-reference rank mismatches=0; max RR error=0; reference hash match=True. |
| D04 | dynasemble_boundary | DB15K / NativE + AdaMF-MAT | DynaSemble | PASS | All seed-specific expert config and checkpoint hashes match the DEV lock. |
| D05 | dynasemble_boundary | DB15K / NativE + AdaMF-MAT | DynaSemble | PASS | provenance_rebind_count=1; each rebind marks test_outcomes_read=false and method_or_hyperparameter_changed=false and records the verified execution-critical hashes. |
| C01 | core_ablation | ALL / ALL | Core ablations | PASS | Matched-contract validation passed; no base retraining, TEST selection, or ablation-specific hyperparameter search. |
| R01 | risk_coverage | ALL / ALL | Anchored Dynamic diagnostic | PASS | fixed_coverage_grid=True; test_used_for_selection=False; test_is_diagnostic_only=True; threshold_modified=False. |
| N01 | negative_transfer | ALL / ALL | Global / Query-soft / DynaSemble / Anchored | PASS | no_training=True; no_test_driven_changes=True; test_final_only=True. |
| H01 | confidence_harm | ALL / ALL | Anchored confidence diagnostic | PASS | Confidence threshold is disabled; non-finite fallback only; strict raw-bounded-vs-Global harm label; TEST diagnostic-only; all source, lock, and output hashes match. |
| B01 | beta_sensitivity | ALL / ALL | Anchored beta diagnostic | PASS | Formal grid is 0.05-0.50; beta=1.0 is diagnostic-only; selection remains the DEV lock; theoretical and core-ablation reproduction checks pass; all source/lock/output hashes match. |
| FBA01 | fallback_audit | ALL / ALL | Anchored fallback diagnostic | PASS | Stored policies reconstruct exactly; the counterfactual removes confidence fallback only; TEST is diagnostic-only; all source, lock, and output hashes match. |
| L01 | global_information_boundary | ALL / ALL | ALL | PASS | Protocols and implementations explicitly separate DEV-only selection from locked/immutable TEST apply; method-specific checks independently validate the resulting hashes and parameters. |

## Failures

No failures were detected. All checked selections are DEV-only and all checked TEST evaluations are immutable applications or diagnostic/final reporting only.

## Interpretation boundary

PASS means that the available final assets, embedded locks, hashes, row-level metadata, grouped-fold assignments, and implementation/protocol evidence are mutually consistent with the declared information boundary. It does not prove the absence of actions outside the recorded repository history; it makes the recorded workflow independently auditable and causes any detected inconsistency to fail closed.

Historical absolute paths in locks are not treated as current filesystem identities. They are mapped to the corresponding repository-relative asset and then checked against the hash recorded in the lock.
