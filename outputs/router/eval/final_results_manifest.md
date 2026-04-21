# Final Results Manifest

- Final status: `FAIL`

## Checks

| check | passed |
| --- | --- |
| all_clean_required_files_exist | True |
| all_posthoc_required_files_exist | True |
| clean_main_table_has_expected_models | True |
| clean_scan_best_model_consistent_with_clean_main_table | False |
| clean_feature_ablation_has_C1_to_C4 | True |
| posthoc_feature_ablation_has_PH1_to_PH4_or_full | True |
| no_illegal_features_in_clean_feature_columns | True |
| clean_subgroup_has_expected_regimes | True |
| posthoc_subgroup_has_expected_regimes | True |
| posthoc_best_ge_clean_best | False |

## Notes

- Main paper should consume clean outputs only.
- Posthoc outputs are analysis-only and may legally use target-aware or confidence-aware fields.
- The legality regression check scans clean model feature_columns.json files for forbidden fields.
- The clean best-model consistency check compares the best learned row in the clean main table against clean threshold scan csvs.