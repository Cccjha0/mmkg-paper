# Final Results Manifest

- Final status: `PASS`

## Canonical Final Outputs

| group | name | status | path |
| --- | --- | --- | --- |
| final_table | main_results_table.csv | final | outputs/router/eval/main_results_table.csv |
| final_table | main_results_table.md | final | outputs/router/eval/main_results_table.md |
| final_table | subgroup_results_table.csv | final | outputs/router/eval/subgroup_results_table.csv |
| final_table | subgroup_results_table.md | final | outputs/router/eval/subgroup_results_table.md |
| final_table | feature_ablation.csv | final | outputs/router/eval/feature_ablation.csv |
| final_table | feature_ablation.md | final | outputs/router/eval/feature_ablation.md |
| final_summary | first_round_takeaways.md | final | outputs/router/eval/first_round_takeaways.md |
| final_figure | threshold_coverage_mrr.png | final | outputs/router/figures/threshold_coverage_mrr.png |
| final_figure | router_feature_importance.png | final | outputs/router/figures/router_feature_importance.png |
| final_figure | selected_fusion_ratio_by_subgroup.png | final | outputs/router/figures/selected_fusion_ratio_by_subgroup.png |
| supporting_input | threshold_scan_logistic_delta_0.01.csv | supporting | outputs/router/eval/threshold_scan_logistic_delta_0.01.csv |
| supporting_input | threshold_scan_xgb_delta_0.01.csv | supporting | outputs/router/eval/threshold_scan_xgb_delta_0.01.csv |

## Consistency Checks

| check | passed |
| --- | --- |
| all_required_files_exist | True |
| main_table_has_expected_models | True |
| subgroup_table_has_expected_models | True |
| all_models_have_three_regimes_in_subgroup | True |
| best_learned_is_xgb_delta_0.01_tau_0.7 | True |
| oracle_beats_best_learned | True |
| best_learned_beats_residual_only | True |
| best_learned_beats_full_model | True |
| learned_beats_rule_based | True |
| feature_ablation_has_f1_to_f4_for_logistic_and_xgb | True |

## Key Numbers

| metric | value |
| --- | --- |
| best_learned_model | xgb |
| best_learned_delta | 0.01 |
| best_learned_tau | 0.7 |
| best_learned_mrr | 0.31595694366688554 |
| oracle_mrr | 0.33373734067328487 |
| residual_only_mrr | 0.2930494303909519 |
| full_model_mrr | 0.21 |
| rule_based_mrr | 0.29394690617851077 |

## Subgroup Coverage

| model | regimes |
| --- | --- |
| Full Model | head_has_img, head_no_img, tail_no_img |
| Gate-only | head_has_img, head_no_img, tail_no_img |
| Oracle | head_has_img, head_no_img, tail_no_img |
| Residual-only | head_has_img, head_no_img, tail_no_img |
| Rule-based | head_has_img, head_no_img, tail_no_img |
| logistic | head_has_img, head_no_img, tail_no_img |
| xgb | head_has_img, head_no_img, tail_no_img |

## Notes

- Canonical final tables are the markdown/csv files under outputs/router/eval refreshed by scripts/make_router_tables_figures.py.
- Threshold scan csv files are treated as supporting inputs to the final tables/figures, not as standalone final presentation artifacts.
- Raw query_eval, gain_labels, priors, feature tables, trained models, and per-tau routing predictions are intermediate or supporting artifacts rather than final paper-facing outputs.