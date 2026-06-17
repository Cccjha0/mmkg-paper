# Supplementary Table Bundle

| Table | TeX file | Source data | Status |
|---|---|---|---|
| S1 | `table_s1_relation_group_sanity_check.tex` | `docs/relation_type_summary_min20.json` | generated |
| S2 | `table_s2_degree_bucket_sanity_check.tex` | `docs/paper_tables/table_degree_bucket_sanity_check.csv` | generated |
| S3 | `table_s3_ordinal_gain_modeling.tex` | clean router ordinal/regression CSVs | generated |
| S4 | `table_s4_delta_sensitivity.tex` | `outputs/router/eval/clean/delta_sensitivity_clean.csv` | generated |
| S5 | `table_s5_per_seed_model_comparison.tex` | `outputs/router/test/*_query_eval_seed*.csv` | generated |
| S6 | `table_s6_bootstrap_implementation_details.tex` | `outputs/score_ensemble/eval/score_aware_bootstrap_ci.csv` | generated |
| S7 | `table_s7_hyperparameter_training_config.tex` | merged configs and router/candidate-router summaries | generated |

Optional alpha-stability table retained as `table_s5_alpha_sweep_stability.tex`; renumber it only if you decide to include it after S7.

S4 should be described as margin sensitivity rather than as full robustness across many deltas; the existing data covers delta values 0.00, 0.01, and 0.02.
