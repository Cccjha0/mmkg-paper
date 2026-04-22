# Clean Significance Rerun Plan

## Goal

Update the significance analysis so it matches the current strongest clean claims in
`docs/PAPER_REVISION_INSTRUCTION_SHEET_AFTER_NEW_RESULTS.md`.

The old significance files only support the old global-threshold clean baseline.
They do not match the current strongest clean methods.

## Current strongest clean candidates

Based on the current result tables in `outputs/router/eval/clean/`:

1. `E5 best`:
   - file: `outputs/router/eval/clean/regression_router_scan_xgb_C4.csv`
   - config: `regressor_type=xgb`, `theta=0.00`
   - `overall_mrr = 0.29818238528001295`

2. `E1 best`:
   - file: `outputs/router/eval/clean/dual_threshold_scan_clean_logistic_delta_0.01_C4.csv`
   - config: `model=logistic`, `tau_head=0.3`, `tau_tail=0.9`
   - `overall_mrr = 0.29743590499633776`

3. `clean rule` baseline:
   - file: `outputs/router/routing/clean/test_router_predictions_clean_rule_delta_0.01_tau_0.5_rule.csv`
   - includes seeds `1,2,3`
   - `overall_mrr = 0.29428194121180534`

## Existing query-level inputs that can be reused

1. `E5 xgb regression scan` query rows:
   - `outputs/router/eval/clean/regression_router_scan_xgb_C4_query_rows.csv`
   - select `config_id == "theta=0.00"`

2. `E1 logistic dual-threshold scan` query rows:
   - `outputs/router/eval/clean/dual_threshold_scan_clean_logistic_delta_0.01_C4_query_rows.csv`
   - select `config_id == "tau_head=0.3|tau_tail=0.9"`

3. `clean rule` routed rows:
   - `outputs/router/routing/clean/test_router_predictions_clean_rule_delta_0.01_tau_0.5_rule.csv`

## Required comparisons

These are the minimum comparisons required by the revision sheet:

1. `E1 best` vs `clean rule`
2. `E5 best` vs `clean rule`
3. `E5 best` vs `Residual-only`

Optional but useful:

4. `E1 best` vs `Residual-only`
5. `E5 best` vs `E1 best`

## Suggested intermediate files

Create a small input folder to avoid repeatedly filtering the large query-row scans:

- `outputs/router/eval/clean/significance_inputs/e1_best_dual_logistic_tau_head_0.3_tau_tail_0.9.csv`
- `outputs/router/eval/clean/significance_inputs/e5_best_regression_xgb_theta_0.00.csv`

## Suggested output files

- `outputs/router/eval/clean/significance_e1_best_vs_rule.json`
- `outputs/router/eval/clean/significance_e1_best_vs_rule.csv`
- `outputs/router/eval/clean/significance_e5_best_vs_rule.json`
- `outputs/router/eval/clean/significance_e5_best_vs_rule.csv`
- `outputs/router/eval/clean/significance_e5_best_vs_residual.json`
- `outputs/router/eval/clean/significance_e5_best_vs_residual.csv`

Optional:

- `outputs/router/eval/clean/significance_e1_best_vs_residual.json`
- `outputs/router/eval/clean/significance_e1_best_vs_residual.csv`
- `outputs/router/eval/clean/significance_e5_best_vs_e1_best.json`
- `outputs/router/eval/clean/significance_e5_best_vs_e1_best.csv`

## PowerShell extraction commands

Create the filtered per-config inputs:

```powershell
New-Item -ItemType Directory -Force outputs/router/eval/clean/significance_inputs

Import-Csv outputs/router/eval/clean/regression_router_scan_xgb_C4_query_rows.csv |
  Where-Object { $_.config_id -eq 'theta=0.00' } |
  Export-Csv outputs/router/eval/clean/significance_inputs/e5_best_regression_xgb_theta_0.00.csv -NoTypeInformation

Import-Csv outputs/router/eval/clean/dual_threshold_scan_clean_logistic_delta_0.01_C4_query_rows.csv |
  Where-Object { $_.config_id -eq 'tau_head=0.3|tau_tail=0.9' } |
  Export-Csv outputs/router/eval/clean/significance_inputs/e1_best_dual_logistic_tau_head_0.3_tau_tail_0.9.csv -NoTypeInformation
```

## Significance commands

### 1. E1 best vs clean rule

```powershell
python scripts/run_router_significance.py `
  --left-file outputs/router/eval/clean/significance_inputs/e1_best_dual_logistic_tau_head_0.3_tau_tail_0.9.csv `
  --left-source final `
  --left-label e1_best_dual_logistic_tau_head_0.3_tau_tail_0.9 `
  --right-file outputs/router/routing/clean/test_router_predictions_clean_rule_delta_0.01_tau_0.5_rule.csv `
  --right-source final `
  --right-label clean_rule_tau_0.5 `
  --comparison e1_best_vs_clean_rule `
  --out-json outputs/router/eval/clean/significance_e1_best_vs_rule.json `
  --out-csv outputs/router/eval/clean/significance_e1_best_vs_rule.csv
```

### 2. E5 best vs clean rule

```powershell
python scripts/run_router_significance.py `
  --left-file outputs/router/eval/clean/significance_inputs/e5_best_regression_xgb_theta_0.00.csv `
  --left-source final `
  --left-label e5_best_regression_xgb_theta_0.00 `
  --right-file outputs/router/routing/clean/test_router_predictions_clean_rule_delta_0.01_tau_0.5_rule.csv `
  --right-source final `
  --right-label clean_rule_tau_0.5 `
  --comparison e5_best_vs_clean_rule `
  --out-json outputs/router/eval/clean/significance_e5_best_vs_rule.json `
  --out-csv outputs/router/eval/clean/significance_e5_best_vs_rule.csv
```

### 3. E5 best vs Residual-only

Use the same routed file and switch the right source to `residual`.

```powershell
python scripts/run_router_significance.py `
  --left-file outputs/router/eval/clean/significance_inputs/e5_best_regression_xgb_theta_0.00.csv `
  --left-source final `
  --left-label e5_best_regression_xgb_theta_0.00 `
  --right-file outputs/router/eval/clean/significance_inputs/e5_best_regression_xgb_theta_0.00.csv `
  --right-source residual `
  --right-label residual_only `
  --comparison e5_best_vs_residual_only `
  --out-json outputs/router/eval/clean/significance_e5_best_vs_residual.json `
  --out-csv outputs/router/eval/clean/significance_e5_best_vs_residual.csv
```

## Optional significance commands

### 4. E1 best vs Residual-only

```powershell
python scripts/run_router_significance.py `
  --left-file outputs/router/eval/clean/significance_inputs/e1_best_dual_logistic_tau_head_0.3_tau_tail_0.9.csv `
  --left-source final `
  --left-label e1_best_dual_logistic_tau_head_0.3_tau_tail_0.9 `
  --right-file outputs/router/eval/clean/significance_inputs/e1_best_dual_logistic_tau_head_0.3_tau_tail_0.9.csv `
  --right-source residual `
  --right-label residual_only `
  --comparison e1_best_vs_residual_only `
  --out-json outputs/router/eval/clean/significance_e1_best_vs_residual.json `
  --out-csv outputs/router/eval/clean/significance_e1_best_vs_residual.csv
```

### 5. E5 best vs E1 best

```powershell
python scripts/run_router_significance.py `
  --left-file outputs/router/eval/clean/significance_inputs/e5_best_regression_xgb_theta_0.00.csv `
  --left-source final `
  --left-label e5_best_regression_xgb_theta_0.00 `
  --right-file outputs/router/eval/clean/significance_inputs/e1_best_dual_logistic_tau_head_0.3_tau_tail_0.9.csv `
  --right-source final `
  --right-label e1_best_dual_logistic_tau_head_0.3_tau_tail_0.9 `
  --comparison e5_best_vs_e1_best `
  --out-json outputs/router/eval/clean/significance_e5_best_vs_e1_best.json `
  --out-csv outputs/router/eval/clean/significance_e5_best_vs_e1_best.csv
```

## Write-up usage rule

Only use strong wording such as
`consistently improves over the clean rule baseline`
if:

1. `mean delta MRR > 0`
2. the paired bootstrap `95% CI` does not cover `0`

Otherwise use softer wording such as
`tends to improve over the clean rule baseline`.
