| category | model | delta | tau | mrr | hits1 | hits3 | hits10 | fusion_coverage | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_expert | Gate-only |  |  | 0.16883089489062678 | 0.0869 | 0.20838333333333334 | 0.32495 |  | query_eval |
| fixed_expert | Residual-only |  |  | 0.2930494303909519 | 0.2328 | 0.33065 | 0.40305 |  | query_eval |
| paper_baseline | Full Model |  |  | 0.21 | 0.1167 | 0.2658 | 0.3794 |  | docs/MAIN_RESULTS_SUMMARY.md |
| oracle | Oracle |  |  | 0.33373734067328487 | 0.2645166666666667 | 0.3774666666666667 | 0.45411666666666667 | 0.48428333333333334 | recomputed_from_router_test_features |
| rule_based | Rule-based | 0.01 | 0.5 | 0.29394690617851077 | 0.23303333333333334 | 0.33145 | 0.40523333333333333 | 0.0187 | recomputed_from_router_test_features |
| learned_router | logistic | 0.01 | 0.3 | 0.2817005499085218 | 0.20693333333333333 | 0.3252833333333333 | 0.4191 | 0.38516666666666666 | router_eval_json |
| learned_router | logistic | 0.01 | 0.5 | 0.3013147290687338 | 0.22836666666666666 | 0.3464333333333333 | 0.43291666666666667 | 0.26811666666666667 | router_eval_json |
| learned_router | logistic | 0.01 | 0.7 | 0.3110909177397595 | 0.2402 | 0.35656666666666664 | 0.4375833333333333 | 0.1788 | router_eval_json |
| learned_router | xgb | 0.01 | 0.3 | 0.29214956639187084 | 0.2191 | 0.3353 | 0.42518333333333336 | 0.33336666666666664 | router_eval_json |
| learned_router | xgb | 0.01 | 0.5 | 0.308446707249557 | 0.23598333333333332 | 0.35385 | 0.43755 | 0.2524166666666667 | router_eval_json |
| learned_router | xgb | 0.01 | 0.7 | 0.31595694366688554 | 0.24476666666666666 | 0.36106666666666665 | 0.44288333333333335 | 0.18483333333333332 | router_eval_json |