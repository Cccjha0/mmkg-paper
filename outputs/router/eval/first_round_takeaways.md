# First Round Takeaways

## Main Outcome

- 当前 best learned router 是 `xgb + delta=0.01 + tau=0.7`，overall MRR = `0.3160`。
- 它高于固定 `Residual-only` 的 query-level test MRR `0.2930`，也高于文档主结果中的 `Full Model` `0.21`。
- 当前 `Oracle` 的 upper bound MRR 为 `0.3337`，说明 learned router 之上仍然存在可提升空间。
- `logistic` 的最优点在 `tau=0.7`，overall MRR = `0.3111`；`xgb` 仍然稳定优于 `logistic`。

## Threshold Scan

- `xgb` 的 threshold scan 呈现清晰 tradeoff：
  - `tau=0.1`: MRR `0.2514`, coverage `0.483`, gain_precision `0.327`
  - `tau=0.3`: MRR `0.2921`, coverage `0.333`, gain_precision `0.452`
  - `tau=0.5`: MRR `0.3084`, coverage `0.252`, gain_precision `0.555`
  - `tau=0.7`: MRR `0.3160`, coverage `0.185`, gain_precision `0.670`
  - `tau=0.9`: MRR `0.3142`, coverage `0.096`, gain_precision `0.846`
- `logistic` 也呈现相同方向的 tradeoff，但整体曲线低于 `xgb`。

## Subgroup Pattern

- learned router 在 `head_has_img` 上明显比 `Residual-only` 更合理，在 `tail_no_img` 上则更接近甚至超过 `Residual-only`。
- 随着 `tau` 提高，`tail_no_img` 的 fusion ratio 明显下降，说明更保守的阈值更符合结构占优场景。

## Feature Ablation

- 当前最强 ablation 组合是 `xgb + F4`，overall MRR = `0.2556`。
- `F1`（仅 `target_has_img`）表现明显不足，说明 router 不是简单地在学“有图就开 fusion”。
- `F2` 相比 `F1` 提升很大，说明 `direction + relation_gain_prior` 是关键特征。
- `F3` 相比 `F2` 增益很小，说明模态一致性特征目前只是弱补充。
- `F4` 再次明显提升，说明 expert confidence 特征（`fusion_margin / struct_margin / delta_margin`）是主要增强项。

## Remaining Gaps

- `Full Model` 的 subgroup 结果现已并入统一 subgroup 汇总表；当前剩余缺口主要在更论文式的文字组织与版式整理。
- 正式图文件依赖 `matplotlib`；如果当前环境缺失该依赖，脚本会保留表格和总结，但跳过 PNG 输出。

## Quick Table

| model | delta | tau | mrr | hits1 | hits3 | hits10 | fusion_coverage | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full Model |  |  | 0.21 | 0.1167 | 0.2658 | 0.3794 |  | docs/MAIN_RESULTS_SUMMARY.md |
| Oracle |  |  | 0.33373734067328487 | 0.2645166666666667 | 0.3774666666666667 | 0.45411666666666667 | 0.48428333333333334 | recomputed_from_router_test_features |
| Residual-only |  |  | 0.2930494303909519 | 0.2328 | 0.33065 | 0.40305 |  | query_eval |
| Rule-based | 0.01 | 0.5 | 0.29394690617851077 | 0.23303333333333334 | 0.33145 | 0.40523333333333333 | 0.0187 | recomputed_from_router_test_features |
| logistic | 0.01 | 0.3 | 0.2817005499085218 | 0.20693333333333333 | 0.3252833333333333 | 0.4191 | 0.38516666666666666 | router_eval_json |
| logistic | 0.01 | 0.5 | 0.3013147290687338 | 0.22836666666666666 | 0.3464333333333333 | 0.43291666666666667 | 0.26811666666666667 | router_eval_json |
| logistic | 0.01 | 0.7 | 0.3110909177397595 | 0.2402 | 0.35656666666666664 | 0.4375833333333333 | 0.1788 | router_eval_json |
| xgb | 0.01 | 0.3 | 0.29214956639187084 | 0.2191 | 0.3353 | 0.42518333333333336 | 0.33336666666666664 | router_eval_json |
| xgb | 0.01 | 0.5 | 0.308446707249557 | 0.23598333333333332 | 0.35385 | 0.43755 | 0.2524166666666667 | router_eval_json |
| xgb | 0.01 | 0.7 | 0.31595694366688554 | 0.24476666666666666 | 0.36106666666666665 | 0.44288333333333335 | 0.18483333333333332 | router_eval_json |