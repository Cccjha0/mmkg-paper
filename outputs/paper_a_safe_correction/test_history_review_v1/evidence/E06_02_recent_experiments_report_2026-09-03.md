# MMKGC 新方向近期实验结果总报告（2026-09-03）

## 1. 范围与结论

本报告整理 2026-09-01 至 2026-09-03 围绕“稳健的 query-dependent 异构专家组合”完成的实验。范围包括：

1. MKG-W 上 Gate-v2-state 与 Structural-v2 的受控互补性验证；
2. MKG-W、DB15K 上 M-Hyper、NativE、AdaMF-MAT 的三 seed 训练与 exact filtered full-ranking；
3. Global/Relation alpha、无锚点 Query-soft 与 Anchored Dynamic 的 grouped DEV cross-fit；
4. P3 的 correction range、fallback、feature group 与 anchor 消融；
5. 四个 strong expert pair 的完整 DEV 锁定和 TEST 应用；
6. 三 seed、head/tail、fallback、saturation 与源文件哈希审计。

核心结果已经闭环：Anchored Dynamic 在 `4/4` 个 dataset-pair TEST 上均优于 DEV-locked Global alpha，四个 original-triple clustered 95% CI 全部高于零；`12/12` 个 pair-seed 单元和 `8/8` 个 pair-direction 单元均为正。无锚点 Query-soft 仅在 `1/4` 个 pair 上为正，在另外三个 pair 上明显退化。

因此，当前最稳妥的主张不是“动态选择总能超过静态组合”，而是：**围绕可靠 Global alpha 的有界、可回退修正，将依赖 expert pair 的动态收益转化为跨数据集和跨 expert pair 的稳定提升。**

## 2. 证据边界

| 证据线 | 数据集 / Pair | 用途 | 证据等级 |
| --- | --- | --- | --- |
| 受控现象验证 | MKG-W / Gate-v2-state + Structural-v2 | 验证 Oracle headroom、score combination 与 seed 稳定性 | exploratory / mechanism |
| Strong-pair 主验证 | MKG-W / M-Hyper + NativE | 互补专家条件下的 clean locked TEST | confirmatory |
| Strong-pair 稳健性 | MKG-W / M-Hyper + AdaMF-MAT | 第二专家明显较弱时的安全性 | confirmatory |
| 跨数据集复现 | DB15K / M-Hyper + NativE、AdaMF-MAT | 验证跨数据集与跨 expert quality 的一致性 | secondary replication |

MKG-W 的方法结构、特征、alpha/beta/fallback 搜索空间先在 grouped DEV cross-fit 中确定，随后在完整 DEV 上拟合并锁定，最后应用一次 TEST。DB15K TEST 在此前的 score-ensemble 工作中已被访问，因此这里的 DB15K strong-pair 结果不能描述为全新 confirmatory holdout，但仍可作为固定方法的辅助复现。

本轮没有使用 TEST 调整 feature set、模型类型、alpha grid、beta grid、fallback threshold grid 或任何 pair-specific 参数。

## 3. 实验资产与训练完成情况

所有正式模型均保留三 seed `best.ckpt`、`config_merged.json` 和训练 metrics。MKG-W 的最终 Structural-v2 三 seed 使用修正后的训练配置；早期被替代的试跑不进入下列表格或后续 full-ranking。

### 3.1 MKG-W 最佳记录 DEV MRR

| 模型 | Seed 1 | Seed 2 | Seed 3 | 正式 run 目录 |
| --- | ---: | ---: | ---: | --- |
| Gate-v2-state | 0.151792 | 0.142424 | 0.146149 | `ml/artifacts/outputs/mkg_w_gate_v2_state/` |
| Structural-v2 | 0.193937 | 0.187823 | 0.191531 | `ml/artifacts/outputs/mkg_w_structural_v2/` |
| M-Hyper | 0.352740 | 0.352830 | 0.352896 | `ml/artifacts/outputs/mkg_w_mhyper/` |
| NativE | 0.328859 | 0.326330 | 0.330140 | `ml/artifacts/outputs/mkg_w_native/` |
| AdaMF-MAT | 0.279470 | 0.297217 | 0.292556 | `ml/artifacts/outputs/mkg_w_adamf_mat/` |

### 3.2 DB15K 最佳记录 DEV MRR

| 模型 | Seed 1 | Seed 2 | Seed 3 | 正式 run 目录 |
| --- | ---: | ---: | ---: | --- |
| M-Hyper | 0.384129 | 0.382037 | 0.381594 | `ml/artifacts/outputs/db15k_mhyper/` |
| NativE | 0.319731 | 0.325735 | 0.317600 | `ml/artifacts/outputs/db15k_native/` |
| AdaMF-MAT | 0.293228 | 0.283881 | 0.298944 | `ml/artifacts/outputs/db15k_adamf_mat/` |

这些训练记录主要用于确认 checkpoints 与 seed 覆盖。论文指标统一采用后续相同协议下重新计算的 exact filtered full-ranking，而不是直接混用训练日志中的评估结果。

## 4. 第一阶段：Gate-v2-state + Structural-v2 受控现象验证

### 4.1 Seed 1 与三 seed 结果

| 设置 | Gate-v2-state | Structural-v2 | Query-zscore 0.5 | Global alpha | Relation alpha | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed 1 DEV | 0.151792 | 0.193937 | 0.224777 | 0.232466 | 0.262453 | 0.278718 |
| Seeds 1/2/3 pooled DEV | 0.146788 | 0.191097 | 0.220225 | 0.229060 | 0.257340 | 0.276074 |

Seed 1 的结论在补齐 seed 2/3 后保持：

- 最强单专家是 Structural-v2，pooled DEV MRR 为 `0.191097`；
- Oracle 相对最强单专家仍有 `+0.084977` MRR headroom；
- Global alpha 相对最强单专家提升 `+0.037963`；
- Relation alpha 的 full-DEV 提升更大，但容量和过拟合风险也更高。

### 4.2 Grouped cross-fit 审计

Global alpha 的 full-DEV 与 five-fold cross-fit MRR 均为 `0.229060`。Relation alpha 从 full-DEV 的 `0.257340` 降至 cross-fit 的 `0.253178`，差值约 `0.004162`，说明高容量 relation-conditioned selection 存在可测的乐观偏差。

以 Gate-v2-state 为表中 expert A 时：

- Global cross-fit delta：`+0.082272`，95% CI `[+0.075840, +0.088704]`；
- Relation cross-fit delta：`+0.106389`，95% CI `[+0.099647, +0.113133]`；
- seed-stripped query 的赢家标签三 seed 全体一致率为 `62.87%`；
- 两两 seed 标签一致率为 `74.89%`。

这一阶段确认了两个事实：query-dependent complementarity 并非单 seed 偶然现象；同时，自由度较高的条件策略需要严格 cross-fit 和低容量约束。

相关产物位于 `outputs/mkg_w/heterogeneous_complementarity/gatev2_state_structuralv2_seed123/`。此前 cross-fit 输出中的硬编码 expert 标签已修复并重新生成，本报告只使用修复后的输出。

## 5. 第二阶段：Strong expert pair 的 DEV 现象验证

### 5.1 完整 DEV full-ranking

| 数据集 | Pair | M-Hyper | Expert B | Global | Relation | Oracle | Oracle vs. best expert |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W | M-Hyper + NativE | 0.352822 | 0.328443 | 0.356356 | 0.364970 | 0.391219 | +0.038397 |
| MKG-W | M-Hyper + AdaMF-MAT | 0.352822 | 0.289747 | 0.352822 | 0.354029 | 0.382418 | +0.029596 |
| DB15K | M-Hyper + NativE | 0.382587 | 0.321022 | 0.382587 | 0.385761 | 0.418821 | +0.036234 |
| DB15K | M-Hyper + AdaMF-MAT | 0.382587 | 0.292018 | 0.382587 | 0.383904 | 0.414467 | +0.031880 |

四个 pair 都保留约 `0.030–0.038` MRR 的 answer-aware Oracle headroom。只有 MKG-W/NativE 的 Global alpha 在完整 DEV 上明显优于 M-Hyper；其余三个 pair 的最优 Global anchor 均为 `alpha0=1.0`，即静态策略选择完全依赖 M-Hyper。这个分布为后续方法提供了有价值的压力测试：动态方法必须只在有可靠信号时引入较弱 expert。

## 6. 第三阶段：Anchored Dynamic 与 P3 消融

Anchored Dynamic 使用：

`alpha(q) = clip(alpha0 + beta * tanh(g(phi(q))), 0, 1)`

其中 `alpha0` 是 DEV 选择的 Global anchor；`phi(q)` 是 13 个 answer-agnostic score-geometry 特征；`g` 为 median imputation、standard scaling 和 class-balanced logistic regression。训练标签为 expert A 的 reciprocal rank 严格大于 expert B，ties 不参与拟合。连续输出舍入到间隔为 `0.05` 的 exact-ranking alpha grid。低置信度或非有限特征回退到 `alpha0`。

Grouped five-fold cross-fit 将同一原始 triple 的所有 seed 和 head/tail 方向放在同一 fold，避免 query identity 跨 fold 泄漏。P3 扩展了 beta 至 `0.50`，并系统比较无锚点 Query-soft、固定 correction range、feature groups 和 fallback threshold。

### 6.1 DEV cross-fit 主结果

| 数据集 / Pair | Global | Query-soft Δ | 初始 Anchored Δ | P3 Expanded Anchored Δ | P3 95% CI | Fallback | Saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W / NativE | 0.355802 | +0.006487 | +0.004209 | +0.006617 | `[+0.005512,+0.007722]` | 9.57% | 18.01% |
| MKG-W / AdaMF-MAT | 0.352822 | -0.007804 | +0.001261 | +0.002382 | `[+0.001694,+0.003070]` | 24.18% | 50.48% |
| DB15K / NativE | 0.382587 | -0.009461 | +0.000606 | +0.001300 | `[+0.000750,+0.001851]` | 5.30% | 51.40% |
| DB15K / AdaMF-MAT | 0.382587 | -0.026513 | +0.000267 | +0.000450 | `[+0.000014,+0.000885]` | 54.32% | 51.59% |

P3 的主要判断如下：

- Expanded Anchored 在 `4/4` 个 DEV pair 上为正，四个 CI 均不跨零；
- 无锚点 Query-soft 只在 MKG-W/NativE 上有效，在较弱 expert pair 上明显失效；
- correction range 增大到 `beta=0.45–0.50` 后，MKG-W/NativE、MKG-W/AdaMF 和 DB15K/NativE 的收益继续增加；
- DB15K/AdaMF 需要较大的 confidence fallback 才能得到小而稳定的正增益；
- `alpha0=1.0` 的 pair 出现约 50% continuous saturation，主要来自上界裁剪，不等于 selector collapse。实际 changed/fallback 指标必须与 saturation 一起解释。

因此最终方法锁定为 P3 Expanded Anchored；初始小 beta 版本保留为 correction-range 消融，Query-soft 保留为 no-anchor 消融，Relation alpha 保留为 secondary diagnostic。

## 7. 第四阶段：完整 DEV lock 与 TEST

### 7.1 正式 TEST 主结果

| 数据集 / Pair | Global | Relation (Δ) | Query-soft (Δ) | Anchored (Δ) | Anchored 95% CI | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W / NativE | 0.364305 | 0.368601 (+0.004296) | **0.370506 (+0.006201)** | 0.370176 (+0.005872) | `[+0.004708,+0.007035]` | 0.401902 |
| MKG-W / AdaMF-MAT | 0.359721 | 0.358281 (-0.001440) | 0.352470 (-0.007251) | **0.361705 (+0.001984)** | `[+0.001370,+0.002598]` | 0.390732 |
| DB15K / NativE | 0.374567 | 0.374329 (-0.000238) | 0.365621 (-0.008946) | **0.375838 (+0.001271)** | `[+0.000768,+0.001775]` | 0.409286 |
| DB15K / AdaMF-MAT | 0.374567 | 0.373398 (-0.001169) | 0.347971 (-0.026595) | **0.375281 (+0.000714)** | `[+0.000339,+0.001089]` | 0.405001 |

置信区间为 original-triple cluster mean 的 paired normal 95% interval；同一 cluster 保留三 seed 和两个预测方向。

MKG-W/NativE 上 Query-soft 名义 MRR 比 Anchored 高 `0.000329`，但 Anchored vs. Query-soft 的 CI 为 `[-0.001228,+0.000569]`，二者没有显著差异。其余三个 pair 上 Anchored 均显著优于 Query-soft。Anchored 相对 Relation 在三个 pair 上显著更好；MKG-W/NativE 的差异 CI 为 `[-0.000042,+0.003193]`，不能声称显著。

### 7.2 三 seed 与方向稳定性

| 数据集 / Pair | Seed 1 Δ | Seed 2 Δ | Seed 3 Δ | Head Δ | Tail Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MKG-W / NativE | +0.006036 | +0.005038 | +0.006541 | +0.003131 | +0.008612 |
| MKG-W / AdaMF-MAT | +0.002057 | +0.001809 | +0.002087 | +0.001494 | +0.002474 |
| DB15K / NativE | +0.001372 | +0.000802 | +0.001640 | +0.001480 | +0.001063 |
| DB15K / AdaMF-MAT | +0.000857 | +0.000648 | +0.000637 | +0.000677 | +0.000751 |

这里的 delta 均为 Anchored 相对同 pair 的 DEV-locked Global alpha。结果达到 `12/12` pair-seed 为正、`8/8` pair-direction 为正。

### 7.3 锁定策略与 DEV/TEST 行为

| 数据集 / Pair | alpha0 | beta | threshold | Fallback DEV/TEST | Saturation DEV/TEST | Changed DEV/TEST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W / NativE | 0.60 | 0.50 | 0.00 | 0.0% / 0.0% | 19.6% / 19.4% | 94.1% / 94.3% |
| MKG-W / AdaMF-MAT | 1.00 | 0.50 | 0.10 | 24.2% / 24.2% | 50.4% / 50.4% | 35.8% / 35.6% |
| DB15K / NativE | 1.00 | 0.50 | 0.00 | 0.0% / 0.0% | 51.4% / 52.2% | 45.4% / 44.4% |
| DB15K / AdaMF-MAT | 1.00 | 0.50 | 0.30 | 67.1% / 67.2% | 51.6% / 52.1% | 7.1% / 7.0% |

DEV 与 TEST 的 fallback、saturation 和 changed rate 基本一致，没有明显 policy behavior shift。DB15K/AdaMF 是最保守的锁：约 67% 查询回退 Global，只有约 7% 查询实际改变 alpha，但仍获得 `+0.000714` MRR 的显著增益。这是 confidence-controlled fallback 发挥安全作用的最直接案例。

## 8. 总体判断

四个 TEST pair 的 Anchored delta 等权描述性均值为 `+0.002460`，Query-soft delta 的等权均值为 `-0.009148`。由于 DB15K 查询数更多且证据等级不同，论文不应把按行数加权的单一均值作为主要结果；主报告单位应保持为 dataset-pair 和对应 CI。

当前证据支持以下结论：

1. 独立训练的 MMKGC experts 即使强弱差距较大，仍存在 query-level Oracle headroom；
2. 自由 Query-soft 的效果高度依赖 expert pair，在弱 expert 条件下会把静态强专家的优势稀释掉；
3. Global anchor、bounded correction 和 confidence fallback 共同提供安全性；
4. Anchored Dynamic 的正收益跨 dataset、pair、seed 和 direction 一致；
5. Anchored 的贡献应表述为 robustness，而不是保证每个 pair 上都取得最高绝对 MRR。

## 9. 工程与复现状态

本轮相关实现提交包括：

- `c983ed3`：修复 cross-fit 输出中的硬编码 expert 标签；
- `f45b778`：加入 Anchored Dynamic grouped cross-fit；
- `5b43015`：加入 P3 Anchored 消融；
- `6c78bdc`：加入 full-DEV lock、TEST apply 和 TEST alpha-grid 导出。

统一汇总由 `scripts/build_anchored_dynamic_four_pair_summary.py` 重建，生成：

- `outputs/anchored_dynamic/four_pair_summary/four_pair_summary.json`；
- `outputs/anchored_dynamic/four_pair_summary/source_manifest.csv`；
- `docs/paper_tables/anchored_dynamic/` 下的主表和附录表；
- `docs/paper_figures/Figure_anchored_dynamic_stability.{pdf,svg,png}`；
- `docs/reports/anchored_dynamic_four_pair_report.md` Methods/Results 英文草稿。

`source_manifest.csv` 记录 20 个关键 P3 summary、DEV lock、DEV/TEST summary 与 TEST query-row 文件的 size 和 SHA256。大型 checkpoints 与 query rows 不进入 Git；Git 只保存可重建脚本、小型汇总、表格、图和报告。

## 10. 接下来的任务

当前不需要继续训练 M-Hyper、NativE 或 AdaMF-MAT，也不应根据四个 TEST 结果重新调 beta、threshold 或 features。下一阶段应转向论文闭环：

1. 将 Methods/Results 草稿并入新版 manuscript；
2. 将 TEST 主表、DEV/TEST stability 表和稳定性图放入正文；
3. 将 per-seed、head/tail、fallback/saturation 表放入附录；
4. 在 Limitations 中明确 DB15K 的 secondary replication 身份；
5. 若需要 OpenBG-IMG 上的 Anchored 结果，应沿用已经冻结的方法配置，把它作为额外 exploratory replication，不再用于方法选择。

当前最关键的研究阶段已经完成：现象、方法、消融、clean TEST 和跨数据集辅助复现均已形成闭环。
