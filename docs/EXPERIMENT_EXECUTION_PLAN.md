# 执行版实验计划（新增实验矩阵收束版）

## 0. 文档目标

本文档将 `docs/ADDITIONAL_EXPERIMENT_MATRIX.md` 中的 12 个新增实验进一步收束成一份可执行、可止损、可分阶段推进的实验计划。

本计划重点回答三个实际问题：

1. **先跑哪个？**
2. **每个实验具体用什么脚本、输入什么、输出什么？**
3. **做到什么程度继续，做不到什么程度就停？**

当前仓库结果已经显示：

- clean 线合法，但 best clean learned 没有明显超过 clean rule；
- clean `C2/C3/C4` 差异很小；
- posthoc stronger selector 仍然有明显 headroom。

因此，本计划不再把目标设定为“必须把 clean learned router 调成显著更强”，而是将目标设定为：

> 用最少但最有价值的新增实验，判断 clean routing 的弱结果究竟是阈值策略过粗、监督目标不合适，还是 deployable query-time signals 本身不足。

---

# 1. 总体推进原则

## 1.1 三阶段推进

本计划分成三阶段：

### 阶段 A：最低成本验证（优先跑）

目标：

- 优先检验“是不是当前策略太粗”
- 尽量复用现有 clean probabilities / eval json / threshold scan
- 尽量不重训模型

对应实验：

- **E1** Direction-specific dual threshold
- **E2** Relation-prior bucket threshold
- **E10** Seed-wise significance / CI

### 阶段 B：中成本策略增强

目标：

- 如果阶段 A 证明结构化策略有希望，再测试 hybrid adaptive strategy
- 仍尽量避免改 router 主训练链路

对应实验：

- **E7** Hybrid prior-first adaptive strategy
- **E12** Cost-aware selective activation
- **E9** Delta sensitivity

### 阶段 C：高成本 supervision / target redesign

目标：

- 若阶段 A/B 仍无法给出清晰结论，再测试当前 binary gain label 是否过粗
- 这阶段才开始改训练目标

对应实验：

- **E4** Dev-calibrated probability routing
- **E5** Regression target routing
- **E6** Ordinal gain bucket routing
- **E8** Conservative fallback policy
- **E11** Oracle gap decomposition（可与任何阶段同步）

---

## 1.2 止损原则

定义几个统一阈值，避免一直无止境调实验。

### 继续阈值（Go）

若某个新实验相对 **clean rule** 或 **best clean learned** 满足任一条件，则进入下一阶段：

1. `overall_mrr` 提升 **≥ 0.0015**
2. `overall_mrr` 提升 **≥ 0.0010** 且 `fusion_coverage` 明显更低（绝对下降 ≥ 0.05）
3. 某关键 subgroup（尤其 `head_has_img`）提升 **≥ 0.0030** 且 overall 不下降超过 `0.0005`

### 停止阈值（Stop）

若新增实验满足以下情况，则不再沿该方向继续深挖：

1. 相对 clean rule / best clean learned，`overall_mrr` 提升 **< 0.0008**
2. 提升主要来自覆盖率几乎降到 0，而整体解释意义不大
3. 统计区间或 seed-wise 差异显示增益不稳（如 CI 覆盖 0）

### 写作转向阈值（Write-up pivot）

若阶段 A + B 全部结束后仍满足：

- best new clean policy 相对 clean rule 的提升 **< 0.0015**
- 且相对 oracle gap 的恢复比例仍很低

则论文主结论直接转向：

> clean query-time observable signals provide only weak routing utility under the current protocol, and most recoverable headroom remains inaccessible without post-hoc information.

---

# 2. 先跑哪个：推荐执行顺序

## Phase A（必须先跑）

### A1. E1 — Direction-specific dual threshold
### A2. E2 — Relation-prior bucket threshold
### A3. E10 — Seed-wise significance / CI

跑完这三项后做一次判断：

- 若 E1/E2 任一项超过继续阈值，则进入 Phase B
- 若 E1/E2 都没有形成可解释提升，但 E10 证明 clean rule 本身相对 residual-only 稳定，则可直接收束论文主结论
- 若 E10 表明当前 clean rule 增益也不稳，则主文要进一步弱化“方法有效性”，转而强调 bounded gain diagnosis 与 deployable gap

## Phase B（只有 Phase A 有希望才继续）

### B1. E7 — Hybrid prior-first adaptive strategy
### B2. E12 — Cost-aware selective activation
### B3. E9 — Delta sensitivity under clean routing

跑完后再判断：

- 若 E7 提升明显，则 hybrid strategy 成为 clean 主线 strongest result
- 若 E7 无明显提升，则停止继续发明更复杂 clean policy
- E12 和 E9 主要用于强化写作，不用于强行“翻盘”

## Phase C（只有当你仍想验证 supervision 形式时再做）

### C1. E4 — Dev-calibrated probability routing
### C2. E5 — Regression target routing
### C3. E6 — Ordinal gain bucket routing
### C4. E8 — Conservative fallback policy
### C5. E11 — Oracle gap decomposition

这一阶段的定位是：

- 不是必须项
- 而是当你需要向导师/审稿人证明“我们已经检查过 supervision / calibration 也无法显著救 clean line”时再做

---

# 3. Phase A 详细执行计划

## A1. E1 — Direction-specific dual threshold

### 目的
测试全局单阈值 `tau` 是否过于粗糙。

### 建议新增脚本
`scripts/run_clean_dual_threshold_scan.py`

### 输入
- clean router predictions（建议来自统一 prediction file）
  - `outputs/router/eval/clean/router_predictions_clean_logistic_delta_0.01_C4.csv`
  - `outputs/router/eval/clean/router_predictions_clean_xgb_delta_0.01_C4.csv`
- clean eval targets / routed rows 所需 metadata
  - 可直接复用 `rr_gate`, `rr_residual`, `target_regime`, `direction`

### 若当前还没有统一 prediction file
先补一个轻量脚本：
- `scripts/run_router_predictions.py`

其输出：
- `outputs/router/eval/clean/router_predictions_clean_<model>_delta_0.01_C4.csv`

列至少应包含：
- `query_id`
- `router_prob`
- `direction`
- `target_regime`
- `rr_gate`
- `rr_residual`
- `seed`

### 核心变量
- `tau_head ∈ {0.3, 0.5, 0.7, 0.9}`
- `tau_tail ∈ {0.3, 0.5, 0.7, 0.9}`

### 输出文件
- `outputs/router/eval/clean/dual_threshold_scan_clean_logistic_delta_0.01_C4.csv`
- `outputs/router/eval/clean/dual_threshold_scan_clean_xgb_delta_0.01_C4.csv`

### 输出表（论文侧）
**T1. Direction-Specific Dual Threshold Scan**

### 继续 / 停止阈值
- 若相对 clean rule (`0.2942819`) 提升 **≥ 0.0015**，继续到 Phase B
- 若相对 best clean learned (`0.2938638`) 提升 **≥ 0.0010** 且 `head_has_img` 提升 ≥ `0.003`，继续到 Phase B
- 若最佳组合提升 **< 0.0008**，停止继续深挖“双阈值方向”

### 预期解释
- 有提升：说明 clean routing 受 direction heterogeneity 限制，global tau 太粗
- 无提升：说明 clean observable signals 的限制比阈值设计更根本

---

## A2. E2 — Relation-prior bucket threshold

### 目的
测试 relation prior 是否更适合做分段策略，而不是统一阈值。

### 建议新增脚本
`scripts/run_clean_prior_bucket_policy.py`

### 输入
- clean router predictions（同 A1）
- relation prior csv
  - `outputs/router/priors/relation_gain_stats_gamma_0.000.csv`

### 核心变量
推荐先用 3 桶：
- `neg`: `relation_gain_prior < 0`
- `mid`: `0 ≤ relation_gain_prior < 0.05`
- `pos`: `relation_gain_prior ≥ 0.05`

每桶单独阈值：
- `tau_neg ∈ {0.5, 0.7, 0.9, force_residual}`
- `tau_mid ∈ {0.3, 0.5, 0.7, 0.9}`
- `tau_pos ∈ {0.3, 0.5, 0.7, 0.9, force_fusion}`

### 输出文件
- `outputs/router/eval/clean/prior_bucket_policy_clean_logistic_delta_0.01_C4.csv`
- `outputs/router/eval/clean/prior_bucket_policy_clean_xgb_delta_0.01_C4.csv`

### 输出表
**T2. Prior-Bucket Threshold Policy Results**

### 继续 / 停止阈值
- 若比 clean rule 提升 **≥ 0.0015**，继续到 B1（Hybrid）
- 若比 best clean learned 提升 **≥ 0.0010** 且 coverage 下降 ≥ `0.05`，继续
- 若最佳 prior-bucket policy 仍 **≤ clean rule + 0.0008**，停止继续做 relation-structure 更复杂的 bucket 扩展

### 预期解释
- 有提升：说明当前 learned router 没学到 policy granularity，但 relation-level structure 确实重要
- 无提升：说明 clean rule 已经接近 saturate deployable relation prior information

---

## A3. E10 — Seed-wise significance / CI

### 目的
验证当前 clean gain 到底稳不稳。

### 建议新增脚本
`scripts/run_router_significance.py`

### 输入
至少比较三组：
- clean best learned vs residual-only
- clean rule vs residual-only
- clean best learned vs clean rule

### 输入文件
- routed rows for clean best learned
- routed rows for clean rule
- baseline residual-only rows

若当前还没有统一 routed rows 文件，建议新增：
- `outputs/router/eval/clean/routed_clean_<model>_delta_0.01_tau_<tau>_<feature>.csv`

### 输出文件
- `outputs/router/eval/clean/significance_clean_best_vs_residual.json`
- `outputs/router/eval/clean/significance_clean_rule_vs_residual.json`
- `outputs/router/eval/clean/significance_clean_best_vs_rule.json`

### 输出表
**T10. Statistical Significance of Clean Routing Gains**

### 继续 / 停止阈值
- 若 clean rule vs residual-only 的 CI 不覆盖 0，但 clean best vs rule 覆盖 0：
  - 继续写论文，但停止继续“让 learned 压过 rule”的调参执念
- 若 clean rule vs residual-only 的 CI 也覆盖 0：
  - clean 方法主张进一步收缩，主文只保留 diagnosis + weak/unstable routing evidence

### 预期解释
- 这是全计划中最重要的“止损实验”之一
- 它决定你后面是继续方法优化，还是转入写作收束

---

# 4. Phase B 详细执行计划

## B1. E7 — Hybrid prior-first adaptive strategy

### 进入条件
只有当 A1 或 A2 至少有一项达到继续阈值时才做。

### 目的
检验 learned router 是否只适合处理中间模糊区，而不适合全局决策。

### 建议新增脚本
`scripts/run_clean_hybrid_prior_first.py`

### 策略模板
- if `relation_gain_prior <= a`: residual
- elif `relation_gain_prior >= b` and `observed_has_img=1`: fusion or low-threshold learned
- else: call learned router

### 自变量
- `a ∈ {-0.02, 0.0}`
- `b ∈ {0.02, 0.05, 0.08}`
- uncertain zone router ∈ {`logistic C4`, `xgb C4`}

### 输入
- clean prediction file
- relation prior csv
- clean feature table（若需要 `observed_has_img`）

### 输出文件
- `outputs/router/eval/clean/hybrid_prior_first_clean_logistic_delta_0.01.csv`
- `outputs/router/eval/clean/hybrid_prior_first_clean_xgb_delta_0.01.csv`

### 输出表
**T7. Hybrid Prior-First Adaptive Policy**

### 继续 / 停止阈值
- 若 best hybrid 比 clean rule 提升 **≥ 0.0015**，则把 hybrid 作为 clean mainline strongest candidate
- 若 best hybrid 提升 **< 0.0010**，则停止继续发明更复杂 clean policy

### 预期解释
- 有提升：说明 learned router 的价值在于处理中间模糊区，而不是全局替代 rule
- 无提升：说明 clean rule 已近似吃完 deployable information

---

## B2. E12 — Cost-aware selective activation

### 进入条件
B1 完成后，无论是否有提升都可做，因为它更偏写作支持。

### 目的
测试在低 coverage 场景下，是否存在更好的保守 operating point。

### 建议新增脚本
`scripts/run_cost_aware_clean_policy.py`

### 输入
- best clean candidate policies（rule / best learned / hybrid）
- routed rows 或 threshold scan results

### 变量
- `λ ∈ {0, 0.01, 0.02, 0.05}`
- utility = `MRR - λ * fusion_coverage`

### 输出文件
- `outputs/router/eval/clean/cost_aware_clean_policy.csv`

### 输出表
**T12. Cost-Aware Operating Point Analysis**

### 继续 / 停止阈值
- 若低 coverage policy 在多个 λ 下 consistently 最优，则继续保留“conservative activation”叙事
- 若 utility 排序与 raw MRR 完全一致，则 E12 不再深挖

### 预期解释
- 该实验不要求数值翻盘，主要用于让论文的系统视角更完整

---

## B3. E9 — Delta sensitivity under clean routing

### 进入条件
建议在 B1/B2 后统一做。

### 目的
检查 clean 弱结论是否只依赖于 `delta=0.01`。

### 建议新增脚本
`scripts/run_clean_delta_sensitivity.py`

### 输入
- clean train tables:
  - `outputs/router/features/router_train_dev_clean_delta_0.00.csv`
  - `outputs/router/features/router_train_dev_clean_delta_0.01.csv`
  - `outputs/router/features/router_train_dev_clean_delta_0.02.csv`
- clean test features

### 输出文件
- `outputs/router/eval/clean/delta_sensitivity_clean.csv`

### 输出表
**T9. Clean Delta Sensitivity Table**

### 继续 / 停止阈值
- 若三个 delta 下 strongest clean policy 结论一致，则停止再调 delta
- 若 delta 改变导致 strongest policy 翻盘且差距 > `0.002`，则需要在论文中单独讨论 label sensitivity

### 预期解释
- 该实验主要服务于论文稳健性，而不是方法翻盘

---

# 5. Phase C 详细执行计划

## C1. E4 — Dev-calibrated probability routing

### 进入条件
只有当你认为 Phase A/B 显示“clean learned 可能只是 calibration 不好”时再做。

### 建议新增脚本
`scripts/run_clean_probability_calibration.py`

### 输入
- dev prediction probabilities
- dev labels
- test prediction probabilities

### 方法
- calibration ∈ {Platt, isotonic}

### 输出文件
- `outputs/router/eval/clean/calibration_scan_clean.csv`

### 输出表
**T4. Calibrated vs Uncalibrated Clean Routing**

### 继续 / 停止阈值
- 若 calibration 后增益 **≥ 0.0015**，可以继续做更细 calibration
- 若增益 **< 0.0008**，停止 calibration 方向

---

## C2. E5 — Regression target routing

### 建议新增脚本
`scripts/train_clean_regression_router.py`
`scripts/eval_clean_regression_router.py`

### 输入
- clean train features
- regression target = `delta_rr`

### 输出文件
- `outputs/router/models/clean_regression/...`
- `outputs/router/eval/clean/regression_router_scan.csv`

### 输出表
**T5. Regression-Based Routing Results**

### 继续 / 停止阈值
- 若明显超过 clean rule（≥ `0.0015`），可继续深入
- 若仍不超过 clean rule + `0.0008`，停止 regression 方向

---

## C3. E6 — Ordinal gain bucket routing

### 建议新增脚本
`scripts/train_clean_ordinal_router.py`
`scripts/eval_clean_ordinal_router.py`

### 输出表
**T6. Ordinal Gain Modeling Results**

### 继续 / 停止阈值
与 E5 相同。

---

## C4. E8 — Conservative fallback policy

### 建议新增脚本
`scripts/run_clean_conservative_policy.py`

### 输入
- relation prior
- observed-side flags

### 核心变量
- `support_min`
- `cosine_min`

### 输出表
**T8. Conservative Clean Fallback Policy**

### 继续 / 停止阈值
- 若在明显更低 coverage 下逼近或超过 clean rule，则保留
- 否则停止

---

## C5. E11 — Oracle gap decomposition

### 说明
这个实验成本极低，可以任何阶段同步做。

### 建议新增脚本
`scripts/run_oracle_gap_decomposition.py`

### 输入
- residual-only
- clean best
- posthoc best
- oracle

### 输出文件
- `outputs/router/eval/oracle_gap_decomposition.json`
- `outputs/router/eval/oracle_gap_decomposition.csv`

### 输出表
**T11. Oracle Gap Recovery Table**

### 继续 / 停止阈值
- 无停止条件，直接作为论文 Discussion / Limitations 的稳定补充

---

# 6. 推荐最小执行集（只做一轮时）

如果你只准备再做一轮最有价值的补充实验，建议只做：

## 最小执行集
1. **A1 / E1** Dual threshold by direction
2. **A2 / E2** Prior-bucket threshold policy
3. **A3 / E10** Significance / CI
4. **B1 / E7** Hybrid prior-first strategy（只有 A1/A2 有希望时才做）

### 这个最小执行集的决策逻辑

- 若 A1/A2 没有任何一项相对 clean rule 提升到 `+0.0015`：
  - 不做 B1
  - 直接收束论文主结论
- 若 A1/A2 有至少一项超过 `+0.0015`：
  - 做 B1
  - 若 B1 仍只小幅波动，则也停止继续复杂化 clean policy

---

# 7. 最终写作转向规则

## 结论分支 A：有小幅但稳定 clean 提升

满足：
- best new clean policy ≥ clean rule + `0.0015`
- 且 E10 显示增益稳定

则写作方向：
- clean routing has limited but real value
- structured policy outperforms naive global thresholding
- most oracle headroom still remains unrecovered

## 结论分支 B：clean 无法形成稳定显著提升

满足：
- A1/A2/B1 全部未达阈值
- 或 E10 表明 CI 覆盖 0

则写作方向：
- the weakness is structural rather than implementation-specific
- deployable query-time signals are insufficient to recover most selective multimodal gains
- posthoc separability remains much stronger than deployable separability

---

# 8. 一句话总结

这份执行版实验计划的真正目标不是“继续把 clean learned router 调强”，而是：

> 用最低成本、最高信息量的结构化实验先判断 clean routing 的弱结果到底是不是策略设计问题；如果不是，就尽快止损，把论文主结论收束到一个更稳、更诚实、也更有说服力的方向上。
