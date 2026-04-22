# 新增实验矩阵（面向 clean routing 结论增强）

## 0. 文档目标

本文档给出一份面向当前结果状态的新增实验矩阵，目标不是“强行把 clean learned router 调成更强”，而是用更严格、更结构化的实验来回答下面三个问题：

1. 当前 clean routing 的弱结果，究竟是 **全局阈值过于粗糙**，还是 **query-time 合法信号本身不足**？
2. 当前 binary threshold label 是否过于粗糙，导致 learned router 学到的概率与最终 MRR 目标不对齐？
3. 在保持 clean legality 的前提下，是否存在比当前 `rule` / `C4 learned` 更合理的 adaptive strategy？

在当前仓库结果下，clean 线已经合法，但最佳 clean learned router 仍未明显超过 clean rule；相反，clean rule 是当前 clean 线最强结果，而 posthoc stronger selector 仍明显存在更大 headroom。因此新增实验的主要目的应该是：

- 验证 clean 线为什么弱
- 确认这种弱是否具有结构性
- 给论文提供更强的 negative / bounded-effect evidence

---

## 1. 当前结果出发点（作为实验矩阵设计依据）

当前结果显示：

- best clean learned ≈ `0.29386`
- clean rule ≈ `0.29428`
- clean `C2/C3/C4` 差异很小
- posthoc `PH_FULL` 明显更强

因此，新实验不应优先继续做“同一个全局阈值上的小调参”，而应优先测试：

- 更结构化的阈值策略
- 更贴近目标的训练目标
- 更强但仍合法的 hybrid clean strategy

---

# 2. 实验矩阵总表

| 编号 | 实验名 | 目的 | 主要变量 | 输出表 | 预期解释 |
|---|---|---|---|---|---|
| E1 | Direction-specific dual threshold | 检查单一全局阈值是否过粗 | `tau_head`, `tau_tail` | T1 | 若优于全局 `tau`，说明 clean routing 受方向异质性限制 |
| E2 | Relation-prior bucket threshold | 检查 relation prior 是否适合分段决策 | `prior_bucket`, `tau_per_bucket` | T2 | 若有增益，说明全局阈值不足，relation-level policy 更合理 |
| E3 | Direction × observed-has-img threshold | 检查 observed-side modality 是否应与方向联动决策 | `direction`, `observed_has_img`, `tau_group` | T3 | 若有增益，说明 observed-side query context 有条件价值 |
| E4 | Dev-calibrated probability routing | 检查当前 learned prob 是否未校准 | calibration method, `tau` | T4 | 若提升，说明主要问题是 calibration 而非 feature 无效 |
| E5 | Regression target routing | 检查二分类 gain label 是否过粗 | target = `delta_rr` | T5 | 若提升，说明 label 设计损失了 gain magnitude 信息 |
| E6 | Ordinal gain bucket routing | 检查 coarse-to-fine gain modeling 是否更合适 | gain bins | T6 | 若提升，说明 routing 应区分“强负/弱负/弱正/强正”而非简单二分类 |
| E7 | Hybrid prior-first adaptive strategy | 检查可解释 hybrid strategy 是否优于 learned C4 | prior gate + learned-on-uncertain-zone | T7 | 若优于 clean rule，说明 learned router 只适合处理中间模糊区 |
| E8 | Confidence-free fallback policy | 检查“不调用 confidence-aware signals”下的稳健 fallback 策略 | conservative fusion policy | T8 | 若稳定，说明 main gain 来自 policy design 而非 classifier capacity |
| E9 | Delta sensitivity under clean routing | 检查结论是否依赖单一 delta=0.01 | `delta ∈ {0.00,0.01,0.02}` | T9 | 若趋势稳定，说明 clean 弱结论不是某个 delta 偶然导致 |
| E10 | Seed-wise significance / CI | 检查 clean gain 是否真实而非噪声 | bootstrap / seed-wise stats | T10 | 若 CI 覆盖 0，说明 clean gain 不可写成强方法增益 |
| E11 | Oracle gap decomposition | 量化 clean 可恢复空间 vs posthoc 可恢复空间 | clean best / posthoc best / oracle | T11 | 用于支撑“deployable gap remains large” |
| E12 | Cost-aware selective activation | 检查低 fusion coverage 下是否仍可获得稳定收益 | utility = MRR - λ·coverage | T12 | 若低 coverage 更优，说明方法更适合写成 conservative activation |

---

# 3. 逐项实验设计

## E1. Direction-specific dual threshold

### 实验名
`clean_dual_tau_by_direction`

### 核心问题
当前全局 `tau` 把 `head` 与 `tail` 两种 query 混在一起，可能过于粗糙。

### 自变量
- `tau_head ∈ {0.3, 0.5, 0.7, 0.9}`
- `tau_tail ∈ {0.3, 0.5, 0.7, 0.9}`
- router model ∈ {`logistic C4`, `xgb C4`}

### 固定项
- clean features = `C4`
- delta = `0.01`

### 实现方式
对同一组 router probabilities：
- 若 `direction=head`，用 `tau_head`
- 若 `direction=tail`，用 `tau_tail`

### 输出表
**T1. Direction-specific Dual Threshold Scan**

列建议：
- model
- feature_set
- delta
- tau_head
- tau_tail
- overall_mrr
- hits1
- hits3
- hits10
- fusion_coverage
- subgroup mrr (`head_has_img`, `head_no_img`, `tail_no_img`)

### 预期解释
- 若优于当前全局最佳 `tau`，说明 clean 线的主要问题之一是 **global threshold too coarse**。
- 若仍几乎不提升，说明 clean signals 的限制是结构性的，不是阈值搜索太粗。

---

## E2. Relation-prior bucket threshold

### 实验名
`clean_tau_by_relation_prior_bucket`

### 核心问题
当前 clean C2/C3/C4 的主要有效信息都来自 relation prior；可能应按 prior bucket 分段决策，而不是统一 `tau`。

### 自变量
- prior bucket 划分方式：
  - B1: `relation_gain_prior < 0`
  - B2: `0 ≤ relation_gain_prior < 0.05`
  - B3: `relation_gain_prior ≥ 0.05`
- 每个 bucket 单独阈值 `tau_b`

### 固定项
- model ∈ {rule-like policy, logistic C4, xgb C4}
- delta = `0.01`

### 输出表
**T2. Prior-Bucket Threshold Policy Results**

列建议：
- bucket definition
- tau_neg
- tau_mid
- tau_pos
- overall_mrr
- fusion_coverage
- gain_precision
- tail_no_img coverage
- head_has_img coverage

### 预期解释
- 若有提升，说明当前 clean learned 没吃到的不是“更复杂特征”，而是 **policy granularity**。
- 若没有提升，说明 relation prior 已经几乎被 clean rule 吃完了。

---

## E3. Direction × observed-has-img threshold

### 实验名
`clean_tau_by_direction_and_observed_img`

### 核心问题
query-time observed-side modality availability 是否应该和 direction 组合起来决定阈值？

### 自变量
分组：
- `(head, observed_has_img=0)`
- `(head, observed_has_img=1)`
- `(tail, observed_has_img=0)`
- `(tail, observed_has_img=1)`

每组单独阈值 `tau_group`

### 输出表
**T3. Group-Conditional Threshold Policy**

列建议：
- group
- tau
- n_queries
- mrr
- fusion_coverage
- gain_precision
- overall aggregated mrr

### 预期解释
- 若提升，说明 observed-side information 的确有用，但需要 conditional policy 才能释放。
- 若不提升，说明 clean observed-side modality 信号本身很弱。

---

## E4. Dev-calibrated probability routing

### 实验名
`clean_prob_calibration`

### 核心问题
当前 clean model 的 probability 可能未校准，使得 threshold scan 不能真实反映 learned score 的可用性。

### 自变量
- calibration ∈ {none, Platt, isotonic}
- router ∈ {logistic C4, xgb C4}
- tau scan ∈ {0.1,0.3,0.5,0.7,0.9}

### 实现方式
- 在 dev 上训练 clean router
- 用 dev 再做 calibration
- 在 test 上扫 calibrated probabilities

### 输出表
**T4. Calibrated vs Uncalibrated Clean Routing**

列建议：
- model
- calibration
- best tau
- best mrr
- best coverage
- delta to uncalibrated best

### 预期解释
- 若 calibration 后提升明显，说明 clean 线的问题部分来自 **probability miscalibration**。
- 若仍无提升，说明问题更深，在 feature / target design 上。

---

## E5. Regression target routing

### 实验名
`clean_delta_rr_regression_router`

### 核心问题
把 routing 目标从“是否大于阈值”改成直接预测 `delta_rr`，是否更符合最终目标？

### 自变量
- target = `delta_rr = rr_gate - rr_residual`
- model ∈ {lightgbm regressor / xgb regressor / linear regressor}
- routing policy = select fusion iff predicted `delta_rr > θ`

### 输出表
**T5. Regression-Based Routing Results**

列建议：
- regressor type
- theta
- overall_mrr
- fusion_coverage
- gain_precision
- best theta

### 预期解释
- 若优于 binary C4，说明原来 label 设计过粗，丢失了 gain magnitude。
- 若不优于 binary C4，说明 clean 可观测信息本身不足，不是 supervision form 的问题。

---

## E6. Ordinal gain bucket routing

### 实验名
`clean_ordinal_gain_router`

### 核心问题
是否应该区分：
- strong negative
n- weak negative
- weak positive
- strong positive

而不是只用一个二分类标签？

### 自变量
- bucket strategy，例如：
  - `delta_rr < -0.01`
  - `-0.01 ≤ delta_rr < 0`
  - `0 ≤ delta_rr < 0.01`
  - `delta_rr ≥ 0.01`
- model ∈ {ordinal classifier / multiclass xgb}
- decision rule: 仅 strong positive 开 fusion 或 weak+strong positive 开 fusion

### 输出表
**T6. Ordinal Gain Modeling Results**

### 预期解释
- 若优于 binary router，说明 clean routing 的主要瓶颈之一是 label granularity。
- 若不优于 binary router，说明更细标签也救不了 clean signal scarcity。

---

## E7. Hybrid prior-first adaptive strategy

### 实验名
`clean_hybrid_prior_first_policy`

### 核心问题
当前最合理的 clean adaptive strategy 也许不是“全量 learned router”，而是：

- 先用 relation prior 做粗筛
- 只在中间模糊区交给 learned router

### 自变量
- prior negative cutoff `a`
- prior positive cutoff `b`
- uncertain zone = `(a, b)`
- uncertain zone router ∈ {logistic C4, xgb C4}

### 策略模板
- if `relation_gain_prior <= a`: force residual
- elif `relation_gain_prior >= b` and `observed_has_img=1`: allow learned decision or direct fusion
- else: call learned router

### 输出表
**T7. Hybrid Prior-First Adaptive Policy**

列建议：
- a
- b
- uncertain-zone model
- overall_mrr
- fusion_coverage
- gain_precision
- fraction handled by direct rule
- fraction handled by learned router

### 预期解释
- 若优于 clean rule，说明 learned router 主要适合处理中间模糊区，而不是全局替代 rule。
- 若不优于 clean rule，说明 clean rule 已几乎吃完 deployable information。

---

## E8. Confidence-free fallback policy

### 实验名
`clean_conservative_fallback_policy`

### 核心问题
在完全不使用任何 confidence-aware / posthoc-style 特征的前提下，是否存在更稳的保守策略？

### 策略示例
- 仅当：
  - `relation_is_visual_prior=1`
  - `observed_has_img=1`
  - `relation_support >= s_min`
  时才允许 fusion
- 否则 residual

### 自变量
- `s_min ∈ {10, 30, 50, 100}`
- 是否要求 `observed_text_img_cosine > c_min`
- `c_min ∈ {0.0, 0.05, 0.1}`

### 输出表
**T8. Conservative Clean Fallback Policy**

### 预期解释
- 若能在更低 coverage 下稳定接近或超过 clean rule，说明论文可把方法写成 **conservative activation** 而非 aggressive routing。
- 若仍无增益，说明 clean observable signals 的可利用性极弱。

---

## E9. Delta sensitivity under clean routing

### 实验名
`clean_delta_sensitivity`

### 核心问题
当前结论是否只是在 `delta=0.01` 下成立？

### 自变量
- `delta ∈ {0.00, 0.01, 0.02}`
- model ∈ {rule, logistic C4, xgb C4}
- best tau per delta

### 输出表
**T9. Clean Delta Sensitivity Table**

列建议：
- delta
- model
- best tau
- best mrr
- best coverage
- vs residual-only gain

### 预期解释
- 若三个 delta 下都只能带来弱提升，则 clean 弱结论更稳。
- 若只有 delta 改了结果就翻盘，说明当前结论对 label threshold 很敏感，需要谨慎表述。

---

## E10. Seed-wise significance / CI

### 实验名
`clean_significance_test`

### 核心问题
当前 clean best 相比 residual-only / clean rule 的差异很小，是否只是噪声？

### 自变量
- compared pairs:
  - clean best learned vs residual-only
  - clean best learned vs clean rule
  - clean rule vs residual-only
- method ∈ {seed-wise mean±std, paired bootstrap CI}

### 输出表
**T10. Statistical Significance of Clean Routing Gains**

列建议：
- comparison
- mean delta MRR
- 95% CI
- p-value (optional)

### 预期解释
- 若 CI 覆盖 0，则不能在主文里把 clean gains 写成 strong method improvement。
- 若 clean rule vs residual-only 显著，而 learned vs rule 不显著，则更支持“simple policy already saturates deployable gain”。

---

## E11. Oracle gap decomposition

### 实验名
`oracle_gap_decomposition`

### 核心问题
clean 与 posthoc 到底各自恢复了 oracle headroom 的多少比例？

### 输入结果
- residual-only
- clean best
- posthoc best (建议使用 `PH_FULL`)
- oracle

### 指标定义
- `recoverable_gap_clean = (clean_best - residual_only) / (oracle - residual_only)`
- `recoverable_gap_posthoc = (posthoc_best - residual_only) / (oracle - residual_only)`

### 输出表
**T11. Oracle Gap Recovery Table**

### 预期解释
- clean 仅恢复很小比例，而 posthoc 恢复明显更多时，可直接支撑：
  **deployable routing gap remains large**。

---

## E12. Cost-aware selective activation

### 实验名
`cost_aware_clean_routing`

### 核心问题
若 fusion 有系统开销，那么在低 coverage 下是否存在更合理的 operating point？

### 自变量
定义 utility：
- `U = MRR - λ * fusion_coverage`

其中：
- `λ ∈ {0, 0.01, 0.02, 0.05}`

比较对象：
- clean rule
- clean logistic C4
- clean xgb C4
- hybrid policy（若已完成 E7）

### 输出表
**T12. Cost-Aware Operating Point Analysis**

列建议：
- model
- tau
- lambda
- mrr
- coverage
- utility
- best operating point per lambda

### 预期解释
- 若低 coverage 操作点更优，说明方法更适合写成 conservative selective activation，而不是 maximize-accuracy routing。

---

# 4. 推荐执行优先级

## 第一优先级（最值得马上做）

1. **E1 Direction-specific dual threshold**
2. **E2 Relation-prior bucket threshold**
3. **E7 Hybrid prior-first adaptive strategy**
4. **E10 Seed-wise significance / CI**

原因：
- 这四项最直接回答“clean 为什么弱、rule 为什么反而更强”
- 且基本不需要改动大模型，只需要在现有输出上做策略层实验

## 第二优先级（若第一优先级没有翻盘，再补强结论）

5. **E4 Dev-calibrated probability routing**
6. **E9 Delta sensitivity**
7. **E11 Oracle gap decomposition**

原因：
- 这组能帮助你把结论写得更稳，尤其适合论文 Discussion / Limitations

## 第三优先级（需要更多实现改动）

8. **E5 Regression target routing**
9. **E6 Ordinal gain bucket routing**
10. **E8 Conservative fallback policy**
11. **E12 Cost-aware selective activation**

原因：
- 这组更像第二轮方法探索，适合在你确认第一轮结论后再做

---

# 5. 建议最终论文里最可能形成的结论分支

## 分支 A：结构化 clean policy 有小幅提升，但仍远低于 oracle/posthoc

则论文结论可写成：

- bounded multimodal gain can be exploited slightly under legal query-time policies
- but most recoverable headroom remains inaccessible to deployable routing

## 分支 B：无论怎么改 clean policy，提升都极小

则论文结论可写成：

- the limitation is structural rather than implementation-specific
- clean query-observable signals are insufficient to recover most of the selective gain

## 分支 C：只有 posthoc / confidence-rich selector 明显有效

则论文结论可写成：

- strong offline separability exists
- but it does not translate to a realistic deployable router under clean feature constraints

---

# 6. 一句话建议

如果你现在只准备做一轮最有价值的补充实验，那么最推荐的组合是：

> **E1 + E2 + E7 + E10**

因为这四项能最有效回答：

- 当前 clean learned 为什么没压过 rule
- 更细粒度 adaptive strategy 是否真的有用
- 如果仍没用，这个“没用”是不是稳健结论而不是偶然结果
