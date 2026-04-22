# 基于新一轮 clean-routing 结果的论文改写指令单

## 0. 文档目的

本文档用于指导当前论文在引入新一轮补充实验结果后，如何系统重写与收束主文叙事。

这轮新结果带来的核心变化是：

- 旧版结论是：**naive clean learned router 不如 clean rule，global-threshold clean routing 很弱**。
- 新版结论应改为：
  1. **naive global-threshold clean router 确实不强**；
  2. 但 **clean routing 并未到头**；
  3. 一旦采用 **direction-specific structured policy** 或 **更贴近目标的 supervision form（如 regression / ordinal gain modeling）**，clean 线可以稳定超过当前 clean rule；
  4. 即便如此，clean strongest result 仍与 Oracle 保持明显差距，因此 **deployable gap remains large**。

因此，论文不能再写成：

> clean query-time signals are almost too weak to support meaningful routing.

而应改写为：

> the main bottleneck of clean routing is not only signal scarcity, but also the coarseness of policy granularity and supervision form.

---

# 1. 需要整体替换的主叙事

## 1.1 旧叙事（需要弱化/替换）

以下叙事不再适合作为主文 strongest claim：

1. clean query-time observable signals are largely insufficient
2. learned clean router fails, therefore deployable routing is mostly ineffective
3. clean result is primarily a negative finding

这些表述在旧结果下是合理的，但在新结果下会低估当前贡献。

## 1.2 新叙事（改为主线）

新版主线应改为四层：

### Layer 1 — Diagnosis still holds

- protocol-defined role–modality asymmetry remains valid
- multimodal gain is conditional rather than uniform
- Oracle headroom remains large

### Layer 2 — Naive clean routing is insufficient

- global-threshold clean learned router is not strong
- a simple clean legal rule is more stable than a naive global learned router

### Layer 3 — Structured clean routing is effective

- direction-specific thresholding clearly improves over global-threshold routing
- regression-style routing targets further improve clean performance
- ordinal gain modeling also helps, though less than regression

### Layer 4 — Remaining deployable gap

- even the strongest clean strategy remains far below Oracle
- thus the problem is only partially recoverable under deployable clean constraints

---

# 2. Abstract 改写指令

## 2.1 Abstract 必须修改的三处

### 第一处：Problem statement

原本若写成类似：

> Under the current protocol, selective multimodal activation is difficult because clean query-time signals are too weak.

需要改成：

> Under the current protocol, selective multimodal activation is difficult not only because multimodal gain is bounded and asymmetric, but also because naive clean routing formulations based on a single global threshold are too coarse to capture the underlying decision structure.

### 第二处：Method statement

不要只写：

> We evaluate a clean query-time router.

要改成分层表述：

> We first examine naive clean routing with global-threshold learned selectors, and then evaluate stronger clean formulations based on direction-specific structured thresholding and more target-aligned supervision such as regression-style gain prediction.

### 第三处：Result statement

不要继续写成：

> clean routing remains weak.

改成：

> While naive global-threshold clean routers do not outperform a simple legal rule, structured clean policies and finer-grained gain modeling consistently improve over that rule, showing that the bottleneck lies not only in deployable signal availability but also in policy granularity and supervision form.

同时结尾必须保留约束：

> However, the best clean strategy still remains far below Oracle, indicating that a substantial deployable gap persists.

## 2.2 建议的 Abstract 结构

建议用四句式：

1. 背景：protocol asymmetry + bounded gain
2. 问题：naive clean routing is too coarse
3. 结果：structured thresholds / regression improve over clean rule
4. 边界：still far from Oracle

## 2.3 可直接替换的 Abstract 结论句模板

可在摘要末尾使用类似句子：

> These results show that the weakness of clean routing is not solely due to insufficient query-time observable signals. Instead, it also arises from coarse policy design and overly simplified binary gain supervision. Structured clean routing can recover additional deployable gains, although a substantial gap to Oracle remains.

---

# 3. Introduction 中 contribution paragraph 改写指令

## 3.1 必须废弃的旧版 contribution 风格

不要再把 contribution 写成：

- We show that clean routing is weak.
- We propose a query-time router but it remains limited.

这会低估你现在的结果。

## 3.2 新 contribution paragraph 应拆成 4 点

### Contribution 1 — Protocol diagnosis

> We provide a protocol-aware diagnosis showing that under the current OpenBG-IMG setup, multimodal gain is conditional, target-position-dependent, and entangled with role-specific modality availability.

### Contribution 2 — Negative result on naive clean routing

> We show that naive clean query-time routing based on a single global threshold is insufficient and can even underperform a simple legal rule-based selector, indicating that deployable routing cannot be characterized adequately by a coarse global decision boundary.

### Contribution 3 — Positive result on structured clean routing

> We demonstrate that clean routing is not fundamentally exhausted: direction-specific thresholding and more target-aligned supervision, especially regression-style gain modeling, consistently outperform the clean rule baseline and recover additional deployable gains.

### Contribution 4 — Remaining gap

> We further show that even these stronger clean strategies remain substantially below Oracle, revealing a persistent gap between deployable separability and oracle-level or post-hoc separability.

## 3.3 Introduction 末尾的论文结构说明要同步更新

在 paper organization 里，Method/Experiments 的描述要从：

- clean router evaluation

改成：

- comparison between naive global clean routing and structured clean formulations
- analysis of policy granularity, supervision form, and remaining oracle gap

---

# 4. Method 部分改写指令

## 4.1 Method 的总标题不应只剩“gain-threshold routing”

如果当前方法标题过于强调单一 gain-threshold router，建议改成更宽的标题，例如：

### 推荐标题 A
`Method: Clean Routing Formulations for Selective Multimodal Activation`

### 推荐标题 B
`Method: From Global Gain-Threshold Routing to Structured Clean Routing`

这样能容纳：
- global-threshold baseline
- direction-specific thresholding
- regression target routing
- ordinal routing

## 4.2 Method 内部结构建议

把原先只写一种 router 的结构，改成下面四节。

### 4.2.1 4.1 Problem formulation and legality constraints

这一节保留并强化：

- clean routing 只能使用 query-time observable features
- posthoc selector 只能作为 analysis-only line
- Oracle / posthoc stronger lines are not deployable

这一节仍然非常重要，因为它保证整篇文章边界清晰。

### 4.2.2 4.2 Naive global-threshold clean routing

在这一节里，把旧 clean learned router 明确降成一个 baseline formulation，而不是最终方案。

要明确写：

- 使用单一 global threshold `tau`
- learned score / probability 转成 binary decision
- 这是最直接但也是最粗糙的 clean routing 形式

并且要加一句解释性文字：

> This formulation serves as a minimal deployable clean baseline rather than the final structured policy proposed by the paper.

### 4.2.3 4.3 Structured threshold policies

新增这一节，正式容纳 E1 / E2 / E3 / E7 这类方法。

#### 需要定义的内容

1. **Direction-specific thresholding**
   - `tau_head`
   - `tau_tail`
   - 决策规则：
     - if query direction is head-side prediction, use `tau_head`
     - else use `tau_tail`

2. **Optional bucketized thresholding**
   - by relation prior bucket
   - by query-observable group

即便 E2/E3 不作为主表 strongest method，也应在 Method 中留一个“structured policy family”框架，便于解释 E1 为何合理。

### 4.2.4 4.4 Target-aligned supervision for clean routing

新增这一节，专门写 E5 / E6。

#### 需要形式化的内容

1. **Binary gain label** 作为旧 baseline
   - `y = 1(delta_rr > delta)`

2. **Regression target**
   - directly predict `delta_rr = rr_fusion - rr_struct`

3. **Ordinal gain bucket**
   - 把 `delta_rr` 划到多级区间中

这一节的核心作用是告诉读者：

> clean routing 的弱结果，并不意味着 clean setup 本身毫无价值；它也可能是 supervision design 太粗造成的。

### 4.2.5 4.5 Analysis-only post-hoc selector

这节保留，但进一步降级为 analysis layer。

要明确写：

- stronger target-aware / confidence-aware posthoc selectors are retained only for analysis and upper-bound-style separability study
- they are not part of the deployable clean claims

---

# 5. Experiments 部分改写指令

## 5.1 Experiments 的整体结构要重排

建议从原先的：

- official results
- clean routing
- posthoc analysis

改成：

### 5.1 Official model comparison and protocol diagnosis
### 5.2 Naive clean routing baseline
### 5.3 Structured clean routing
### 5.4 Target-aligned clean supervision
### 5.5 Post-hoc analysis and remaining oracle gap

这样逻辑是：

1. 先确立问题
2. 再说明 naive clean 失败
3. 再展示 structured clean 改进
4. 再展示更强 supervision 的 clean 改进
5. 最后收束 remaining gap

## 5.2 5.2 Naive clean routing baseline 要怎么写

这一小节的任务是：

- 不回避旧 global clean learned 失败
- 但把它定位成 baseline，不是最终方案

### 建议表述模板

> Under the clean legality constraint, a naive global-threshold learned router does not surpass the simple clean rule baseline. This indicates that the main difficulty of deployable routing is not merely classification capacity, but the mismatch between a single global decision boundary and the highly asymmetric gain structure induced by the protocol.

## 5.3 5.3 Structured clean routing 要怎么写

这是新主文的核心小节。

### 主文最推荐放的结果

1. **Direction-specific dual threshold (E1)**
2. **Direction × observed grouping (若你最后觉得它与 E1 高度重复，则只在文中一句话提到)**
3. **Hybrid prior-first（若它只是小幅增益，可放附录）**

### 这一节的主结论句模板

> Once the clean policy is allowed to use direction-specific thresholds instead of a single global threshold, performance increases substantially above the clean rule baseline. The best operating point consistently uses asymmetric thresholds for head-side and tail-side queries, confirming that global thresholding is too coarse for the current protocol.

### 需要明确强调的点

- E1 是 **最干净** 的结果，因为它不需要改训练，只改 policy
- 所以它最能直接证明：问题首先在 policy granularity

## 5.4 5.4 Target-aligned clean supervision 要怎么写

这一节主打 E5，E6 做 supporting。

### 建议组织方式

先写 binary baseline 不足，再写 regression 强，ordinal 次强。

### 建议结论句模板

> Replacing the coarse binary gain label with a more target-aligned supervision signal yields a further improvement in clean routing. In particular, regression-based prediction of gain magnitude outperforms both the binary clean baseline and the clean rule, indicating that the earlier weak clean result was partly caused by supervision mismatch rather than by deployable signal scarcity alone.

### 对 ordinal 的写法

> Ordinal gain modeling also improves over the binary baseline, although its gains remain below the regression formulation, suggesting that preserving magnitude information is especially useful under the current protocol.

## 5.5 5.5 Calibration / conservative fallback / delta sensitivity 怎么安排

### Calibration（E4）
写成短小节或附录支持：

> Probability calibration provides only limited gains, suggesting that miscalibration is not the primary bottleneck.

### Conservative fallback（E8）
写成 negative support result：

> Extremely conservative clean activation policies do not recover the main gains, indicating that the key issue is not simply to activate fusion less often, but to activate it with the right structural granularity.

### Delta sensitivity（E9）
这个很值得进主文或至少 main appendix。

建议写成：

> The effectiveness of clean routing is sensitive to the gain label threshold. Stronger gains appear under more permissive targets, whereas the original `delta=0.01` setting is substantially harder. This confirms that part of the earlier clean-routing weakness is tied to the label formulation itself.

---

# 6. Discussion 部分改写指令

## 6.1 Discussion 的主问题要重写

原来若是围绕：

- clean signal is too weak
- deployable routing mostly fails

现在应改成围绕三个新问题：

1. Why does naive global-threshold clean routing fail?
2. Why do structured policies help?
3. Why does a large Oracle gap still remain even after these improvements?

## 6.2 Discussion 第一段：解释 naive clean failure

### 建议写法

> The weak performance of the original clean learned router should not be interpreted as evidence that deployable clean routing is impossible in principle. Instead, the new results suggest that its main limitation lies in the use of an overly coarse global threshold and an overly simplified binary gain label. Under the current protocol, the routing boundary is highly asymmetric across query directions, and a single threshold is too rigid to capture this structure.

## 6.3 Discussion 第二段：解释 structured threshold 的意义

### 建议写法

> The strong performance of direction-specific thresholding indicates that the clean decision boundary is not globally homogeneous. Head-side and tail-side queries operate under different gain regimes, so the routing policy must be structured accordingly. This finding strengthens the protocol-aware interpretation of the task: the difficulty is not merely whether multimodal gain exists, but how its decision boundary is organized under role-specific asymmetry.

## 6.4 Discussion 第三段：解释 regression / ordinal 为什么有效

### 建议写法

> The gains from regression-style routing further show that supervision form matters. A coarse binary gain label discards information about gain magnitude and collapses qualitatively different cases into a single decision class. Predicting a richer gain signal allows the router to align more closely with the actual utility difference between experts, which is especially important when the gain boundary is shallow and highly uneven.

## 6.5 Discussion 第四段：重新定义“deployable gap”

这段非常关键。

不要再把 deployable gap 写成：

- clean signals too weak, end of story

而要写成：

> Even after structured thresholding and stronger supervision, the strongest clean strategy remains substantially below Oracle. This means that the deployable gap is real, but its interpretation must be refined. The gap does not arise solely because clean query-time observable signals are useless; rather, it persists because the recoverable decision boundary remains only partially visible under legal query-time information, even after policy and supervision are improved.

## 6.6 Discussion 第五段：如何定位 posthoc line

### 建议写法

> The post-hoc selector remains useful as an analysis tool because it reveals stronger offline separability than what can be realized in a deployable clean router. This contrast is now even more informative: once clean routing is strengthened, the remaining distance to post-hoc or Oracle performance becomes a more precise indicator of how much separability is still hidden from legal query-time observation.

---

# 7. Limitations 部分建议同步补一句

你现在的 limitations 不应只写：

- clean routing is limited

而应补成：

> Although structured thresholding and richer clean supervision improve over the earlier global-threshold binary formulation, the resulting gains are still protocol-specific and remain substantially below Oracle. Moreover, some supporting analyses, such as oracle-gap decomposition and auxiliary hit-based summaries, require further verification before they can be treated as final quantitative evidence.

这句很重要，因为你自己已经发现：

- oracle_gap_decomposition 有冲突
- 新实验中的 hits 字段有异常

这部分要提前边界化。

---

# 8. 结果表与图的改写指令

## 8.1 主文主表建议

建议新增或重排成如下主表：

### Table X. Clean Routing Comparison Under Increasing Structural Strength

行建议：
1. Residual-only
2. Clean rule
3. Naive global clean learned router (best global-threshold C4)
4. Direction-specific dual-threshold clean policy (E1)
5. Regression-based clean router (E5)
6. Oracle

这样读者一眼就能看到：
- naive clean 不强
- structured clean 变强
- 但仍远低于 Oracle

## 8.2 建议的 supporting table

### Table Y. Structured Policy and Supervision Variants

放：
- prior bucket (E2)
- hybrid prior-first (E7)
- ordinal gain (E6)
- calibration (E4)
- conservative fallback (E8)

## 8.3 推荐图

### Figure A
Global threshold vs direction-specific threshold

### Figure B
Binary vs regression vs ordinal clean supervision

### Figure C
Best clean strategy vs Oracle gap

---

# 9. 必须避免的写法

以下表述现在不要再用：

1. `clean routing is largely ineffective`
2. `deployable signals are almost entirely insufficient`
3. `the main message is a negative result on clean routing`
4. `the clean line cannot outperform the legal rule`

这些都已经被新结果部分推翻。

同时也不要写成另一个极端：

1. `we solve clean routing`
2. `structured clean routing closes the gap`
3. `deployable routing is now strong enough`

因为 Oracle gap 仍明显存在。

---

# 10. 最终建议的一句话主结论

如果整篇文章最后只保留一句最核心的主结论，建议改成：

> Under the current protocol, the weakness of clean routing is not solely caused by insufficient query-time observable signals. It also stems from overly coarse global thresholding and binary gain supervision. Once the clean policy is structured more appropriately and trained with a more target-aligned objective, deployable routing can recover additional gains beyond a simple legal rule, although a substantial gap to Oracle still remains.

---

# 11. 最后执行顺序建议

建议你按下面顺序改稿：

1. **先改 Abstract 和 Introduction contribution paragraph**
   - 先把 strongest claim 改对
2. **再改 Method 结构**
   - 把 naive global-threshold baseline 与 structured clean routing 分开
3. **再改 Experiments**
   - 重排成 “naive clean → structured clean → target-aligned supervision → remaining gap”
4. **最后改 Discussion / Limitations**
   - 把“clean signal too weak”改成“policy granularity + supervision form + remaining deployable gap”

---

# 12. 一句话总结

这轮新结果要求论文从“clean routing mostly fails”的旧叙事，转向：

> **naive clean routing fails, but structured clean routing and richer supervision partially recover deployable gains; however, the remaining Oracle gap still shows that the problem is far from fully solved.**
