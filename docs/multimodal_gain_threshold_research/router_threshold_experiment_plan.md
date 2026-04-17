# 多模态融合收益阈值模型：Router Baseline + Hard-Threshold 实验方案（详细执行版）

## 0. 文档目的

本文档用于将当前已经讨论完成的 **router baseline + hard-threshold 实验方案** 固化为一份尽可能无歧义、可直接拆解为代码任务与论文任务的执行文档。

本文档的目标不是继续做概念讨论，而是把以下内容全部明确下来：

1. 本轮方法扩展到底要解决什么问题。
2. 为什么这个问题与当前论文初稿严格衔接。
3. 两个 expert 分别是谁，为什么这样选。
4. gain label 如何定义，在哪个数据集上构造，如何避免泄漏。
5. router baseline 有哪些，分别承担什么角色。
6. hard-threshold routing 的数学定义是什么。
7. router 输入特征到底有哪些，哪些是首版必须做的，哪些暂时不做。
8. 首轮实验矩阵怎么排，哪些是主实验，哪些是消融。
9. 首轮成功标准是什么，什么情况下说明这条路线成立。
10. 实现时应该输出哪些中间文件、表格和图。

---

## 1. 当前研究起点与问题重述

### 1.1 当前论文初稿已经得到的核心结论

当前初稿已经不是在回答“多模态有没有用”这种泛问题，而是在回答一个更精确的问题：

> 在 OpenBG-IMG 的当前 protocol 下，多模态信息在什么条件下有用，为什么这种帮助无法转化为整体最优？

根据当前初稿，已有结论可以概括为：

- 当前 setting 存在明显的 **role–modality asymmetry**。
- `tail_no_img` 在整体评估中占很大比重，并显著偏向 structural modeling。
- `head_has_img` 是当前 protocol 下最清晰的 multimodal-favorable regime。
- 全局 strongest competitor 不是最完整的 multimodal model，而是更强的 structural compensation path。
- 因此，multimodal gain 是 **conditional / bounded / protocol-aware** 的，而不是 globally uniform。

这意味着当前论文的问题已经不再是“怎么做一个更复杂的 fusion 模型”，而是：

> 既然 multimodal gain 是可条件化的、可分区间的，那么模型是否应该在所有 query 上一律启用 fusion？

答案显然是否定的。

---

### 1.2 新方法扩展的核心命题

因此，本轮方法扩展不再追求：

- 设计一个更复杂、端到端更重的 multimodal architecture；
- 或者声称 multimodal globally dominates structural baselines。

而是转向一个更稳、更符合已有证据的命题：

> 在 role-conditioned missing-modality setting 下，学习一个 gain-aware router，用于预测某个 query 是否值得启用 multimodal fusion；当预测收益超过阈值时，使用 fusion expert；否则退回 structural expert。

也就是说，这一轮的核心从 **“如何融合”** 进一步推进为 **“何时融合”**。

---

### 1.3 本轮方案在论文叙事中的位置

本轮方案不是推翻旧稿，而是对旧稿的自然延伸。

旧稿主张：

- multimodal gain 是有边界的；
- 边界受 protocol、relation context、modality availability 和 residual-dominant branch preference 共同影响；
- 更显式控制的 multimodal compensation strategy 是自然的 future direction。

本轮方案正是把这一句 future work 落成正文方法：

> 将“分析型发现”转化为“选择性激活机制”。

因此，新方法的定位必须写成：

- 不是试图证明 fusion should always be on；
- 而是证明 fusion should be **selectively activated**。

---

## 2. 本轮方法方案的总览

### 2.1 方法名称建议

以下名称都可用，后续可再精修：

- **Threshold-Aware Adaptive Fusion Router**
- **Protocol-Aware Gain Router**
- **Selective Multimodal Activation via Gain Thresholding**
- **Threshold-Aware Gain Routing for MMKGC**

若要稳妥，建议暂时使用中性描述，不急着定 branding：

> A gain-aware hard-threshold router for selective multimodal activation

---

### 2.2 方法总体结构

整个方案由三部分组成：

1. **Fusion expert**
2. **Structural expert**
3. **Gain router / threshold-based selector**

其基本逻辑为：

- 对每个 query，router 先预测“启用 fusion 是否值得”；
- 若值得，则最终使用 fusion expert 的分数；
- 若不值得，则最终使用 structural expert 的分数。

形式化可写为：

\[
\alpha(q)=\mathbf{1}[p(q)>\tau]
\]

\[
s_{\text{final}}(q,e)=\alpha(q)\, s_f(q,e)+(1-\alpha(q))\, s_s(q,e)
\]

其中：

- \(q\)：一个 query，可能是 head query 或 tail query；
- \(e\)：候选实体；
- \(s_f(q,e)\)：fusion expert 对候选实体 \(e\) 的打分；
- \(s_s(q,e)\)：structural expert 对候选实体 \(e\) 的打分；
- \(p(q)\)：router 预测当前 query 为 gain-positive 的概率；
- \(\tau\)：hard threshold；
- \(\alpha(q)\in\{0,1\}\)：最终 expert 选择开关。

---

## 3. Expert 固定方案

### 3.1 为什么必须先固定 expert

在当前阶段，最危险的做法是同时修改：

- gain label 定义；
- router 结构；
- expert 组成；
- routing 粒度；
- score-level / representation-level mixing。

这会导致实验爆炸，并且无法知道结果到底来自哪里。

因此，首轮实验必须先把 expert 固定下来。

---

### 3.2 首版 expert 选择

本方案首版固定为：

- **Fusion expert = Gate-only**
- **Structural expert = Residual-only**

这是首轮最优选择，原因如下。

#### 原因 1：与当前论文主分析严格一致

当前论文中，`Gate-only` 与 `Residual-only` 本来就是最核心的两个 diagnostic variants：

- `Gate-only`：代表 fusion-only behavior；
- `Residual-only`：代表 structure-heavy compensation。

因此，用这两个作为 router 的两端专家，最容易与当前分析闭环。

#### 原因 2：两者条件互补清晰

当前稿子已经显示：

- 在 `head_has_img` 中，`Gate-only` 更强；
- 在 `tail_no_img` 中，`Residual-only` 更强；
- 这正好说明 query-level routing 有意义。

#### 原因 3：能避免 Full Model 的混合解释干扰

如果直接拿 `Full Model` 当 fusion 端，会引入额外问题：

- `Full Model` 本身就已经带 residual branch；
- 其内部 mixture weight 不是纯 fusion；
- 这样 router 的解释会变混乱。

所以首轮不建议用：

- Fusion expert = Full Model
- Structural expert = ComplEx

首轮必须优先保证解释清晰。

---

### 3.3 为什么不先用 ComplEx 做 structural expert

不是不能做，而是首轮不建议。

原因：

- 当前论文的 strongest internal structural competitor 是 `Residual-only`；
- `Residual-only` 与 `Gate-only` 来自同一内部 family，更便于分析“融合 vs 结构补偿”；
- 若直接用 ComplEx，会把 discussion 从“选择性激活 multimodal evidence”扯到“不同模型家族对比”，首轮不划算。

后续可以补做扩展实验：

- Structural expert = ComplEx
- Structural expert = max(Residual-only, ComplEx)

但这不是首轮必做项。

---

## 4. Router 的粒度与位置

### 4.1 Router 粒度：固定为 query-level

首版 router 固定为 **query-level routing**。

也就是：

- 对于 tail prediction，query 形如 \((h,r,?)\)
- 对于 head prediction，query 形如 \((?,r,t)\)

每个 query 输出一个单独的决策：

- 选 fusion expert；或
- 选 structural expert。

#### 为什么不用 triple-level

因为你的评估单位本来就是 query 下的 candidate ranking，而不是单个正负样本三元组。

#### 为什么不用 entity-level

因为当前 gain boundary 的关键是：

- direction
- target-side modality availability
- relation context

这些都天然是 query-conditioned 的，而不是 entity-only 的。

因此，query-level 是最符合当前论文 protocol-aware 逻辑的粒度。

---

### 4.2 Router 位置：固定为 score-level routing

首轮只做 **score-level routing**，不做 representation-level routing。

即：

\[
s_{\text{final}}(q,e)=\alpha(q)\, s_f(q,e)+(1-\alpha(q))\, s_s(q,e)
\]

当 \(\alpha(q)=1\) 时，用 fusion expert 的 candidate scores；
当 \(\alpha(q)=0\) 时，用 structural expert 的 candidate scores。

#### 为什么不用 representation-level mixing

因为那样会新增以下不确定性：

- 两个表示空间是否对齐；
- decoder 是否共用；
- joint training 是否必要；
- 最终提升来自 routing 还是来自更大的模型容量。

首轮实验的目的，是先验证：

> “选择谁来回答这个 query” 这件事本身，能否带来收益。

所以必须先把变化控制在 score-level。

---

### 4.3 是否做 soft routing

首轮不做。

首轮只做：

- **hard-threshold routing**

即：

\[
\alpha(q)=\mathbf{1}[p(q)>\tau]
\]

soft routing 可以作为后续扩展：

\[
\alpha(q)=\sigma((p(q)-\tau)/T)
\]

但首轮不建议，以免把“收益阈值模型”的概念搞虚。

---

## 5. Gain Label 定义（首轮固定版）

### 5.1 基本原则

gain label 不是在预测“是否有图”，也不是预测“当前 query 是不是视觉关系”。

它真正要表达的是：

> 在当前 query 条件下，fusion expert 相比 structural expert 是否带来了有意义的实际收益。

因此，label 必须来自 **expert outcome difference**，而不是 shallow heuristic。

---

### 5.2 首轮 label 形式：二分类 margin label

定义：

- 设 \(rank_f(q)\) 为 fusion expert 在 query \(q\) 上对真实目标实体的 filtered rank；
- 设 \(rank_s(q)\) 为 structural expert 在 query \(q\) 上对真实目标实体的 filtered rank；
- 定义 reciprocal ranks：

\[
RR_f(q)=\frac{1}{rank_f(q)}, \qquad RR_s(q)=\frac{1}{rank_s(q)}
\]

定义 gain 差值：

\[
\Delta(q)=RR_f(q)-RR_s(q)
\]

然后定义二分类 label：

\[
y(q)=
\begin{cases}
1, & \Delta(q) > \delta \\
0, & \Delta(q) \le \delta
\end{cases}
\]

其中：

- \(y(q)=1\)：当前 query 是 gain-positive，值得启用 fusion；
- \(y(q)=0\)：当前 query 不值得启用 fusion，应该退回 structural expert；
- \(\delta\)：margin 超参数，用来避免把几乎没有意义的极小差异也算作 gain-positive。

---

### 5.3 首轮 margin 候选值

首轮只试以下三个值：

- `delta = 0.00`
- `delta = 0.01`
- `delta = 0.02`

说明：

- `0.00`：最宽松，只要 fusion 略优即视为正样本；
- `0.01`：推荐首选值；
- `0.02`：更保守，正样本更少，但更纯。

首轮推荐主报告围绕 `delta = 0.01` 展开，其他两个作为敏感性分析。

---

### 5.4 为什么不用 rank 更小就算正样本

不要使用：

\[
y(q)=\mathbf{1}[rank_f(q)<rank_s(q)]
\]

原因是这会把大量几乎无意义的边界差异也标成正样本。

例如：

- fusion rank = 102
- structural rank = 103

虽然 fusion technically 更好，但这种差异未必有方法意义。

而 reciprocal-rank 差更贴近 MRR 逻辑，也更符合当前论文的整体评估口径。

---

### 5.5 label 的数据来源

label 只能从：

- train
n或者
- dev

上构造。

**绝对不能从 test 上构造。**

否则会造成 label leakage，使得 test 结论不可用。

首轮推荐方案：

- 先在 **dev** 上构造 label，完成原型验证；
- 若需要扩大样本量，再考虑在 train 上用同样方式构造 router supervision。

---

### 5.6 label 构造粒度

首轮固定为：

- **query-level gain label**

不是：

- sample-level
- triple-level
- candidate-level

原因：

1. query-level 与最终 ranking task 一致；
2. 当前 subgroup 分析也是 query-side decomposition；
3. router 本身就是在决定“当前 query 交给哪个 expert”。

---

## 6. Router Baseline 设计

本轮 router baseline 必须按层次来排，不能只做 learned router。

首轮固定做 4 种：

1. Oracle router
2. Rule-based router
3. Logistic Regression router
4. XGBoost router

---

### 6.1 R0：Oracle Router（上界）

#### 定义

对每个 query，若 fusion expert 更优，则选 fusion；否则选 structural。

可写为：

\[
\alpha_{oracle}(q)=
\begin{cases}
1, & RR_f(q) > RR_s(q) \\
0, & RR_f(q) \le RR_s(q)
\end{cases}
\]

或使用与 label 相同的 \(\Delta(q)\) 判断。

#### 作用

Oracle router 不是正式方法，而是上界。

它用来回答：

> 在当前 setting 下，“动态选择 expert” 这件事到底有没有理论空间？

#### 必须关注的判断

- 如果 Oracle 只比 `Residual-only` 高一点点，则说明 routing 的空间很有限；
- 如果 Oracle 明显高于 `Full Model`，则说明 selective routing 值得做；
- 如果 Oracle 也明显高于 `Residual-only`，说明这条路线潜力更大。

首轮必须先跑 Oracle。

---

### 6.2 R1：Rule-Based Router

#### 定义思想

这是最关键的 heuristic baseline。

它不使用 learned model，而是只利用当前论文已经发现的两类核心条件：

1. target-side modality availability
2. relation-conditioned gain prior

#### 首版推荐规则

\[
\alpha_{rule}(q)=1
\quad \text{iff} \quad
(target\_has\_img=1) \land (G_r > \gamma)
\]

否则：

\[
\alpha_{rule}(q)=0
\]

其中：

- `target_has_img`：当前 query 的真实 target 是否 image-available；
- \(G_r\)：relation \(r\) 的历史 gain prior；
- \(\gamma\)：rule threshold。

#### relation gain prior 的定义

在 train/dev 上统计：

\[
G_r = \mathbb{E}_{q \in r}[RR_f(q)-RR_s(q)]
\]

也可额外保存：

\[
W_r = \mathbb{E}_{q \in r}[\mathbf{1}(RR_f(q)>RR_s(q))]
\]

其中：

- \(G_r\)：平均 gain 大小；
- \(W_r\)：fusion win rate。

首轮只要求用 \(G_r\)，不要同时引入太多规则分支。

#### gamma 首轮候选值

- `gamma = 0.000`
- `gamma = 0.005`

#### 这个 baseline 的作用

Rule-based router 用来回答：

> 如果我只用当前论文已经揭示出来的协议条件与 relation prior，是否已经能做出有效的 selective activation？

如果 learned router 不能明显优于它，那么论文方法贡献会偏弱。

---

### 6.3 R2：Logistic Regression Router

#### 为什么需要它

这是最干净、最可解释的 learned baseline。

优点：

- 简单；
- 稳定；
- 可解释；
- 适合作为 paper 中的第一 learned model。

#### 输入

固定使用本文第 7 节定义的 router features。

#### 输出

输出：

\[
p(q)=P(y(q)=1\mid x_q)
\]

然后做硬阈值：

\[
\alpha(q)=\mathbf{1}[p(q)>\tau]
\]

#### 作用

Logistic 主要是证明：

- 简单学习器是否已经能超过 heuristic；
- 特征是否具有足够信息量；
- gain prediction 是否可学习。

---

### 6.4 R3：XGBoost Router

#### 为什么需要它

XGBoost 更适合首轮 tabular feature setting，通常比 Logistic 更能拟合非线性交互。

考虑到你的特征中包含：

- relation prior
- modality flag
- confidence margin
- text-image consistency

这些特征之间可能存在非线性交互，所以 XGBoost 是一个非常合理的首轮强基线。

#### 作用

XGBoost 的定位是：

- 作为 learned router 的主力版本；
- 如果 Logistic 能赢 heuristic，而 XGBoost 再进一步提升，那么方法会更稳。

#### 注意

首轮就到这里，不要立刻再上：

- MLP
- GNN router
- Transformer router
- end-to-end router

否则会过早复杂化。

---

## 7. Router 输入特征（首版固定集）

首轮 router 特征必须遵循两个原则：

1. 必须服务于当前论文已证明的 gain boundary 机制；
2. 必须避免一开始就做成超大杂烩。

因此，首版只做以下 8 类特征。

---

### 7.1 协议条件特征

#### F1. `direction`

取值：

- `head`
- `tail`

编码方式：

- binary indicator 或 one-hot

作用：

- 当前 protocol 下，direction 与 modality availability 强耦合；
- 是 gain boundary 的第一层条件。

#### F2. `target_has_img`

取值：

- `0`
- `1`

定义：

- 当前 query 的真实 target entity 是否 image-available。

作用：

- 这是最基础但不能单独依赖的条件。

#### F3. `relation_id`

表示方式二选一：

- one-hot / embedding index
- target encoding

首轮建议：

- 对 Logistic：不直接喂 relation_id，避免维度膨胀；
- 对 XGBoost：可以考虑 target encoding 或 relation gain prior 替代。

#### F4. `relation_gain_prior`

定义：

\[
relation\_gain\_prior(r)=\mathbb{E}_{q \in r}[RR_f(q)-RR_s(q)]
\]

这是首轮最重要的 relation-level feature。

作用：

- 让 router 看到“该 relation historically 是否适合 fusion”。

---

### 7.2 模态一致性特征

#### F5. `text_img_cosine`

定义：

- 当前 target entity 的 text embedding 与 image embedding 的 cosine similarity。

作用：

- 若文本与图像高度一致，则说明视觉信号较可能可靠；
- 若两者冲突严重，则 fusion 的风险更高。

注意：

- 对于无图实体，需要定义缺省处理；
- 可以设为 0，或单独配合 `img_is_missing_replaced` 使用。

#### F6. `img_is_missing_replaced`

定义：

- 当前 target entity 的 image 是否为真实图像特征；
- 若实际使用的是 `v_missing`，则记为 1；否则记为 0。

作用：

- 明确告诉 router：这个 query 的“图像模态”是否只是占位符。

---

### 7.3 expert 置信度特征

#### F7. `fusion_margin`

定义：

- fusion expert 对当前 query 的 top1 score 与 top2 score 之差。

\[
fusion\_margin(q)=s_f^{(1)}(q)-s_f^{(2)}(q)
\]

作用：

- 表示 fusion expert 自己对答案有多自信。

#### F8. `struct_margin`

定义：

- structural expert 的 top1 score 与 top2 score 之差。

\[
struct\_margin(q)=s_s^{(1)}(q)-s_s^{(2)}(q)
\]

作用：

- 表示 structural expert 的自信程度。

#### 为什么 margin 而不是只用 top1 score

因为 raw score 在不同模型之间可能尺度不完全一致，而 margin 更像 query-internal confidence。

---

### 7.4 首轮暂时不纳入的特征

以下特征不是没价值，而是首轮先不做：

- `mix_w_fusion`
- `mix_w_residual`
- `g_mean_img`
- `g_mean_noimg`
- `grad_projection`
- relation frequency 的过多变体
- graph degree 的复杂统计
- candidate entropy 的复杂近似

原因：

- 它们更适合作为第二轮增强或 behavior analysis；
- 首轮先验证主路线，不要一次把所有诊断量塞进 router。

---

## 8. Hard-Threshold Routing 规则

### 8.1 概率输出

router 输出：

\[
p(q)=P(y(q)=1\mid x_q)
\]

解释为：

- 当前 query 为 gain-positive 的概率；
- 即“启用 fusion 值得”的概率。

---

### 8.2 硬阈值决策

定义：

\[
\alpha(q)=\mathbf{1}[p(q)>\tau]
\]

其中：

- \(\alpha(q)=1\)：使用 fusion expert；
- \(\alpha(q)=0\)：使用 structural expert。

然后：

\[
s_{\text{final}}(q,e)=\alpha(q)\, s_f(q,e)+(1-\alpha(q))\, s_s(q,e)
\]

这意味着：

- 不是对两个 expert 的 score 做 soft blend；
- 而是整条 query 只交给一个 expert。

这正是“threshold-based selective activation”的最纯净表达。

---

### 8.3 首轮 threshold 候选值

主实验固定使用：

- `tau = 0.3`
- `tau = 0.5`
- `tau = 0.7`

扩展扫描可额外做：

- `tau = 0.1`
- `tau = 0.9`

解释：

- `0.3`：更激进，更容易打开 fusion；
- `0.5`：中性；
- `0.7`：更保守，只在较确信时才启用 fusion。

---

### 8.4 对阈值曲线的预期

随着 \(\tau\) 增大，理论上会发生：

- fusion coverage 下降；
- 被分给 fusion 的 query 纯度上升；
- harmful fusion 下降；
- overall performance 可能先升后降。

这会形成一条典型的 threshold–coverage–performance trade-off 曲线。

这张图对论文非常重要，因为它能真正把“收益阈值模型”可视化。

---

## 9. 实验矩阵（首轮固定版）

为防止实验爆炸，本轮实验矩阵必须严格收敛。

---

### 9.1 主实验比较组

主结果表固定比较以下 7 个系统：

1. `Residual-only`
2. `Gate-only`
3. `Full Model`
4. `Oracle Router`
5. `Rule-based Router`
6. `Logistic Hard-Threshold Router`
7. `XGBoost Hard-Threshold Router`

说明：

- `Residual-only`：最关键 structural anchor
- `Gate-only`：最关键 fusion-only anchor
- `Full Model`：当前完整 multimodal 参考系
- `Oracle Router`：路由理论上界
- `Rule-based Router`：heuristic baseline
- `Logistic`：简单 learned router
- `XGBoost`：强 tabular learned router

---

### 9.2 主实验评估指标

必须与当前论文协议保持一致：

- MRR
- Hits@1
- Hits@3
- Hits@10

并继续使用：

- filtered ranking
- direction = both
- 三个随机种子

不要另起一套协议，否则新旧稿会脱节。

---

### 9.3 subgroup evaluation

必须固定报告以下三个 subgroup：

- `head_has_img`
- `head_no_img`
- `tail_no_img`

不能省略。

因为当前 setting 下这三个 regime 是 protocol 内生的，而不是后加分析。

#### subgroup 的核心观察目标

不是要求所有 subgroup 全涨，而是观察是否出现：

- `head_has_img`：router 更接近 `Gate-only`
- `tail_no_img`：router 更接近 `Residual-only`
- `overall`：router 优于 `Full Model`，并尽量逼近 Oracle

若出现该模式，则说明 router 学到了“何时融合”。

---

### 9.4 threshold scan

固定扫描：

\[
\tau \in \{0.1,0.3,0.5,0.7,0.9\}
\]

至少报告：

1. overall MRR
2. fusion coverage
3. gain-positive precision

其中：

- **fusion coverage**：被 router 分配给 fusion expert 的 query 占比；
- **gain-positive precision**：被判给 fusion 的 query 中，真实 gain-positive 的比例。

这一组图是 paper 中阐释 threshold concept 的关键证据。

---

### 9.5 feature ablation

首轮只做 4 个版本，不再扩张：

#### A1：Only modality flag

- `target_has_img`

#### A2：Protocol-aware basic

- `target_has_img`
- `direction`
- `relation_gain_prior`

#### A3：Add modality consistency

- A2
- `text_img_cosine`
- `img_is_missing_replaced`

#### A4：Add expert confidence

- A3
- `fusion_margin`
- `struct_margin`

#### 目标

这组消融要证明：

- router 不是在学“有图就融合”；
- 而是在逐步综合 protocol condition、relation prior、modality consistency 和 expert confidence。

---

### 9.6 label / router / threshold 的组合控制

首轮不要做全排列扩展，只固定：

#### label

\[
delta \in \{0.00,0.01,0.02\}
\]

#### learned router

- Logistic Regression
- XGBoost

#### threshold

\[
\tau \in \{0.3,0.5,0.7\}
\]

因此首轮主组合数：

\[
3 \times 2 \times 3 = 18
\]

这是可控的。

---

## 10. 成功标准（首轮必须提前写死）

在首轮实验开始前，必须先定义“什么叫成功”，否则很容易跑完实验后再临时改口径。

---

### 10.1 成功标准 S1：Oracle 必须显示出明显空间

要求：

- `Oracle Router` 明显高于 `Full Model`
- 最好也高于 `Residual-only`

解释：

若 Oracle 没有明显收益，说明“路由”这个问题本身空间不大。

---

### 10.2 成功标准 S2：Learned router 必须优于 Rule-based router

要求：

- `Logistic` 或 `XGBoost` 至少有一个明显优于 `Rule-based Router`

解释：

这说明方法不只是 heuristic，而是真正在学习更细粒度的 gain condition。

---

### 10.3 成功标准 S3：subgroup 模式必须正确

要求出现：

- router 在 `head_has_img` 上更像 `Gate-only`
- router 在 `tail_no_img` 上更像 `Residual-only`

解释：

这说明 router 学到了 protocol-aware gain boundary，而不是纯噪声。

---

### 10.4 成功标准 S4：feature ablation 应该显示累进提升

理想模式：

- A2 > A1
- A3 ≥ A2
- A4 ≥ A3

解释：

这能支持你的论文叙事：

> gain prediction 不是单一 modality availability 能解释的，而是由多种条件共同决定的。

---

### 10.5 不应设定的过强目标

首轮不要把成功标准写成：

- “必须显著超过 Residual-only”

因为这不一定现实，也不完全符合你当前分析型论文的逻辑。

更合理的成功表述是：

> selective routing 能够在保留 multimodal-favorable regime 中的局部收益的同时，抑制 structurally unfavorable regime 中的 harmful fusion，并整体上优于 naive always-fuse / global fixed-mix 策略。

---

## 11. 数据与泄漏控制

### 11.1 严禁的做法

以下做法禁止：

- 用 test query 构造 gain label
- 用 test query 统计 relation gain prior
- 用 test 结果来回调 threshold 或 feature 设计

---

### 11.2 首轮推荐的数据流

#### 路线 A：快速原型验证

- 在 dev 上导出 expert outputs
- 在 dev 上构造 label 与训练 router
- 在 test 上仅做最终评估

#### 路线 B：更规范版本

- train 上构造 router supervision
- dev 上调 \(\delta\)、\(\tau\)、feature 与 router 超参数
- test 上最终报告

首轮建议先用路线 A 快速闭环，再升级到路线 B。

---

### 11.3 relation prior 的统计边界

relation-level statistics 只能来自：

- train
- dev

不能在全量数据上先算完再切分，否则会信息污染。

---

## 12. 输出文件与中间产物规范

本轮开发必须输出规范化中间文件，否则后期很难复现实验。

---

### 12.1 expert 输出文件

建议文件名：

- `gateonly_dev_query_outputs.csv`
- `residualonly_dev_query_outputs.csv`
- `gateonly_test_query_outputs.csv`
- `residualonly_test_query_outputs.csv`

每行至少包含：

- `query_id`
- `direction`
- `relation_id`
- `target_entity_id`
- `target_has_img`
- `rank`
- `rr`
- `top1_score`
- `top2_score`
- `top1_margin`

---

### 12.2 router feature 表

建议文件名：

- `router_dev_features.csv`
- `router_test_features.csv`

字段至少包含：

- `query_id`
- `direction`
- `relation_id`
- `target_has_img`
- `img_is_missing_replaced`
- `text_img_cosine`
- `fusion_margin`
- `struct_margin`
- `relation_gain_prior`
- `rr_f`
- `rr_s`
- `delta_rr`
- `gain_label_delta_0.00`
- `gain_label_delta_0.01`
- `gain_label_delta_0.02`

注意：

- test feature 表中不应包含 test-derived label；
- 若保留 `rr_f/rr_s`，必须仅用于分析，不可在 test inference 中被 router 直接读取。

---

### 12.3 relation prior 文件

建议文件名：

- `relation_gain_stats.csv`

字段建议：

- `relation_id`
- `support`
- `avg_delta_rr`
- `fusion_win_rate`
- `head_query_ratio`
- `target_has_img_ratio`

首轮主用字段是 `avg_delta_rr`。

---

### 12.4 router 训练输出

建议文件名：

- `router_logistic_delta001.pkl`
- `router_xgb_delta001.pkl`
- `router_dev_probabilities_delta001.csv`

字段建议：

- `query_id`
- `p_gain`
- `pred_label_tau_03`
- `pred_label_tau_05`
- `pred_label_tau_07`

---

### 12.5 最终评估输出

建议文件名：

- `eval_router_logistic_delta001_tau03.json`
- `eval_router_logistic_delta001_tau05.json`
- `eval_router_logistic_delta001_tau07.json`
- `eval_router_xgb_delta001_tau03.json`
- `eval_router_xgb_delta001_tau05.json`
- `eval_router_xgb_delta001_tau07.json`

每个 json 建议包含：

- overall MRR / Hits@1/3/10
- subgroup metrics
- fusion coverage
- gain-positive precision
- selected expert counts

---

## 13. 论文中应生成的表与图

首轮建议固定做以下 4 类结果展示。

---

### 13.1 表 1：主结果表

列：

- Model
- MRR
- Hits@1
- Hits@3
- Hits@10

行：

- Residual-only
- Gate-only
- Full Model
- Oracle Router
- Rule-based Router
- Logistic Router
- XGBoost Router

---

### 13.2 表 2：subgroup 结果表

列：

- Model
- head_has_img
- head_no_img
- tail_no_img

目标：

- 明确展示 selective activation 的 regime-wise 行为。

---

### 13.3 图 1：threshold–coverage–MRR 曲线

推荐至少画：

- x 轴：threshold \(\tau\)
- y 轴 1：overall MRR
- y 轴 2：fusion coverage

可以分两张图画，避免双轴过乱。

---

### 13.4 表 3：feature ablation

列：

- Feature set
- Dev AUC/F1
- Test MRR
- Test fusion coverage

行：

- A1
- A2
- A3
- A4

---

## 14. 开发顺序建议（按最小闭环）

### Step 1：导出 expert 的 dev query outputs

先完成：

- Gate-only 的 dev query 输出
- Residual-only 的 dev query 输出

确保能拿到：

- rank
- reciprocal rank
- top1 / top2 score

这是整个 router 流水线的起点。

---

### Step 2：构造 gain label 与 relation prior

完成：

- `delta_rr`
- `gain_label_delta_0.00`
- `gain_label_delta_0.01`
- `gain_label_delta_0.02`
- `relation_gain_prior`

这是形成 router feature 表的关键一步。

---

### Step 3：实现 Rule-based Router

优先写 heuristic baseline。

因为它最简单，也最能先验证“选择性激活是否有用”。

---

### Step 4：训练 Logistic / XGBoost Router

建议先跑：

- delta = 0.01
- tau = 0.5

只要这一个组合先跑通，就能快速判断方向值不值得继续。

---

### Step 5：做完整 threshold 扫描与 feature ablation

当主链路跑通后，再扩展：

- tau 扫描
- A1–A4 消融
- delta 敏感性

---

### Step 6：生成论文图表与方法节草稿

当实验矩阵定型后，再正式写论文方法节，避免反复返工。

---

## 15. 首轮不做的内容清单

为了防止项目范围失控，以下内容明确列为**首轮不做**：

- end-to-end joint training
- representation-level routing
- soft routing
- 多 expert routing（超过两个 expert）
- Gain regression（连续值回归）
- 三分类 gain label
- ComplEx / TuckER 一起纳入 router 多专家版本
- GNN / Transformer router
- 复杂 uncertainty estimation
- 大量 graph topology feature

这些都可以作为第二阶段扩展，但不是首轮必需。

---

## 16. 一句话执行摘要

本轮方案的首轮执行版本可总结为：

> 以 Gate-only 作为 fusion expert、Residual-only 作为 structural expert，基于 query-level gain label 训练一个轻量 router，用 hard threshold 决定每个 query 是否启用 fusion，并在 unified protocol 下通过 overall、subgroup、threshold scan 与 feature ablation 验证 selective multimodal activation 是否优于 always-fuse 与 fixed global mixing。

---

## 17. 当前拍板版（供后续所有实现统一引用）

### 固定 expert

- Fusion expert = Gate-only
- Structural expert = Residual-only

### 固定 label

\[
y(q)=\mathbf{1}\left[\frac{1}{rank_f(q)}-\frac{1}{rank_s(q)}>\delta\right]
\]

其中：

- `delta ∈ {0.00, 0.01, 0.02}`
- 主推荐值：`delta = 0.01`

### 固定 router

- Oracle Router
- Rule-based Router
- Logistic Router
- XGBoost Router

### 固定路由形式

- query-level
- score-level
- hard-threshold

### 固定 threshold

- `tau ∈ {0.3, 0.5, 0.7}`
- 扩展扫描：`{0.1,0.3,0.5,0.7,0.9}`

### 固定主实验比较组

- Residual-only
- Gate-only
- Full Model
- Oracle Router
- Rule-based Router
- Logistic Router
- XGBoost Router

### 固定 subgroup

- head_has_img
- head_no_img
- tail_no_img

### 固定 feature set 递进

- A1: target_has_img
- A2: A1 + direction + relation_gain_prior
- A3: A2 + text_img_cosine + img_is_missing_replaced
- A4: A3 + fusion_margin + struct_margin

### 固定成功标准

- Oracle 明显高于 Full Model
- Learned router 明显优于 Rule-based router
- subgroup 呈现正确的 regime-aware 选择模式
- feature ablation 呈现累进改进趋势

---

## 18. 后续衔接建议

在本方案执行完成后，最自然的下一份文档应为：

1. **开发任务清单（代码拆解版）**
2. **论文方法节骨架（可直接写入 paper）**
3. **实验结果模板表格**

建议顺序是：

- 先按本文档开发
- 跑出第一轮结果
- 再反推方法节正文

这样返工最少。
