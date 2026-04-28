# 论文压缩与重构修改方案

## 0. 修改背景

本次修改基于导师反馈和当前论文版本的实际问题。导师提出的核心意见可以概括为：

1. 论文整体过长，最好压缩到 **15–20 页**。
2. 文章逻辑还不够连贯，尤其是 **Abstract、Introduction、Related Work** 需要更自然地引出研究问题。
3. 领域内常见和通用的方法、实验流程、评价指标不需要过度解释，应压缩篇幅。
4. 论文中属于自己创新的部分，可以通过 **黑体加粗** 的方式突出。
5. 附录中不重要的内容可以直接删除，部分重要内容可以放回实验部分。
6. 当前第三部分标题 **Task Setting and Method** 不够自然，需要考虑更符合论文规范的标题。

本文件的目标不是简单“删字”，而是帮助论文从当前的“实验记录型长文”收束为一篇更清晰的 **analysis-driven method paper**。

---

## 1. 总体修改方向

当前论文并不是缺少内容，而是内容过多、层级过散。接下来应从“全面展示所有实验”转向“围绕主问题组织证据”。

新版论文的主线建议为：

> **We first identify a protocol-induced role–modality asymmetry in OpenBG-IMG, which makes multimodal gain local and bounded. To exploit this bounded gain, we evaluate selective activation under two information boundaries: strict metadata-only routing and score-aware non-answer-aware expert combination. Results show that strict clean routing only recovers modest gain, while score-aware expert combination substantially improves performance, indicating that the main bottleneck is not router complexity but the available deployable information boundary.**

中文理解为：

> 我们首先发现 OpenBG-IMG 当前协议下存在 role–modality asymmetry，使得多模态增益不是全局可靠的，而是局部且有边界的。为利用这种 bounded gain，我们在两种可部署信息边界下研究 selective activation：一种是严格的 metadata-only clean routing，另一种是 score-aware non-answer-aware expert combination。实验表明，严格 clean routing 只能带来有限增益，而 score-aware expert combination 能显著提升性能，这说明关键瓶颈不只是 router 复杂度，而是可部署信息边界。

因此，论文逻辑应调整为：

```text
发现问题
→ 解释问题为什么重要
→ 提出两级解决方案
→ 实验证明两类信息边界下的可恢复增益
→ 给出受控结论
```

---

## 2. 对导师意见的逐条吸收

### 2.1 论文太长，需要压缩到 15–20 页

导师这个意见是合理的。当前版本篇幅过长，正文和附录中包含较多过程性说明、中间实验、重复表格和实现细节。

建议目标页数如下：

| 部分 | 建议页数 |
|---|---:|
| Abstract | 0.5 页 |
| Introduction | 2 页 |
| Related Work | 2 页 |
| Problem Formulation and Methodology | 4–5 页 |
| Experiments | 6–8 页 |
| Conclusion | 0.5–1 页 |
| Appendix | 只保留关键补充 |

如果导师要求 **15–20 页含参考文献**，正文最好控制在 **12–15 页** 左右。

如果 **15–20 页不含参考文献**，正文控制在 **18 页左右** 是合理的。

需要优先压缩的内容包括：

- 过细的 routing training procedure；
- 重复解释 clean legality 的段落；
- 多个相似 routing variant 的完整表格；
- RQ 的过长铺垫；
- 已经由表格说明清楚的内容；
- appendix split 的非核心内容；
- seed-wise 或 threshold-scan 等过程性实验结果。

---

### 2.2 文章逻辑不够连贯，需要重新组织 Abstract / Introduction / Related Work

当前论文的 Abstract、Introduction 和 Related Work 已经包含主要信息，但问题是术语密度高、逻辑跳跃较多，读者第一遍不容易抓住主线。

建议三部分都围绕同一个推理链展开：

```text
MMKGC 理论上可以利用图像和文本增强 KGC
→ 但商品知识图谱中视觉支持不完整
→ OpenBG-IMG 当前协议进一步造成 role–modality asymmetry
→ 因此 multimodal gain 是 local and bounded，而不是 globally reliable
→ 所以问题从“如何更强地融合模态”转向“何时激活多模态证据”
→ 本文比较 strict clean routing 与 score-aware expert combination
```

#### Abstract 建议结构

Abstract 建议控制在 **180–230 words**，采用四句话结构：

1. **Background + Problem**  
   MMKGC assumes auxiliary modalities can improve KGC, but in product KGs visual support is incomplete and uneven.

2. **Finding**  
   On OpenBG-IMG, the current protocol induces role–modality asymmetry: head-side targets may have images, while tail-side targets are image-unavailable, causing multimodal gain to be local and bounded.

3. **Method**  
   We study selective activation between fusion and structural experts under two deployable information boundaries: strict metadata-only routing and score-aware non-answer-aware combination.

4. **Result + Conclusion**  
   Strict clean routing recovers only modest gain, while score-aware combination reaches stronger performance; this shows the key bottleneck is the available deployable expert information rather than router complexity.

建议 Abstract 中只保留少量关键结果，例如：

- strict clean routing best: **0.2982 MRR**；
- CA-S2: **0.3142 MRR**；
- score interpolation: **0.3428 MRR**。

不要在 Abstract 中堆太多数值和术语。

---

### 2.3 通用方法不需要详细解释

导师指出的这一点非常重要。论文中与领域共识相关的内容应当压缩，避免写成教程。

建议压缩如下：

| 内容 | 正文处理方式 |
|---|---|
| KGC 基本任务 | 一段即可 |
| filtered ranking | 一句话说明 |
| MRR / Hits@K | 不详细解释，只说明作为指标 |
| ComplEx scoring | 可给公式，但不展开基础原理 |
| early stopping / checkpoint selection | 放实验设置一句话 |
| bootstrap | 只说明 paired bootstrap，详细过程可删或放附录 |
| threshold selection | 保留 “selected on validation set only” |
| router features C1–C4 | 不建议全部保留，只保留最终使用特征或压缩成小表 |
| negative sampling / training details | 若不是本文创新，压缩为实现设置 |

正文应重点解释的是自己的创新部分：

1. **Protocol-aware bounded-gain diagnosis**；
2. **Strict metadata-only clean routing**；
3. **Score-aware non-answer-aware expert combination**；
4. **Information boundary distinction**。

---

### 2.4 创新部分可以加粗

可以采用导师建议，但需要克制使用。不要到处加粗，而是在关键位置突出“本文自己的贡献”。

建议加粗位置：

- Introduction 的 contributions；
- Method 每个核心模块第一次出现时；
- Experiment 每个 RQ 的核心结论句；
- Discussion 中总结信息边界时。

建议将贡献点从 4 个压缩为 3 个：

```text
1. **Protocol-aware bounded-gain diagnosis.**
   We show that the current OpenBG-IMG protocol induces role–modality asymmetry, making multimodal gain local rather than globally reliable.

2. **Selective activation under strict clean information boundaries.**
   We evaluate whether bounded multimodal gain can be recovered using only metadata available at query time.

3. **Score-aware expert combination under non-answer-aware constraints.**
   We show that fixed-expert score patterns provide much stronger deployable signal than strict query-level metadata, and that simple score interpolation can outperform more complex learned routers.
```

这样更集中，也更符合当前实验结果。

---

### 2.5 附录中不重要的部分可以删除，重要部分可以放入实验

需要注意：导师不是说“一定要大量删除正文图表”，而是说：

> 附录中不重要的部分可以不展示；正文需要压缩，所以一部分低价值图表也可以合并或删除。

因此，图表处理原则不是机械删，而是判断其是否直接支撑主线。

建议处理为三层：

#### 第一层：正文必须保留

这些图表直接支撑论文主线，不建议删除：

| 图表类型 | 是否保留 | 理由 |
|---|---|---|
| Role–modality asymmetry 图 | 保留 | 这是论文问题来源 |
| Dataset statistics / target-side regime counts | 保留，最好合并 | 证明协议不对称 |
| Official model comparison | 保留 | 证明 always-on multimodal fusion 不是全局最优 |
| Subgroup MRR by target-side regime | 保留 | 回答 multimodal gain 出现在哪里 |
| Main clean routing comparison | 保留 | 支撑 strict clean routing 只能带来 modest gain |
| Main score-aware combination / interpolation result | 保留 | 支撑 score-aware expert combination 的核心贡献 |

#### 第二层：正文可保留，但建议压缩或合并

这些内容可以保留，但要提高信息密度：

| 图表类型 | 建议 |
|---|---|
| Information boundary 表 | 建议保留，但压缩成小表 |
| Progressive clean-routing ablation | 可与 main clean routing 表合并 |
| Bootstrap significance 表 | 可简化成 main table 的 CI 列 |
| Relation-group analysis | 如果支撑 RQ1 可保留，否则放附录 |
| Candidate-router bootstrap 表 | 可压缩，不必列所有 comparison |
| Alpha diagnostics | 更适合附录，正文只写关键观察 |

#### 第三层：优先删除或不展示

这些更符合导师所说的“附录中不重要的部分可以不展示”：

| 内容 | 建议 |
|---|---|
| 不改变主结论的 threshold scan | 删除或极简附录 |
| 多个相似 routing variant 的完整表 | 删除，只保留代表性 baseline |
| appendix split 的完整结果 | 如果不是必须证明 robustness，可删除 |
| 过细的 training details 表 | 删除或改成一句话 |
| 大量 seed-wise 细节 | 删除，保留 mean ± std 即可 |
| 重复说明 evaluation line 的长表 | 压缩或转成文字 |

关键判断标准：

> **如果这张表能回答一个 RQ，保留。  
> 如果它只是解释某个 RQ 的中间过程，合并。  
> 如果它只是证明“我也试过这个”，删除或放附录。**

---

### 2.6 第三部分标题 Task Setting and Method 是否合适

导师的疑问是合理的。

**Task Setting and Method** 不是绝对错误，但不够像成熟论文中的标准标题。它的问题是：

- “Task Setting” 比较偏任务说明；
- “Method” 比较宽泛；
- 当前 Section 3 实际上包含 problem setting、protocol、routing method、score-aware combination 和 information boundary；
- 因此标题需要更能概括这一节的功能。

更推荐改为：

```text
3 Problem Formulation and Methodology
```

这个标题比 Task Setting and Method 更自然，也更符合当前论文强调 protocol-aware setting 的特点。

其他可选标题：

```text
3 Problem Setup and Methodology
3 Problem Formulation and Selective Activation
3 Methodology
3 Task Definition and Proposed Method
```

其中最推荐：

> **3 Problem Formulation and Methodology**

因为你的论文不只是提出方法，还要先定义当前协议下的问题结构。

---

## 3. 新版章节结构建议

建议新版论文结构如下：

```text
Abstract

1 Introduction
   1.1 Motivation
   1.2 Protocol-induced bounded multimodal gain
   1.3 Contributions

2 Related Work
   2.1 Multimodal Knowledge Graph Completion
   2.2 Missing Modality and Modality Imbalance
   2.3 Selective Activation and Score-aware Expert Combination

3 Problem Formulation and Methodology
   3.1 Problem Definition and OpenBG-IMG Protocol
   3.2 Bounded-Gain Diagnosis
   3.3 Strict Query-Level Clean Routing
   3.4 Score-Aware Expert Combination
   3.5 Information Boundaries

4 Experiments
   4.1 Experimental Setup
   4.2 RQ1: Is multimodal gain globally reliable?
   4.3 RQ2: How much bounded gain can strict clean routing recover?
   4.4 RQ3: Does score-aware combination recover more gain?
   4.5 Discussion

5 Conclusion

Appendix
   A Additional implementation details
   B Additional ablation results
   C Additional diagnostics
```

---

## 4. RQ 结构调整建议

当前版本中 RQ 较多，容易让文章显得碎。建议从 4 个 RQ 压缩为 3 个 RQ。

### 原问题

原版本大致包含：

1. Where does multimodal gain appear?
2. Why does naive clean routing fail?
3. How far can structured query-level clean routing go?
4. Can score-aware expert combination recover more deployable gain?

### 建议合并后

```text
RQ1. Where does multimodal gain appear under the OpenBG-IMG protocol?

RQ2. How much bounded gain can strict metadata-only routing recover?

RQ3. Does score-aware non-answer-aware combination recover substantially more deployable gain?
```

这样逻辑更紧凑：

```text
RQ1: 先证明 gain 是 bounded/local
RQ2: 再证明 strict clean routing 只能恢复一部分
RQ3: 最后证明 score-aware combination 可以恢复更多
```

这也更符合“发现问题 → 尝试严格解决 → 扩展信息边界后解决得更好”的推理路线。

---

## 5. 正文图表保留与压缩方案

根据导师意见和当前论文主线，正文图表不需要机械限制数量，但每张图表都必须服务主线。

更合理的正文图表数量范围：

| 论文长度目标 | 合理正文图表数量 |
|---|---:|
| 15 页左右 | 6–8 个 |
| 18–20 页 | 8–10 个 |
| 20 页以上 | 10–12 个也可以，但要紧凑 |

建议正文保留大约 **8–9 个核心图表**：

| 编号 | 内容 | 处理方式 |
|---|---|---|
| Figure 1 | Role–modality asymmetry | 保留 |
| Table 1 | Dataset statistics + target-side regime counts | 合并 |
| Table 2 | Official model comparison | 保留 |
| Table 3 | Subgroup MRR by target-side regime | 保留 |
| Table 4 | Information boundary | 压缩保留 |
| Table 5 | Clean routing main result + ablation + CI | 合并保留 |
| Table 6 | Score-aware candidate router result | 保留或并入 Table 7 |
| Table 7 | Score interpolation vs CA-S2 | 保留 |
| Figure/Table 8 | Main takeaway visualization | 可选，例如 clean vs score-aware gain ladder |

### 建议合并的表格

#### Clean routing 相关表格合并

可以将以下内容合并为一个紧凑表：

- clean rule；
- naive global clean router；
- direction-specific threshold；
- regression-based clean router；
- hard-selection oracle；
- MRR；
- delta；
- 关键 CI。

这样可以避免 clean routing 部分出现多个相似表格。

#### Score-aware 相关表格合并

可以将以下内容合并：

- Residual-only；
- E5；
- CA-S1；
- CA-S2；
- CA-S3；
- score interpolation；
- delta vs E5；
- 是否 score-aware；
- 是否 answer-aware。

这样能直接呈现最核心结论：

> score-aware expert combination 明显强于 strict metadata-only routing。

---

## 6. 需要压缩或删除的具体内容

### 6.1 Abstract

需要重写，不是小修。

当前 Abstract 信息量过大，建议改为：

- 减少术语堆叠；
- 只保留最关键结果；
- 重点突出 bounded gain 和 information boundary；
- 控制在 180–230 words。

---

### 6.2 Introduction

需要重写逻辑，而不只是删字。

建议按以下顺序：

1. MMKGC 的基本动机；
2. 商品知识图谱中 visual support incomplete；
3. OpenBG-IMG 当前协议下存在 role–modality asymmetry；
4. 因此 multimodal gain 是 local and bounded；
5. 研究问题变成 when to activate multimodal evidence；
6. 本文提出 strict clean routing 和 score-aware combination 两条路线；
7. 总结 3 个贡献点。

建议删除或压缩：

- 过长的 4 个 RQ 展开；
- 与后文重复的实验结果；
- 过细的模型名和数值；
- “This paper makes four linked contributions...” 中过长的表格。

---

### 6.3 Related Work

Related Work 应服务于引出本文，而不是全面综述所有 KGC / MMKGC 方法。

建议保留 3 个小节：

```text
2.1 Multimodal Knowledge Graph Completion
2.2 Missing Modality and Modality Imbalance
2.3 Selective Activation and Score-aware Expert Combination
```

每个小节控制在 2–3 段。

压缩重点：

- 不展开所有 KGC baseline；
- 不详细介绍每类 fusion 方法；
- 不做太多 citation 堆叠；
- 每节最后一句都要引回你的研究 gap。

---

### 6.4 Section 3

将标题改为：

```text
3 Problem Formulation and Methodology
```

建议重组为：

```text
3.1 Problem Definition and OpenBG-IMG Protocol
3.2 Bounded-Gain Diagnosis
3.3 Strict Query-Level Clean Routing
3.4 Score-Aware Expert Combination
3.5 Information Boundaries
```

需要合并或压缩：

- 3.2 Query-level Clean Routing Formulations；
- 3.3 Clean Router Features, Training, and Evaluation Basis；
- 3.5 Router Training Details。

这些内容可以整合到 3.3 中，不需要单独长篇展开。

### 6.5 Experiments

建议按 3 个 RQ 重排，而不是按实验过程堆叠。

新版结构：

```text
4 Experiments
   4.1 Experimental Setup
   4.2 RQ1: Is multimodal gain globally reliable?
   4.3 RQ2: How much bounded gain can strict metadata-only routing recover?
   4.4 RQ3: Does score-aware combination recover more gain?
   4.5 Discussion
```

每个 RQ 只回答一个核心问题：

- RQ1：证明 multimodal gain local and bounded；
- RQ2：证明 strict clean routing 有增益但有限；
- RQ3：证明 score-aware combination 显著更强。

---

## 7. 建议保留的论文定位

当前论文不适合写成：

> We propose a stronger MMKGC model.

因为实验结果并不支持“新模型全局最强”这个说法。

更适合写成：

> We provide a protocol-aware analysis and selective expert-combination framework for MMKGC under incomplete visual support.

或者：

> This paper is an analysis-driven method paper: it first diagnoses why multimodal gain is bounded under the current OpenBG-IMG protocol, then shows how much of this bounded gain can be recovered under different deployable information boundaries.

中文理解为：

> 这篇论文是一篇分析驱动的方法论文。它首先诊断为什么当前 OpenBG-IMG 协议下多模态增益是有限且局部的，然后研究在不同可部署信息边界下，这部分 bounded gain 能被恢复到什么程度。

这个定位最安全，也最符合当前结果。

---

## 8. 建议在论文中反复强调的核心表述

以下表达可以作为全文主线句反复使用，但不要每处都完全重复。

### 8.1 关于问题

> Under the current OpenBG-IMG protocol, multimodal gain is not globally reliable but local and bounded due to role–modality asymmetry.

### 8.2 关于研究重点转移

> The central question is therefore not simply how to design a stronger fusion module, but when multimodal evidence should be activated under deployable information constraints.

### 8.3 关于 strict clean routing

> Strict metadata-only clean routing can recover statistically supported but modest gain, indicating that query-level legal metadata captures only part of the bounded multimodal signal.

### 8.4 关于 score-aware combination

> Once fixed-expert score patterns are available without using answer-aware target information, substantially more deployable gain can be recovered.

### 8.5 关于最终结论

> The main bottleneck is not necessarily router complexity, but the deployable information boundary under which expert combination is performed.

---

## 9. 下一步执行清单

建议按以下顺序修改：

### 第一阶段：结构重排

- [ ] 将 Section 3 标题改为 **Problem Formulation and Methodology**。
- [ ] 将 RQ 从 4 个压缩为 3 个。
- [ ] 将 Experiments 按 3 个 RQ 重排。
- [ ] 删除或合并重复的 routing / evaluation line 表格。
- [ ] 明确正文和附录边界。

### 第二阶段：重写前两章

- [ ] 重写 Abstract，控制在 180–230 words。
- [ ] 重写 Introduction，形成自然推理链。
- [ ] Contributions 从 4 个压缩为 3 个。
- [ ] Related Work 压缩为 3 个小节，每节 2–3 段。

### 第三阶段：压缩 Method

- [ ] 将通用 KGC / filtered ranking / metrics 说明压缩。
- [ ] 将 clean routing feature details 压缩。
- [ ] 删除过细 router training details。
- [ ] 保留 strict clean / score-aware / oracle 的信息边界说明。
- [ ] 用黑体突出本文创新方法。

### 第四阶段：压缩 Experiments

- [ ] 保留 official model comparison。
- [ ] 保留 target-side subgroup analysis。
- [ ] 合并 clean routing main result、ablation 和 CI。
- [ ] 合并 score-aware candidate router 与 score interpolation 结果。
- [ ] 删除或转移不影响主结论的 appendix split 和 threshold scan。
- [ ] 每个 RQ 后加一句明确 conclusion。

### 第五阶段：附录清理

- [ ] 删除不重要的附录表格。
- [ ] 保留必要的 implementation details。
- [ ] 保留关键补充 ablation。
- [ ] 保留少量 diagnostics。
- [ ] 如果 robustness split 不能显著增强主结论，暂时删除。

---

## 10. 最终修改原则

本轮修改的总原则可以概括为：

> **新版论文要从“全面展示所有实验”改为“围绕 bounded multimodal gain 这一主问题，压缩背景、突出创新、用最少但最关键的实验支撑最强结论”。**

具体来说：

1. 不再追求展示所有实验；
2. 不再长篇解释领域通用方法；
3. 不再把 appendix 当作实验仓库；
4. 正文图表不机械减少，但必须服务主线；
5. 每个章节都要服务于 bounded gain → selective activation → information boundary 这条主线；
6. 最终论文定位应是 **protocol-aware analysis + selective expert-combination framework**，而不是“更强 MMKGC 模型”。

---

## 11. 推荐使用的新版标题和核心说法

### Section 3 标题

最推荐：

```text
3 Problem Formulation and Methodology
```

### 论文定位句

```text
This paper provides a protocol-aware analysis and selective expert-combination framework for MMKGC under incomplete visual support.
```

### 主线句

```text
Under the current OpenBG-IMG protocol, multimodal gain is local and bounded rather than globally reliable.
```

### 方法句

```text
We study selective activation under two deployable information boundaries: strict metadata-only routing and score-aware non-answer-aware expert combination.
```

### 结论句

```text
The results show that the main bottleneck is not necessarily router complexity, but the deployable information boundary under which expert combination is performed.
```
