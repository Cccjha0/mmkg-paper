# 论文压缩重构可执行版方案

> 适用对象：当前论文《Score-Aware Expert Combination for Protocol-Aware Multimodal Knowledge Graph Completion under Incomplete Visual Support》  
> 目标：根据导师反馈，将当前偏长、偏实验记录型的论文压缩重构为一篇 15–20 页左右、逻辑更连贯、主线更清晰的 analysis-driven method paper。  
> 核心主线：**protocol-induced role–modality asymmetry → bounded multimodal gain → strict clean routing only recovers modest gain → score-aware non-answer-aware expert combination recovers more deployable gain**。

---

## 0. 最终目标版本定义

### 0.1 篇幅目标

优先目标：

```text
正文 + 图表：约 15–18 页
参考文献：另计
附录：可选，尽量短
```

如果学校或期刊模板要求更紧：

```text
全文含参考文献：15–20 页
正文：12–15 页
```

### 0.2 论文定位

当前论文不应定位为：

```text
We propose a globally stronger MMKGC model.
```

应定位为：

```text
We provide a protocol-aware analysis and selective expert-combination framework for MMKGC under incomplete visual support.
```

或者更明确：

```text
This paper is an analysis-driven method paper. It first diagnoses why multimodal gain is bounded under the current OpenBG-IMG protocol, then studies how much of this bounded gain can be recovered under different deployable information boundaries.
```

### 0.3 最终核心结论

新版全文必须反复服务于以下结论：

```text
Under the current OpenBG-IMG protocol, multimodal gain is local and bounded rather than globally reliable. Strict metadata-only routing can recover only modest gain, while score-aware non-answer-aware expert combination recovers substantially more. The key bottleneck is therefore not only router architecture complexity, but the deployable information boundary under which expert combination is performed.
```

---

## 1. 总体执行顺序

不要从局部润色开始。建议严格按以下顺序执行：

```text
Step 1: 重写 Abstract
Step 2: 重写 Introduction
Step 3: 压缩 Related Work
Step 4: 重命名并重组 Section 3
Step 5: 按 3 个 RQ 重组 Experiments
Step 6: 合并正文核心表格
Step 7: 清理 Appendix
Step 8: 全文统一术语和 claim 边界
Step 9: 最后一轮语言压缩
```

原因：

- Abstract 和 Introduction 决定整篇文章主线；
- Section 3 决定方法是否清楚；
- Experiments 决定证据链是否紧凑；
- Appendix 最后处理，避免一开始纠结删不删某个表。

---

## 2. 新版整体目录

建议将当前结构改为：

```text
Abstract

1 Introduction
   1.1 Motivation
   1.2 Protocol-induced Bounded Multimodal Gain
   1.3 Contributions

2 Related Work
   2.1 Multimodal Knowledge Graph Completion
   2.2 Missing Modality and Modality Imbalance
   2.3 Selective Activation and Score-aware Expert Combination

3 Problem Formulation and Methodology
   3.1 Problem Definition and OpenBG-IMG Protocol
   3.2 Protocol-Aware Bounded-Gain Diagnosis
   3.3 Strict Query-Level Clean Routing
   3.4 Score-Aware Expert Combination
   3.5 Information Boundaries

4 Experiments
   4.1 Experimental Setup
   4.2 RQ1: Is multimodal gain globally reliable?
   4.3 RQ2: How much bounded gain can strict metadata-only routing recover?
   4.4 RQ3: Does score-aware combination recover more gain?
   4.5 Discussion

5 Conclusion

Appendix
   A Additional Implementation Details
   B Additional Ablation Results
   C Additional Diagnostics
```

---

## 3. RQ 重构方案

### 3.1 当前问题

当前版本使用 4 个 RQ：

```text
RQ1. Where does multimodal gain actually appear?
RQ2. Why does naive strict query-level clean routing fail?
RQ3. How far can structured query-level clean routing go?
RQ4. Can score-aware expert combination recover more deployable gain?
```

问题是：

- RQ2 和 RQ3 区分过细；
- 读者会觉得像实验流水线；
- Introduction 也因此变长；
- 论文主线显得散。

### 3.2 新版 RQ

建议压缩为 3 个 RQ：

```text
RQ1. Is multimodal gain globally reliable under the OpenBG-IMG protocol?

RQ2. How much bounded gain can strict metadata-only routing recover?

RQ3. Does score-aware non-answer-aware expert combination recover substantially more deployable gain?
```

### 3.3 RQ 与实验对应关系

| 新 RQ | 核心问题 | 对应证据 |
|---|---|---|
| RQ1 | Multimodal gain 是否全局可靠？ | official model comparison + target-side subgroup analysis |
| RQ2 | strict clean routing 能恢复多少 gain？ | clean rule / naive router / direction-specific threshold / regression router |
| RQ3 | score-aware combination 是否更强？ | CA-S2 / CA-S3 / score interpolation |

### 3.4 旧 RQ 的处理方式

| 旧 RQ | 新位置 |
|---|---|
| RQ1 Where does gain appear? | 新 RQ1 |
| RQ2 Why naive clean routing fails? | 并入新 RQ2 |
| RQ3 How far structured clean routing can go? | 并入新 RQ2 |
| RQ4 Score-aware combination | 新 RQ3 |

---

## 4. Abstract 可执行修改方案

### 4.1 当前问题

当前 Abstract 信息完整，但存在以下问题：

- 术语过多；
- 数值过多；
- 句子过长；
- “发现问题 → 解决问题 → 结论”的链条不够干净；
- 一开始就出现大量 routing / interpolation / CA-S2 细节，读者负担较大。

### 4.2 修改目标

控制在：

```text
180–230 words
一段
4–5 个长句
只保留 3 个关键数值
```

保留的关键数值：

```text
0.2982 MRR: strongest strict clean routing
0.3142 MRR: CA-S2 learned score-aware router
0.3428 MRR: best score interpolation
```

### 4.3 推荐 Abstract 结构

```text
Sentence 1: MMKGC 背景 + incomplete visual support 问题
Sentence 2: 本文发现 OpenBG-IMG 中 protocol-induced role–modality asymmetry
Sentence 3: 因此 multimodal gain 是 local and bounded
Sentence 4: 本文研究 two deployable information boundaries
Sentence 5: 结果：strict clean modest, score-aware stronger
Sentence 6: 结论：main bottleneck is deployable information boundary rather than router complexity alone
```

### 4.4 可直接替换的 Abstract 草稿

```text
Multimodal knowledge graph completion (MMKGC) aims to improve link prediction by incorporating auxiliary textual and visual evidence, yet in product-oriented knowledge graphs visual support is often incomplete and uneven. This paper studies OpenBG-IMG under a unified filtered-ranking protocol and shows that the current split induces a protocol-defined role–modality asymmetry: head-side targets may be image-supported, whereas tail-side targets are effectively image-unavailable. As a result, multimodal gain is not globally reliable, but local and bounded under the current evaluation protocol. To exploit this bounded gain, we study selective activation between a fusion expert and a structural expert under two deployable information boundaries: strict metadata-only query-level routing and score-aware non-answer-aware expert combination. Experiments show that strict clean routing recovers only modest gain, with regression-based clean routing reaching 0.2982 MRR. In contrast, exposing fixed-expert score patterns substantially improves recoverable gain: a learned candidate-level score-aware router reaches 0.3142 MRR, while simple full-ranking score interpolation reaches up to 0.3428 MRR. These results indicate that the main bottleneck is not merely router complexity, but the deployable information boundary under which expert combination is performed.
```

---

## 5. Introduction 可执行修改方案

### 5.1 当前问题

当前 Introduction 已经包含主要信息，但太长，且结构偏“把所有实验提前讲一遍”。主要问题：

- RQ 列表过细；
- contribution table 占篇幅；
- 贡献点 4 个偏多；
- 实验结果提前展开过多；
- “why this study is necessary” 可以更自然。

### 5.2 新版 Introduction 目标

控制在：

```text
约 2 页
6–7 段
不放大表格
贡献点 3 个
不长篇列 RQ
```

### 5.3 段落级执行方案

#### Paragraph 1：领域背景

保留 MMKGC 背景，但压缩。

要表达：

```text
KGC aims to infer missing triples.
MMKGC uses text/images to enrich entity representations.
In product KGs, visual evidence seems naturally useful.
```

不要详细列举 KGC baseline。

#### Paragraph 2：现实挑战

引出 incomplete visual support。

要表达：

```text
However, product-oriented KGs often have incomplete and uneven visual support.
Therefore, multimodal usefulness should not be assumed to be uniform.
```

#### Paragraph 3：本文核心观察

强调 OpenBG-IMG 当前协议的 role–modality asymmetry。

要表达：

```text
Under the current OpenBG-IMG protocol, target position is entangled with modality availability.
Head-side targets may have images.
Tail-side targets are effectively image-unavailable.
This creates a protocol-shaped missing-modality setting.
```

#### Paragraph 4：问题转化

从“如何更强 fusion”转到“何时激活 multimodal evidence”。

要表达：

```text
The central question is no longer simply how to build a stronger fusion model.
Instead, we ask when multimodal evidence should be activated under deployable information constraints.
```

#### Paragraph 5：方法路线

简述两级方法：

```text
First, strict metadata-only query-level routing.
Second, score-aware non-answer-aware expert combination.
```

不要在这里讲太多公式。

#### Paragraph 6：实验结论

简要给结果：

```text
Strict clean routing only modestly improves.
Score-aware combination substantially improves.
Simple interpolation can outperform learned CA-S2.
```

#### Paragraph 7：贡献点

贡献点压缩为 3 个。

### 5.4 建议删除或压缩的当前 Introduction 内容

| 当前内容 | 操作 |
|---|---|
| 4 个 RQ 的逐条展开 | 压缩成 1 段或删掉 |
| Table 1 Problem–response–validation | 建议删除或移到 Appendix，不建议放 Introduction |
| 四个 contributions | 压缩为三个 |
| 过多实验数值 | 只保留核心数值 |
| 过长的 “controlled claim” 段落 | 压缩到 2–3 句 |

### 5.5 新版 Contributions 草稿

```text
This paper makes three contributions:

1. **Protocol-aware bounded-gain diagnosis.** We show that the current OpenBG-IMG protocol induces role–modality asymmetry, which makes multimodal gain local and bounded rather than globally reliable.

2. **Selective activation under strict clean information boundaries.** We evaluate whether bounded multimodal gain can be recovered using only legal metadata available at query time, and show that direction-specific thresholding and regression-based gain prediction provide statistically supported but modest improvements.

3. **Score-aware expert combination under non-answer-aware constraints.** We show that fixed-expert score patterns expose substantially stronger deployable signal than strict query-level metadata. Learned candidate-level routing improves over strict clean routing, while simple score interpolation further indicates that the main bottleneck lies in the information boundary rather than router complexity alone.
```

---

## 6. Related Work 可执行修改方案

### 6.1 当前问题

当前 Related Work 内容整体合理，但偏长。问题主要是：

- 对 classical KGC 方法列举过多；
- 对 MMKGC fusion 方法铺得太开；
- selective routing 部分有些内容可以压缩；
- 最后解释本文定位的段落很好，但可以保留精简版。

### 6.2 目标

控制在：

```text
约 2 页
3 个小节
每小节 2–3 段
每节最后一句都引回本文
```

### 6.3 小节处理

#### 2.1 Multimodal Knowledge Graph Completion

保留：

- KGC 是基础；
- MMKGC 使用 text/image；
- fusion methods 包括 concatenation, gates, attention, relation-aware fusion；
- 本文不是 stronger-fusion paper。

压缩：

- 不详细列 KGC model categories；
- 不逐一介绍 TransE / DistMult / ComplEx / TuckER / ConvE；
- citation 可以合并。

结尾句建议：

```text
Unlike prior work that mainly asks how to fuse available modalities, this paper asks when multimodal evidence should be activated under protocol-shaped incomplete visual support.
```

#### 2.2 Missing Modality and Modality Imbalance

保留：

- incomplete modality；
- modality imbalance；
- visual evidence may be noisy or uneven；
- OpenBG-IMG 是 protocol-shaped missingness。

结尾句建议：

```text
Our focus is not only robustness to missing modalities, but the protocol-defined boundary under which multimodal gain becomes local and bounded.
```

#### 2.3 Selective Activation and Score-aware Expert Combination

保留：

- expert routing / mixture of experts；
- selective activation；
- score-based ensemble / ranking combination；
- strict metadata-only vs score-aware vs oracle 的信息边界。

压缩：

- 大规模 MoE 的背景不需要太多；
- 不需要过长解释 generic ensemble；
- 强 structural baseline 的解释保留，但压缩。

结尾句建议：

```text
This motivates our separation between strict clean routing, score-aware non-answer-aware combination, and answer-aware oracle analysis.
```

---

## 7. Section 3 可执行修改方案

### 7.1 标题修改

将当前：

```text
3 Task Setting and Method
```

改为：

```text
3 Problem Formulation and Methodology
```

理由：

- 更符合论文规范；
- 能覆盖 task setting、protocol、method 和 information boundary；
- 比 Task Setting and Method 更自然。

### 7.2 当前 Section 3 的问题

当前 Section 3 包含：

```text
3.1 Task Setting and Protocol
3.2 Query-level Clean Routing Formulations for Selective Activation
3.3 Clean Router Features, Training, and Evaluation Basis
3.4 Score-aware Expert Combination
3.5 Router Training Details
```

问题：

- 3.2、3.3、3.5 有较多重复；
- clean legality 被多次解释；
- router training details 过细；
- feature table C1–C4 可以压缩；
- evaluation lines 表有用，但太长；
- Section 3 占篇幅偏多。

### 7.3 新 Section 3 结构

建议改成：

```text
3 Problem Formulation and Methodology

3.1 Problem Definition and OpenBG-IMG Protocol
3.2 Protocol-Aware Bounded-Gain Diagnosis
3.3 Strict Query-Level Clean Routing
3.4 Score-Aware Expert Combination
3.5 Information Boundaries
```

---

### 7.4 Section 3.1 执行方案

#### 新标题

```text
3.1 Problem Definition and OpenBG-IMG Protocol
```

#### 保留内容

- triple `(h, r, t)`；
- tail prediction `(h, r, ?)`；
- head prediction `(?, r, t)`；
- filtered ranking；
- direction=both；
- OpenBG-IMG paper_split；
- dataset statistics；
- target-side regime counts。

#### 压缩内容

- KGC / MMKGC 基础解释；
- filtered ranking 细节；
- early stopping 和 checkpoint selection；
- train/dev/test workflow 只保留一句。

#### 表格处理

将当前 dataset statistics 和 regime counts 合并为一个表：

```text
Table 1: OpenBG-IMG paper_split statistics and target-side regimes.
```

表中包含两块：

```text
A. Dataset scale
#Entities, #Relations, #Train, #Valid, #Test, #Image entities, Image coverage

B. Target-side regimes
head_has_img, head_no_img, tail_no_img, tail_has_img
```

这样可以减少一个表格。

---

### 7.5 Section 3.2 执行方案

#### 新标题

```text
3.2 Protocol-Aware Bounded-Gain Diagnosis
```

#### 作用

这一节不是正式实验结果，而是定义本文为什么要做 selective activation。

#### 要表达

```text
The current protocol creates role–modality asymmetry.
This asymmetry makes multimodal gain local and bounded.
Therefore, always-on multimodal fusion may not be globally optimal.
```

#### 保留内容

- Figure 1；
- role–modality asymmetry 解释；
- bounded gain 概念；
- target position entangled with modality availability。

#### 压缩内容

- 不提前展开所有 subgroup result；
- 不要把实验结论写得太长；
- 具体 MRR 放到 Experiments。

---

### 7.6 Section 3.3 执行方案

#### 新标题

```text
3.3 Strict Query-Level Clean Routing
```

#### 合并来源

这一节应合并当前：

```text
3.2 Query-level Clean Routing Formulations
3.3 Clean Router Features, Training, and Evaluation Basis
3.5 Router Training Details 中关于 clean routing 的部分
```

#### 保留内容

必须保留：

- fusion expert = Gate-only；
- structural expert = Residual-only；
- clean legality constraint；
- global threshold baseline；
- direction-specific threshold；
- regression-based gain prediction；
- final score formula。

#### 可以删除或压缩

| 内容 | 处理 |
|---|---|
| C1–C4 feature families 表 | 可删，改为文字说明使用 direction, relation priors, observed-side modality indicators |
| relation_gain_prior 的详细定义 | 可放附录或一句话 |
| observed_text_img_cosine 等细节 | 可放附录 |
| ordinal gain buckets | 如果不是核心结果，放附录 |
| bucketized threshold variants | 如果不支撑主结论，放附录 |
| router training six-family list | 删除，太像实验记录 |

#### 建议保留公式

保留以下三个公式即可：

```text
s_final(q, e) = α(q)s_f(q, e) + (1 - α(q))s_s(q, e)
```

```text
α(q) = 1[p(q) > τ]
```

```text
Δ(q) = RR_f(q) - RR_s(q)
```

方向特定阈值可用文字或简短公式说明，不要展开太长。

#### 本节目标长度

```text
1.5–2 页
```

---

### 7.7 Section 3.4 执行方案

#### 新标题

```text
3.4 Score-Aware Expert Combination
```

#### 保留内容

- 为什么 strict clean routing 太粗；
- score-aware non-answer-aware setting；
- simple score interpolation；
- CA-S2 / CA-S3 candidate-level router；
- candidate-level mixing score；
- pairwise ranking loss 可保留，但不要长篇解释。

#### 可删除或压缩

| 内容 | 处理 |
|---|---|
| CA-S1 的方法细节 | 简短说明是 clean candidate metadata baseline |
| hard candidate set training 细节 | 放附录 |
| full filtered ranking 细节 | 一句话 |
| score/rank/confidence/disagreement 全部展开 | 压缩为 “score, rank, confidence, and disagreement features” |
| 过细训练过程 | 附录或删除 |

#### 保留公式

```text
s_mix(q, e) = α(q, e)s_f(q, e) + (1 - α(q, e))s_s(q, e)
```

```text
α(q, e) = σ(g_θ(x_q, x_e, z_{q,e}))
```

pairwise loss 可选保留：

```text
L_pair = -log σ(s_mix(q, e^+) - s_mix(q, e^-))
```

如果篇幅紧，可以把 pairwise loss 放附录。

---

### 7.8 Section 3.5 执行方案

#### 新标题

```text
3.5 Information Boundaries
```

#### 目的

用一个压缩表清楚区分：

```text
Strict clean
Score-aware non-answer-aware
Oracle/post-hoc
```

#### 表格压缩方案

当前 Table 6 信息很重要，但太长。建议改成压缩版：

| Signal Type | Strict Clean | Score-aware | Oracle |
|---|---|---|---|
| Query direction / relation priors | ✓ | ✓ | ✓ |
| Observed-side modality | ✓ | ✓ | ✓ |
| Fixed-expert candidate scores | ✗ | ✓ | ✓ |
| Rank / confidence / disagreement | ✗ | ✓ | ✓ |
| Hidden target identity | ✗ | ✗ | ✓ |
| Target-side regime / RR | ✗ | ✗ | ✓ |

这样足够说明边界。

#### 本节长度

```text
0.5 页以内
```

---

## 8. Experiments 可执行修改方案

### 8.1 当前问题

当前 Experiments 有 4 个 RQ，表格较多，部分解释重复。需要改为 3 个 RQ，并让每个 RQ 只回答一个核心问题。

### 8.2 新结构

```text
4 Experiments
   4.1 Experimental Setup
   4.2 RQ1: Is multimodal gain globally reliable?
   4.3 RQ2: How much bounded gain can strict metadata-only routing recover?
   4.4 RQ3: Does score-aware combination recover more gain?
   4.5 Discussion
```

---

### 8.3 Section 4.1 Experimental Setup

#### 保留内容

- current OpenBG-IMG paper_split；
- filtered ranking；
- direction=both；
- three seeds；
- dev used for checkpoint / threshold selection；
- test used once；
- paired bootstrap only for main routing comparisons。

#### 压缩内容

| 内容 | 操作 |
|---|---|
| 2,000 bootstrap resamples 的详细解释 | 压缩成一句 |
| seed-wise vs query-wise 详细说明 | 删除或附录 |
| appendix split 介绍 | 移到 Appendix，不在正文 setup 里展开 |
| 四条 evaluation lines 的长解释 | 已在 Section 3.5 信息边界表处理，这里不重复 |

#### 推荐长度

```text
0.5–1 页
```

---

### 8.4 Section 4.2 / RQ1

#### 标题

```text
4.2 RQ1: Is multimodal gain globally reliable?
```

#### 本节要回答

```text
No. Multimodal gain is local and bounded under the current OpenBG-IMG protocol.
```

#### 保留表格

1. Official model comparison；
2. Subgroup MRR by target-side regime；
3. Relation-group analysis 可选。

#### Official model comparison 表格处理

保留：

```text
ComplEx
Text-only
Early Fusion
Gate-only
Full Model
Residual-only
```

TuckER 是否保留：

- 如果 TuckER 是正式 baseline，保留；
- 如果篇幅紧，可以保留但不长解释。

表格列：

```text
Model | Type | MRR | Hits@1 | Hits@3 | Hits@10
```

不需要 Notes 太长。

#### Subgroup 表格处理

保留：

```text
Residual-only
Full Model
Gate-only
```

列：

```text
head_has_img | head_no_img | tail_no_img
```

这张表很关键，必须保留。

#### Relation-group analysis

建议处理：

- 如果正文还有空间，保留一个压缩版；
- 如果压缩压力大，移到 Appendix；
- 正文可以用一句话总结 relation-group result。

#### 本节结论句

必须以清晰结论收尾：

```text
These results show that multimodal gain is not globally reliable. It appears only in local protocol-shaped regimes, while the dominant tail_no_img region remains structure-favorable.
```

---

### 8.5 Section 4.3 / RQ2

#### 标题

```text
4.3 RQ2: How much bounded gain can strict metadata-only routing recover?
```

#### 本节要回答

```text
Strict clean routing can recover statistically supported but modest gain.
```

#### 合并表格

将当前以下表格合并：

```text
Main clean routing comparison
Progressive clean-routing ablation
Paired bootstrap significance
```

合并成：

```text
Table X: Strict query-level clean routing results.
```

推荐列：

```text
Method
Policy / Target
MRR
Delta vs Residual-only
Delta vs Clean rule
95% CI
Interpretation
```

推荐行：

```text
Residual-only
Clean rule
Naive global clean router
Direction-specific threshold
Regression-based clean router
Hard-selection Oracle
```

#### 表格示例结构

| Method | Policy / Target | MRR | Δ vs Residual | Δ vs Rule | 95% CI | Role |
|---|---|---:|---:|---:|---|---|
| Residual-only | fixed structural | 0.2930 | – | – | – | anchor |
| Clean rule | legal rule | 0.2943 | +0.0012 | – | – | baseline |
| Naive global router | global / binary | 0.2939 | +0.0008 | -0.0004 | – | too coarse |
| Direction-specific | direction thresholds / binary | 0.2974 | +0.0044 | +0.0032 | positive | policy granularity helps |
| Regression clean router | thresholded scalar gain | 0.2982 | +0.0051 | +0.0039 | positive | strongest strict clean |
| Hard-selection Oracle | answer-aware | 0.3337 | +0.0407 | +0.0395 | – | diagnostic only |

#### 文字压缩策略

不要逐段详细解释 E1、E5、delta sensitivity、ordinal buckets。改为：

```text
Direction-specific thresholding improves over a single global threshold, showing that policy granularity matters. Regression-based gain prediction further improves performance, suggesting that binary gain labels are too coarse. However, the best strict clean result remains far below hard-selection Oracle routing, indicating that metadata-only query-level signals expose only part of the bounded gain.
```

#### 移到 Appendix 的内容

- ordinal gain modeling；
- delta sensitivity；
- calibration；
- bucket-specific thresholding；
- detailed threshold scan；
- full bootstrap table。

---

### 8.6 Section 4.4 / RQ3

#### 标题

```text
4.4 RQ3: Does score-aware combination recover more gain?
```

#### 本节要回答

```text
Yes. Score-aware non-answer-aware expert combination recovers substantially more deployable gain than strict clean routing.
```

#### 保留两个结果层次

1. Learned candidate-level router；
2. Simple score interpolation。

#### 表格合并方案 A：分两张表

如果篇幅允许，保留两张表：

```text
Table X: Candidate-level score-aware router results.
Table Y: Full-ranking score interpolation results.
```

#### 表格合并方案 B：合并成一张大表

如果篇幅紧，合并成：

```text
Table X: Score-aware expert combination results.
```

推荐列：

```text
Method
Level
Information type
MRR
Delta vs Residual
Delta vs E5
Interpretation
```

推荐行：

```text
Residual-only
E5 regression clean router
CA-S1
CA-S2
CA-S3
Global score interpolation
Direction-specific interpolation
Relation-specific interpolation
Hard-selection Oracle
```

但如果表太长，Hard-selection Oracle 可只在 RQ2 中出现一次，不重复。

#### 推荐正文叙述顺序

```text
1. CA-S1 negative result: static candidate metadata alone is insufficient.
2. CA-S2 / CA-S3 improve substantially: score-aware features matter.
3. CA-S2 and CA-S3 are close; do not overclaim CA-S2 dominates.
4. Score interpolation further outperforms learned routers.
5. Therefore, the key is score-aware expert-combination information, not router complexity alone.
```

#### 必须避免的错误表述

不要写：

```text
CA-S2 is the best method.
```

因为 score interpolation 更强。

应该写：

```text
CA-S2 is the strongest learned candidate-level score-aware router, while simple score interpolation provides the strongest score-aware combination result.
```

#### 本节结论句

```text
These results answer RQ3: once non-answer-aware fixed-expert score patterns are available, substantially more deployable gain can be recovered than under strict metadata-only routing. The strongest result comes from simple score interpolation, suggesting that the key factor is the score-aware combination boundary rather than learned router complexity alone.
```

---

### 8.7 Section 4.5 Discussion

#### 作用

这一节负责收束全文，不要再引入大量新实验。

建议包含 3 个小段：

#### Discussion point 1：Why always-on fusion is not enough

```text
The protocol creates a large structure-favorable region, so always-on fusion is not globally reliable.
```

#### Discussion point 2：Why strict clean routing is limited

```text
Strict query-level metadata is legally deployable but too coarse to observe candidate-level expert behavior.
```

#### Discussion point 3：Why score-aware combination changes the boundary

```text
Score-aware methods remain non-answer-aware but expose candidate-level expert disagreement and confidence patterns, which contain much richer deployable signal.
```

#### 必须强调

```text
Oracle is diagnostic, not deployable.
Score interpolation is not bounded by hard-selection Oracle because it can create new mixed rankings.
Claims are protocol-aware, not universal.
```

---

## 9. 正文图表处理执行表

### 9.1 必须保留在正文

| 当前图表 | 新位置 | 操作 |
|---|---|---|
| Figure 1 role–modality asymmetry | Section 3.2 | 保留 |
| Dataset statistics | Section 3.1 | 与 regime counts 合并 |
| Target-side regime counts | Section 3.1 | 与 dataset statistics 合并 |
| Official model comparison | Section 4.2 | 保留 |
| Subgroup MRR by target-side regime | Section 4.2 | 保留 |
| Clean routing main comparison | Section 4.3 | 与 ablation / CI 合并 |
| Candidate router results | Section 4.4 | 保留或并入 score-aware main table |
| Score interpolation results | Section 4.4 | 保留 |

### 9.2 可以正文压缩保留

| 当前图表 | 操作 |
|---|---|
| Information boundary table | 压缩保留 |
| Progressive clean-routing ablation | 合并进 clean routing main table |
| Paired bootstrap clean table | 合并为 CI 列 |
| Candidate-router bootstrap table | 简化为一句或合并 |
| Relation-group analysis | 可留正文，也可放附录 |

### 9.3 优先移到 Appendix 或删除

| 当前内容 | 操作 |
|---|---|
| Full threshold scan | Appendix 或删除 |
| Delta sensitivity | Appendix |
| Calibration analysis | Appendix 或删除 |
| Ordinal gain bucket details | Appendix |
| Alpha diagnostics | Appendix |
| Degree confounding diagnostics | Appendix，正文只一句 |
| Additional split robustness check | Appendix 或删除 |
| Seed-wise detailed tables | 删除，保留 mean ± std |

---

## 10. Appendix 清理方案

### 10.1 Appendix 的保留原则

Appendix 只保留：

```text
1. 支撑 reviewer 复查的实现细节
2. 正文结论的关键补充验证
3. 放正文会打断主线但有价值的诊断
```

不要把 Appendix 当成“所有做过实验的仓库”。

### 10.2 建议保留

```text
A. Additional Implementation Details
- router feature construction
- threshold selection details
- candidate router training details

B. Additional Ablation Results
- delta sensitivity
- optional threshold variants
- ordinal gain if still有价值

C. Additional Diagnostics
- alpha diagnostics
- degree confounding diagnostics
```

### 10.3 建议删除或极简化

```text
1. appendix_split_seed20260427 完整表格
```

处理建议：

- 如果导师非常强调压缩：直接删除；
- 如果担心 robustness 不够：保留一小段 + 一张小表；
- 不建议保留长篇解释。

可压缩为：

```text
An additional relation-stratified split shows the same qualitative trend: strict clean routing improves only modestly over Residual-only, while CA-S2 remains substantially stronger. We therefore treat it as robustness evidence rather than a replacement for the main split.
```

---

## 11. 术语统一清单

全文必须统一以下术语。

### 11.1 推荐使用

```text
role–modality asymmetry
protocol-aware bounded gain
bounded multimodal gain
strict metadata-only query-level routing
strict clean routing
score-aware non-answer-aware expert combination
deployable information boundary
answer-aware Oracle analysis
fixed fusion expert
fixed structural expert
```

### 11.2 避免混用

| 不建议混用 | 推荐统一 |
|---|---|
| clean router / legal router / deployable router | strict clean routing |
| score-aware router / candidate router / CA router | score-aware non-answer-aware expert combination |
| post-hoc oracle / oracle upper bound | answer-aware Oracle analysis |
| multimodal benefit / fusion gain / gain | bounded multimodal gain |
| protocol-shaped missingness / asymmetric setting | role–modality asymmetry |

### 11.3 Claim 边界

必须避免：

```text
Our method solves missing modality.
Our router is universally stronger.
Score-aware routing fully closes the oracle gap.
CA-S2 is the best method.
```

推荐表述：

```text
Under the current OpenBG-IMG protocol...
Within this protocol-aware setting...
The results suggest...
The score-aware line recovers substantially more deployable gain...
The main bottleneck is the deployable information boundary rather than router complexity alone.
```

---

## 12. 每章字数 / 页数控制

| 章节 | 目标长度 | 操作重点 |
|---|---:|---|
| Abstract | 180–230 words | 重写 |
| Introduction | 2 页 | 重新组织逻辑 |
| Related Work | 2 页 | 压缩 citation 和背景 |
| Section 3 | 4–5 页 | 合并方法细节 |
| Experiments | 6–8 页 | 3 个 RQ，核心表格 |
| Discussion | 可并入 Experiments | 不单独拉太长 |
| Conclusion | 0.5–1 页 | 收束主线 |
| Appendix | 越短越好 | 删除低价值内容 |

---

## 13. 具体删除 / 合并清单

### 13.1 删除

优先删除：

```text
- Introduction 中的 Problem–response–validation 表
- 过长的四个 RQ 展开
- Section 3.5 Router Training Details 作为独立小节
- Clean feature families C1–C4 的完整表格
- 多个 routing family 的流水账式列举
- appendix split 的长篇结果解释
- 不改变主结论的 threshold scan
- calibration 细节
- seed-wise 详细结果
```

### 13.2 合并

必须合并：

```text
- Dataset statistics + regime counts
- Clean routing main result + progressive ablation + CI
- Candidate router results + score interpolation results
```

可选合并：

```text
- Information boundary table + method explanation
- Relation-group analysis + subgroup analysis
```

### 13.3 保留

必须保留：

```text
- Role–modality asymmetry Figure
- Official model comparison
- Target-side subgroup MRR
- Best strict clean routing result
- CA-S2 / CA-S3 result
- Score interpolation result
- Information boundary distinction
```

---

## 14. 修改后的核心实验叙事

全文实验部分应形成如下递进：

### 14.1 RQ1 叙事

```text
First, we verify that multimodal gain is not globally reliable. Residual-only is globally strongest among the internal family, while Gate-only shows relative advantage only in local target-side regimes. This confirms that the current protocol induces bounded multimodal gain.
```

### 14.2 RQ2 叙事

```text
Second, we test whether strict metadata-only routing can exploit this bounded gain. A naive global threshold is insufficient, direction-specific thresholding improves the result, and regression-based gain prediction gives the best strict clean performance. However, the improvement remains modest, showing that query-level legal metadata captures only part of the recoverable signal.
```

### 14.3 RQ3 叙事

```text
Third, we test whether score-aware non-answer-aware expert combination can recover more gain. CA-S2 and CA-S3 substantially outperform strict clean routing, while simple score interpolation performs even better. This shows that fixed-expert score patterns expose much richer deployable signal than metadata-only query features.
```

---

## 15. 最终检查清单

修改完成后逐项检查：

### 15.1 逻辑检查

- [ ] Abstract 是否清楚表达“发现问题 → 方法 → 结果 → 结论”？
- [ ] Introduction 是否自然推出研究问题？
- [ ] Related Work 是否服务于本文 gap？
- [ ] Section 3 是否从 problem formulation 过渡到 methodology？
- [ ] Experiments 是否按 3 个 RQ 组织？
- [ ] 每个 RQ 是否有明确结论句？
- [ ] 全文是否始终围绕 bounded multimodal gain？

### 15.2 篇幅检查

- [ ] Introduction 是否控制在约 2 页？
- [ ] Related Work 是否控制在约 2 页？
- [ ] Section 3 是否不超过 5 页？
- [ ] Experiments 是否不超过 8 页？
- [ ] Appendix 是否明显变短？
- [ ] 是否删除了低价值附录内容？

### 15.3 图表检查

- [ ] 每张正文图表是否回答一个核心问题？
- [ ] 是否合并了 dataset statistics 和 regime counts？
- [ ] 是否合并了 clean routing 相关表格？
- [ ] 是否合并或压缩了 score-aware 相关表格？
- [ ] 是否删除了只证明“我试过”的表格？
- [ ] 是否避免正文和附录重复展示同一结论？

### 15.4 Claim 边界检查

- [ ] 是否避免声称提出 globally strongest MMKGC model？
- [ ] 是否明确 claim 是 under current OpenBG-IMG protocol？
- [ ] 是否明确 Oracle 是 diagnostic，不是 deployable？
- [ ] 是否明确 score interpolation 不受 hard-selection Oracle 上界约束？
- [ ] 是否避免说 CA-S2 是 overall best？
- [ ] 是否强调 information boundary 是关键瓶颈？

### 15.5 术语检查

- [ ] role–modality asymmetry 是否全篇统一？
- [ ] bounded multimodal gain 是否全篇统一？
- [ ] strict clean routing 是否全篇统一？
- [ ] score-aware non-answer-aware expert combination 是否全篇统一？
- [ ] deployable information boundary 是否全篇统一？

---

## 16. 推荐下一步实际操作

建议下一步直接进入正文修改，但不要全文一起改。

最优顺序：

```text
1. 先重写 Abstract
2. 再重写 Introduction
3. 确认新版主线后，再改 Section 3
4. Section 3 定稿后，再重排 Experiments
5. 最后处理 Appendix
```

原因：

- Abstract 和 Introduction 是主线源头；
- 如果主线不先定，后面 Method 和 Experiments 会反复返工；
- Section 3 和 Section 4 的压缩都依赖 Introduction 中如何定义贡献。

因此，下一步最具体的任务应该是：

```text
先重写 Abstract + Introduction。
```

完成这两部分后，再继续处理 Section 3 和 Experiments。
