# Paper Revision Master Checklist

## Purpose

本文档用于将当前论文初稿的总体修改建议整理成一份**按优先级排列、可直接执行的整篇修改清单**。

当前稿件已经具备以下几个关键优点：

- 主线已经稳定为 **analysis-driven first**
- 核心问题明确为 **multimodal gain is real but bounded**
- 协议意识较强，能够把 `paper_split` 的非对称性纳入解释与 limitation
- 主结果、gain-boundary analysis、behavior analysis、case study 之间已经形成较完整的证据链

因此，当前阶段的任务不是推翻重写，而是把这篇论文从“结构正确的初稿”推进到“证据完整、文献扎实、表达统一、可投稿的规范稿”。

---

## P0：必须先完成的修改

### 1. 补全真正的论文前置件

当前主文已经包含 01–09 的正文文件，但从“完整论文初稿”的角度看，仍缺少至少两块关键内容：

- `Abstract`
- `References / citation integration`

#### 当前问题
- 目前正文主线已经成熟，但摘要尚未正式落稿。
- Related Work 还未真正接入具体文献引用，因此参考文献系统尚未成型。

#### 修改目标
- 写出一版正式摘要，并确保其与正文主线完全一致。
- 建立完整参考文献体系，特别是补齐 Related Work 所依赖的代表性论文引用。

#### 摘要必须覆盖的核心内容
- `Residual-only` 在当前协议下全局最强
- `Full Model` 明显强于更简单的 multimodal baselines
- 多模态收益最明显的局部 regime 出现在 `head_has_img`
- 论文贡献是 **gain-boundary analysis**，而不是 globally superior multimodal architecture

---

### 2. 把 Related Work 从“概念综述”升级成“论文综述”

这是当前最明显的短板之一。

#### 当前问题
当前 Related Work 的逻辑框架是对的，但还更像“方向综述”而不是“论文综述”。它已经区分了：

- MMKGC foundations
- adaptive / relation-aware fusion
- missing-modality / robustness

但目前仍缺少：

- 真实代表论文
- 方法差异说明
- 与本文工作的明确对比定位

#### 修改目标
把 Related Work 写成真正的“论文对论文”的综述，而不是“概念对概念”的综述。

#### 必做事项
- 每一类研究方向至少补入若干篇代表性论文
- 每类明确写出：
  - 别人解决了什么问题
  - 别人主要强调什么
  - 本文与其差异在哪
- 在 Related Work 结尾明确落脚：
  - 本文不是 another stronger fusion module paper
  - 本文关注的是 **when multimodal gain appears and why it remains bounded under missing-visual conditions**

---

### 3. 冻结主结果口径

当前 Main Results 已经明确了总排序：

- `Residual-only > ComplEx > Full Model > Gate-only > Early Fusion > Text-only > TuckER`

#### 当前问题
后续所有分析都建立在主表之上，因此必须确认主结果口径已经完全冻结。否则后面极易出现“正文一套结果、图表一套结果、答辩又是另一套结果”的不一致。

#### 修改目标
确认主结果是全篇唯一官方版本。

#### 必查事项
- 是否全部来自最终 `paper_split`
- 是否全部基于最终统一协议
- 是否全部基于最终 3-seed aggregation
- 是否所有 baseline 使用同一 reporting rule
- 是否各章引用的排名、优势关系与主表完全一致

---

### 4. 做出完整的主文表图

当前正文里仍有 `Table X` 这类 placeholder，说明图表体系还没有真正落地。

#### 修改目标
尽快完成主文所需的核心表图，使论文从“文字初稿”升级为“完整 paper package”。

#### 主文最少应包含的图表
1. Main results 总表  
2. `has_img / no_img` subgroup 表  
3. relation-type grouped / MIN20 表  
4. behavior summary figure 或 compact table  
5. case study summary table  

#### 强烈建议加入
6. `paper_split` 协议非对称性示意图（protocol asymmetry figure）

---

## P1：最重要的内容强化

### 5. 给 Introduction 补一组清晰的 numbered contributions

#### 当前问题
Introduction 的逻辑已经很完整，但还缺少一组足够清晰、正式的 contributions 列表。  
目前正文已经表达了研究问题与主线，但如果没有明确贡献点，整体上会更像“项目说明”而不是“论文开头”。

#### 修改目标
在 Introduction 末尾补上一组编号式贡献点，并压缩为 3 条左右。

#### 推荐结构
1. 提供 unified empirical comparison：系统比较 structural 与 multimodal model family  
2. 揭示 multimodal gain 的边界条件：target position、image availability、relation characteristics  
3. 通过 behavior analysis 与 case evidence 解释 gain boundary 的形成机制  

---

### 6. 强化 protocol-aware limitation 的前置提醒

#### 当前问题
Task Setting 中已经把 `paper_split` 的非对称性讲清楚了，但这类 limitation 只放在中间章节仍然不够。  
如果 reviewer 只读摘要、引言和主结果，很可能会忽略这个限制条件，从而误读结论。

#### 修改目标
让 protocol limitation 不只存在于 Task Setting，而要在全文关键位置形成“前置提醒—中段展开—结尾回收”的闭环。

#### 建议加入的位置
- Introduction 末尾：轻度提醒  
- Task Setting：正式展开  
- Discussion / Limitations：明确回收并强调泛化边界  

---

### 7. 把 TuckER 的定位彻底稳定下来

#### 当前问题
现在对 TuckER 的处理已经比之前成熟很多，但后续最怕的就是全篇口径不统一。  
你已经明确：TuckER 在当前协议下不具竞争性，但它仍是 classical reference baseline，而不是“无意义 baseline”。

#### 修改目标
确保 TuckER 的定位在全文任何位置都不再漂移。

#### 全文统一口径
- TuckER 是 classical structural reference baseline  
- 在当前 protocol and split 下，它 **does not emerge as a competitive structural baseline**
- 当前真正构成全局竞争压力的是 `Residual-only` 和 `ComplEx`

#### 必须避免的写法
- “TuckER is not a strong baseline”
- “TuckER is weak in general”
- “TuckER is not suitable for this task”

---

### 8. 把 protocol asymmetry figure 升为主文默认图

#### 当前问题
虽然之前把它列为 optional，但从正文逻辑来看，这张图实际上已经接近“主文必要图”。

#### 修改目标
将其正式纳入正文，而不是放到 appendix 或仅作为可选项。

#### 这张图的作用
- 解释为什么 `head_has_img` 是最有利的 multimodal regime
- 解释为什么 `tail_no_img` 主导整体指标
- 解释为什么 gain-boundary 结论必须 protocol-aware

---

## P2：结构与篇幅优化

### 9. 精简 Introduction，避免把后文分析提前讲太满

#### 当前问题
Introduction 的逻辑是对的，但略微有些“把后文分析提前展开”。  
当前 introduction 已经涉及：

- subgroup analysis
- relation-type analysis
- behavior analysis
- case analysis

这些内容可以保留，但应该更高层、更压缩。

#### 修改目标
让 Introduction 更像“提出问题与总设计”，而不是“提前展开分析正文”。

#### 建议处理
- 保留 tension
- 保留 main question
- 保留 unified protocol / multi-view analysis 的高层概述
- 把具体分析内容压缩成一句或一个短段落

---

### 10. 减少 Task Setting 与 Models 之间的重复

#### 当前问题
Task Setting 已经写了 Compared Models 的概览，而 Models 章节又完整展开所有模型家族，这会造成内容重复。

#### 修改目标
让章节边界更清楚：

- `Task Setting and Protocol` 只负责：
  - task
  - dataset
  - split
  - evaluation
  - reporting rule
- `Models and Compared Methods` 负责：
  - 各模型设计逻辑
  - internal family 对比意义
  - structural baselines 的角色

---

### 11. 压缩 Behavior Analysis 的篇幅，让它始终服务于主问题

#### 当前问题
Behavior Analysis 现在已经写得很成熟，但这一章最容易失控：  
如果写得过大，它会让论文从“gain-boundary paper”变成“内部机制分析 paper”。

#### 修改目标
严格把行为分析限定在“解释层”。

#### 这一章只应该回答的两个问题
1. 为什么 `Full Model` 能稳定优于 `Gate-only`  
2. 为什么 `Full Model` 仍然难以全局超过 `Residual-only`

#### 不应做的事
- 不再引入与 gain-boundary 并列的新中心故事
- 不把 behavior 章节写成单独的理论贡献

---

### 12. 缩短 Case Study 主文篇幅

#### 当前问题
Case Study 的选例逻辑很好，也有 selection rule 来避免 cherry-picking，但正文中如果案例过多，会造成篇幅失衡。

#### 修改目标
保留 case study 的支撑作用，同时让它不喧宾夺主。

#### 建议主文保留
- 1–2 个 success cases
- 1–2 个 failure cases

其余案例放到 appendix。

#### 主文中 case study 的职责
- 验证前文的 statistical pattern
- 说明 gain boundary 在 sample level 上可见
- 不负责替代 subgroup / relation / behavior 的主证据地位

---

## P3：最后的投稿级打磨

### 13. 统一全文的“句型口径”

#### 当前问题
你现在已经形成了一套很好的安全表达，但正式投稿前还需要做一次全局统一检查。

#### 必须统一复用的安全表达
- multimodal gain is real but bounded
- under the current protocol
- structure-heavy / structure-dominant compensation
- local, conditional, bounded
- not globally dominant

#### 必须统一避免的危险表达
- globally superior multimodal architecture
- visual relations always favor multimodal models
- TuckER is not a strong baseline
- multimodal information consistently outperforms structural alternatives

---

### 14. 去掉 markdown / 工程痕迹，统一 paper 风格

#### 当前问题
从 manuscript 角度看，正文已经很像论文，但最后还需要做一次 paper-style normalization。

#### 检查清单
- 模型名反引号是否全篇统一
- `mean +- std` 是否统一成正式论文写法
- `head_has_img / tail_noimg` 命名是否一致
- section title 是否保持正式论文语气
- 避免过多工程化表述（例如“当前项目”“paper-facing numbers”等）

---

### 15. 为 appendix 提前规划内容

#### 当前问题
你现在已有很多分析材料，不可能全部塞进主文。  
如果没有提前做 appendix 规划，后面很容易要么主文过长，要么删掉很多有价值证据。

#### 修改目标
提前分层管理主文与附录。

#### 建议放入 appendix 的内容
- 更细的 relation-level 表
- 更多案例
- 更细的 behavior statistics
- baseline 训练细节
- 更完整的 subgroup 明细
- 额外 ablation / supplementary notes

---

## 推荐执行顺序

### Step 1
补全 `Abstract`、参考文献体系、主文核心表图。  
这是把当前稿件从“正文初稿”升级成“完整 paper package”的关键一步。

### Step 2
重写并补强 `Related Work`。  
这是当前最明显的内容短板，也是 reviewer 最容易抓住的问题之一。

### Step 3
精修 `Introduction`，补 3 条 contribution，并略微压缩分析细节。  
让全文开头更像论文而不是项目说明。

### Step 4
压缩 `Behavior Analysis` 与 `Case Study`，确保它们服务主线，而不与主线竞争。

### Step 5
做最后的 protocol-aware polishing：  
统一 TuckER 口径、统一 gain-boundary 口径、统一 limitation 表述。

---

## Final Summary

当前这篇论文最需要的不是“大改方向”，而是把**结构正确的初稿**推进成**文献扎实、表图完整、语气统一、边界明确的投稿稿**。

一句话总结：

> 主线已经成立，剩下是精修工程。

具体来说，当前最关键的三件事是：

1. 补全摘要、参考文献与主文表图  
2. 强化 Related Work 的真实文献支撑  
3. 压缩局部章节篇幅，让 gain-boundary 主线更加突出
