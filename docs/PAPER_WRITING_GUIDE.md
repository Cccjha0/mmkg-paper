# Paper Writing Guide

## 0. Thesis Statement

本文的核心论点不是“提出一个全局最优的多模态模型”，而是：

> 在当前 OpenBG-IMG 协议下，多模态收益是真实存在的，但这种收益具有明确边界；其大小受到目标位置、模态可用性、关系类型以及结构补偿强度的共同限制。

更具体地说，本文希望回答三个问题：

- 多模态信息何时真正有用？
- 为什么这种收益没有在当前协议下成为全局主导？
- 为什么强结构补偿路径，尤其是 residual-dominant 路径，仍然保持整体优势？

当前论文身份应明确为：

- `analysis-driven first`
- `method-improvement second`

## 1. Purpose

本文档用于把当前项目材料重组为“论文写作导向”的结构。

它不再承担实验计划或进度管理职责，而是回答一个实际问题：

> 如果现在开始正式写论文，每一节应该使用哪些文档、主张哪些结论、避免哪些过度表述？

当前项目的实验闭环已经基本完成，核心证据链已经具备：

- 主结果
- 分组分析
- 行为分析
- 案例分析
- 最终主线判定

因此，这份 guide 的作用是把“已成熟的证据”映射到“可直接落稿的章节结构”。

## 2. Paper Identity and Claim Hierarchy

### 2.1 Paper Identity

当前最稳妥、也最符合证据的论文身份是：

- `analysis-driven first`
- `method-improvement second`

也就是说，论文的主要价值不在于宣称 `Full Model` 全局最强，而在于：

- 诊断多模态收益的出现条件
- 解释多模态收益为何受限
- 说明结构补偿为何在当前协议下保持主导

### 2.2 Strongest Safe Claim

当前最强、但仍然安全的主张是：

- under the current protocol, multimodal gain in OpenBG-IMG is real but bounded
- gain depends on target position, image availability, relation characteristics, and branch competition
- stronger structural compensation remains globally dominant under the current protocol

### 2.3 Claims to Avoid

以下表述应避免：

- “we propose a globally stronger multimodal model”
- “our Full Model is the best overall model”
- “visual relations always favor multimodal models”
- “TuckER is not a strong baseline”

关于 `TuckER`，更准确的说法是：

> under the current protocol and split, TuckER does not emerge as a competitive structural baseline

## 3. Suggested Paper Structure

推荐章节顺序如下：

1. Introduction
2. Related Work
3. Task Setting and Experimental Protocol
4. Models and Compared Methods
5. Main Results
6. Gain-Boundary Analysis
7. Behavior Analysis
8. Case Study
9. Discussion, Limitations, and Conclusion

这套结构的优点是：

- 先用主结果建立张力
- 再用 gain-boundary analysis 给出核心贡献
- 最后用 behavior 与 case study 解释机制和边界

## 4. Introduction

### 4.1 Goal of the Section

本节需要完成三件事：

1. 引出 MMKGC 在缺失视觉模态场景下的现实问题
2. 建立“多模态承诺”与“结构主导现实”之间的张力
3. 明确本文的核心问题不是“如何证明多模态一定更强”，而是：
   - when does multimodal information help?
   - why is that gain bounded under the current protocol?

### 4.2 Best Source Files

- [FINAL_NARRATIVE_DECISION.md](/E:/learn/R&D/mmkg-project-research/docs/FINAL_NARRATIVE_DECISION.md)
- [GAIN_BOUNDARY_JUDGMENT.md](/E:/learn/R&D/mmkg-project-research/docs/GAIN_BOUNDARY_JUDGMENT.md)
- [PAPER_PLAN.md](/E:/learn/R&D/mmkg-project-research/docs/PAPER_PLAN.md)
- [README.md](/E:/learn/R&D/mmkg-project-research/README.md)

### 4.3 Intro Claims to Carry

建议在 Introduction 中明确：

- 本文不以证明 `Full Model` 全局最优为目标
- 本文关注多模态收益的边界条件
- 当前协议下最强竞争因素不是纯融合模型，而是强结构补偿
- 在缺失视觉模态条件下，多模态并不天然优于结构方法

## 5. Related Work Strategy

### 5.1 Goal of the Section

相关工作不应写成“我们提出了另一个融合模块”，而应写成：

1. 现有研究如何处理 MMKGC 中的多模态融合
2. 现有研究如何处理 missing modality / modality imbalance / modality noise
3. 为什么本文虽然也包含 fusion / residual 结构，但研究主线不同

### 5.2 Three Related-Work Groups

#### (1) MMKGC foundations

介绍 MMKGC 的基本任务设定与多模态使用方式。

#### (2) Relation-aware / adaptive fusion

介绍 relation-aware fusion、adaptive fusion、link-aware fusion 等思路。

#### (3) Missing-modality / robustness studies

介绍缺失模态、模态不均衡、模态质量问题相关研究。

### 5.3 Our Positioning

建议明确写成：

> 本文并不声称首次研究缺失视觉模态或关系感知融合，而是在已有研究基础上，进一步关注在当前协议下，多模态收益何时出现、为何受限，以及结构补偿为何仍保持全局主导。

### 5.4 What Makes This Paper Different

这一节最好补一句，避免被读成常规“新模型论文”：

- 本文不是单纯提出一个新的 fusion 模块
- 本文也不是单纯构造一个 missing-modality benchmark
- 本文把“多模态收益边界”本身作为研究对象

## 6. Task Setting and Experimental Protocol

### 6.1 Goal of the Section

本节需要说明：

- dataset and split
- missing-modality asymmetry in the current `paper_split`
- unified evaluation protocol
- seed reporting / early stopping / filtered ranking rules

### 6.2 Best Source Files

- [EXPERIMENT_EXECUTION_TODO.md](/E:/learn/R&D/mmkg-project-research/docs/EXPERIMENT_EXECUTION_TODO.md)
- [HAS_IMG_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/HAS_IMG_ANALYSIS.md)
- [HAS_IMG_SPLIT_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/HAS_IMG_SPLIT_SUMMARY.md)
- [RELATION_TYPE_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_TYPE_ANALYSIS.md)
- [build_openbg_img_paper_split.py](/E:/learn/R&D/mmkg-project-research/ml/training/scripts/build_openbg_img_paper_split.py)
- [check_paper_split_has_img_distribution.py](/E:/learn/R&D/mmkg-project-research/ml/training/scripts/check_paper_split_has_img_distribution.py)

### 6.3 Key Protocol Facts

必须明确写出：

- evaluation uses filtered ranking on `test`
- model selection uses `best.ckpt` based on `dev`
- evaluation direction is `both`
- each reported model uses at least 3 seeds
- in the current `paper_split`, head targets may have images, while tail targets are effectively always `no_img`

### 6.4 Protocol-Specific Limitation

这一点要上升到 limitation 层级：

> 当前 `paper_split` 存在显著的 target-position × modality-availability 非对称性：head 侧仍可能具备图像支持，而 tail 侧在目标预测上基本处于 `no_img` 状态。

这意味着：

- target position 不只是普通设置差异，而是结论的重要解释变量
- 本文关于 gain boundary 的结论是在当前协议下成立的
- 该不对称性既解释了多模态局部收益的出现，也限制了结论的泛化范围

## 7. Models and Compared Methods

### 7.1 Goal of the Section

定义所有比较模型，并说明各自的概念角色，而不是只堆模型名。

### 7.2 Best Source Files

- [EXPERIMENT_EXECUTION_TODO.md](/E:/learn/R&D/mmkg-project-research/docs/EXPERIMENT_EXECUTION_TODO.md)
- [MAIN_RESULTS_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/MAIN_RESULTS_SUMMARY.md)
- [build_model.py](/E:/learn/R&D/mmkg-project-research/ml/training/src/models/build_model.py)
- [openbg_img_gated_lp.py](/E:/learn/R&D/mmkg-project-research/ml/training/src/models/openbg_img_gated_lp.py)

### 7.3 Model Groups to Introduce

#### Internal model family

- `Text-only`
- `Early Fusion`
- `Gate-only`
- `Residual-only`
- `Full Model`

#### Structural baselines

- `ComplEx`
- `TuckER`

### 7.4 How to Frame Each Group

- `Text-only`：最弱的单模态文本参考
- `Early Fusion`：最直接的多模态融合参考
- `Gate-only`：关系感知融合主干
- `Residual-only`：强结构补偿参考，也是当前协议下最强内部竞争者之一
- `Full Model`：最完整的 multimodal variant，用来研究 fusion 与 residual 的协同/竞争关系
- `ComplEx`：当前结果上真正具有竞争力的 structural baseline
- `TuckER`：经典 structural reference baseline，但在当前协议下没有表现出竞争性的强基线效果

### 7.5 Special Note on TuckER

必须明确：

> 在当前 `direction=both` 与 `paper_split` 的 head/tail 不对称设置下，TuckER 并未表现出具有竞争力的强基线效果，尤其在 head-side prediction 上表现较弱。

但不要写成：

- TuckER 本身不强
- TuckER 不适合这个任务

更准确的写法是：

> Under the current protocol and data distribution, TuckER does not emerge as a competitive structural baseline.

## 8. Main Results

### 8.1 Goal of the Section

这一节的任务是建立全局经验事实，而不是解释机制。

### 8.2 Best Source Files

- [MAIN_RESULTS_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/MAIN_RESULTS_SUMMARY.md)
- [main_results_summary.json](/E:/learn/R&D/mmkg-project-research/docs/main_results_summary.json)
- [RESULT_INDEX.md](/E:/learn/R&D/mmkg-project-research/docs/RESULT_INDEX.md)

### 8.3 Writing Role

这一节要建立一种“张力”：

- `Full Model` 相比更简单的 multimodal baselines 有提升
- 但它仍未在全局上超过更强的结构补偿替代方案

这就是后面所有分析章节的起点。

### 8.4 Main-Results Writing Principles

主结果部分必须做到：

1. 先给出整体 test results
2. 明确 strongest competitors 是谁
3. 区分：
   - strong global competitors
   - classical but non-competitive reference baselines

因此，在当前协议下：

- `Residual-only` 与 `ComplEx` 是真正构成张力的强对手
- `TuckER` 可作为经典参考模型保留，但不应写成 strongest structural baseline

### 8.5 Important Reminder

主结果本身不能直接充当全文最终结论。

它的作用是建立经验张力，真正的论文结论来自：

- `6` 的 gain-boundary analysis
- `7` 的 behavior analysis
- `8` 的 case study

也就是说，Main Results 回答的是“发生了什么”，后续分析回答的是“为什么如此、边界在哪”。

## 9. Gain-Boundary Analysis

这是论文最重要的分析章节。

### 9.1 Section Goal

说明 multimodal gain：

- is real
- is local
- is conditional
- is bounded

### 9.2 `has_img / no_img`

#### Best Source Files

- [HAS_IMG_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/HAS_IMG_ANALYSIS.md)
- [HAS_IMG_SPLIT_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/HAS_IMG_SPLIT_SUMMARY.md)
- [has_img_split_summary.json](/E:/learn/R&D/mmkg-project-research/docs/has_img_split_summary.json)

#### Core Conclusion

- multimodal gain 最有利的 regime 出现在 `head_has_img`
- overall metrics 仍被 `tail_noimg` 主导

#### Writing Role

说明多模态收益与：

- target position
- modality availability

密切相关。

### 9.3 Relation Type

#### Best Source Files

- [RELATION_TYPE_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_TYPE_ANALYSIS.md)
- [RELATION_TYPE_SUMMARY_ALL.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_TYPE_SUMMARY_ALL.md)
- [RELATION_TYPE_SUMMARY_MIN20.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_TYPE_SUMMARY_MIN20.md)
- [relation_type_summary_all.json](/E:/learn/R&D/mmkg-project-research/docs/relation_type_summary_all.json)
- [relation_type_summary_min20.json](/E:/learn/R&D/mmkg-project-research/docs/relation_type_summary_min20.json)

#### Core Conclusion

- `Full Model` 在不少支持度足够的 relation 上优于 `Gate-only`
- 但相对 `Residual-only` 的优势仅出现在少数 relation
- 粗粒度视觉分组无法支持“视觉关系必然利好多模态”的简单叙事

### 9.4 Gain-Boundary Synthesis

#### Best Source Files

- [GAIN_BOUNDARY_JUDGMENT.md](/E:/learn/R&D/mmkg-project-research/docs/GAIN_BOUNDARY_JUDGMENT.md)
- [FINAL_NARRATIVE_DECISION.md](/E:/learn/R&D/mmkg-project-research/docs/FINAL_NARRATIVE_DECISION.md)

#### Core Conclusion

- multimodal gain is local
- multimodal gain is conditional
- multimodal gain is bounded by:
  - target position
  - modality availability
  - relation characteristics
  - structural compensation strength

### 9.5 Important Writing Constraint

这里必须写清楚：

> 这些 gain-boundary 结论是在当前 protocol 下成立的；它们不是“所有 MMKGC 数据集都如此”的普遍定律，而是当前 split 与模型行为共同作用下的经验规律。

## 10. Behavior Analysis

### 10.1 Goal of the Section

解释机制问题：

- 为什么 `Full Model` 优于 `Gate-only`
- 但为什么它仍然没能全局超过 `Residual-only`

### 10.2 Best Source Files

- [BEHAVIOR_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/BEHAVIOR_SUMMARY.md)
- [RELATION_AWARE_GATE_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_AWARE_GATE_SUMMARY.md)
- [GATE_RELATION_JUDGMENT.md](/E:/learn/R&D/mmkg-project-research/docs/GATE_RELATION_JUDGMENT.md)
- [RELATION_AWARE_RESIDUAL_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_AWARE_RESIDUAL_SUMMARY.md)
- [FUSION_VS_RESIDUAL_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/FUSION_VS_RESIDUAL_ANALYSIS.md)

### 10.3 Section Role

这一节在论文中的角色是“解释层”，不是新的主发现层。

它的任务是解释：

- why the gain boundary appears
- why fusion helps locally but does not dominate globally

而不是引入一个与 gain-boundary 并列竞争的新中心结论。

### 10.4 Gate Behavior

#### Core Conclusion

- gate 会随 relation context 变化
- 但 gate 本身不足以单独决定最终性能格局

短句可以保留：

- `gate is relation-aware, but not relation-dominant`

### 10.5 Residual Behavior

#### Core Conclusion

- residual contribution 在 image support 弱的场景下更强
- 尤其在 `head_noimg` 或其他模态受限子群中更明显

### 10.6 Fusion vs Residual

#### Core Conclusion

- fusion 确实提供局部增益
- 但最终分支偏好仍然呈现 residual-dominant

短句可以保留：

- `residual-dominant asymmetric complementarity`

### 10.7 Evidence Standard

这一节所有机制性结论，必须对应至少一个定量证据，例如：

- relation-level gate variance
- subgroup-level residual-strength shift
- branch preference distribution
- fusion vs residual preference summary

不要只写“看起来像”，必须写“统计上支持”。

## 11. Case Study

### 11.1 Goal of the Section

Case study 的作用是支持前面的分析，而不是代替分析。

### 11.2 Best Source Files

- [CASE_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/CASE_ANALYSIS.md)
- [case_analysis.json](/E:/learn/R&D/mmkg-project-research/docs/case_analysis.json)
- [CASE_ANALYSIS_INTERPRETATION.md](/E:/learn/R&D/mmkg-project-research/docs/CASE_ANALYSIS_INTERPRETATION.md)
- [CASE_CONCLUSION.md](/E:/learn/R&D/mmkg-project-research/docs/CASE_CONCLUSION.md)

### 11.3 Recommended Main-Text Cases

Success cases:

- `佩戴方式`
- `裙长`
- `细分风格`

Failure cases:

- `适用场景`
- `材质`
- `净含量`

### 11.4 Case Selection Rule

为避免 cherry-picking，建议明确：

- 案例应从已定义 subgroup 中按固定规则选取
- 优先选：
  - 模型差距显著
  - 语义可解释
  - 与前面 subgroup 分析一致的样本
- 每类案例应至少包含：
  - 一个 multimodal-favorable case
  - 一个 structure-favorable case

### 11.5 Writing Role

最终要让 case study 支撑下面这个判断：

- multimodal-favorable cases 更多集中在 `head + has_img`
- structure-favorable failures 更多集中在 `tail + no_img`

## 12. Discussion, Limitations, and Conclusion

### 12.1 Best Source Files

- [FINAL_NARRATIVE_DECISION.md](/E:/learn/R&D/mmkg-project-research/docs/FINAL_NARRATIVE_DECISION.md)
- [GAIN_BOUNDARY_JUDGMENT.md](/E:/learn/R&D/mmkg-project-research/docs/GAIN_BOUNDARY_JUDGMENT.md)
- [CASE_CONCLUSION.md](/E:/learn/R&D/mmkg-project-research/docs/CASE_CONCLUSION.md)

### 12.2 Discussion Goals

Discussion 不是重复结果，而是完成三件事：

1. 解释为什么多模态收益没有成为全局主导
2. 说明当前协议对结论的影响
3. 给出对未来 MMKGC 研究的启示

### 12.3 Key Points to Emphasize

- multimodal information is useful, but not uniformly useful
- stronger structural compensation remains globally dominant under the current protocol
- the right research question is not only how to fuse modalities, but when fusion actually matters under missing-visual conditions

### 12.4 Limitations

至少应写出：

- 当前 `paper_split` 的 target-position × modality-availability asymmetry
- gain-boundary 结论在当前 protocol 下成立
- `TuckER` 等模型未做单独最优调参，因此其结果应理解为统一协议下的参考表现，而非其绝对能力上限

### 12.5 Conclusion Writing Advice

结论不要承诺：

- globally superior architecture
- universal multimodal dominance

结论应强调：

- careful diagnosis
- bounded gain
- protocol-aware interpretation
- implications for future MMKGC under incomplete modality settings

## 13. Figures and Tables Checklist

最少应包含：

1. **Main results table**
   source: [MAIN_RESULTS_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/MAIN_RESULTS_SUMMARY.md)
2. **has_img / no_img subgroup table**
   source: [HAS_IMG_SPLIT_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/HAS_IMG_SPLIT_SUMMARY.md), [HAS_IMG_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/HAS_IMG_ANALYSIS.md)
3. **Relation-type grouped table**
   source: [RELATION_TYPE_SUMMARY_MIN20.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_TYPE_SUMMARY_MIN20.md), [RELATION_TYPE_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_TYPE_ANALYSIS.md)
4. **Behavior summary figure / compact table**
   source: [BEHAVIOR_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/BEHAVIOR_SUMMARY.md), [FUSION_VS_RESIDUAL_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/FUSION_VS_RESIDUAL_ANALYSIS.md)
5. **Case-study table**
   source: [CASE_ANALYSIS_INTERPRETATION.md](/E:/learn/R&D/mmkg-project-research/docs/CASE_ANALYSIS_INTERPRETATION.md), [CASE_CONCLUSION.md](/E:/learn/R&D/mmkg-project-research/docs/CASE_CONCLUSION.md)

### Optional but Recommended

1. **Protocol asymmetry figure**
   直观展示：
   - head / tail
   - has_img / no_img
   的不对称分布

这张图会非常有助于 reviewer 理解全文主线。

## 14. Chapter-to-Document Map

如果写作时需要快速定位材料，可以按下面映射：

| 论文部分 | 优先文档 |
|---|---|
| Introduction | `FINAL_NARRATIVE_DECISION`, `GAIN_BOUNDARY_JUDGMENT` |
| Task Setting | `HAS_IMG_ANALYSIS`, `EXPERIMENT_EXECUTION_TODO` |
| Models | `EXPERIMENT_EXECUTION_TODO`, `MAIN_RESULTS_SUMMARY` |
| Main Results | `MAIN_RESULTS_SUMMARY`, `RESULT_INDEX` |
| Gain-Boundary Analysis | `HAS_IMG_ANALYSIS`, `RELATION_TYPE_ANALYSIS`, `GAIN_BOUNDARY_JUDGMENT` |
| Behavior Analysis | `GATE_RELATION_JUDGMENT`, `RELATION_AWARE_RESIDUAL_SUMMARY`, `FUSION_VS_RESIDUAL_ANALYSIS` |
| Case Study | `CASE_ANALYSIS_INTERPRETATION`, `CASE_CONCLUSION` |
| Discussion / Conclusion | `FINAL_NARRATIVE_DECISION`, `CASE_CONCLUSION` |

## 15. Minimal Writing Order

最高效的写作顺序：

1. `Main Results`
2. `Gain-Boundary Analysis`
3. `Behavior Analysis`
4. `Case Study`
5. `Related Work`
6. `Introduction`
7. `Discussion, Limitations, and Conclusion`

原因：

- 当前最成熟的是结果和分析
- Introduction 应该最后写，保证和证据完全一致

## 16. Final Reminder

如果后面写作时在两种说法之间犹豫，优先选择更贴证据的版本。

### Good

- we analyze when multimodal gain appears and why it remains bounded
- under the current protocol, structural compensation remains globally dominant
- TuckER does not emerge as a competitive structural baseline under the current split

### Bad

- we propose a globally stronger multimodal model
- multimodal information consistently outperforms structural alternatives
- TuckER is not a strong baseline
