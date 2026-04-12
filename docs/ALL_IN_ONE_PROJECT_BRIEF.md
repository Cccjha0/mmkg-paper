# 项目与论文总览

## 1. 这份文档的用途

这是一份面向“第一次接触该项目的人”的 all-in-one 总览文档。

它的目标是用一份材料把以下信息串起来：

- 项目背景与研究动机
- 当前论文的目标与主线
- 数据集、协议、模型与实验设计
- 已完成的所有关键实验及其主要结果
- 当前论文已经写到什么程度
- 接下来阅读现有文档的建议顺序

如果你已经拿到了 `docs/paper_manuscript/` 目录下的论文草稿文件，那么这份文档可以作为进入该目录之前的“上下文说明书”。

---

## 2. 项目背景

本项目研究的是商品领域的多模态知识图谱补全（MMKGC, Multimodal Knowledge Graph Completion）。

具体问题是：当知识图谱中的实体同时具有结构信息、文本信息和图像信息时，能否利用多模态信息提升链接预测性能；以及在图像模态不完整、不均衡甚至缺失时，多模态信息到底什么时候真正有用。

项目使用的数据集是 OpenBG-IMG。这个数据集天然包含商品实体及其属性关系，同时部分实体带有图片，因此非常适合研究“视觉信息是否真的能帮助知识图谱补全”这一问题。

---

## 3. 项目最初目标与当前目标

### 3.1 最初目标

项目一开始更接近“方法改进型论文”路线，即：

- 设计一个包含 gate、fusion、residual 的完整多模态模型
- 期待 `Full Model` 在整体指标上优于结构基线与简化版多模态模型

### 3.2 当前目标

随着正式实验完成，项目主线已经收敛为一篇**分析驱动型论文**，而不是“提出一个全局最强新模型”的论文。

当前正式主线已经在 [FINAL_NARRATIVE_DECISION.md](/E:/learn/R&D/mmkg-project-research/docs/FINAL_NARRATIVE_DECISION.md) 中定稿：

**在当前 OpenBG-IMG 协议下，多模态收益是真实存在的，但它是局部的、条件化的、且有边界的；本文的核心贡献是解释多模态信息何时有效，以及为什么它没有成为全局主导。**

换句话说，当前论文不再试图证明：

- `Full Model` 是全局最强模型

而是重点回答：

- 多模态信息在哪些条件下有效？
- 为什么这些收益没有扩展成全局优势？
- 为什么最终仍是结构补偿路径更强？

---

## 4. 当前论文的核心研究问题

目前整篇论文围绕以下三个问题展开：

1. 在统一协议下，`Full Model`、`Residual-only`、`Gate-only`、结构基线之间的整体关系是什么？
2. 多模态收益是否只出现在某些特定子条件下，例如有图实体、特定关系类型、特定预测方向？
3. 从 gate、residual、mix weight 和具体案例的角度，为什么 `Full Model` 能稳定优于简化多模态模型，却仍然难以超过 `Residual-only`？

---

## 5. 数据集与实验协议

### 5.1 数据集

项目使用 OpenBG-IMG 商品知识图谱数据集。

任务是链接预测（link prediction），统一在 `train / dev / test` 协议下进行，使用 filtered ranking 汇报：

- MRR
- Hits@1
- Hits@3
- Hits@10

### 5.2 当前统一评测协议

当前所有正式结果都基于统一协议：

- `direction=both`
- `dev` 仅用于 early stopping 和 best checkpoint 选择
- 最终统一在 `test` 上汇报
- 每个模型至少 3 个 seeds

这部分已在 [EXPERIMENT_EXECUTION_TODO.md](/E:/learn/R&D/mmkg-project-research/docs/EXPERIMENT_EXECUTION_TODO.md) 中固化。

### 5.3 一个非常关键的协议前提

当前 `paper_split` 具有明显的非对称性：

- `head` 方向的目标实体中，约 70% 有图
- `tail` 方向的目标实体，100% 无图

这意味着：

- 图像可用性的细分分析主要发生在 `head` 方向
- `tail` 方向天然更偏结构预测

因此，后续所有关于“多模态收益边界”的结论，都必须被理解为**protocol-aware** 的结论，而不是可以不加说明地推广到所有 MMKGC 数据集上的一般规律。

---

## 6. 模型设置

### 6.1 五组内部主模型

项目最终固定了五组核心模型：

- `Text-only`
- `Early Fusion`
- `Gate-only`
- `Residual-only`
- `Full Model`

它们分别代表：

- 单模态文本下界
- 直接早期融合
- 关系感知 gate 融合
- 纯结构补偿路径
- 融合与 residual 并存的完整模型

### 6.2 结构参考基线

项目也纳入了两个结构模型：

- `ComplEx`
- `TuckER`

当前更准确的定位是：

- `ComplEx` 是当前协议下真正具有竞争力的结构基线
- `TuckER` 是 classical structural reference baseline，但在当前协议下并未表现出竞争性的强基线效果

不应把当前结果写成“`TuckER` 一般很弱”，更准确的说法是：

- under the current protocol, `TuckER` does not emerge as a competitive structural baseline

---

## 7. 已完成的主实验与核心结果

### 7.1 主结果实验

主结果已经完成 7 个模型的 3-seed 正式汇总，结果见：

- [MAIN_RESULTS_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/MAIN_RESULTS_SUMMARY.md)
- [RESULT_INDEX.md](/E:/learn/R&D/mmkg-project-research/docs/RESULT_INDEX.md)

当前主结果排序为：

1. `Residual-only`
2. `ComplEx`
3. `Full Model`
4. `Gate-only`
5. `Early Fusion`
6. `Text-only`
7. `TuckER`

其中主表的关键数值为：

- `Residual-only`: `MRR = 0.2930 ± 0.0008`
- `ComplEx`: `MRR = 0.2588 ± 0.0018`
- `Full Model`: `MRR = 0.2100 ± 0.0097`
- `Gate-only`: `MRR = 0.1739 ± 0.0044`

由此得到第一层结论：

- `Full Model` 明显优于更简单的多模态模型，尤其优于 `Gate-only`
- 但 `Full Model` 没有超过 `Residual-only`
- 当前任务在整体上仍由更强的结构补偿路径主导

### 7.2 Residual dominance 诊断

项目专门做了 residual dominance 排查，包括：

- 梯度统计
- `residual_scale_value`
- `mix_w_fusion / mix_w_residual`
- delayed-residual
- weaker-residual
- stronger residual regularization

对应结论已写入：

- [EXPERIMENT_EXECUTION_TODO.md](/E:/learn/R&D/mmkg-project-research/docs/EXPERIMENT_EXECUTION_TODO.md)
- [FUSION_VS_RESIDUAL_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/FUSION_VS_RESIDUAL_ANALYSIS.md)

当前结论不是“训练 bug 导致 residual 偷跑”，而是：

- fusion 与 projection 一直在学习
- 但最终表示混合长期偏向 residual
- 更合理的解释是：当前任务本身更偏好结构补偿路径

---

## 8. 已完成的收益边界分析

这是当前论文最核心的部分。

### 8.1 `has_img / no_img` 分析

相关文档：

- [HAS_IMG_SPLIT_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/HAS_IMG_SPLIT_SUMMARY.md)
- [HAS_IMG_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/HAS_IMG_ANALYSIS.md)

已得到的关键结论：

- `head_has_img` 是最有利于多模态收益的局部 regime
- 在 `head_has_img` 上，`Gate-only > Full Model > Residual-only`
- 但 overall 仍被 `tail_no_img` 上的结构优势拉回

这说明：

- 多模态收益不是全局存在
- 它受到目标位置和模态可用性的强约束

### 8.2 relation-type 分析

相关文档：

- [RELATION_TYPE_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_TYPE_ANALYSIS.md)
- [RELATION_TYPE_SUMMARY_ALL.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_TYPE_SUMMARY_ALL.md)
- [RELATION_TYPE_SUMMARY_MIN20.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_TYPE_SUMMARY_MIN20.md)

当前关系被粗分为三组：

- `visual_relations`
- `weak_visual_relations`
- `ambiguous_material_relations`

组级结果显示：

- 三个关系组上都保持 `Residual-only > Full Model > Gate-only`

`MIN20` relation-level 结果显示：

- `Full Model` 在多数支持度足够的关系上优于 `Gate-only`
- 但只在少数关系上优于 `Residual-only`

这说明：

- 多模态收益是真实的
- 但它并没有在“视觉关系组”层面形成普遍优势
- relation-type 更像是收益边界的限制性证据，而不是“视觉关系天然更适合多模态”的直接证据

### 8.3 收益边界综合判断

对应总结文档：

- [GAIN_BOUNDARY_JUDGMENT.md](/E:/learn/R&D/mmkg-project-research/docs/GAIN_BOUNDARY_JUDGMENT.md)

当前已经正式定稿的结论是：

- 多模态收益是局部的
- 多模态收益是条件化的
- 多模态收益是有边界的

这些边界至少由以下因素共同决定：

- target position
- image availability
- relation characteristics
- structural compensation strength

---

## 9. 已完成的行为分析

行为分析用于解释“为什么会出现上述收益边界”，它是解释层，不是另一个独立主线。

### 9.1 全局行为汇总

相关文档：

- [BEHAVIOR_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/BEHAVIOR_SUMMARY.md)

它汇总了：

- gate mean / std
- `g_mean_img / g_mean_noimg`
- `residual_scale_value`
- `mix_w_fusion / mix_w_residual`
- gradient statistics

### 9.2 Gate 分析

相关文档：

- [RELATION_AWARE_GATE_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_AWARE_GATE_SUMMARY.md)
- [GATE_RELATION_JUDGMENT.md](/E:/learn/R&D/mmkg-project-research/docs/GATE_RELATION_JUDGMENT.md)

最终结论已经明确为：

**gate is relation-aware, but not relation-dominant**

即：

- gate 的确会随 relation 变化
- 但这种变化不足以单独解释主性能格局

### 9.3 Residual 分析

相关文档：

- [RELATION_AWARE_RESIDUAL_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_AWARE_RESIDUAL_SUMMARY.md)

当前结论是：

- 在真实 test `(entity, relation)` 对上，`Full Model` 的 residual 有效贡献在 `head_noimg` 上系统高于 `head_has_img`
- 缺图条件下 residual 补偿更强

### 9.4 Fusion vs residual

相关文档：

- [FUSION_VS_RESIDUAL_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/FUSION_VS_RESIDUAL_ANALYSIS.md)

最终结论是：

**residual-dominant asymmetric complementarity**

即：

- fusion 不是失活的
- 它确实帮助 `Full Model` 稳定优于 `Gate-only`
- 但最终主导权仍然长期在 residual 手里

---

## 10. 已完成的案例分析

案例分析用于把“统计层结论”落到具体查询上。

相关文档：

- [CASE_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/CASE_ANALYSIS.md)
- [CASE_ANALYSIS_INTERPRETATION.md](/E:/learn/R&D/mmkg-project-research/docs/CASE_ANALYSIS_INTERPRETATION.md)
- [CASE_CONCLUSION.md](/E:/learn/R&D/mmkg-project-research/docs/CASE_CONCLUSION.md)

当前正文推荐成功案例：

- `佩戴方式`
- `裙长`
- `细分风格`

这些成功案例的共同特点是：

- `head`
- `target_has_img=True`
- 关系更接近外观或局部可感知属性

当前正文推荐失败案例：

- `适用场景`
- `材质`
- `净含量`

这些失败案例的共同特点是：

- `tail`
- `target_has_img=False`
- 更依赖结构、元数据或尾实体可预测性

案例层最终支持的结论是：

- success cases 支持局部多模态收益
- failure cases 支持结构主导边界

---

## 11. 当前论文已经完成了什么

项目现在不只是“实验做完了”，而且已经完成了论文主线的定稿与章节草稿搭建。

### 11.1 已完成的论文主线定稿

相关文档：

- [FINAL_NARRATIVE_DECISION.md](/E:/learn/R&D/mmkg-project-research/docs/FINAL_NARRATIVE_DECISION.md)

当前正式叙事已经明确：

- 不是“`Full Model` 全局最强”
- 也不是“多模态无用”
- 而是“多模态收益是真实但有边界的”

### 11.2 已完成的写作导航与大纲

相关文档：

- [PAPER_WRITING_GUIDE.md](/E:/learn/R&D/mmkg-project-research/docs/PAPER_WRITING_GUIDE.md)
- [PAPER_DETAILED_OUTLINE.md](/E:/learn/R&D/mmkg-project-research/docs/PAPER_DETAILED_OUTLINE.md)

这些文档已经把论文按章节结构规划清楚。

### 11.3 已完成的正文草稿

当前统一主文档目录为：

- [00_MANUSCRIPT_README.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/00_MANUSCRIPT_README.md)

正文已按章节放在：

- [01_introduction.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/01_introduction.md)
- [02_related_work.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/02_related_work.md)
- [03_task_setting_and_protocol.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/03_task_setting_and_protocol.md)
- [04_models_and_compared_methods.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/04_models_and_compared_methods.md)
- [05_main_results.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/05_main_results.md)
- [06_gain_boundary_analysis.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/06_gain_boundary_analysis.md)
- [07_behavior_analysis.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/07_behavior_analysis.md)
- [08_case_study.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/08_case_study.md)
- [09_discussion_limitations_conclusion.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/09_discussion_limitations_conclusion.md)

也就是说，当前状态已经不是“只有实验记录”，而是已经有一套可以继续润色的论文主文档框架。

### 11.4 已完成的修改建议清单

当前 revision 层的总清单在：

- [paper_revision_master_checklist.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/paper_revision_master_checklist.md)

它的作用是：

- 不再讨论“实验做什么”
- 而是讨论“论文下一步怎么改”

---

## 12. 当前还没有完成、但已经不构成主线阻碍的内容

从实验角度看，非必须但仍可追加的主要是：

- 第 5 阶段优化路径排查
  - 两阶段训练
  - 分组学习率
  - 若干超参数敏感性

但按照当前已经定稿的论文主线，这部分不是必要前提。

原因是：

- 当前论文不是方法改进型主线
- 现有证据已经足够支撑 analysis-driven paper

因此，当前工作的重点更适合放在：

- 文稿润色
- 图表补齐
- Related Work 文献接入
- 摘要、参考文献、主图表等 paper package 完成

---

## 13. 给第一次接手者的建议阅读顺序

如果对方第一次接手这个项目，建议按下面顺序阅读。

### 第一轮：先理解项目主线

1. [ALL_IN_ONE_PROJECT_BRIEF.md](/E:/learn/R&D/mmkg-project-research/docs/ALL_IN_ONE_PROJECT_BRIEF.md)
2. [FINAL_NARRATIVE_DECISION.md](/E:/learn/R&D/mmkg-project-research/docs/FINAL_NARRATIVE_DECISION.md)
3. [EXPERIMENT_EXECUTION_TODO.md](/E:/learn/R&D/mmkg-project-research/docs/EXPERIMENT_EXECUTION_TODO.md)

### 第二轮：看核心证据

1. [MAIN_RESULTS_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/MAIN_RESULTS_SUMMARY.md)
2. [HAS_IMG_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/HAS_IMG_ANALYSIS.md)
3. [RELATION_TYPE_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/RELATION_TYPE_ANALYSIS.md)
4. [GATE_RELATION_JUDGMENT.md](/E:/learn/R&D/mmkg-project-research/docs/GATE_RELATION_JUDGMENT.md)
5. [FUSION_VS_RESIDUAL_ANALYSIS.md](/E:/learn/R&D/mmkg-project-research/docs/FUSION_VS_RESIDUAL_ANALYSIS.md)
6. [CASE_CONCLUSION.md](/E:/learn/R&D/mmkg-project-research/docs/CASE_CONCLUSION.md)

### 第三轮：进入当前论文草稿

1. [00_MANUSCRIPT_README.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/00_MANUSCRIPT_README.md)
2. 依次阅读 `01` 到 `09`
3. 最后看 [paper_revision_master_checklist.md](/E:/learn/R&D/mmkg-project-research/docs/paper_manuscript/paper_revision_master_checklist.md)

---

## 14. 一页式总结

如果只用几句话概括整个项目，可以这样说：

- 这是一个围绕 OpenBG-IMG 的 MMKGC 研究项目。
- 项目最初希望通过 gate + fusion + residual 的完整多模态模型获得全局最优结果。
- 正式实验表明：`Full Model` 能稳定优于简化多模态模型，但仍弱于 `Residual-only` 和 `ComplEx`。
- 因此论文主线转向分析：多模态收益并非不存在，而是只在某些局部条件下出现。
- 这些条件包括：`head` 方向、有图目标实体、以及部分关系类型。
- 行为分析进一步说明：gate 的确感知关系，但最终决策长期仍由 residual 主导。
- 案例分析也与这一点一致：成功案例集中在 `head + has_img`，失败案例集中在 `tail + no_img`。
- 所以当前论文不是“提出一个全局最强的新模型”，而是“系统解释多模态收益何时有效、为何受限”的分析型论文。

---

## 15. 当前最需要记住的结论

当前项目最重要的一句话是：

**multimodal gain in OpenBG-IMG is real but bounded; the paper is about explaining when it helps and why it remains bounded under the current protocol**
