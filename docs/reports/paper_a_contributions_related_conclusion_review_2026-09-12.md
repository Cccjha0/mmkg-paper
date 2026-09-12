# A10–A12：贡献证据映射、问题导向相关工作与条件化结论

本轮为文稿结构与文献修订，沿用已核验的全部结果、策略锁和历史证据地位，不新增实验、调参或 TEST 选择。A10–A12 的关闭依据如下。

## A10：三项贡献各回答一个问题

| 贡献 | 回答的问题与支持结果 | 论文定位 | 不作出的推论 |
|---|---|---|---|
| 现象诊断：互补机会与实际效用分开 | 六 pair 都有端点 Oracle 机会，但 D-N、D-NA 等仍出现略负或接近零的 ADC 收益；winner／gold-rank 分组将机会连接到实际效用贡献 | `tab:complement-pairs`、`tab:complement-winners`、`tab:complement-rank_bins`；§6 互补性诊断与附录 | Oracle gap／静态收益不证明动态策略可学；分组关联不作因果解释 |
| 方法设计：显式参照与行动约束 | DEV 静态参考、拒绝调整时精确回退、规定网格条件下的分数扰动界；最近邻机制差异表交代已有组成 | `eq:fallback`、`eq:bound`、`tab:nearest-methods` | 不把动态集成、收缩、tanh、clip 或拒绝本身当发明；系数／分数界不是 RR 风险保证 |
| 跨条件检验与机制证据：界定调整的条件价值 | 共享拟合对象和预算的 DEV/TEST 比较限制 tanh 特有优势；固定其余组件的半径／回退反事实显示组件损益；健康动态控制显示效用—损失取舍 | `tab:matched-dev`、`tab:matched-test`、`tab:rerun-ablation`、`tab:claims-main`、`tab:conservative-fair` | 不把“做了实验”作为创新；不声称单独识别 anchor 的因果作用，也不将某一策略写成普遍赢家 |

三条正文贡献均直接带内部交叉引用。第一项回答“有没有机会、实现了多少”，第二项回答“行动被如何约束”，第三项回答“哪些控制改变了损益、优势是否专属于 ADC”。第一项的分组诊断与第三项的策略比较／固定组件反事实分开。

## A11：按问题衔接文献

相关工作改为三个主题：**冻结模型融合与静态参考 → query 权重与偏好校准 → 拒绝调整与风险定义**。基础模型架构缩为解释既定 pair 的一句，保留 MoSE 与 DynaSemble 最近邻差异表。增加四项原始文献，参考文献库从 9 项变为 13 项；这不是穷尽性新颖性检索。

| 原始来源及已核对位置 | 支持的桥接 | 与本研究的界限 |
|---|---|---|
| Wolpert (1992), *Stacked Generalization*, pp.241–243，尤其 Figure 1 的 held-out predictions；[DOI](https://doi.org/10.1016/S0893-6080(05)80023-1)，[原论文 PDF 镜像](https://cafri-labs.github.io/lab-manual/papers/wolpert1992.pdf) | 用已有预测训练组合器已有基础；hold-out 的位置需要明确 | 本研究冻结基础模型，对 DEV 上组合器拟合／选参做分组评估；不声称逐折重训基础模型或发明 stacking |
| Diebold & Shin (2019), *Machine Learning for Regularized Survey Forecast Combination: Partially-Egalitarian LASSO and its Derivatives*, 摘要、Introduction 与 §2；[出版商原文](https://www.sciencedirect.com/science/article/pii/S0169207018301596) | 选择预测器并把保留权重收缩向等权，说明简单参照与收缩是既有方法 | 该文是预测组合权重正则化；ADC 是 query 修正，参考为 DEV Global、未必等权。我们的 shrinkage 是应有强对照，不是该文算法复现或独创机制 |
| Guo et al. (2017), *On Calibration of Modern Neural Networks*, 摘要中的校准定义；[PMLR 原文](https://proceedings.mlr.press/v70/guo17a.html) | 概率必须对应明确事件及其正确频率 | 该文不验证 ADC 的 logistic 校准；endpoint-winner 概率即使校准也不是有益调整／无害调整概率。ADC 使用 class-balanced 拟合与 DEV MRR 阈值，不能冒充风险校准 |
| Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, 摘要；[JMLR 原文](https://jmlr.org/papers/v11/cawley10a.html) | 有限样本选参标准也会被过拟合，评估需区分拟合、选择与报告 | 分组 DEV 和 feature/filter 单测不恢复历史 TEST 独立性；后验地位由本项目 C07/C08 时间线支持，未借文献重新宣称确认性 |

已有 [MoSE](https://aclanthology.org/2022.emnlp-main.719/) 与 [DynaSemble](https://aclanthology.org/2024.acl-short.20/) 的机制定位继续沿用 `docs/protocols/paper_a_nearest_methods_reference.json` 的公式／代码核查。已有 [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html) 用于区分拒绝预测的 risk–coverage 与拒绝调整后仍返回排名；不将 ADC 的 intervention rate 改称 prediction coverage。MoSE 与四篇桥接文献均未新增本地性能复现实验。

全部新增引文使用转述；书目信息核对了作者、年份、刊物／会议、页码及 DOI 或正式出版链接。未将用户提供的 S5、S7 标签猜配给具体文献。

## A12：结尾交代何时考虑、付出什么及不覆盖什么

结论用短结果段保留四主＋两附加的范围、D-N 略负及三个跨零区间、W-N 的 Relation／健康 DynaSemble 优势、clip 的 MRR 优势和 Global 零参照损失，并指向全六 pair 区间表。不再重复两个附加 pair 的六位小数、历史时间的精确倍数或组件清单；这些数值仍完整保留在正文与正式表格。

最后一段可独立阅读：已有两套冻结 scorer、DEV 上可靠 primary 和有效静态参考时，ADC 是限制额外调整的候选；只有 held-out DEV 收益值得承担 RR 损失与双评分成本时才值得考虑，保留 Global 或采用更强控制也是有效选择。这里表达部署考虑，不新增已验证的联合效用目标、损失阈值或选择算法。

Discussion 保留标签错位、无概率保证、模态支持关联的局限、历史 TEST 使用和双模型成本；补充 D-N 在可靠 primary 条件下仍有略负 DEV/TEST 效用，明确该条件并不充分。结尾保留修正版计时未测、无可靠 primary／分布漂移／模态专属适配未确立和缺乏独立确认的范围，不引入未经实验的新效用学习方案。

## 验证

沿用 LaTeX 编译、交叉引用／引文解析、全文来源和表格 hash 校验、PDF 渲染与页面检查。现有校验不再强迫结论重复精确结果数值，改为保留全部不利证据、六 pair 区间表引用及历史／修正版计时界限。数字资产和训练代码均未修改，无需服务器重跑。

新增引文的首次编译暴露了已有构建顺序问题：pdfLaTeX 读取源码目录中的旧 `main.bbl`，而脚本直到最后才同步 BibTeX 的新输出。已把同步移至 BibTeX 完成后、后续 LaTeX 引用解析轮次前，确保一次构建即可使用新参考文献。

最终检查通过：44 页、13 项参考文献、39 张表、4 幅图和 2,575 项来源快照；编译无 overfull 或未解析引用，完整来源校验退出码为 0。已检查贡献／相关工作及 Discussion／Conclusion／书目页面，并处理相关工作页尾的孤立首行。当前 PDF 贡献见第 3 页，相关工作见第 3–5 页，Discussion 与 Conclusion 见第 17–18 页。修正版计时仍未测，两个数据集的 confirmatory 标记仍为 false。
