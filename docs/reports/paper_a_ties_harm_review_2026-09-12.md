# B15/B16 核验：RR 平局的部署后果与 harm 判别分母

本轮完成 B15/B16 的固定策略诊断与正文修改。复用 information_boundary_v2 的六个 pair、三 base seeds、双方向数据，共 474,732 条观察；没有拟合新 selector、重跑基础模型或根据 TEST 选择策略。新计算在单 CPU 线程约 96 秒完成，不需要服务器算力。两个数据集仍属于历史 TEST 已被查看后的回顾性评估。

## B15：平局只退出分类器拟合，仍接受动态动作

平局采用与原训练代码相同的精确条件 `rr_a == rr_b`，并逐行验证其与 `rank_a == rank_b` 等价。DEV 拟合排除的 tie 占比为 18.11%–23.73%，TEST 中仍有 17.51%–24.12%。这些样本保留在 anchor/action 的训练部分选参、OOF 评估和 TEST 应用中。部署并不知道指定 gold 的端点 RR，因此没有新增 gold-aware tie gate。

以下分母均为**全部 TEST ties**，包含没有改变权重的 ties；U 为相对原 anchor 的平均 RR 净变化。完整 DEV/TEST × pair × seed × direction 的 72 个 tie cell 已纳入 PDF 的两张正式表，而非仅留在外部缓存。

| Pair | Tie 数/全体数 | Tie 占比 | Tie 权重改变率 | Harm 数/率 | U |
|---|---:|---:|---:|---:|---:|
| W-N | 6,186/25,644 | 24.12% | 79.44% | 48 / 0.776% | +0.000334 |
| W-A | 5,218/25,644 | 20.35% | 4.27% | 5 / 0.096% | +0.001278 |
| D-N | 12,832/59,412 | 21.60% | 6.21% | 12 / 0.094% | +0.000184 |
| D-A | 10,403/59,412 | 17.51% | 0.26% | 1 / 0.010% | +0.000013 |
| W-NA | 5,208/25,644 | 20.31% | 3.17% | 4 / 0.077% | +0.000036 |
| D-NA | 10,881/59,412 | 18.31% | 3.84% | 6 / 0.055% | +0.000118 |

W-N TEST 有 4,914 条 tie 改变权重，63 条收益、48 条受损；tie 子集无条件平均损失为 0.001306，受损 ties 的条件平均损失为 **0.1683 RR**。正净收益不代表没有较大的单次损失。W-N 和 W-NA 的 DEV tie 净收益分别为 −0.000066、−0.000077；均保留。六个 TEST tie 净收益点估计为正，但本轮不对 tie 子集作显著性宣称。

端点平局不能推出任意混合的 rank 不变。单测反例：gold 分数为 0，两个候选在 A 下为 (2,−1)、B 下为 (−1,2)，两端 gold 都排第 2，中点却排第 3。两个向量的均值、方差相同，因此该例经 population 标准化仍成立。

`tie_summary.csv` 包含 all/ties/non_ties 三个组，`tie_cells.csv` 保留其全部 216 个联合 cell。除了上表指标，还报告 benefit 数/率、平均绝对权重位移、无条件损失、条件损失和对总体净收益的贡献。已验证 tie 与 non-tie 的计数、收益贡献守恒，并与 B13 的原结果一致。B13 的 `tied_endpoints` 指**已执行动作中的 ties**，本轮主 tie 分母是**全部 ties**；两者收益贡献一致，条件均值不能混用。

## B16：三个动作总体，分别报告基线及区间

在生成本轮结果前写定协议，固定逆偏好强度 `r = 1-c`，其中 `c = abs(2*p_A-1)`；没有看完结果后翻转方向。raw proposal 是取消 gate 后的原半径提议，仍保留 clipping、0.05 投影及原 tie-break。按三个总体报告：

1. **All**：全部有效的未门控提议，含投影/裁剪后回到 anchor 的行。
2. **Prop.**：All 中投影后权重不等于 anchor 的非零提议。
3. **Exec.**：经过原 gate 后实际执行的非零修正。

前两组按未门控动作的 RR 标记 harm，第三组按实际动作；harm 均指相对对应 anchor 的 RR 下降超过 1e-12。已验证 Exec. 是 Prop. 的子集，且该子集 raw/deployed 动作及 RR 完全一致。非零权重变化也可能不改变 RR。

以下为四个主 pair 的 **TEST Exec.**，括号内为条件 95% 区间。AP 为非插值 average precision，基线是**同一子集**的 harm prevalence；Lift 为 AP 减该 prevalence，单位为百分点。

| Pair | n / harm | Prevalence | AUROC [95% CI] | AP | AP Lift [95% CI] |
|---|---:|---:|---:|---:|---:|
| W-N | 16,240 / 3,885 | 23.92% | 0.525 [0.511, 0.540] | 25.58% | +1.65 [+0.86, +2.63] |
| W-A | 5,204 / 928 | 17.83% | 0.500 [0.478, 0.522] | 17.46% | −0.37 [−1.24, +0.78] |
| D-N | 4,678 / 846 | 18.08% | 0.489 [0.467, 0.511] | 17.57% | −0.52 [−1.47, +0.74] |
| D-A | 267 / 56 | 20.97% | 0.544 [0.467, 0.631] | 22.77% | +1.80 [−1.46, +9.23] |

完整 PDF 表格保留全部六个 pair、两 split、三总体，包含 prevalence、AUROC、AP、lift 的点值和区间；CSV 另有各类 cluster 数、每个指标的有效 bootstrap 次数及正/负样本数。DEV 用原 OOF 各 fold 的模型、anchor、半径和阈值，不能把 full-DEV lock 套到所有 OOF 行。

结果含义：

- 修正后的四个主 pair，在 TEST **All** 上 AUROC 均低于 0.5，AP 均低于 prevalence，lift 区间均低于 0；固定的逆 confidence harm 排序在该总体上失败。低 margin 可能被投影死区消除，且裁剪也会产生零动作；这些行不能 harm，却仍影响 All 的判别指标。
- 限制到 **Exec.** 后，W-N 有很弱的正判别，其余三个主 pair 的 AUROC 区间都跨 0.5，AP lift 区间都跨 0。两个附加 pair 的 Exec. 也都如此。不能把较好的子集替代全部提议的结果，更不能由判别分数推出概率校准。
- D-A 的 267 条执行观察来自 156 个原始三元组；56 次 harm 来自 46 个三元组，区间相对宽。其全体观察 harm 率是 56/59,412 = **0.094%**，执行子集是 56/267 = **20.97%**。正文现已同时说明这两个分母；低总体 harm 部分来自低干预率。
- 不据本轮结果翻转 score、设新阈值或部署新 gate。本文 confidence 仅为 margin/gate strength，不能叫自然胜率或无害概率。固定逆 confidence 的弱/失败排序也不等于证明任何其他信息或方向都无法预测 harm。

区间用 **2,000 次原始三元组 cluster bootstrap**，每次从完整 split 抽样，保留该三元组的三 seeds × 双方向，再限制到各固定子集；三个子集共享抽样。AP lift 在每次抽样内重新减去对应 prevalence。所有报告指标均有 2,000 个有效 replicate。单测覆盖空子集、单一类别等未定义情况；不将其填为零。区间条件于既有模型、fold 和 seeds，不含重训练/选参不确定性，不是多重比较校正结果，也不保证图中相关三元组彼此独立。

常数 score 在两类俱全时 AUROC 为 0.5，有 harm 时 AP 等于 prevalence；Global 的零动作/零 harm 意味着没有可评价的执行判别总体，不能解释为完美风险分类器。AP 定义采用 [scikit-learn 文档](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html)的非插值规则，percentile 区间参照 [SciPy 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)；本研究的 cluster 单位由上述协议另行规定。

## 历史结果与当前结果不混用

历史 `confidence_harm` 输出保留不改：原四个主 pair 的 TEST AUROC 为 0.527–0.569，D-N/D-A 的 AP 低于 prevalence。新的 manifest 将该历史表及报告作为独立版本绑定。旧数据来自边界修复前，且旧 DEV 诊断使用最终锁定的 anchor/radius，因此不能拿旧数值填充本轮修正结果，也不能延续旧报告中关于识别“especially safe queries”的解读。

论文在方法段、fallback 分母说明、Discussion 和新增附录中分别说明训练/apply 总体差异、tie 后果以及三个 harm 总体。摘要用 margin-strength gate 表述机制；没有新增安全概率主张。

## 可复核文件及验证范围

- 协议：`docs/protocols/paper_a_ties_harm_review.md`。
- 数值：`outputs/paper_a_safe_correction/ties_harm_review_v1/{tie_summary,tie_cells,harm_discrimination}.csv`。
- 回放与来源审计：同目录 `audit.json`、`asset_audit.json`。
- 绑定：`paper_a_draft/ties_harm_source_manifest.json`；五张表位于 `paper_a_draft/tables/ties_harm/`。
- 11 项单测：tie 反例与分组守恒、动作总体区分、weighted AUROC/AP 对照 sklearn、cluster 权重与显式整组三元组重复对照、空/单类样本规则。
- 全量回放：12 个 pair/split 单元、474,732 条观察的权重、fallback、anchor/动作 RR 一致；无新增无效行，未调用基础模型。

实现入口为 `scripts/analyze_paper_a_ties_harm.py` 和 `scripts/build_paper_a_ties_harm_assets.py`。分析脚本拒绝覆盖已完成的版本，后续变更分析应使用新版本目录。只提交小型源码、正式表及摘要证据，不提交逐 query 大文件。B15/B16 可按本轮诊断及文字条件关闭，结论是公开分布差异和弱判别边界，而非证明 ADC 安全。
