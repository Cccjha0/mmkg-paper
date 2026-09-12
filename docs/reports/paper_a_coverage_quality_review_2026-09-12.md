# D11/D12 核验：覆盖率机制、选择质量、聚合与 AP

本轮完成 D11/D12 的统计及报告要求。使用 `information_boundary_v2` 的冻结状态和缓存，对六个 pair、原 grouped OOF DEV 与锁定 TEST 共 474,732 个观察作离线诊断。没有基础模型评分、拟合或新策略选择。历史 coverage 图仍是独立旧版本；正文新增的是修正版诊断。

## D11：单调性是构造性质，不是选择有效性的证据

固定每行 ungated projected proposal 及其 RR 后，在嵌套接受集合中，受损观察累计数不会减少；因此以全部 N 为分母的 harm 不会下降。随机排序同样满足此性质。接受子集和真正非零干预子集的条件风险则没有这一单调保证。

新增三个分母：`h/N`、`h/n_accept`、`h/n_active`，并报告对应计数、实际干预率、净 RR 效用、无条件损失与平均权重位移。接受仍映回原 proposal，拒绝返回 anchor，不重新计算幅度或重选 β/τ。

每个决策单位为 `(seed, fitted fold, direction, relation, known entity)`，同一可观测 query 的多 gold 记录共同接受或拒绝。按 seed/fold/可观测重复次数分层，每层配额为 `floor(decile*n_units/10)`，预先固定 0–10 全部 decile。配额是分层 unit 比例，不是单个置信度阈值或全观察覆盖率。随机参照在相同层内生成 512 个均匀随机排列，每个排列跨 decile 复用，接受数逐次精确匹配。All pool 包括零 proposal；Active pool 仅含投影后非零 proposal，后者还精确匹配实际干预数。

图中的随机线为 512 次实现的指标均值，阴影为逐点 2.5/97.5 随机化分位数，**不是总体置信区间或显著性检验**。条件比例先在每次实现上计算，之后才平均；不能用期望 harm 数除以期望 active 数替代。空子集为未定义，并保留有效实现次数。补充 CSV 另存可解析的各加法总量期望。

TEST、Active pool、预先固定 50% unit 配额：

| Pair | 实际干预数 | 置信度 harm 数 | 置信度条件 harm | 随机条件 harm 均值 | 随机 95% 范围 | 置信度 U | 随机 U |
|---|---:|---:|---:|---:|---:|---:|---:|
| W-N | 10,241 | 2,156 | 21.05% | 24.05% | 23.45%–24.69% | +0.001938 | +0.001118 |
| W-A | 3,862 | 712 | 18.44% | 16.40% | 15.58%–17.19% | +0.000608 | +0.000286 |
| D-N | 9,740 | 1,581 | 16.23% | 15.53% | 14.99%–16.06% | −0.000049 | −0.000081 |
| D-A | 12,454 | 2,808 | 22.55% | 19.41% | 18.89%–19.96% | −0.001800 | −0.001475 |
| W-NA | 9,186 | 2,249 | 24.48% | 25.78% | 25.10%–26.45% | −0.000689 | −0.001547 |
| D-NA | 7,797 | 1,003 | 12.86% | 10.99% | 10.51%–11.49% | −0.000199 | −0.000332 |

W-N 的这个点具有更低 harm，但不能推广到其它 pair 或所有配额。W-A 更高净收益与更高 harm 频率并存。D-A 的原部署 gate 只改变 267 个观察，而这里的半配额诊断改变 12,454 个，两者不能混写。随机排序已控制数量，仍没有匹配方向/幅度；此前 `rejection_controls_v1` 的 Random-shape 结果继续约束“独立选择信息”解释。

导出重复记录的 confidence 存在小数值差异：全审计最大 unit 内范围 `4.843782985819445e-7`，245 个 unit 的范围超过 `1e-12`。所有这些 unit 的 raw/anchor/deployed 投影权重一致。新 batch 排序使用同一可观测 unit 的均值，避免依据首个 gold 记录取代表值；没有修改原模型或原结果。AP 核验仍使用原逐观察分数，不将 unit 均值替代原分数。

## D12：macro、micro 的分母和权重

四主 pair 的同数据集观察 key 重合率为 100%，两 pair 重复评价同一三 seed/两方向观察。不能把它们视作四个独立样本源。

| Split | W-N、W-A 各 N | D-N、D-A 各 N | MKG-W 每 pair micro 权重 | DB15K 每 pair micro 权重 | 合计 pair-observations | 不重复 dataset triples |
|---|---:|---:|---:|---:|---:|---:|
| DEV | 25,656 | 47,532 | 17.5275% | 32.4725% | 146,376 | 12,198 |
| TEST | 25,644 | 59,412 | 15.0748% | 34.9252% | 170,112 | 14,176 |

Macro 对四 pair 的每项指标等权 1/4，未定义的 pair 不会静默删除。Micro 对加法总量先合并再计算比值；全分母指标的权重为 N 比例，接受/干预条件 harm 则按对应子集数加权，每个配额和随机实现都可能不同。随机聚合也是先逐实现聚合、后计算均值与范围。附加两 pair 全部单列，不加入四主 pair 汇总。聚合没有以 pair 为独立单位计算总体区间。

这一差别有实际影响：TEST Active pool 的 50% 配额，置信度顺序 U 的 macro 为 **+0.000174**，micro 为 **−0.000262**；All pool 对应为 **+0.000151** 与 **−0.000342**。全部配额和两种聚合一起报告，不挑选有利平均。

## AP 与梯形 PR 面积

实际算法是按相同 score 分组、对阈值处 precision 以 recall 增量加权：`AP=sum((R_j-R_(j-1))*P_j)`。它是 **non-interpolated average precision**，不是 precision–recall 曲线的梯形积分。这与 [scikit-learn 官方 AP 定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) 一致；实际核验使用本机 scikit-learn 1.7.1，版本写入审计。

- 原 `scripts/analyze_paper_a_confidence_harm.py::average_precision`、现 `router/ties_harm_diagnostics.py::RankedBinary.evaluate` 与 `sklearn.metrics.average_precision_score`，在全部 36 个 pair/split/population 点一致，最大差异 `5.551115123125783e-17`。
- 历史 CSV `auprc_risk_to_harm` 实为 AP；`auprc_lift_over_prevalence` 实为 AP 减同一子集 prevalence。历史文件不改写，正式新表使用 AP。
- AP lift 是差值，不是比值；百分数展示时用百分点 pp。原 per-pair CI 已按每个聚类 bootstrap 样本自己的 prevalence 重新计算差值，本轮不改其区间。
- Macro AP 为四个 AP 的等权平均；pooled/micro AP 必须拼接原 score/label 后重新排序计算，不能加权平均单 pair AP。跨 pair 分数未校准，额外产生的跨 pair 排序效果并非可部署 harm 预测证据。

例如 TEST Exec.：macro prevalence/AP 为 20.20%/20.85%，lift +0.64 pp；pooled prevalence/AP 为 21.66%/22.57%，lift +0.91 pp。DEV Exec. 的 macro lift 为 −0.44 pp，pooled lift 为 +0.50 pp。新正式表保留全部 All/Prop./Exec. 与 DEV/TEST 两种聚合，仅作描述点，不替代已有逐 pair 区间。

## 核验与产物

6 项单测覆盖机械单调与条件风险反例、分母、随机计数及端点、可观测 query 决策、macro/micro 权重、AP tie 处理及与梯形面积的差异。新审计验证 12 个 pair/split 单元、528 条曲线行、176 条聚合行、48 条 AP 行和 8 条权重记录；运行约 82 秒，CPU 单线程，无服务器任务。六张正式图、四张正式表及来源 manifest 与代码/协议/小型结果绑定。论文需编译及 PDF 视觉核验后交付。
