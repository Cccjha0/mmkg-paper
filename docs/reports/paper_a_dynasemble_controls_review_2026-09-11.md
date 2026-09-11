# D01–D03 回传结果核验（2026-09-11）

**核心结论：健康 DynaSemble 改变了比较结果。** DEV 选择的 R3 在 W-N、D-N 的 MRR 高于 ADC；ADC 在 W-A、D-A 更高，且四组相对 Global 的 harm rate 均更低。不能继续用历史塌缩版本支持“ADC 优于正常动态集成”或统一 SOTA 的结论。

本次读取回传结果、检查小型 selector 参数、复算统计并重放冻结 selector 前向；未在本机运行真实 scorer、训练或大 bootstrap。输入为 `outputs/paper_a_safe_correction/dynasemble_controls_v1_review/` 及补包 `dynasemble_controls_v1_cache_evidence/`。实验代码 commit 为 `1c0dbcb2aa0e06702add55cc8bd8f7574eb92dfd`，服务器 Python 3.9.1 / PyTorch 2.6.0+cu126 / NumPy 2.0.2 / A100-SXM4-40GB。

## 关闭状态

| 风险 | 本次已完成 | 状态 |
|---|---|---|
| D01：塌缩基线 | 保留 R0 全部失败 seed；五个新版本各 3 base × 3 selector seeds；DEV 调参、训练健康和完整 TEST 比较已完成；补包与权重重放通过 | 按既定最低条件关闭；健康版本已进入主表 |
| D02：论文/公开代码/迁移 | 固定公开 commit、逐项对照和实际 core diff；R1 上游参数/前向/loss/梯度一致性单测；服务器源码与 commit 哈希一致 | 按既定最低条件关闭；不能称完全复现原论文实验 |
| D03：固定系数偏置 | R4 使用 R3 同一 DEV 配置/预算，全部 seed 保留；系数比例、gold rank、完整排序探针齐全；已修改绝对端点表述 | 按既定最低条件关闭；没有证据表明交换角色是一项有效修复 |

**补包核验完成，D01–D03 按本轮既定最低关闭条件关闭。** 原打包器遗漏的24份 cache manifest、DEV/TEST 原始行索引和四维 feature 数组现已回传，共72个文件、15,287,400字节。全部文件与原 DEV/TEST 锁及 manifest 的哈希一致；按实际原始三元组重建的折分配与全部训练记录一致，fit/validation 互斥、覆盖完整，head/tail 与各 base/selector 重复保持同折。180个冻结 selector 的全部2,551,680个 TEST 权重已在 CPU 上重放通过，无需重训。

这是上述三项问题的关闭，不是对整篇论文可投稿或所有历史实验有效性的认证。完整候选分数、normalization 和排名的独立重算仍需服务器大 cache；本次没有声称完成这一额外工作。

这批实验是在已查看历史 TEST 后针对问题设计的补充实验；内部设置只用 DEV 选择，不将整项研究称为新盲测确认。W-A 选中 10 轮、1e-4 的网格边界，本次保留这个预定预算，不根据 TEST 追加搜索或选方向。

## 核验范围与证据

审计脚本：`scripts/audit_paper_a_dynasemble_review.py`。机器可读结果：`outputs/paper_a_safe_correction/dynasemble_review_audit/audit.json`，`status=review_artifact_checks_passed`，`small_cache_evidence_verified=true`，无失败项。此标志表示小型 cache 证据和 selector 重放核验通过，不能解释为全部候选分数已独立重算。

- 4 个 DEV lock、4 个 TEST-start lock 绑定、1,975 个输入文件哈希（含72个补包文件）；记录源码与执行 commit 的 LF 规范化哈希一致。
- 324 条 CV 轨迹（每组 81 条），1,296 个 held-out checkpoint 记录；每个学习率/轮数组合包含所有 3 base × 3 selector × 3 folds。
- 180 个最终 selector、504 份模型与 trace、全部逐层梯度/参数更新、最终健康标志及 180 行 health summary 相互一致。模型均可安全加载，参数均有限。
- 独立复算 DEV 观测数加权 MRR、tie-break 和四个配置选择；R1/R2/R3-fixed 固定一轮；R4 与 R3 的配置、seed 和预算一致。
- 180 份 TEST CSV 与 sidecar/锁/模型绑定一致；完整查询覆盖、无重复；gold 实体、rank 与 RR 一致。基础 A/B 端点与 B07/C04 v2 的 ADC 导出逐查询完全一致。
- 36 行方法 point summary、456 个 base × selector × direction 单元、harm/benefit/unchanged 及条件幅度全部从逐查询结果复算通过。R0 三个 seed 全部保留，没有将它复制为九个独立运行。
- 每个完整排序探针的选择可由 observable query hash 重建；探针不依据策略表现或 gold 分数选择。
- 12份 DEV 与12份 TEST cache 的专家资产、split、query 身份和 feature 哈希均一致；DEV manifest 不包含 TEST 数据文件，DEV 过滤为 TRAIN∪DEV，TEST 为全事实 union。全部12份 DEV 的原始三元组折分配完成独立重建。

| 组合 | 最终 selector 数 | 重放 TEST 权重数 | CPU 重放与服务器导出的最大绝对差 |
|---|---:|---:|---:|
| W-N | 45 | 384,660 | 4.768372e-7 |
| W-A | 45 | 384,660 | 3.814697e-6 |
| D-N | 45 | 891,180 | 2.861023e-6 |
| D-A | 45 | 891,180 | 3.814697e-6 |

重放使用此前已提交审计脚本中的 `atol=2e-5, rtol=2e-6`，没有根据本次结果放宽容差。重放权重仅使用已绑定的四维缓存特征与冻结参数，不将 gold/filter 传入 selector。全部通过数值容差核验，**不声称 CPU 与 GPU 输出逐 bit 相等**。补包核验后，方法汇总、所有 seed/direction 值和训练健康汇总的文件字节与补包前一致。

局限：未重放完整候选排名和 normalization；输入数据/原始 cache 大数组的字节核查依赖服务器执行路径。TEST 的先后顺序由锁绑定和代码屏障支持，没有外部时间戳证明。10,000 次 bootstrap 区间为经哈希核验的服务器产物；本机复算 point metrics 和完整 seed/direction 矩阵，没有再次运行大 bootstrap。

## D01：健康对照的真实比较

W-N/W-A 分别为 MKG-W 的 M-Hyper+NativE / M-Hyper+AdaMF-MAT；D-N/D-A 为 DB15K 对应组合。A 始终指可靠 primary M-Hyper。

| 方法：TEST MRR | W-N | W-A | D-N | D-A |
|---|---:|---:|---:|---:|
| R0 historical | .347957 | .310320 | .333222 | .304616 |
| R1 source core | .368292 | .348538 | .373725 | .360812 |
| R2 init | .368411 | .350739 | .374584 | .362820 |
| R3 Softplus fixed | .368641 | .349840 | .373897 | .361762 |
| **R3 Softplus DEV-selected** | **.369530** | .359075 | **.374984** | .372353 |
| R4 role swap | .369441 | .315878 | .352841 | .300898 |
| Global | .364665 | .359721 | .374567 | .374567 |
| Query-soft | .366543 | .340618 | .360990 | .338760 |
| ADC | .367112 | **.360735** | .374485 | **.374778** |

| 组合 | R3 − ADC MRR | 原三元组 bootstrap 95% CI | R3 harm | ADC harm |
|---|---:|---|---:|---:|
| W-N | +.002418 | [+.001220, +.003585] | 16.823% | 15.150% |
| W-A | −.001660 | [−.002338, −.001017] | 11.010% | 3.619% |
| D-N | +.000499 | [+.000071, +.000938] | 16.281% | 1.424% |
| D-A | −.002426 | [−.002878, −.001989] | 14.661% | 0.094% |

这些是固定训练结果下的 pointwise 区间，未做多重比较校正。先平均 selector repeats，再按原始三元组抽样，保留 base/direction 相关性；不能把区间说成训练随机性区间。Harm 在实际 selector 观测上计算，不先平均 RR。D-N 的 R3−ADC 区间下界大于零，但 R3−Global 区间跨零，两者不矛盾。

健康训练与性能优劣分开判断。R0 每组 seed 1、3 全部权重为零，seed 2 有活动；它是历史迁移的可复现失败案例。所有新版本的 180 次最终训练，以及 324 次 CV 训练，都没有全零权重或永久零梯度标志，**每层在每个 epoch 均有正的参数更新范数**。不能从 R1 的正常结果反推“初始化顺序是唯一失败原因”：R0→R1 还改变了训练统计支持集，且 selector seed 设计不同。

| 组合 | DEV 选出的 LR / 轮数 | pooled held-out DEV MRR | R3 TEST weight SD（九次运行范围） | primary ratio SD（九次运行均值） |
|---|---|---:|---|---:|
| W-N | 1e-5 / 1 | .352361 | .003215–.005763 | .000994 |
| W-A | 1e-4 / 10 | .343739 | .654516–.817693 | .003110 |
| D-N | 5e-5 / 3 | .370354 | .245735–.410799 | .006974 |
| D-A | 1e-4 / 5 | .368916 | .715459–.944084 | .003544 |

DEV 值是配置选择分数，不是选后无偏泛化估计。新 DEV 的评估过滤为 TRAIN∪DEV，旧 ADC/R0 DEV 使用全事实 union，因此不直接比较两者 DEV MRR。full-DEV resubstitution 记录没有用于二次选 checkpoint。

R3 的 36 次最终训练中，最小逐层 epoch 更新范数为 .001064，最小逐层 epoch 最大梯度范数为 .003517。多轮 W-A、D-N、D-A 的首末 epoch 平均 loss（九次运行平均）为 1.176→.362、1.300→.467、.951→.253。W-N 只选一轮，不宣称跨 epoch loss 下降。其比例变化很小，可能近似静态组合；当前结果不证明其 MRR 收益必须依靠 query dependence。

更稳妥的论文结论是：**ADC 在这四组给出较低 harm frequency，但 MRR 优势取决于组合。** 不把该横向差异孤立归因于 bound：DynaSemble 还使用 min-max、四维特征、margin supervision 和不同搜索预算；ADC 使用 z-score、13 维特征、winner supervision。

## D02：来源与迁移边界

公开代码固定为 `dair-iitd/KGC-Ensemble@48d66b915f64899798f736129fa8c4d0a40fdb78`。完整逐项表见冻结的 `docs/protocols/paper_a_dynasemble_controls.md`；来源 manifest 及两份真实 core diff 见 `docs/audits/dynasemble_source/`。不要改写冻结协议来适配已经看到的 TEST。

关键核验：公开首个 MLP 的结构为 Linear→Linear→ReLU→Linear→ReLU；只有隐藏循环中的第二 Linear.weight uniform[0,2]。首层、输出层、所有 bias 为默认初始化。上游先初始化隐藏 weight 再构造输出层，R0 顺序不同，R1 已对齐。公开训练先算完整候选分布的 min-max/1−mean/无偏 variance 再 gather；R0 相反，R1–R4 已对齐完整未过滤分布。

论文对初始化的概括、mean 与代码 1−mean、10,000 negatives 与 YAML 的 9,999+gold、论文 validation 与默认 loader dataset[0] 映射 TRAIN，以及 best_epoch 记录但未自动 reload 的差异均已显式列明。未猜测作者曾手工调换 split。迁移保留本项目双模型宽度16、batch16；不能称所有上游三模型 YAML 参数原样复制。R2 和 R3/R4 的输出初始化/激活变更单列。

此前实现与结果审计阶段的 `tests/test_dynasemble_controls.py` 及审计回归测试共 **19 passed**，覆盖公开 core 的参数/前向/loss/梯度一致性、特征在采样前且不接收 gold/filter、端点与角色、完整 synthetic pipeline，以及审计器拒绝缺 seed、重复 CV 记录和跨折/方向错误；补包测试验证72文件字节保真及拒绝被修改的数组。本轮执行实际回传补包的全量审计和权重重放。

## D03：交换固定对象的诊断

R3 是 `wA+B`，primary 比例 `w/(1+w)`；R4 是 `A+wB`，比例 `1/(1+w)`。Softplus 下有限实数 logit 给出 w>0，两者都不能严格达到零/一系数端点；**有限 w 仍可能产生与 primary 完全相同的有限集合排序**。这与 gold rank 恰好相同是三个不同概念。

| 组合 | R3 平均 primary 比例 | R4 平均 primary 比例 | R3 gold rank 等于 primary | R4 gold rank 等于 primary | R4−R3 MRR |
|---|---:|---:|---:|---:|---:|
| W-N | .5158 | .4841 | 34.54% | 33.99% | −.000089 |
| W-A | .9353 | .0500 | 49.83% | 22.84% | −.043197 |
| D-N | .8542 | .1053 | 50.82% | 27.32% | −.022143 |
| D-A | .9355 | .0474 | 53.71% | 19.73% | −.071454 |

R4 使用 R3 同一配置，**未额外调到“反向最优”**；差值为描述性诊断，不用于选择部署方向。后三组中，R4 在正常梯度/更新和下降 loss 的同时让弱模型占据较大比例，说明同一 margin 目标下参数化方向非常重要，不能写“交换固定对象后问题自然消失”。

每个新 selector 每方向32个 observable-query hash 探针，共 180×64=11,520 个 probe observations；与 normalized primary 的完整未过滤弱序（包含 ties）相同的计数为0。不同模型反复评估同一批 query，不能称11,520个独立query；该样本也不证明所有 query 或所有有限 w 均不能恢复 primary 排序。gold rank 比较对象是 raw primary filtered rank，两种检查口径已分别标注。

新版本同 observable-query 的 learned-weight 最大自然跨度为 `3.814697265625e-6`；W-N 的 R3/R4 为精确相同，其他组有微小连续数值差别。此结果不能单独证明差别由 gold 引起，也不能声称全部连续输出逐 bit 相同。已有受控 gold/filter 干预单测与本次缓存 feature 重放提供不同层次的检查；均未跨 gold 后处理平均。

## 补包已完成：复核命令

证据保存在 `outputs/paper_a_safe_correction/dynasemble_controls_v1_cache_evidence/`。此次执行以下只读核验，Python 环境包含 torch/numpy/pandas；不再需要用户补包或重新运行原 `Dev/Test` 阶段：

```powershell
python scripts/audit_paper_a_dynasemble_review.py --cache-evidence-root outputs/paper_a_safe_correction/dynasemble_controls_v1_cache_evidence
```

审计状态和稿件中的待补包说明已更新；可用 `scripts/build_paper_a_dynasemble_review_tables.py` 重建表格及来源 manifest。原实验协议和锁没有修改，完整候选排名的独立重放仍需原服务器大 cache。

## 已修改的论文位置

主表并列 R0/R3，新增全部预定版本、健康训练、DEV 选择、角色交换及 harm 对比表。摘要、Introduction、§5 的过滤口径、Results、Conclusion、DynaSemble appendix 均纳入健康对照。统计口径明确 selector-repeat averaging 与 realized harm，以及已看过历史 TEST 的研究地位。稿件没有继续使用“所有 DynaSemble 均失败”“无法恢复 primary ranking”或仅凭塌缩基线得出全面优势的结论。

精简核验产物及所有 seed/direction 记录位于 `outputs/paper_a_safe_correction/dynasemble_review_audit/`；大 CSV、selector 和 cache 继续留在 gitignore 范围内。
