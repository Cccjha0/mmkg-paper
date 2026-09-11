# A02 / D04–D06：最邻近替代方案核验

2026-09-12。四项最低补实验要求已落实：简单收缩、相同拟合对象与候选数的映射比较、完整静态表、显式 Global 候选及预算披露。**结果不支持 tanh 的独特准确率优势**：Clip-g 与 Linear-g 在四组 TEST 的 MRR 点估计均略高于 ADC。因此关闭的是“缺少对照”的问题，同时收窄论文贡献，不能把本次补实验写成 tanh 获胜。

这是一项已看过历史 TEST 后的问题驱动补充。新增动作先完成所有四组、24 个 DEV 家族锁，再统一应用 TEST；没有按 TEST 选择家族或调整参数。原 ADC、Query-soft 与 DynaSemble 结果均保留。

## 1. D04：确实共享同一个拟合对象

核查 [最终锁定与应用路径](../../scripts/lock_apply_anchored_dynamic.py) 的 `lock_policy` / `apply_locked_policy`：只取得同一个 `model_outputs(model, matrix)` 的 g、p，然后分别调用 ADC 与 Query-soft 映射。当前论文使用的 `rr_query_soft_locked` 不是另一个逻辑回归模型的输出。

[P3 OOF 路径](../../scripts/ablate_anchored_dynamic.py) 也从同一 `group_outputs['full_geometry']` 给两种策略传递输出。本轮重新构建 20 个原 full_geometry 折模型，逐项重现原分组、训练 anchor、非零 ADC 选参、留出权重/RR/fallback 与 Query-soft RR。最终四组直接复用原 `anchored_model.pkl`，在 DEV 和 TEST 将 g、p 与已有导出按绝对容差 `10^-12` 比较，全部通过。

六种新映射共享：13 维特征、监督标签、相同中位数插补/标准化、同一 balanced/liblinear/C=1 逻辑回归对象、g/p、anchor、五折三元组划分、候选实体、z-score、filtered RR、0.05 精确权重网格与投影 tie-break。映射函数没有 gold、RR、filter 参数；完整 DEV/TEST 的重复可观测 query 在不同 gold 下保持相同网格权重。

历史 ADC DEV 的评估过滤仍使用 TRAIN∪DEV∪TEST 的已知事实集合；没有把 TEST 当作正例监督或推理特征。它与新 Dyna 严格 DEV 的 TRAIN∪DEV 过滤范围不同，不能混作同一 DEV 条件的直接比较。Dyna 还使用 4 特征网络与 min-max 分数，所以仍不作为“只变化一个映射”的证据。

## 2. A02 / D06：六个等候选数家族

[协议](../protocols/paper_a_matched_alternatives.md)在计算新映射前固定。记 a=α₀、p=σ(g)；各提案都经过 [0,1] clip、共同 gate（纯 Shrink 除外）、共同网格投影，非有限输入返回 a。

| 家族 | gate 之前的提案 | 非零强度 | gate τ | 总候选数 |
|---|---|---|---|---:|
| ADC+Global | a+β tanh(g) | .05 至 .50，步长 .05 | 0/.1/.2/.3 | 40+1 |
| Shrink | (1−λ)a+λp | .025 至 1，步长 .025 | 固定0 | 40+1 |
| Shrink-gate | (1−λ)a+λp | .1 至 1，步长 .1 | 0/.1/.2/.3 | 40+1 |
| Linear-p | a+β(2p−1) | .05 至 .50，步长 .05 | 0/.1/.2/.3 | 40+1 |
| Clip-g | a+β clip(g,−1,1) | .05 至 .50，步长 .05 | 0/.1/.2/.3 | 40+1 |
| Linear-g | a+γg | .05 至 .50，步长 .05 | 0/.1/.2/.3 | 40+1 |

每族有一个显式的零强度 Global 配置，无需通过阈值间接实现。Shrink 的 λ=0 确切返回 Global，λ=1 等于当前 Query-soft。Linear-p 与 Clip-g 保留 ADC 的局部半径；Linear-g 仅受 [0,1] clip，γ 不构成同样的局部半径。

每次选择先最大化该训练部分的 MRR，`10^-12` 数值平局内优先 Global，其次较小强度、较大 τ。外层训练部分选参、留出三元组评价；最终完整 DEV 只选择动作参数，不再为每个家族训练分类器。完整 DEV 的选择分数是 full-fit 分数，不冒充留出性能。

总计保存 5,904 次候选评价、120 个外层折家族选择、24 个最终家族锁。配置经过网格投影可能重合，因此另列实际不同动作向量数，最终完整 DEV 为31–41，不把它们假称为相同有效复杂度。纯 Shrink 用预算获得更细 λ，其他家族用部分预算选择 gate；这种搜索分配区别已披露。

| 家族 | W-N：强度/τ | W-A：强度/τ | D-N：强度/τ | D-A：强度/τ |
|---|---|---|---|---|
| ADC+Global | .45/.1 | .50/.1 | .20/.2 | .50/.3 |
| Shrink | .80/0 | **0/0** | **0/0** | **0/0** |
| Shrink-gate | .80/0 | **0/0** | .30/.3 | .90/.3 |
| Linear-p | .50/0 | .40/.1 | .40/.2 | .35/.3 |
| Clip-g | .50/.1 | .50/.1 | .20/.2 | .50/.3 |
| Linear-g | .50/.1 | .50/.1 | .20/.2 | .50/.3 |

显式 Global 实际赢得三组纯 Shrink 和 W-A Shrink-gate 的最终选择。ADC 的20个外层折和四个最终选择均未因加入 Global 而改变；不能声称加入零配置便消除了其 D-N TEST 损害。Global 的存在只保证选择集合允许不干预，不能保证泛化收益。

## 3. 留出 DEV 与冻结 TEST 结果

下面 DEV 为原三元组五折留出 MRR；TEST 为所有 DEV 配置锁定后的点估计。各表保留全部配对与所有新家族，Query-soft 和原 ADC 的对应数值也在论文表中保留。

| 方法 | DEV W-N | DEV W-A | DEV D-N | DEV D-A |
|---|---:|---:|---:|---:|
| Global | 0.355750 | 0.352822 | 0.382587 | 0.382587 |
| ADC+Global | 0.358821 | 0.354026 | 0.382578 | 0.382623 |
| Shrink | 0.357168 | 0.352712 | 0.382587 | 0.382587 |
| Shrink-gate | 0.357116 | 0.352822 | 0.382479 | 0.382742 |
| Linear-p | 0.358220 | 0.353435 | 0.382560 | 0.382609 |
| Clip-g | 0.359154 | 0.354103 | 0.382658 | 0.382566 |
| Linear-g | 0.359154 | 0.354102 | 0.382658 | 0.382566 |

| 方法 | TEST W-N | TEST W-A | TEST D-N | TEST D-A |
|---|---:|---:|---:|---:|
| Global | 0.364665 | 0.359721 | 0.374567 | 0.374567 |
| ADC+Global（与原 ADC 相同） | 0.367112 | 0.360735 | 0.374485 | 0.374778 |
| Shrink | 0.366161 | 0.359721 | 0.374567 | 0.374567 |
| Shrink-gate | 0.366161 | 0.359721 | 0.374523 | 0.374834 |
| Linear-p | 0.366519 | 0.360626 | 0.374496 | 0.374634 |
| Clip-g | 0.367223 | 0.360983 | 0.374498 | 0.374806 |
| Linear-g | 0.367223 | 0.360983 | 0.374499 | 0.374807 |

ADC 相对纯 Shrink 的 TEST 增量在 W-N/W-A/D-A 分别为 `+0.000950 / +0.001014 / +0.000212`，D-N 为 `−0.000081`。三个正向配对在三种 seed 和两种方向上均为正，但本轮没有新增 bootstrap，不能据此宣称统计显著。

更接近 ADC 的 Clip-g 已保留相同半径、gate、输入、学习器和预算，TEST 四组均稍高于 ADC；ADC−Clip-g 分别为 `−0.000111 / −0.000248 / −0.000013 / −0.000028`。Linear-g 的结果也四组略高。DEV 上 Clip-g 三组更高、D-A 更低；因此不存在“所有划分、所有配对均优于”的结论。D-N 上这些动态映射仍未优于 Global。

这要求修改贡献表述：**ADC 的特定 tanh 映射增量价值尚未建立**。可以报告某些配对上参考相对行动设计优于直接概率权重或纯收缩，但不能把它写成 tanh 必要性的证据，也不能换报 TEST 上最好的一种映射。

一个与结果无关的代数区别有助于解释：当 a=1 时，只要 λ>0 且 p<1，朝 p 收缩就会离开纯 primary，即便 p>0.5 已偏好 primary；而以 g 的正负决定偏移的 tanh、Linear-p、Clip-g，在 g≥0 时经 [0,1] clip 仍保持 primary。这个区别不依赖 tanh，不能归为其独特贡献。

## 4. 与上一轮一致的风险口径

所有新家族在同一 z-score 坐标中比较，统一报告 U、无条件平均损失 L₋、受损率 H、条件损失 C、权重改变率 Iα 与平均偏移 Dα，保留全部 seed/direction。

- W-N：ADC 比 Shrink 的 MRR 高，但 L₋ 更高（0.005463 对 0.004874），受损率也略高（15.150% 对 15.033%）。相对 Clip-g，ADC 的 MRR 较低、L₋ 较低（0.005463 对 0.005836）。
- W-A：Clip-g MRR 较高、L₋ 也较高（0.001461 对 ADC 的0.001425）；Linear-p MRR 稍低但 L₋ 明显更低（0.000598）。
- D-N：Clip-g 的 MRR 稍高且 L₋ 稍低（0.000608 对0.000617），受损频率却稍高；纯 Shrink 选择 Global，零损失且 MRR 更高。
- D-A：Clip-g 的 MRR 稍高、L₋ 与 H 略低，C 则极略高。Shrink-gate 的 MRR 更高，但 L₋ 为0.000201、H 为2.319%，均高于 ADC 的0.000048、0.094%。

因此既不能把“ADC 更保守”推广到所有邻近映射，也不能把 Clip-g 的微小 MRR 优势当作所有风险量的优势。论文主文与新增风险表均逐项披露。

## 5. D05：静态方法和 Oracle 在 PDF 中都有结果

完整表覆盖六个修正后配对。下表简称与论文一致，NA 为 NativE + AdaMF-MAT。

| 方法 | W-N | W-A | D-N | D-A | W-NA | D-NA |
|---|---:|---:|---:|---:|---:|---:|
| Primary | .359721 | .359721 | .374567 | .374567 | .338037 | .312761 |
| Secondary | .338037 | .296319 | .312761 | .282588 | .296319 | .282588 |
| Equal z-score | .364968 | .336645 | .360744 | .338108 | .327268 | .300405 |
| Equal RRF | .345546 | .322471 | .354296 | .326784 | .296458 | .284988 |
| Global | .364665 | .359721 | .374567 | .374567 | .344127 | .312761 |
| Relation | .368921 | .358672 | .374491 | .373179 | .343964 | .312726 |

当前主表在 Dyna 修订后已列 Relation，因此旧问题“Relation 不在主表”不再成立；本轮进一步把 RRF、两个专家、等权及所有附加配对补齐。W-N 等权 z-score 的 TEST MRR 略高于 DEV-selected Global，Relation 高于 ADC，均保留，不能把 Global 称为 TEST 最佳静态方案。

Oracle 在 PDF 单独一表报告：W-N `.401902`、W-A `.390732`、D-N `.409286`、D-A `.405001`、W-NA `.377348`、D-NA `.357574`。它只是答案知情的逐观测专家选择最大值。新增单测给出两专家各自 gold rank=2、混合后 rank=1 的例子，因此该 Oracle 不能被当作混合分数最佳可实现上界。原始结果快照中的泛称 upper bound 不被继承为当前论文解释。

## 6. D06：预算、停止规则与零选择

| 方法 | 搜索预算 | 拟合/停止与选择 |
|---|---|---|
| 固定专家、等权、RRF | 无调参搜索 | RRF k=60、等权=.5固定 |
| Global | 21权重 | DEV MRR；所有匹配映射共享 |
| Relation | 每个够支持关系21权重 | 60个seed×direction观测；W46个、D91个关系 |
| 原 Query-soft | 1个固定映射 | 同一个已拟合对象，无额外动作调参 |
| 原 ADC | 10β×4τ=40 | 不含显式Global，保留历史结果 |
| 每个新家族 | 40非零+1 Global | 共享对象，DEV MRR、零动作优先 tie-break |
| Dyna R0/R1/R2/R3-fixed | 每种1设置 | 1epoch，学习率5e−5 |
| Dyna R3 | 3学习率×4checkpoint | 每组81条CV轨迹，最多10epoch，DEV选择 |
| Dyna R4 | 继承R3设置 | 不另按角色方向调参 |

新家族没有单独的分类器拟合、学习率搜索或早停。逻辑回归保持 max_iter=2000，四个原最终拟合对象记录的收敛迭代数为4/5/6/5。Dyna 的训练与信息预算不同，表列透明，但不声称与小型逻辑回归逐项等同。

## 7. 核验、修正记录与复现

[动作模块](../../router/matched_actions.py)、[两阶段分析](../../scripts/analyze_paper_a_matched_alternatives.py)、[出表脚本](../../scripts/build_paper_a_matched_assets.py)、[测试](../../tests/test_matched_actions.py)。新增10项测试通过，涵盖等候选数、零动作、Shrink端点、生产投影平局、ADC重现、有界与无局部界映射、gold信息边界、DEV路径拒绝TEST及专家Oracle反例。

真实数据审核状态为 `matched_alternative_checks_passed`，失败列表为空，`test_used_for_selection=false`。原四个最终模型来自 sklearn1.7.2，本机为1.7.1；不假设反序列化天然兼容，而是强制所有最终 DEV/TEST logits、概率与原导出在10^-12内重现，检查通过。

首次 apply 在 Relation 权重检查处停止，尚未生成任何新映射 TEST 结果。原因是评估器按 z_a.dtype 将 JSON float64 权重转换为 float32，再导出为 Python float，而初版审核直接比较未转换的 JSON 值。修正只复现这个实际转换，没有放松 RR 或动态映射核验。保留 `provenance/initial_dev_locks.json`、原分析源码和逐行 diff；修正后 DEV 重放的**全部输出哈希和24个家族参数完全一致**，记录于 `provenance/validation_fix.json`。这不是重新按 TEST 选参。

输入、锁、汇总、预算和8张新增表绑定在 [matched_source_manifest.json](../../paper_a_draft/matched_source_manifest.json)。逐查询输出置于被 Git 忽略的 `query_rows/`，可用于后续服务器统计分析；本轮只提交小型汇总和必要来源。无需基础模型训练或 GPU 重评。

最终论文29页、24张表、3幅图；四轮合计2,393个去重来源的哈希、全部引用/表图、页面边界检查通过。编译没有 overfull 或未解析引用；已检查所有页面缩略图，并放大核验新增主结果、静态、预算和风险表。

已有冻结结果可直接核验出表；首次复现则按顺序执行两个阶段：

```powershell
Set-Location G:\mmkg-project-research
$pythonExe = 'E:\develop\Miniconda3\python.exe'
& $pythonExe -m pytest tests/test_matched_actions.py -q
# 仅首次生成；已有 test_audit.json 时 lock-dev 会拒绝覆盖锁。
& $pythonExe scripts/analyze_paper_a_matched_alternatives.py lock-dev
& $pythonExe scripts/analyze_paper_a_matched_alternatives.py apply-test
& $pythonExe scripts/build_paper_a_matched_assets.py
& $pythonExe paper_a_draft/scripts/compile.py
& $pythonExe paper_a_draft/scripts/verify_draft.py
```

补实验完整并不意味着新的核心主张已获支持。当前适当结论是“参考相对动作有经验性取舍，特定 tanh 优势未建立”；若要提出一个优于这些强对照的新方法，应另立 DEV 协议与验证计划，而不能从本轮 TEST 中挑选替代方案后改称主方法。
