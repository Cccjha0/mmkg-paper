# C09/C10：选择层级、开发适应性与 seed 波动

日期：2026-09-12。保持原六 pair、三 base seed、两方向和 TEST 策略；所有新增选择只使用 DEV。执行了 108 次小型 logistic 拟合，未重训/评分基础模型，无需服务器强算力任务。协议在分析前写入 `docs/protocols/paper_a_selection_seed_review.md`。

## C09：原设计的有效范围

原外层算法可以评价，不能仅因外层训练部分同时用于拟合和选 β/τ 就称为 holdout 泄漏。源码路径为 `ablate_anchored_dynamic.py` 的外层循环：

1. 以原三元组分组，三个 seed × 两方向共六行进入同一 fold；按关系分层 SHA256 排序，5fold，seed=20260901。
2. 外层训练部分选择 Global α0；非平局训练行拟合 median、StandardScaler 和 balanced logistic。
3. 同一外层训练部分上的 fitted preferences 给40个 β/τ 候选打分，按 MRR、较小β、较大τ的精确字典序选择。
4. 将训练所得 α0/预处理/模型/β/τ 应用于外层 holdout；holdout 的 RR 不参与这一次选择。

最终 `lock_apply_anchored_dynamic.py` 检查 P3 与 full-DEV 的 feature order、β/τ/grid 一致，再对 full DEV 按同一训练规则重做 anchor、拟合和训练部分选参。外层 full-geometry 拟合 random_state=20260902+(fold−1)×10+2；最终 full DEV 为20260902。样本量和随机状态不同，算法层级一致。历史代码的 `nested-selection` 字符串不意味着做过 inner CV；正式文字明确区分。

**评价对象是组合层，条件于已冻结的基础模型、A/B角色和特征族。** 基础 checkpoint 的完整 DEV 选择不在每个 outer fold 重做；不能把这份 OOF 称为从基础训练到组合部署的整个流水线验证。缓存 RR 也保留已披露的 TRAIN∪DEV∪TEST filtered fact 规则；新增 inner OOF 没有改变该评分协议。

**开发适应性另行披露。** 已有可追溯时间线显示：9月1日加入 grouped OOF；9月2日初始 ADC β上限.20，随后 P3扩到.50并加特征/anchor/fallback消融；9月3日加入 full-DEV lock，四pair TEST数值与锁摘要首次在同一提交中被找到；后续修复、半径与 matched-family 工作均晚于已记录的 TEST曝光。历史报告描述过 DEV-led development，但没有完整人类访问/决策日志可确定每次 OOF查看、迭代次数和因果影响。既不保证“从未适应OOF”，也不将无法排除影响说成已证明某次 TEST调参。新 inner OOF不能消除过去开发适应性；全部结果继续是 retrospective fixed-family evidence。

**grouped不等于query独立。** 原三元组没有跨fold，但不同gold可能共享相同可观测query。外层holdout中，其 `(seed,direction,relation,known entity)` 已在外层训练出现的行比例：MKG-W 8,277/25,656=32.26%，DB15K 15,288/47,532=32.16%，每数据集三pair相同。这不是 gold进入特征的证据，也不是 entity-ID错位；它限定OOF衡量的是留出三元组，不能自动外推为新实体、新关系或全新可观测query的泛化。

## 新增训练内 OOF 敏感性

原五fold不变。每个外层训练部分再分三组，inner fold seed=2026091209；每个内层训练部分独立选择anchor和拟合预处理/classifier，只在其inner holdout上计算40候选RR。以各holdout行数加权合并，再选β/τ；将其用于原外层训练的anchor和原classifier后评价外层holdout。另以Global作为第41项单独报告。没有增加β范围、feature family、inner fold数或随机重复搜索。

六个full-DEV范围执行同样三内折选参，仅报告选择，**不应用新TEST策略**。共6×(5+1)×3=108次小型拟合，liblinear均在3–7次迭代收敛；重放30份原balanced外层模型状态、6份full-DEV锁；144项外层/full-DEV选择和2,952条候选分数全部保留。7项单测覆盖内层拟合边界、完整三元组分组、holdout标签不改变训练范围、精确tie规则、Global候选、sample SD及六观察聚类。

| Pair | 原40 OOF MRR | Inner40 OOF MRR | Inner−原，条件95%CI |
|---|---:|---:|---|
| W-N | 0.358821 | 0.358700 | −0.000120 [−0.000689, 0.000456] |
| W-A | 0.354026 | 0.354026 | 0，逐行相同 |
| D-N | 0.382578 | 0.382573 | −0.000005 [−0.000080, 0.000068] |
| D-A | 0.382623 | 0.382635 | +0.000011 [−0.000003, 0.000030] |
| W-NA | 0.335339 | 0.335379 | +0.000040 [−0.000179, 0.000246] |
| D-NA | 0.321052 | 0.321037 | −0.000016 [−0.000113, 0.000095] |

六个差异区间均包含零，不能据此宣称两算法等价或inner选择更优。区间以2,000次原三元组bootstrap计算，保留全部六行，条件于已拟合模型/选择/folds；不重新拟合，也不覆盖模型开发适应性和重叠训练集带来的不确定性。

outer β/τ在W-N/W-A/D-N/D-A/W-NA/D-NA中分别改变4/0/2/2/4/3个fold，共15/30。所有36个范围的resub-41和inner-41均未选Global，结果分别与各自40项版本相同。W-N的intervention由63.29%升至83.93%，无条件平均损失由.004701升至.006263；D-NA的干预由3.85%降至2.13%、损失由.000393降至.000186，但净MRR也略降。取舍全部披露。

full-DEV inner选择为W-N .50/.1、W-A .50/.1、D-N .20/0、D-A .35/.3、W-NA .40/.2、D-NA .45/.1；相对原锁四组变化。论文及TEST继续使用原参数。这个诊断既不授权追试新TEST，也不把原外层程序重新定义成无效。

## C10：两个不确定性对象

所有76个主比较method×pair分别公开三seed绝对MRR、均值与sample SD（ddof=1），以及456个seed×direction绝对MRR；70个ADC−对照差异先在每seed内配对，再报告其SD，不能减两个边际SD。R3先在每个base-seed/direction/query平均三selector重复，再计算base-seed MRR，与主表一致。不是9个base seed。

| Pair | seed1 ΔMRR | seed2 ΔMRR | seed3 ΔMRR | 平均Δ ± seed SD | 条件query 95%CI |
|---|---:|---:|---:|---|---|
| W-N | +.001837 | +.002707 | +.002796 | +.002447 ± .000530 | [.001481,.003425] |
| W-A | +.000612 | +.001321 | +.001110 | +.001014 ± .000364 | [.000503,.001549] |
| D-N | −.000100 | +.000010 | −.000153 | −.000081 ± .000083 | [−.000254,.000091] |
| D-A | +.000193 | +.000338 | +.000104 | +.000212 ± .000118 | [.000070,.000378] |
| W-NA | +.000636 | −.000059 | −.000160 | +.000139 ± .000433 | [−.000249,.000547] |
| D-NA | −.000039 | −.000061 | +.000056 | −.000015 ± .000062 | [−.000163,.000129] |

Δ均为当前ADC−Global。最后一列直接保留已绑定10,000次原三元组聚类bootstrap，而非重新调随机seed。它反映固定模型条件下的query抽样波动；中间SD描述这三个已观察checkpoint pair在同一query集上的离散程度。SD不是SE、总体训练seed的CI，也不补偿选择器拟合或历史开发不确定性。ADC本来就是一套pooled selector跨三个base seed，因此也不能把三行称为三个独立selector训练。

例如W-N的Global/ADC绝对MRR分别为.364665±.002019和.367112±.001494，但配对增益SD为.000530；共享query和checkpoint使边际波动与配对波动不同。D-N和两个附加pair的seed效应正负混合。失败Dyna R0仍完整呈现，其四主pair的seed SD为.016710/.029854/.036519/.030444。它不能被删除失败seed后重新包装。

三seed结果不是自动无效；新增训练seed是进一步评估稳定性的可选实验。当前结果不支持把seed×direction或跨pair的12/12、8/8等计数当独立重复次数。六张新正式表同时列选择敏感性、每fold/full-DEV参数、所有方法的绝对seed指标和query/seed两种统计口径。

生成物位于 `outputs/paper_a_safe_correction/selection_seed_v1`；代码、冻结协议、候选分数、内层范围、结果表、源哈希和报告随提交保留。无新score矩阵、checkpoint或逐query大文件。
