# C11/C12：query key 分组边界与统一成对区间

日期：2026-09-12。协议先于本轮分析写入 `docs/protocols/paper_a_query_pair_review.md`。使用已核验的六个 pair、三 base seed、两个方向、原五个 DEV folds 和冻结 TEST 策略。新增 60 次 CPU 单线程小型 logistic 拟合（3–6 次迭代收敛），没有基础模型训练、候选评分或新 TEST 策略。

## C11：原三元组隔离不等于可观测 query 隔离

可观测语义 key 为 `(direction,r,known entity)`，即 tail 的 `(tail,r,h)` 与 head 的 `(head,r,t)`。三个 seed 是同一语义请求在不同 checkpoint 上的重复，统计唯一 key 时不重复计数。一个原三元组贡献一行 head 与一行 tail；一个 key 可对应多个正确答案。`query_inventory_dev.csv`、`query_inventory_test.csv`、`query_folds.csv` 分别提供 split、方向与每 fold 的完整分母。

| DEV 数据／方向 | 答案行 N | 唯一 key K | 多答案 key／K | 多答案行／N | 跨 fold key／K | 留出行有训练 key／N |
|---|---:|---:|---:|---:|---:|---:|
| MKG-W head | 4,276 | 2,303 | 685 / 29.74% | 62.16% | 622 / 27.01% | 59.12% |
| MKG-W tail | 4,276 | 4,133 | 130 / 3.15% | 6.38% | 109 / 2.64% | 5.40% |
| DB15K head | 7,922 | 4,350 | 928 / 21.33% | 56.80% | 820 / 18.85% | 53.98% |
| DB15K tail | 7,922 | 7,394 | 465 / 6.29% | 12.53% | 380 / 5.14% | 10.35% |

跨 fold key 占据至少两个原三元组 fold。该 key 上所有留出答案行都能在对应训练部分找到同 key，因此跨 fold 行与逐 fold 训练重叠行的总数一致。合并方向、恢复三 seed 后，MKG-W 为 8,277/25,656=32.26%，DB15K 为 15,288/47,532=32.16%，复现 C09 数字。三个 pair 的计数、三元组及分组逐一一致。头预测重复更集中，最大单 key 答案数 DEV 为 442/660、TEST 为 467/859（MKG-W/DB15K）。这也限制把不同图事实当独立抽样单位的解释。

TEST 头／尾方向在 DEV 中已有相同 key 的答案行分别为 MKG-W 2,700/241（各以4,274为分母）、DB15K 5,708/1,252（各以9,902为分母）。所有 key 使用可观测实体和方向；没有把 gold 身份作为 selector 新输入。原报告现在明确为**转导 query 分布上的留出三元组评估**，不声称全新 query、实体或关系的端到端泛化。

## DEV 清除共享 key 的敏感性

保留原外层 holdout，删除外层训练中只要 head 或 tail key 与 holdout 重合的整个三元组；全部六个观察一起移除。所有 30 个 Purged 训练范围与 holdout 的 key 交集为空。原训练范围的34xx／63xx三元组分别缩减为 MKG-W 2,127–2,194、DB15K 3,784–3,840；删去比例为35.77–37.86%／39.40–40.33%。

另设 Random-size：每 fold、每 relation 保留与 Purged 完全相同的三元组数，按固定 SHA256 排序取样。它仅是一个预定样本，不能代表训练子集随机性的分布。两种新范围分别重选 anchor、拟合 median/scaler/balanced logistic、在训练部分选原40项 β/τ；random_state 与原 fold 相同。与历史原模型共90组记录、3,600项候选分数，原30组选择与逐行输出全部复现。六观察、双方向共享 key、等量关系配额和 holdout RR 不参与清除规则均有单测。

| Pair | 原 OOF MRR | Purged MRR | Random-size MRR | Purged−原 [条件95%CI] |
|---|---:|---:|---:|---|
| W-N | .358821 | .356817 | .358091 | −.002004 [−.003658, −.000190] |
| W-A | .354026 | .354239 | .354076 | +.000214 [−.000202, .000616] |
| D-N | .382578 | .382764 | .382758 | +.000186 [−.000112, .000473] |
| D-A | .382623 | .382889 | .382862 | +.000266 [−.000013, .000557] |
| W-NA | .335339 | .335397 | .335338 | +.000059 [−.000218, .000338] |
| D-NA | .321052 | .321032 | .321028 | −.000021 [−.000182, .000144] |

W-N Purged−Random-size 为 −.001275 [−.003119,.000583]；所有六组该差异区间均跨零，不是等效证明。W-N 的下降主要落在原本就没有训练 key 匹配的留出子集（MRR .462655→.459730），共享 key 子集为 .140802→.140733。因此不能把下降直接解释为去除了“泄漏红利”。训练组成、样本量和 anchor 都可能改变，且 Random-size 只有一个固定样本。Purged 的 W-N 平均损失从 .004701 增为 .008147；D-A 虽有小幅正 MRR，干预从 .003640 增为 .160692、损失从 .000125 增为 .001102。各方法的损失相对自己的训练所得 anchor，不能混为同一参照下的纯 selector 效应。

`dev_effects.csv` 同时给 Purged−原、Random-size−原和 Purged−Random-size；`dev_fits.csv` 给每 fold 的训练/拟合类别数、参数、收敛迭代和范围哈希；`dev_seed_metrics.csv`、`dev_key_subsets.csv`、`dev_summary.csv` 保留 seed、子集和风险数值。2,000次三元组 percentile 区间条件于已拟合模型、固定子集、fold和专家，未重拟合，不能覆盖重叠训练集、图依赖、开发适应性或随机子集不确定性。基础 checkpoint 用完整 DEV 选择，未随 fold 重训，因而**即使 Purged 也只证实组合层训练的 query 隔离**。不据此变更最终模型或新试 TEST。

## C12：同一成对 percentile 流程

所有70项 ADC−方法以及12项 full−expanded-radius／full−no-fallback，共82项，使用相同的逐观察成对流程。先按 `(seed,direction,h,r,t)` 精确匹配，拒绝缺失／重复记录；R3 在同一观察内平均3个 selector 重复；随后差分并平均每个原三元组的6个观察。按三元组字典序，用 NumPy PCG64 每次有放回抽取N个三元组，10,000次。MKG-W seed=2026091212、DB15K seed=2026091213；同数据集全部pair及比较共用抽样顺序。区间为线性插值2.5/97.5 percentile，无 normal approximation。

所有旧成对点估计与 component MRR 复现；full−no-fallback 与 B18 直接效应复现。只重新统一区间抽样，没有重新挑选政策或 bootstrap seed。主 Global 表和 query-CI／seed-SD 并列表也更新为本轮结果。旧文件与哈希保持原状；保留的 DynaSemble 10,000次、拒绝分析2,000次历史诊断明确标注，当前方法对比引用统一表。全部82项有完整CI、3个seed效应及sample SD，并保留492个seed×direction单元和聚类顺序哈希。

重点 W-N：

- ADC−Query-soft：+.000569 [−.000324,.001448]，差异未确定，不是等效。
- ADC−Relation：−.001809 [−.003473,−.000062]，此次未校正点态区间支持 Relation 更高，不能再引用旧 normal 区间跨零来概括。
- full−expanded-radius：+.000046 [−.000860,.000958]。
- full−no-fallback：+.000010 [−.000475,.000480]。

四主pair的 full−expanded-radius 依次为 +.000046、−.001170、+.000233、−.000112；W-A、D-A 区间低于零，W-N、D-N跨零。两个附加pair分别 −.000095 [−.000384,.000155]、−.000240 [−.000403,−.000073]。较小半径并非普遍免费。

full−no-fallback 的 W-A 为 −.000033 [−.000333,.000277]，D-N +.000066 [−.000121,.000259]，D-A +.003201 [.002653,.003754]；附加 W-NA +.003389 [.002211,.004592]、D-NA +.000649 [.000360,.000944]。fallback 的条件性结论保留。当前修复后六锁均τ>0，没有一项比较逐观察恒为零；不将跨零或近零写成相同策略。

主四pair ADC−Global仍是中心科学比较，但历史TEST盲性未建立，不能称本轮确认性假设检验。其他比较与敏感性均为探索性。全部区间未做多重比较调整，不宣称家族错误率保证、不以显著项数量计独立证据；它们仅反映固定模型下三元组抽样的不确定性，不包含seed重训、query/图依赖或开发过程不确定性。合适的下一步由该边界决定，而非凭跨零推断等效。

## 交付与验证

分析：`scripts/analyze_paper_a_query_pair.py`（DEV先完成，TEST只读统计另阶段）；正式表：`scripts/build_paper_a_query_pair_assets.py`；结果：`outputs/paper_a_safe_correction/query_pair_v1/`；哈希：`paper_a_draft/query_pair_source_manifest.json`。

6项合成单测通过，覆盖可观测key的gold不变性、整三元组清除、关系等量控制、分母区别、显式六观察重采样与反向区间、行顺序不变和不完整聚类拒绝。完整论文核验另外核对60次拟合、30组重放、82个区间、492个配对单元、同数据集的共同cluster顺序与零TEST选参标记。生成物仅为小型CSV/JSON和论文表，不含模型、分数矩阵或逐query大文件。
