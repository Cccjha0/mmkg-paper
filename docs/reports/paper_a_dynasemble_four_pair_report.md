# Paper A：DynaSemble 四主模型组合实验报告

## 结论摘要

在冻结的 faithful reproduction 协议下，DynaSemble 在四个 strong-primary MMKGC pairs 上都显著低于 DEV-locked Global。它的 pooled TEST 损失从 `-0.016030` 到 `-0.070642`，四个 original-triple clustered bootstrap 95% CI 均完全低于 0。

对核心问题最准确的回答是：

> **DynaSemble 的退化严重程度和 active-selector 效果具有明显 pair sensitivity，但 pooled 结果的方向并不 pair-sensitive——四组全部为负。更突出的风险是 paired-seed fragility：seed 1 和 seed 3 在四组实验中均发生 100% zero-weight collapse，只有 seed 2 保持活动。**

当只看没有 collapse 的 seed 2 时，pair sensitivity 更清楚：M-Hyper + NativE 在 MKG-W 和 DB15K 分别相对 Global 提升 `+0.004598` 和 `+0.001095`，其中前者显著、后者 CI 跨 0；换成 AdaMF-MAT 后分别显著下降 `-0.014371` 和 `-0.037227`。因此 published dynamic mechanism 不是稳定无效，而是其效果同时依赖 secondary identity 和 paired seed，不能被视为 strong-primary 场景中的安全组合方法。

## 1. 完整性与信息边界审计

- 四个 pair 的 `lock.json` 均与 TEST summary 记录的 SHA-256 一致；
- 12 个 selector state 的当前 SHA-256 均与 DEV lock 一致；
- 四组均锁定并运行于 source commit `39242f76365b501de0df403170c6a1c3eee9d7f9`；
- exact full-ranking reference audit 的 rank mismatch 均为 0，最大 reciprocal-rank 误差均为 0；
- Query-soft / Anchored comparison rows 均按 `query_id` 一一匹配；
- MKG-W 每组 TEST 为 25,644 rows / 4,274 original triples；DB15K 每组 TEST 为 59,412 rows / 9,902 original triples；
- 每个 original-triple cluster 内保留三个 seeds 和 head/tail 两个方向；bootstrap 使用预先锁定的 10,000 次重采样与 seed `20260909`；
- selector 只在 DEV 上训练，TEST 未用于 hyperparameter search、early stopping 或 orientation selection；
- Anchored Dynamic 的模型、lock 和结果没有被修改。

据此，本轮结果可以作为 Paper A 的正式 matched external-baseline evidence。

## 2. TEST 主结果

| Dataset | Pair | Primary | Global | Query-soft | DynaSemble | Anchored | Dyna − Global | Bootstrap 95% CI | 相对 Global 降幅 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W | M-Hyper + NativE | 0.359721 | 0.364305 | 0.370506 | 0.348275 | 0.370176 | -0.016030 | [-0.019059, -0.013063] | 4.40% |
| MKG-W | M-Hyper + AdaMF-MAT | 0.359721 | 0.359721 | 0.352470 | 0.310413 | 0.361705 | -0.049308 | [-0.053466, -0.045062] | 13.71% |
| DB15K | M-Hyper + NativE | 0.374567 | 0.374567 | 0.365621 | 0.333422 | 0.375838 | -0.041145 | [-0.043514, -0.038823] | 10.98% |
| DB15K | M-Hyper + AdaMF-MAT | 0.374567 | 0.374567 | 0.347971 | 0.303924 | 0.375281 | -0.070642 | [-0.073491, -0.067838] | 18.86% |

四组结论都不是统计边界情况。DynaSemble 相对 Query-soft 也全部显著更差：四组差值依次为 `-0.022231`、`-0.042058`、`-0.032199` 和 `-0.044047`，对应 bootstrap CI 均完全低于 0。

相比之下，Anchored Dynamic 相对 Global 在四组分别为 `+0.005872`、`+0.001984`、`+0.001271` 和 `+0.000714`。这一对比直接支持 Paper A 的核心论点：问题不在于“动态”本身，而在于 strong-primary 条件下缺少 anchor、bound 和 fallback 的动态调整不具备可靠性。

## 3. DEV–TEST 一致性

| Dataset | Pair | DEV Dyna − Global | TEST Dyna − Global |
| --- | --- | ---: | ---: |
| MKG-W | M-Hyper + NativE | -0.016672 | -0.016030 |
| MKG-W | M-Hyper + AdaMF-MAT | -0.050515 | -0.049308 |
| DB15K | M-Hyper + NativE | -0.041850 | -0.041145 |
| DB15K | M-Hyper + AdaMF-MAT | -0.069188 | -0.070642 |

DEV 与 TEST 的差值高度接近。这说明主要失败不是 TEST distribution shift，也不是在 TEST 上偶然出现的性能波动；selector collapse 和错误的相对权重在 DEV 阶段已经形成，并稳定迁移到 TEST。DEV DynaSemble 指标仍只应称为 in-sample training diagnostics，不能替代 TEST 证据。

## 4. Selector collapse 与 paired-seed fragility

| Dataset | Pair | Seed 1 Δ | Seed 2 Δ [95% CI] | Seed 3 Δ | Seed 1/3 zero fraction | Seed 2 mean effective primary α |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MKG-W | M-Hyper + NativE | -0.028705 | +0.004598 [0.001653, 0.007600] | -0.023982 | 100% / 100% | 0.2755 |
| MKG-W | M-Hyper + AdaMF-MAT | -0.072717 | -0.014371 [-0.018652, -0.010073] | -0.060837 | 100% / 100% | 0.2738 |
| DB15K | M-Hyper + NativE | -0.060874 | +0.001095 [-0.000568, 0.002841] | -0.063655 | 100% / 100% | 0.4135 |
| DB15K | M-Hyper + AdaMF-MAT | -0.090699 | -0.037227 [-0.040218, -0.034241] | -0.084001 | 100% / 100% | 0.4351 |

seed 1 和 seed 3 的 learned primary weight 对每一个 DEV/TEST query 都恰为 0，因此 endpoint safeguard 使 DynaSemble 精确退化为 secondary：

- seed 1/3 的 M-Hyper + NativE DynaSemble MRR 分别与对应 NativE MRR 完全相等；
- seed 1/3 的 M-Hyper + AdaMF-MAT DynaSemble MRR 分别与对应 AdaMF-MAT MRR 完全相等。

进一步读取 locked selector state，并用保存的四维 query features 重算 final-layer pre-activation 后发现：seed 1/3 在四组 TEST 上的所有 pre-activation 都严格小于 0，整体范围约为 `[-0.8130, -0.1216]`；经过 released final ReLU 后全部变为 0。seed 2 的 pre-activation 则全部大于 0，整体范围约为 `[0.2168, 0.9180]`。因此结果与 final-output dead-ReLU collapse 完全一致，而不是 CSV 汇总或 endpoint rank 的实现错误；它也复现了此前 OpenBG-IMG 实验中 seed 1/3 collapse、seed 2 活动的模式。

本协议将 base-model checkpoint seed 与 selector random seed 配对，因此严格来说，主实验识别的是 **paired-seed instability**，不能仅凭这四组结果把全部方差因果归于 selector initialization。不过，跨两个数据集、两个 secondary identities 以及此前 OpenBG-IMG 都重复出现相同的 seed 1/3 final-ReLU collapse，使 selector optimization/initialization 成为最符合现有证据的机制解释。

由于 released architecture 和 initialization 是预先冻结的，本实验不应通过替换 activation、增加 bias initialization、重启失败 seeds 或挑选最佳 seed 来“修复”该现象。collapse 本身就是 faithful reproduction 的重要结果。

## 5. Pair sensitivity

pair sensitivity 在两个层面成立：

1. **退化幅度依赖 pair。** 在两个数据集上，AdaMF-MAT pair 都比对应 NativE pair 更差；DB15K 又比 MKG-W 更差。四组 pooled 损失相差超过 4 倍。
2. **active selector 的符号依赖 pair。** 固定使用唯一活动的 seed 2 后，两个 NativE pairs 均略有正收益，而两个 AdaMF-MAT pairs 均为负。这排除了“所有差异都只是 dead ReLU”这一解释。

这与 secondary 质量和 frozen parameterization 共同相关。seed 1/3 collapse 时模型完全回到 weaker secondary；即便 seed 2 活动，平均 effective primary α 也只有约 0.27–0.44。四组中除 MKG-W / M-Hyper + NativE 的 Global `alpha=0.6` 外，其余三组 Global 都是纯 primary (`alpha=1`)；released `w_a * s_a + 1 * s_b` 参数化只能在 `w_a -> infinity` 时逼近 primary endpoint，任何有限权重都会保留 secondary。这个结构性限制在 AdaMF-MAT 和 DB15K 上尤其昂贵。

因此论文应将结论写成“released dynamic ensemble 在 strong-primary pairs 上缺乏稳健性，并表现出 pair- and seed-dependent degradation”，而不是笼统声称所有动态融合都无效。

## 6. Head/Tail 方向分析

| Dataset | Pair | Head Δ Dyna − Global | Tail Δ Dyna − Global |
| --- | --- | ---: | ---: |
| MKG-W | M-Hyper + NativE | -0.007848 | -0.024212 |
| MKG-W | M-Hyper + AdaMF-MAT | -0.035252 | -0.063365 |
| DB15K | M-Hyper + NativE | -0.018413 | -0.063876 |
| DB15K | M-Hyper + AdaMF-MAT | -0.043616 | -0.097669 |

四组均为 tail 方向损失更大。对应地，Primary 与 Secondary 的 TEST MRR 差距在 tail 上也始终更大：

- MKG-W / NativE：head `0.006529`，tail `0.036840`；
- MKG-W / AdaMF-MAT：head `0.045124`，tail `0.081680`；
- DB15K / NativE：head `0.029626`，tail `0.093985`；
- DB15K / AdaMF-MAT：head `0.057468`，tail `0.126488`。

因此 tail 退化不是孤立方向效应，而是当 primary 在该方向上的可靠性优势更大时，collapse 或过度引入 secondary 会付出更高代价。

## 7. Query-level negative-transfer 诊断

以下指标是在 TEST 结果生成后进行的描述性诊断，未用于选择 DynaSemble 配置；其中 `ΔRR <= -0.1` 的 severe-harm threshold 不是预注册 confirmatory 指标，应在正文中明确标注为 retrospective analysis。

| Dataset | Pair | Harm rate | Improve rate | Severe harm | CVaR10% ΔRR | Top-1 loss | Top-1 gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MKG-W | M-Hyper + NativE | 30.77% | 39.37% | 9.46% | -0.4120 | 6.26% | 2.47% |
| MKG-W | M-Hyper + AdaMF-MAT | 39.69% | 37.63% | 15.86% | -0.6024 | 10.75% | 2.49% |
| DB15K | M-Hyper + NativE | 41.31% | 31.33% | 14.50% | -0.5781 | 8.97% | 2.54% |
| DB15K | M-Hyper + AdaMF-MAT | 46.88% | 31.77% | 20.21% | -0.6175 | 13.70% | 2.22% |

MKG-W / M-Hyper + NativE 尤其说明只报告“多少 query 变好”是不够的：其 improvement rate 高于 harm rate，但 conditional mean harm 为 `-0.1419`，conditional mean gain 只有 `+0.0702`，最终 MRR 仍显著下降。换言之，少量或中等收益被更重的 lower-tail damage 抵消。这正是 Paper A 需要显式 negative-transfer risk 和 conservative correction 的原因。

## 8. 与 Query-soft 和 Anchored Dynamic 的关系

- Query-soft 只在 MKG-W / M-Hyper + NativE 上相对 Global 为正；其余三组分别下降 `-0.007251`、`-0.008946`、`-0.026595`。
- DynaSemble 在四组上比 Query-soft 更差，说明把 query policy 换成 published score-statistics MLP 并不能自动解决 strong-primary negative transfer。
- Anchored Dynamic 在四组全部为正，且 12/12 seed cells、8/8 dataset-pair directions 均为正；其收益虽小，但稳定性与两个 unconstrained baselines 形成直接对照。
- 对 Paper A 而言，最有价值的结论不是 Anchored 获得最大绝对 MRR，而是它避免了 DynaSemble 的 catastrophic endpoint collapse，也避免了 Query-soft 在三组 strong-primary pairs 上的系统性负迁移。

## 9. 建议的论文表述

主文可以采用如下结论：

> A faithful reproduction of DynaSemble exhibits substantial pair and seed sensitivity on strong-primary MMKGC ensembles. Although its active seed slightly improves over the static global solution for the two M-Hyper–NativE pairs, two of three selector seeds collapse to the weaker secondary expert across all four pairs, and the pooled result is significantly negative in every case. These findings motivate conservative adaptation around a reliable DEV-locked anchor rather than unconstrained query-dependent reweighting.

需要同时保留三条边界：

1. 这是 released mechanism 在四个 frozen MMKGC pairs 上的外部泛化结果，不是对 DynaSemble 原论文任务和全部数据集的全面否定；
2. 三组 Global 为 primary endpoint，而 released asymmetric parameterization 不能用有限权重精确表示该 endpoint；这不是本实验的实现偏差，而是需要透明报告的适用边界；
3. 不应报告 best-seed DynaSemble 作为主结果。最佳 seed 会掩盖 2/3 seeds 的 deterministic collapse；主表必须保留 pooled three-seed protocol，并把逐 seed 结果作为稳定性证据。

## 10. 结果文件

- 主总表：`outputs/paper_a_safe_correction/dynasemble/four_pair_summary.csv`
- selector 汇总：`outputs/paper_a_safe_correction/dynasemble/four_pair_selector_summary.csv`
- retrospective risk 汇总：`outputs/paper_a_safe_correction/dynasemble/four_pair_risk_summary.csv`
- 每组 bootstrap：`outputs/paper_a_safe_correction/dynasemble/<dataset>/<pair>/clustered_bootstrap_ci.json`
- 每组完整审计：对应目录下的 `lock.json`、`dev_summary.json`、`test_summary.json` 与 query rows。
