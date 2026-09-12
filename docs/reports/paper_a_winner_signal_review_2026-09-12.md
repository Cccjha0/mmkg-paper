# B13 / B14：winner-label 错位与 class-balanced 偏好分数

本轮完成六个 pair 的实际错位分析，并加入共享特征／分组／预算的 DEV 类别权重对照。B13 不再只停留在 limitations；B14 明确区分偏好分数、自然 winner 概率与行动相关的伤害概率。原有模型与 TEST 结果保持不变，无需服务器重新评分。

## B13：量化标签方向与 anchor 效用的错位

使用原有 21 点 RR 缓存。DEV 使用原有五折训练部分确定的 anchor、radius 和 held-out ADC action；TEST 使用原始 full-DEV locks。对每个带答案观察，只在原 radius 内考察较低／较高 α 是否存在比 anchor RR 高超过 1e-12 的点，分别保留下调有益、上调有益、两侧都有、两侧都无。表中按 A winner、B winner 和 tie 分层，覆盖六 pair × DEV/TEST，共 36 行列联表、474,732 个观察。

这些是 gold-aware 的存在性诊断，不是推理特征、新训练标签或可部署的 oracle 策略。它们也不是连续导数，继续使用报告中的 raw endpoint dispatch。

- W-N：在非并列、有获益机会且 winner 建议方向可行的观察中，DEV 有 889/15,974（5.57%）、TEST 有 883/15,780（5.60%）只能往相反方向获益。
- W-NA：对应错位为 14.66% 和 14.82%。
- W-A、D-N、D-A、D-NA 的零错位率具有结构原因：primary 端点只允许下调，条件已经要求建议方向可行且有获益机会，不能将该零值解释为方向预测完美。被端点阻断的建议另行报告；例如 W-A TEST 有 3,928 个 A winner 观察能通过下调改善 anchor，却无法向 A label 建议的上调方向移动。
- 实际修正幅度仍可造成损害。W-N TEST 有 6,806 个非零行动与端点 winner 方向一致，其中 66.34% 受益、7.66% 受损；受损者平均损失 0.0772 RR。方向一致并不表示具体幅度正确。
- 这些 aligned 行动的条件效用为 +0.02095，说明信号在部分条件下有用，但该 gold 条件不能拿来做部署 gate。其全体效用贡献为 +0.005561；winner-opposed 行动贡献 −0.003195；tie 行动贡献 +0.000081；未调整为 0，合计仍为原有 +0.002447。
- 3,555 个 W-N TEST 观察虽然在建议方向存在有益的 radius 内点，但邻接的 0.05 步并无正收益。机会、方向和幅度三个问题不能合并。

完整 CSV 还报告条件损失、无条件平均损失、受损幅度的 95 分位数、全部 alignment 分组及相邻步长效用。空组保留为空，不写成零损失。

## B14：balanced 与 unweighted 的 DEV 对照

对六个 pair 各执行五折：复现 30 个 balanced 模型，另拟合 30 个 class_weight=None 模型。13 个特征、pooled 非并列 DEV 行、imputer/scaler、C=1、liblinear、迭代上限和随机种子均相同。预处理逐项相同；原 balanced 的 held-out ADC 权重／RR、Query-soft RR、fold 分组及 anchor 全部复现，共 219,564 个 DEV 观察。最多七次迭代；单 CPU 线程；不调用基础模型评分。

两个 learner 各有同一组 41 个 ADC+Global 候选，按 outer training portion 的 MRR 选择，包括始终 Global；合计 2,460 次候选评估。Query-soft 直接使用各自相同的拟合对象，不另拟合或调参。保存全部候选、选择、fitting scope、模型和预处理数值状态。没有生成最终 full-DEV 控制模型，没有将 unweighted 控制应用到 TEST；DEV 产物在 TEST 固定策略诊断前已序列化。

概率指标在自然频率加权的 held-out **非并列** winner 观察上计算；效用／损失／干预率则包含全部观察。记录 Brier、log loss、ROC AUC、10 个固定等宽分箱的 ECE，并保留训练折 winner prior 这个概率评估参照。Brier/log loss 不是纯校准指标，ECE 依赖分箱；prior 的近零 ECE 并不带来折内 query 区分能力。

| Pair | Balanced ECE | Unweighted ECE | Balanced ADC+Global ΔMRR | Unweighted ADC+Global ΔMRR |
|---|---:|---:|---:|---:|
| W-N | 7.79% | 1.63% | +0.003070 | +0.003681 |
| W-A | 4.72% | 0.69% | +0.001204 | +0.000678 |
| D-N | 13.61% | 0.84% | −0.000009 | −0.000003 |
| D-A | 14.16% | 1.04% | +0.000036 | +0.000027 |
| W-NA | 12.98% | 0.28% | +0.000147 | 0 |
| D-NA | 1.39% | 0.23% | +0.000031 | +0.000082 |

六组 Brier 与 ECE 均下降，而策略效用并非统一提高。W-N 的无条件平均 RR 损失由 0.00470 增至 0.00674，干预率由 63.29% 增至 82.85%；W-A 与 D-A 效用下降；W-NA 每折都选择 Global；D-N 两种拟合均略负。Unweighted Query-soft 相比 balanced 版本在六组都有改善，但五组仍低于 Global。没有据此宣布新的最佳 weighting 或替换 TEST 结果。

## 概念与证据边界

实现的 balanced 权重为 n/(2n_c)。在固定类别权重、无正则且模型不受限的理想 weighted log-loss 最优解下，若 π 是自然非并列胜率，则 q*=w1π/[w1π+w0(1−π)]。该公式解释加权目标与自然胜率为何不同，并不声称实际正则化 logistic 达到理想解。模型又排除了 RR tie，因此不能直接把任何该二元输出说成全体 query 的自然胜率。

方法使用 signed margin g 与其 sigmoid 偏好分数 p_A。正文把 confidence 限定为 margin strength；p_A、1−p_A 和 |2p_A−1| 均不被解释为伤害概率。B14 对照是移除类别权重，不是 post-hoc calibration，也没有产生概率安全保证。

已核查的主来源：[scikit-learn 1.7 LogisticRegression 的 class_weight 定义](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)、[概率校准指南及 Brier/log-loss 解释](https://scikit-learn.org/1.7/modules/calibration.html)。论文还沿用 Guo et al. (2017) 的相关工作引用；旧的修复前 calibration/coverage 快照未当作新证据。

## 产物与核验

- 协议：`docs/protocols/paper_a_winner_signal_review.md`
- 纯诊断逻辑：`router/winner_signal_diagnostics.py`
- 两阶段分析：`scripts/analyze_paper_a_winner_signal.py`
- 表格构建：`scripts/build_paper_a_winner_signal_assets.py`
- 证据与全部数值：`outputs/paper_a_safe_correction/winner_signal_review_v1/`
- 绑定清单：`paper_a_draft/winner_signal_source_manifest.json`
- 论文新增三张表：完整 winner/方向列联、方向与幅度错位、DEV 类别权重对照。

11 项单测通过，包括端点 winner 不能推出有益方向、同一方向的幅度错位、端点不可行、tie 类别、损失分解、缓存异常、weighted-loss 目标及自然频率指标。全部原始策略和候选设置按预定核查通过。DEV receipt 已冻结，`dev` 阶段拒绝覆盖；重复核查现有产物可执行 `diagnose`、表格构建及论文校验，无需重训。

最终论文编译无 overfull box 或未解析引用。全稿 60 页、52 表、4 图的 2,982 个源文件绑定及页面边界检查通过；人工检查第 42–45 页，新增 Tables 30–32 与正文公式可读，无遮挡。此次提交的文件总量约 0.86 MB，最大单文件约 334 KB；原有大型结果文件未包含在提交中。
