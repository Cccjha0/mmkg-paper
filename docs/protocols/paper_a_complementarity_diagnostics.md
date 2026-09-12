# A01/A04/E01：互补性、模态支持与贡献范围诊断

本协议在本轮聚合分析之前记录，属于已查看历史 TEST 后的描述性补分析，不是独立预注册。沿用 C07/C08 的后验证据地位。使用全部六个既定 pair、三个 base seed、head/tail 两方向；不训练、不选新模型、参数、pair、策略或 TEST 子集。

## 输入与指标

- 只使用 B07/C04 修正后的、已绑定 SHA-256 的端点 ranks/RR、P3 grouped held-out DEV 策略输出和锁定 TEST 输出。DEV 的 ADC 和 Global 必须来自相同 OOF observation；不使用 full-DEV 拟合值代替 OOF 结果。所有 join 用完整 query identity 并检查一一对应。
- 报告 A/B/tie 端点 winner 份额；A 是既定 DEV primary。`oracle = mean(max(RR_A, RR_B))`；`best single = max(mean(RR_A), mean(RR_B))`。Oracle gap 是两个端点的答案感知诊断，不是所有混合可达到的上界。
- 排名分歧为 gold 端点 rank 的不同率、`mean(abs(log2(rank_A/rank_B)))` 及 gold Hits@10 分歧。它们不等于完整候选排序的 Kendall/Spearman 距离，不能作为可部署输入。固定 rank-ratio 桶：相同、(1,2]、(2,4]、>4；用整数 rank 比值分桶。
- `delta = RR_ADC - RR_Global`；报告均值、平均正收益、平均损失、受损频率、权重调整率及显式 fallback 率。每个分层同时报告条件均值及 `n_group/n_total * mean(delta_group)`，并检验后者之和恢复整体 delta。
- 同时按 winner、rank-ratio 桶、方向、已知实体模态支持、关系以及关系×方向分层。所有关系与稀疏/零支持状态留在 CSV；图只显示 DEV 和 TEST 同时至少含 30 个原始 triples 的关系×方向，门槛仅用于可读性，不用于选择策略或支撑显著性。
- 明确支持计数：seed-direction observations、原始 triples、去 seed 的 directional query 数分别记录。没有把多 seed 当成独立样本，也不新增显著性检验或 bootstrap。

## 可观测模态支持边界

从每套数据已冻结 M-Hyper seed 1 checkpoint 提取 `has_img/has_text`，核对 checkpoint SHA-256 与之前 C01–C03 审计、mask tensor hashes 与 dataset manifest；先前审计已经核对全部 18 个 checkpoint 的相同 buffers。此轮仅提取布尔 mask，禁止训练。

tail query `(h,r,?)` 只查已知 `h` 的支持；head query `(?,r,t)` 只查已知 `t` 的支持。四个状态固定为 neither、image-only、text-only、both。不访问 gold 的属性；不向现有 13 特征 selector 追加任何变量。加入单测：改变未知端的 gold 或 gold 的 mask，已知 query 的支持分组保持不变；修改已知端的 mask 应能改变分组。

覆盖标记说明输入是否存在，不是图像/文本质量、语义相关性或因果效应。关系、实体、模型与缺失机制可能混杂。ADC 接口只有两套分数及方向，不能把观察到的差异归因于多模态独有机制；论文将其定位为在 MMKGC 场景研究的一般分数后处理。

## 最近邻方法比较

逐项引用 DynaSemble §3（pp.206–207）、MoSE §§3.5–3.6/Appendix A（pp.10530–10531/10536），并区分 MoSE 的平均、关系 boosting、实例元学习三种版本。DynaSemble 的固定一个系数不等于围绕 DEV Global 的有界修正；无显式拒绝不等于不能数值上输出零系数。ADC 的拒绝是返回 Global、仍给预测，不是拒绝回答。

把行动映射、范围与拒绝语义作为比较对象，不声称首创动态集成、分数几何输入、tanh/clip 或收缩。共享拟合对象和预算的 Shrink/Clip-g 对照已显示 tanh 的独特增量价值未获支持；此次不改写该结论。MoSE 仅为有文献依据的机制比较，没有本研究的新运行数值。
