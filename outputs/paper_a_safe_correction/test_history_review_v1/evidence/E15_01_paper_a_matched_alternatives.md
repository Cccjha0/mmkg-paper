# A02 / D04–D06：最邻近替代方案的匹配比较协议

2026-09-11，在本轮新增映射计算前固定。已有历史 TEST 可见，本研究是问题驱动补充，不是新的盲测。保留原 ADC/Query-soft/DynaSemble 结果；新增政策先完成四组 DEV 锁，再统一应用 TEST，不依 TEST 选择家族、参数或报告子集。

## 共享对象与数据

四主配对 W-N/W-A/D-N/D-A；三 base seeds、head/tail、原三元组分组五折（fold seed 20260901）。所有新映射共用同一 13 维几何、标签、预处理、拟合对象、g 与 p。每个折仅重建一次原 full_geometry 逻辑回归，必须复现原 held-out ADC/Query-soft 权重或 RR、fallback 和原选参；失败则停止。最终四组直接加载已锁的同一个 `anchored_model.pkl`，先在 DEV 重放输出核验，随后序列化新动作参数锁。

候选实体、z-score、filtered RR、0.05 权重网格及投影 tie-break 均复用 B07 修正结果。特征不接收 gold 或 filter。历史 ADC DEV 的 filtered evaluation 事实范围仍为 TRAIN∪DEV∪TEST；不新增 TEST 正例监督，不把该 DEV 值直接等同于严格 DEV Dyna 分数。新锁定阶段不得打开任何 TEST 逐查询输入。

## 六个等预算家族

记 a 为 anchor，p=σ(g)，共同外层 clip 将权重限制到 [0,1]，随后映射到相同精确排名网格。对有 gate 的家族，`|2p−1|<τ` 时返回 a；非有限输入对所有家族返回 a。

| 家族 | gate 之前的提案 | 非零强度网格 | τ | 候选数 |
|---|---|---|---|---:|
| ADC+Global | a+β tanh(g) | β=.05,.10,…,.50 | 0,.1,.2,.3 | 40+1 |
| Shrink | (1−λ)a+λp | λ=.025,.050,…,1.000 | 固定0 | 40+1 |
| Shrink-gate | (1−λ)a+λp | λ=.1,.2,…,1.0 | 0,.1,.2,.3 | 40+1 |
| Linear-p | a+β(2p−1) | β=.05,.10,…,.50 | 0,.1,.2,.3 | 40+1 |
| Clip-g | a+β clip(g,−1,1) | β=.05,.10,…,.50 | 0,.1,.2,.3 | 40+1 |
| Linear-g | a+γg | γ=.05,.10,…,.50 | 0,.1,.2,.3 | 40+1 |

每族另有一个**显式始终 Global**候选，强度=0、τ=0，不通过 gate 阈值模拟。Shrink λ=0 确切返回 Global、λ=1（无 gate）等于 Query-soft。纯 Shrink 用预算换更细 λ，gated 家族用预算选择阈值；不以无效重复项填充搜索次数。Linear-g 只有 [0,1] clip，不具备 β 局部半径，须单独说明；Linear-p 与 Clip-g 保持与 ADC 相同的局部半径。

所有家族每次选择均评估 41 个声明参数配置；实际网格权重可能重合，另记录实际不同动作向量数。共享 anchor 的 21 点搜索、共享分类器拟合只执行一次，不重复计作每族独有预算。原未选参 Query-soft 另报固定一个配置；原 ADC 为40个配置、不含Global，作为历史对照保留。

## 选择、锁定与评价

外层每个训练部分使用该部分 MRR 选择每个家族的配置，再评价同一留出三元组。选择规则：最大 MRR；数值平局阈值 1e-12 内优先 Global，然后较小强度、较大 τ。各家族不交叉选冠军。逻辑回归沿用 balanced/liblinear/C=1/max_iter=2000 的相同对象；无家族专属重拟合、学习率搜索或早停。记录收敛迭代数。

完整 DEV 使用原已锁模型与 anchor，并按同样的每族 41 点规则选择新动作参数，明确是 full-fit 选择分数，不冒充留出性能。四组共24个最终家族锁写入独立文件、绑定协议/代码/模型/DEV输入哈希后，另一个 apply-test 阶段才读 TEST；不覆盖旧锁。允许纯 Global 赢得任意家族选择，不保证留出或 TEST 不受损。

结果报告全部家族在 DEV OOF 与冻结 TEST 的 MRR、相对同一 Global 的净效用、H、条件损失、无条件平均损失、权重改变率与偏移，保留全部 seed/direction。对 ADC+Global 与其余五族逐一报告配对差异，不事后只挑最好或最差基线。新比较为给定拟合对象的描述性点估计，不新增昂贵 bootstrap，不将小差异写为统计显著的 tanh 优势。

## 静态对照与预算披露

完整静态表覆盖全部六个已重评配对：Primary、Secondary、Equal z-score、Equal RRF(k=60)、Global、Relation；每种已宣称比较的方法在 PDF 有数值。Oracle 单独作为 `max(RR_primary, RR_secondary)` 的答案知情专家选择诊断报告，它只在两个专家的选择集合内逐观测取最大，既不可部署，也不是混合分数的最佳可实现上界。Relation 使用原DEV锁、support=60，不重新按 TEST 调整。

预算表列示固定静态方法0搜索、Global21权重、Relation每个够支持关系21权重、Query-soft1配置、原ADC40、新六族各41，以及Dyna R0/R1/R2/R3-fixed固定预算、R3的3学习率×4checkpoint、R4继承R3不另搜。Dyna输入/学习器/规范化不同，不能作为单一映射变化的证据。

最低关闭条件：共享对象证据与回归测试通过；新增简单收缩及线性/clip数值完整披露；Global实际参与DEV选择；全部静态/Oracle数值及预算进入PDF；明确ADC能否超出简单替代方案，不预设结论。
