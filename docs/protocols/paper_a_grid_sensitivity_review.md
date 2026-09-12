# B09：固定偏好与参数的网格敏感性协议

在新增细网格／连续权重评分前固定本协议。已有历史 TEST 已可见，本轮为后验敏感性补充，不恢复独立确认性。当前 B09 待服务器结果，不能预先宣称结论不依赖量化。

## 对象与预先固定的配置

四主 pair W-N/W-A/D-N/D-A，三个 base seed，head/tail 全实体 filtered 排名。每个观察使用同一个冻结偏好信号、锚点与参数，仅改变最后的投影。

- 三种执行方式：原 0.05 网格、0.01 网格、连续权重（取消投影；仍保留 gate、tanh 和 [0,1] 裁剪）。
- 两种动态映射：ADC、同一拟合对象的 Query-soft。共六个动态配置，每个 pair 均完整报告，不筛选 TEST 胜者。
- Global 使用原锚点，另重算 Primary／Secondary 作为评分复现检查。两网格均包含全部原锚点，fallback 不会被移到另一个静态权重。
- 本轮不更换部署网格，不按 TEST 选择网格、pair、seed 或参数。没有按网格重新调 β、τ、α0；结果解释为“给定原拟合与 DEV 选定参数的量化敏感性”，不声称连续／细网格已经最优调参。

## DEV 与 TEST 分离

1. `prepare` 只读取既有 DEV 逐观察结果和参数，以及已有 checkpoint 账本／配置元信息。按原三元组五折（seed 20260901）重建 20 个小型 13D logistic 模型：相同标签／tie 排除、median/scaler、balanced liblinear、C=1、max_iter=2000、fold random state=20260902+10*k+2。每折只使用训练部分拟合，沿用既有该折 α0、β、τ；必须复现原 OOF ADC／Query-soft 的 0.05 权重或 RR 与 fallback。保存四份 DEV OOF 动作计划及全部 full-DEV 参数锁；不读取 TEST 逐观察结果，账本中的历史 TEST 汇总不参与拟合或配置决策。
2. `score-dev` 在服务器对完整 DEV 评分。只比较预先固定的三种执行方式；不选新参数。DEV query 属于原 OOF 的留出部分。冻结评分器本身未按 selector 外折重训，与原协议相同。
3. `score-test` 要求全部 DEV 评分 cell 和完成记录已经存在且哈希通过，再读取四组既有 TEST 的完整拟合信号。执行同样六个事先固定的动态配置，不根据 DEV 或 TEST 缩减配置集合。
4. `report` 统一核查所有结果并输出 paired 效应；不产生“获胜网格”或覆盖旧模型。未来若另行选择新部署网格，只能用 DEV 协议并另行冻结，不依据本轮 TEST 结果挑选。

## 评分信息边界和数值复现

继续使用原修正评估路径：原始未过滤分数计算 population z-score，gold 单独用候选矩转换，随后对评估拷贝屏蔽其他已知事实，保留当前 gold；端点使用对应单模型原始 rank。两阶段评价均继承历史 ADC 的 TRAIN∪DEV∪TEST 事实过滤范围，因此服务器 DEV loader 可以读取 TEST 事实用于标准 filtered evaluation，不能称为完全不打开 TEST 文件。特征与冻结动作计划不接收 gold 分数或事实索引；prepare 不读取 TEST 结果。

评分使用原两个模型的 query batch 参数，outer batch=max(两模型 batch)，按 canonical split 顺序；完整候选分数只保留于内存 batch，不导出大型 score cache。所有候选／gold／标准化矩必须有限，否则停止并写错误记录。加载数据的 split／mapping／feature 哈希必须与 checkpoint 内嵌配置记录一致；代码、计划、逐 cell 输入与输出均绑定哈希。连续权重指取消网格投影，最终混合沿用原排名函数的 float32 权重转换，不代表无限精度。

原 0.05、Global、Primary、Secondary 的新评分与保存 RR 逐行比较。若发生数值重执行差异，完整保留差异计数和效应，不静默替换旧结果；最终状态转为需复核。所有网格比较始终使用本次同一候选评分，避免把旧／新评分差异混进网格效应。保留多 gold 可观测 query 的网格权重一致性与连续权重最大跨度；不通过按 gold 平均来修复差异。连续权重受既有浮点可重复性局限约束。

## 指标、成对比较与预算

DEV OOF 和 TEST 各报告 MRR、相对同一次 Global 的净效用、无条件 RR loss、harm frequency、conditional loss、权重改变率、相对连续动作的量化误差与死区抑制比例。保留 pair×seed×方向、完整逐观察 RR／权重，所有策略在同一原三元组上配对。

预先固定比较：每种执行方式 ADC−Global、Query-soft−Global、ADC−Query-soft；每种动态方法 fine−0.05、continuous−0.05。所有点估计报告，无新显著性口号；本轮不新增 bootstrap，原 0.05 的历史区间不移植到新网格。若收益符号或 ADC 对 Query-soft 的排序改变，必须披露，不能预设量化只是计算细节。

网格有行动正则化作用：包含锚点的等间距网格在锚点附近产生半格宽死区，tie 朝锚点；小提议可退回 Global。0.05／0.01 分别对应约 0.025／0.005 的半宽（实际遵循浮点 lexicographic tie 规则）。一般投影位移不超过半格，但不能据此界定 RR 损失。已有 21 点 reciprocal ranks 不能插值得到细网格或连续权重排名。

仅服务器执行基础模型评分。`prepare` 可使用单 CPU 线程重建小型 selector；无基础模型训练。每个 scoring cell 独立落盘并记录计划／代码／结果哈希，可以校验后续跑；禁止用不同计划覆盖已完成 cell。回传包不含候选分数或模型，只含计划、检查、逐观察权重与 RR、汇总和成对效应。B09 需在回传结果审阅后才可关闭。
