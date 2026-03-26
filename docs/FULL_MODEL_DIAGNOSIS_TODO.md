# Full Model 诊断 Todo

## 1. 目的

这份文档要回答一个具体问题：

> 在已经切换到 raw mode、并且已经把投影层移入模型内部之后，如果 `Gate+Residual` 仍然弱于 `Residual-Only`，还可能有哪些原因，应该如何排查？

当前默认前提是：

- raw cache 模式已经启用
- `text_proj` 和 `img_proj` 都是可训练的
- `Gate-Only`、`Residual-Only`、`Gate+Residual` 已经在模型定义层面显式拆分

因此，之前那个解释：

- 冻结的随机投影缓存让多模态分支天然吃亏

已经不再是当前的主要假设。

接下来的诊断重点应转向：

- 模型行为
- 优化过程
- 数据本身的模态性质
- 评测协议与结果展示方式

## 2. 当前判断

如果 `Gate+Residual` 仍然打不过 `Residual-Only`，那么最可能的原因已经不再是输入流程问题，更可能是：

1. `Residual-Only` 在这个数据集上本来就是一个很强的 baseline。
2. 多模态特征虽然存在，但与 KG 补全任务的对齐程度不足。
3. 融合分支没有学到真正的互补性。
4. 优化路径天然偏向 residual 这条捷径。
5. 当前评测口径或结果粒度掩盖了多模态分支真正有效的局部场景。
6. 数据集本身并不支持一个很强的“视觉模态全面提升性能”的命题。

这些可能性并不互斥。

## 3. 假设 A：Residual-Only 天生就是一个强 baseline

### 3.1 为什么合理

`Residual-Only` 不是一个弱消融。它本质上相当于：

- 一个针对每个实体的可训练 embedding 表
- 再加上一个 `ComplEx` 解码器

在封闭实体集的链接预测任务里，这样的结构本身就可能非常强。如果图结构已经足够有信息量，那么多模态特征平均下来提供的增益可能很有限。

### 3.2 典型现象

- `Residual-Only` 收敛又快又稳。
- `Gate+Residual` 的有效能力主要来自 residual 分支。
- 削弱多模态分支，最终结果变化不大。
- 削弱 residual 分支，结果变化很明显。

### 3.3 要检查什么

- 比较以下参数组的梯度量级：
  - residual 参数
  - 文本/图像投影层
  - fusion 参数
- 如果启用了 normalized mix，记录 fusion 与 residual 的实际混合权重
- 比较以下模型的学习曲线：
  - `Residual-Only`
  - `Gate-Only`
  - `Gate+Residual`

### 3.4 针对性实验

1. 降低 residual 分支容量
2. 增强 residual 参数的正则化
3. 前几个 epoch 冻结 residual
4. 让 `Gate+Residual` 使用更小的 residual scale 初值

### 3.5 判断标准

- 如果 residual 一旦被削弱或延迟启用，`Gate+Residual` 就明显变强，那么 residual dominance 就是主要原因。
- 如果削弱 residual 只会让整体更差，那说明这个任务本身可能高度依赖结构信息。

### 3.6 Todo

- [ ] 记录 residual、projection、fusion 三组参数的梯度范数
- [ ] 如果启用 mix，记录 normalized-mix 的统计信息
- [ ] 跑一组带 residual warmup delay 的 `Gate+Residual`
- [ ] 跑一组削弱 residual 能力的 `Gate+Residual`
- [ ] 与当前 `Residual-Only` 做直接比较

## 4. 假设 B：多模态特征存在，但与任务对齐较弱

### 4.1 为什么合理

即使文本和图像特征在一般意义上质量不错，它们也未必能直接帮助 KG 关系预测。

例如：

- 文本描述可能偏介绍性，而不是关系判别性
- 图像可能表达外观，但很多关系其实是抽象语义关系
- 真正受益于视觉信息的关系，可能只占很小一部分

### 4.2 典型现象

- 收益只出现在某些子集，而不是整体
- 有图实体上可能提升，但总平均没有提升
- 某些关系类型受益，另一些关系反而变差
- gate 长期偏向忽略某一个模态

### 4.3 要检查什么

- 按以下维度做分组评测：
  - 有图实体 vs 无图实体
  - 不同关系类型
  - 高频关系 vs 长尾关系
  - 视觉相关关系 vs 抽象关系
- 查看 gate 在不同关系上的分布
- 比较图像去除与文本去除的结果

### 4.4 针对性实验

1. 只在 `has_img` 实体子集上评测
2. 只在更可能依赖视觉线索的关系上评测
3. 比较：
   - 完整多模态
   - 去掉文本
   - 去掉图像

### 4.5 判断标准

- 如果多模态收益只存在于很窄的子集，那么论文主线应该转向“收益边界分析”，而不是“整体性能全面提升”。

### 4.6 Todo

- [ ] 建立 `has_img` vs `no_img` 的分组评测
- [ ] 建立按关系类型分组的评测
- [ ] 粗略定义一批视觉相关关系
- [ ] 跑 text-drop 和 image-drop 消融
- [ ] 判断收益是否只存在于窄场景

## 5. 假设 C：融合分支没有学到真正的互补性

### 5.1 为什么合理

relation-aware gate 只有在以下条件成立时才有意义：

- 不同模态确实提供了不同且有用的信息
- gate 真的学到了与关系相关的模态偏好
- 融合表示相比更简单的方案确实有额外收益

如果这些条件不成立，那么 gate 很可能只是在对噪声做加权。

### 5.2 典型现象

- `Gate-Only` 持续偏弱
- `Gate+Residual` 也无法超过 `Residual-Only`
- gate 值塌缩到长期偏向某一个模态
- 不同关系下的 gate 分布非常相似

### 5.3 要检查什么

- gate 的均值和方差
- gate 在不同关系上的分布
- relation-aware bias 是否真的影响了 gate
- 更简单的融合方式是否表现相近甚至更好

### 5.4 针对性实验

1. 用当前 relation-aware gate 对比：
   - scalar gate
   - concat + MLP
   - 固定平均
   - early fusion
2. 比较不同 run 和不同 seed 下的 gate 统计
3. 验证 relation-aware bias 是否真的被用到了

### 5.5 判断标准

- 如果更简单的融合方式就能达到同样甚至更好的结果，那么当前 gate 设计就缺乏充分性。
- 如果 gate 模式在关系之间几乎没有差异，那么“relation-aware”这个说法就比较弱。

### 5.6 Todo

- [ ] 训练过程中记录 gate 的均值和方差
- [ ] 记录按关系划分的 gate 分布
- [ ] 与 scalar gate baseline 比较
- [ ] 与 concat + MLP baseline 比较
- [ ] 与 early fusion 比较
- [ ] 判断 relation-aware 行为是否真实存在

## 6. 假设 D：优化路径天然偏向 residual 捷径

### 6.1 为什么合理

即使 `Gate+Residual` 在表达能力上足够强，训练过程也可能优先选择最容易收敛的路径。在多分支模型里，residual 分支很容易变成 shortcut。

特别是在以下情况下，这个问题更容易出现：

- residual 参数更容易拟合
- fusion 分支需要更多 epoch 才能变得有用
- early stopping 基于 dev 曲线，而 dev 曲线本身偏向 shortcut 学习

### 6.2 典型现象

- `Gate+Residual` 前期看起来有潜力，后面却逐渐塌缩成 residual 主导
- seed 方差较大
- 稍微改一点超参数，结果就明显波动
- gate 相关参数学习速度明显慢于 residual

### 6.3 要检查什么

- 完整训练曲线：
  - train loss
  - dev MRR
  - gate 统计
  - mix 统计
- 比较多个 seed
- 比较不同学习率策略下各参数组的行为

### 6.4 针对性实验

1. 两阶段训练：
   - 先训练 fusion
   - 后启用 residual
2. 设置分组学习率：
   - residual 分支
   - projection/fusion 分支
3. 调整：
   - `batch_size`
   - `early_stop_patience`
   - `img_dropout`
   - gate regularization

### 6.5 判断标准

- 如果优化策略一改，`Gate+Residual` 就能超过 `Residual-Only`，那主要问题就是训练动力学，而不是模型结构本身。

### 6.6 Todo

- [ ] 为 `Gate+Residual` 跑两阶段训练
- [ ] 尝试 residual 与 fusion path 分组学习率
- [ ] 扫 `batch_size`
- [ ] 扫 `early_stop_patience`
- [ ] 扫 `img_dropout`
- [ ] 扫 gate regularization 设置
- [ ] 比较优化改动前后的 seed 方差

## 7. 假设 E：评测粒度掩盖了有效行为

### 7.1 为什么合理

一个全局 MRR 数字，可能会掩盖多模态分支真正有帮助的局部场景。

可能出现的情况是：

- `Gate+Residual` 在整体上输了
- 但在有图实体或某些关系类型上是赢的

如果只报告聚合指标，模型看起来会“全面更差”，但实际上它可能揭示了一个有价值的科学现象。

### 7.2 典型现象

- dev 层面的排名与分组行为不一致
- 提升只出现在某些切片上
- 不同 seed 在整体排序上不一致，但在子集收益上比较一致

### 7.3 要检查什么

- 不只看全局指标，还要看分组指标
- 在统一协议下报告最终 `test`
- 检查 dev 选择与 test 结论是否一致

### 7.4 针对性实验

1. 同时报告全局指标和分组指标
2. 避免只凭一个 dev-best run 下结论
3. 对所有 baseline 使用相同 seed 数和相同报告标准

### 7.5 判断标准

- 如果 `Gate+Residual` 虽然整体输，但在真正有科学意义的子集上赢，那么这仍然可能构成论文贡献，只是论文叙事必须调整。

### 7.6 Todo

- [ ] 统一 `test` 报告标准
- [ ] 增加分组结果表
- [ ] 重新检查 dev 与 test 的一致性
- [ ] 确保各 baseline 的 seed 数可比

## 8. 假设 F：数据集本身不支持强“视觉提升”命题

### 8.1 为什么合理

这是最根本的一种可能性。

OpenBG-IMG 虽然包含图像，但这并不意味着：

- 图像就是当前任务中主要缺失的信息
- 图像对大多数关系都有帮助
- 多模态融合理应在平均意义上超过强结构 baseline

如果这个数据集本身是结构主导型，那么强结构 baseline 取胜并不奇怪。

### 8.2 典型现象

- `Residual-Only` 持续很强
- `Gate-Only` 持续很弱
- `Gate+Residual` 只有有限或不稳定的收益
- 视觉收益只出现在少数局部场景

### 8.3 要检查什么

- 人工检查一些关系和实体样本
- 判断图像是否真的与关系判别相关
- 比较视觉相关关系与抽象关系

### 8.4 判断标准

- 如果视觉有用性本身就很窄，那么论文主线就应该从：
  - “我们提出了一个更强的多模态 MMKGC 方法”
  转向：
  - “我们分析多模态信号何时有效、何时失效，以及 residual 补偿如何改变这一边界”

### 8.5 Todo

- [ ] 人工抽样检查若干关系与实体
- [ ] 标记视觉相关关系与抽象关系
- [ ] 判断视觉分支是否只在窄场景有效
- [ ] 重新评估论文是否应继续坚持方法改进型主线

## 9. 推荐排查顺序

最理性的顺序是：

1. 先检查 residual 是否主导了训练。
2. 再检查优化路径是否天然偏向 shortcut。
3. 再检查多模态收益是否只存在于特定子集。
4. 再检查当前 gate 设计是否真的必要。
5. 最后重新评估数据集是否支持当前论文命题。

之所以推荐这个顺序，是因为它能以较小实验成本换取较高可解释性。

## 10. 最小实验包

如果第一轮只能做一小组排查实验，建议先跑：

1. `Residual-Only` 多 seed
2. `Gate-Only` 多 seed
3. `Gate+Residual` 多 seed
4. 不带 normalized mix 的 `Gate+Residual`
5. 使用两阶段训练的 `Gate+Residual`

并同时配套以下分析：

- gate 统计
- mix 统计
- `has_img` vs `no_img` 分组评测
- 按关系类型分组评测

## 11. 最终决策逻辑

可以按下面规则判断后续论文方向：

- 如果 `Gate+Residual` 在优化与 residual 平衡修正后变强，那么继续走“方法改进型”主线。
- 如果 `Gate+Residual` 全局仍然较弱，但在明确子集上有稳定收益，那么转向“收益边界分析型”主线。
- 如果 `Residual-Only` 在各个层面都持续占优，那么更诚实的结论就是：这个数据集上真正起主导作用的是结构补偿，论文也应据此重新定位。

## 12. 执行清单

### 12.1 Residual dominance

- [ ] 记录 residual 相关梯度
- [ ] 记录 residual/fusion 混合行为
- [ ] 跑 delayed-residual training
- [ ] 跑 weaker-residual training

### 12.2 Optimization

- [ ] 跑两阶段训练
- [ ] 尝试分组学习率
- [ ] 扫 patience
- [ ] 扫 batch size
- [ ] 扫 dropout 与 gate regularization

### 12.3 Data and subgroup analysis

- [ ] 评测 `has_img` vs `no_img`
- [ ] 按关系组评测
- [ ] 建立视觉相关关系子集
- [ ] 跑 text-drop 与 image-drop 消融

### 12.4 Fusion mechanism

- [ ] 与 scalar gate 比较
- [ ] 与 concat + MLP 比较
- [ ] 与 early fusion 比较
- [ ] 检查 relation-aware gate 统计

### 12.5 Paper direction

- [ ] 重新评估当前方法改进型主线是否仍然站得住
- [ ] 如有必要，转向收益边界分析叙事
