# Gate / Residual 模型拆分重构方案

## 1. 文档目的

当前项目中的 `OpenBGImgGatedLP` 通过配置项：

- `use_fusion`
- `use_residual`
- `use_normalized_mix`

在同一个模型实现中切换出多种行为。  
这种做法在工程早期便于快速实验，但在当前论文推进阶段已经暴露出明显问题：

- baseline 语义不够独立
- 模型命名与实际作用路径容易混淆
- `residual-only` 等模型在论文叙事中不够清晰
- 后续实验对照关系不够直观

因此，本方案建议：

**将 `gate-only`、`residual-only`、`gate+residual` 拆分为三个明确模型类，但保留共享基类，避免复制三份几乎相同的大文件。**

本文档将说明这一重构的目标、结构设计与逐步执行 Todo。

## 2. 重构目标

这次重构的核心目标不是“为了代码更好看”，而是为了让模型定义、实验对照和论文叙事严格对应。

具体目标如下：

1. 让 `gate-only`、`residual-only`、`gate+residual` 成为语义清晰的独立模型
2. 保证三类模型共享尽可能多的底层实现，避免重复代码
3. 保证实验对照公平，只改变目标模块，不引入额外实现差异
4. 为后续论文写作提供更清晰的模型命名和方法图结构

## 3. 当前实现存在的问题

### 3.1 配置开关掩盖了模型语义

当前通过 `use_fusion/use_residual` 切换行为，会导致：

- 类名仍叫 `OpenBGImgGatedLP`
- 但实际可能根本不走 gate 分支
- `residual-only` 从代码入口看不够独立

这在论文对外表述时不够清晰。

### 3.2 baseline 容易被误解

例如当前 `residual-only`：

- 仍然会实例化文本和图像投影层
- 但最终打分表示不真正依赖 fusion 分支

这种设计在代码上可运行，但在研究口径上容易让人误解它是不是“某种弱化版多模态模型”。

### 3.3 后续继续扩展会越来越乱

如果未来还想继续加：

- 普通 gate
- scalar gate
- concat + MLP

继续依赖大量布尔开关，会让一个文件里堆积越来越多的条件分支，不利于维护，也不利于实验解释。

## 4. 建议的重构方案

### 4.1 总体原则

不是写三份完全独立的大文件，而是采用：

- 一个共享基类
- 三个明确子类

### 4.2 建议的类结构

建议引入如下结构：

#### 共享基类

- `BaseOpenBGImgLP`

负责统一处理：

- 原始缓存特征读取后的投影
- `has_img` / `v_missing`
- 图像 dropout
- `ComplEx` 解码器
- loss 计算
- 训练与推理公共接口

#### 子类一：Gate-Only

- `OpenBGImgGateOnlyLP`

职责：

- 只保留文本/图像融合路径
- 不包含实体 residual 分支
- 直接测试“关系感知融合”能力

#### 子类二：Residual-Only

- `OpenBGImgResidualOnlyLP`

职责：

- 只保留实体 residual 分支
- 不再混入 gate 路径
- 明确作为强结构补偿 baseline

#### 子类三：Gate + Residual

- `OpenBGImgGateResidualLP`

职责：

- 保留 gate 路径
- 保留 residual 分支
- 支持直接相加或 normalized mix
- 作为当前主模型

## 5. 共享基类建议包含的内容

### 5.1 输入与缓存相关

- `text_feat`
- `img_feat`
- `has_img`
- `text_proj`
- `img_proj`
- `v_missing`
- 图像 dropout 逻辑

### 5.2 训练与打分相关

- `decoder`
- `score`
- `score_eval`
- `self_adversarial_loss`
- `forward`

### 5.3 可被子类复用的辅助函数

- `_entity_text`
- `_entity_image`
- `_build_entity_representation(...)` 的统一调用接口

其中“如何构造实体表示”应当由子类覆盖。

## 6. 三个子类的职责边界

### 6.1 Gate-Only

实体表示仅由：

- 文本投影
- 图像投影
- gate 融合

构成。

不包含：

- residual 分支
- normalized mix

### 6.2 Residual-Only

实体表示仅由：

- residual embedding

构成。

这里需要强调：

- 它可以继续继承基类中的缓存接口和公共训练接口
- 但其最终表示构造不依赖 gate 输出
- 如果文本/图像投影层在该模型中完全不参与最终表示，则应考虑是否继续实例化，或明确标记为不使用

建议采用更清晰的方式：

- 在 `ResidualOnly` 中不实例化 fusion 模块
- 不依赖 gate 相关统计接口

### 6.3 Gate + Residual

实体表示由：

- gate 融合结果
- residual 补偿

共同构成。

支持：

- 直接加和
- normalized mix

## 7. `residual-only` 是否保留投影层

这是当前最需要明确的一个设计点。

### 7.1 建议结论

建议：

- `ResidualOnly` 不应在最终表示路径中依赖文本和图像投影层

原因是：

- 它的研究意义就是“只保留实体级补偿能力”
- 如果仍引入模态投影并参与最终表示，它就不再是真正的 residual-only

### 7.2 工程实现建议

可以允许它继承基类，但：

- 不调用文本/图像融合路径
- 不使用 gate 模块
- 不输出 gate 统计

这样可以保证语义最清晰。

## 8. `gate+residual` 作为主模型的建议口径

重构完成后，建议将当前主模型明确命名为：

- `GateResidual`
- 或 `RelationAwareGateResidual`

这样比当前“一个 gated 模型通过开关切不同模式”更适合论文表述。

## 9. 配置层重构建议

当前配置依赖：

- `model.name = openbg_img_gated`
- 再配合 `use_fusion/use_residual`

重构后建议改为直接使用明确的模型名：

- `model.name = openbg_img_gate_only`
- `model.name = openbg_img_residual_only`
- `model.name = openbg_img_gate_residual`

好处是：

- 配置语义更直接
- `build_model.py` 更清晰
- 结果目录与模型定义更一致

## 10. 迁移策略

为了避免一次性改太多，建议分阶段重构。

### 第一阶段：先抽公共基类

目标：

- 不改变外部行为
- 先把公共逻辑收拢

### 第二阶段：引入三个明确子类

目标：

- 保证当前实验逻辑可复现
- 让三种模型路径清晰分离

### 第三阶段：修改配置入口

目标：

- `build_model.py` 改为按明确模型名分发
- 不再依赖 `use_fusion/use_residual` 做主控制

### 第四阶段：清理旧接口

目标：

- 删除不再需要的兼容开关
- 更新文档、结果索引和 baseline 定义

## 11. 风险与注意事项

### 11.1 风险一：不小心破坏现有实验可比性

应对：

- 每次重构后先做最小前向验证
- 再做单 seed smoke test

### 11.2 风险二：三类模型共享逻辑不干净

应对：

- 明确哪些逻辑在基类
- 明确哪些逻辑只能由子类负责

### 11.3 风险三：重构后结果和旧版不一致

应对：

- 这是预期现象之一
- 关键是确认变化来自语义更清晰的模型定义，而不是实现 bug

## 12. 分步 Todo

### 12.1 设计阶段

- [x] 明确三类模型的标准定义
- [x] 明确 `ResidualOnly` 是否完全不依赖模态路径
- [x] 明确 `GateResidual` 是否保留 normalized mix
- [x] 明确新的模型命名规范

### 12.2 代码重构阶段

- [x] 新建 `BaseOpenBGImgLP`
- [x] 抽取公共输入处理逻辑
- [x] 抽取公共 loss 和 decoder 逻辑
- [x] 新建 `OpenBGImgGateOnlyLP`
- [x] 新建 `OpenBGImgResidualOnlyLP`
- [x] 新建 `OpenBGImgGateResidualLP`
- [x] 保证三类模型都能独立前向

### 12.3 构建入口阶段

- [x] 修改 `build_model.py`
- [x] 让 `build_model.py` 支持三个明确模型名
- [x] 保留短期兼容逻辑
- [x] 补充必要的日志输出

### 12.4 配置重构阶段

- [x] 新建 gate-only 配置
- [x] 新建 residual-only 配置
- [x] 新建 gate+residual 配置
- [x] 逐步废弃旧的 `use_fusion/use_residual` 主控方式

### 12.5 验证阶段

- [x] 分别检查三类模型是否真正走到了预期路径
- [x] 检查 gate-only 是否不含 residual
- [x] 检查 residual-only 是否不依赖 gate
- [x] 检查 gate+residual 是否同时包含两者
- [x] 做最小训练 smoke test

### 12.6 文档同步阶段

- [x] 更新 `VALID_BASELINES.md`
- [x] 更新 `RESULT_INDEX.md` 的模型口径
- [x] 更新 `PAPER_PLAN.md` 中的方法描述
- [x] 更新导师汇报文档中的模型定义

## 13. 当前建议

当前这个重构是值得做的，因为它解决的不是“代码风格问题”，而是：

- baseline 定义不清
- 论文叙事不稳
- 后续实验对照不够直观

但重构时必须坚持一个原则：

**三类模型在研究语义上要分开，在工程实现上要尽量共用。**

这样既能让论文表达清晰，又不会因为复制代码而引入新的实现偏差。

## 14. 12.1 设计阶段结论

本节给出后续代码重构必须遵守的正式定义。后续实现若与此不一致，应优先修改实现，而不是再改研究口径。

### 14.1 三类模型的标准定义

#### 1. Gate-Only

标准定义：

- 输入：文本原始特征、图像原始特征、关系 id
- 路径：文本/图像投影 -> 模态适配 -> 关系感知 gate 融合
- 输出表示：仅由 gate 融合结果构成

必须满足：

- 不包含实体 residual 分支
- 不使用 normalized mix
- 允许使用 `v_missing`
- 允许使用 image dropout

论文口径：

- 用于测试关系感知多模态融合本身的能力

#### 2. Residual-Only

标准定义：

- 输入：实体 id
- 路径：实体 residual embedding -> residual scale
- 输出表示：仅由 residual 分支构成

必须满足：

- 不依赖文本投影输出
- 不依赖图像投影输出
- 不依赖 gate 模块
- 不输出 gate 相关统计

论文口径：

- 用于测试实体级结构补偿/参数化表示本身的能力
- 应明确视为强结构 baseline，而不是多模态模型

#### 3. Gate + Residual

标准定义：

- 输入：文本原始特征、图像原始特征、关系 id、实体 id
- 路径：文本/图像投影 -> gate 融合 + residual 分支
- 输出表示：由 gate 分支与 residual 分支联合构成

必须满足：

- 同时保留 gate 与 residual 两条路径
- 允许直接加和
- 允许使用 normalized mix
- 允许使用 `v_missing`
- 允许使用 image dropout

论文口径：

- 作为当前主模型
- 用于测试融合机制与实体补偿机制是否具有互补性

### 14.2 关于 `ResidualOnly` 的正式结论

正式结论如下：

- `ResidualOnly` 不应在最终表示路径中依赖文本和图像模态特征
- 即使工程实现上继承共享基类，也不得让文本/图像投影结果进入最终实体表示
- 它不是“无 gate 的多模态模型”，而是“无模态融合的实体补偿 baseline”

因此，后续实现中若 `ResidualOnly` 仍实例化 `text_proj` / `img_proj`，这些层也必须处于无效状态，不参与最终表示计算。

### 14.3 关于 `GateResidual` 与 normalized mix 的正式结论

正式结论如下：

- `normalized mix` 只属于 `GateResidual`
- `GateOnly` 不应包含 normalized mix
- `ResidualOnly` 不应包含 normalized mix

原因是：

- normalized mix 的意义是调节 gate 分支与 residual 分支的联合贡献
- 若模型本身只保留一条分支，则该机制失去定义意义

因此，后续实现中：

- `GateResidual` 允许配置 `use_normalized_mix`
- 其他模型不应暴露该开关

### 14.4 新的模型命名规范

后续建议统一采用以下命名：

#### Python 类名

- `BaseOpenBGImgLP`
- `OpenBGImgGateOnlyLP`
- `OpenBGImgResidualOnlyLP`
- `OpenBGImgGateResidualLP`

#### 配置中的 `model.name`

- `openbg_img_gate_only`
- `openbg_img_residual_only`
- `openbg_img_gate_residual`

#### 结果表与论文中的模型口径

- `Gate-Only`
- `Residual-Only`
- `Gate+Residual`

### 14.5 本阶段输出

12.1 阶段完成后，后续重构必须以本节定义为准。下一步直接进入：

- 抽取 `BaseOpenBGImgLP`
- 为三类模型建立独立子类
- 修改 `build_model.py` 与配置入口
