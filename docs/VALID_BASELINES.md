# 当前有效 Baseline 清单

## 1. 说明

本清单用于回答一个更具体的问题：在当前仓库和当前阶段，哪些 baseline 已经具备进入后续实验比较和论文分析的基础，哪些还不具备。

这里的“有效”不是指“性能强”，而是指：

- 定义相对清晰
- 结果文件完整或基本完整
- 当前至少具备被拿来对比的最低条件

## 2. 当前有效 baseline

### 2.1 Gate-Only

定义：

- 模型名为 `openbg_img_gate_only`
- 实现类为 `OpenBGImgGateOnlyLP`
- 仅保留关系感知 gate 融合路径
- 不包含 residual 分支

当前状态：

- 已有 3 个 seed
- metrics 文件完整
- 当前是较完整的对照组

对应 run：

- [seed1](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gate_only/20260308_180434_seed1)
- [seed2](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gate_only/20260308_201856_seed2)
- [seed3](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gate_only/20260309_001045_seed3)

建议标签：

- 当前有效 baseline

### 2.2 Full Model

定义：

- 模型名为 `openbg_img_gate_residual`
- 实现类为 `OpenBGImgGateResidualLP`
- 同时保留 gate 融合与 residual 分支
- 当前配置中启用 `use_normalized_mix`
- 当前目录仍为 `openbg_img_gated_vec_res_rel`

当前状态：

- 已有 5 个 seed
- metrics 文件完整
- 当前是最完整的主模型组

对应 run：

- [seed1](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260314_212911_seed1)
- [seed2](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260314_223034_seed2)
- [seed3](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260314_231558_seed3)
- [seed4](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260315_000727_seed4)
- [seed5](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260315_005115_seed5)

建议标签：

- 当前有效主模型

## 3. 当前部分有效 baseline

### 3.1 Residual-Only

定义：

- 模型名为 `openbg_img_residual_only`
- 实现类为 `OpenBGImgResidualOnlyLP`
- 仅保留 residual 分支
- 最终表示不依赖 gate 路径
- 当前目录为 `openbg_img_residual_only`

当前状态：

- 当前只有 1 个 seed
- 结果完整，但不够稳定
- 可用来观察现象，不足以作为正式多 seed baseline
- 已与 gate-only 在代码实现上明确分离

对应 run：

- [seed1](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_residual_only/20260313_161055_seed1)

建议标签：

- 部分有效 baseline
- 重点复核对象

## 4. 当前无有效结果的 baseline

以下 baseline 在概念上应该存在，但当前仓库里还没有形成可直接纳入比较的统一结果。

### 4.1 Text-Only

当前状态：

- 尚未建立统一结果索引
- 尚未形成可直接比较的当前版本结果组

建议动作：

- 尽快纳入最小关键实验组

### 4.2 Early Fusion

当前状态：

- 尚未建立统一结果索引
- 尚未形成可直接比较的当前版本结果组

建议动作：

- 尽快纳入最小关键实验组

## 5. 当前 baseline 使用建议

### 5.1 可以直接用于后续对照的

- `gate-only`
- `full model`

### 5.2 可以用于提出问题、但不能直接形成稳定结论的

- `residual-only`

### 5.3 当前不宜直接纳入正式主表的

- `text-only`
- `early fusion`

原因不是这些 baseline 不重要，而是当前仓库里还没有与现阶段口径一致、结果完整的版本。

## 6. 当前最关键的 baseline 缺口

- `text-only`
- `early fusion`
- `residual-only` 多 seed

如果要进入下一阶段正式比较，这三类缺口应优先补齐。

### 6.1 缺口优先级说明

#### 第一优先级

- `text-only`
  - 这是后续判断多模态是否真正带来收益的最低对照组
  - 如果缺少这一组，很多结论都无法成立

- `early fusion`
  - 这是判断当前 relation-aware gated fusion 是否真的优于简单融合的关键对照
  - 如果缺少这一组，就无法区分“多模态有效”与“当前融合设计有效”

#### 第二优先级

- `residual-only` 多 seed
  - 当前单个 seed 结果异常强
  - 必须补齐多 seed 才能判断这是否是稳定现象

### 6.2 当前结论

当前最关键的对照组缺口已经可以明确归纳为：

1. `text-only`
2. `early fusion`
3. `residual-only` 多 seed

阶段 0 到此可以视为完成，后续可正式进入阶段 1 和阶段 2。

## 7. 下一步建议

- [ ] 将 `text-only` 纳入统一索引体系
- [ ] 将 `early fusion` 纳入统一索引体系
- [ ] 为 `residual-only` 补足多 seed
- [ ] 在补齐后更新主结果表和 baseline 清单
