# 实验执行清单

## 1. 目标

这份清单用于把当前论文计划落到实验执行层，目标不是继续讨论方向，而是明确：

- 先做什么
- 每一步产出什么
- 哪些结果足以支持论文主线调整

当前默认论文主线为：

- 分析驱动型为主
- 方法改进型为辅

## 2. 第一阶段：协议与口径统一

### 2.1 主模型固定

- [x] 确认并固定五组主模型：
  - Text-only
  - Early Fusion
  - Gate-only
  - Residual-only
  - Full Model
- [x] 确认每组模型配置文件路径
- [x] 确认每组模型都能正常构建并启动训练

当前固定的主模型映射如下：

| 论文名称 | 配置文件 | `model.name` | 实际构建类 | 当前状态 |
|---|---|---|---|---|
| Text-only | `ml/configs/openbg_img_text_only.yaml` | `openbg_img_text_only` | `OpenBGImgTextOnlyLP` | 已验证可构建 |
| Early Fusion | `ml/configs/openbg_img_early.yaml` | `openbg_img_early` | `OpenBGImgEarlyLP` | 已验证可构建 |
| Gate-only | `ml/configs/openbg_img_gate_only.yaml` | `openbg_img_gate_only` | `OpenBGImgGateOnlyLP` | 已验证可构建 |
| Residual-only | `ml/configs/openbg_img_residual_only.yaml` | `openbg_img_residual_only` | `OpenBGImgResidualOnlyLP` | 已验证可构建 |
| Full Model | `ml/configs/openbg_img_gated_vec_res_rel.yaml` | `openbg_img_gate_residual` | `OpenBGImgGateResidualLP` | 已验证可构建 |

补充说明：

- `TextRGCN` 已从当前主实验路线中移除，不再作为 `Text-only` 入口。
- 当前 `Text-only` 已纳入与其余四组相同的 OpenBG-IMG 主模型家族，以保证模型口径一致。
- 五组模型均已验证能够在 raw cache 模式下正确构建。

### 2.2 强基线固定

- [ ] 明确保留的结构强基线：
  - ComplEx
  - TuckER
  - RSME
- [ ] 整理这些基线已有结果的来源与口径
- [ ] 标记哪些结果可以直接用于论文，哪些只能暂作参考

### 2.3 评测协议统一

- [x] 明确 train/dev/test 的统一流程
- [x] 明确 dev 仅用于 early stopping 和 model selection
- [x] 明确最终统一在 test 上汇报
- [x] 明确使用 filtered ranking 指标：
  - MRR
  - Hits@1
  - Hits@3
  - Hits@10
- [ ] 明确每组至少跑 3 个 seeds

当前已完成的协议改造：

- `run_train.py` 已统一读取 `train/dev/test`
- 训练时仅使用 `dev` 做 early stopping 和 checkpoint 选择
- 训练结束后会自动加载 `best.ckpt` 并在 `test` 上评测
- `test` 指标会保存为 `test_metrics.json`
- `filtered_ranking_eval` 已支持 `tail / head / both`，当前默认协议为 `both`
- `common.yaml` 和 `common_smoke.yaml` 已显式固定 `evaluation.direction: both`

当前尚未完成的唯一项：

- 五组主模型的正式主实验还没有按统一协议补齐 `3 seeds`

### 2.4 输入流程确认

- [ ] 确认 raw cache 已稳定启用
- [ ] 确认 `text_proj` / `img_proj` 均为可训练层
- [ ] 确认旧缓存只作为兼容，不再作为主流程证据

## 3. 第二阶段：主结果实验

### 3.1 五组主模型重跑

- [ ] 跑 `Text-only` 3 seeds
- [ ] 跑 `Early Fusion` 3 seeds
- [ ] 跑 `Gate-only` 3 seeds
- [ ] 跑 `Residual-only` 3 seeds
- [ ] 跑 `Full Model` 3 seeds

### 3.2 结果汇总

- [ ] 汇总每组模型的 test MRR / Hits@1 / Hits@3 / Hits@10
- [ ] 汇总每组模型的 mean ± std
- [ ] 整理主结果表
- [ ] 对比结构强基线结果

### 3.3 初步判断

- [ ] 判断 `Full Model` 是否稳定优于 `Gate-only`
- [ ] 判断 `Full Model` 是否稳定优于 `Residual-only`
- [ ] 判断 `Full Model` 与强结构基线差距是否缩小
- [ ] 记录主结果对论文主线的影响

## 4. 第三阶段：Residual dominance 排查

这一阶段直接对应 `FULL_MODEL_DIAGNOSIS_TODO.md` 中的 `12.1`。

### 4.1 观测量补齐

- [ ] 记录 residual 相关梯度
- [ ] 记录 fusion 相关梯度
- [ ] 记录 projection 相关梯度
- [ ] 记录 `residual_scale`
- [ ] 记录 mix 权重

### 4.2 干预实验

- [ ] 跑 delayed-residual training
- [ ] 跑 weaker-residual training
- [ ] 跑更强 residual 正则
- [ ] 比较干预前后 `Full Model` 的变化

### 4.3 结果判断

- [ ] 判断 residual 是否长期主导训练
- [ ] 判断 `Full Model` 弱于 `Residual-only` 是否主要由 residual shortcut 导致
- [ ] 输出一版 residual dominance 诊断结论

## 5. 第四阶段：优化路径排查

### 5.1 两阶段训练

- [ ] 跑 fusion first / residual later 的两阶段训练
- [ ] 比较与默认训练的差异

### 5.2 超参数敏感性

- [ ] 扫 `batch_size`
- [ ] 扫 `early_stop_patience`
- [ ] 扫 `img_dropout`
- [ ] 扫 gate regularization

### 5.3 结果判断

- [ ] 判断 `Full Model` 是否对训练策略高度敏感
- [ ] 判断问题更像“结构不足”还是“优化不足”

## 6. 第五阶段：缺模态与关系分组实验

### 6.1 缺模态分组

- [ ] 建立 `has_img` / `no_img` 的 test 分组评测
- [ ] 比较五组模型在两类实体上的表现
- [ ] 判断多模态收益是否主要出现在有图实体

### 6.2 关系分组

- [ ] 粗略定义视觉相关关系
- [ ] 粗略定义视觉弱相关或抽象关系
- [ ] 比较五组模型在不同关系组上的表现
- [ ] 判断多模态收益是否具有明显关系依赖性

### 6.3 结果判断

- [ ] 判断多模态收益是否是“局部有效”
- [ ] 判断论文是否可以建立“收益边界”核心结论

## 7. 第六阶段：行为分析

### 7.1 Gate 行为

- [ ] 统计 gate 均值与方差
- [ ] 按关系统计 gate 分布
- [ ] 判断 gate 是否真的随关系变化

### 7.2 Residual 行为

- [ ] 统计 `residual_scale`
- [ ] 分析 residual 在不同实体子集上的行为
- [ ] 判断 residual 是否在缺图实体上更强

### 7.3 Fusion 与 residual 竞争关系

- [ ] 结合 mix 权重分析 fusion / residual 的主导关系
- [ ] 判断 fusion 是否长期被 residual 压制
- [ ] 形成“互补还是竞争”的分析结论

## 8. 第七阶段：案例分析

### 8.1 成功案例

- [ ] 选取 `Full Model` 明显优于 `Residual-only` 的样本
- [ ] 分析此类样本是否依赖图像或文本线索

### 8.2 失败案例

- [ ] 选取 `Residual-only` 明显优于 `Full Model` 的样本
- [ ] 分析此类样本是否更依赖结构模式

### 8.3 案例结论

- [ ] 提炼哪些场景更适合多模态
- [ ] 提炼哪些场景更适合强结构表示

## 9. 第八阶段：论文主线最终判定

### 9.1 若出现以下结果

- `Full Model` 经规范训练后稳定优于 `Gate-only`
- 在部分关键子集上明显优于 `Residual-only`
- 行为分析支持 fusion 与 residual 具有互补性

则保留：

- “方法改进型为辅、分析驱动型为主”的写法

### 9.2 若出现以下结果

- `Full Model` 全局仍不如 `Residual-only`
- 但在 `has_img` 或某些关系组上存在稳定收益

则转向：

- “多模态收益边界分析”主线

### 9.3 若出现以下结果

- `Residual-only` 在几乎所有层面都持续占优
- 多模态局部收益也不明显

则进一步转向：

- “为什么该数据集上强结构表示天然占优”的分析型论文

## 10. 写作前必须完成的最小清单

- [ ] 五组主模型都有规范 test 结果
- [ ] 主结果表完成
- [ ] `has_img` / `no_img` 分组结果完成
- [ ] relation type 分组结果完成
- [ ] gate / residual 行为分析完成
- [ ] 至少 2 组成功案例与 2 组失败案例完成
- [ ] 论文主线最终定稿
