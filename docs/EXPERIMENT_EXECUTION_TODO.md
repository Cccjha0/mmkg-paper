# 实验执行清单

## 1. 目标

这份清单用于把当前论文计划落实到实验执行层。当前默认主线为：
- 分析驱动型为主
- 方法改进型为辅

## 2. 第一阶段：协议与口径统一

### 2.1 主模型固定
- [x] 固定五组主模型
  - Text-only
  - Early Fusion
  - Gate-only
  - Residual-only
  - Full Model
- [x] 固定每组模型的配置文件和 `model.name`
- [x] 确认五组模型都能正常构建并启动训练

当前映射：

| 论文名称 | 配置文件 | `model.name` | 实际类 |
|---|---|---|---|
| Text-only | `ml/configs/openbg_img_text_only.yaml` | `openbg_img_text_only` | `OpenBGImgTextOnlyLP` |
| Early Fusion | `ml/configs/openbg_img_early.yaml` | `openbg_img_early` | `OpenBGImgEarlyLP` |
| Gate-only | `ml/configs/openbg_img_gate_only.yaml` | `openbg_img_gate_only` | `OpenBGImgGateOnlyLP` |
| Residual-only | `ml/configs/openbg_img_residual_only.yaml` | `openbg_img_residual_only` | `OpenBGImgResidualOnlyLP` |
| Full Model | `ml/configs/openbg_img_gated_vec_res_rel.yaml` | `openbg_img_gate_residual` | `OpenBGImgGateResidualLP` |

### 2.2 强基线固定
- [x] 固定结构强基线
  - ComplEx
  - TuckER
- [x] 接入当前统一训练与评测链路
- [x] 补跑正式结果

### 2.3 评测协议统一
- [x] 统一 train/dev/test 流程
- [x] dev 仅用于 early stopping 和 model selection
- [x] 最终统一在 test 上汇报
- [x] 使用 filtered ranking
  - MRR
  - Hits@1
  - Hits@3
  - Hits@10
- [x] 使用 `direction=both`
- [x] 每组至少 3 个 seeds

### 2.4 输入流程确认
- [x] raw cache 已稳定启用
- [x] `text_proj` / `img_proj` 为可训练层
- [x] 旧缓存只保留兼容用途，不再作为主证据

## 3. 第二阶段：主结果实验

### 3.1 五组主模型 3 seeds
- [x] Text-only
- [x] Early Fusion
- [x] Gate-only
- [x] Residual-only
- [x] Full Model

### 3.2 结果汇总
- [x] 汇总五组主模型 test 指标
- [x] 汇总 `mean ± std`
- [x] 整理主结果表
- [x] 补入结构强基线结果

### 3.3 初步判断
- [x] 判断 `Full Model` 是否稳定优于 `Gate-only`
- [x] 判断 `Full Model` 是否稳定优于 `Residual-only`
- [x] 判断 `Full Model` 与强结构基线差距是否缩小
- [x] 记录主结果对论文主线的影响

当前主结果排序：
1. Residual-only
2. ComplEx
3. Full Model
4. Gate-only
5. Early Fusion
6. Text-only
7. TuckER

## 4. 第三阶段：Residual dominance 排查

### 4.1 观测量补齐
- [x] 记录 residual 梯度
- [x] 记录 fusion 梯度
- [x] 记录 projection 梯度
- [x] 记录 `residual_scale_value`
- [x] 记录 `mix_w_fusion / mix_w_residual`

### 4.2 干预实验
- [x] 跑 delayed-residual training
- [x] 跑 weaker-residual training
- [x] 跑 stronger residual regularization
- [x] 比较干预前后 `Full Model` 的变化

已完成配置：
- `ml/configs/openbg_img_gated_vec_res_rel_reswarmup10.yaml`
- `ml/configs/openbg_img_gated_vec_res_rel_weakres.yaml`
- `ml/configs/openbg_img_gated_vec_res_rel_strongresreg.yaml`

### 4.3 结果判断
- [x] 判定 residual 长期主导最终表示混合
- [x] 判定 `Full Model` 弱于 `Residual-only` 不能仅归因于梯度级 residual shortcut
- [x] 输出 residual dominance 阶段诊断结论

当前阶段结论：
- residual 的主导性主要体现在最终表示混合层面，而不是梯度层面
- fusion 和 projection 一直在学习，但最终模型仍持续更依赖 residual
- delayed-residual、weaker-residual、strongresreg 都没有超过默认 `Full Model`
- 当前更合理的解释是：结构补偿路径本身更符合当前任务偏好

## 5. 第四阶段：优化路径排查

### 5.1 训练策略
- [ ] fusion first / residual later 的两阶段训练
- [ ] 分组学习率

### 5.2 超参数敏感性
- [ ] `batch_size`
- [ ] `early_stop_patience`
- [ ] `img_dropout`
- [ ] gate regularization

### 5.3 结果判断
- [ ] 判断问题更像“结构不足”还是“优化不足”

## 6. 第五阶段：缺模态与关系分组实验

### 6.1 `has_img / no_img`
- [x] 建立 `has_img / no_img` test 分组
- [x] 比较关键模型在子组上的表现
- [x] 判断多模态收益是否主要出现在有图实体

当前正式口径说明：
- 保留当前 `paper_split`，不再为 `has_img / no_img` 分析重新划分数据
- 已确认当前 `paper_split` 中：
  - head 目标实体约 70% 有图
  - tail 目标实体 100% 无图
- 因此 `6.1` 的正式分析方式为：
  - `head target has_img / no_img`
  - `tail target no_img`
- 论文中需要明确说明：当前划分下，图像可用性分组分析主要发生在 head 方向，tail 方向不存在 `has_img` 子组

当前 3-seed 结果支持的结论：
- 整体上仍是 `Residual-only > Full Model > Gate-only`
- 在 `head_has_img` 子组中，排序变为 `Gate-only > Full Model > Residual-only`
- 在 `head_no_img` 子组中，`Full Model > Gate-only > Residual-only`
- 因此多模态收益不是全局存在，而是具有目标位置与模态可用性边界

### 6.2 relation type
- [x] 粗略定义视觉相关关系
- [x] 粗略定义视觉弱相关或抽象关系
- [x] 比较各模型在不同关系组上的表现
- [x] 判断多模态收益是否具有关系依赖性

当前正式口径说明：
- 基于 `docs/relation_type_groups_draft.json` 将关系粗分为：
  - `visual_relations`
  - `weak_visual_relations`
  - `ambiguous_material_relations`
- 使用与主实验一致的 `paper_split`、`best.ckpt`、`test`、filtered ranking、`direction=both`
- 正式汇总文件为：
  - `docs/RELATION_TYPE_SUMMARY_ALL.md`
  - `docs/relation_type_summary_all.json`
  - `docs/RELATION_TYPE_SUMMARY_MIN20.md`
  - `docs/relation_type_summary_min20.json`
- 当前口径说明文档为：
  - `docs/RELATION_TYPE_ANALYSIS.md`

当前 3-seed 结果支持的结论：
- 在 7 模型 grouped result 中，三个关系组上的总体排序都保持为：`Residual-only > ComplEx > Full Model`
- 在聚焦 3 模型的 grouped result 中，三个关系组上的总体排序都保持为：`Residual-only > Full Model > Gate-only`
- 在 `visual_relations` 上，未观察到多模态模型形成组级优势；`Full Model` 仍明显低于 `Residual-only`
- 在 `weak_visual_relations` 上，`Full Model` 的组级 MRR 高于其在 `visual_relations` 上的表现，因此当前粗分组结果不支持“视觉关系天然更适合多模态”的强结论
- 在 `MIN20` relation-level 结果中：
  - `visual_relations` 上 `Full Model > Gate-only` 为 `22 / 24`
  - `weak_visual_relations` 上 `Full Model > Gate-only` 为 `15 / 18`
  - `ambiguous_material_relations` 上 `Full Model > Gate-only` 为 `9 / 9`
- 但在同一套 `MIN20` relation-level 结果中，`Full Model > Residual-only` 只出现在少数关系上：
  - `visual_relations` 为 `4 / 24`
  - `weak_visual_relations` 为 `4 / 18`
  - `ambiguous_material_relations` 为 `1 / 9`
- 因此 relation-type 分析更支持这样的判断：`Full Model` 相比 `Gate-only` 在多数中等以上支持度关系上具有稳定增益，但这种增益仍不足以在组级层面或大多数关系上超过 `Residual-only`
- `6.2` 当前更适合作为“收益边界”叙事中的限制性证据：多模态收益是局部、条件化且有边界的，而不是“视觉关系上多模态普遍占优”的直接证据

### 6.3 结果判断
- [x] 判断多模态收益是否是“局部有效”
- [x] 判断论文是否可以建立“收益边界”核心结论

当前正式判定：
- 综合 `docs/HAS_IMG_ANALYSIS.md`、`docs/RELATION_TYPE_ANALYSIS.md`、`docs/GAIN_BOUNDARY_JUDGMENT.md` 与 residual dominance 阶段诊断结果，当前证据已足以判定：多模态收益是局部、条件化、且有边界的
- `6.1` 说明收益受目标位置与模态可用性约束：最强局部收益出现在 `head_has_img`，但 overall 仍被 `tail_no_img` 上的结构优势主导
- `6.2` 说明收益受关系特征约束：`Full Model` 虽然在多数中等以上支持度关系上稳定优于 `Gate-only`，但仍很少超过 `Residual-only`
- residual dominance 诊断说明：`Residual-only` 的优势不能简单归因于训练 bug 或 residual shortcut，更合理的解释是当前任务本身更偏好结构补偿路径
- 因此当前论文主线可以正式收敛为：多模态收益边界分析，而不是“Full Model 全局优于结构基线”的方法改进叙事

## 7. 第六阶段：行为分析

### 7.1 Gate
- [x] 统计 gate 均值与方差
- [x] 按关系统计 gate 分布
- [x] 判断 gate 是否真的随关系变化

当前进展：
- 已基于 run 目录中的 `metrics_seed1.csv` 完成第一轮 gate 行为汇总，结果见：
  - `docs/BEHAVIOR_SUMMARY.md`
  - `docs/behavior_summary.json`
- 已进一步完成 relation-aware gate 汇总，结果见：
  - `docs/RELATION_AWARE_GATE_SUMMARY.md`
  - `docs/relation_aware_gate_summary.json`
- 当前汇总方式为：对 `Gate-only / Full Model / Residual-only` 的最新 3-seed 正式 run，取各自 `best dev epoch` 的训练期诊断统计
- 当前已确认：
  - `Gate-only` 与 `Full Model` 都具备稳定的 gate 统计项（`g_mean_all / g_std_all / g_mean_img / g_mean_noimg`）
  - 在 best epoch 上，`Gate-only` 的 `g_mean_all` 约为 `0.5346 ± 0.0107`，`Full Model` 的 `g_mean_all` 约为 `0.4288 ± 0.0058`
  - 两个模型都表现出 `g_mean_img < g_mean_noimg`，其中：
    - `Gate-only` 的 `img-noimg gap` 约为 `-0.1863`
    - `Full Model` 的 `img-noimg gap` 约为 `-0.1510`
  - relation-aware gate 结果表明：在真实 test `(entity, relation)` 对上，`Gate-only` 与 `Full Model` 的 gate 在三个 relation group 上都存在系统差异
  - 在 group-level 上，`Full Model` 的 gate mean 在 `visual / weak_visual / ambiguous` 三组中都低于 `Gate-only`
  - 两个模型在 relation-aware 统计下仍稳定表现为 `target_has_img mean < target_noimg mean`
- 已新增 `7.1` 的正式判断文档：
  - `docs/GATE_RELATION_JUDGMENT.md`
- 当前正式判断为：
  - gate 的确会随关系变化，这一点在 relation-group level 和 per-relation level 上都成立
  - 例如 `Full Model` 的 group-level gate mean 在三个关系组上分别为：
    - `visual_relations`: `0.4719 ± 0.0056`
    - `weak_visual_relations`: `0.4955 ± 0.0073`
    - `ambiguous_material_relations`: `0.4814 ± 0.0040`
  - 同时，同组内部不同关系也存在明显差异，例如 `visual_relations` 中 `rel_0009` (`裤长`) 与 `rel_0044` (`闭合方式`) 的 gate mean 就明显不同
  - 但这种 gate 变化幅度仍不足以单独解释 `6.2` 的主性能格局；更完整的解释仍然需要结合 residual 偏好、mix 权重以及缺图子集上的 residual 增强
  - 因此 `7.1` 的最终收口结论为：`gate is relation-aware, but not relation-dominant`

### 7.2 Residual
- [x] 统计 `residual_scale`
- [x] 分析 residual 在不同实体子集上的行为
- [x] 判断 residual 是否在缺图实体上更强

当前进展：
- 已基于同一批 `metrics_seed1.csv` 完成第一轮 residual / mix 行为汇总，结果同样记录在：
  - `docs/BEHAVIOR_SUMMARY.md`
  - `docs/behavior_summary.json`
- 已进一步完成 relation-aware residual 汇总，结果见：
  - `docs/RELATION_AWARE_RESIDUAL_SUMMARY.md`
  - `docs/relation_aware_residual_summary.json`
- 当前已确认：
  - `Full Model` 在 best epoch 的 `residual_scale_value` 约为 `0.5198 ± 0.1801`
  - `Residual-only` 在 best epoch 的 `residual_scale_value` 约为 `0.2585 ± 0.0034`
  - `Full Model` 在 best epoch 的混合权重大致为：
    - `mix_w_fusion ≈ 0.2049 ± 0.0402`
    - `mix_w_residual ≈ 0.7951 ± 0.0402`
  - 从 first eval 到 best epoch，`Full Model` 的 `residual_scale_value` 平均上升约 `0.3728`，同时 `mix_w_residual` 继续上升、`mix_w_fusion` 相应下降
- 这进一步支持了已有 residual dominance 判断：最终表示混合层面仍然明显偏向 residual 路径
- relation-aware residual 结果进一步表明：在真实 test `(entity, relation)` 对上，`Full Model` 的 residual 有效贡献在 `head_noimg` 上稳定高于 `head_has_img`
  - `visual_relations` 的 head-side `effective residual norm`：`0.7462 > 0.6776`
  - `weak_visual_relations` 的 head-side `effective residual norm`：`0.7976 > 0.6909`
  - `ambiguous_material_relations` 的 head-side `effective residual norm`：`0.7504 > 0.6812`
- 同时，`Full Model` 的 head-side `residual / fused ratio` 也在 `head_noimg` 上系统更高：
  - `visual_relations`：`0.4461 > 0.4212`
  - `weak_visual_relations`：`0.4733 > 0.4279`
  - `ambiguous_material_relations`：`0.4490 > 0.4232`
- 因此 `7.2` 当前可以正式收口为：在 `Full Model` 内部，缺图 head 目标实体确实更依赖 residual 补偿；tail 方向由于当前 `paper_split` 中目标实体恒为 `no_img`，主要体现为结构路径整体主导而非额外的 `has_img/no_img` 对照
- `Residual-only` 的 subgroup 差异相对更弱或更混合，因此“缺图实体上 residual 更强”的核心证据主要来自 `Full Model` 的内部行为，而不是 `Residual-only` 的幅度变化

### 7.3 Fusion vs residual
- [x] 结合 mix 权重分析主导关系
- [x] 判断 fusion 是否长期被 residual 压制
- [x] 形成“互补还是竞争”的结论

当前进展：
- 已形成 `7.3` 的正式综合分析文档：
  - `docs/FUSION_VS_RESIDUAL_ANALYSIS.md`
- 当前判断综合了三类证据：
  - `docs/BEHAVIOR_SUMMARY.md`
  - `docs/RELATION_AWARE_GATE_SUMMARY.md`
  - `docs/RELATION_AWARE_RESIDUAL_SUMMARY.md`
- 当前已确认：
  - 在 `Full Model` 的 best epoch 上，最终混合权重长期偏向 residual：
    - `mix_w_fusion ≈ 0.2049 ± 0.0402`
    - `mix_w_residual ≈ 0.7951 ± 0.0402`
  - 从 first eval 到 best epoch，`mix_w_residual` 持续上升、`mix_w_fusion` 持续下降，因此 residual 偏好不是偶然快照，而是训练后期稳定形成的趋势
  - 但 fusion 并未失活：
    - `grad_fusion` 与 `grad_projection` 持续非零
    - relation-aware gate 在不同 relation group 上仍有系统差异
    - `6.2 MIN20` 已表明 `Full Model` 在多数中等支持度关系上稳定优于 `Gate-only`
  - relation-aware residual 进一步表明：当 head 目标实体缺图时，`Full Model` 内部 residual 贡献会系统上升，因此 residual 不只是“全局默认更强”，而是在弱视觉支持条件下被进一步偏好
- 因此 `7.3` 的正式收口结论为：
  - fusion 与 residual 不是纯冗余关系
  - 两者存在真实互补，因为 fusion 的加入能够把 `Gate-only` 推进到 `Full Model`
  - 但这种互补是明显不对称的：最终表示混合与弱图像条件下的行为都更偏向 residual
  - 当前最准确的描述是：`residual-dominant asymmetric complementarity`

## 8. 第七阶段：案例分析

### 8.1 成功案例
- [ ] 选择 `Full Model` 优于 `Residual-only` 的样本
- [ ] 分析是否依赖图像或文本线索

### 8.2 失败案例
- [ ] 选择 `Residual-only` 优于 `Full Model` 的样本
- [ ] 分析是否更依赖结构模式

### 8.3 案例结论
- [ ] 总结哪些场景更适合多模态
- [ ] 总结哪些场景更适合强结构表示

## 9. 第八阶段：论文主线最终判定

### 9.1 若出现以下结果
- `Full Model` 在关键子集上稳定优于 `Residual-only`
- 行为分析支持 fusion 与 residual 具有互补性

则保留：
- 分析驱动为主，方法改进为辅

### 9.2 若出现以下结果
- `Full Model` 全局仍不如 `Residual-only`
- 但在 `has_img` 或部分关系组上存在稳定收益

则转向：
- 多模态收益边界分析

### 9.3 若出现以下结果
- `Residual-only` 在几乎所有层面都持续占优
- 多模态局部收益也不明显

则进一步转向：
- 为什么该数据集上强结构表示天然占优

## 10. 写作前必须完成的最小清单
- [x] 五组主模型正式 test 结果完成
- [x] 主结果表完成
- [x] `has_img / no_img` 分组结果完成
- [x] relation type 分组结果完成
- [ ] gate / residual 行为分析完成
- [ ] 至少 2 组成功案例与 2 组失败案例完成
- [ ] 论文主线最终定稿
