# 统一结果索引表

## 1. 说明

本表用于统一记录当前 `ml/artifacts/outputs` 下可见实验结果，便于后续：

- 快速定位 run 目录
- 查找对应 baseline 与 seed
- 判断哪些结果可直接用于论文比较
- 判断哪些结果仍不完整，只能作为参考

该文档由 `ml/training/scripts/build_result_index.py` 自动生成。

## 2. 字段说明

- `实验组`: 输出目录名，对应一个实验配置族
- `模型口径`: 按当前项目语义整理后的模型解释
- `run目录`: 相对 `ml/artifacts/outputs` 的运行目录
- `seed`: 当前运行的随机种子
- `配置`: 关键开关组合
- `最佳MRR`: 当前 metrics 文件中记录到的最佳 `mrr`
- `最佳epoch`: 达到最佳 `mrr` 的 epoch
- `状态`: 当前结果是否适合直接纳入后续正式比较
- `备注`: 对命名异常或文件完整性的补充说明

## 3. 统一结果索引

| 实验组 | 模型口径 | run目录 | seed | 配置 | 最佳MRR | 最佳epoch | 状态 | 备注 |
|---|---|---|---:|---|---:|---:|---|---|
| `openbg_img_gate_only` | `gate-only` | `openbg_img_gate_only/20260308_180434_seed1` | 1 | `use_fusion=True, use_residual=False, img_dropout=0.2` | 0.3743 | 70 | 可用 | 配置与结果完整 |
| `openbg_img_gate_only` | `gate-only` | `openbg_img_gate_only/20260308_201856_seed2` | 2 | `use_fusion=True, use_residual=False, img_dropout=0.2` | 0.3472 | 170 | 可用 | 配置与结果完整 |
| `openbg_img_gate_only` | `gate-only` | `openbg_img_gate_only/20260309_001045_seed3` | 3 | `use_fusion=True, use_residual=False, img_dropout=0.2` | 0.3621 | 50 | 可用 | metrics 文件为 `metrics_gate_residual_seed2.csv` |
| `openbg_img_gated_vec_res_rel` | `Gate+Residual` | `openbg_img_gated_vec_res_rel/20260314_212911_seed1` | 1 | `model.name=openbg_img_gate_residual, use_normalized_mix=True, img_dropout=0.2` | 0.4788 | 50 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_gated_vec_res_rel` | `Gate+Residual` | `openbg_img_gated_vec_res_rel/20260314_223034_seed2` | 2 | `model.name=openbg_img_gate_residual, use_normalized_mix=True, img_dropout=0.2` | 0.5175 | 16 | 可用 | metrics 文件为 `metrics_seed2.csv` |
| `openbg_img_gated_vec_res_rel` | `Gate+Residual` | `openbg_img_gated_vec_res_rel/20260314_231558_seed3` | 3 | `model.name=openbg_img_gate_residual, use_normalized_mix=True, img_dropout=0.2` | 0.4978 | 30 | 可用 | metrics 文件为 `metrics_seed3.csv` |
| `openbg_img_gated_vec_res_rel` | `Gate+Residual` | `openbg_img_gated_vec_res_rel/20260315_000727_seed4` | 4 | `model.name=openbg_img_gate_residual, use_normalized_mix=True, img_dropout=0.2` | 0.4881 | 14 | 可用 | metrics 文件为 `metrics_seed4.csv` |
| `openbg_img_gated_vec_res_rel` | `Gate+Residual` | `openbg_img_gated_vec_res_rel/20260315_005115_seed5` | 5 | `model.name=openbg_img_gate_residual, use_normalized_mix=True, img_dropout=0.2` | 0.4364 | 16 | 可用 | 配置与结果完整 |
| `openbg_img_residual_only` | `residual-only` | `openbg_img_residual_only/20260313_161055_seed1` | 1 | `use_fusion=False, use_residual=True, img_dropout=0.1` | 0.5998 | 29 | 可用 | 配置与结果完整 |

## 4. 当前可直接使用的结果分组

### 4.1 当前相对完整的组

- `Gate+Residual`
  - 当前已有 5 个可用 run，seed: 1, 2, 3, 4, 5
  - 可作为后续主结果比较与汇总的基础
- `gate-only`
  - 当前已有 3 个可用 run，seed: 1, 2, 3
  - 可作为后续主结果比较与汇总的基础

### 4.2 当前部分可用的组

- `residual-only`
  - 当前有 1 个可用 run，总计 1 个目录
  - 可用于观察趋势，但不足以直接形成稳定结论

## 5. 当前主要缺口

- 缺少统一整理后的 `text-only` 结果索引
- 缺少统一整理后的 `early fusion` 结果索引
- `residual-only` 缺少多 seed 结果
- 当前 `results` 聚合目录为空，仅有 `.gitkeep`

## 6. 当前建议标签

- `可用`
  - 配置明确
  - metrics 完整
  - 可直接进入后续比较或汇总

- `参考可用`
  - 存在部分配置、checkpoint 或结果文件
  - 可用于观察趋势，但不足以形成正式结论

- `不完整`
  - 缺失关键文件
  - 只能保留目录信息，不能直接进入论文汇报

## 7. 下一步维护任务

- [ ] 将后续新实验统一追加到本索引表
- [ ] 为 `text-only` 建立独立索引行
- [ ] 为 `early fusion` 建立独立索引行
- [ ] 补齐 `residual-only` 的多 seed metrics
- [ ] 将 `results` 目录真正作为聚合输出目录使用
