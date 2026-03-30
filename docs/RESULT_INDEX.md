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
| `openbg_img_complex` | `ComplEx` | `openbg_img_complex/20260328_004926_seed1` | 1 | `-` | 0.2610 | 5 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_complex` | `ComplEx` | `openbg_img_complex/20260328_005017_seed2` | 2 | `-` | 0.2625 | 5 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_complex` | `ComplEx` | `openbg_img_complex/20260328_005322_seed3` | 3 | `-` | 0.2606 | 5 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_early` | `Early Fusion` | `openbg_img_early/20260326_235410_seed1` | 1 | `-` | 0.1670 | 90 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_early` | `Early Fusion` | `openbg_img_early/20260327_144829_seed2` | 2 | `-` | 0.1638 | 125 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_early` | `Early Fusion` | `openbg_img_early/20260327_153219_seed3` | 3 | `-` | 0.1626 | 130 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_gate_only` | `Gate-only` | `openbg_img_gate_only/20260327_173820_seed1` | 1 | `img_dropout=0.2` | 0.1726 | 70 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_gate_only` | `Gate-only` | `openbg_img_gate_only/20260327_181147_seed2` | 2 | `img_dropout=0.2` | 0.1705 | 75 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_gate_only` | `Gate-only` | `openbg_img_gate_only/20260327_185713_seed3` | 3 | `img_dropout=0.2` | 0.1763 | 120 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_gated_vec_res_rel` | `Full Model` | `openbg_img_gated_vec_res_rel/20260327_191500_seed1` | 1 | `use_normalized_mix=True, img_dropout=0.2` | 0.2204 | 30 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_gated_vec_res_rel` | `Full Model` | `openbg_img_gated_vec_res_rel/20260327_211759_seed3` | 3 | `use_normalized_mix=True, img_dropout=0.2` | 0.2068 | 40 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_gated_vec_res_rel` | `Full Model` | `openbg_img_gated_vec_res_rel/20260327_211818_seed2` | 2 | `use_normalized_mix=True, img_dropout=0.2` | 0.2027 | 60 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_residual_only` | `Residual-only` | `openbg_img_residual_only/20260326_171401_seed1` | 1 | `img_dropout=0.1` | 0.2967 | 90 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_residual_only` | `Residual-only` | `openbg_img_residual_only/20260326_172327_seed2` | 2 | `img_dropout=0.1` | 0.2972 | 90 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_residual_only` | `Residual-only` | `openbg_img_residual_only/20260326_173129_seed3` | 3 | `img_dropout=0.1` | 0.2971 | 95 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_text_only` | `Text-only` | `openbg_img_text_only/20260326_184048_seed1` | 1 | `img_dropout=0.0` | 0.1220 | 320 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_text_only` | `Text-only` | `openbg_img_text_only/20260326_191652_seed2` | 2 | `img_dropout=0.0` | 0.1298 | 320 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_text_only` | `Text-only` | `openbg_img_text_only/20260326_211111_seed3` | 3 | `img_dropout=0.0` | 0.1257 | 315 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_tucker` | `TuckER` | `openbg_img_tucker/20260328_011550_seed1` | 1 | `-` | 0.0886 | 20 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_tucker` | `TuckER` | `openbg_img_tucker/20260328_131930_seed2` | 2 | `-` | 0.0881 | 15 | 可用 | metrics 文件为 `metrics_seed1.csv` |
| `openbg_img_tucker` | `TuckER` | `openbg_img_tucker/20260328_170728_seed3` | 3 | `-` | 0.0865 | 15 | 可用 | metrics 文件为 `metrics_seed1.csv` |

## 4. 当前可直接使用的结果分组

### 4.1 当前相对完整的组

- `ComplEx`
  - 当前已有 3 个可用 run，seed: 1, 2, 3
  - 可作为后续主结果比较与汇总的基础
- `Early Fusion`
  - 当前已有 3 个可用 run，seed: 1, 2, 3
  - 可作为后续主结果比较与汇总的基础
- `Full Model`
  - 当前已有 3 个可用 run，seed: 1, 2, 3
  - 可作为后续主结果比较与汇总的基础
- `Gate-only`
  - 当前已有 3 个可用 run，seed: 1, 2, 3
  - 可作为后续主结果比较与汇总的基础
- `Residual-only`
  - 当前已有 3 个可用 run，seed: 1, 2, 3
  - 可作为后续主结果比较与汇总的基础
- `Text-only`
  - 当前已有 3 个可用 run，seed: 1, 2, 3
  - 可作为后续主结果比较与汇总的基础
- `TuckER`
  - 当前已有 3 个可用 run，seed: 1, 2, 3
  - 可作为后续主结果比较与汇总的基础

### 4.2 当前部分可用的组

- 当前无部分可用组

## 5. 当前主要缺口

- 当前 `results` 聚合目录为空，仅有 `.gitkeep`
- 当前五组主模型与两组结构强基线均已形成 3-seed 可用结果
- 当前主要工作重心应从“补齐主模型结果”转向“强基线对比、分组分析与原因诊断”

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
- [ ] 汇总七组模型的 `mean ± std` 并形成最终主结果表
- [ ] 将 `test_metrics.json` 纳入后续自动索引与汇总逻辑
- [ ] 开始 `Residual-only > Full Model` 的原因排查实验
- [ ] 开始 `Full Model` 与 `ComplEx / TuckER` 的正式对比分析
- [ ] 将 `results` 目录真正作为聚合输出目录使用
