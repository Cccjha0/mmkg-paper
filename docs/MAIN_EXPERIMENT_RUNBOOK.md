# 主实验运行清单

## 1. 目标

当前主实验统一基于：

- 五组主模型
- `paper_split` 三列 `train/dev/test`
- `dev` 用于 early stopping / model selection
- `test` 用于最终汇报
- 每组 `3 seeds`

五组主模型如下：

1. `Text-only`
2. `Early Fusion`
3. `Gate-only`
4. `Residual-only`
5. `Full Model`

## 2. 推荐运行顺序

建议按以下顺序跑：

1. `Residual-only`
2. `Text-only`
3. `Early Fusion`
4. `Gate-only`
5. `Full Model`

原因：

- 先拿到强结构基线
- 再补简单多模态模型
- 最后跑最复杂的 `Full Model`

## 3. 公共配置

正式实验不要再用 `common_smoke.yaml`。

请使用：

- `ml/configs/common_seed1.yaml`
- `ml/configs/common_seed2.yaml`
- `ml/configs/common_seed3.yaml`

这三份配置与正式 `common.yaml` 保持一致，只是固定了不同随机种子：

- `seed=1`
- `seed=2`
- `seed=3`

## 4. 各模型正式运行命令

### 4.1 Residual-only

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_residual_only.yaml `
  --common ml/configs/common_seed1.yaml
```

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_residual_only.yaml `
  --common ml/configs/common_seed2.yaml
```

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_residual_only.yaml `
  --common ml/configs/common_seed3.yaml
```

### 4.2 Text-only

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_text_only.yaml `
  --common ml/configs/common_seed1.yaml
```

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_text_only.yaml `
  --common ml/configs/common_seed2.yaml
```

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_text_only.yaml `
  --common ml/configs/common_seed3.yaml
```

### 4.3 Early Fusion

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_early.yaml `
  --common ml/configs/common_seed1.yaml
```

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_early.yaml `
  --common ml/configs/common_seed2.yaml
```

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_early.yaml `
  --common ml/configs/common_seed3.yaml
```

### 4.4 Gate-only

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_gate_only.yaml `
  --common ml/configs/common_seed1.yaml
```

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_gate_only.yaml `
  --common ml/configs/common_seed2.yaml
```

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_gate_only.yaml `
  --common ml/configs/common_seed3.yaml
```

### 4.5 Full Model

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_gated_vec_res_rel.yaml `
  --common ml/configs/common_seed1.yaml
```

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_gated_vec_res_rel.yaml `
  --common ml/configs/common_seed2.yaml
```

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_gated_vec_res_rel.yaml `
  --common ml/configs/common_seed3.yaml
```

## 5. 每次运行后检查什么

每个 run 目录里至少应有：

- `config_merged.json`
- `metrics_seed1.csv` 或对应 metrics 文件
- `best.ckpt`
- `test_metrics.json`

重点确认：

- `metrics` 中有 `mrr / hits@1 / hits@3 / hits@10`
- `test_metrics.json` 已生成
- `run_dir` 路径和 seed 对应正确

## 6. 跑完后的统一整理

每组 3 seeds 跑完后，建议立即执行：

```powershell
python ml/training/scripts/build_result_index.py
```

然后再汇总：

- 每组 `test MRR / Hits@1 / Hits@3 / Hits@10`
- `mean ± std`
- 主结果表

## 7. 执行纪律

- 不要在正式 3 seeds 期间再改模型定义
- 不要混用 `raw` 旧路径和 `paper_split` 路径
- 不要混用 `common_smoke.yaml` 和正式 `common_seed*.yaml`
- 五组主模型全部跑完之前，不要提前下结论

## 8. 当前建议

最稳妥的执行方式是：

1. 先完整跑完 `Residual-only` 3 seeds
2. 确认输出结构无误
3. 再按顺序推进其余四组

这样可以最快得到最关键的结构基线结果。
