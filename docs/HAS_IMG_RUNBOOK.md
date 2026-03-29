# 6.1 has_img / no_img 运行清单

## 1. 目标

当前 `6.1` 先优先补齐三组模型的 `3 seeds`：

- `Residual-only`
- `Gate-only`
- `Full Model`

运行完成后，每个 run 目录下的 `test_metrics.json` 都会包含：

- `head_has_img_mrr`
- `head_no_img_mrr`
- `tail_no_img_mrr`

## 2. Residual-only

```powershell
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_residual_only.yaml --common ml/configs/common_seed1.yaml
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_residual_only.yaml --common ml/configs/common_seed2.yaml
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_residual_only.yaml --common ml/configs/common_seed3.yaml
```

## 3. Gate-only

```powershell
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_gate_only.yaml --common ml/configs/common_seed1.yaml
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_gate_only.yaml --common ml/configs/common_seed2.yaml
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_gate_only.yaml --common ml/configs/common_seed3.yaml
```

## 4. Full Model

```powershell
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_gated_vec_res_rel.yaml --common ml/configs/common_seed1.yaml
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_gated_vec_res_rel.yaml --common ml/configs/common_seed2.yaml
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_gated_vec_res_rel.yaml --common ml/configs/common_seed3.yaml
```

## 5. 运行后检查

每组重点检查：

- run 目录下存在 `test_metrics.json`
- `test_metrics.json` 中存在：
  - `head_has_img_mrr`
  - `head_no_img_mrr`
  - `tail_no_img_mrr`
  - `head_has_img_count`
  - `head_no_img_count`
  - `tail_no_img_count`

## 6. 后处理

三组 `3 seeds` 跑完后，执行：

```powershell
python ml/training/scripts/build_has_img_split_summary.py
```

将会更新：

- `docs/HAS_IMG_SPLIT_SUMMARY.md`
- `docs/has_img_split_summary.json`
