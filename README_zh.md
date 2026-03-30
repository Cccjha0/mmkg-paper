# MMKG Project Research

当前仓库的重点已经收敛到 **OpenBG-IMG 上的多模态知识图谱补全研究**。它现在首先是一个论文实验仓库。

## 当前目录结构

```text
mmkg-project-research/
  data/
    datasets/
    cache/
  docs/
  ml/
    artifacts/
    configs/
    inference/
    training/
```

## 当前研究重点

- OpenBG-IMG 统一训练流程
- 论文专用 `train/dev/test` 划分
- 原始文本/图像特征缓存与模型内可训练投影
- 论文主模型与强结构基线的统一评测
- 图像可用性分组分析

## 当前主模型

- `Text-only`
- `Early Fusion`
- `Gate-only`
- `Residual-only`
- `Full Model`
- `ComplEx`
- `TuckER`

主要配置文件：

- `ml/configs/openbg_img_text_only.yaml`
- `ml/configs/openbg_img_early.yaml`
- `ml/configs/openbg_img_gate_only.yaml`
- `ml/configs/openbg_img_residual_only.yaml`
- `ml/configs/openbg_img_gated_vec_res_rel.yaml`
- `ml/configs/openbg_img_complex.yaml`
- `ml/configs/openbg_img_tucker.yaml`

## 环境建议

```powershell
conda create -n mmkg python=3.10 -y
conda activate mmkg
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 数据与缓存

原始数据默认放在：

```text
data/datasets/openbg_img/raw/
```

构建文本缓存：

```powershell
python ml/training/scripts/build_cache_openbg_img_text.py `
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv `
  --cache_dir data/cache/openbg_img
```

构建图像缓存：

```powershell
python ml/training/scripts/build_cache_openbg_img_image.py `
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv `
  --images_root data/datasets/openbg_img/raw/OpenBG-IMG_images `
  --cache_dir data/cache/openbg_img
```

当前缓存流程：

- 文本使用 `text_feat_raw.pt`
- 图像使用 `img_feat_raw.pt`
- 兼容旧版 `img_emb_raw.pt`
- `text_proj` / `img_proj` 在模型内部训练

## 论文专用数据划分

正式实验统一使用：

```text
data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_train.tsv
data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_dev.tsv
data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_test.tsv
```

重建命令：

```powershell
python ml/training/scripts/build_openbg_img_paper_split.py
```

## 训练入口

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_gated_vec_res_rel.yaml `
  --common ml/configs/common_seed1.yaml
```

输出目录：

```text
ml/artifacts/outputs/<exp_name>/<timestamp>_seed<seed>/
```

## 当前统一评测协议

- 三列 `train/dev/test`
- `dev` 只做 early stopping 和选模
- 最终统一在 `test` 上汇报
- filtered ranking
- `direction: both`

当前结果不能直接和早期 `tail-only` 内部结果横向比较。

## 常用文档

- `docs/MAIN_EXPERIMENT_RUNBOOK.md`
- `docs/HAS_IMG_RUNBOOK.md`
- `docs/RESULT_INDEX.md`
- `docs/MAIN_RESULTS_SUMMARY.md`
- `docs/INITIAL_JUDGMENT.md`
- `docs/HAS_IMG_SPLIT_SUMMARY.md`
