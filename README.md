# MMKG Project Research

This repository currently focuses on the ML research workflow for multimodal knowledge graph completion on OpenBG-IMG. The active work is a paper-oriented training, evaluation, and analysis codebase.

## Repository Layout

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

## Current Focus

- unified OpenBG-IMG training pipeline
- paper split with 3-column `train/dev/test`
- raw text/image cache loading with trainable projection layers
- explicit model lineup for the paper
- structural baselines under the same evaluation protocol
- subgroup analysis for image availability

## Main Models

- `Text-only`
- `Early Fusion`
- `Gate-only`
- `Residual-only`
- `Full Model`
- `ComplEx`
- `TuckER`

Main experiment configs:

- `ml/configs/openbg_img_text_only.yaml`
- `ml/configs/openbg_img_early.yaml`
- `ml/configs/openbg_img_gate_only.yaml`
- `ml/configs/openbg_img_residual_only.yaml`
- `ml/configs/openbg_img_gated_vec_res_rel.yaml`
- `ml/configs/openbg_img_complex.yaml`
- `ml/configs/openbg_img_tucker.yaml`

## Environment

```powershell
conda create -n mmkg python=3.10 -y
conda activate mmkg
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data and Cache

Raw OpenBG-IMG files are expected under:

```text
data/datasets/openbg_img/raw/
```

Build text cache:

```powershell
python ml/training/scripts/build_cache_openbg_img_text.py `
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv `
  --cache_dir data/cache/openbg_img
```

Build image cache:

```powershell
python ml/training/scripts/build_cache_openbg_img_image.py `
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv `
  --images_root data/datasets/openbg_img/raw/OpenBG-IMG_images `
  --cache_dir data/cache/openbg_img
```

Current cache behavior:

- text uses `text_feat_raw.pt`
- image uses `img_feat_raw.pt`
- legacy `img_emb_raw.pt` remains compatible
- model-side `text_proj` and `img_proj` are trainable

## Paper Split

Formal paper experiments use:

```text
data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_train.tsv
data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_dev.tsv
data/datasets/openbg_img/paper_split/OpenBG-IMG_paper_test.tsv
```

Rebuild the split with:

```powershell
python ml/training/scripts/build_openbg_img_paper_split.py
```

## Training

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_gated_vec_res_rel.yaml `
  --common ml/configs/common_seed1.yaml
```

Artifacts are written to:

```text
ml/artifacts/outputs/<exp_name>/<timestamp>_seed<seed>/
```

## Evaluation Protocol

Current formal protocol:

- 3-column `train/dev/test`
- `dev` for early stopping and model selection
- final reporting on `test`
- filtered ranking
- `direction: both`

Do not compare current `both` results directly with older `tail-only` internal runs.

## Useful Documents

- `docs/MAIN_EXPERIMENT_RUNBOOK.md`
- `docs/HAS_IMG_RUNBOOK.md`
- `docs/RESULT_INDEX.md`
- `docs/MAIN_RESULTS_SUMMARY.md`
- `docs/INITIAL_JUDGMENT.md`
- `docs/HAS_IMG_SPLIT_SUMMARY.md`

## Current Result Snapshot

Current ordering under the unified paper protocol:

1. `Residual-only`
2. `ComplEx`
3. `Full Model`
4. `Gate-only`
5. `Early Fusion`
6. `Text-only`
7. `TuckER`
