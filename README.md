# MMKG Project Research

This repository contains the paper-oriented code and artifacts for multimodal knowledge graph completion on **OpenBG-IMG**. The current focus is protocol-aware analysis of fixed experts, clean routing, candidate-level routing, and score-aware expert combination.

## Dataset: OpenBG-IMG

Expected raw data layout:

```text
data/datasets/openbg_img/raw/
  OpenBG-IMG_train.tsv
  OpenBG-IMG_dev.tsv
  OpenBG-IMG_test.tsv
  OpenBG-IMG_entity2text.tsv
  OpenBG-IMG_relation2text.tsv
  OpenBG-IMG_images/
```

Formal paper experiments use the unified `paper_split`:

```text
data/datasets/openbg_img/paper_split/
  OpenBG-IMG_paper_train.tsv
  OpenBG-IMG_paper_dev.tsv
  OpenBG-IMG_paper_test.tsv
```

The paper protocol uses filtered bidirectional ranking. The dev split is used for checkpoint/model selection and score-interpolation alpha selection; final reported results use the test split.

## Environment

```powershell
conda create -n mmkg python=3.10 -y
conda activate mmkg
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## How To Prepare Data

Build the paper split:

```powershell
python ml/training/scripts/build_openbg_img_paper_split.py
```

Build text and image feature caches:

```powershell
python ml/training/scripts/build_cache_openbg_img_text.py `
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv `
  --cache_dir data/cache/openbg_img

python ml/training/scripts/build_cache_openbg_img_image.py `
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv `
  --images_root data/datasets/openbg_img/raw/OpenBG-IMG_images `
  --cache_dir data/cache/openbg_img
```

Expected cache files include:

```text
data/cache/openbg_img/text_feat_raw.pt
data/cache/openbg_img/img_feat_raw.pt
data/cache/openbg_img/has_img.pt
```

Optional sanity checks and dataset statistics:

```powershell
python scripts/build_dataset_statistics.py
python scripts/build_protocol_audit_sanity.py
python scripts/build_target_side_regime_counts.py
```

## How To Train Fixed Experts

The fixed experts used in the paper are:

- `ComplEx`
- `TuckER`
- `Text-only`
- `Early Fusion`
- `Gate-only`
- `Residual-only`
- `Full Model`

Train one model/seed:

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_residual_only.yaml `
  --common ml/configs/common_seed1.yaml
```

Run all three seeds by changing the common config:

```powershell
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_residual_only.yaml --common ml/configs/common_seed1.yaml
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_residual_only.yaml --common ml/configs/common_seed2.yaml
python ml/training/scripts/run_train.py --config ml/configs/openbg_img_residual_only.yaml --common ml/configs/common_seed3.yaml
```

Replace `openbg_img_residual_only.yaml` with the other model configs:

```text
ml/configs/openbg_img_complex.yaml
ml/configs/openbg_img_tucker.yaml
ml/configs/openbg_img_text_only.yaml
ml/configs/openbg_img_early.yaml
ml/configs/openbg_img_gate_only.yaml
ml/configs/openbg_img_residual_only.yaml
ml/configs/openbg_img_gated_vec_res_rel.yaml
```

Artifacts are written to:

```text
ml/artifacts/outputs/<exp_name>/<timestamp>_seed<seed>/
```

For router and score-combination analyses, export locked per-query evaluations after fixed-expert training:

```powershell
python scripts/export_query_eval.py
```

The downstream scripts expect files such as:

```text
outputs/router/test/gate_only_query_eval_seed1.csv
outputs/router/test/residual_only_query_eval_seed1.csv
outputs/router/test/full_model_query_eval_seed1.csv
```

## How To Reproduce Tables 3-9

The main paper tables are generated from saved model artifacts and evaluation CSVs. If the fixed-expert and router artifacts already exist, these commands are table-generation/evaluation commands rather than full retraining commands.

| Table | Purpose | Command or output |
|---|---|---|
| Table 3 | OpenBG-IMG dataset statistics | `python scripts/build_dataset_statistics.py` |
| Table 4 | Fixed-expert model comparison | `python ml/training/scripts/build_main_results_summary.py` |
| Table 5 | Clean routing baselines and oracle bounds | `python scripts/build_paper_tables_router_v2.py` and `outputs/router/eval/clean/main_results_table_clean.csv` |
| Table 6 | Candidate-aware router comparison | `python scripts/build_candidate_router_paper_tables.py` |
| Table 7 | Candidate-router subgroup/significance diagnostics | `python scripts/build_candidate_router_paper_tables.py` |
| Table 8 | Score-aware expert-combination baselines | `python scripts/eval_score_ensemble_baselines.py` |
| Table 9 | Score-aware bootstrap confidence intervals | `python scripts/run_score_aware_bootstrap_ci.py` |

Useful direct regeneration commands:

```powershell
python scripts/build_dataset_statistics.py
python ml/training/scripts/build_main_results_summary.py
python scripts/build_paper_tables_router_v2.py
python scripts/build_candidate_router_paper_tables.py
python scripts/eval_score_ensemble_baselines.py
python scripts/run_score_aware_bootstrap_ci.py
```

Important output locations:

```text
docs/paper_tables/
outputs/router/eval/
outputs/candidate_router/eval/tables/
outputs/score_ensemble/eval/
```

`scripts/eval_score_ensemble_baselines.py` performs full score interpolation/ranking evaluation and can be slow on CPU or laptop GPUs. It supports progress logging and checkpoint/resume behavior for long local runs.

## How To Reproduce Supplementary Tables S1-S7

Supplementary tables are collected under:

```text
docs/paper_tables/supplementary/
```

Current supplementary table bundle:

| Table | Purpose | Command |
|---|---|---|
| S1 | Relation-group sanity check | `python scripts/build_supplementary_tables.py` |
| S2 | Degree-bucket sanity check | `python scripts/build_degree_bucket_sanity_table.py` |
| S3 | Ordinal/scalar gain modeling | `python scripts/build_supplementary_tables.py` |
| S4 | Delta sensitivity of clean routing | `python scripts/build_supplementary_tables.py` |
| S5 | Per-seed fixed-expert model comparison | `python scripts/build_supplementary_tables_s5_s7.py` |
| S6 | Bootstrap implementation details | `python scripts/build_supplementary_tables_s5_s7.py` |
| S7 | Hyperparameter/training configuration summary | `python scripts/build_supplementary_tables_s5_s7.py` |

Recommended regeneration order:

```powershell
python scripts/build_degree_bucket_sanity_table.py
python scripts/build_supplementary_tables.py
python scripts/build_supplementary_tables_s5_s7.py
```

Notes:

- S1-S4 depend on the relation-group, clean-router, ordinal-router, regression-router, and delta-sensitivity outputs.
- S5-S7 depend on locked per-query evaluation CSVs, score-aware bootstrap outputs, and saved training/config summaries.
- If you only need the newest S5-S7 files, run `python scripts/build_supplementary_tables_s5_s7.py`.

## Expected Hardware / Approximate Runtime

Approximate runtimes depend heavily on GPU, disk speed, and whether artifacts are already cached.

| Step | Expected hardware | Approximate runtime |
|---|---|---|
| Text cache build | CPU acceptable | minutes |
| Image cache build | NVIDIA GPU recommended | tens of minutes to a few hours |
| One fixed-expert seed | NVIDIA GPU recommended; 8GB+ VRAM preferred | about 1-6 hours per seed/model |
| Full fixed-expert sweep | NVIDIA GPU strongly recommended | multiple days on one laptop GPU |
| Query-eval export | GPU recommended | tens of minutes per model/seed |
| Clean router training/evaluation | CPU acceptable | minutes |
| Candidate-router training | GPU recommended | tens of minutes to a few hours |
| Score-interpolation alpha sweep / full ranking | GPU recommended; resume enabled | about 1-3 hours on a laptop RTX 3050 for a full run, depending on batch/chunk settings |
| Bootstrap CI generation | CPU acceptable | seconds to minutes |
| LaTeX table generation from existing outputs | CPU acceptable | seconds |

For local laptop runs, prefer smaller `--query-batch-size` / `--chunk-size` if memory is tight. Long score-ensemble evaluations should be run with checkpoints enabled so interrupted runs can resume.

## Reproducibility Notes

- Use `paper_split` for all paper-numbered results.
- Use three seeds (`common_seed1.yaml`, `common_seed2.yaml`, `common_seed3.yaml`) for fixed experts.
- Report final metrics on the test split only.
- Do not compare current bidirectional filtered-ranking results with older tail-only exploratory runs.
- For bootstrap CIs, the preferred unit is the seed-averaged original query, not raw seed-query records.
