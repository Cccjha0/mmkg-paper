# ML Module

`ml` contains the active research code for the project.

## Subdirectories

- `training`: model implementations, training loops, evaluation code, and experiment scripts
- `inference`: lightweight loaders and predictor utilities for completed run directories
- `artifacts`: experiment outputs, checkpoints, metrics, and generated summaries
- `configs`: main experiment configs, seed configs, smoke configs, and ablation configs

## Current OpenBG-IMG Lineup

- `Text-only`
- `Early Fusion`
- `Gate-only`
- `Residual-only`
- `Full Model`
- `ComplEx`
- `TuckER`

## Important Entry Points

- `training/scripts/run_train.py`
- `training/scripts/build_openbg_img_paper_split.py`
- `training/scripts/build_result_index.py`
- `training/scripts/build_main_results_summary.py`
- `training/scripts/build_has_img_split_summary.py`
