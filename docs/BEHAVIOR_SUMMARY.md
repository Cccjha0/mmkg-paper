# Behavior Summary

## 1. Purpose

This document summarizes the current first-round behavior analysis based on training-time diagnostics already saved in run directories.

Current focus:

- gate mean / std behavior
- residual scale behavior
- fusion vs residual mixture weights
- gradient-group diagnostics

The statistics below are summarized at each run's best-dev epoch, i.e. the epoch with the highest recorded dev MRR in the metrics CSV.

## 2. Selected Runs

Outputs root: `ml/artifacts/outputs`

- `Gate-only`: `openbg_img_gate_only/20260329_202546_seed1`, `openbg_img_gate_only/20260329_234132_seed2`, `openbg_img_gate_only/20260330_010533_seed3`
- `Full Model`: `openbg_img_gated_vec_res_rel/20260329_205816_seed1`, `openbg_img_gated_vec_res_rel/20260330_130244_seed2`, `openbg_img_gated_vec_res_rel/20260330_134147_seed3`
- `Residual-only`: `openbg_img_residual_only/20260329_202016_seed1`, `openbg_img_residual_only/20260329_231234_seed2`, `openbg_img_residual_only/20260329_233525_seed3`

Duplicate handling:
- `Gate-only` seed `1` had multiple runs; selected latest: `openbg_img_gate_only/20260329_202546_seed1`
- `Gate-only` seed `2` had multiple runs; selected latest: `openbg_img_gate_only/20260329_234132_seed2`
- `Gate-only` seed `3` had multiple runs; selected latest: `openbg_img_gate_only/20260330_010533_seed3`
- `Full Model` seed `1` had multiple runs; selected latest: `openbg_img_gated_vec_res_rel/20260329_205816_seed1`
- `Full Model` seed `2` had multiple runs; selected latest: `openbg_img_gated_vec_res_rel/20260330_130244_seed2`
- `Full Model` seed `3` had multiple runs; selected latest: `openbg_img_gated_vec_res_rel/20260330_134147_seed3`
- `Residual-only` seed `1` had multiple runs; selected latest: `openbg_img_residual_only/20260329_202016_seed1`
- `Residual-only` seed `2` had multiple runs; selected latest: `openbg_img_residual_only/20260329_231234_seed2`
- `Residual-only` seed `3` had multiple runs; selected latest: `openbg_img_residual_only/20260329_233525_seed3`

## 3. Gate Statistics At Best Epoch

| Model | Seeds | Gate Mean (All) | Gate Std (All) | Gate Mean (Img) | Gate Mean (NoImg) | Img-NoImg Gap |
|---|---:|---:|---:|---:|---:|---:|
| Gate-only | 3 | 0.5346 +/- 0.0107 | 0.1015 +/- 0.0016 | 0.4445 +/- 0.0117 | 0.6308 +/- 0.0083 | -0.1863 +/- 0.0000 |
| Full Model | 3 | 0.4288 +/- 0.0058 | 0.1044 +/- 0.0050 | 0.3562 +/- 0.0035 | 0.5073 +/- 0.0104 | -0.1510 +/- 0.0000 |

## 4. Residual And Mix Statistics At Best Epoch

| Model | Seeds | Residual Scale | Mix Fusion | Mix Residual | Grad Residual | Grad Fusion | Grad Projection |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gate-only | 3 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.8093 +/- 0.1042 | 0.9581 +/- 0.1208 |
| Full Model | 3 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 | 0.1104 +/- 0.0463 | 0.3788 +/- 0.1921 | 0.5161 +/- 0.1237 |
| Residual-only | 3 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0791 +/- 0.0057 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |

## 5. Delta From First Eval To Best Epoch

| Model | Gate Mean Delta | Gate Img Delta | Gate NoImg Delta | Residual Scale Delta | Mix Fusion Delta | Mix Residual Delta |
|---|---:|---:|---:|---:|---:|---:|
| Gate-only | -0.0146 +/- 0.0115 | -0.0833 +/- 0.0111 | 0.0571 +/- 0.0137 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| Full Model | 0.0063 +/- 0.0139 | -0.0051 +/- 0.0130 | 0.0150 +/- 0.0214 | 0.3728 +/- 0.1808 | 0.1234 +/- 0.0404 | -0.1234 +/- 0.0404 |
| Residual-only | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.1346 +/- 0.0034 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |

## 6. Per-Run Best-Epoch Detail

| Model | Seed | Run | Best Epoch | Best Dev MRR | Gate Mean (All) | Gate Mean (Img) | Gate Mean (NoImg) | Residual Scale | Mix Fusion | Mix Residual |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gate-only | 1 | `openbg_img_gate_only/20260329_202546_seed1` | 70 | 0.1726 | 0.5361 | 0.4477 | 0.6318 | 0.0000 | 0.0000 | 0.0000 |
| Gate-only | 2 | `openbg_img_gate_only/20260329_234132_seed2` | 75 | 0.1705 | 0.5232 | 0.4314 | 0.6220 | 0.0000 | 0.0000 | 0.0000 |
| Gate-only | 3 | `openbg_img_gate_only/20260330_010533_seed3` | 120 | 0.1763 | 0.5445 | 0.4542 | 0.6386 | 0.0000 | 0.0000 | 0.0000 |
| Full Model | 1 | `openbg_img_gated_vec_res_rel/20260329_205816_seed1` | 30 | 0.2204 | 0.4295 | 0.3534 | 0.5140 | 0.3607 | 0.1673 | 0.8327 |
| Full Model | 2 | `openbg_img_gated_vec_res_rel/20260330_130244_seed2` | 60 | 0.2027 | 0.4227 | 0.3552 | 0.4953 | 0.7154 | 0.2472 | 0.7528 |
| Full Model | 3 | `openbg_img_gated_vec_res_rel/20260330_134147_seed3` | 40 | 0.2068 | 0.4341 | 0.3602 | 0.5126 | 0.4834 | 0.2002 | 0.7998 |
| Residual-only | 1 | `openbg_img_residual_only/20260329_202016_seed1` | 90 | 0.2967 | 0.0000 | 0.0000 | 0.0000 | 0.2565 | 0.0000 | 0.0000 |
| Residual-only | 2 | `openbg_img_residual_only/20260329_231234_seed2` | 90 | 0.2972 | 0.0000 | 0.0000 | 0.0000 | 0.2566 | 0.0000 | 0.0000 |
| Residual-only | 3 | `openbg_img_residual_only/20260329_233525_seed3` | 95 | 0.2971 | 0.0000 | 0.0000 | 0.0000 | 0.2624 | 0.0000 | 0.0000 |

## 7. First-Round Takeaways

- This summary is based only on run-level diagnostics already saved during training; it does not yet provide relation-aware behavior analysis.
- Gate statistics can already support the first half of `7.1` (mean/std and image-availability comparison).
- Residual-scale, mix-weight, and gradient-group statistics can already support the first round of `7.2` and `7.3`.
- The next follow-up should be a relation-aware behavior script to connect these diagnostics back to the completed `6.2 relation type` analysis.
