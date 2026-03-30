# Relation-Aware Residual Summary

## 1. Purpose

This document summarizes residual-branch behavior on real `(entity, relation)` pairs from the test split.

Current focus:

- residual norm and effective residual contribution
- fused-branch norm and effective fused contribution
- residual-to-fused ratio on real subgroup and relation-group slices
- per-relation residual details with `test triples >= 20`

## 2. Selected Runs

Outputs root: `ml/artifacts/outputs`

Grouping source: `E:/learn/R&D/mmkg-project-research/docs/relation_type_groups_draft.json`

- `Full Model`: `openbg_img_gated_vec_res_rel/20260329_205816_seed1`, `openbg_img_gated_vec_res_rel/20260330_130244_seed2`, `openbg_img_gated_vec_res_rel/20260330_134147_seed3`
- `Residual-only`: `openbg_img_residual_only/20260329_202016_seed1`, `openbg_img_residual_only/20260329_231234_seed2`, `openbg_img_residual_only/20260329_233525_seed3`

Duplicate handling:
- `Full Model` seed `1` had multiple runs; selected latest: `openbg_img_gated_vec_res_rel/20260329_205816_seed1`
- `Full Model` seed `2` had multiple runs; selected latest: `openbg_img_gated_vec_res_rel/20260330_130244_seed2`
- `Full Model` seed `3` had multiple runs; selected latest: `openbg_img_gated_vec_res_rel/20260330_134147_seed3`
- `Residual-only` seed `1` had multiple runs; selected latest: `openbg_img_residual_only/20260329_202016_seed1`
- `Residual-only` seed `2` had multiple runs; selected latest: `openbg_img_residual_only/20260329_231234_seed2`
- `Residual-only` seed `3` had multiple runs; selected latest: `openbg_img_residual_only/20260329_233525_seed3`

## 3. Group Definition and Test Coverage

| Group | Defined Relations | Relations in Test | Test Triples | Preview |
|---|---:|---:|---:|---|
| `visual_relations` | 28 | 28 | 3823 | `rel_0000` (图案), `rel_0006` (流行元素), `rel_0008` (色泽), `rel_0009` (裤长), `rel_0012` (是否带帽子), `rel_0020` (设计元素), `rel_0024` (旗袍款式), `rel_0027` (裤款式) |
| `weak_visual_relations` | 39 | 37 | 3383 | `rel_0004` (包装数量), `rel_0005` (地市), `rel_0010` (品牌), `rel_0011` (品牌类型), `rel_0014` (储存条件), `rel_0016` (关联场景), `rel_0019` (品牌归属地), `rel_0021` (食品工艺) |
| `ambiguous_material_relations` | 15 | 15 | 1201 | `rel_0003` (面料), `rel_0029` (鞋面材质), `rel_0035` (里料材质), `rel_0046` (服饰工艺), `rel_0059` (内胆材质), `rel_0069` (帮面材质), `rel_0073` (面料材质), `rel_0074` (里料) |

## 4.1 `visual_relations`


### head_has_img

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.8685 +/- 0.4832 | 0.6776 +/- 0.3444 | 7.4213 +/- 1.5233 | 1.5614 +/- 0.6203 | 0.4212 +/- 0.0498 | -0.8838 +/- 0.2762 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.1074 +/- 0.9789 | 1.6495 +/- 0.6835 | 8.4744 +/- 2.2356 | 1.7963 +/- 0.8146 | 0.9311 +/- 0.0445 | -0.1468 +/- 0.1317 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3655 +/- 0.0107 | 0.3655 +/- 0.0107 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3655 +/- 0.0107 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.8580 +/- 0.0541 | 1.8580 +/- 0.0541 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.8580 +/- 0.0541 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### head_noimg

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.9560 +/- 0.5231 | 0.7462 +/- 0.3723 | 7.7441 +/- 1.7868 | 1.6346 +/- 0.6904 | 0.4461 +/- 0.0378 | -0.8885 +/- 0.3181 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.1175 +/- 0.9833 | 1.6574 +/- 0.6866 | 8.4953 +/- 2.2451 | 1.8008 +/- 0.8174 | 0.9332 +/- 0.0455 | -0.1434 +/- 0.1315 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3664 +/- 0.0107 | 0.3664 +/- 0.0107 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3664 +/- 0.0107 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.8739 +/- 0.0533 | 1.8739 +/- 0.0533 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.8739 +/- 0.0533 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### tail_noimg

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.8932 +/- 0.4945 | 0.6970 +/- 0.3523 | 7.5125 +/- 1.5977 | 1.5821 +/- 0.6401 | 0.4282 +/- 0.0464 | -0.8851 +/- 0.2880 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.1102 +/- 0.9801 | 1.6517 +/- 0.6844 | 8.4803 +/- 2.2383 | 1.7976 +/- 0.8154 | 0.9317 +/- 0.0448 | -0.1459 +/- 0.1316 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3658 +/- 0.0107 | 0.3658 +/- 0.0107 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3658 +/- 0.0107 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.8625 +/- 0.0538 | 1.8625 +/- 0.0538 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.8625 +/- 0.0538 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### target_has_img

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.8685 +/- 0.4832 | 0.6776 +/- 0.3444 | 7.4213 +/- 1.5233 | 1.5614 +/- 0.6203 | 0.4212 +/- 0.0498 | -0.8838 +/- 0.2762 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.1074 +/- 0.9789 | 1.6495 +/- 0.6835 | 8.4744 +/- 2.2356 | 1.7963 +/- 0.8146 | 0.9311 +/- 0.0445 | -0.1468 +/- 0.1317 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3655 +/- 0.0107 | 0.3655 +/- 0.0107 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3655 +/- 0.0107 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.8580 +/- 0.0541 | 1.8580 +/- 0.0541 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.8580 +/- 0.0541 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### target_noimg

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.9070 +/- 0.5008 | 0.7078 +/- 0.3567 | 7.5636 +/- 1.6394 | 1.5937 +/- 0.6512 | 0.4322 +/- 0.0445 | -0.8859 +/- 0.2946 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.1118 +/- 0.9808 | 1.6530 +/- 0.6849 | 8.4836 +/- 2.2398 | 1.7983 +/- 0.8158 | 0.9320 +/- 0.0449 | -0.1453 +/- 0.1316 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3659 +/- 0.0107 | 0.3659 +/- 0.0107 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3659 +/- 0.0107 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.8650 +/- 0.0537 | 1.8650 +/- 0.0537 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.8650 +/- 0.0537 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### all_targets

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.8932 +/- 0.4945 | 0.6970 +/- 0.3523 | 7.5125 +/- 1.5977 | 1.5821 +/- 0.6401 | 0.4282 +/- 0.0464 | -0.8851 +/- 0.2880 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.1102 +/- 0.9801 | 1.6517 +/- 0.6844 | 8.4803 +/- 2.2383 | 1.7976 +/- 0.8154 | 0.9317 +/- 0.0448 | -0.1459 +/- 0.1316 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3658 +/- 0.0107 | 0.3658 +/- 0.0107 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3658 +/- 0.0107 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.8625 +/- 0.0538 | 1.8625 +/- 0.0538 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.8625 +/- 0.0538 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

Per-relation residual preview (`test triples >= 20`; kept 24 relations):

| Relation | Chinese | Model | Subgroup | Side | Seeds | Effective Residual | Effective Fused | Residual/Fused Ratio | Test Triples |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `rel_0000` | 图案 | Full Model | head_has_img | head | 3 | 0.6714 +/- 0.3430 | 1.5651 +/- 0.6189 | 0.4159 +/- 0.0512 | 278 |
| `rel_0000` | 图案 | Full Model | head_has_img | tail | 3 | 1.6972 +/- 0.7093 | 1.8126 +/- 0.8225 | 0.9488 +/- 0.0618 | 278 |
| `rel_0000` | 图案 | Full Model | head_noimg | head | 3 | 0.7470 +/- 0.3700 | 1.6331 +/- 0.6880 | 0.4473 +/- 0.0365 | 278 |
| `rel_0000` | 图案 | Full Model | head_noimg | tail | 3 | 1.7096 +/- 0.6922 | 1.8139 +/- 0.8237 | 0.9597 +/- 0.0767 | 278 |
| `rel_0000` | 图案 | Full Model | tail_noimg | head | 3 | 0.6910 +/- 0.3500 | 1.5827 +/- 0.6368 | 0.4240 +/- 0.0474 | 278 |
| `rel_0000` | 图案 | Full Model | tail_noimg | tail | 3 | 1.7004 +/- 0.7048 | 1.8129 +/- 0.8228 | 0.9517 +/- 0.0656 | 278 |
| `rel_0000` | 图案 | Residual-only | head_has_img | head | 3 | 0.3581 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 278 |
| `rel_0000` | 图案 | Residual-only | head_has_img | tail | 3 | 1.8901 +/- 0.0412 | 0.0000 +/- 0.0000 | n/a | 278 |
| `rel_0000` | 图案 | Residual-only | head_noimg | head | 3 | 0.3709 +/- 0.0113 | 0.0000 +/- 0.0000 | n/a | 278 |
| `rel_0000` | 图案 | Residual-only | head_noimg | tail | 3 | 1.8512 +/- 0.0429 | 0.0000 +/- 0.0000 | n/a | 278 |
| `rel_0000` | 图案 | Residual-only | tail_noimg | head | 3 | 0.3614 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 278 |
| `rel_0000` | 图案 | Residual-only | tail_noimg | tail | 3 | 1.8800 +/- 0.0416 | 0.0000 +/- 0.0000 | n/a | 278 |
| `rel_0006` | 流行元素 | Full Model | head_has_img | head | 3 | 0.6471 +/- 0.3345 | 1.5736 +/- 0.6273 | 0.3984 +/- 0.0504 | 122 |
| `rel_0006` | 流行元素 | Full Model | head_has_img | tail | 3 | 1.6262 +/- 0.6367 | 1.7996 +/- 0.8126 | 0.9198 +/- 0.0586 | 122 |
| `rel_0006` | 流行元素 | Full Model | head_noimg | head | 3 | 0.7274 +/- 0.3592 | 1.6414 +/- 0.6964 | 0.4342 +/- 0.0316 | 122 |
| `rel_0006` | 流行元素 | Full Model | head_noimg | tail | 3 | 1.7033 +/- 0.7022 | 1.8078 +/- 0.8144 | 0.9532 +/- 0.0436 | 122 |
| `rel_0006` | 流行元素 | Full Model | tail_noimg | head | 3 | 0.6701 +/- 0.3416 | 1.5930 +/- 0.6472 | 0.4087 +/- 0.0450 | 122 |
| `rel_0006` | 流行元素 | Full Model | tail_noimg | tail | 3 | 1.6483 +/- 0.6554 | 1.8020 +/- 0.8131 | 0.9294 +/- 0.0540 | 122 |
| `rel_0006` | 流行元素 | Residual-only | head_has_img | head | 3 | 0.3580 +/- 0.0116 | 0.0000 +/- 0.0000 | n/a | 122 |
| `rel_0006` | 流行元素 | Residual-only | head_has_img | tail | 3 | 1.4772 +/- 0.0675 | 0.0000 +/- 0.0000 | n/a | 122 |
| `rel_0006` | 流行元素 | Residual-only | head_noimg | head | 3 | 0.3880 +/- 0.0111 | 0.0000 +/- 0.0000 | n/a | 122 |
| `rel_0006` | 流行元素 | Residual-only | head_noimg | tail | 3 | 1.7087 +/- 0.0668 | 0.0000 +/- 0.0000 | n/a | 122 |
| `rel_0006` | 流行元素 | Residual-only | tail_noimg | head | 3 | 0.3666 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 122 |
| `rel_0006` | 流行元素 | Residual-only | tail_noimg | tail | 3 | 1.5437 +/- 0.0672 | 0.0000 +/- 0.0000 | n/a | 122 |
| `rel_0008` | 色泽 | Full Model | head_has_img | head | 3 | 0.6895 +/- 0.3467 | 1.5643 +/- 0.6254 | 0.4288 +/- 0.0475 | 480 |
| `rel_0008` | 色泽 | Full Model | head_has_img | tail | 3 | 1.5568 +/- 0.6892 | 1.8313 +/- 0.8492 | 0.8569 +/- 0.0336 | 480 |
| `rel_0008` | 色泽 | Full Model | head_noimg | head | 3 | 0.7611 +/- 0.3733 | 1.6465 +/- 0.6997 | 0.4529 +/- 0.0333 | 480 |
| `rel_0008` | 色泽 | Full Model | head_noimg | tail | 3 | 1.5578 +/- 0.6873 | 1.8302 +/- 0.8488 | 0.8582 +/- 0.0358 | 480 |
| `rel_0008` | 色泽 | Full Model | tail_noimg | head | 3 | 0.7105 +/- 0.3545 | 1.5884 +/- 0.6473 | 0.4359 +/- 0.0433 | 480 |
| `rel_0008` | 色泽 | Full Model | tail_noimg | tail | 3 | 1.5571 +/- 0.6886 | 1.8310 +/- 0.8491 | 0.8573 +/- 0.0342 | 480 |
| `rel_0008` | 色泽 | Residual-only | head_has_img | head | 3 | 0.3444 +/- 0.0112 | 0.0000 +/- 0.0000 | n/a | 480 |
| `rel_0008` | 色泽 | Residual-only | head_has_img | tail | 3 | 1.7755 +/- 0.0461 | 0.0000 +/- 0.0000 | n/a | 480 |
| `rel_0008` | 色泽 | Residual-only | head_noimg | head | 3 | 0.3523 +/- 0.0118 | 0.0000 +/- 0.0000 | n/a | 480 |
| `rel_0008` | 色泽 | Residual-only | head_noimg | tail | 3 | 1.7664 +/- 0.0445 | 0.0000 +/- 0.0000 | n/a | 480 |
| `rel_0008` | 色泽 | Residual-only | tail_noimg | head | 3 | 0.3467 +/- 0.0114 | 0.0000 +/- 0.0000 | n/a | 480 |
| `rel_0008` | 色泽 | Residual-only | tail_noimg | tail | 3 | 1.7728 +/- 0.0456 | 0.0000 +/- 0.0000 | n/a | 480 |
| `rel_0009` | 裤长 | Full Model | head_has_img | head | 3 | 0.6852 +/- 0.3460 | 1.5696 +/- 0.6307 | 0.4242 +/- 0.0471 | 118 |
| `rel_0009` | 裤长 | Full Model | head_has_img | tail | 3 | 1.7113 +/- 0.6829 | 1.6483 +/- 0.7118 | 1.0520 +/- 0.0589 | 118 |
| `rel_0009` | 裤长 | Full Model | head_noimg | head | 3 | 0.7650 +/- 0.3916 | 1.6392 +/- 0.6996 | 0.4549 +/- 0.0428 | 118 |
| `rel_0009` | 裤长 | Full Model | head_noimg | tail | 3 | 1.7317 +/- 0.6973 | 1.6491 +/- 0.7136 | 1.0640 +/- 0.0550 | 118 |
| `rel_0009` | 裤长 | Full Model | tail_noimg | head | 3 | 0.7041 +/- 0.3568 | 1.5861 +/- 0.6471 | 0.4315 +/- 0.0460 | 118 |
| `rel_0009` | 裤长 | Full Model | tail_noimg | tail | 3 | 1.7162 +/- 0.6863 | 1.6485 +/- 0.7122 | 1.0548 +/- 0.0580 | 118 |
| `rel_0009` | 裤长 | Residual-only | head_has_img | head | 3 | 0.3379 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 118 |
| `rel_0009` | 裤长 | Residual-only | head_has_img | tail | 3 | 1.7291 +/- 0.0550 | 0.0000 +/- 0.0000 | n/a | 118 |
| `rel_0009` | 裤长 | Residual-only | head_noimg | head | 3 | 0.3386 +/- 0.0118 | 0.0000 +/- 0.0000 | n/a | 118 |
| `rel_0009` | 裤长 | Residual-only | head_noimg | tail | 3 | 1.7643 +/- 0.0531 | 0.0000 +/- 0.0000 | n/a | 118 |
| `rel_0009` | 裤长 | Residual-only | tail_noimg | head | 3 | 0.3381 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 118 |
| `rel_0009` | 裤长 | Residual-only | tail_noimg | tail | 3 | 1.7375 +/- 0.0545 | 0.0000 +/- 0.0000 | n/a | 118 |
| `rel_0012` | 是否带帽子 | Full Model | head_has_img | head | 3 | 0.6859 +/- 0.3529 | 1.5430 +/- 0.6108 | 0.4300 +/- 0.0558 | 38 |
| `rel_0012` | 是否带帽子 | Full Model | head_has_img | tail | 3 | 1.7380 +/- 0.6717 | 1.8174 +/- 0.8360 | 0.9806 +/- 0.0861 | 38 |
| `rel_0012` | 是否带帽子 | Full Model | head_noimg | head | 3 | 0.7198 +/- 0.3633 | 1.6236 +/- 0.6764 | 0.4323 +/- 0.0416 | 38 |
| `rel_0012` | 是否带帽子 | Full Model | head_noimg | tail | 3 | 1.7484 +/- 0.6836 | 1.8243 +/- 0.8410 | 0.9812 +/- 0.0787 | 38 |
| `rel_0012` | 是否带帽子 | Full Model | tail_noimg | head | 3 | 0.6957 +/- 0.3559 | 1.5663 +/- 0.6297 | 0.4306 +/- 0.0517 | 38 |
| `rel_0012` | 是否带帽子 | Full Model | tail_noimg | tail | 3 | 1.7410 +/- 0.6751 | 1.8194 +/- 0.8375 | 0.9807 +/- 0.0840 | 38 |
| `rel_0012` | 是否带帽子 | Residual-only | head_has_img | head | 3 | 0.3085 +/- 0.0118 | 0.0000 +/- 0.0000 | n/a | 38 |
| `rel_0012` | 是否带帽子 | Residual-only | head_has_img | tail | 3 | 1.8139 +/- 0.0934 | 0.0000 +/- 0.0000 | n/a | 38 |
| `rel_0012` | 是否带帽子 | Residual-only | head_noimg | head | 3 | 0.3475 +/- 0.0107 | 0.0000 +/- 0.0000 | n/a | 38 |
| `rel_0012` | 是否带帽子 | Residual-only | head_noimg | tail | 3 | 1.8692 +/- 0.0934 | 0.0000 +/- 0.0000 | n/a | 38 |
| `rel_0012` | 是否带帽子 | Residual-only | tail_noimg | head | 3 | 0.3198 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 38 |
| `rel_0012` | 是否带帽子 | Residual-only | tail_noimg | tail | 3 | 1.8299 +/- 0.0934 | 0.0000 +/- 0.0000 | n/a | 38 |
| `rel_0020` | 设计元素 | Full Model | head_has_img | head | 3 | 0.6954 +/- 0.3410 | 1.5591 +/- 0.6066 | 0.4340 +/- 0.0461 | 25 |
| `rel_0020` | 设计元素 | Full Model | head_has_img | tail | 3 | 1.6065 +/- 0.6499 | 1.8029 +/- 0.8157 | 0.9039 +/- 0.0523 | 25 |
| `rel_0020` | 设计元素 | Full Model | head_noimg | head | 3 | 0.7200 +/- 0.3652 | 1.6298 +/- 0.6797 | 0.4308 +/- 0.0411 | 25 |
| `rel_0020` | 设计元素 | Full Model | head_noimg | tail | 3 | 1.7289 +/- 0.6974 | 1.8145 +/- 0.8234 | 0.9698 +/- 0.0683 | 25 |
| `rel_0020` | 设计元素 | Full Model | tail_noimg | head | 3 | 0.7082 +/- 0.3536 | 1.5958 +/- 0.6446 | 0.4323 +/- 0.0435 | 25 |
| `rel_0020` | 设计元素 | Full Model | tail_noimg | tail | 3 | 1.6701 +/- 0.6745 | 1.8090 +/- 0.8197 | 0.9381 +/- 0.0604 | 25 |
| `rel_0020` | 设计元素 | Residual-only | head_has_img | head | 3 | 0.3778 +/- 0.0101 | 0.0000 +/- 0.0000 | n/a | 25 |
| `rel_0020` | 设计元素 | Residual-only | head_has_img | tail | 3 | 1.6916 +/- 0.0566 | 0.0000 +/- 0.0000 | n/a | 25 |
| `rel_0020` | 设计元素 | Residual-only | head_noimg | head | 3 | 0.3798 +/- 0.0111 | 0.0000 +/- 0.0000 | n/a | 25 |
| `rel_0020` | 设计元素 | Residual-only | head_noimg | tail | 3 | 1.9189 +/- 0.0518 | 0.0000 +/- 0.0000 | n/a | 25 |
| `rel_0020` | 设计元素 | Residual-only | tail_noimg | head | 3 | 0.3789 +/- 0.0107 | 0.0000 +/- 0.0000 | n/a | 25 |
| `rel_0020` | 设计元素 | Residual-only | tail_noimg | tail | 3 | 1.8098 +/- 0.0541 | 0.0000 +/- 0.0000 | n/a | 25 |
| `rel_0032` | 领型 | Full Model | head_has_img | head | 3 | 0.6843 +/- 0.3443 | 1.5545 +/- 0.6136 | 0.4275 +/- 0.0494 | 231 |
| `rel_0032` | 领型 | Full Model | head_has_img | tail | 3 | 1.4633 +/- 0.5571 | 1.7826 +/- 0.8087 | 0.8426 +/- 0.0721 | 231 |
| `rel_0032` | 领型 | Full Model | head_noimg | head | 3 | 0.7327 +/- 0.3705 | 1.6260 +/- 0.6832 | 0.4392 +/- 0.0414 | 231 |
| `rel_0032` | 领型 | Full Model | head_noimg | tail | 3 | 1.4736 +/- 0.5483 | 1.7787 +/- 0.8051 | 0.8533 +/- 0.0818 | 231 |
| `rel_0032` | 领型 | Full Model | tail_noimg | head | 3 | 0.6979 +/- 0.3517 | 1.5746 +/- 0.6332 | 0.4308 +/- 0.0471 | 231 |
| `rel_0032` | 领型 | Full Model | tail_noimg | tail | 3 | 1.4662 +/- 0.5546 | 1.7815 +/- 0.8077 | 0.8456 +/- 0.0748 | 231 |
| `rel_0032` | 领型 | Residual-only | head_has_img | head | 3 | 0.3743 +/- 0.0092 | 0.0000 +/- 0.0000 | n/a | 231 |
| `rel_0032` | 领型 | Residual-only | head_has_img | tail | 3 | 1.8623 +/- 0.0444 | 0.0000 +/- 0.0000 | n/a | 231 |
| `rel_0032` | 领型 | Residual-only | head_noimg | head | 3 | 0.3753 +/- 0.0100 | 0.0000 +/- 0.0000 | n/a | 231 |
| `rel_0032` | 领型 | Residual-only | head_noimg | tail | 3 | 1.8929 +/- 0.0423 | 0.0000 +/- 0.0000 | n/a | 231 |
| `rel_0032` | 领型 | Residual-only | tail_noimg | head | 3 | 0.3746 +/- 0.0094 | 0.0000 +/- 0.0000 | n/a | 231 |
| `rel_0032` | 领型 | Residual-only | tail_noimg | tail | 3 | 1.8709 +/- 0.0438 | 0.0000 +/- 0.0000 | n/a | 231 |
| `rel_0038` | 服装版型 | Full Model | head_has_img | head | 3 | 0.6747 +/- 0.3440 | 1.5531 +/- 0.6167 | 0.4214 +/- 0.0502 | 170 |
| `rel_0038` | 服装版型 | Full Model | head_has_img | tail | 3 | 1.9113 +/- 0.8330 | 1.8202 +/- 0.8159 | 1.0574 +/- 0.0296 | 170 |
| `rel_0038` | 服装版型 | Full Model | head_noimg | head | 3 | 0.7239 +/- 0.3625 | 1.6231 +/- 0.6861 | 0.4357 +/- 0.0375 | 170 |
| `rel_0038` | 服装版型 | Full Model | head_noimg | tail | 3 | 1.9394 +/- 0.8364 | 1.8161 +/- 0.8111 | 1.0761 +/- 0.0332 | 170 |
| `rel_0038` | 服装版型 | Full Model | tail_noimg | head | 3 | 0.6892 +/- 0.3494 | 1.5737 +/- 0.6371 | 0.4256 +/- 0.0464 | 170 |
| `rel_0038` | 服装版型 | Full Model | tail_noimg | tail | 3 | 1.9196 +/- 0.8340 | 1.8190 +/- 0.8145 | 1.0629 +/- 0.0307 | 170 |
| `rel_0038` | 服装版型 | Residual-only | head_has_img | head | 3 | 0.3887 +/- 0.0089 | 0.0000 +/- 0.0000 | n/a | 170 |
| `rel_0038` | 服装版型 | Residual-only | head_has_img | tail | 3 | 2.1848 +/- 0.0540 | 0.0000 +/- 0.0000 | n/a | 170 |
| `rel_0038` | 服装版型 | Residual-only | head_noimg | head | 3 | 0.3960 +/- 0.0073 | 0.0000 +/- 0.0000 | n/a | 170 |
| `rel_0038` | 服装版型 | Residual-only | head_noimg | tail | 3 | 2.1993 +/- 0.0553 | 0.0000 +/- 0.0000 | n/a | 170 |
| `rel_0038` | 服装版型 | Residual-only | tail_noimg | head | 3 | 0.3909 +/- 0.0084 | 0.0000 +/- 0.0000 | n/a | 170 |
| `rel_0038` | 服装版型 | Residual-only | tail_noimg | tail | 3 | 2.1891 +/- 0.0544 | 0.0000 +/- 0.0000 | n/a | 170 |
| `rel_0043` | 颜色分类 | Full Model | head_has_img | head | 3 | 0.6795 +/- 0.3464 | 1.5618 +/- 0.6252 | 0.4223 +/- 0.0498 | 341 |
| `rel_0043` | 颜色分类 | Full Model | head_has_img | tail | 3 | 1.5687 +/- 0.6864 | 1.8289 +/- 0.8470 | 0.8666 +/- 0.0454 | 341 |
| `rel_0043` | 颜色分类 | Full Model | head_noimg | head | 3 | 0.7518 +/- 0.3737 | 1.6343 +/- 0.6940 | 0.4503 +/- 0.0368 | 341 |
| `rel_0043` | 颜色分类 | Full Model | head_noimg | tail | 3 | 1.6064 +/- 0.7080 | 1.8350 +/- 0.8508 | 0.8831 +/- 0.0367 | 341 |
| `rel_0043` | 颜色分类 | Full Model | tail_noimg | head | 3 | 0.6975 +/- 0.3532 | 1.5799 +/- 0.6424 | 0.4293 +/- 0.0465 | 341 |
| `rel_0043` | 颜色分类 | Full Model | tail_noimg | tail | 3 | 1.5781 +/- 0.6917 | 1.8304 +/- 0.8479 | 0.8707 +/- 0.0432 | 341 |
| `rel_0043` | 颜色分类 | Residual-only | head_has_img | head | 3 | 0.3461 +/- 0.0109 | 0.0000 +/- 0.0000 | n/a | 341 |
| `rel_0043` | 颜色分类 | Residual-only | head_has_img | tail | 3 | 1.8133 +/- 0.0500 | 0.0000 +/- 0.0000 | n/a | 341 |
| `rel_0043` | 颜色分类 | Residual-only | head_noimg | head | 3 | 0.3556 +/- 0.0113 | 0.0000 +/- 0.0000 | n/a | 341 |
| `rel_0043` | 颜色分类 | Residual-only | head_noimg | tail | 3 | 1.9006 +/- 0.0438 | 0.0000 +/- 0.0000 | n/a | 341 |
| `rel_0043` | 颜色分类 | Residual-only | tail_noimg | head | 3 | 0.3484 +/- 0.0110 | 0.0000 +/- 0.0000 | n/a | 341 |
| `rel_0043` | 颜色分类 | Residual-only | tail_noimg | tail | 3 | 1.8350 +/- 0.0484 | 0.0000 +/- 0.0000 | n/a | 341 |
| `rel_0044` | 闭合方式 | Full Model | head_has_img | head | 3 | 0.6315 +/- 0.3140 | 1.5962 +/- 0.6481 | 0.3860 +/- 0.0369 | 53 |
| `rel_0044` | 闭合方式 | Full Model | head_has_img | tail | 3 | 1.7186 +/- 0.6153 | 1.7811 +/- 0.8022 | 0.9915 +/- 0.0932 | 53 |
| `rel_0044` | 闭合方式 | Full Model | head_noimg | head | 3 | 0.7332 +/- 0.3558 | 1.6665 +/- 0.7161 | 0.4318 +/- 0.0273 | 53 |
| `rel_0044` | 闭合方式 | Full Model | head_noimg | tail | 3 | 1.6831 +/- 0.6379 | 1.7953 +/- 0.8134 | 0.9593 +/- 0.0751 | 53 |
| `rel_0044` | 闭合方式 | Full Model | tail_noimg | head | 3 | 0.6469 +/- 0.3203 | 1.6068 +/- 0.6584 | 0.3929 +/- 0.0353 | 53 |
| `rel_0044` | 闭合方式 | Full Model | tail_noimg | tail | 3 | 1.7133 +/- 0.6187 | 1.7832 +/- 0.8039 | 0.9867 +/- 0.0904 | 53 |
| `rel_0044` | 闭合方式 | Residual-only | head_has_img | head | 3 | 0.3070 +/- 0.0121 | 0.0000 +/- 0.0000 | n/a | 53 |
| `rel_0044` | 闭合方式 | Residual-only | head_has_img | tail | 3 | 1.6120 +/- 0.0966 | 0.0000 +/- 0.0000 | n/a | 53 |
| `rel_0044` | 闭合方式 | Residual-only | head_noimg | head | 3 | 0.3422 +/- 0.0107 | 0.0000 +/- 0.0000 | n/a | 53 |
| `rel_0044` | 闭合方式 | Residual-only | head_noimg | tail | 3 | 1.6387 +/- 0.0995 | 0.0000 +/- 0.0000 | n/a | 53 |
| `rel_0044` | 闭合方式 | Residual-only | tail_noimg | head | 3 | 0.3123 +/- 0.0119 | 0.0000 +/- 0.0000 | n/a | 53 |
| `rel_0044` | 闭合方式 | Residual-only | tail_noimg | tail | 3 | 1.6160 +/- 0.0970 | 0.0000 +/- 0.0000 | n/a | 53 |
| `rel_0048` | 设计细节 | Full Model | head_has_img | head | 3 | 0.6633 +/- 0.3389 | 1.5618 +/- 0.6184 | 0.4118 +/- 0.0504 | 234 |
| `rel_0048` | 设计细节 | Full Model | head_has_img | tail | 3 | 1.6612 +/- 0.6543 | 1.7969 +/- 0.8133 | 0.9412 +/- 0.0573 | 234 |
| `rel_0048` | 设计细节 | Full Model | head_noimg | head | 3 | 0.7493 +/- 0.3778 | 1.6338 +/- 0.6852 | 0.4474 +/- 0.0416 | 234 |
| `rel_0048` | 设计细节 | Full Model | head_noimg | tail | 3 | 1.6645 +/- 0.6478 | 1.7995 +/- 0.8176 | 0.9423 +/- 0.0619 | 234 |
| `rel_0048` | 设计细节 | Full Model | tail_noimg | head | 3 | 0.6872 +/- 0.3497 | 1.5818 +/- 0.6370 | 0.4217 +/- 0.0479 | 234 |
| `rel_0048` | 设计细节 | Full Model | tail_noimg | tail | 3 | 1.6621 +/- 0.6525 | 1.7976 +/- 0.8145 | 0.9415 +/- 0.0585 | 234 |
| `rel_0048` | 设计细节 | Residual-only | head_has_img | head | 3 | 0.3838 +/- 0.0108 | 0.0000 +/- 0.0000 | n/a | 234 |
| `rel_0048` | 设计细节 | Residual-only | head_has_img | tail | 3 | 1.6341 +/- 0.0666 | 0.0000 +/- 0.0000 | n/a | 234 |
| `rel_0048` | 设计细节 | Residual-only | head_noimg | head | 3 | 0.3780 +/- 0.0126 | 0.0000 +/- 0.0000 | n/a | 234 |
| `rel_0048` | 设计细节 | Residual-only | head_noimg | tail | 3 | 1.6278 +/- 0.0693 | 0.0000 +/- 0.0000 | n/a | 234 |
| `rel_0048` | 设计细节 | Residual-only | tail_noimg | head | 3 | 0.3822 +/- 0.0113 | 0.0000 +/- 0.0000 | n/a | 234 |
| `rel_0048` | 设计细节 | Residual-only | tail_noimg | tail | 3 | 1.6324 +/- 0.0674 | 0.0000 +/- 0.0000 | n/a | 234 |
| `rel_0062` | 裙长 | Full Model | head_has_img | head | 3 | 0.6756 +/- 0.3488 | 1.5634 +/- 0.6083 | 0.4175 +/- 0.0573 | 66 |
| `rel_0062` | 裙长 | Full Model | head_has_img | tail | 3 | 1.6843 +/- 0.5582 | 1.7385 +/- 0.7617 | 1.0070 +/- 0.1181 | 66 |
| `rel_0062` | 裙长 | Full Model | head_noimg | head | 3 | 0.7156 +/- 0.3571 | 1.6276 +/- 0.6763 | 0.4290 +/- 0.0410 | 66 |
| `rel_0062` | 裙长 | Full Model | head_noimg | tail | 3 | 1.6646 +/- 0.5567 | 1.7473 +/- 0.7683 | 0.9888 +/- 0.1195 | 66 |
| `rel_0062` | 裙长 | Full Model | tail_noimg | head | 3 | 0.6884 +/- 0.3514 | 1.5838 +/- 0.6299 | 0.4211 +/- 0.0520 | 66 |
| `rel_0062` | 裙长 | Full Model | tail_noimg | tail | 3 | 1.6780 +/- 0.5577 | 1.7413 +/- 0.7638 | 1.0012 +/- 0.1185 | 66 |
| `rel_0062` | 裙长 | Residual-only | head_has_img | head | 3 | 0.3799 +/- 0.0117 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0062` | 裙长 | Residual-only | head_has_img | tail | 3 | 1.8406 +/- 0.0885 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0062` | 裙长 | Residual-only | head_noimg | head | 3 | 0.3947 +/- 0.0116 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0062` | 裙长 | Residual-only | head_noimg | tail | 3 | 1.7566 +/- 0.0857 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0062` | 裙长 | Residual-only | tail_noimg | head | 3 | 0.3846 +/- 0.0116 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0062` | 裙长 | Residual-only | tail_noimg | tail | 3 | 1.8139 +/- 0.0876 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0064` | 细分风格 | Full Model | head_has_img | head | 3 | 0.6722 +/- 0.3420 | 1.5426 +/- 0.6158 | 0.4229 +/- 0.0488 | 52 |
| `rel_0064` | 细分风格 | Full Model | head_has_img | tail | 3 | 1.4525 +/- 0.5973 | 1.8103 +/- 0.8267 | 0.8137 +/- 0.0385 | 52 |
| `rel_0064` | 细分风格 | Full Model | head_noimg | head | 3 | 0.7161 +/- 0.3524 | 1.6339 +/- 0.6939 | 0.4296 +/- 0.0330 | 52 |
| `rel_0064` | 细分风格 | Full Model | head_noimg | tail | 3 | 1.3968 +/- 0.5644 | 1.8069 +/- 0.8241 | 0.7861 +/- 0.0441 | 52 |
| `rel_0064` | 细分风格 | Full Model | tail_noimg | head | 3 | 0.6883 +/- 0.3457 | 1.5760 +/- 0.6443 | 0.4254 +/- 0.0429 | 52 |
| `rel_0064` | 细分风格 | Full Model | tail_noimg | tail | 3 | 1.4322 +/- 0.5853 | 1.8091 +/- 0.8257 | 0.8036 +/- 0.0405 | 52 |
| `rel_0064` | 细分风格 | Residual-only | head_has_img | head | 3 | 0.4100 +/- 0.0081 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0064` | 细分风格 | Residual-only | head_has_img | tail | 3 | 1.6832 +/- 0.0766 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0064` | 细分风格 | Residual-only | head_noimg | head | 3 | 0.3923 +/- 0.0073 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0064` | 细分风格 | Residual-only | head_noimg | tail | 3 | 1.4888 +/- 0.0754 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0064` | 细分风格 | Residual-only | tail_noimg | head | 3 | 0.4036 +/- 0.0078 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0064` | 细分风格 | Residual-only | tail_noimg | tail | 3 | 1.6122 +/- 0.0761 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0065` | 腰型 | Full Model | head_has_img | head | 3 | 0.6798 +/- 0.3423 | 1.5689 +/- 0.6236 | 0.4210 +/- 0.0471 | 104 |
| `rel_0065` | 腰型 | Full Model | head_has_img | tail | 3 | 1.4459 +/- 0.5973 | 1.8171 +/- 0.8284 | 0.8056 +/- 0.0343 | 104 |
| `rel_0065` | 腰型 | Full Model | head_noimg | head | 3 | 0.7198 +/- 0.3697 | 1.6370 +/- 0.6901 | 0.4274 +/- 0.0452 | 104 |
| `rel_0065` | 腰型 | Full Model | head_noimg | tail | 3 | 1.4452 +/- 0.5900 | 1.8180 +/- 0.8289 | 0.8061 +/- 0.0380 | 104 |
| `rel_0065` | 腰型 | Full Model | tail_noimg | head | 3 | 0.6921 +/- 0.3507 | 1.5898 +/- 0.6440 | 0.4230 +/- 0.0463 | 104 |
| `rel_0065` | 腰型 | Full Model | tail_noimg | tail | 3 | 1.4457 +/- 0.5950 | 1.8174 +/- 0.8285 | 0.8058 +/- 0.0355 | 104 |
| `rel_0065` | 腰型 | Residual-only | head_has_img | head | 3 | 0.3364 +/- 0.0103 | 0.0000 +/- 0.0000 | n/a | 104 |
| `rel_0065` | 腰型 | Residual-only | head_has_img | tail | 3 | 1.9059 +/- 0.0355 | 0.0000 +/- 0.0000 | n/a | 104 |
| `rel_0065` | 腰型 | Residual-only | head_noimg | head | 3 | 0.3314 +/- 0.0108 | 0.0000 +/- 0.0000 | n/a | 104 |
| `rel_0065` | 腰型 | Residual-only | head_noimg | tail | 3 | 2.0291 +/- 0.0350 | 0.0000 +/- 0.0000 | n/a | 104 |
| `rel_0065` | 腰型 | Residual-only | tail_noimg | head | 3 | 0.3349 +/- 0.0105 | 0.0000 +/- 0.0000 | n/a | 104 |
| `rel_0065` | 腰型 | Residual-only | tail_noimg | tail | 3 | 1.9438 +/- 0.0353 | 0.0000 +/- 0.0000 | n/a | 104 |
| `rel_0066` | 袖型 | Full Model | head_has_img | head | 3 | 0.6867 +/- 0.3501 | 1.5535 +/- 0.6098 | 0.4282 +/- 0.0535 | 130 |
| `rel_0066` | 袖型 | Full Model | head_has_img | tail | 3 | 1.9935 +/- 0.8948 | 1.8169 +/- 0.8273 | 1.0973 +/- 0.0098 | 130 |
| `rel_0066` | 袖型 | Full Model | head_noimg | head | 3 | 0.7565 +/- 0.3819 | 1.6245 +/- 0.6825 | 0.4541 +/- 0.0416 | 130 |
| `rel_0066` | 袖型 | Full Model | head_noimg | tail | 3 | 1.9327 +/- 0.8814 | 1.8201 +/- 0.8300 | 1.0595 +/- 0.0035 | 130 |
| `rel_0066` | 袖型 | Full Model | tail_noimg | head | 3 | 0.7055 +/- 0.3587 | 1.5726 +/- 0.6294 | 0.4352 +/- 0.0502 | 130 |
| `rel_0066` | 袖型 | Full Model | tail_noimg | tail | 3 | 1.9771 +/- 0.8912 | 1.8178 +/- 0.8280 | 1.0871 +/- 0.0078 | 130 |
| `rel_0066` | 袖型 | Residual-only | head_has_img | head | 3 | 0.4113 +/- 0.0109 | 0.0000 +/- 0.0000 | n/a | 130 |
| `rel_0066` | 袖型 | Residual-only | head_has_img | tail | 3 | 2.3156 +/- 0.0484 | 0.0000 +/- 0.0000 | n/a | 130 |
| `rel_0066` | 袖型 | Residual-only | head_noimg | head | 3 | 0.4043 +/- 0.0090 | 0.0000 +/- 0.0000 | n/a | 130 |
| `rel_0066` | 袖型 | Residual-only | head_noimg | tail | 3 | 2.2579 +/- 0.0488 | 0.0000 +/- 0.0000 | n/a | 130 |
| `rel_0066` | 袖型 | Residual-only | tail_noimg | head | 3 | 0.4094 +/- 0.0104 | 0.0000 +/- 0.0000 | n/a | 130 |
| `rel_0066` | 袖型 | Residual-only | tail_noimg | tail | 3 | 2.3000 +/- 0.0485 | 0.0000 +/- 0.0000 | n/a | 130 |
| `rel_0086` | 样式 | Full Model | head_has_img | head | 3 | 0.6797 +/- 0.3445 | 1.5658 +/- 0.6229 | 0.4216 +/- 0.0486 | 518 |
| `rel_0086` | 样式 | Full Model | head_has_img | tail | 3 | 1.5291 +/- 0.6186 | 1.7778 +/- 0.7938 | 0.8746 +/- 0.0420 | 518 |
| `rel_0086` | 样式 | Full Model | head_noimg | head | 3 | 0.7631 +/- 0.3769 | 1.6429 +/- 0.6956 | 0.4550 +/- 0.0350 | 518 |
| `rel_0086` | 样式 | Full Model | head_noimg | tail | 3 | 1.5637 +/- 0.6329 | 1.7870 +/- 0.8014 | 0.8894 +/- 0.0441 | 518 |
| `rel_0086` | 样式 | Full Model | tail_noimg | head | 3 | 0.7032 +/- 0.3537 | 1.5875 +/- 0.6434 | 0.4310 +/- 0.0447 | 518 |
| `rel_0086` | 样式 | Full Model | tail_noimg | tail | 3 | 1.5389 +/- 0.6226 | 1.7804 +/- 0.7959 | 0.8787 +/- 0.0426 | 518 |
| `rel_0086` | 样式 | Residual-only | head_has_img | head | 3 | 0.3578 +/- 0.0112 | 0.0000 +/- 0.0000 | n/a | 518 |
| `rel_0086` | 样式 | Residual-only | head_has_img | tail | 3 | 1.7420 +/- 0.0459 | 0.0000 +/- 0.0000 | n/a | 518 |
| `rel_0086` | 样式 | Residual-only | head_noimg | head | 3 | 0.3426 +/- 0.0116 | 0.0000 +/- 0.0000 | n/a | 518 |
| `rel_0086` | 样式 | Residual-only | head_noimg | tail | 3 | 1.7944 +/- 0.0480 | 0.0000 +/- 0.0000 | n/a | 518 |
| `rel_0086` | 样式 | Residual-only | tail_noimg | head | 3 | 0.3535 +/- 0.0113 | 0.0000 +/- 0.0000 | n/a | 518 |
| `rel_0086` | 样式 | Residual-only | tail_noimg | tail | 3 | 1.7568 +/- 0.0465 | 0.0000 +/- 0.0000 | n/a | 518 |
| `rel_0098` | 佩戴方式 | Full Model | head_has_img | head | 3 | 0.6799 +/- 0.3434 | 1.5536 +/- 0.6164 | 0.4252 +/- 0.0485 | 68 |
| `rel_0098` | 佩戴方式 | Full Model | head_has_img | tail | 3 | 2.1135 +/- 0.8676 | 1.7781 +/- 0.7936 | 1.2039 +/- 0.0605 | 68 |
| `rel_0098` | 佩戴方式 | Full Model | head_noimg | head | 3 | 0.7780 +/- 0.3907 | 1.6212 +/- 0.6854 | 0.4683 +/- 0.0430 | 68 |
| `rel_0098` | 佩戴方式 | Full Model | head_noimg | tail | 3 | 2.1146 +/- 0.8675 | 1.7703 +/- 0.7871 | 1.2101 +/- 0.0657 | 68 |
| `rel_0098` | 佩戴方式 | Full Model | tail_noimg | head | 3 | 0.7059 +/- 0.3559 | 1.5715 +/- 0.6347 | 0.4366 +/- 0.0468 | 68 |
| `rel_0098` | 佩戴方式 | Full Model | tail_noimg | tail | 3 | 2.1138 +/- 0.8675 | 1.7761 +/- 0.7919 | 1.2056 +/- 0.0618 | 68 |
| `rel_0098` | 佩戴方式 | Residual-only | head_has_img | head | 3 | 0.4014 +/- 0.0103 | 0.0000 +/- 0.0000 | n/a | 68 |
| `rel_0098` | 佩戴方式 | Residual-only | head_has_img | tail | 3 | 2.2353 +/- 0.0566 | 0.0000 +/- 0.0000 | n/a | 68 |
| `rel_0098` | 佩戴方式 | Residual-only | head_noimg | head | 3 | 0.4017 +/- 0.0093 | 0.0000 +/- 0.0000 | n/a | 68 |
| `rel_0098` | 佩戴方式 | Residual-only | head_noimg | tail | 3 | 2.2462 +/- 0.0566 | 0.0000 +/- 0.0000 | n/a | 68 |
| `rel_0098` | 佩戴方式 | Residual-only | tail_noimg | head | 3 | 0.4015 +/- 0.0101 | 0.0000 +/- 0.0000 | n/a | 68 |
| `rel_0098` | 佩戴方式 | Residual-only | tail_noimg | tail | 3 | 2.2382 +/- 0.0566 | 0.0000 +/- 0.0000 | n/a | 68 |
| `rel_0101` | 裤门襟 | Full Model | head_has_img | head | 3 | 0.6892 +/- 0.3491 | 1.5686 +/- 0.6296 | 0.4270 +/- 0.0472 | 52 |
| `rel_0101` | 裤门襟 | Full Model | head_has_img | tail | 3 | 1.8334 +/- 0.6404 | 1.7662 +/- 0.7861 | 1.0684 +/- 0.1006 | 52 |
| `rel_0101` | 裤门襟 | Full Model | head_noimg | head | 3 | 0.7490 +/- 0.3784 | 1.6397 +/- 0.7047 | 0.4464 +/- 0.0369 | 52 |
| `rel_0101` | 裤门襟 | Full Model | head_noimg | tail | 3 | 1.7832 +/- 0.6370 | 1.7874 +/- 0.8021 | 1.0254 +/- 0.0929 | 52 |
| `rel_0101` | 裤门襟 | Full Model | tail_noimg | head | 3 | 0.6949 +/- 0.3519 | 1.5754 +/- 0.6368 | 0.4288 +/- 0.0462 | 52 |
| `rel_0101` | 裤门襟 | Full Model | tail_noimg | tail | 3 | 1.8286 +/- 0.6400 | 1.7682 +/- 0.7876 | 1.0643 +/- 0.0998 | 52 |
| `rel_0101` | 裤门襟 | Residual-only | head_has_img | head | 3 | 0.3532 +/- 0.0128 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0101` | 裤门襟 | Residual-only | head_has_img | tail | 3 | 1.7042 +/- 0.0902 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0101` | 裤门襟 | Residual-only | head_noimg | head | 3 | 0.3587 +/- 0.0109 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0101` | 裤门襟 | Residual-only | head_noimg | tail | 3 | 1.6778 +/- 0.0913 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0101` | 裤门襟 | Residual-only | tail_noimg | head | 3 | 0.3537 +/- 0.0126 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0101` | 裤门襟 | Residual-only | tail_noimg | tail | 3 | 1.7016 +/- 0.0903 | 0.0000 +/- 0.0000 | n/a | 52 |
| `rel_0106` | 袖长 | Full Model | head_has_img | head | 3 | 0.6747 +/- 0.3471 | 1.5515 +/- 0.6129 | 0.4211 +/- 0.0533 | 230 |
| `rel_0106` | 袖长 | Full Model | head_has_img | tail | 3 | 1.5917 +/- 0.7189 | 1.7586 +/- 0.8027 | 0.9063 +/- 0.0113 | 230 |
| `rel_0106` | 袖长 | Full Model | head_noimg | head | 3 | 0.7629 +/- 0.3826 | 1.6279 +/- 0.6844 | 0.4571 +/- 0.0407 | 230 |
| `rel_0106` | 袖长 | Full Model | head_noimg | tail | 3 | 1.5564 +/- 0.6863 | 1.7677 +/- 0.8101 | 0.8848 +/- 0.0170 | 230 |
| `rel_0106` | 袖长 | Full Model | tail_noimg | head | 3 | 0.6962 +/- 0.3558 | 1.5701 +/- 0.6303 | 0.4299 +/- 0.0501 | 230 |
| `rel_0106` | 袖长 | Full Model | tail_noimg | tail | 3 | 1.5831 +/- 0.7110 | 1.7608 +/- 0.8045 | 0.9010 +/- 0.0120 | 230 |
| `rel_0106` | 袖长 | Residual-only | head_has_img | head | 3 | 0.3731 +/- 0.0100 | 0.0000 +/- 0.0000 | n/a | 230 |
| `rel_0106` | 袖长 | Residual-only | head_has_img | tail | 3 | 2.0638 +/- 0.0534 | 0.0000 +/- 0.0000 | n/a | 230 |
| `rel_0106` | 袖长 | Residual-only | head_noimg | head | 3 | 0.3388 +/- 0.0099 | 0.0000 +/- 0.0000 | n/a | 230 |
| `rel_0106` | 袖长 | Residual-only | head_noimg | tail | 3 | 2.0614 +/- 0.0602 | 0.0000 +/- 0.0000 | n/a | 230 |
| `rel_0106` | 袖长 | Residual-only | tail_noimg | head | 3 | 0.3647 +/- 0.0100 | 0.0000 +/- 0.0000 | n/a | 230 |
| `rel_0106` | 袖长 | Residual-only | tail_noimg | tail | 3 | 2.0632 +/- 0.0551 | 0.0000 +/- 0.0000 | n/a | 230 |
| `rel_0107` | 衣门襟 | Full Model | head_has_img | head | 3 | 0.6902 +/- 0.3564 | 1.5530 +/- 0.6126 | 0.4300 +/- 0.0564 | 132 |
| `rel_0107` | 衣门襟 | Full Model | head_has_img | tail | 3 | 1.9603 +/- 0.7789 | 1.8071 +/- 0.8124 | 1.1021 +/- 0.0574 | 132 |
| `rel_0107` | 衣门襟 | Full Model | head_noimg | head | 3 | 0.7427 +/- 0.3727 | 1.6274 +/- 0.6829 | 0.4452 +/- 0.0417 | 132 |
| `rel_0107` | 衣门襟 | Full Model | head_noimg | tail | 3 | 1.8786 +/- 0.7630 | 1.8033 +/- 0.8092 | 1.0547 +/- 0.0454 | 132 |
| `rel_0107` | 衣门襟 | Full Model | tail_noimg | head | 3 | 0.7093 +/- 0.3623 | 1.5801 +/- 0.6381 | 0.4356 +/- 0.0509 | 132 |
| `rel_0107` | 衣门襟 | Full Model | tail_noimg | tail | 3 | 1.9306 +/- 0.7731 | 1.8057 +/- 0.8112 | 1.0848 +/- 0.0531 | 132 |
| `rel_0107` | 衣门襟 | Residual-only | head_has_img | head | 3 | 0.3938 +/- 0.0098 | 0.0000 +/- 0.0000 | n/a | 132 |
| `rel_0107` | 衣门襟 | Residual-only | head_has_img | tail | 3 | 2.0779 +/- 0.0713 | 0.0000 +/- 0.0000 | n/a | 132 |
| `rel_0107` | 衣门襟 | Residual-only | head_noimg | head | 3 | 0.3955 +/- 0.0111 | 0.0000 +/- 0.0000 | n/a | 132 |
| `rel_0107` | 衣门襟 | Residual-only | head_noimg | tail | 3 | 1.9868 +/- 0.0611 | 0.0000 +/- 0.0000 | n/a | 132 |
| `rel_0107` | 衣门襟 | Residual-only | tail_noimg | head | 3 | 0.3944 +/- 0.0102 | 0.0000 +/- 0.0000 | n/a | 132 |
| `rel_0107` | 衣门襟 | Residual-only | tail_noimg | tail | 3 | 2.0448 +/- 0.0676 | 0.0000 +/- 0.0000 | n/a | 132 |
| `rel_0109` | 版型 | Full Model | head_has_img | head | 3 | 0.6618 +/- 0.3348 | 1.5604 +/- 0.6172 | 0.4113 +/- 0.0492 | 137 |
| `rel_0109` | 版型 | Full Model | head_has_img | tail | 3 | 1.7848 +/- 0.7624 | 1.8328 +/- 0.8306 | 0.9840 +/- 0.0361 | 137 |
| `rel_0109` | 版型 | Full Model | head_noimg | head | 3 | 0.7242 +/- 0.3607 | 1.6305 +/- 0.6913 | 0.4340 +/- 0.0360 | 137 |
| `rel_0109` | 版型 | Full Model | head_noimg | tail | 3 | 1.8207 +/- 0.7923 | 1.8243 +/- 0.8249 | 1.0051 +/- 0.0262 | 137 |
| `rel_0109` | 版型 | Full Model | tail_noimg | head | 3 | 0.6814 +/- 0.3429 | 1.5824 +/- 0.6404 | 0.4184 +/- 0.0450 | 137 |
| `rel_0109` | 版型 | Full Model | tail_noimg | tail | 3 | 1.7961 +/- 0.7718 | 1.8301 +/- 0.8288 | 0.9906 +/- 0.0330 | 137 |
| `rel_0109` | 版型 | Residual-only | head_has_img | head | 3 | 0.3991 +/- 0.0099 | 0.0000 +/- 0.0000 | n/a | 137 |
| `rel_0109` | 版型 | Residual-only | head_has_img | tail | 3 | 2.2040 +/- 0.0539 | 0.0000 +/- 0.0000 | n/a | 137 |
| `rel_0109` | 版型 | Residual-only | head_noimg | head | 3 | 0.3633 +/- 0.0090 | 0.0000 +/- 0.0000 | n/a | 137 |
| `rel_0109` | 版型 | Residual-only | head_noimg | tail | 3 | 2.0446 +/- 0.0509 | 0.0000 +/- 0.0000 | n/a | 137 |
| `rel_0109` | 版型 | Residual-only | tail_noimg | head | 3 | 0.3879 +/- 0.0096 | 0.0000 +/- 0.0000 | n/a | 137 |
| `rel_0109` | 版型 | Residual-only | tail_noimg | tail | 3 | 2.1539 +/- 0.0530 | 0.0000 +/- 0.0000 | n/a | 137 |
| `rel_0117` | 衣长 | Full Model | head_has_img | head | 3 | 0.6860 +/- 0.3497 | 1.5566 +/- 0.6145 | 0.4270 +/- 0.0525 | 112 |
| `rel_0117` | 衣长 | Full Model | head_has_img | tail | 3 | 1.7960 +/- 0.7620 | 1.8316 +/- 0.8401 | 0.9916 +/- 0.0422 | 112 |
| `rel_0117` | 衣长 | Full Model | head_noimg | head | 3 | 0.7373 +/- 0.3784 | 1.6221 +/- 0.6854 | 0.4425 +/- 0.0436 | 112 |
| `rel_0117` | 衣长 | Full Model | head_noimg | tail | 3 | 1.8369 +/- 0.7757 | 1.8408 +/- 0.8458 | 1.0105 +/- 0.0457 | 112 |
| `rel_0117` | 衣长 | Full Model | tail_noimg | head | 3 | 0.7048 +/- 0.3602 | 1.5806 +/- 0.6405 | 0.4327 +/- 0.0492 | 112 |
| `rel_0117` | 衣长 | Full Model | tail_noimg | tail | 3 | 1.8110 +/- 0.7670 | 1.8350 +/- 0.8422 | 0.9985 +/- 0.0435 | 112 |
| `rel_0117` | 衣长 | Residual-only | head_has_img | head | 3 | 0.4028 +/- 0.0102 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0117` | 衣长 | Residual-only | head_has_img | tail | 3 | 2.0628 +/- 0.0680 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0117` | 衣长 | Residual-only | head_noimg | head | 3 | 0.3916 +/- 0.0097 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0117` | 衣长 | Residual-only | head_noimg | tail | 3 | 2.2080 +/- 0.0673 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0117` | 衣长 | Residual-only | tail_noimg | head | 3 | 0.3987 +/- 0.0100 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0117` | 衣长 | Residual-only | tail_noimg | tail | 3 | 2.1160 +/- 0.0678 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0123` | 裙型 | Full Model | head_has_img | head | 3 | 0.6738 +/- 0.3432 | 1.5600 +/- 0.6078 | 0.4186 +/- 0.0529 | 58 |
| `rel_0123` | 裙型 | Full Model | head_has_img | tail | 3 | 1.5201 +/- 0.5704 | 1.7117 +/- 0.7401 | 0.9020 +/- 0.0606 | 58 |
| `rel_0123` | 裙型 | Full Model | head_noimg | head | 3 | 0.7323 +/- 0.3623 | 1.6302 +/- 0.6751 | 0.4388 +/- 0.0385 | 58 |
| `rel_0123` | 裙型 | Full Model | head_noimg | tail | 3 | 1.5651 +/- 0.5826 | 1.7098 +/- 0.7405 | 0.9331 +/- 0.0712 | 58 |
| `rel_0123` | 裙型 | Full Model | tail_noimg | head | 3 | 0.6960 +/- 0.3504 | 1.5866 +/- 0.6333 | 0.4263 +/- 0.0474 | 58 |
| `rel_0123` | 裙型 | Full Model | tail_noimg | tail | 3 | 1.5372 +/- 0.5750 | 1.7110 +/- 0.7402 | 0.9138 +/- 0.0645 | 58 |
| `rel_0123` | 裙型 | Residual-only | head_has_img | head | 3 | 0.3971 +/- 0.0105 | 0.0000 +/- 0.0000 | n/a | 58 |
| `rel_0123` | 裙型 | Residual-only | head_has_img | tail | 3 | 1.3730 +/- 0.0651 | 0.0000 +/- 0.0000 | n/a | 58 |
| `rel_0123` | 裙型 | Residual-only | head_noimg | head | 3 | 0.3960 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 58 |
| `rel_0123` | 裙型 | Residual-only | head_noimg | tail | 3 | 1.3663 +/- 0.0629 | 0.0000 +/- 0.0000 | n/a | 58 |
| `rel_0123` | 裙型 | Residual-only | tail_noimg | head | 3 | 0.3967 +/- 0.0108 | 0.0000 +/- 0.0000 | n/a | 58 |
| `rel_0123` | 裙型 | Residual-only | tail_noimg | tail | 3 | 1.3705 +/- 0.0642 | 0.0000 +/- 0.0000 | n/a | 58 |
| `rel_0134` | 基础风格 | Full Model | head_has_img | head | 3 | 0.6699 +/- 0.3431 | 1.5518 +/- 0.6195 | 0.4189 +/- 0.0503 | 55 |
| `rel_0134` | 基础风格 | Full Model | head_has_img | tail | 3 | 1.5987 +/- 0.6403 | 1.7967 +/- 0.8276 | 0.9086 +/- 0.0667 | 55 |
| `rel_0134` | 基础风格 | Full Model | head_noimg | head | 3 | 0.7058 +/- 0.3556 | 1.6299 +/- 0.6891 | 0.4225 +/- 0.0391 | 55 |
| `rel_0134` | 基础风格 | Full Model | head_noimg | tail | 3 | 1.5980 +/- 0.6470 | 1.8059 +/- 0.8346 | 0.9028 +/- 0.0627 | 55 |
| `rel_0134` | 基础风格 | Full Model | tail_noimg | head | 3 | 0.6797 +/- 0.3465 | 1.5731 +/- 0.6385 | 0.4199 +/- 0.0471 | 55 |
| `rel_0134` | 基础风格 | Full Model | tail_noimg | tail | 3 | 1.5985 +/- 0.6421 | 1.7992 +/- 0.8295 | 0.9070 +/- 0.0656 | 55 |
| `rel_0134` | 基础风格 | Residual-only | head_has_img | head | 3 | 0.3684 +/- 0.0109 | 0.0000 +/- 0.0000 | n/a | 55 |
| `rel_0134` | 基础风格 | Residual-only | head_has_img | tail | 3 | 1.9294 +/- 0.0617 | 0.0000 +/- 0.0000 | n/a | 55 |
| `rel_0134` | 基础风格 | Residual-only | head_noimg | head | 3 | 0.3953 +/- 0.0070 | 0.0000 +/- 0.0000 | n/a | 55 |
| `rel_0134` | 基础风格 | Residual-only | head_noimg | tail | 3 | 1.9209 +/- 0.0638 | 0.0000 +/- 0.0000 | n/a | 55 |
| `rel_0134` | 基础风格 | Residual-only | tail_noimg | head | 3 | 0.3757 +/- 0.0099 | 0.0000 +/- 0.0000 | n/a | 55 |
| `rel_0134` | 基础风格 | Residual-only | tail_noimg | tail | 3 | 1.9271 +/- 0.0622 | 0.0000 +/- 0.0000 | n/a | 55 |

## 4.2 `weak_visual_relations`


### head_has_img

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.8853 +/- 0.4901 | 0.6909 +/- 0.3490 | 7.4595 +/- 1.5666 | 1.5704 +/- 0.6310 | 0.4279 +/- 0.0470 | -0.8796 +/- 0.2823 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.0508 +/- 0.9570 | 1.6050 +/- 0.6693 | 8.5720 +/- 2.2915 | 1.8178 +/- 0.8303 | 0.8940 +/- 0.0387 | -0.2128 +/- 0.1610 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3407 +/- 0.0122 | 0.3407 +/- 0.0122 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3407 +/- 0.0122 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 2.0065 +/- 0.0456 | 2.0065 +/- 0.0456 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 2.0065 +/- 0.0456 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### head_noimg

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 1.0210 +/- 0.5318 | 0.7976 +/- 0.3766 | 7.8700 +/- 1.8682 | 1.6626 +/- 0.7122 | 0.4733 +/- 0.0229 | -0.8650 +/- 0.3357 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 1.9241 +/- 0.9003 | 1.5058 +/- 0.6298 | 8.5403 +/- 2.2706 | 1.8107 +/- 0.8248 | 0.8397 +/- 0.0351 | -0.3049 +/- 0.1951 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3882 +/- 0.0154 | 0.3882 +/- 0.0154 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3882 +/- 0.0154 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.8128 +/- 0.0450 | 1.8128 +/- 0.0450 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.8128 +/- 0.0450 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### tail_noimg

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.9272 +/- 0.5029 | 0.7238 +/- 0.3575 | 7.5862 +/- 1.6596 | 1.5989 +/- 0.6560 | 0.4419 +/- 0.0395 | -0.8751 +/- 0.2987 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.0117 +/- 0.9395 | 1.5744 +/- 0.6571 | 8.5622 +/- 2.2850 | 1.8156 +/- 0.8286 | 0.8773 +/- 0.0376 | -0.2412 +/- 0.1715 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3553 +/- 0.0132 | 0.3553 +/- 0.0132 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3553 +/- 0.0132 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.9467 +/- 0.0454 | 1.9467 +/- 0.0454 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.9467 +/- 0.0454 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### target_has_img

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.8853 +/- 0.4901 | 0.6909 +/- 0.3490 | 7.4595 +/- 1.5666 | 1.5704 +/- 0.6310 | 0.4279 +/- 0.0470 | -0.8796 +/- 0.2823 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.0508 +/- 0.9570 | 1.6050 +/- 0.6693 | 8.5720 +/- 2.2915 | 1.8178 +/- 0.8303 | 0.8940 +/- 0.0387 | -0.2128 +/- 0.1610 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3407 +/- 0.0122 | 0.3407 +/- 0.0122 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3407 +/- 0.0122 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 2.0065 +/- 0.0456 | 2.0065 +/- 0.0456 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 2.0065 +/- 0.0456 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### target_noimg

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.9493 +/- 0.5097 | 0.7412 +/- 0.3620 | 7.6531 +/- 1.7088 | 1.6139 +/- 0.6693 | 0.4493 +/- 0.0356 | -0.8727 +/- 0.3074 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 1.9910 +/- 0.9302 | 1.5582 +/- 0.6507 | 8.5571 +/- 2.2816 | 1.8144 +/- 0.8277 | 0.8684 +/- 0.0370 | -0.2562 +/- 0.1771 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3631 +/- 0.0137 | 0.3631 +/- 0.0137 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3631 +/- 0.0137 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.9151 +/- 0.0453 | 1.9151 +/- 0.0453 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.9151 +/- 0.0453 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### all_targets

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.9272 +/- 0.5029 | 0.7238 +/- 0.3575 | 7.5862 +/- 1.6596 | 1.5989 +/- 0.6560 | 0.4419 +/- 0.0395 | -0.8751 +/- 0.2987 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.0117 +/- 0.9395 | 1.5744 +/- 0.6571 | 8.5622 +/- 2.2850 | 1.8156 +/- 0.8286 | 0.8773 +/- 0.0376 | -0.2412 +/- 0.1715 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3553 +/- 0.0132 | 0.3553 +/- 0.0132 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3553 +/- 0.0132 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.9467 +/- 0.0454 | 1.9467 +/- 0.0454 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.9467 +/- 0.0454 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

Per-relation residual preview (`test triples >= 20`; kept 18 relations):

| Relation | Chinese | Model | Subgroup | Side | Seeds | Effective Residual | Effective Fused | Residual/Fused Ratio | Test Triples |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `rel_0010` | 品牌 | Full Model | head_has_img | head | 3 | 0.6722 +/- 0.3412 | 1.5626 +/- 0.6267 | 0.4180 +/- 0.0470 | 334 |
| `rel_0010` | 品牌 | Full Model | head_has_img | tail | 3 | 1.5100 +/- 0.5589 | 1.7697 +/- 0.8032 | 0.8753 +/- 0.0733 | 334 |
| `rel_0010` | 品牌 | Full Model | head_noimg | head | 3 | 0.7451 +/- 0.3688 | 1.6394 +/- 0.6954 | 0.4452 +/- 0.0353 | 334 |
| `rel_0010` | 品牌 | Full Model | head_noimg | tail | 3 | 1.4201 +/- 0.5429 | 1.7745 +/- 0.8085 | 0.8178 +/- 0.0592 | 334 |
| `rel_0010` | 品牌 | Full Model | tail_noimg | head | 3 | 0.6944 +/- 0.3496 | 1.5860 +/- 0.6477 | 0.4263 +/- 0.0434 | 334 |
| `rel_0010` | 品牌 | Full Model | tail_noimg | tail | 3 | 1.4825 +/- 0.5540 | 1.7712 +/- 0.8048 | 0.8578 +/- 0.0689 | 334 |
| `rel_0010` | 品牌 | Residual-only | head_has_img | head | 3 | 0.3543 +/- 0.0105 | 0.0000 +/- 0.0000 | n/a | 334 |
| `rel_0010` | 品牌 | Residual-only | head_has_img | tail | 3 | 1.4680 +/- 0.0509 | 0.0000 +/- 0.0000 | n/a | 334 |
| `rel_0010` | 品牌 | Residual-only | head_noimg | head | 3 | 0.3441 +/- 0.0096 | 0.0000 +/- 0.0000 | n/a | 334 |
| `rel_0010` | 品牌 | Residual-only | head_noimg | tail | 3 | 1.3520 +/- 0.0476 | 0.0000 +/- 0.0000 | n/a | 334 |
| `rel_0010` | 品牌 | Residual-only | tail_noimg | head | 3 | 0.3512 +/- 0.0102 | 0.0000 +/- 0.0000 | n/a | 334 |
| `rel_0010` | 品牌 | Residual-only | tail_noimg | tail | 3 | 1.4326 +/- 0.0499 | 0.0000 +/- 0.0000 | n/a | 334 |
| `rel_0011` | 品牌类型 | Full Model | head_has_img | head | 3 | 0.6983 +/- 0.3554 | 1.5987 +/- 0.6534 | 0.4264 +/- 0.0442 | 39 |
| `rel_0011` | 品牌类型 | Full Model | head_has_img | tail | 3 | 1.8607 +/- 0.8272 | 1.8269 +/- 0.8504 | 1.0283 +/- 0.0536 | 39 |
| `rel_0011` | 品牌类型 | Full Model | head_noimg | head | 3 | 0.7849 +/- 0.3608 | 1.6769 +/- 0.7276 | 0.4668 +/- 0.0164 | 39 |
| `rel_0011` | 品牌类型 | Full Model | head_noimg | tail | 3 | 1.8760 +/- 0.8361 | 1.8215 +/- 0.8434 | 1.0392 +/- 0.0543 | 39 |
| `rel_0011` | 品牌类型 | Full Model | tail_noimg | head | 3 | 0.7272 +/- 0.3571 | 1.6248 +/- 0.6781 | 0.4399 +/- 0.0342 | 39 |
| `rel_0011` | 品牌类型 | Full Model | tail_noimg | tail | 3 | 1.8658 +/- 0.8302 | 1.8251 +/- 0.8481 | 1.0319 +/- 0.0538 | 39 |
| `rel_0011` | 品牌类型 | Residual-only | head_has_img | head | 3 | 0.3006 +/- 0.0171 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0011` | 品牌类型 | Residual-only | head_has_img | tail | 3 | 1.8605 +/- 0.0579 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0011` | 品牌类型 | Residual-only | head_noimg | head | 3 | 0.3583 +/- 0.0160 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0011` | 品牌类型 | Residual-only | head_noimg | tail | 3 | 1.7734 +/- 0.0651 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0011` | 品牌类型 | Residual-only | tail_noimg | head | 3 | 0.3198 +/- 0.0167 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0011` | 品牌类型 | Residual-only | tail_noimg | tail | 3 | 1.8315 +/- 0.0602 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0016` | 关联场景 | Full Model | head_noimg | head | 3 | 1.0824 +/- 0.4340 | 1.7191 +/- 0.7562 | 0.6383 +/- 0.0278 | 122 |
| `rel_0016` | 关联场景 | Full Model | head_noimg | tail | 3 | 0.6854 +/- 0.2743 | 1.7605 +/- 0.7805 | 0.3942 +/- 0.0178 | 122 |
| `rel_0016` | 关联场景 | Full Model | tail_noimg | head | 3 | 1.0824 +/- 0.4340 | 1.7191 +/- 0.7562 | 0.6383 +/- 0.0278 | 122 |
| `rel_0016` | 关联场景 | Full Model | tail_noimg | tail | 3 | 0.6854 +/- 0.2743 | 1.7605 +/- 0.7805 | 0.3942 +/- 0.0178 | 122 |
| `rel_0016` | 关联场景 | Residual-only | head_noimg | head | 3 | 0.7337 +/- 0.0355 | 0.0000 +/- 0.0000 | n/a | 122 |
| `rel_0016` | 关联场景 | Residual-only | head_noimg | tail | 3 | 0.6637 +/- 0.0290 | 0.0000 +/- 0.0000 | n/a | 122 |
| `rel_0016` | 关联场景 | Residual-only | tail_noimg | head | 3 | 0.7337 +/- 0.0355 | 0.0000 +/- 0.0000 | n/a | 122 |
| `rel_0016` | 关联场景 | Residual-only | tail_noimg | tail | 3 | 0.6637 +/- 0.0290 | 0.0000 +/- 0.0000 | n/a | 122 |
| `rel_0019` | 品牌归属地 | Full Model | head_has_img | head | 3 | 0.7144 +/- 0.3674 | 1.5810 +/- 0.6384 | 0.4386 +/- 0.0519 | 186 |
| `rel_0019` | 品牌归属地 | Full Model | head_has_img | tail | 3 | 1.7516 +/- 0.7690 | 1.8222 +/- 0.8445 | 0.9706 +/- 0.0421 | 186 |
| `rel_0019` | 品牌归属地 | Full Model | head_noimg | head | 3 | 0.7476 +/- 0.3599 | 1.6731 +/- 0.7228 | 0.4406 +/- 0.0251 | 186 |
| `rel_0019` | 品牌归属地 | Full Model | head_noimg | tail | 3 | 1.6900 +/- 0.7482 | 1.8210 +/- 0.8433 | 0.9352 +/- 0.0363 | 186 |
| `rel_0019` | 品牌归属地 | Full Model | tail_noimg | head | 3 | 0.7255 +/- 0.3649 | 1.6117 +/- 0.6665 | 0.4392 +/- 0.0429 | 186 |
| `rel_0019` | 品牌归属地 | Full Model | tail_noimg | tail | 3 | 1.7311 +/- 0.7621 | 1.8218 +/- 0.8441 | 0.9588 +/- 0.0402 | 186 |
| `rel_0019` | 品牌归属地 | Residual-only | head_has_img | head | 3 | 0.3282 +/- 0.0148 | 0.0000 +/- 0.0000 | n/a | 186 |
| `rel_0019` | 品牌归属地 | Residual-only | head_has_img | tail | 3 | 1.8589 +/- 0.0568 | 0.0000 +/- 0.0000 | n/a | 186 |
| `rel_0019` | 品牌归属地 | Residual-only | head_noimg | head | 3 | 0.3253 +/- 0.0141 | 0.0000 +/- 0.0000 | n/a | 186 |
| `rel_0019` | 品牌归属地 | Residual-only | head_noimg | tail | 3 | 1.8010 +/- 0.0544 | 0.0000 +/- 0.0000 | n/a | 186 |
| `rel_0019` | 品牌归属地 | Residual-only | tail_noimg | head | 3 | 0.3272 +/- 0.0146 | 0.0000 +/- 0.0000 | n/a | 186 |
| `rel_0019` | 品牌归属地 | Residual-only | tail_noimg | tail | 3 | 1.8396 +/- 0.0560 | 0.0000 +/- 0.0000 | n/a | 186 |
| `rel_0022` | 是否精酿 | Full Model | head_has_img | head | 3 | 0.6690 +/- 0.3299 | 1.5760 +/- 0.6281 | 0.4141 +/- 0.0417 | 66 |
| `rel_0022` | 是否精酿 | Full Model | head_has_img | tail | 3 | 1.7263 +/- 0.6356 | 1.7870 +/- 0.7983 | 0.9907 +/- 0.0851 | 66 |
| `rel_0022` | 是否精酿 | Full Model | head_noimg | head | 3 | 0.7355 +/- 0.3610 | 1.6392 +/- 0.6986 | 0.4405 +/- 0.0326 | 66 |
| `rel_0022` | 是否精酿 | Full Model | head_noimg | tail | 3 | 1.5989 +/- 0.5949 | 1.7816 +/- 0.8030 | 0.9204 +/- 0.0781 | 66 |
| `rel_0022` | 是否精酿 | Full Model | tail_noimg | head | 3 | 0.6882 +/- 0.3388 | 1.5942 +/- 0.6484 | 0.4217 +/- 0.0390 | 66 |
| `rel_0022` | 是否精酿 | Full Model | tail_noimg | tail | 3 | 1.6896 +/- 0.6239 | 1.7854 +/- 0.7997 | 0.9705 +/- 0.0831 | 66 |
| `rel_0022` | 是否精酿 | Residual-only | head_has_img | head | 3 | 0.3855 +/- 0.0088 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0022` | 是否精酿 | Residual-only | head_has_img | tail | 3 | 1.6834 +/- 0.0673 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0022` | 是否精酿 | Residual-only | head_noimg | head | 3 | 0.3605 +/- 0.0117 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0022` | 是否精酿 | Residual-only | head_noimg | tail | 3 | 1.4944 +/- 0.0651 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0022` | 是否精酿 | Residual-only | tail_noimg | head | 3 | 0.3783 +/- 0.0096 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0022` | 是否精酿 | Residual-only | tail_noimg | tail | 3 | 1.6290 +/- 0.0667 | 0.0000 +/- 0.0000 | n/a | 66 |
| `rel_0031` | 适用场景 | Full Model | head_has_img | head | 3 | 0.7024 +/- 0.3570 | 1.5702 +/- 0.6289 | 0.4343 +/- 0.0505 | 112 |
| `rel_0031` | 适用场景 | Full Model | head_has_img | tail | 3 | 1.7012 +/- 0.7104 | 1.8150 +/- 0.8211 | 0.9481 +/- 0.0420 | 112 |
| `rel_0031` | 适用场景 | Full Model | head_noimg | head | 3 | 0.7655 +/- 0.3585 | 1.6549 +/- 0.7076 | 0.4564 +/- 0.0207 | 112 |
| `rel_0031` | 适用场景 | Full Model | head_noimg | tail | 3 | 1.6273 +/- 0.6736 | 1.8174 +/- 0.8194 | 0.9049 +/- 0.0420 | 112 |
| `rel_0031` | 适用场景 | Full Model | tail_noimg | head | 3 | 0.7170 +/- 0.3573 | 1.5899 +/- 0.6471 | 0.4395 +/- 0.0436 | 112 |
| `rel_0031` | 适用场景 | Full Model | tail_noimg | tail | 3 | 1.6840 +/- 0.7019 | 1.8155 +/- 0.8207 | 0.9381 +/- 0.0420 | 112 |
| `rel_0031` | 适用场景 | Residual-only | head_has_img | head | 3 | 0.3442 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0031` | 适用场景 | Residual-only | head_has_img | tail | 3 | 1.7982 +/- 0.0573 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0031` | 适用场景 | Residual-only | head_noimg | head | 3 | 0.3573 +/- 0.0104 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0031` | 适用场景 | Residual-only | head_noimg | tail | 3 | 1.7899 +/- 0.0493 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0031` | 适用场景 | Residual-only | tail_noimg | head | 3 | 0.3473 +/- 0.0112 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0031` | 适用场景 | Residual-only | tail_noimg | tail | 3 | 1.7963 +/- 0.0555 | 0.0000 +/- 0.0000 | n/a | 112 |
| `rel_0045` | 适用季节 | Full Model | head_has_img | head | 3 | 0.6834 +/- 0.3451 | 1.5626 +/- 0.6230 | 0.4251 +/- 0.0478 | 427 |
| `rel_0045` | 适用季节 | Full Model | head_has_img | tail | 3 | 1.6055 +/- 0.7525 | 1.8303 +/- 0.8352 | 0.8741 +/- 0.0109 | 427 |
| `rel_0045` | 适用季节 | Full Model | head_noimg | head | 3 | 0.7431 +/- 0.3710 | 1.6379 +/- 0.6919 | 0.4437 +/- 0.0375 | 427 |
| `rel_0045` | 适用季节 | Full Model | head_noimg | tail | 3 | 1.5929 +/- 0.7445 | 1.8295 +/- 0.8348 | 0.8681 +/- 0.0099 | 427 |
| `rel_0045` | 适用季节 | Full Model | tail_noimg | head | 3 | 0.6995 +/- 0.3521 | 1.5829 +/- 0.6415 | 0.4301 +/- 0.0450 | 427 |
| `rel_0045` | 适用季节 | Full Model | tail_noimg | tail | 3 | 1.6021 +/- 0.7504 | 1.8301 +/- 0.8351 | 0.8725 +/- 0.0106 | 427 |
| `rel_0045` | 适用季节 | Residual-only | head_has_img | head | 3 | 0.3506 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 427 |
| `rel_0045` | 适用季节 | Residual-only | head_has_img | tail | 3 | 2.1356 +/- 0.0324 | 0.0000 +/- 0.0000 | n/a | 427 |
| `rel_0045` | 适用季节 | Residual-only | head_noimg | head | 3 | 0.3325 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 427 |
| `rel_0045` | 适用季节 | Residual-only | head_noimg | tail | 3 | 2.1515 +/- 0.0315 | 0.0000 +/- 0.0000 | n/a | 427 |
| `rel_0045` | 适用季节 | Residual-only | tail_noimg | head | 3 | 0.3457 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 427 |
| `rel_0045` | 适用季节 | Residual-only | tail_noimg | tail | 3 | 2.1399 +/- 0.0321 | 0.0000 +/- 0.0000 | n/a | 427 |
| `rel_0052` | 成分 | Full Model | head_has_img | head | 3 | 0.7049 +/- 0.3544 | 1.5681 +/- 0.6310 | 0.4373 +/- 0.0459 | 39 |
| `rel_0052` | 成分 | Full Model | head_has_img | tail | 3 | 1.4992 +/- 0.6348 | 1.7813 +/- 0.8059 | 0.8468 +/- 0.0267 | 39 |
| `rel_0052` | 成分 | Full Model | head_noimg | head | 3 | 0.8691 +/- 0.4018 | 1.6544 +/- 0.7042 | 0.5192 +/- 0.0238 | 39 |
| `rel_0052` | 成分 | Full Model | head_noimg | tail | 3 | 1.4593 +/- 0.6076 | 1.7827 +/- 0.8049 | 0.8236 +/- 0.0298 | 39 |
| `rel_0052` | 成分 | Full Model | tail_noimg | head | 3 | 0.7554 +/- 0.3689 | 1.5947 +/- 0.6535 | 0.4625 +/- 0.0387 | 39 |
| `rel_0052` | 成分 | Full Model | tail_noimg | tail | 3 | 1.4869 +/- 0.6264 | 1.7817 +/- 0.8056 | 0.8397 +/- 0.0274 | 39 |
| `rel_0052` | 成分 | Residual-only | head_has_img | head | 3 | 0.3605 +/- 0.0101 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0052` | 成分 | Residual-only | head_has_img | tail | 3 | 1.3650 +/- 0.0590 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0052` | 成分 | Residual-only | head_noimg | head | 3 | 0.3227 +/- 0.0133 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0052` | 成分 | Residual-only | head_noimg | tail | 3 | 1.3376 +/- 0.0477 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0052` | 成分 | Residual-only | tail_noimg | head | 3 | 0.3489 +/- 0.0111 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0052` | 成分 | Residual-only | tail_noimg | tail | 3 | 1.3565 +/- 0.0555 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0061` | 是否为有机食品 | Full Model | head_has_img | head | 3 | 0.7081 +/- 0.3739 | 1.6204 +/- 0.6642 | 0.4236 +/- 0.0520 | 41 |
| `rel_0061` | 是否为有机食品 | Full Model | head_has_img | tail | 3 | 1.9772 +/- 0.8361 | 1.8343 +/- 0.8332 | 1.0881 +/- 0.0338 | 41 |
| `rel_0061` | 是否为有机食品 | Full Model | head_noimg | head | 3 | 0.7007 +/- 0.3257 | 1.7296 +/- 0.7628 | 0.4028 +/- 0.0114 | 41 |
| `rel_0061` | 是否为有机食品 | Full Model | head_noimg | tail | 3 | 1.9552 +/- 0.8281 | 1.8347 +/- 0.8340 | 1.0756 +/- 0.0329 | 41 |
| `rel_0061` | 是否为有机食品 | Full Model | tail_noimg | head | 3 | 0.7047 +/- 0.3515 | 1.6710 +/- 0.7099 | 0.4139 +/- 0.0329 | 41 |
| `rel_0061` | 是否为有机食品 | Full Model | tail_noimg | tail | 3 | 1.9670 +/- 0.8324 | 1.8345 +/- 0.8336 | 1.0823 +/- 0.0334 | 41 |
| `rel_0061` | 是否为有机食品 | Residual-only | head_has_img | head | 3 | 0.3058 +/- 0.0195 | 0.0000 +/- 0.0000 | n/a | 41 |
| `rel_0061` | 是否为有机食品 | Residual-only | head_has_img | tail | 3 | 2.4583 +/- 0.0640 | 0.0000 +/- 0.0000 | n/a | 41 |
| `rel_0061` | 是否为有机食品 | Residual-only | head_noimg | head | 3 | 0.2927 +/- 0.0216 | 0.0000 +/- 0.0000 | n/a | 41 |
| `rel_0061` | 是否为有机食品 | Residual-only | head_noimg | tail | 3 | 2.4479 +/- 0.0637 | 0.0000 +/- 0.0000 | n/a | 41 |
| `rel_0061` | 是否为有机食品 | Residual-only | tail_noimg | head | 3 | 0.2997 +/- 0.0205 | 0.0000 +/- 0.0000 | n/a | 41 |
| `rel_0061` | 是否为有机食品 | Residual-only | tail_noimg | tail | 3 | 2.4535 +/- 0.0639 | 0.0000 +/- 0.0000 | n/a | 41 |
| `rel_0078` | 上市年份季节 | Full Model | head_has_img | head | 3 | 0.6844 +/- 0.3451 | 1.5593 +/- 0.6225 | 0.4267 +/- 0.0480 | 438 |
| `rel_0078` | 上市年份季节 | Full Model | head_has_img | tail | 3 | 1.4901 +/- 0.5812 | 1.8299 +/- 0.8360 | 0.8330 +/- 0.0649 | 438 |
| `rel_0078` | 上市年份季节 | Full Model | head_noimg | head | 3 | 0.7757 +/- 0.3865 | 1.6355 +/- 0.6906 | 0.4638 +/- 0.0391 | 438 |
| `rel_0078` | 上市年份季节 | Full Model | head_noimg | tail | 3 | 1.4898 +/- 0.5860 | 1.8301 +/- 0.8359 | 0.8319 +/- 0.0619 | 438 |
| `rel_0078` | 上市年份季节 | Full Model | tail_noimg | head | 3 | 0.7069 +/- 0.3553 | 1.5781 +/- 0.6393 | 0.4358 +/- 0.0458 | 438 |
| `rel_0078` | 上市年份季节 | Full Model | tail_noimg | tail | 3 | 1.4900 +/- 0.5824 | 1.8300 +/- 0.8360 | 0.8327 +/- 0.0641 | 438 |
| `rel_0078` | 上市年份季节 | Residual-only | head_has_img | head | 3 | 0.3466 +/- 0.0119 | 0.0000 +/- 0.0000 | n/a | 438 |
| `rel_0078` | 上市年份季节 | Residual-only | head_has_img | tail | 3 | 2.1325 +/- 0.0317 | 0.0000 +/- 0.0000 | n/a | 438 |
| `rel_0078` | 上市年份季节 | Residual-only | head_noimg | head | 3 | 0.3441 +/- 0.0117 | 0.0000 +/- 0.0000 | n/a | 438 |
| `rel_0078` | 上市年份季节 | Residual-only | head_noimg | tail | 3 | 2.0865 +/- 0.0328 | 0.0000 +/- 0.0000 | n/a | 438 |
| `rel_0078` | 上市年份季节 | Residual-only | tail_noimg | head | 3 | 0.3460 +/- 0.0119 | 0.0000 +/- 0.0000 | n/a | 438 |
| `rel_0078` | 上市年份季节 | Residual-only | tail_noimg | tail | 3 | 2.1211 +/- 0.0319 | 0.0000 +/- 0.0000 | n/a | 438 |
| `rel_0089` | 适合季节 | Full Model | head_has_img | head | 3 | 0.7014 +/- 0.3475 | 1.5855 +/- 0.6572 | 0.4318 +/- 0.0405 | 23 |
| `rel_0089` | 适合季节 | Full Model | head_has_img | tail | 3 | 1.5849 +/- 0.7301 | 1.8337 +/- 0.8376 | 0.8639 +/- 0.0081 | 23 |
| `rel_0089` | 适合季节 | Full Model | head_noimg | head | 3 | 0.8527 +/- 0.3794 | 1.6825 +/- 0.7332 | 0.5042 +/- 0.0047 | 23 |
| `rel_0089` | 适合季节 | Full Model | head_noimg | tail | 3 | 1.5522 +/- 0.6415 | 1.8252 +/- 0.8348 | 0.8627 +/- 0.0393 | 23 |
| `rel_0089` | 适合季节 | Full Model | tail_noimg | head | 3 | 0.7212 +/- 0.3517 | 1.5982 +/- 0.6671 | 0.4413 +/- 0.0358 | 23 |
| `rel_0089` | 适合季节 | Full Model | tail_noimg | tail | 3 | 1.5806 +/- 0.7185 | 1.8326 +/- 0.8373 | 0.8638 +/- 0.0083 | 23 |
| `rel_0089` | 适合季节 | Residual-only | head_has_img | head | 3 | 0.3073 +/- 0.0117 | 0.0000 +/- 0.0000 | n/a | 23 |
| `rel_0089` | 适合季节 | Residual-only | head_has_img | tail | 3 | 2.2132 +/- 0.0283 | 0.0000 +/- 0.0000 | n/a | 23 |
| `rel_0089` | 适合季节 | Residual-only | head_noimg | head | 3 | 0.2956 +/- 0.0154 | 0.0000 +/- 0.0000 | n/a | 23 |
| `rel_0089` | 适合季节 | Residual-only | head_noimg | tail | 3 | 2.1839 +/- 0.0420 | 0.0000 +/- 0.0000 | n/a | 23 |
| `rel_0089` | 适合季节 | Residual-only | tail_noimg | head | 3 | 0.3058 +/- 0.0121 | 0.0000 +/- 0.0000 | n/a | 23 |
| `rel_0089` | 适合季节 | Residual-only | tail_noimg | tail | 3 | 2.2094 +/- 0.0300 | 0.0000 +/- 0.0000 | n/a | 23 |
| `rel_0090` | 功能 | Full Model | head_has_img | head | 3 | 0.6664 +/- 0.3293 | 1.5762 +/- 0.6386 | 0.4128 +/- 0.0393 | 160 |
| `rel_0090` | 功能 | Full Model | head_has_img | tail | 3 | 1.3606 +/- 0.5264 | 1.7895 +/- 0.8124 | 0.7764 +/- 0.0543 | 160 |
| `rel_0090` | 功能 | Full Model | head_noimg | head | 3 | 0.7169 +/- 0.3535 | 1.6559 +/- 0.7090 | 0.4250 +/- 0.0309 | 160 |
| `rel_0090` | 功能 | Full Model | head_noimg | tail | 3 | 1.3980 +/- 0.5219 | 1.7955 +/- 0.8128 | 0.7983 +/- 0.0676 | 160 |
| `rel_0090` | 功能 | Full Model | tail_noimg | head | 3 | 0.6781 +/- 0.3349 | 1.5946 +/- 0.6549 | 0.4156 +/- 0.0373 | 160 |
| `rel_0090` | 功能 | Full Model | tail_noimg | tail | 3 | 1.3693 +/- 0.5254 | 1.7909 +/- 0.8125 | 0.7814 +/- 0.0573 | 160 |
| `rel_0090` | 功能 | Residual-only | head_has_img | head | 3 | 0.3200 +/- 0.0120 | 0.0000 +/- 0.0000 | n/a | 160 |
| `rel_0090` | 功能 | Residual-only | head_has_img | tail | 3 | 1.2913 +/- 0.0554 | 0.0000 +/- 0.0000 | n/a | 160 |
| `rel_0090` | 功能 | Residual-only | head_noimg | head | 3 | 0.3193 +/- 0.0131 | 0.0000 +/- 0.0000 | n/a | 160 |
| `rel_0090` | 功能 | Residual-only | head_noimg | tail | 3 | 1.3228 +/- 0.0605 | 0.0000 +/- 0.0000 | n/a | 160 |
| `rel_0090` | 功能 | Residual-only | tail_noimg | head | 3 | 0.3198 +/- 0.0122 | 0.0000 +/- 0.0000 | n/a | 160 |
| `rel_0090` | 功能 | Residual-only | tail_noimg | tail | 3 | 1.2986 +/- 0.0566 | 0.0000 +/- 0.0000 | n/a | 160 |
| `rel_0091` | 上市时间 | Full Model | head_has_img | head | 3 | 0.7152 +/- 0.3588 | 1.5751 +/- 0.6306 | 0.4422 +/- 0.0474 | 37 |
| `rel_0091` | 上市时间 | Full Model | head_has_img | tail | 3 | 1.2764 +/- 0.5578 | 1.8209 +/- 0.8292 | 0.7065 +/- 0.0267 | 37 |
| `rel_0091` | 上市时间 | Full Model | head_noimg | head | 3 | 0.7336 +/- 0.3580 | 1.6488 +/- 0.6941 | 0.4366 +/- 0.0304 | 37 |
| `rel_0091` | 上市时间 | Full Model | head_noimg | tail | 3 | 1.2538 +/- 0.5646 | 1.8254 +/- 0.8337 | 0.6906 +/- 0.0309 | 37 |
| `rel_0091` | 上市时间 | Full Model | tail_noimg | head | 3 | 0.7191 +/- 0.3586 | 1.5910 +/- 0.6444 | 0.4410 +/- 0.0438 | 37 |
| `rel_0091` | 上市时间 | Full Model | tail_noimg | tail | 3 | 1.2715 +/- 0.5592 | 1.8219 +/- 0.8302 | 0.7030 +/- 0.0273 | 37 |
| `rel_0091` | 上市时间 | Residual-only | head_has_img | head | 3 | 0.3703 +/- 0.0111 | 0.0000 +/- 0.0000 | n/a | 37 |
| `rel_0091` | 上市时间 | Residual-only | head_has_img | tail | 3 | 1.2772 +/- 0.0707 | 0.0000 +/- 0.0000 | n/a | 37 |
| `rel_0091` | 上市时间 | Residual-only | head_noimg | head | 3 | 0.3746 +/- 0.0104 | 0.0000 +/- 0.0000 | n/a | 37 |
| `rel_0091` | 上市时间 | Residual-only | head_noimg | tail | 3 | 1.3389 +/- 0.0701 | 0.0000 +/- 0.0000 | n/a | 37 |
| `rel_0091` | 上市时间 | Residual-only | tail_noimg | head | 3 | 0.3712 +/- 0.0109 | 0.0000 +/- 0.0000 | n/a | 37 |
| `rel_0091` | 上市时间 | Residual-only | tail_noimg | tail | 3 | 1.2906 +/- 0.0705 | 0.0000 +/- 0.0000 | n/a | 37 |
| `rel_0096` | 是否进口 | Full Model | head_has_img | head | 3 | 0.6973 +/- 0.3495 | 1.5696 +/- 0.6296 | 0.4324 +/- 0.0464 | 538 |
| `rel_0096` | 是否进口 | Full Model | head_has_img | tail | 3 | 1.9772 +/- 0.8361 | 1.8343 +/- 0.8332 | 1.0881 +/- 0.0338 | 538 |
| `rel_0096` | 是否进口 | Full Model | head_noimg | head | 3 | 0.7566 +/- 0.3674 | 1.6536 +/- 0.7042 | 0.4499 +/- 0.0301 | 538 |
| `rel_0096` | 是否进口 | Full Model | head_noimg | tail | 3 | 1.9738 +/- 0.8341 | 1.8342 +/- 0.8334 | 1.0864 +/- 0.0342 | 538 |
| `rel_0096` | 是否进口 | Full Model | tail_noimg | head | 3 | 0.7140 +/- 0.3546 | 1.5934 +/- 0.6507 | 0.4374 +/- 0.0418 | 538 |
| `rel_0096` | 是否进口 | Full Model | tail_noimg | tail | 3 | 1.9762 +/- 0.8355 | 1.8343 +/- 0.8333 | 1.0876 +/- 0.0339 | 538 |
| `rel_0096` | 是否进口 | Residual-only | head_has_img | head | 3 | 0.3388 +/- 0.0121 | 0.0000 +/- 0.0000 | n/a | 538 |
| `rel_0096` | 是否进口 | Residual-only | head_has_img | tail | 3 | 2.4583 +/- 0.0640 | 0.0000 +/- 0.0000 | n/a | 538 |
| `rel_0096` | 是否进口 | Residual-only | head_noimg | head | 3 | 0.3419 +/- 0.0119 | 0.0000 +/- 0.0000 | n/a | 538 |
| `rel_0096` | 是否进口 | Residual-only | head_noimg | tail | 3 | 2.4461 +/- 0.0639 | 0.0000 +/- 0.0000 | n/a | 538 |
| `rel_0096` | 是否进口 | Residual-only | tail_noimg | head | 3 | 0.3397 +/- 0.0121 | 0.0000 +/- 0.0000 | n/a | 538 |
| `rel_0096` | 是否进口 | Residual-only | tail_noimg | tail | 3 | 2.4548 +/- 0.0640 | 0.0000 +/- 0.0000 | n/a | 538 |
| `rel_0104` | 包装方式 | Full Model | head_has_img | head | 3 | 0.7079 +/- 0.3505 | 1.6332 +/- 0.6848 | 0.4258 +/- 0.0333 | 34 |
| `rel_0104` | 包装方式 | Full Model | head_has_img | tail | 3 | 1.6103 +/- 0.6536 | 1.7783 +/- 0.7955 | 0.9164 +/- 0.0474 | 34 |
| `rel_0104` | 包装方式 | Full Model | head_noimg | head | 3 | 0.7522 +/- 0.3551 | 1.7157 +/- 0.7532 | 0.4346 +/- 0.0168 | 34 |
| `rel_0104` | 包装方式 | Full Model | head_noimg | tail | 3 | 1.7071 +/- 0.7005 | 1.8058 +/- 0.8183 | 0.9563 +/- 0.0429 | 34 |
| `rel_0104` | 包装方式 | Full Model | tail_noimg | head | 3 | 0.7222 +/- 0.3519 | 1.6599 +/- 0.7069 | 0.4287 +/- 0.0277 | 34 |
| `rel_0104` | 包装方式 | Full Model | tail_noimg | tail | 3 | 1.6416 +/- 0.6687 | 1.7872 +/- 0.8029 | 0.9293 +/- 0.0454 | 34 |
| `rel_0104` | 包装方式 | Residual-only | head_has_img | head | 3 | 0.3476 +/- 0.0204 | 0.0000 +/- 0.0000 | n/a | 34 |
| `rel_0104` | 包装方式 | Residual-only | head_has_img | tail | 3 | 1.3683 +/- 0.0638 | 0.0000 +/- 0.0000 | n/a | 34 |
| `rel_0104` | 包装方式 | Residual-only | head_noimg | head | 3 | 0.3494 +/- 0.0198 | 0.0000 +/- 0.0000 | n/a | 34 |
| `rel_0104` | 包装方式 | Residual-only | head_noimg | tail | 3 | 1.5512 +/- 0.0623 | 0.0000 +/- 0.0000 | n/a | 34 |
| `rel_0104` | 包装方式 | Residual-only | tail_noimg | head | 3 | 0.3482 +/- 0.0200 | 0.0000 +/- 0.0000 | n/a | 34 |
| `rel_0104` | 包装方式 | Residual-only | tail_noimg | tail | 3 | 1.4275 +/- 0.0633 | 0.0000 +/- 0.0000 | n/a | 34 |
| `rel_0115` | 净含量 | Full Model | head_has_img | head | 3 | 0.7015 +/- 0.3538 | 1.5782 +/- 0.6319 | 0.4328 +/- 0.0466 | 116 |
| `rel_0115` | 净含量 | Full Model | head_has_img | tail | 3 | 1.9562 +/- 0.8348 | 1.8305 +/- 0.8434 | 1.0781 +/- 0.0383 | 116 |
| `rel_0115` | 净含量 | Full Model | head_noimg | head | 3 | 0.7468 +/- 0.3478 | 1.6623 +/- 0.7087 | 0.4441 +/- 0.0209 | 116 |
| `rel_0115` | 净含量 | Full Model | head_noimg | tail | 3 | 2.0139 +/- 0.8580 | 1.8322 +/- 0.8443 | 1.1098 +/- 0.0404 | 116 |
| `rel_0115` | 净含量 | Full Model | tail_noimg | head | 3 | 0.7140 +/- 0.3521 | 1.6014 +/- 0.6531 | 0.4359 +/- 0.0392 | 116 |
| `rel_0115` | 净含量 | Full Model | tail_noimg | tail | 3 | 1.9721 +/- 0.8412 | 1.8310 +/- 0.8436 | 1.0869 +/- 0.0389 | 116 |
| `rel_0115` | 净含量 | Residual-only | head_has_img | head | 3 | 0.3559 +/- 0.0121 | 0.0000 +/- 0.0000 | n/a | 116 |
| `rel_0115` | 净含量 | Residual-only | head_has_img | tail | 3 | 2.1148 +/- 0.0525 | 0.0000 +/- 0.0000 | n/a | 116 |
| `rel_0115` | 净含量 | Residual-only | head_noimg | head | 3 | 0.3090 +/- 0.0104 | 0.0000 +/- 0.0000 | n/a | 116 |
| `rel_0115` | 净含量 | Residual-only | head_noimg | tail | 3 | 2.1914 +/- 0.0539 | 0.0000 +/- 0.0000 | n/a | 116 |
| `rel_0115` | 净含量 | Residual-only | tail_noimg | head | 3 | 0.3430 +/- 0.0116 | 0.0000 +/- 0.0000 | n/a | 116 |
| `rel_0115` | 净含量 | Residual-only | tail_noimg | tail | 3 | 2.1360 +/- 0.0529 | 0.0000 +/- 0.0000 | n/a | 116 |
| `rel_0124` | 食品口味 | Full Model | head_has_img | head | 3 | 0.7103 +/- 0.3713 | 1.6154 +/- 0.6639 | 0.4281 +/- 0.0502 | 28 |
| `rel_0124` | 食品口味 | Full Model | head_has_img | tail | 3 | 1.8964 +/- 0.8218 | 1.8108 +/- 0.8422 | 1.0573 +/- 0.0332 | 28 |
| `rel_0124` | 食品口味 | Full Model | head_noimg | head | 3 | 0.6711 +/- 0.2883 | 1.7216 +/- 0.7560 | 0.3912 +/- 0.0044 | 28 |
| `rel_0124` | 食品口味 | Full Model | head_noimg | tail | 3 | 1.7643 +/- 0.7675 | 1.8077 +/- 0.8373 | 0.9833 +/- 0.0288 | 28 |
| `rel_0124` | 食品口味 | Full Model | tail_noimg | head | 3 | 0.6963 +/- 0.3415 | 1.6533 +/- 0.6968 | 0.4149 +/- 0.0311 | 28 |
| `rel_0124` | 食品口味 | Full Model | tail_noimg | tail | 3 | 1.8492 +/- 0.8024 | 1.8097 +/- 0.8405 | 1.0309 +/- 0.0316 | 28 |
| `rel_0124` | 食品口味 | Residual-only | head_has_img | head | 3 | 0.3207 +/- 0.0167 | 0.0000 +/- 0.0000 | n/a | 28 |
| `rel_0124` | 食品口味 | Residual-only | head_has_img | tail | 3 | 1.6947 +/- 0.1182 | 0.0000 +/- 0.0000 | n/a | 28 |
| `rel_0124` | 食品口味 | Residual-only | head_noimg | head | 3 | 0.3063 +/- 0.0196 | 0.0000 +/- 0.0000 | n/a | 28 |
| `rel_0124` | 食品口味 | Residual-only | head_noimg | tail | 3 | 1.5817 +/- 0.1079 | 0.0000 +/- 0.0000 | n/a | 28 |
| `rel_0124` | 食品口味 | Residual-only | tail_noimg | head | 3 | 0.3156 +/- 0.0177 | 0.0000 +/- 0.0000 | n/a | 28 |
| `rel_0124` | 食品口味 | Residual-only | tail_noimg | tail | 3 | 1.6543 +/- 0.1145 | 0.0000 +/- 0.0000 | n/a | 28 |
| `rel_0130` | 产地 | Full Model | head_has_img | head | 3 | 0.6977 +/- 0.3546 | 1.5669 +/- 0.6300 | 0.4328 +/- 0.0487 | 564 |
| `rel_0130` | 产地 | Full Model | head_has_img | tail | 3 | 1.3483 +/- 0.5559 | 1.8260 +/- 0.8440 | 0.7513 +/- 0.0399 | 564 |
| `rel_0130` | 产地 | Full Model | head_noimg | head | 3 | 0.7834 +/- 0.3775 | 1.6580 +/- 0.7087 | 0.4650 +/- 0.0275 | 564 |
| `rel_0130` | 产地 | Full Model | head_noimg | tail | 3 | 1.4608 +/- 0.6105 | 1.8222 +/- 0.8422 | 0.8154 +/- 0.0430 | 564 |
| `rel_0130` | 产地 | Full Model | tail_noimg | head | 3 | 0.7230 +/- 0.3613 | 1.5937 +/- 0.6532 | 0.4423 +/- 0.0425 | 564 |
| `rel_0130` | 产地 | Full Model | tail_noimg | tail | 3 | 1.3814 +/- 0.5719 | 1.8249 +/- 0.8435 | 0.7702 +/- 0.0406 | 564 |
| `rel_0130` | 产地 | Residual-only | head_has_img | head | 3 | 0.3274 +/- 0.0122 | 0.0000 +/- 0.0000 | n/a | 564 |
| `rel_0130` | 产地 | Residual-only | head_has_img | tail | 3 | 2.2612 +/- 0.0276 | 0.0000 +/- 0.0000 | n/a | 564 |
| `rel_0130` | 产地 | Residual-only | head_noimg | head | 3 | 0.3531 +/- 0.0140 | 0.0000 +/- 0.0000 | n/a | 564 |
| `rel_0130` | 产地 | Residual-only | head_noimg | tail | 3 | 2.1766 +/- 0.0347 | 0.0000 +/- 0.0000 | n/a | 564 |
| `rel_0130` | 产地 | Residual-only | tail_noimg | head | 3 | 0.3349 +/- 0.0127 | 0.0000 +/- 0.0000 | n/a | 564 |
| `rel_0130` | 产地 | Residual-only | tail_noimg | tail | 3 | 2.2363 +/- 0.0297 | 0.0000 +/- 0.0000 | n/a | 564 |

## 4.3 `ambiguous_material_relations`


### head_has_img

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.8730 +/- 0.4843 | 0.6812 +/- 0.3450 | 7.4290 +/- 1.5337 | 1.5633 +/- 0.6230 | 0.4232 +/- 0.0488 | -0.8821 +/- 0.2782 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.2487 +/- 1.0467 | 1.7599 +/- 0.7321 | 8.5231 +/- 2.2574 | 1.8069 +/- 0.8214 | 0.9844 +/- 0.0400 | -0.0469 +/- 0.0900 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3623 +/- 0.0118 | 0.3623 +/- 0.0118 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3623 +/- 0.0118 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.9239 +/- 0.0611 | 1.9239 +/- 0.0611 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.9239 +/- 0.0611 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### head_noimg

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.9613 +/- 0.5230 | 0.7504 +/- 0.3720 | 7.7477 +/- 1.8006 | 1.6357 +/- 0.6936 | 0.4490 +/- 0.0362 | -0.8853 +/- 0.3215 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.2744 +/- 1.0610 | 1.7800 +/- 0.7423 | 8.5368 +/- 2.2671 | 1.8099 +/- 0.8241 | 0.9938 +/- 0.0398 | -0.0299 +/- 0.0828 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3557 +/- 0.0111 | 0.3557 +/- 0.0111 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3557 +/- 0.0111 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.9420 +/- 0.0604 | 1.9420 +/- 0.0604 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.9420 +/- 0.0604 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### tail_noimg

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.8978 +/- 0.4951 | 0.7006 +/- 0.3526 | 7.5184 +/- 1.6085 | 1.5836 +/- 0.6428 | 0.4305 +/- 0.0452 | -0.8830 +/- 0.2903 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.2559 +/- 1.0507 | 1.7656 +/- 0.7350 | 8.5270 +/- 2.2602 | 1.8077 +/- 0.8222 | 0.9871 +/- 0.0400 | -0.0421 +/- 0.0880 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3604 +/- 0.0116 | 0.3604 +/- 0.0116 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3604 +/- 0.0116 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.9290 +/- 0.0609 | 1.9290 +/- 0.0609 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.9290 +/- 0.0609 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### target_has_img

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.8730 +/- 0.4843 | 0.6812 +/- 0.3450 | 7.4290 +/- 1.5337 | 1.5633 +/- 0.6230 | 0.4232 +/- 0.0488 | -0.8821 +/- 0.2782 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.2487 +/- 1.0467 | 1.7599 +/- 0.7321 | 8.5231 +/- 2.2574 | 1.8069 +/- 0.8214 | 0.9844 +/- 0.0400 | -0.0469 +/- 0.0900 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3623 +/- 0.0118 | 0.3623 +/- 0.0118 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3623 +/- 0.0118 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.9239 +/- 0.0611 | 1.9239 +/- 0.0611 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.9239 +/- 0.0611 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### target_noimg

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.9117 +/- 0.5012 | 0.7115 +/- 0.3569 | 7.5687 +/- 1.6506 | 1.5950 +/- 0.6539 | 0.4345 +/- 0.0432 | -0.8835 +/- 0.2971 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.2600 +/- 1.0530 | 1.7687 +/- 0.7366 | 8.5291 +/- 2.2617 | 1.8082 +/- 0.8226 | 0.9886 +/- 0.0399 | -0.0394 +/- 0.0868 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3594 +/- 0.0115 | 0.3594 +/- 0.0115 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3594 +/- 0.0115 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.9318 +/- 0.0608 | 1.9318 +/- 0.0608 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.9318 +/- 0.0608 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

### all_targets

| Model | Side | Seeds | Residual Norm | Effective Residual | Fused Norm | Effective Fused | Residual/Fused Ratio | Residual-Fused Gap | Residual Scale | Mix Fusion | Mix Residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Model | head | 3 | 0.8978 +/- 0.4951 | 0.7006 +/- 0.3526 | 7.5184 +/- 1.6085 | 1.5836 +/- 0.6428 | 0.4305 +/- 0.0452 | -0.8830 +/- 0.2903 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Full Model | tail | 3 | 2.2559 +/- 1.0507 | 1.7656 +/- 0.7350 | 8.5270 +/- 2.2602 | 1.8077 +/- 0.8222 | 0.9871 +/- 0.0400 | -0.0421 +/- 0.0880 | 0.5198 +/- 0.1801 | 0.2049 +/- 0.0402 | 0.7951 +/- 0.0402 |
| Residual-only | head | 3 | 0.3604 +/- 0.0116 | 0.3604 +/- 0.0116 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 0.3604 +/- 0.0116 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Residual-only | tail | 3 | 1.9290 +/- 0.0609 | 1.9290 +/- 0.0609 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | n/a | 1.9290 +/- 0.0609 | 0.2585 +/- 0.0034 | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

Per-relation residual preview (`test triples >= 20`; kept 9 relations):

| Relation | Chinese | Model | Subgroup | Side | Seeds | Effective Residual | Effective Fused | Residual/Fused Ratio | Test Triples |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `rel_0003` | 面料 | Full Model | head_has_img | head | 3 | 0.6717 +/- 0.3417 | 1.5574 +/- 0.6163 | 0.4185 +/- 0.0501 | 235 |
| `rel_0003` | 面料 | Full Model | head_has_img | tail | 3 | 1.6587 +/- 0.6672 | 1.7787 +/- 0.8019 | 0.9469 +/- 0.0478 | 235 |
| `rel_0003` | 面料 | Full Model | head_noimg | head | 3 | 0.7495 +/- 0.3743 | 1.6348 +/- 0.6883 | 0.4477 +/- 0.0394 | 235 |
| `rel_0003` | 面料 | Full Model | head_noimg | tail | 3 | 1.6467 +/- 0.6620 | 1.7815 +/- 0.8063 | 0.9389 +/- 0.0502 | 235 |
| `rel_0003` | 面料 | Full Model | tail_noimg | head | 3 | 0.6943 +/- 0.3511 | 1.5798 +/- 0.6371 | 0.4269 +/- 0.0469 | 235 |
| `rel_0003` | 面料 | Full Model | tail_noimg | tail | 3 | 1.6552 +/- 0.6657 | 1.7795 +/- 0.8032 | 0.9446 +/- 0.0485 | 235 |
| `rel_0003` | 面料 | Residual-only | head_has_img | head | 3 | 0.3721 +/- 0.0104 | 0.0000 +/- 0.0000 | n/a | 235 |
| `rel_0003` | 面料 | Residual-only | head_has_img | tail | 3 | 1.6579 +/- 0.0644 | 0.0000 +/- 0.0000 | n/a | 235 |
| `rel_0003` | 面料 | Residual-only | head_noimg | head | 3 | 0.3471 +/- 0.0110 | 0.0000 +/- 0.0000 | n/a | 235 |
| `rel_0003` | 面料 | Residual-only | head_noimg | tail | 3 | 1.6393 +/- 0.0652 | 0.0000 +/- 0.0000 | n/a | 235 |
| `rel_0003` | 面料 | Residual-only | tail_noimg | head | 3 | 0.3648 +/- 0.0106 | 0.0000 +/- 0.0000 | n/a | 235 |
| `rel_0003` | 面料 | Residual-only | tail_noimg | tail | 3 | 1.6525 +/- 0.0646 | 0.0000 +/- 0.0000 | n/a | 235 |
| `rel_0029` | 鞋面材质 | Full Model | head_has_img | head | 3 | 0.6600 +/- 0.3412 | 1.5966 +/- 0.6494 | 0.4010 +/- 0.0484 | 27 |
| `rel_0029` | 鞋面材质 | Full Model | head_has_img | tail | 3 | 1.4449 +/- 0.4623 | 1.7891 +/- 0.8112 | 0.8417 +/- 0.1115 | 27 |
| `rel_0029` | 鞋面材质 | Full Model | head_noimg | head | 3 | 0.7166 +/- 0.3603 | 1.6560 +/- 0.7207 | 0.4246 +/- 0.0298 | 27 |
| `rel_0029` | 鞋面材质 | Full Model | head_noimg | tail | 3 | 1.4232 +/- 0.4383 | 1.8013 +/- 0.8252 | 0.8262 +/- 0.1224 | 27 |
| `rel_0029` | 鞋面材质 | Full Model | tail_noimg | head | 3 | 0.6705 +/- 0.3447 | 1.6076 +/- 0.6626 | 0.4053 +/- 0.0449 | 27 |
| `rel_0029` | 鞋面材质 | Full Model | tail_noimg | tail | 3 | 1.4409 +/- 0.4570 | 1.7914 +/- 0.8138 | 0.8388 +/- 0.1123 | 27 |
| `rel_0029` | 鞋面材质 | Residual-only | head_has_img | head | 3 | 0.3227 +/- 0.0165 | 0.0000 +/- 0.0000 | n/a | 27 |
| `rel_0029` | 鞋面材质 | Residual-only | head_has_img | tail | 3 | 1.4483 +/- 0.1002 | 0.0000 +/- 0.0000 | n/a | 27 |
| `rel_0029` | 鞋面材质 | Residual-only | head_noimg | head | 3 | 0.3128 +/- 0.0112 | 0.0000 +/- 0.0000 | n/a | 27 |
| `rel_0029` | 鞋面材质 | Residual-only | head_noimg | tail | 3 | 1.7075 +/- 0.1234 | 0.0000 +/- 0.0000 | n/a | 27 |
| `rel_0029` | 鞋面材质 | Residual-only | tail_noimg | head | 3 | 0.3209 +/- 0.0155 | 0.0000 +/- 0.0000 | n/a | 27 |
| `rel_0029` | 鞋面材质 | Residual-only | tail_noimg | tail | 3 | 1.4963 +/- 0.1045 | 0.0000 +/- 0.0000 | n/a | 27 |
| `rel_0035` | 里料材质 | Full Model | head_has_img | head | 3 | 0.6847 +/- 0.3432 | 1.5827 +/- 0.6400 | 0.4209 +/- 0.0447 | 33 |
| `rel_0035` | 里料材质 | Full Model | head_has_img | tail | 3 | 1.7470 +/- 0.7433 | 1.7972 +/- 0.8118 | 0.9795 +/- 0.0283 | 33 |
| `rel_0035` | 里料材质 | Full Model | head_noimg | head | 3 | 0.7028 +/- 0.3477 | 1.6566 +/- 0.7096 | 0.4157 +/- 0.0306 | 33 |
| `rel_0035` | 里料材质 | Full Model | head_noimg | tail | 3 | 1.4753 +/- 0.6373 | 1.7963 +/- 0.8100 | 0.8248 +/- 0.0209 | 33 |
| `rel_0035` | 里料材质 | Full Model | tail_noimg | head | 3 | 0.6896 +/- 0.3444 | 1.6029 +/- 0.6590 | 0.4195 +/- 0.0409 | 33 |
| `rel_0035` | 里料材质 | Full Model | tail_noimg | tail | 3 | 1.6729 +/- 0.7144 | 1.7970 +/- 0.8113 | 0.9373 +/- 0.0261 | 33 |
| `rel_0035` | 里料材质 | Residual-only | head_has_img | head | 3 | 0.3472 +/- 0.0152 | 0.0000 +/- 0.0000 | n/a | 33 |
| `rel_0035` | 里料材质 | Residual-only | head_has_img | tail | 3 | 1.7308 +/- 0.0599 | 0.0000 +/- 0.0000 | n/a | 33 |
| `rel_0035` | 里料材质 | Residual-only | head_noimg | head | 3 | 0.3556 +/- 0.0135 | 0.0000 +/- 0.0000 | n/a | 33 |
| `rel_0035` | 里料材质 | Residual-only | head_noimg | tail | 3 | 1.5854 +/- 0.0588 | 0.0000 +/- 0.0000 | n/a | 33 |
| `rel_0035` | 里料材质 | Residual-only | tail_noimg | head | 3 | 0.3495 +/- 0.0148 | 0.0000 +/- 0.0000 | n/a | 33 |
| `rel_0035` | 里料材质 | Residual-only | tail_noimg | tail | 3 | 1.6912 +/- 0.0596 | 0.0000 +/- 0.0000 | n/a | 33 |
| `rel_0046` | 服饰工艺 | Full Model | head_has_img | head | 3 | 0.6588 +/- 0.3384 | 1.5569 +/- 0.6132 | 0.4094 +/- 0.0534 | 43 |
| `rel_0046` | 服饰工艺 | Full Model | head_has_img | tail | 3 | 1.9073 +/- 0.7239 | 1.8017 +/- 0.8111 | 1.0804 +/- 0.0756 | 43 |
| `rel_0046` | 服饰工艺 | Full Model | head_noimg | head | 3 | 0.8342 +/- 0.4142 | 1.6245 +/- 0.6909 | 0.5016 +/- 0.0457 | 43 |
| `rel_0046` | 服饰工艺 | Full Model | head_noimg | tail | 3 | 1.9534 +/- 0.7357 | 1.8053 +/- 0.8094 | 1.1057 +/- 0.0782 | 43 |
| `rel_0046` | 服饰工艺 | Full Model | tail_noimg | head | 3 | 0.6873 +/- 0.3507 | 1.5679 +/- 0.6259 | 0.4244 +/- 0.0519 | 43 |
| `rel_0046` | 服饰工艺 | Full Model | tail_noimg | tail | 3 | 1.9148 +/- 0.7258 | 1.8023 +/- 0.8108 | 1.0845 +/- 0.0759 | 43 |
| `rel_0046` | 服饰工艺 | Residual-only | head_has_img | head | 3 | 0.4353 +/- 0.0091 | 0.0000 +/- 0.0000 | n/a | 43 |
| `rel_0046` | 服饰工艺 | Residual-only | head_has_img | tail | 3 | 1.8294 +/- 0.0706 | 0.0000 +/- 0.0000 | n/a | 43 |
| `rel_0046` | 服饰工艺 | Residual-only | head_noimg | head | 3 | 0.3947 +/- 0.0117 | 0.0000 +/- 0.0000 | n/a | 43 |
| `rel_0046` | 服饰工艺 | Residual-only | head_noimg | tail | 3 | 1.8737 +/- 0.0733 | 0.0000 +/- 0.0000 | n/a | 43 |
| `rel_0046` | 服饰工艺 | Residual-only | tail_noimg | head | 3 | 0.4287 +/- 0.0095 | 0.0000 +/- 0.0000 | n/a | 43 |
| `rel_0046` | 服饰工艺 | Residual-only | tail_noimg | tail | 3 | 1.8366 +/- 0.0710 | 0.0000 +/- 0.0000 | n/a | 43 |
| `rel_0069` | 帮面材质 | Full Model | head_has_img | head | 3 | 0.6479 +/- 0.3279 | 1.5940 +/- 0.6522 | 0.3952 +/- 0.0435 | 44 |
| `rel_0069` | 帮面材质 | Full Model | head_has_img | tail | 3 | 1.3488 +/- 0.4786 | 1.7858 +/- 0.8073 | 0.7785 +/- 0.0783 | 44 |
| `rel_0069` | 帮面材质 | Full Model | head_noimg | head | 3 | 0.6728 +/- 0.3346 | 1.6587 +/- 0.7116 | 0.3976 +/- 0.0321 | 44 |
| `rel_0069` | 帮面材质 | Full Model | head_noimg | tail | 3 | 1.4008 +/- 0.4673 | 1.7883 +/- 0.8138 | 0.8116 +/- 0.0947 | 44 |
| `rel_0069` | 帮面材质 | Full Model | tail_noimg | head | 3 | 0.6541 +/- 0.3296 | 1.6102 +/- 0.6670 | 0.3958 +/- 0.0406 | 44 |
| `rel_0069` | 帮面材质 | Full Model | tail_noimg | tail | 3 | 1.3618 +/- 0.4755 | 1.7864 +/- 0.8089 | 0.7868 +/- 0.0819 | 44 |
| `rel_0069` | 帮面材质 | Residual-only | head_has_img | head | 3 | 0.3204 +/- 0.0160 | 0.0000 +/- 0.0000 | n/a | 44 |
| `rel_0069` | 帮面材质 | Residual-only | head_has_img | tail | 3 | 1.2990 +/- 0.0900 | 0.0000 +/- 0.0000 | n/a | 44 |
| `rel_0069` | 帮面材质 | Residual-only | head_noimg | head | 3 | 0.3122 +/- 0.0141 | 0.0000 +/- 0.0000 | n/a | 44 |
| `rel_0069` | 帮面材质 | Residual-only | head_noimg | tail | 3 | 1.3560 +/- 0.0998 | 0.0000 +/- 0.0000 | n/a | 44 |
| `rel_0069` | 帮面材质 | Residual-only | tail_noimg | head | 3 | 0.3184 +/- 0.0154 | 0.0000 +/- 0.0000 | n/a | 44 |
| `rel_0069` | 帮面材质 | Residual-only | tail_noimg | tail | 3 | 1.3132 +/- 0.0925 | 0.0000 +/- 0.0000 | n/a | 44 |
| `rel_0082` | 材质 | Full Model | head_has_img | head | 3 | 0.6920 +/- 0.3482 | 1.5600 +/- 0.6187 | 0.4310 +/- 0.0489 | 345 |
| `rel_0082` | 材质 | Full Model | head_has_img | tail | 3 | 1.6801 +/- 0.6971 | 1.8047 +/- 0.8194 | 0.9411 +/- 0.0368 | 345 |
| `rel_0082` | 材质 | Full Model | head_noimg | head | 3 | 0.7703 +/- 0.3761 | 1.6311 +/- 0.6924 | 0.4630 +/- 0.0331 | 345 |
| `rel_0082` | 材质 | Full Model | head_noimg | tail | 3 | 1.7330 +/- 0.7097 | 1.8045 +/- 0.8183 | 0.9726 +/- 0.0429 | 345 |
| `rel_0082` | 材质 | Full Model | tail_noimg | head | 3 | 0.7151 +/- 0.3565 | 1.5810 +/- 0.6405 | 0.4405 +/- 0.0441 | 345 |
| `rel_0082` | 材质 | Full Model | tail_noimg | tail | 3 | 1.6957 +/- 0.7008 | 1.8047 +/- 0.8191 | 0.9504 +/- 0.0385 | 345 |
| `rel_0082` | 材质 | Residual-only | head_has_img | head | 3 | 0.3585 +/- 0.0113 | 0.0000 +/- 0.0000 | n/a | 345 |
| `rel_0082` | 材质 | Residual-only | head_has_img | tail | 3 | 1.8760 +/- 0.0592 | 0.0000 +/- 0.0000 | n/a | 345 |
| `rel_0082` | 材质 | Residual-only | head_noimg | head | 3 | 0.3743 +/- 0.0112 | 0.0000 +/- 0.0000 | n/a | 345 |
| `rel_0082` | 材质 | Residual-only | head_noimg | tail | 3 | 1.8264 +/- 0.0600 | 0.0000 +/- 0.0000 | n/a | 345 |
| `rel_0082` | 材质 | Residual-only | tail_noimg | head | 3 | 0.3632 +/- 0.0113 | 0.0000 +/- 0.0000 | n/a | 345 |
| `rel_0082` | 材质 | Residual-only | tail_noimg | tail | 3 | 1.8613 +/- 0.0595 | 0.0000 +/- 0.0000 | n/a | 345 |
| `rel_0092` | 薄厚 | Full Model | head_has_img | head | 3 | 0.6909 +/- 0.3507 | 1.5563 +/- 0.6224 | 0.4311 +/- 0.0498 | 353 |
| `rel_0092` | 薄厚 | Full Model | head_has_img | tail | 3 | 2.0240 +/- 0.9079 | 1.8393 +/- 0.8439 | 1.1040 +/- 0.0174 | 353 |
| `rel_0092` | 薄厚 | Full Model | head_noimg | head | 3 | 0.7511 +/- 0.3749 | 1.6324 +/- 0.6930 | 0.4499 +/- 0.0374 | 353 |
| `rel_0092` | 薄厚 | Full Model | head_noimg | tail | 3 | 2.0420 +/- 0.9156 | 1.8419 +/- 0.8466 | 1.1128 +/- 0.0160 | 353 |
| `rel_0092` | 薄厚 | Full Model | tail_noimg | head | 3 | 0.7091 +/- 0.3580 | 1.5794 +/- 0.6438 | 0.4368 +/- 0.0461 | 353 |
| `rel_0092` | 薄厚 | Full Model | tail_noimg | tail | 3 | 2.0294 +/- 0.9102 | 1.8401 +/- 0.8447 | 1.1067 +/- 0.0169 | 353 |
| `rel_0092` | 薄厚 | Residual-only | head_has_img | head | 3 | 0.3582 +/- 0.0121 | 0.0000 +/- 0.0000 | n/a | 353 |
| `rel_0092` | 薄厚 | Residual-only | head_has_img | tail | 3 | 2.4340 +/- 0.0500 | 0.0000 +/- 0.0000 | n/a | 353 |
| `rel_0092` | 薄厚 | Residual-only | head_noimg | head | 3 | 0.3475 +/- 0.0106 | 0.0000 +/- 0.0000 | n/a | 353 |
| `rel_0092` | 薄厚 | Residual-only | head_noimg | tail | 3 | 2.4572 +/- 0.0486 | 0.0000 +/- 0.0000 | n/a | 353 |
| `rel_0092` | 薄厚 | Residual-only | tail_noimg | head | 3 | 0.3550 +/- 0.0116 | 0.0000 +/- 0.0000 | n/a | 353 |
| `rel_0092` | 薄厚 | Residual-only | tail_noimg | tail | 3 | 2.4410 +/- 0.0496 | 0.0000 +/- 0.0000 | n/a | 353 |
| `rel_0102` | 鞋底材质 | Full Model | head_has_img | head | 3 | 0.6432 +/- 0.3186 | 1.5941 +/- 0.6527 | 0.3940 +/- 0.0373 | 39 |
| `rel_0102` | 鞋底材质 | Full Model | head_has_img | tail | 3 | 1.2911 +/- 0.5027 | 1.7794 +/- 0.8126 | 0.7416 +/- 0.0548 | 39 |
| `rel_0102` | 鞋底材质 | Full Model | head_noimg | head | 3 | 0.6908 +/- 0.3481 | 1.6594 +/- 0.7074 | 0.4066 +/- 0.0377 | 39 |
| `rel_0102` | 鞋底材质 | Full Model | head_noimg | tail | 3 | 1.2964 +/- 0.4892 | 1.7800 +/- 0.8100 | 0.7460 +/- 0.0584 | 39 |
| `rel_0102` | 鞋底材质 | Full Model | tail_noimg | head | 3 | 0.6591 +/- 0.3284 | 1.6159 +/- 0.6709 | 0.3982 +/- 0.0374 | 39 |
| `rel_0102` | 鞋底材质 | Full Model | tail_noimg | tail | 3 | 1.2928 +/- 0.4982 | 1.7796 +/- 0.8117 | 0.7431 +/- 0.0558 | 39 |
| `rel_0102` | 鞋底材质 | Residual-only | head_has_img | head | 3 | 0.3124 +/- 0.0123 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0102` | 鞋底材质 | Residual-only | head_has_img | tail | 3 | 1.4647 +/- 0.0845 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0102` | 鞋底材质 | Residual-only | head_noimg | head | 3 | 0.3176 +/- 0.0103 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0102` | 鞋底材质 | Residual-only | head_noimg | tail | 3 | 1.4599 +/- 0.0806 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0102` | 鞋底材质 | Residual-only | tail_noimg | head | 3 | 0.3141 +/- 0.0115 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0102` | 鞋底材质 | Residual-only | tail_noimg | tail | 3 | 1.4631 +/- 0.0832 | 0.0000 +/- 0.0000 | n/a | 39 |
| `rel_0119` | 填充料 | Full Model | head_has_img | head | 3 | 0.6881 +/- 0.3491 | 1.5580 +/- 0.6166 | 0.4286 +/- 0.0496 | 31 |
| `rel_0119` | 填充料 | Full Model | head_has_img | tail | 3 | 1.9557 +/- 0.7690 | 1.8031 +/- 0.8204 | 1.1041 +/- 0.0765 | 31 |
| `rel_0119` | 填充料 | Full Model | head_noimg | head | 3 | 0.7544 +/- 0.3724 | 1.6341 +/- 0.6749 | 0.4510 +/- 0.0378 | 31 |
| `rel_0119` | 填充料 | Full Model | head_noimg | tail | 3 | 2.1738 +/- 0.8708 | 1.8140 +/- 0.8327 | 1.2203 +/- 0.0794 | 31 |
| `rel_0119` | 填充料 | Full Model | tail_noimg | head | 3 | 0.7010 +/- 0.3536 | 1.5728 +/- 0.6279 | 0.4329 +/- 0.0474 | 31 |
| `rel_0119` | 填充料 | Full Model | tail_noimg | tail | 3 | 1.9979 +/- 0.7887 | 1.8052 +/- 0.8227 | 1.1266 +/- 0.0770 | 31 |
| `rel_0119` | 填充料 | Residual-only | head_has_img | head | 3 | 0.3858 +/- 0.0140 | 0.0000 +/- 0.0000 | n/a | 31 |
| `rel_0119` | 填充料 | Residual-only | head_has_img | tail | 3 | 1.7583 +/- 0.0788 | 0.0000 +/- 0.0000 | n/a | 31 |
| `rel_0119` | 填充料 | Residual-only | head_noimg | head | 3 | 0.4756 +/- 0.0120 | 0.0000 +/- 0.0000 | n/a | 31 |
| `rel_0119` | 填充料 | Residual-only | head_noimg | tail | 3 | 1.9782 +/- 0.0829 | 0.0000 +/- 0.0000 | n/a | 31 |
| `rel_0119` | 填充料 | Residual-only | tail_noimg | head | 3 | 0.4032 +/- 0.0136 | 0.0000 +/- 0.0000 | n/a | 31 |
| `rel_0119` | 填充料 | Residual-only | tail_noimg | tail | 3 | 1.8008 +/- 0.0796 | 0.0000 +/- 0.0000 | n/a | 31 |

## 5. Next Step

- Compare `head_has_img / head_noimg / tail_noimg` to judge whether residual dominance is stronger on missing-image targets.
- Connect residual subgroup patterns back to the completed `6.1` and `6.2` analyses.
- Combine this with relation-aware gate results to finish `7.2` and prepare `7.3 fusion vs residual`.
