# 历史实验结果清单

## 1. 说明

本清单用于汇总当前仓库中已经整理出的历史实验结果，重点回答三个问题：

- 当前有哪些实验结果可以直接找到
- 哪些 run 目前仍值得保留和引用
- 哪些结果适合进入后续论文分析，哪些只适合做参考

本清单基于 [RESULT_INDEX.md](/E:/learn/R&D/mmkg-project-research/docs/RESULT_INDEX.md) 整理，侧重历史结果回顾，不替代自动索引表。

## 2. 当前仍有价值的 run 目录

### 2.1 Gate-Only

- [openbg_img_gate_only/20260308_180434_seed1](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gate_only/20260308_180434_seed1)
- [openbg_img_gate_only/20260308_201856_seed2](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gate_only/20260308_201856_seed2)
- [openbg_img_gate_only/20260309_001045_seed3](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gate_only/20260309_001045_seed3)

### 2.2 Full Model

- [openbg_img_gated_vec_res_rel/20260314_212911_seed1](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260314_212911_seed1)
- [openbg_img_gated_vec_res_rel/20260314_223034_seed2](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260314_223034_seed2)
- [openbg_img_gated_vec_res_rel/20260314_231558_seed3](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260314_231558_seed3)
- [openbg_img_gated_vec_res_rel/20260315_000727_seed4](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260315_000727_seed4)
- [openbg_img_gated_vec_res_rel/20260315_005115_seed5](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260315_005115_seed5)

### 2.3 Residual-Only

- [openbg_img_residual_only/20260313_161055_seed1](/E:/learn/R&D/mmkg-project-research/ml/artifacts/outputs/openbg_img_residual_only/20260313_161055_seed1)

## 3. 历史实验摘要

### 3.1 Gate-Only 结果摘要

| seed | run目录 | 最佳MRR | 最佳epoch | 备注 |
|---:|---|---:|---:|---|
| 1 | `openbg_img_gate_only/20260308_180434_seed1` | 0.3743 | 70 | 结果完整 |
| 2 | `openbg_img_gate_only/20260308_201856_seed2` | 0.3472 | 170 | 结果完整 |
| 3 | `openbg_img_gate_only/20260309_001045_seed3` | 0.3621 | 50 | metrics 文件名异常，但配置正确 |

结论：

- 这是当前较完整的一组 baseline 历史结果
- 已具备 3 个 seed，可作为现阶段对照参考

### 3.2 Full Model 结果摘要

| seed | run目录 | 最佳MRR | 最佳epoch | 备注 |
|---:|---|---:|---:|---|
| 1 | `openbg_img_gated_vec_res_rel/20260314_212911_seed1` | 0.4788 | 50 | 结果完整 |
| 2 | `openbg_img_gated_vec_res_rel/20260314_223034_seed2` | 0.5175 | 16 | 当前最优 |
| 3 | `openbg_img_gated_vec_res_rel/20260314_231558_seed3` | 0.4978 | 30 | 结果完整 |
| 4 | `openbg_img_gated_vec_res_rel/20260315_000727_seed4` | 0.4881 | 14 | 结果完整 |
| 5 | `openbg_img_gated_vec_res_rel/20260315_005115_seed5` | 0.4364 | 16 | 相对偏低，但结果完整 |

结论：

- 这是当前仓库内最完整的主模型多 seed 结果组
- seed2 最佳，seed5 明显低于其他 seed，需要后续注意稳定性解释

### 3.3 Residual-Only 结果摘要

| seed | run目录 | 最佳MRR | 最佳epoch | 备注 |
|---:|---|---:|---:|---|
| 1 | `openbg_img_residual_only/20260313_161055_seed1` | 0.5998 | 29 | 当前结果异常强 |

结论：

- 当前只有 1 个 seed
- 虽然结果突出，但不能直接作为稳定结论
- 这是后续最值得重点复核的实验现象之一

## 4. 当前历史结果的使用建议

### 4.1 可直接用于后续分析的历史结果

- `full model` seed1-5
- `gate-only` seed1-3

### 4.2 可作为重点问题线索的历史结果

- `residual-only` seed1

### 4.3 当前不应直接据此下结论的部分

- 不能仅凭 `residual-only` 单个 seed 就认定其稳定优于 full model
- 不能仅凭现有结果就认定多模态融合无效
- 不能忽略输入特征流程和评测协议对结果的影响

## 5. 当前历史结果还不能覆盖的内容

- `text-only`
- `early fusion`
- `residual-only` 多 seed
- 统一 test 集最终汇报

## 6. 维护建议

- [ ] 每次新增重要 run 后更新本清单
- [ ] 只保留当前仍有解释价值的目录
- [ ] 若后续重跑正式实验，旧 run 可转入“历史参考”分区
