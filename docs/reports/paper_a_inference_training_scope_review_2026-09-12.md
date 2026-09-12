# B11 / B12：离线 rank 缓存、拟合信息量与部署单位

结论：B11 的训练／评估缓存与真实推理边界已经区分；B12 的 pooled ADC 与逐 base seed 拟合的 DynaSemble 已按代码和保存状态核实。此次只补核查、单测和文字，不重新拟合科学实验模型、不运行基础模型、不改 TEST 结果。

## B11：缓存供离线评估使用

- `router/query_geometry.py::query_geometry_tensor` 只接收两个未过滤候选分数矩阵与方向。gold、单独打分的 gold reference 和 evaluator filter 不进入特征函数。
- `scripts/crossfit_anchored_dynamic.py::apply_policy` 先由特征预测计算 gate、位移和投影权重，再读取该权重对应的 RR 列。DEV 的 standalone RR 构造监督标签，21 个权重的 RR 列用于 DEV 选参与固定策略的离线评估。
- 历史 `lock_apply_anchored_dynamic.py` 的实验 CLI 要求 RR 和关联列，是因为它同时生成评估结果；不能将其输入 schema 当作部署输入。
- `scripts/apply_anchored_safe.py::apply_frame` 只需 13 个特征列、已拟合的模型和 lock，输出权重与有效性诊断。可选 seed/query 元信息只是复制到输出用于关联，不参与预测。该函数是 feature-to-weight helper，并不是加载基础模型的完整 serving adapter。
- Algorithm 1 改为输出选定权重和候选分数。对新 query，两个冻结 checkpoint 评分一次，提取几何特征，确定一个权重，只构造该混合并执行一次排序或 top-k。端点使用活跃专家的原始候选分数向量。部署无需已知正确实体、gold RR 或 21 次完整排序。
- 21 点离线 exact gold rank 也可通过 strict-greater 计数获得，不必执行 21 次全排序。gold 保留、已知答案屏蔽和单独 gold reference 评分属于有标签评估，位于推理之后。
- 上述区分不等于低部署成本：特征提取和两个基础模型的候选评分仍需执行，fallback 并不取消这些上游工作。

`tests/test_inference_training_scope.py` 的六项测试通过：无答案／rank 列推理；三类损坏或无意义的 gold、RR、seed/filter 元信息不改变预测；有限特征逐条与批量应用一致；缺少必需特征时不得用 RR 列替代。测试使用微型合成模型，只验证接口信息边界；不将它视为真实评分 batch invariance 或新 checkpoint 泛化证明。

## B12：每个 pair 拟合几套 selector

ADC 每个 pair 只有 **1 套最终** imputer + scaler + logistic 模型，对三个 base seed 和两个方向共享。seed ID 不是特征。五个 OOF 模型分别在四折训练、在另一折评估，不作为部署集成。

| Pair | DEV 原始三元组 | 三 seed × 两方向 | 非并列拟合行 | 排除并列 |
|---|---:|---:|---:|---:|
| W-N | 4,276 | 25,656 | 19,567 | 6,089 |
| W-A | 4,276 | 25,656 | 20,564 | 5,092 |
| D-N | 7,922 | 47,532 | 37,005 | 10,527 |
| D-A | 7,922 | 47,532 | 38,925 | 8,607 |
| W-NA | 4,276 | 25,656 | 20,522 | 5,134 |
| D-NA | 7,922 | 47,532 | 38,515 | 9,017 |

核查已将六个最终模型的 hash、imputer 中位数、scaler 均值和 `n_samples_seen_` 与非并列 pooled DEV 子集逐项核对。全部通过。另重建并核对 30 个 full-geometry OOF 训练／留出范围，没有重新拟合或声称读取原先未保存的 fold 模型。

Query-soft、行动消融、匹配收缩／线性映射共享 full-geometry 模型；六个 action-family lock 不代表六个 classifier。量化敏感性也复用最终模型。P3 特征子集各需五个 OOF 拟合；9D/13D 补充比较中每个主 pair 新拟合一套最终 9D 模型，13D 则复用原模型。论文新增方法范围表覆盖静态、动态、行动／网格变体、特征变体和不可部署的 Oracle。

DynaSemble 的范围不同：

- R0 每个主 pair 有 3 个最终网络，每个网络只看对应 base seed 的 DEV 分数；selector 随机种子也取该 base seed。
- R1、R2、R3-fixed、R3、R4 每个变体每 pair 有 9 个最终网络：3 个 base seed × selector seeds 11、23、37。每个网络仍只读取对应 base seed 的缓存。
- 对含 N 个 DEV 三元组的单个网络，新控制缓存有 2N 个方向行，但每个 epoch 将每个 batch 前一半分配给 tail、其余给 head，每个三元组只贡献一个方向的 loss，共 N 行／epoch。R0 也只处理 N 个方向 loss 行。不可把 2N 个可用方向缓存行写成每轮都训练了 2N 行。
- loss 为采样候选排序损失，每条 loss query 使用 gold 加 9,999 个负样本，未按 endpoint RR 并列排除；不能直接与 ADC 二元 winner-label 行数等价比较。
- R3 每 pair 的 81 条 CV 轨迹来自 3 base seeds × 3 selector seeds × 3 folds × 3 learning rates，各训练至 10 epoch。学习率和停止 epoch 从 pooled held-out 指标选择，但每个网络的特征训练仍按 base seed 分开。R4 继承 R3 设置。

已核对 12 个 R0、180 个新控制最终网络的身份；新控制同时核对 cache hash、fit IDs、诊断／验证 IDs、model hash。324 条 R3 CV 轨迹均核对 base-seed 绑定、训练／留出分组及学习率／epoch。完整列表存入 `inference_training_scope_v1` 的五个 CSV。

## 统计与部署含义

ADC 的三 base-seed 结果来自对同一 pooled selector 的重复评估，并非三个独立拟合的 ADC。Dyna 新控制先在相同原始三元组/base seed/方向内平均三个 selector 的 RR，再合并 base seed/方向；损害与损失按平均之前的实际 selector 输出计算。这里平均的是评估结果，不是把网络、系数或候选分数融合为一套部署策略。

实际一个 ADC 部署单位为一组冻结的 primary/secondary checkpoint 加一套共享 selector 和 lock，不需同时运行全部三个 base seed。Dyna 也只需适用于该 checkpoint pair 的一个网络；报告的重复均值不代表用 TEST 选择最佳部署 seed。

ADC 的 14 参数不包括预处理状态，更不能抹去 pooled 训练带来的观察量和 score geometry 多样性。ADC--Dyna 比较同时改变信息范围、loss、学习器和预算，因此不能独自定位行动映射贡献；共享实际模型对象的 Query-soft／匹配行动对照提供该证据。

未来新训练、未评估的 checkpoint pair 是否能直接使用现有 pooled selector，没有实验支持。若需要重新拟合或适配，应在声明的 DEV 信息范围内完成并锁定后再评估。本次关闭 B12 的报告缺口，不增添未测的跨 checkpoint 泛化主张。

## 可追溯产物与核验

最终核验通过：6 项接口单测；六个 ADC 保存状态及全部 Dyna 拟合范围核查；全文编译无 overfull box 或未解析引用；56 页、49 表、4 图的 2,962 个源文件绑定与页面边界检查通过。人工检查第 8、23–27 页，Algorithm 1 与新增 Tables 10–11 排版可读，无遮挡。未改变科学实验结果。

- 协议：`docs/protocols/paper_a_inference_training_scope.md`
- 元信息和保存状态核查：`scripts/audit_paper_a_inference_training_scope.py`
- 论文资产构建：`scripts/build_paper_a_inference_scope_assets.py`
- 证据：`outputs/paper_a_safe_correction/inference_training_scope_v1/`
- 源文件／小模型／表格绑定：`paper_a_draft/inference_scope_source_manifest.json`
- 论文：Algorithm 1、方法与统计说明、Appendix 的缓存／拟合／部署说明，以及新增方法范围与拟合数量两张表。

复核命令（均不调用基础模型评分）：

```powershell
python -m pytest tests/test_inference_training_scope.py -q
python scripts/audit_paper_a_inference_training_scope.py
python scripts/build_paper_a_inference_scope_assets.py
python paper_a_draft/scripts/compile.py
python paper_a_draft/scripts/verify_draft.py
```
