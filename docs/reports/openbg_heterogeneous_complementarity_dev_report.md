# OpenBG-IMG 强异构模型互补性：DEV 与 5-fold Cross-fitting 报告

**实验对象：** M-Hyper + AdaMF-MAT；M-Hyper + NativE

**数据划分：** OpenBG-IMG `paper_split` DEV

**统计规模：** 3 seeds × 5,000 triples × head/tail，共 30,000 个 seed-direction observations

**报告性质：** TEST 解封前的 DEV 方法审计，不是最终泛化结论

## 1. 执行摘要

本轮审计得到四个主要结论。

1. **强模型之间确实存在可回收的 complementarity，但目前可靠证据主要来自 query-wise score normalization 与 global alpha，而不是 relation alpha。**
2. Full-DEV relation alpha 明显过于乐观。按原始 triple 分组的 5-fold cross-fitting 后，AdaMF-MAT pair 从 `0.366958` 降至 `0.360448`，NativE pair 从 `0.370643` 降至 `0.363851`。AdaMF relation 低于 cross-fit global；NativE relation 仅高 `0.000230`，没有形成可靠的额外增量。
3. **M-Hyper + NativE 是当前更强的 external-validation pair。**固定、不读取 DEV 标签的 Query-zscore 0.5 达到 `0.363821`，三个 seed 均提升；cross-fit global alpha 达到 `0.363621`。
4. Full-DEV relation alpha 的总体增益约 `93–94%` 来自 tail prediction。该区域在 OpenBG protocol 中全部属于 `tail_no_img`，因此 strong heterogeneous line 验证的是 **query-dependent expert utility / architecture complementarity**，而不是 multimodal selective activation 的直接 backbone generalization。

综合判断：**relation alpha cross-fitting 未通过“具有额外泛化价值”的 sanity check；但固定 normalization 与 global alpha 仍提供了足够理由运行一次已经锁定的 TEST。** Relation alpha 应保留为预注册的 secondary diagnostic，不能再作为 DEV 主结果或继续调参。

## 2. 研究问题与两条实验线

本研究应明确保留两条不同但互补的证据线。

### Controlled line：Gate-only + Residual-only

回答 OpenBG protocol、target-side modality support 和 multimodal activation 为什么不是 uniformly beneficial。两个 expert 是受控构造，用于隔离 fusion activation 与 structural fallback 的机制。

### Strong heterogeneous line：M-Hyper + AdaMF-MAT / NativE

回答 independently designed strong MMKGC models 是否表现出稳定、可回收的 query-dependent utility。该实验检验 external validity 与 architecture-level complementarity，不直接证明 multimodal activation 可以跨 backbone 泛化。

因此，strong line 更准确的名称是：

> **external validation of query-dependent expert utility**

或：

> **strong-model validation of score-aware complementarity**

## 3. 协议、信息边界与端点审计

所有方法均使用 exact filtered full-entity ranking，而不是 top-k 近似。两个专家分别沿用 checkpoint 保存的 `chunk_size/query_batch_size`，head scoring 也遵循各模型的正式 evaluator contract。

| 方法 | 参数来源 | 信息边界 |
| --- | --- | --- |
| Equal RRF | 固定 `k=60` | rank-aware、answer-agnostic |
| Query-zscore 0.5 | 固定等权 | score-aware、answer-agnostic |
| Global alpha | `0.00:0.05:1.00` | score-aware、answer-agnostic；DEV selected |
| Relation alpha | 同一 alpha grid；低支持 relation 回退 global | relation-conditioned、score-aware；DEV selected |
| Oracle | 每 query 取较高 expert RR | answer-aware upper bound |

两组实验共 12 个固定专家端点，均复现原 checkpoint DEV MRR；最大绝对误差低于 `2.1e-8`，远小于预设容差 `5e-7`。因此以下差异不是 M-Hyper batch shape、head scorer 或 checkpoint 重载错误造成的。

## 4. Support 独立性修正

原始 full-DEV relation policy 使用 `support ≥ 60` 个 seed-direction observations。这个 support 不能解释为 60 个独立 DEV queries：

- 同一个 directional query 在三个模型 seed 中重复出现，因此 60 observations 最多对应约 20 个 seed-stripped directional query identities；
- 如果进一步把同一原始 triple 的 head/tail 两个方向作为一个相关 cluster，则约只对应 10 个 original-triple clusters。

因此 full-DEV relation alpha 的有效样本量显著小于表面 support，存在较高的 relation-level selection overfitting 风险。

## 5. 5-fold Cross-fitting 设计

Cross-fitting 完全使用已经导出的 DEV query rows，不重新评分模型，也不改变任何方法参数。

- 单位：5,000 个 unique original triples。
- 分折：按 relation 分层，对 triple key 做固定 SHA256 排序后，以 relation-specific deterministic offset round-robin 分配到 5 folds。
- 泄漏保护：同一 triple 的三个 seed 和 head/tail 两个方向，共 6 个 observations，必须进入同一 fold。
- 每轮：在其余 4 folds 选择 global/relation alpha，在 held-out fold 评估。
- 保持不变：query-zscore normalization、alpha grid、`support ≥ 60`、低支持回退规则、alpha tie-break。

每个 fold 包含 997–1,010 个 held-out triples；训练 folds 中达到 relation-specific support 的 relation 使用 relation alpha，其余回退 global。

## 6. 主结果：Full-DEV fit 与 Cross-fit 对照

### 6.1 M-Hyper + AdaMF-MAT

| 方法 | DEV MRR | Delta vs. M-Hyper | Oracle gap recovery |
| --- | ---: | ---: | ---: |
| M-Hyper | 0.357397 | +0.000000 | 0.00% |
| AdaMF-MAT | 0.315629 | -0.041768 | -91.11% |
| Equal RRF | 0.327998 | -0.029399 | -64.13% |
| Query-zscore 0.5 | 0.357919 | +0.000521 | 1.14% |
| Global alpha, full-DEV | 0.362081 | +0.004684 | 10.22% |
| **Global alpha, 5-fold cross-fit** | **0.362081** | **+0.004684** | **10.22%** |
| Relation alpha, full-DEV | 0.366958 | +0.009561 | 20.86% |
| **Relation alpha, 5-fold cross-fit** | **0.360448** | **+0.003051** | **6.65%** |
| Oracle | 0.403240 | +0.045843 | 100.00% |

五个 training-fold policies 都选择 global alpha `0.95`，所以 cross-fit global 与 full-DEV global 完全一致。Relation alpha 的增益从 `+0.009561` 降至 `+0.003051`，只保留原始增益的约 31.9%，并低于 global alpha `0.001633`。

### 6.2 M-Hyper + NativE

| 方法 | DEV MRR | Delta vs. M-Hyper | Oracle gap recovery |
| --- | ---: | ---: | ---: |
| M-Hyper | 0.357397 | +0.000000 | 0.00% |
| NativE | 0.308105 | -0.049292 | -118.62% |
| Equal RRF | 0.319983 | -0.037414 | -90.04% |
| Query-zscore 0.5 | 0.363821 | +0.006424 | 15.46% |
| Global alpha, full-DEV | 0.364721 | +0.007323 | 17.62% |
| **Global alpha, 5-fold cross-fit** | **0.363621** | **+0.006224** | **14.98%** |
| Relation alpha, full-DEV | 0.370643 | +0.013246 | 31.87% |
| **Relation alpha, 5-fold cross-fit** | **0.363851** | **+0.006454** | **15.53%** |
| Oracle | 0.398952 | +0.041555 | 100.00% |

五个 training-fold policies 选择的 global alpha 为 `0.55–0.70`。Relation alpha 的增益从 `+0.013246` 降至 `+0.006454`，只保留原始增益约 48.7%；其 pooled MRR 只比 cross-fit global 高 `0.000230`，远小于 full-DEV relation 相对 global 的 `0.005922`，并与固定 Query-zscore 0.5 基本处于同一水平。

## 7. Cross-fit 的 fold 与 seed 稳定性

### 7.1 Fold 结果

| Pair | Fold | M-Hyper | Cross-fit global | Cross-fit relation |
| --- | ---: | ---: | ---: | ---: |
| AdaMF-MAT | 1 | 0.349550 | 0.353547 | 0.355051 |
| AdaMF-MAT | 2 | 0.362288 | 0.364881 | 0.361319 |
| AdaMF-MAT | 3 | 0.363736 | 0.368566 | 0.364637 |
| AdaMF-MAT | 4 | 0.357227 | 0.363517 | 0.364904 |
| AdaMF-MAT | 5 | 0.354186 | 0.359875 | 0.356274 |
| NativE | 1 | 0.349550 | 0.356634 | 0.353776 |
| NativE | 2 | 0.362288 | 0.366508 | 0.369824 |
| NativE | 3 | 0.363736 | 0.367821 | 0.370255 |
| NativE | 4 | 0.357227 | 0.364808 | 0.365077 |
| NativE | 5 | 0.354186 | 0.362317 | 0.360304 |

Cross-fit relation 相对 M-Hyper 在十个 pair-fold combinations 中仍全部为正，但相对 cross-fit global，AdaMF-MAT 只在 2/5 folds 更好，NativE 只在 3/5 folds 更好。AdaMF pooled 后更差；NativE pooled 后仅有 `+0.000230`，不足以支持 relation conditioning 具有稳定的额外价值。

### 7.2 Seed 结果

| Pair / seed | M-Hyper | Cross-fit global | Cross-fit relation |
| --- | ---: | ---: | ---: |
| AdaMF-MAT / 1 | 0.356943 | 0.360512 | 0.357724 |
| AdaMF-MAT / 2 | 0.358324 | 0.363773 | 0.363465 |
| AdaMF-MAT / 3 | 0.356925 | 0.361957 | 0.360155 |
| NativE / 1 | 0.356943 | 0.362096 | 0.362897 |
| NativE / 2 | 0.358324 | 0.366878 | 0.366046 |
| NativE / 3 | 0.356925 | 0.361889 | 0.362609 |

Cross-fit global 在两个 pair 的全部 seeds 上均为正，是当前比 relation alpha 更稳定的低容量结果。

## 8. Clustered uncertainty

以 original triple 为 cluster，把三个 seed 和 head/tail 的六个 observations 一起计算 paired delta：

| Pair | Cross-fit 方法 | Delta vs. M-Hyper | 95% normal interval |
| --- | --- | ---: | ---: |
| AdaMF-MAT | Global alpha | +0.004684 | [+0.003905, +0.005462] |
| AdaMF-MAT | Relation alpha | +0.003051 | [+0.001442, +0.004659] |
| NativE | Global alpha | +0.006224 | [+0.004582, +0.007866] |
| NativE | Relation alpha | +0.006454 | [+0.004607, +0.008300] |

这些区间说明 cross-fit policy 相对 M-Hyper 的聚合增益仍为正；但 relation conditioning 本身没有展示出超越 global policy 的可靠增量。

## 9. 方向分解与增益来源

### 9.1 Full-DEV relation alpha

| Pair | Head delta | Tail delta | Tail 在两方向增益和中的占比 |
| --- | ---: | ---: | ---: |
| M-Hyper + AdaMF-MAT | +0.001294 | +0.017828 | 93.2% |
| M-Hyper + NativE | +0.001640 | +0.024851 | 93.8% |

总体 MRR 是 head/tail MRR 的等权平均，因此上表的贡献分解直接反映总体增益来源：两组 full-DEV relation gain 均约 `93–94%` 来自 tail prediction。

### 9.2 Cross-fit policies

| Pair | Policy | Head delta | Tail delta | Tail contribution |
| --- | --- | ---: | ---: | ---: |
| AdaMF-MAT | Global | +0.000394 | +0.008973 | 95.8% |
| AdaMF-MAT | Relation | +0.000599 | +0.005502 | 90.2% |
| NativE | Global | +0.001159 | +0.011289 | 90.7% |
| NativE | Relation | +0.001210 | +0.011697 | 90.6% |

即使消除 full-DEV relation fitting，约 90% 以上的可回收增益仍来自 tail。OpenBG 的 tail targets 全部属于 `tail_no_img`，因此 strong-model complementarity 不能主要解释为“有图时开启 multimodal fusion、无图时关闭”。更合理的解释是：

> 不同 MMKGC architectures 在 structure-dominant / difficult query regions 中具有互补 inductive biases。

可能参与这一条件异质性的因素包括 relation semantics、structural pattern、score calibration、representation geometry 与模型归纳偏置。当前结果只能定位现象，不能区分这些原因。

## 10. Query 与跨模型稳定性

- AdaMF-MAT 胜出率：35.15%；NativE 胜出率：33.70%。
- 两者同时胜过 M-Hyper：24.89%；至少一个胜出：43.96%。
- 两个胜出集合 Jaccard：`0.566`；winner sign agreement：69.17%。
- 两个 pair 的 seed-wise winner label 完全一致率约 45%，pairwise seed agreement 约 62%。
- 两个 full-DEV relation-alpha maps 在 65 个高支持 relations 上的 Pearson correlation 为 `r=0.423`。

这些结果支持“存在共享但不完全相同的困难区域”，也说明单 query 胜者和最佳 relation 权重都具有明显噪声与 pair specificity。论文不应声称 query winner 已经高度稳定或 relation policy 可以直接跨模型迁移。

## 11. 方法判断

### Equal RRF：明确失败

两组 pair 都显著低于 M-Hyper。Error diversity 不自动等价于可利用 complementarity；弱专家的完整排序会稀释强 anchor。

### Query-zscore 0.5：NativE pair 的最干净证据

它不读取 DEV 标签、不选择参数，在三个 seed 上均提高 MRR。其 `+0.006424` 增益与 NativE cross-fit global/relation 几乎相当，是目前最有说服力的 low-capacity recovery 结果。

### Global alpha：通过 cross-fitting sanity check

两个 pair 的 cross-fit global 均在所有 folds/seeds 上高于 M-Hyper，且 clustered intervals 不跨 0。它应成为 TEST 的主要 DEV-selected baseline。

### Relation alpha：未证明可靠的额外泛化价值

Cross-fitting 后，它仍高于 M-Hyper；但 AdaMF pair 低于 global alpha，NativE pair 仅高 `0.000230` 且 fold/seed 优势不一致。Full-DEV 的大幅提升主要来自 relation-level in-sample fitting。当前不应提高容量、降低 support threshold 或根据 cross-fit 重新调参，否则会继续消耗 DEV。

## 12. TEST 解封与论文叙事建议

### TEST 决策

现在可以运行一次已经锁定的 TEST，但优先级应预先规定：

1. Primary：固定 Query-zscore 0.5；DEV-locked global alpha。
2. Secondary diagnostic：原先已锁定的 relation alpha，不修改任何参数。
3. Negative control：Equal RRF。
4. Upper bound：Oracle。

无论 TEST relation alpha 是否偶然较高，都必须同时呈现 cross-fit 结果，避免用单次 TEST 掩盖其 DEV 过拟合迹象。

### 论文叙事

Controlled line 继续说明 protocol-shaped visual availability 会造成 fusion/structure utility 差异。Strong heterogeneous line 则进一步说明：conditional model utility 不只来自 visual availability；在 target visual support 完全缺失的 tail region，不同强 MMKGC architectures 仍表现出可回收的 score-aware complementarity。

因此不应使用 “backbone generalization of multimodal selective activation”。更稳健的主张是：

> Independently trained MMKGC models exhibit query-dependent utility differences that extend beyond target-side modality availability; a non-trivial portion is recoverable through low-capacity, answer-agnostic score calibration, while fine-grained relation policies are vulnerable to validation overfitting.

## 13. 可复现来源

- `outputs/openbg_img/heterogeneous_complementarity/mhyper_adamf/crossfit/`
- `outputs/openbg_img/heterogeneous_complementarity/mhyper_native/crossfit/`
- `scripts/crossfit_heterogeneous_dev_policies.py`
- `docs/openbg_heterogeneous_complementarity_protocol.md`
