# Paper A：DynaSemble 四主模型组合补充实验协议

## 1. 研究问题

本实验只回答一个问题：

> Published unconstrained/dynamic score ensembling 是否在 strong-primary MMKGC pairs 上表现 pair-sensitive？

“pair-sensitive”不预设为正面或负面结论。判断依据是四个预先指定组合上，DynaSemble 相对 DEV-locked Global 的 TEST MRR 差值、original-triple clustered bootstrap 95% CI、head/tail 结果以及逐 seed 结果是否一致。

## 2. 审计结论与实现来源

实现以 `docs/openbg_dynasemble_protocol.md` 与 `scripts/eval_openbg_dynasemble.py` 中已经完成的 OpenBG-IMG faithful reproduction 为冻结依据。DynaSemble 上游来源为：

- Nandi et al., *DynaSemble: Dynamic Ensembling of Textual and Structure-Based Models for Knowledge Graph Completion*, ACL 2024；
- released repository：`dair-iitd/KGC-Ensemble`；
- released source commit：`48d66b915f64899798f736129fa8c4d0a40fdb78`；
- 已审计文件：`NBFNet/script/selector.py`、`NBFNet/script/train_selector.py` 及 two-model YAML configurations。

当前仓库审计发现：旧 OpenBG evaluator 的 selector 与训练逻辑本身可以复用，但数据集名、pair 名、专家名和若干输出字段写死为 OpenBG / M-Hyper / NativE；同时它已落后于公共 full-ranking scorer 的三返回值接口。此次只泛化这些接口与审计输出，不改变 released architecture、loss 或 hyperparameters。

## 3. 冻结实验对象

四个组合均将 M-Hyper 作为 `expert_a` / reliable primary，将另一模型作为 `expert_b` / secondary：

| Dataset | Pair ID | expert_a | expert_b |
| --- | --- | --- | --- |
| MKG-W | `mhyper_native` | M-Hyper | NativE |
| MKG-W | `mhyper_adamf` | M-Hyper | AdaMF-MAT |
| DB15K | `mhyper_native` | M-Hyper | NativE |
| DB15K | `mhyper_adamf` | M-Hyper | AdaMF-MAT |

每组固定使用同 seed 配对的 1、2、3 三个基础模型 checkpoint。基础模型不重新训练。运行路径由 `scripts/run_paper_a_dynasemble_four_pair.ps1` 固定，并在 DEV lock 中以 `config_merged.json` 和 `best.ckpt` 的 SHA-256 再次锁定。

## 4. DynaSemble 冻结配置

对每个 query，两个专家的 filtered candidate scores 分别做 min-max normalization。每个专家提取 released code 的两个特征：

1. `1 - mean(normalized_scores)`；
2. unbiased variance of `normalized_scores`。

四维特征依次输入：

`Linear(4,16) -> Linear(16,16) -> ReLU -> Linear(16,1) -> ReLU`

其余配置冻结为：

- learned weight expert：`expert_a`（M-Hyper）；
- fixed-weight expert：`expert_b`（NativE 或 AdaMF-MAT）；
- fixed weight：1；
- Adam learning rate：`5e-5`；
- `MultiMarginLoss` margin：2.0；
- strict negatives：9,999，加 gold candidate；
- selector training batch size：16；
- epochs：1；
- 第二个 hidden linear layer weight 初始化：uniform `[0,2]`；
- 其余参数：PyTorch default initialization；
- selector random seed：paired base-model seed；
- 每个 paired seed 独立训练一个 selector，不跨 seed 共享样本或参数。

因此最终 score 为：

`score(q,e) = w_a(q) * minmax(s_a(q,e)) + 1 * minmax(s_b(q,e))`，其中 `w_a(q) >= 0`。

这与已审计 OpenBG adaptation 的参数化一致：在 M-Hyper + NativE 中，NativE 固定为 1、M-Hyper 学习非负权重；推广到 AdaMF-MAT 时只把 fixed secondary 的身份替换为 AdaMF-MAT。released architecture 和所有数值超参数均不改变。

该参数化的有效 primary 比例为 `w_a / (w_a + 1)`：`w_a=0` 能精确表示 secondary endpoint，但任何有限权重都不能精确表示 primary endpoint，只能在 `w_a -> infinity` 时渐近接近。四组中除 MKG-W / M-Hyper + NativE (`alpha=0.6`) 外，其余三组当前 DEV-locked Global 均为 primary endpoint (`alpha=1`)；为保持 faithful baseline，不增加 primary bypass、上界、fallback 或重参数化。论文解释必须明确：这些 pair 上 DynaSemble 相对 Global 的结果同时反映 learned selector 与 released asymmetric parameterization 在 strong-primary 场景中的行为。

## 5. 信息边界

### 5.1 DEV

- selector 只以 DEV triples 为优化样本；
- DEV triples 以 paired seed 确定性 shuffle；
- 每个 batch 前半作为 tail query、后半作为 head query，保留 released training-direction behavior；
- 不进行 hyperparameter grid、early stopping、pair orientation selection 或跨 seed 训练；
- Global alpha 直接读取既有 DEV-only `selection.json`，不重新选择；
- Anchored/Query-soft rows 只在 selector 训练和 DynaSemble 排名完成后按 `query_id` 附加，用于 matched reporting，不进入特征、loss 或选择。

与 OpenBG 已审计协议一致，filtered truth universe 使用 TRAIN+DEV+TEST 的已知真事实来移除 false negatives。TEST 的 reciprocal rank、方法输出和任何效果统计均不参与 selector 训练或配置选择；TEST triples 在 DEV 阶段只作为标准 filtered-evaluation 的真事实集合成员。

### 5.2 DEV lock

每个 pair 完成 DEV 后生成 `lock.json`，包含：

- 完整 selector configuration；
- paired selector random seeds；
- 每个 expert 的 config/checkpoint 路径与 SHA-256；
- 每个 selector state 的 SHA-256；
- dataset、pair、protocol version；
- DynaSemble released source repository/commit；
- 当前仓库 commit、evaluator SHA-256、公共 full-ranking evaluator SHA-256；
- 本协议文件 SHA-256；
- Global selection 文件 SHA-256；
- DEV training summary。

存在 `lock.json` 后，DEV 重跑只允许载入既有 selector，不允许重训或改写 lock。TEST 必须验证 selector、expert assets、Global selection、协议和 evaluator hashes 与 lock 完全一致。

### 5.3 TEST

- TEST 不做搜索、early stopping、orientation selection 或阈值选择；
- 已存在的其他方法 TEST 结果不得用于修改 DynaSemble 配置；
- TEST 只加载 DEV-locked selectors；
- Oracle 只作 answer-aware reporting upper bound；
- Anchored Dynamic 本身及其 lock/model/rows 均为只读，不作修改。

## 6. Ranking 与数值协议

- DEV/TEST 均使用 exact filtered full-entity ranking；
- 禁止 top-k candidate approximation；
- candidate scoring 复用 `scripts/eval_heterogeneous_complementarity.py` 的 dataset-independent 分块 scorer、true-fact filtering 与 rank functions；
- evaluator 逐 seed、逐 direction 生成可恢复 checkpoint；
- DynaSemble learned weight 恰为 0 时，rank 明确回退到 fixed expert_b 的已审计 endpoint rank，避免浮点 min-max rounding 改变数学上相同的 endpoint；
- 新输出的 Primary、Global、Query-zscore 0.5 必须与既有 exact full-ranking query rows 逐 query 对齐；rank 不一致或 reciprocal-rank 最大误差超过 `5e-7` 时立即失败。

## 7. 统计汇总

主比较为：

`Delta_DynaSemble_vs_Global = MRR_DynaSemble - MRR_Global`

每个 pair 至少报告：

- pooled TEST；
- head 与 tail；
- seed 1、2、3；
- seed × direction；
- Primary、Global、Query-soft、DynaSemble、Anchored Dynamic；
- DynaSemble − Global 与 Anchored − Global。

95% CI 使用 original triple 为 cluster。每个 cluster 内保留两个预测方向和三个 paired seeds，进行 10,000 次 nonparametric cluster bootstrap，固定 bootstrap seed `20260909`。同时保留旧 OpenBG 报告使用的 cluster-mean normal interval，仅作兼容诊断；论文主报告以 bootstrap CI 为准。

## 8. 输出契约

每组写入：

`outputs/paper_a_safe_correction/dynasemble/<dataset>/<pair>/`

- `dev_summary.json`
- `dev_query_rows.csv`
- `test_summary.json`
- `test_query_rows.csv`
- `results_by_seed.csv`
- `results_by_direction.csv`
- `clustered_bootstrap_ci.json`
- `lock.json`

额外的 selector states、逐 stage results、weight diagnostics 与 seed×direction checkpoints 保存在同一 pair 目录中，用于审计与断点续跑。

四组完成后生成：

- `outputs/paper_a_safe_correction/dynasemble/four_pair_summary.csv`
- `outputs/paper_a_safe_correction/dynasemble/four_pair_summary.md`
- `outputs/paper_a_safe_correction/dynasemble/four_pair_summary_audit.json`

## 9. A100 执行顺序

建议把 DEV 与 TEST 分成两个明确阶段，中间人工确认四个 `lock.json` 已生成：

```powershell
pwsh -File scripts/run_paper_a_dynasemble_four_pair.ps1 -Stage dev -Python python -Device cuda
pwsh -File scripts/run_paper_a_dynasemble_four_pair.ps1 -Stage test -Python python -Device cuda
```

脚本也支持 `-Stage all`，但正式留痕建议使用上述两阶段命令。若服务器 Python 不在 PATH，以 `-Python /absolute/path/to/python` 指定。默认启用 checkpoint resume；只有审计性重算 query rows 时才使用 `-NoResume`，它不会绕过既有 lock 或重训已锁定 selector。

## 10. 预先规定的解释边界

- 若四组 `DynaSemble − Global` 的符号、置信区间或 seed/direction 稳定性明显不同，则支持“published dynamic ensembling 在 strong-primary MMKGC pair 上具有 pair sensitivity”；
- 若四组均稳定同向且幅度接近，则不支持 pair-sensitive 叙事，应如实报告其一致性；
- DynaSemble 不受 bounded correction 或 fallback 保护，因此任何负迁移都是该 published unconstrained baseline 在当前冻结协议下的结果，不得事后调参修复；
- 本实验不能被表述为对 DynaSemble 原论文任务设定的全面否定，只能回答其 released mechanism 在这四个 frozen MMKGC pairs 上的外部泛化表现。
