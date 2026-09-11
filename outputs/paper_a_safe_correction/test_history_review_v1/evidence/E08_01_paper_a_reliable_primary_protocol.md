# Paper A：DEV-only Reliable-Primary Selection Protocol

## 1. 目的

本协议在运行或解释任何 TEST 结果之前，仅用 standalone exact filtered DEV 结果确定模型组合中的 primary，以排除长期固定 M-Hyper 所可能带来的 post-hoc / cherry-picking 质疑。

## 2. 冻结输入

每个 dataset/model pair 读取现有 `full_ranking/dev_query_rows.csv` 中两个专家的 `rr_a` 与 `rr_b`。输入必须同时满足：

- `split` 仅为 `dev`；
- paired seeds 恰为 1、2、3；
- direction 恰为 head、tail；
- 每个 raw triple 包含 3 seeds × 2 directions 共 6 个 observation；
- reciprocal rank 均为 finite；
- 同一 dataset 中重复出现的 standalone model，其 pooled DEV MRR 在不同 pair asset 中完全一致（容差 `1e-12`）。

脚本不读取任何 TEST 文件。filtered ranking 沿用已有资产的 exact full-entity evaluation 结果，不重新训练 base model 或 combiner。

## 3. Primary 选择

对两个模型分别计算所有 DEV observation 上的 pooled MRR。pooled DEV MRR 较高者为 primary，另一模型为 secondary。若二者完全相同，则不声明可靠 primary，并归入 boundary / non-reliable-primary。

定义：

`delta(q) = rr_primary(q) - rr_secondary(q)`。

进一步计算 pooled DEV delta，以及三个 paired seed 内分别汇总 head/tail 后的 seed delta。

## 4. Clustered bootstrap

采用 10,000 次 percentile bootstrap，固定随机种子 `20260909`。采样单位为 original raw triple `(head_id, relation_id, tail_id)`；同一 triple 的三个 seeds 与两个 directions 始终保留在同一个 cluster 内，禁止把 6 个 seed-direction observations 当作独立样本。

报告 pooled `MRR_primary - MRR_secondary` 的 95% CI。

## 5. 决策规则

一个 pair 当且仅当同时满足以下三项时标记为 `reliable-primary`：

1. pooled DEV delta > 0；
2. 3/3 paired seeds 的 delta > 0；
3. original-triple clustered 95% CI lower bound > 0。

否则标记为 `boundary / non-reliable-primary`。协议不设置任何额外 MRR margin；不得在看到结果后改变条件、随机种子、bootstrap 单位或 pair orientation。

## 6. 预注册应用范围

主审计固定包括：

- MKG-W：M-Hyper vs NativE、M-Hyper vs AdaMF-MAT、NativE vs AdaMF-MAT；
- DB15K：M-Hyper vs NativE、M-Hyper vs AdaMF-MAT、NativE vs AdaMF-MAT。

其中前四个包含 M-Hyper 的组合对应当前 Paper A 四主 pair。NativE vs AdaMF-MAT 用于检验其是否在 standalone DEV 意义上属于 boundary。该判定与后续组合方法是否产生 negative transfer 是两个不同问题。

## 7. 审计与信息边界

`reliable_primary_manifest.json` 记录每个 DEV asset 的 SHA-256、selection timestamp、配置版本、primary 决策及 `TEST_NOT_USED=true`。TEST outcome 不参与 primary、pair eligibility、阈值或方法选择；审计结果“不好看”也不得触发规则修改。

执行命令：

```powershell
python scripts/audit_reliable_primary_regime.py
```
