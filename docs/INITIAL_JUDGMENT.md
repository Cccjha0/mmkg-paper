# 主结果初步判断

## 1. 依据

本判断基于 [MAIN_RESULTS_SUMMARY.md](/E:/learn/R&D/mmkg-project-research/docs/MAIN_RESULTS_SUMMARY.md) 中五组主模型在统一协议下的 `test` 结果：

- 数据划分：`paper_split`
- 评测协议：filtered ranking, `direction=both`
- 结果形式：每组 `3 seeds`, 汇总 `mean ± std`

当前五组主模型的 `test MRR mean ± std` 为：

- `Text-only`: `0.1261 ± 0.0043`
- `Early Fusion`: `0.1666 ± 0.0013`
- `Gate-only`: `0.1739 ± 0.0044`
- `Full Model`: `0.2100 ± 0.0097`
- `Residual-only`: `0.2930 ± 0.0008`

## 2. 判断 1：Full Model 是否稳定优于 Gate-only

结论：

- 是，`Full Model` 稳定优于 `Gate-only`

依据：

- `Full Model` 平均 `MRR = 0.2100`
- `Gate-only` 平均 `MRR = 0.1739`
- 平均差值约为 `+0.0361`

解释：

- 这说明在当前统一协议下，`Residual` 分支的引入并不是无效的
- 相比只做关系感知融合，`Fusion + Residual` 组合确实带来了稳定增益

这条结论对论文是有正面意义的，因为它至少支持：

- `Residual` 与 `Fusion` 的组合优于纯 `Fusion`

## 3. 判断 2：Full Model 是否稳定优于 Residual-only

结论：

- 否，`Full Model` 明显弱于 `Residual-only`

依据：

- `Full Model` 平均 `MRR = 0.2100`
- `Residual-only` 平均 `MRR = 0.2930`
- 平均差值约为 `-0.0830`

解释：

- 当前结果不支持“Full Model 全局优于 Residual-only”的方法论文主线
- 相反，它强化了当前论文更合理的问题意识：
  - 为什么强结构/实体补偿分支在该数据集上更占优
  - 为什么多模态融合没有进一步跨过这一强基线

因此，后续分析重点应转向：

- `Residual-only` 是否形成 shortcut
- `Fusion` 是否被长期压制
- 多模态收益是否只在局部场景存在

## 4. 判断 3：Full Model 与强结构基线差距是否缩小

当前结论：

- 暂时不能正式下结论

原因：

- `ComplEx / TuckER` 还没有在当前统一协议下补跑
- 因此目前只能做五组主模型内部判断，不能做与强结构基线的规范横向比较

这也是当前 `3.3` 中唯一仍依赖 `2.2` 的部分。

## 5. 判断 4：主结果对论文主线的影响

当前最合理的论文主线判断是：

- 保持“分析驱动型为主，方法改进型为辅”

理由如下：

1. `Full Model` 虽然优于 `Gate-only`

这说明融合与补偿的组合有价值，但还不足以支撑“提出一个全面更强的新模型”。

2. `Full Model` 明显弱于 `Residual-only`

这意味着当前最关键的研究问题不是“如何证明 full model 最强”，而是：

- 为什么结构补偿更强
- 多模态为何没有稳定转化为全局增益

3. `both` 协议下 `head_mrr` 普遍极低

当前所有模型都呈现：

- `tail_mrr` 明显高于 `head_mrr`

这说明正式论文协议比早期内部 `tail-only` 观察严格得多，也说明：

- 当前结果更适合做“现象分析”
- 不适合直接拿早期高分去支撑方法领先叙事

## 6. 当前阶段结论

截至目前，可以明确写入论文推进判断的是：

1. 五组主模型的正式 `test` 结果已经齐备。
2. `Full Model` 稳定优于 `Gate-only`，说明 `Fusion + Residual` 组合优于纯融合。
3. `Full Model` 明显弱于 `Residual-only`，说明强结构补偿仍然主导当前数据集表现。
4. 当前论文最稳妥的方向仍是：
   - 以“多模态收益边界与结构补偿作用关系分析”为主线
5. 下一步最关键的工作不是继续猜测，而是：
   - 补齐强结构基线
   - 进入 residual dominance 与分组分析
