# has_img 分组分析

## 1. 分析目的

本节用于回答一个更具体的问题：在当前 OpenBG-IMG `paper_split` 下，多模态收益是否主要出现在有图目标实体上。

这项分析不是重新定义主评测协议，而是在统一的 `direction=both`、filtered ranking、3-column `train/dev/test` 协议下，对测试样本按目标实体图像可用性进行分组统计。

## 2. 当前 split 的限制

通过 `ml/training/scripts/check_paper_split_has_img_distribution.py` 已确认：

- head 目标实体约 70% 有图
- tail 目标实体 100% 无图

因此，当前 `paper_split` 下不能把 `6.1` 写成对称的“整体 has_img vs no_img”分析。更准确的写法应是：

- `head target has_img / no_img`
- `tail target no_img`

也就是说，图像可用性带来的差异主要发生在 head 方向，tail 方向不存在 `has_img` 子组。

## 3. 参与模型

本轮先分析三组最关键模型：

- `Gate-only`
- `Full Model`
- `Residual-only`

这三组足以回答以下问题：

- 多模态融合本身是否在有图目标实体上体现价值
- 结构补偿是否在整体上压制了多模态收益
- `Full Model` 的优势是否主要来自某些局部子组

## 4. 3-seed 结果摘要

| 模型 | Seeds | Overall MRR | Head has_img MRR | Head no_img MRR | Tail no_img MRR |
|---|---:|---:|---:|---:|---:|
| Gate-only | 3 | 0.1739 ± 0.0044 | 0.0125 ± 0.0004 | 0.0108 ± 0.0014 | 0.3359 ± 0.0087 |
| Full Model | 3 | 0.2100 ± 0.0097 | 0.0069 ± 0.0009 | 0.0119 ± 0.0014 | 0.4116 ± 0.0193 |
| Residual-only | 3 | 0.2930 ± 0.0008 | 0.0017 ± 0.0001 | 0.0055 ± 0.0003 | 0.5832 ± 0.0015 |

分组规模固定为：

- `head_has_img_count = 7048`
- `head_no_img_count = 2952`
- `tail_has_img_count = 0`
- `tail_no_img_count = 10000`

## 5. 结果解读

### 5.1 整体结果仍然由结构路径主导

在 overall MRR 上，排序仍然是：

1. `Residual-only`
2. `Full Model`
3. `Gate-only`

这与主结果表保持一致，说明当前统一协议下的总体趋势没有因为分组分析而改变。

### 5.2 `head_has_img` 子组中，多模态路径更有竞争力

在 `head_has_img` 子组中，排序变为：

1. `Gate-only`
2. `Full Model`
3. `Residual-only`

这说明当被预测的目标实体有图像时，纯结构补偿路径并不占优；相反，直接利用多模态信息的模型更具优势。

这也是当前最关键的局部收益证据。

### 5.3 `head_no_img` 子组中，Full Model 优于 Gate-only

在 `head_no_img` 子组中，排序变为：

1. `Full Model`
2. `Gate-only`
3. `Residual-only`

这表明 `Full Model` 的价值不只是“有图就用图”，还在于它能在无图目标实体上通过结构补偿和融合路径共同工作，获得比 `Gate-only` 更稳的表现。

### 5.4 `tail_no_img` 重新拉回总体排序

由于 `tail` 方向全部是 `no_img`，而且 `Residual-only` 在 `tail_no_img` 上显著最强，所以最终 overall MRR 仍由结构路径主导。

这解释了为什么：

- `head_has_img` 中多模态路径更强
- 但 overall 结果仍然是 `Residual-only` 领先

原因不是前后矛盾，而是测试集分布与目标位置不同导致的结果叠加。

## 6. 对论文主线的意义

这组结果支持一个更稳健的论文结论：

- 多模态收益不是全局存在的
- 它具有明显的目标位置和模态可用性边界
- 在当前 OpenBG-IMG `paper_split` 下，图像相关收益主要体现在 `head_has_img` 子组
- 而整体结果仍被 `tail_no_img` 上的强结构表现主导

因此，本文不应继续把主线写成“多模态模型全面优于结构基线”，而应写成：

- 多模态收益具有条件性
- 强结构表示与多模态表示在不同子组上作用不同
- `Full Model` 的价值更多体现在局部场景中的折中与协同，而不是全局压倒性领先

## 7. 可直接写入论文的表述

可在实验分析部分使用类似表述：

> 在当前 OpenBG-IMG `paper_split` 下，测试集 tail 目标实体全部无图，因此图像可用性分析主要发生在 head 方向。3-seed 结果表明，`Gate-only` 和 `Full Model` 在 `head_has_img` 子组中均显著优于 `Residual-only`，说明多模态信息在目标实体具备视觉模态时能够带来局部收益。然而，在 `tail_no_img` 子组中，`Residual-only` 仍保持明显优势，导致 overall 指标依然由结构路径主导。这说明多模态收益并非全局存在，而是受到目标位置与模态可用性的共同约束。

## 8. 下一步

在 `6.1` 完成后，下一步应进入：

- `6.2 relation type`
- `7. 行为分析`

这样可以继续判断多模态收益是否还具有关系依赖性，以及 `fusion` 与 `residual` 是互补还是竞争。
