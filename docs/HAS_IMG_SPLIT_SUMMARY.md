# has_img / no_img 分组结果摘要

## 1. 说明

本摘要基于 `ml/artifacts/outputs` 下 run 目录中的 `test_metrics.json` 自动汇总。

当前采用的正式口径是：

- 保留当前 `paper_split`，不重划分数据
- `head` 方向按真实 head 实体是否有图分组
- `tail` 方向按真实 tail 实体是否有图分组
- 但当前 `paper_split` 中 tail 目标实体全部无图，因此 tail 方向只有 `tail_no_img` 子组

## 2. 分组规模

- `head_has_img_count = 7048`
- `head_no_img_count = 2952`
- `tail_has_img_count = 0`
- `tail_no_img_count = 10000`

## 3. 主表

| 模型 | Seeds | Overall MRR | Head has_img MRR | Head no_img MRR | Tail no_img MRR |
|---|---:|---:|---:|---:|---:|
| Gate-only | 1 | 0.1703 ± 0.0000 | 0.0127 ± 0.0000 | 0.0096 ± 0.0000 | 0.3287 ± 0.0000 |
| Full Model | 1 | 0.2204 ± 0.0000 | 0.0076 ± 0.0000 | 0.0103 ± 0.0000 | 0.4324 ± 0.0000 |
| Residual-only | 1 | 0.2924 ± 0.0000 | 0.0018 ± 0.0000 | 0.0055 ± 0.0000 | 0.5819 ± 0.0000 |

## 4. 初步观察

- 当前 split 下，tail 方向不存在 `has_img` 子组，因此图像可用性分析主要发生在 head 方向。
- `tail_no_img_mrr` 基本等价于 tail 方向主结果，可直接与整体 `tail_mrr` 对应理解。
- 更值得关注的是：不同模型在 `head_has_img_mrr` 与 `head_no_img_mrr` 上的相对排序是否变化。
