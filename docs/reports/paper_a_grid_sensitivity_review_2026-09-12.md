# B09：量化敏感性准备核验（服务器排名待回传）

当前状态：**B09 未关闭；定义已修订，固定比较与本地准备检查通过，等待完整 DEV／TEST 排名**。本轮没有基础模型训练或本机全实体评分，没有新选网格，也没有修改现有论文数值。

## 定义与比较范围

0.05 网格是论文所报告策略的一部分，同时便于复用 exact-ranking 缓存。它可以消除锚点附近的微小动作；不能仅称为评估加速。论文方法与附录已明确这一点，并保留“尚不能确认收益不依赖取整”的限制。

依照 [固定协议](../protocols/paper_a_grid_sensitivity_review.md)，四主 pair、三 seed、两个预测方向均比较 ADC／同一拟合对象的 Query-soft × {0.05、0.01、连续}，另保留 Global。连续仅取消投影，保留原映射、gate、裁剪与 float32 排名算术。每个 split 一次重算候选分数并共用于全部配置，不从原 21 个 reciprocal ranks 插值。

DEV 使用原三元组五折 OOF；TEST 使用原 full-DEV 拟合信号和参数锁。β、τ、α0 均不按网格重调，因此是给定原参数的量化敏感性，不是各网格最优能力比较。全部六个动态配置在新评分前固定，TEST 阶段须先校验完整 DEV 完成记录。历史 TEST 暴露使本轮仍为后验补充。

## 已完成的本地检查

- 20 个小型 logistic 模型，单 CPU 线程；最多 7 次 solver iteration，未触及 2000 上限。
- W-N／W-A 各 25,656、D-N／D-A 各 47,532 条 OOF 观察，共 146,376 条。原 fold 分配／anchor、ADC 0.05 权重与 fallback、ADC／Query-soft 的原 OOF RR 均通过复现检查。
- Python 3.13.11、NumPy 2.4.2、sklearn 1.7.1。代码／协议／输入共 94 项哈希保存在本地准备快照；不能仅用快照中的旧 HEAD 指代未提交的新代码，真实实现以文件哈希为准。
- 35 项单测通过，覆盖两网格的原始 tie 规则、死区、非有限 fallback、端点、实际过滤排名路径、同 query 的 gold／事实改变、DEV 前置条件、全阶段模拟、哈希续跑和结果汇总。PowerShell 脚本语法检查通过。模拟评分器只用于测试流程，不作为实验结果。
- 论文编译通过，48 页、41 表、4 图、13 篇参考文献；既有 2,622 项来源、引用和页面边界校验通过，目视核对改动第 7／23 页及后续第 24 页，未见溢出或遮挡。该全稿校验不代表 B09 新排名已经完成。
- 所有 OOF 同一可观测 query 在同一拟合 fold 内的 0.05／0.01 权重完全一致。连续 ADC 最大跨度约 1.65e-7，保留原浮点差异，不按 gold 平均，不宣称任意 batching 下逐位一致。

### 已可确定的动作诊断

以下为原 OOF 信号与原锁定参数下的动作统计，不是 MRR 敏感性结果。“死区抑制”指连续动作已偏离锚点，但投影后回到锚点；比率以全部观察为分母。

| Pair | ADC 改动率（三种精度均相同） | ADC 0.05 死区抑制 | Query-soft 0.05 死区抑制 | Query-soft 0.01 死区抑制 |
|---|---:|---:|---:|---:|
| W-N | 63.29% | 0.00% | 17.93% | 3.76% |
| W-A | 19.48% | 0.00% | 0.00% | 0.00% |
| D-N | 10.11% | 0.00% | 0.00% | 0.00% |
| D-A | 0.36% | 0.00% | 0.00% | 0.00% |

ADC 的 gate 和裁剪已经消除了这些配置中的锚点邻近小动作，不能把通用死区性质误报成这里额外拒绝了大量 ADC 动作。量化仍改变保留动作的幅度，仍可能改变排名。W-N Query-soft 的改动率分别为 82.07%、96.24%、100%；这也是让两种映射共用三种精度的原因。上述现象不能代替服务器 MRR／RR loss 对照。

详细快照：`outputs/paper_a_safe_correction/grid_sensitivity_review_v1/local_preparation_audit.json`。它是本地准备审计，不是服务器应直接执行的 `plan.json`。逐观察计划、运行中计划、cell 和回传 ZIP 已加入 `.gitignore`，服务器首次执行会生成自己的计划并在评分前锁定。

## 服务器运行

在服务器仓库根目录，使用先前能加载 18 个 checkpoint 的 CUDA Python 环境：

```powershell
git pull --ff-only
powershell -ExecutionPolicy Bypass -File scripts/run_paper_a_grid_sensitivity.ps1
```

如需指定环境：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_paper_a_grid_sensitivity.ps1 -Python 'E:\path\to\python.exe'
```

包装脚本顺序执行 prepare、score-dev、score-test、report --pack。中断后运行同一命令，已完成 cell 经哈希校验后跳过，不重新选参数；失败会留下 `last_error.json`。代码或输入变更时拒绝复用旧计划，不应删除完成结果后无记录地反复试跑。DEV 评分仍按原协议用 TRAIN∪DEV∪TEST 事实过滤，但这些事实不进入特征和权重。

结束后回传：

```text
outputs/paper_a_safe_correction/grid_sensitivity_review_v1_return.zip
```

ZIP 含计划、逐观察权重／rank、每 cell 校验、汇总和 paired 效应；不含 checkpoint 或候选矩阵。`summary.csv` 应有 56 行、`by_seed_direction.csv` 336 行、`paired_effects.csv` 104 行，覆盖全部 pair／split／配置，另有 56 行动作诊断。

## 回传后的关闭条件

核验 48 个评分 cell 的来源与完整性，复核原 0.05／Global／单模型排名重执行差异。若有差异，完整保留并解释，不静默替换旧表。将 DEV、固定 TEST 的 utility、无条件 RR loss、harm、改动率与 paired ADC−Query-soft、fine／continuous−0.05 效应写入正式补表。收益若消失或排序反转同样披露；不按结果改部署网格、删 seed 或移植旧置信区间。完成这些结果审阅与入稿后才关闭 B09。
