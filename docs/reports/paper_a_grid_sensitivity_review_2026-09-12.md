# B09：固定网格敏感性回传核验

当前状态：**B09 按固定参数的量化敏感性口径关闭**。48 个评分 cell、316,488 条观察完整，原 0.05／Global／单模型排名逐行零差异，全部新汇总与成对效应通过独立重算；DEV、TEST 和成对变化已进入三张正式补表。没有基础模型训练或本机全实体评分，没有新选网格，也没有替换原论文结果。

## 回传结果

下表为 ADC 相对 Global 的 TEST MRR 差，**已乘以 10,000**。全部六个预设动态配置均保留，没有按该表更换网格。

| Pair | 0.05 | 0.01 | 连续 |
|---|---:|---:|---:|
| W-N | +24.47 | +25.05 | +23.28 |
| W-A | +10.14 | +10.82 | +11.48 |
| D-N | −0.81 | −0.84 | −0.80 |
| D-A | +2.12 | +2.22 | +2.08 |

三组正向 TEST 效应在细网格和连续权重下保留，D-N 仍略为负向。因此不能把已观察到的三组增益完全归因于 0.05 取整；也不能写成所有 pair 正向、与网格无关或统计等价。相对原 0.05，ADC 的最大绝对 TEST MRR 变化为 0.000134；没有给新设置移植旧区间或新增显著性结论。

更细不总是更好：连续权重使 W-N／D-A 的 TEST MRR 略降、W-A／D-N 略升。DEV 上 D-N 的小效应从两种网格下的负向变为连续权重下的正向。ADC 对同一拟合对象 Query-soft 的 MRR 优势在各精度、两 split 上均保留，但不等于优于所有健康动态基线。W-N 的 0.01 相对 0.05 TEST 变化在 head 为 −0.000783、tail 为 +0.000899，不能用微小平均变化宣称每个方向／query 都不受量化影响。

公平风险比较仍有条件：W-N TEST 的 Query-soft 在 0.05 下无条件 RR loss 为 0.005369，略低于 ADC 的 0.005463；在 0.01 下分别为 0.006821／0.005632，连续下为 0.006973／0.005564，排序反转。W-N TEST Query-soft 被压回锚点的比例从 0.05 的 20.21% 降至 0.01 的 4.14%；连续执行全部权重偏离锚点，但不等于全部 gold rank 改变。两 split 的 ADC 均无额外提议被投影压回锚点，gate／裁剪已移除了这些小动作；量化仍改变保留动作幅度与排名。

## 返回证据与独立核验

接收目录为 `outputs/paper_a_safe_correction/grid_sensitivity_review_v1_return`，保留原服务器元信息，不覆盖本地准备计划。逐观察大文件与准备矩阵已加入 `.gitignore`，Git 只提交小型证据、汇总、代码和表格。

服务器 revision：`a2c1728e3c4b20f4a4daf52d6be1a4e63a528663`；计划 SHA-256：`6f274e29bd202e885a848358d2eb8b365de6327a9a8f6aec3764bbbe1edc9875`。2026-09-12 UTC 记录依次为计划 08:11:40、最后 DEV cell 08:31:52、首个 TEST 完成 cell 08:32:45、最后 TEST cell 08:55:19。代码要求完整 DEV 完成记录后才开放 TEST；这些记录不恢复历史 holdout 独立性。

- 18 个 checkpoint 身份与原账本／配置／来源 manifest 一致，完整三元组顺序、seed、head/tail、候选数、batch、每折与 full-DEV 参数均核对。
- 94 项实现与准备来源中 67 项字节一致，27 项只有可证明的整文件 LF／CRLF 转换；未接受其他代码差异。新 B09 脚本本身字节一致。
- 服务器 Python 3.9.1、NumPy 2.0.2、sklearn 1.6.1，A100 40GB、PyTorch 2.6.0+cu126；本地准备使用 sklearn 1.7.1。已按实际运行时披露，不声称跨版本连续 logit 或原始分数逐位一致。
- 逐观察重建三种精度的 ADC／Query-soft 权重、float32 执行权重和 fallback；与计划和旧 RR 对齐。原 0.05 ADC、Query-soft、Global、两单模型共 1,582,440 个 rank 比较全部一致。
- 独立重算 56 行效用／损失／动作汇总、336 行 seed/方向汇总、104 项成对效应、56 行量化诊断，与回传 CSV 在 1e-12 绝对容差内一致。无缺配置、删 seed 或汇总筛选。
- 新排名由服务器固定代码计算，本机未重跑基础评分器。候选矩阵未回传；本地核验代码／来源／权重／原排名复现及返回新 rank 的统计，不声称独立重算了每个新候选分数。
- 新增 10 项回传测试，加原 16 项网格测试共 26 项通过，覆盖错误 rank、伪造汇总、动作／执行权重、顺序／覆盖与路径检查。
- 更新后全稿编译及校验通过：50 页、44 表、4 图、13 篇参考文献，2,779 项来源快照；已目视核对补表与相邻页面，无溢出或遮挡。

正式证据为 `outputs/paper_a_safe_correction/grid_sensitivity_return_review_v1/audit.json` 与 `paper_a_draft/grid_sensitivity_source_manifest.json`。正文方法段与 Appendix A.3 已更新；三张补表报告完整 DEV／TEST MRR、相对 Global 效用、无条件损失、harm、干预率及预设聚合网格变化。

## 定义与比较范围

0.05 网格是论文所报告策略的一部分，同时便于复用 exact-ranking 缓存。它可以消除锚点附近微小动作，也改变保留动作幅度，不能仅称为评估加速。正文已根据上述实测结果限定“收益不完全依赖取整”的范围。

依照 [固定协议](../protocols/paper_a_grid_sensitivity_review.md)，四主 pair、三 seed、两个预测方向均比较 ADC／同一拟合对象的 Query-soft × {0.05、0.01、连续}，另保留 Global。连续仅取消投影，保留原映射、gate、裁剪与 float32 排名算术。每个 split 一次重算候选分数并共用于全部配置，不从原 21 个 reciprocal ranks 插值。

DEV 使用原三元组五折 OOF；TEST 使用原 full-DEV 拟合信号和参数锁。β、τ、α0 均不按网格重调，因此是给定原参数的量化敏感性，不是各网格最优能力比较。全部六个动态配置在新评分前固定，TEST 阶段须先校验完整 DEV 完成记录。历史 TEST 暴露使本轮仍为后验补充。

## 先前本地准备检查（保留阶段记录）

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

## 原服务器运行命令（本轮已完成，无需重跑）

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

## 关闭口径

来源、覆盖、原排名重执行与新汇总核验均已通过，完整敏感性已入正式补表。关闭表示已排查“收益完全由 0.05 取整产生”的替代解释并披露效用／风险变化；不证明最优精度、统计等价、普遍正收益或无 RR 损害。0.05 仍为原算法网格，未依据此次 TEST 选择新配置。本地可运行 `python scripts/review_paper_a_grid_sensitivity.py` 和 `python scripts/build_paper_a_grid_sensitivity_assets.py` 重做核验与建表，无需再次服务器评分。
