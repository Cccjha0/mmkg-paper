# B01–B03：标准化定义、差值坐标及理论性质核查

本轮完成 B02 的匹配补实验和 B03 的性质定位。B01 的标准化定义及边界测试已补齐，但发现异常输入的真实遗留局限，**原始分数异常是否影响已有结果仍待服务器审计，B01 不作整体关闭**。

## B01：以实际论文导出路径为准

`scripts/eval_heterogeneous_complementarity.py` 的路径为 `score_expert_block → query_zscore_with_reference → filtered_copy → endpoint_safe_mixed_ranks`。`router/score_combination.py` 的单模型 `normalize_candidate_scores(..., "query_zscore")` 与其候选归一化一致；该文件另一个 `combine_expert_scores` 会进行 finite fill，**不能用它解释当前论文导出的 interior ranks**。

| 输入情况／步骤 | 实际处理 |
|---|---|
| 两模型实体集合 | 实体 ID 对齐，每模型独立选择原始未过滤的有限值集合；不要求 finite mask 相同，不取交集或并集计算矩 |
| 统计量 | `n=max(1, finite_count)`；有限值均值；population 方差除以 n；开平方后加 `1e-8`，不是 variance+eps 或 clamp(std,eps) |
| NaN、+∞、−∞ | 都不参与矩计算，标准化后都为 −∞ |
| 常数有限集合、仅一个有限元素 | 矩运算保持有限时，有限位置标准化为 0，原本非有限位置仍为 −∞ |
| 全非有限行 | 内部均值、标准差为 0，输出整行 −∞；不自动删 query，也不自动触发 selector fallback |
| gold | 单独评分后沿用候选矩转换，不添入候选均值/方差；只在评估分支使用 |
| 过滤 | 先计算未过滤候选的归一化结果，再对拷贝 mask，当前 gold 始终保留 |
| interior mixture | 直接插值；任一模型无有限得分的候选成为 −∞；共享评估 mask 也保持 −∞ |
| 两端点 | 显式使用原始对应 expert rank，绕开 `0×−∞` 并保留端点评估 |
| 数值精度 | 保留输入 tensor dtype，当前导出为 float32，没有自动用 float64 累加；极端有限输入仍可能溢出 |

另一区别也已写入附录：几何特征的 score_std 是 `torch.std` 的 sample correction=1；z-score 归一化使用 population correction=0。几何的 sanitizer 取整个 expert batch 的有限 min−1/max+1，空有限集合使用 −100/100，因而含异常输入时几何可能随 batch 其他行变化；它不是逐行安全回退。classifier 只会把**剩余**非有限特征 flag 为 fallback，不能检测已被上游 sanitize 的异常。

独立标量实现与两个实际归一化函数核对了普通行、常数行、近常数行、单个有限值、混合非有限值和全非有限行；另检验独立 finite mask、归一化 batch 不变性、样本标准差、interior/endpoint 语义与原 B07/C04 gold/filter 边界。

**发现的未关闭问题：**如果 gold 参考也是 NaN，旧 strict-greater rank 函数的比较均为假，可返回 rank 1。测试明确记录此反例，不把它当成合理缺失值策略。当前保留的有限特征无法证明原始评分无异常，因此本轮未声称历史结果已经排除这一影响，也未悄悄改动旧评分路径或替换结果。

服务器审计脚本 `scripts/audit_paper_a_raw_score_contract.py` 读取已绑定 checkpoint/split 清单，验证 18 个 checkpoint 与配置的 SHA-256，遍历两个数据集、三个模型、三 seed、DEV/TEST 与双方向共 72 个 cell。统计部分／全非有限行、常数行、gold 异常、矩溢出、归一化异常及最多 20 个例子；不训练、不调参、不替换历史 RR。它核验重新执行的冻结模型，不能证明未保存的历史原始分数 bitwise 一致。

在服务器仓库根目录运行：

```powershell
python scripts/audit_paper_a_raw_score_contract.py --plan-only
python scripts/audit_paper_a_raw_score_contract.py --device cuda
```

回传 `outputs/paper_a_safe_correction/raw_score_contract_audit.json`。发现异常时脚本写完完整报告后以退出码 2 结束，不应删除异常行继续报最好结果。若异常存在，后续需确定受影响模型／pair、制定显式拒绝或修复策略，再重导出并重评；若没有异常，也只支持这次冻结 checkpoint 重执行的数值条件。当前仅在本机运行 plan-only 与合成数据单测，没有执行全量评分。

## B02：差值改变参数化，未增加独立线性信号

精确算术、完整输入下，13D = D×9D，其中 D 包含 9 维恒等行和四个 A−B 差值行。带截距的线性／logistic 表示空间不变。独立标准化后，增广映射 M 把 `[z9,1]` 映射为 `[z13,1]`，13D 实现有效系数 v 的最小 L2 代价为 `vᵀ(MᵀM)⁻¹v`，通常不是 9D 的 `vᵀv`。本配置 liblinear 的 unit synthetic intercept 也受惩罚。未缩放的 `(a,b,a−b)` 例子给出 `(2v_a²+2v_av_b+2v_b²)/3`。

列中位数插补可破坏该依赖，已经用反例测试；本轮四主 pair 的完整 DEV/TEST 导出及各折训练部分没有非有限特征，插补不是本轮差异的来源。浮点差值残差最大 `1.9073486e-6`；将 13D 拟合折叠成 9D 原坐标仿射系数后，已检查数据上的最大 decision 差为 `3.1638975e-7`。不将浮点残差误称新独立信号。

按预先写入本轮协议的五折、标签、随机种子、预处理、balanced/liblinear/C=1 和同一 41 动作预算完成 9／13D 对照。先完成四 pair×两维度的八个 DEV 锁，再独立应用 TEST；原 13D 结果全部保留。44 次小型拟合、复用 4 个原模型，另有 4 次 DEV-only runtime 核验拟合；没有基础模型训练，均限制为单 CPU 线程，最多 7 次迭代。

| Pair | ADC ΔDEV OOF：13D−9D | ADC ΔTEST：13D−9D |
|---|---:|---:|
| W-N | +1.7477e-6 | −9.3461e-6 |
| W-A | +2.5701e-5 | +1.1289e-6 |
| D-N | +9.9725e-9 | +4.4907e-7 |
| D-A | +1.8907e-5 | +1.0221e-8 |

TEST 的最大绝对差不足 1e-5，W-N 是 9D 略高；Query-soft 的 13D−9D 符号也混合。两张正式 PDF 补表给出 ADC／Query-soft 的 DEV/TEST MRR、两维度的锁定动作、无条件损失、harm 与干预率；CSV 保留全部 seed/direction。没有新增显著性检验，也没有用 TEST 改报最好维度。结论是额外差值坐标未显示稳定的准确率优势，而不是冗余属于实现错误。

运行环境为 Python 3.13.11 / NumPy 2.4.2 / scikit-learn 1.7.1，原 pickle 为 scikit-learn 1.7.2。复用模型的完整 DEV/TEST 输出通过原严格 replay；另在当前版本重拟合四个完整 13D 模型，预处理状态相同、最大系数差约 `5.58e-12`、最大 decision 差约 `9.68e-11`，全部 42 个 DEV 动作向量（41 ADC+Query-soft）逐元素相同。没有把这些微小差异说成 bitwise 一致，也不把 MRR 差唯一归因于 L2，优化器／舍入因素尚未被单独消融。

## B03：清晰性质而非主要理论创新

原 Proposition 改名为 Property，明确它是线性混合恒等式和 `|tanh|≤1` 的直接推论，描述行动／分数约束。只对两模型归一化分数均有限的候选使用有限实数代数；非有限项不纳入该式。

附录 margin 条件明确为充分、非必要条件；不满足不代表真实动作一定改变排名。认证 top candidate 只需逐一对竞争者满足该充分条件，但不保证其它候选的排序。全文排序证书需要相应的全部严格候选对条件，本文没有证明所有 query 都满足。小动作仍可造成 RR 从 1 到 1/2 的既有反例和实际 harm 报告均保留，另补充“充分条件未满足但实际排序不变”的测试。

## 验证与版本

43 项针对性测试通过，包括原信息边界、匹配动作和风险算术测试。`feature_dimension_source_manifest.json` 绑定本轮协议、代码、来源、锁、结果和两张表；`.gitignore` 排除拟合模型与逐 query 大文件，`.gitattributes` 保持 hash 绑定文件的原始字节。论文构建和全稿校验使用新增 `feature_dimension_review_v1`，并保留 `raw_score_finiteness_certified=false`。

最终 PDF 为 47 页、41 张表、4 张图和 13 条参考文献，编译无未解析引用或 overfull 警告；全稿校验通过，核对 2,611 项来源快照。已逐页目视检查本轮修改涉及的公式、异常分数说明、margin 条件及两张维度对照表。新增结果只提交小型审计记录与汇总表，不包含模型、逐 query 缓存或 PDF 大文件。
