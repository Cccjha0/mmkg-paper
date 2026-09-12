# B10 回传核验：端点并非无条件等价，主 pair 的固定策略效应不变

日期：2026-09-12。前置记录：[B08/B10 代码核查](paper_a_inference_endpoint_contract_review_2026-09-12.md)。本轮没有模型训练、基础模型评分、TEST 选参或历史结果替换。

## 结论与关闭口径

**共享 evaluator 的端点一致性已通过全量核验；标准化端点的无条件等价性被实际数据否定。** B10 应以“明确执行约定、保留全部差异并量化影响”收束，不能写成“标准化不会改变端点 rank”。

原方法保留 exact alpha=0/1 时的 raw standalone rank，内部权重使用标准化混合。四个主 pair 的固定 TEST ADC−Global 逐行 RR 差在替换端点后完全不变（检查容差 1e-12，观察到最大差为 0），因此这四个既有相对效应不依赖本次发现的端点 rank 差异。附加 pair、其他比较及重新训练/选参不享有同样结论，均明确披露。

本次核验不要求替换已报告的原协议主结果，也不采用标准化端点作为新的“更好”TEST 协议。如以后改成另一套端点、监督或选参定义，必须另行运行完整 DEV 拟合/选择流程，不能把当前固定策略敏感性当成该流程的复现。

## 回传及来源核验

- 服务器提交：`19b4d39e704c88d6d716ecbd3df4c23282ad1beb`。
- 18 个 checkpoint、72 个 DEV/TEST/方向 cell、474,732 个 expert-query 观察，146 个小型回传文件。
- 76 个源码/配置/协议来源：52 个字节完全一致，24 个 Python 文件仅换行格式不同；不接受其他源码差异。
- checkpoint、config、canonical split 顺序、每份差异文件哈希、样例、计数、RR 总和守恒均通过独立核验。
- 运行时：PyTorch 2.6.0+cu126，CUDA 12.6，NVIDIA A100-SXM4-40GB。
- raw export 对 shared rank、alpha=0/1 对 shared rank：**全部零差异**；原始分数异常行：**零**。
- normalized 对 raw rank：**1,293 行差异，约 0.272%**，全部使 normalized RR 上升。不能只报告前述零差异而省略此结果。

差异以 `(dataset, model, seed, split, direction, triple)` 对应历史输出，所有 1,293 行在两个相关 pair 中都找到，raw rank 均完全匹配。每个历史 cell 的 raw RR 总和也匹配，normalized RR 增量由差异行独立重算一致。返回未包含全部原始候选分数，这不是任意运行时/批次原始 score 字节相同的证明；也不能仅凭总和证明所有未返回行的历史 raw rank 都逐行相同。

## 标准化端点差异的规模

下表 ΔMRR 为 normalized−raw，不是 ADC 效应。

| 数据集 / 模型 | DEV 差异行 | DEV ΔMRR | TEST 差异行 | TEST ΔMRR |
| --- | ---: | ---: | ---: | ---: |
| MKG-W / M-Hyper | 0 | 0 | 1 | 约 3.0e-12 |
| MKG-W / NativE | 592 | +0.008601 | 590 | +0.008694 |
| MKG-W / AdaMF-MAT | 34 | +0.000663 | 20 | +0.000390 |
| DB15K / M-Hyper | 1 | 约 2.8e-9 | 2 | +0.00000842 |
| DB15K / NativE | 12 | +0.000126 | 23 | +0.000188 |
| DB15K / AdaMF-MAT | 6 | +0.00005435 | 12 | +0.000101 |

MKG-W NativE 占 1,182 行。全部差异中有 924 行从 rank 2 变为 1；但回传只记录 rank，没有候选级分数和 reference 的具体差距，不能推断每次变化都由 gold 候选自身的舍入导致。论文保留一般浮点反例，并报告实际差异，不猜测具体候选机制。

## 对原固定策略的影响

固定原来的模型、参数、anchor 和逐行 action，只在 alpha 恰为 0 或 1 时替换为已审计 normalized rank，内部权重的缓存 rank 保持原样。Global、ADC、Query-soft、Relation 全部报告；Equal、RRF 的绝对 rank 不变，但其相对 Global 的差仍可能随 Global 改变。DEV 此处为 full-fit replay，不是 OOF 泛化估计。

| Pair | 原 TEST ADC−Global | 替换端点后 | ADC−Global 差改变行数 |
| --- | ---: | ---: | ---: |
| W-N | +0.002446649 | +0.002446649 | 0 |
| W-A | +0.001013912 | +0.001013912 | 0 |
| D-N | −0.000081223 | −0.000081223 | 0 |
| D-A | +0.000211554 | +0.000211554 | 0 |
| W-NA | +0.000138920 | +0.000190914 | 4 |
| D-NA | −0.000014872 | −0.000031703 | 2 |

四个主 pair 的 ADC 与 Global 在受影响的端点观察上获得相同 RR 增量，所以相对效应逐行抵消；不是“所有绝对 MRR 都没有变化”。既有三正一负的主 pair 点估计保持。附加 pair 的改变量不能忽略，且原来的不显著结论仍按原报告保留，不为替代设置新增或转移显著性主张。

其他对照亦不被省略：W-NA Relation−Global 从 −0.000162725 变为 +0.000163738。D-N 上 ADC 与 Relation 的微小点估计顺序也会变化。WN 的 Relation 仍高于 ADC，Query-soft 与 ADC 的既有损失取舍仍保留。完整正式补充 CSV 覆盖六个 policy、六个 pair、两个 split、两种端点约定：144 个汇总行和 864 个 seed/方向行，包含 MRR、相对 Global 效应、平均损失、harm 频率及发生变化的观察数。

## DEV anchor 与监督边界

仅把原 21 列 DEV rank grid 的两端替换，再使用相同 tie-break，六组中的五组最优 Global 不变；**W-NA 从 0.95 变为 1.00**。这个候选只作为 DEV 诊断，不用于产生新的 TEST 结果。

若连 standalone-winner 监督定义也改用 normalized rank，则六组 winner/tie 类别变化分别为 429、34、12、5、489、17 行。因此不能从固定 TEST 策略的稳定性推导“整个方法重训后等价”。本轮不改监督、不重训、不重选 β/τ，也不选择更有利的 TEST 约定。

## 实现和交付

- 独立核验及敏感性：`scripts/review_paper_a_endpoint_contract.py`。
- 表格生成：`scripts/build_paper_a_endpoint_assets.py`。
- 新增回传测试：`tests/test_endpoint_contract_return.py`，16 项通过，覆盖假零差异、重复/缺失 cell、错误哈希身份/顺序、伪造差异行与 RR 守恒、端点精确替换和匹配 Global 参照。
- 输出：`outputs/paper_a_safe_correction/endpoint_contract_return_review_v1/`。
- 论文：方法端点定义继续保留；Discussion 加入数值约定边界；附录新增模型端点、固定策略与 DEV anchor 三张表。
- 来源清单：`paper_a_draft/endpoint_contract_source_manifest.json`。小型回传 receipt/difference JSON 作为已核验的证据保留；原始候选矩阵、checkpoint、运行目录和压缩包不新增到 Git。

论文编译没有 overfull box 或未解析引用；全稿核验通过 2,948 项来源快照、47 张表和 4 张图。当前 PDF 为 53 页，已目视检查第 18、24–26 页，新增表格为 Tables 11–13。
