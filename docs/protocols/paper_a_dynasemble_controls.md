# D01–D03：DynaSemble 失效对照与修复实验协议

版本 `dynasemble_controls_v1`，2026-09-10。此为已查看历史 TEST 后的问题驱动补充实验，不宣称新的盲测确认。当前尚无新服务器实验结果，D01/D03 不标记关闭。

## 1. 来源与明确差异（D02）

论文：[ACL 2024 DynaSemble](https://aclanthology.org/2024.acl-short.20.pdf)，§3、§4，PDF 第 2–4 页。代码固定为 `dair-iitd/KGC-Ensemble@48d66b915f64899798f736129fa8c4d0a40fdb78`。本地干净 checkout 的逐文件 SHA 与固定链接见 `docs/audits/dynasemble_source/source_manifest.json`。

| 项目 | 论文 | 公开代码（固定 commit） | 既有迁移 R0 | 新实验的选择 |
|---|---|---|---|---|
| 网络 | 每模型 MLP；双模型宽度 16、三模型 32 | `selector.py:28–36`：Linear→Linear→ReLU→Linear→ReLU，首层后无 ReLU | 4→16→16→1；369 参数 | 同一骨架；R3/R4 单独标明末端 Softplus |
| 初始化 | §4 笼统描述 MLP weights uniform [0,2]；未逐层说明 bias | 仅隐藏循环中的 Linear.weight 在 init 非零时 uniform；首层、输出层、所有 bias 默认初始化 | 仅第二个 Linear.weight uniform [0,2] | R1 相同；R2/R3/R4 明确改变输出 weight/bias，见下一节 |
| 特征 | 拼接 mean、variance | `selector.py:85–105` 返回 `1−mean`、`torch.var`（默认无偏）；top-k 等计算但不返回 | 返回 1−mean、无偏方差 | 保留代码实际返回的四维特征 |
| 分布范围 | 完整实体集合、逐 query min-max | 先对完整 scorer 分布 normalize/feature，再在 training 时 gather(t_index)；NBF 分支未先 gather node features | 训练先抽样 gold+负例，再 normalize/feature；推理为完整未过滤分布 | 修正为训练/推理均完整未过滤分布；仅 loss gather 采样候选。独立保留 R0 |
| 非有限/常数行 | 未给完整边界规则 | 直接除 max-min，无显式常数行保护 | 有有限值统计、常数行报错 | 复用 R0 边界保护，不把这一点称为上游逐字复现 |
| 系数 | 固定文本模型系数 1，学习其他模型 | 配置 flags 决定；NBF/Sim/RotatE 等可分别学习或固定 | 学习 M-Hyper，固定 NativE/AdaMF 为 1 | R1–R3 保持；R4 学习 secondary、固定 primary；特征顺序 A/B 不换 |
| Loss | 负样本 hinge 求和，margin m；提到 CE 也可用 | `train_selector.py:86–105` MultiMarginLoss(margin=adversarial_temperature)，`loss=margin` 分支；默认 reduction 按候选数和 batch 平均 | margin=2、默认 MultiMarginLoss | 同样保留未归一化系数和 margin=2；不把分数改成凸组合后继续使用不可满足的 margin |
| 学习率 | Adam 5e-5 | optimizer 来自 YAML；检查的配置为 Adam 5e-5 | Adam 5e-5 | 固定版相同；R3 仅 DEV 网格 {1e-5,5e-5,1e-4} |
| 负样本 | 写 10,000 negatives | YAML num_negative=9999，另加第 0 列 positive；strict mask、均匀有放回采样 | 9999+gold；本地独立随机流 | 同数目、同采样语义；随机实现不同，不承诺上游随机序列逐位一致 |
| batch/配置 | 未逐数据集列出全部 YAML 设置 | 当前 WN18RR/FB15k237/CoDex YAML 为三模型、宽度32；batch 为16/24/16 | 双模型宽度16，batch16 | 明确为按论文双模型宽度和本项目固定 batch16 的迁移，不称“所有 YAML 超参数照搬” |
| 训练 split | validation | `train_selector.py:47,431,444` 从 dataset[0] 构建训练三元组；`nbfnet/util.py:149–160,211–220` 默认映射原 train split | 明确 DEV 正监督 | 按论文意图使用 DEV。不能猜测作者是否曾在外部重排数据文件；检出的默认路径与文字不一致需披露 |
| epoch/checkpoint | 单轮收敛 | 按配置轮数保存 epoch checkpoint，验证并记录 best_epoch；函数结束未自动重新加载 best；测试分支也不等于加载该记录 | 一轮后保存最终 selector | R1/R2/R3-fixed 一轮；R3 内层 DEV 选训练轮数后 full-DEV 重训，无 TEST early stopping |
| 负过滤与评估 | filtered evaluation | strict sampler 使用训练 data 图；评估可使用全事实图 | 修复后训练负过滤 TRAIN∪DEV；评估 TRAIN∪DEV∪TEST | 内层训练 TRAIN∪DEV-fit；full refit TRAIN∪DEV；内层 DEV 评估 TRAIN∪DEV，且 DEV loader 不打开 TEST；最终 TEST 评估全事实 union |

两份迁移 diff 是实际执行的 normalization/MLP 核心片段比较；框架装载、采样、数据分配、角色及 checkpoint 的额外差异由上表说明。R1 名称表示核心实现对齐，不声称在新模型/数据集上完全重现原论文实验。

初始化还有随机数调用顺序差异：上游先构造首层/隐藏层、uniform隐藏权重、再构造输出层；R0先构造全部层，再uniform隐藏权重。同seed下参数并不相等，尽管各层初始化分布相同。R1按上游顺序执行，单测对照上游首个MLP的实际初始参数、前向、loss和梯度；不把这项差异隐去。

## 2. 冻结实验矩阵

四组：MKG-W/DB15K × M-Hyper+NativE / M-Hyper+AdaMF-MAT。原基础模型 checkpoint 不训练；路径见 `configs/paper_a_dynasemble_controls.json`。

| 版本 | 激活、初始化及训练 | 目的 |
|---|---|---|
| R0_historical | `information_boundary_v2/*/dynasemble` 的现有结果，三个原配对 seed 全保留 | 历史迁移失败案例，包括已查出的 sampled-feature 差异 |
| R1_source | 完整分布特征；原代码初始化/末端 ReLU；lr5e-5、一轮 | 源码对齐后的固定对照 |
| R2_init | R1 的输出 weight 改为 uniform[-1e-3,1e-3]，bias=1；其他层不变；一轮 | 输出区间初始化修复 |
| R3_softplus_fixed | R2 的末端激活改为 Softplus，输出 bias=log(expm1(1))，初始 learned weight 近1；lr5e-5、一轮 | 隔离平滑激活，不混入调参收益 |
| R3_softplus | 同 R3-fixed，独立 DEV 选择 lr/训练轮数 | 健康动态强基线候选，不保证其表现或健康标志必通过 |
| R4_swap | 同已选 R3 配置/种子/训练预算，固定 A 系数1、学习 B 系数 | 系数对象交换诊断，不能根据 TEST 选择方向 |

R2/R3 的小非零输出 weights 避免初始所有隐藏层梯度都被人为截断；两个激活初始有效比例近 0.5，精确分布写入诊断。Softplus 没有 ReLU 的严格零区，但仍可能饱和或缺乏有效更新，不能仅因权重大于零宣称健康。

每个新版本使用基础 seed {1,2,3} × selector seed {11,23,37} 共9个模型/组合。R0仍报告原配对3个运行，并标注其随机性设计不同；不复制成9个“独立运行”。每组最终45个 selector，四组合共180个，全部报告。

## 3. DEV 选择与信息边界

三折、按 relation 分层 SHA256 排序并轮转分配原始三元组；同一原三元组的 head/tail 和各基础 seed 同折。只有 R3-softplus 搜索 lr {1e-5,5e-5,1e-4}，每条轨迹最多10轮，在 {1,3,5,10} 评估；所有 fold/base/selector seed 都进入 pooled held-out MRR。每组81条 CV 轨迹，checkpoint 共用同一轨迹，不能删除失败 seed 或选最佳 selector seed。MRR 并列优先更少轮数，再更小 lr。

选出每组一个配置后，全 DEV 从相同预定 selector seed 重新训练。R4继承R3配置，目的为受控角色诊断，不称独立充分调参的反向最优基线。full-DEV 记录标为 resubstitution，不用来二次选择 checkpoint。四组 DEV lock 齐全后才开始任一 TEST 评分。更改源码、协议、配置、基础资产或数据内容会触发锁/缓存校验失败；不得通过 rebind 绕过，新设计需新目录并重新披露。

新 DEV 是 strict TRAIN∪DEV 评估，与 R0/旧 ADC 开发选择曾采用的全事实过滤不同；本次报告明确区分，不把 DEV 数值直接当相同协议比较。最终 TEST 使用相同完整实体和 TRAIN∪DEV∪TEST 过滤，并与既有 ADC export 的基础端点逐查询核对。gold 仅进入训练监督、loss 候选采样和独立评估 mask；完整分布 normalization/features 不接收 gold/filter。

## 4. 训练诊断及关闭条件

每个 CV 和 full-DEV 运行保存初始及逐 epoch：preactivation、learned weight、有效 primary 比例的 min/p05/median/p95/max/std，零权重与负 preactivation 比例，逐层梯度 mean/max/零梯度 batch 比例、参数更新范数、loss首末/均值；评估 checkpoint 同时记录 held-out MRR及权重诊断。完整失败日志和 trace 保留，非有限 loss/gradient 会停止，不自动丢弃重抽。

训练健康需综合梯度、更新、loss轨迹及 held-out 表现判断；不能仅检查“非零权重”，也不要求超过Global才算训练正常。若正常优化趋近静态策略，如实报告。完成所有运行只标为 `experiments_complete_requires_health_and_scientific_review`，不自动关闭 D01。

D01：存在至少一个完成合理DEV选择、优化有效的动态对照，与失败R0并列；D02：来源表/diff/数值与梯度一致性单测齐全；D03：角色交换结果完整，系数端点和排序端点表述正确。任何主对照塌缩仍须保留，不能只报恢复的seed。

## 5. D03 的数学和统计口径

原方向 `S=w*A+B`，primary系数比例 `w/(1+w)`；交换方向 `S=A+w*B`，primary比例 `1/(1+w)`。Softplus 下有限 logit 的 w>0，两方向都不精确达到0/1系数端点；角色交换改变接近强模型端点所需的权重方向，不能声称它保证精确端点。有限w仍可能给出与A完全一样的有限候选集弱序，取决于原排序间隔和B的扰动。

每个 TEST query 报告 gold rank 是否与 raw primary 一致；另按 observed-query hash 确定每方向32条观测，比较整个未过滤候选集合与 normalized primary 的完整弱序（包括 ties）。该64条探针的完整排序结果不外推所有query；gold rank一致也不等于完整排序一致。probe选取不使用gold分数或策略结果。

主表含 R0/R1/R2/R3-fixed/R3/R4、Global、Query-soft、ADC。报告MRR、相对Global与ADC增量、harm、每base×selector×direction结果。三元组 bootstrap 为10,000次、seed20260910：先对selector重复取RR均值，再保留每三元组原来的六个base×direction观测；区间只衡量三元组抽样不确定性，不冒充训练随机性区间。harm在各实际selector观测上计算，不在平均RR后计算。随机性用完整矩阵额外呈现。

## 6. 服务器运行（PowerShell）

本机仅源码审计、synthetic CPU 单测和 Plan；真实候选评分/训练/大bootstrap在服务器。需要已有 processed 数据、冻结基础模型和 B07/C04 v2 导出。本流程缓存 float32 完整 min-max候选矩阵，四组合全部DEV+TEST约32 GiB，建议至少60 GiB可用磁盘；不需要加载整个缓存到内存。单GPU串行执行，不同时运行相同输出目录。

在服务器仓库根目录、已安装项目依赖且CUDA可用的Python环境中：

```powershell
git pull --ff-only
powershell -ExecutionPolicy Bypass -File scripts/run_paper_a_dynasemble_controls.ps1 -Stage Plan -Python python
powershell -ExecutionPolicy Bypass -File scripts/run_paper_a_dynasemble_controls.ps1 -Stage All -Python python
```

如使用Conda环境，先激活，或将 `-Python python` 换成该环境 `python.exe` 完整路径。`All`依次执行Dev→Test→Summarize→Bundle；可独立指定 `-Stage Dev`、`Test`、`Summarize`、`Bundle`。中断后重复同一命令，已完成且哈希一致的cache/fit/test会复用，未完成fit从本次预定初始化重跑，不换seed。CUDA不可用则直接报错，不自动转CPU。源码、commit和Python/PyTorch/NumPy/CUDA环境一并锁定，运行中不要升级或pull新提交。

输出 `outputs/paper_a_safe_correction/dynasemble_controls_v1/`：首先回传 `summary.csv`、`seed_direction.csv`、`health_summary.csv`、`summary_manifest.json`、`completion.json`、每组 `dev_cv.csv`、`dev_lock.json`、`fits/**/trace.json`、`fits/**/complete.json` 及 `test/*.json`；后续逐query核验需要 `test/*.csv`。候选cache无需先传回。运行前仅有计划和代码，不应将缺少服务器结果写成实验关闭。

`All` 最后自动生成 `outputs/paper_a_safe_correction/dynasemble_controls_v1_review.zip`，包含上述证据、逐查询结果和selector，排除约32 GiB的候选cache。运行成功后优先将这个zip传回。
