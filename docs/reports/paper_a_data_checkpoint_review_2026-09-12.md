# C01–C03：数据及 checkpoint 可比性核验

日期：2026-09-12。审查对象是当前 `information_boundary_v2` 结果和其中实际使用的 18 个基础模型。没有训练模型、执行全候选 scorer、修改 split/checkpoint，或用 TEST 选择参数。

## 结论与关闭范围

| 风险 | 本次结论 | 状态 |
|---|---|---|
| C01 有效 split 与公开 split 不一致 | 原文件哈希、构造代码、训练 manifest、全部导出三元组已形成完整链条；DB15K DEV 是从 TRAIN 留出，TEST 没有另行抽样 | 最低关闭条件已满足 |
| C02 实际数据元信息 | 实际统计、对齐/缺失策略、checkpoint 特征已核验；服务器六份原始输入指纹一致；用户下载说明与 MMRNS 官方仓库、发布目录对应 | 使用发布的预计算特征这一设定下，最低报告条件已满足；原始编码器重建仍是限制 |
| C03 基础模型低于公开值 | 公共论文/固定代码/当前迁移的差异已列明；六个 M-Hyper 跑满 200 epoch，最佳 DEV 位于 15–55 epoch；18 个模型的导出 DEV 与日志最优值吻合 | 当前协议下 checkpoint 合理性的最低证据已满足；不声称公开 SOTA 复现 |

**未解信息不以猜测填补。** 下载来源已补充为 MMRNS 官方仓库链接的预计算 HDF5 发布目录，见文末补证。文本特征 manifest 仍明确写 `unknown (upstream HDF5 metadata not provided)`；BEIT 图像编码器名称也仅来自文件名。原始编码器具体版本和抽取参数尚不能恢复，因此不声称从原始图文重新抽取特征可复现；这不改变已明确采用发布特征、记录精确文件哈希的实验设定，也不触发 TEST 追分。

## C01：数量流水账

公开列来自 [M-Hyper Table 7](https://aclanthology.org/2026.acl-long.1289.pdf)，文件列来自本次重新读取并核验 SHA-256 的 NativE 固定版本 `c72291c3d571b36c754897935d21718aa4568fe7`。两者不混用。

| 数据集/阶段 | 实体 | 关系 | TRAIN | valid / DEV | TEST |
|---|---:|---:|---:|---:|---:|
| MKG-W 论文公布 | 15,000 | 169 | 34,196 | 4,276 | 4,274 |
| MKG-W 所用原文件 | 15,000 | 169 | 34,196 | 4,276 | 4,274 |
| MKG-W 本研究有效 | 15,000 | 169 | 34,196 | 4,276 | 4,274 |
| DB15K 论文公布 | 12,842 | 279 | 79,222 | 9,902 | 9,904 |
| DB15K 所用原文件 | 12,842 | 279 | 79,222 | 9,904 | 9,902 |
| DB15K 本研究有效 | 12,842 | 279 | 71,300 | 7,922 | 9,902 |

这不是将 9,904 条 TEST 去掉两条得到 9,902 条。所用仓库的 `test2id.txt` 原本就是 9,902 条。没有证据证明它与论文表中另一列是同一组三元组，因此没有将差异解释为“论文写错”或擅自交换标签。

DB15K 的 `valid2id.txt` 有 9,904 行、7,481 个唯一三元组，即 2,423 个首次出现之外的重复行。唯一三元组中有 6,584 个与 TRAIN 重合、824 个与 TEST 重合；TRAIN 与 TEST 不重合，另有 73 个仅在原 valid 中。预处理**整份排除原 valid**，包括这 73 个，而不是仅删除重复行后继续用余下部分。上述判断限定于哈希固定的镜像文件，不外推到作者论文实际使用的数据版本。

替代 DEV 从原 TRAIN 取 10%，`round(79222 * 0.1)=7922`。代码 `build_db15k_train_holdout` 使用关系配额、seed=2025 的稳定 SHA-256 排序，并保留 DEV 实体与关系在 TRAIN 中的支持；必要时 seed=2026 的第二排序填充配额。TRAIN/DEV 保留源 TRAIN 内的相对行序，TEST 原样保留。所有有效 split 无重复、无交集；不因模态缺失删除任何实体或三元组。

本次重建了六个有效 TSV 的精确哈希（包括 CRLF 行尾），与所有 18 份训练配置中的 manifest 一致。OpenKE 的 `(h,t,r)` 转为 canonical `(h,r,t)`，ID 映射保持不变。逐行压缩账本记录 source split、原数据行序、三元组、canonical split/行序和排除原因。导出通过三元组与 seed/direction 联接，不误把字符串 `query_id` 当作数字行号。

六组配对、每组两个 split，共 12 个导出通过：MKG-W 每组 DEV/TEST 为 25,656/25,644 行，DB15K 为 47,532/59,412 行，均等于三元组数 × 3 seed × 2 方向。缺失、额外和重复观察均为零。

## C02：实际模态输入

| 数据集 | 图像实体数/覆盖率 | 图像维度 | 文本实体数/覆盖率 | 文本维度 |
|---|---|---:|---|---:|
| MKG-W 有效 | 14,224 / 94.8267% | 383 | 13,333 / 88.8867% | 384 |
| DB15K 有效 | 12,818 / 99.8131% | 383 | 9,078 / 70.6899% | 384 |

作为对照，M-Hyper Table 7 的 MKG-W 图像/文本覆盖数是 14,463/14,123；DB15K 特征维度是图像 4096、文本 768。相同数据集名称不代表相同模态输入。

数据原始来源补引 [MKG-W / Relation-enhanced Negative Sampling](https://doi.org/10.1145/3503161.3548388) 和 [DB15K / MMKG](https://arxiv.org/abs/1903.05485)；实际获取的 split/mapping 明确指向固定 NativE 镜像。OpenBG 不在当前纠正后数值结果中，不能拿它的统计充当本轮实验数据。

实际文件是 `MKG_W_img_BEIT_16-224.h5`、`MKG_W_description_sentences.h5`、`MMKB_img_BEIT_16-224.h5`、`MMKB_description_sentences.h5`，完整哈希在生成的 split manifest 中。MKG-W 经固定 QID→title crosswalk 对齐；DB15K 文本经资源 basename、图像经 SameAs→Freebase MID 对齐。四个 DB15K 实体关联多个图像 key，跨 key 的原始行一起取均值。禁止按 HDF5 枚举顺序对齐。

缺失文本/图像分别为 MKG-W 1,667/776、DB15K 3,764/24。缺失行填零，显式 mask 来自 key 可用性，不能由数值为零推断缺失。当前 general adapters 屏蔽不可用模态的投影，M-Hyper 同时屏蔽独立模态嵌入；不添加 learned missing vector。结构表示、实体候选和三元组仍保留。

原 HDF5/crosswalk 不在本地，但 **18 个实际 checkpoint 保存了 `text_feat`、`img_feat`、`has_text`、`has_img`**。本次全部加载到 CPU，验证四个 tensor 的 dtype/shape/value 哈希与 manifest 一致，复算覆盖数、有限性和缺失行零值。这是对模型实际输入的直接核验；不是声称重做了原始特征抽取。

## C03：训练、checkpoint 与公开实现

公开代码锁定 [M-Hyper cb390b6a54e08a6ac76d664cf2c2f6e1ff13ff76](https://github.com/zjukg/M-Hyper/tree/cb390b6a54e08a6ac76d664cf2c2f6e1ff13ff76)。本地 checkout 的 HEAD 与锁相符；所读文件哈希进入审计。

| 维度 | 固定公开实现/材料 | 当前迁移 | 可比性含义 |
|---|---|---|---|
| split | 论文 Table 7 与所用镜像的 DB15K valid/TEST 计数标签不同 | 独立 TRAIN holdout DEV，保留镜像 TEST | DB15K 非相同 split 复现 |
| 模态 | 公开输入为 `img_features.pth` / `text_features.pth`；论文统计见上 | 共享 canonical HDF5 输入与显式缺失 mask | DB15K 维度、MKG-W 覆盖不一致 |
| PCA | `models.py` 在输入的全实体矩阵上 `fit_transform` | 只在 TRAIN-visible 且对应模态可用的实体上拟合，再 transform | 初始化信息范围不同 |
| 优化配置 | README 为 rank128、Adagrad、LR0.1、batch1000、wN3=0.005、200epoch、每5epoch评估 | 六个 M-Hyper 使用这些值 | README 配置对齐；不等于论文选优配置被证明一致 |
| loss | CE + weighted N3 + reconstruction/consistency；返回值未含已计算的 independence 项 | 保留这些选择，也保留 reconstruction 最后一项 structural target | 代码选择透明，不声称每个论文公式完全一致 |
| 训练样本 | 正向+逆关系、全实体 CE | 同；generic `neg_ratio=10` 字段在此引擎不生效 | 不是 10 负样本训练 |
| checkpoint | `datasets.py` 把 valid 读为 `test.pickle`，`run.py` 按 valid 最大 MRR 保存 | 独立 canonical DEV，strict improvement；训练时 `run_test=false` | 只证明该代码行为；不能据此断言论文用了 TEST 选优 |
| filter/排序 | train/valid/test union；保存 gold 分数后用有限 sentinel 屏蔽 gold 和其他答案；计数 `>=` | canonical union，保留当前 gold，屏蔽其他答案；计数 `>` | 双向 filtered 主体相同，具体事实集合及 ties 不同 |

历史 base-model DEV 评估仍使用 canonical TRAIN∪DEV∪TEST 事实作为 evaluation filter，与当前 ADC 的历史 DEV 合同一致。这不把 TEST 三元组当训练正例；也不能误写为训练阶段从不打开 TEST 文件。新 DynaSemble 控制的严格 DEV 过滤范围是另一条已披露的边界。

M-Hyper 六个 run 的最佳 DEV/最终 epoch：

| 数据集 | seed | 最佳 epoch | 最佳 DEV MRR | 最终 epoch | 最终 DEV MRR |
|---|---:|---:|---:|---:|---:|
| MKG-W | 1 | 15 | 0.352740 | 200 | 0.349935 |
| MKG-W | 2 | 25 | 0.352830 | 200 | 0.350654 |
| MKG-W | 3 | 20 | 0.352896 | 200 | 0.349855 |
| DB15K | 1 | 25 | 0.384129 | 200 | 0.378360 |
| DB15K | 2 | 40 | 0.382037 | 200 | 0.379183 |
| DB15K | 3 | 55 | 0.381594 | 200 | 0.375938 |

所有六个 M-Hyper 均有完整 40 次评估，最佳点后至少 29 次没有更高 DEV；首末记录 loss 均下降。另 12 个 NativE/AdaMF run 也满足日志中的早停规则，18 个 checkpoint 参数有限。完整设置与全部 seed 诊断已进入 PDF，而不是只呈现最好的 seed。

18 个模型从 corrected DEV 导出复算的 standalone MRR 与日志最大值的最大误差是 **4.1973866e-8**。预先使用绝对容差 1e-7，原因是日志以 float32 reciprocal rank 求均值、导出以 float64 求均值。24 份后续 DynaSemble DEV/TEST cache manifest 的模型/config 哈希均与当前 checkpoint 相符。checkpoint 是裸 state_dict，无 epoch metadata，因此证据链是保存代码规则 + 日志最大值 + 导出 DEV 一致 + 后续 cache 的 checkpoint 哈希；不能声称重新执行了训练过程。

多份目录里的文件都叫 `metrics_seed1.csv`，但 `config_merged.json` 中 seed 分别为 1/2/3，与冻结 run map 一致。本次不通过 metrics 文件名推断 seed，也不改名破坏来源引用。

公开/当前 M-Hyper TEST MRR 分别为 MKG-W 0.3702/0.359721、DB15K 0.4125/0.374567，描述性差值为 -0.010479/-0.037933。存在上述协议差异，不能直接判为同协议复现失败；没有控制重跑也不能声称某一差异解释了多少分数。当前 “strong primary” 已限定为同一输入协议下所测专家池内较强，不再以公开 SOTA 能力作担保。

## 实现、验证及服务器补证指令

生成物目录：`outputs/paper_a_safe_correction/data_checkpoint_review_v1/`。包括 `audit.json`、两个完整 split manifest、两个 source-row 压缩账本、`split_counts.csv`、`export_coverage.csv`、`modality_statistics.csv`、`checkpoint_diagnostics.csv` 和 `training_settings.json`。最大的单文件约 1.08 MB；不提交 raw HDF5、checkpoint 或 score 矩阵。

主脚本 `scripts/audit_paper_a_data_checkpoint.py` 只读输入，不执行 scorer；`scripts/build_paper_a_data_checkpoint_assets.py` 生成五张正式表格及来源 manifest。21 项测试涵盖重复/漏样/错 gold/错三元组/错 RR 拒绝、split 哈希顺序和行尾、非最佳 epoch 拒绝、训练中断拒绝，以及现有 preprocess/PCA/逆关系/候选一致性测试。最初 pytest 临时目录被 Windows 沙箱拒绝访问；同组测试在正常权限下运行后 **21 passed**，没有修改测试规避错误。

首次 C01–C03 核验后的 PDF 为 32 页、28 张表、3 幅图、9 条参考文献；当时全部 2,494 个来源哈希、表格依赖、审计状态、引用和页面边界检查通过。数据流水账、模态表和 checkpoint 页已做视觉检查，编译无 overfull 或未解析引用。服务器与下载来源补证在此基础上继续增加来源绑定。

本轮不需要 GPU 重训。此前用于收集服务器原始 HDF5 元信息的 PowerShell 指令如下，现已执行并回传，无需重复运行：

```powershell
Set-Location 'G:\mmkg-project-research'
git pull --ff-only
python scripts/collect_paper_a_feature_provenance.py --output outputs/paper_a_safe_correction/data_checkpoint_feature_provenance.json
```

该脚本只读取文件哈希、HDF5 attributes 和少量对象的 shape，不执行训练、评分或 TEST 选参。回传结果及其证据范围见下一节。

## 服务器回传补证

已核验 `outputs/paper_a_safe_correction/data_checkpoint_feature_provenance.json`。验证脚本不只读取其中的 `status` 或 `hash_matches`，而是将实际回传的 SHA-256、字节数逐项与**此前锁定的** `docs/EXTERNAL_SOURCES_LOCK.json` 及两个训练 split manifest 比对，并要求 dataset/kind 恰好组成六个预期条目，拒绝缺项、重复和额外项。

| 数据集 | 文件类型 | 字节数 | HDF5 root keys | 抽查对象特征维度 | 与原锁一致 |
|---|---|---:|---:|---:|---|
| MKG-W | 图像 HDF5 | 554,792,032 | 14,951 | 383 | 是 |
| MKG-W | 文本 HDF5 | 78,853,653 | 14,149 | 384 | 是 |
| MKG-W | QID/title crosswalk | 834,296 | — | — | 是 |
| DB15K | 图像 HDF5 | 551,681,896 | 14,845 | 383 | 是 |
| DB15K | 文本 HDF5 | 84,013,072 | 9,083 | 384 | 是 |
| DB15K | SameAs 对齐文件 | 875,463 | — | — | 是 |

四个 HDF5 的 `root_attributes` 全为空，每个文件抽查的前五个对象也都没有 attributes，共 20 个对象。root-key 数与原 manifest 的 HDF5 key 统计一致；它不是对齐后的实体覆盖数。抽查对象的第二维与实际 383/384 维特征一致。

**该 JSON 能确认原始文件身份记录一致，不能单独确认编码器或下载来源。** 哈希和字节数由服务器回传，本地独立比对旧记录；原 HDF5 未回传，不能声称本地重新散列了这些源文件。也不能把根节点和 20 个对象为空外推为文件中任何位置都不存在元信息。OpenBG 自己的文本缓存脚本不适用于 MKG-W/DB15K，不能用它的模型名填补这里的未知项。下载来源通过用户随后提供的链接和下述官方材料另行确认。

新增 `scripts/audit_paper_a_feature_provenance.py` 和 `data_checkpoint_review_v1/feature_provenance_audit.json`；正式来源 manifest 绑定原始回传 JSON、收集脚本、验证脚本、下载来源记录与验证结果。新增测试拒绝“自报 expected/actual 相同却不符旧锁”的哈希、错误字节数、缺失/重复条目、错误维度、错误 key 数、TEST 选参标记及不匹配的下载目录/文件映射；与前轮数据审计测试合计 **20 passed**。

C01/C03 的既有关闭结论不变，实验分数及 checkpoint 未改变。无需重复同一 metadata 收集指令，也没有由此触发重训的依据。

## 用户下载说明与官方发布目录

用户确认四个 `.h5` 直接从 [MMRNS-Datasets 文件夹](https://drive.google.com/drive/folders/1sFC-P9RKnikqNXjmLcj0IX7x5zvRs-Yj) 下载。进一步核验发现，[MMRNS 官方 README](https://github.com/quqxui/MMRNS/blob/a4a41fc8f991df768e1dceb1b113685641e70693/README.md#datasets) 的数据下载链接指向**完全相同的 folder ID**。本次固定的源码 commit 是 `a4a41fc8f991df768e1dceb1b113685641e70693`；这表明发布项目归属，不表示这些输入由本研究重新编码。

[emb_data 子目录](https://drive.google.com/drive/folders/1MAzRC0kEbnwlGe7EIZdqWyhUfDQotndB) 确实列出全部四个文件，目录显示大小与服务器字节数换算后的值一致：

| 文件 | 发布目录显示大小 | Google Drive 文件 ID |
|---|---:|---|
| MKG_W_description_sentences.h5 | 75.2 MB | `1C0VrxuhzTY7m5mGSZeqmTJ1gr2ZUgbSt` |
| MKG_W_img_BEIT_16-224.h5 | 529.1 MB | `1jBW5aTV6nRSyGSy9m2DuGrBl5SjlX8PA` |
| MMKB_description_sentences.h5 | 80.1 MB | `19XMPd1nRCtKGvjEBmysux04YFVpLy0OA` |
| MMKB_img_BEIT_16-224.h5 | 526.1 MB | `16PCQGg3ZHHDVyySskaoFIoU9ykwTWp93` |

来源链为：用户确认直接下载 → 官方 README 的同一发布目录 → 目录中的四个文件名及显示大小 → 服务器实际文件 SHA-256 → 此前训练 manifest/18 个 checkpoint 的 canonical 输入哈希。目录显示的 MB 与按 `bytes / 1024^2` 保留一位小数一致。此次没有重新下载并散列远端四个大文件，因此**不把显示大小和名称相同说成独立验证了远端内容 SHA-256，也不声称查明了历史远端文件 revision**。精确使用版本仍由本研究保存的 SHA-256 固定。

`feature_download_source.json` 保存用户下载说明、官方仓库/commit/README、文件夹及四个 file ID、显示大小、既有文件哈希、网页观察哈希和证据限制。此处建立的是使用发布特征的可追溯来源。C02 要求的数据统计、模态特征来源与缺失值策略现已齐全，其最低报告条件可关闭；编码器具体 checkpoint、版本和生成配置仍未知，作为**特征抽取层面的复现限制**保留，不能据此虚写 BERT 或任何具体模型版本。

补证后的最终 PDF 为 33 页、28 张表、3 幅图；全部 2,499 个来源绑定检查通过，编译无溢出或未解析引用。更新页已重新渲染并视觉检查，20 项相关单测通过，原有各轮数值审计仍通过。
