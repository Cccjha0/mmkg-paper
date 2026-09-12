# C05/C06：跨模型列对齐与基础模型重建记录

日期：2026-09-12。范围为已使用的 18 个 checkpoint（两数据集 × 三架构 × 三 seed）。本轮不重训、不运行真实模型全候选评分、不选新参数、不替换历史结果。

## 结论

- **C05 最低核验条件满足**：公开两个数据集的完整 entity/relation ID 数组、三种映射哈希及生产导出路径测试。18 个 checkpoint 的完整实体支持、36 次映射核验、6 个 M-Hyper 逆关系 buffer 检查全部通过。既有 72-cell 实际评分回传无非有限分数；已有 endpoint 回传证明部署端点与共享 evaluator 的逐行一致性，仍保留规范化端点不同于原始端点的限制。
- **C06 已补齐可重建配置与可核验 checkpoint 标识**：六个 dataset×architecture 设置、18 份完整配置和 checksum、所有 seed 最佳/最终 epoch、来源归属、源码快照及服务器重建入口进入正式补充材料。旧运行的精确执行 commit 和环境无法从现存记录恢复，明确保留缺口；不把这项工作称为历史训练完全复现。

新增 16 项测试，加上已有方向、逆关系和数据合同测试，共 **22 passed**。测试使用六实体模型，没有本机强算力任务。所有结果仍为原 checkpoint 和原策略。

## C05：具体保证什么

代码链为 `read_openke_mapping` → canonical manifest / `load_openke_mmkg` → `DatasetBundle.validate` → `validate_pair` → `score_expert_block` → `direction_scorer` → 原始矩阵 → 标准化/组合 → gold-preserving filtered evaluator。`validate_pair` 比较完整字典及有序 split，而非只比较数量。生产 exporter 明确按 `torch.arange(N)` 分块；column j 对应 ID j。

| 架构 | tail | head | 内部关系 ID | 分数符号 |
|---|---|---|---|---|
| M-Hyper | `(h,r,j)` | reciprocal `(t,r+R,j)` | base 0…R−1；inverse R…2R−1，buffer 为 involution | query/entity 点积，越大越好 |
| NativE | `(h,r,j)` | `(j,r,t)` | base 0…R−1，不添加逆关系 | margin − RotatE distance |
| AdaMF-MAT adapter | `(h,r,j)` | `(j,r,t)` | base 0…R−1，不添加逆关系 | margin − RotatE distance |

MKG-W N=15,000/R=169；DB15K N=12,842/R=279。同数据集九个 run 的映射完全相同。新 `mapping_inventory.json` 记录：原始 OpenKE 文件 SHA、canonical name→ID 对象 SHA、按 ID 排序的 name 数组 SHA。相应完整数组已随提交公开。哈希序列化定义在协议中，文件字节哈希另由 source manifest 绑定，不能把不同序列化的哈希混用。

真实 checkpoint 检查所有实体 embedding、模态 buffer/mask 第一维；M-Hyper 额外验证 2R 关系和 exact inverse buffer，另两架构验证 R 关系。微型实际架构测试检查非排序候选排列、首尾 ID、跨 chunk 和 query batch、head/tail 及隐藏答案替换后的原始矩阵一致性。错位 mapping、缺少实体行、错误关系语义均被拒绝。

**没有共同子集或缺失模型候选补值策略**：当前路径要求所有实体可评分，不符合则拒绝。模态缺失是另一回事：保留结构实体候选，原始模态行零填充并带独立 mask。非有限分数也不能用来解释 ID 不齐。实际全量评分回传为 72 cells / 474,732 rows、零异常，故本轮不需要修复或重评。synthetic 数值一致容差为 atol=rtol=2e-6；精确 ID/字典/buffer 比较没有数值容差。没有声称历史未保留 raw 矩阵的 bitwise 一致性。

## C06：设置、来源与未知项

| 架构 | MKG-W / DB15K 生效配置 | 来源及本地差异 |
|---|---|---|
| M-Hyper | rank128；Adagrad LR0.1；batch1000；全实体 CE；N3=.005；init=.001；noise=.2；200epoch；每5epoch DEV | 固定 README 命令值；本地 train-visible+available PCA、共享特征/mask、canonical split/evaluator |
| NativE | dim250；margin4/12；Adam LR1e-4；batch1024；128负例；temperature2；generator LR1e-4；μ1e-4；regularization1e-5；GP=.1；cap1000/patience10 | 固定 dataset scripts 的设置；本地共享特征/mask、TRAIN-filtered sampler、DEV 早停与 evaluator |
| AdaMF | dim200/250；margin12；Adam LR2e-5；batch1024；128负例；temperature2；generator LR1e-4；μ0；regularization默认0；cap1000/patience10 | 固定 dataset scripts，而非 README 中 LR1e-4 的示例；本地冻结 raw feature buffers（上游 AdaMF 可训练）、mask、采样与 DEV 早停 |

两采样引擎 generator noise=64、hidden=512；RotatE epsilon=2，实体范围 (margin+epsilon)/(2d)、关系范围 (margin+epsilon)/d。Adam 默认 betas=(.9,.999)、eps=1e-8；Adagrad 默认 initial_accumulator=0、lr_decay=0、eps=1e-10；optimizer weight_decay=0，无 scheduler/gradient clipping。Python、NumPy、PyTorch seed=1/2/3，deterministic=false。M-Hyper `embedding.d=256`、`neg_ratio=10`、`adv_temperature=2` 不生效，不能直接照抄 generic 字段描述训练。

**AdaMF μ=0** 同时来自上游数据集脚本和当前实际配置。generator 更新仍执行，但不进入 base-model gradient。该结果不能代表完整 MAT 的贡献。这个限制原附录已披露，本轮强化来源及其含义，不悄悄更名替代或删掉结果。

**设置来源不是历史选择证明**：数值与发布脚本匹配，不等于已证明作者为何采用它。现存记录没有一份完整 base hyperparameter search ledger，因此不称六组为 DEV 调参最优。可核验的是 DEV checkpoint 选择：每5epoch完整双向 filtered DEV，strict MRR improvement 保存，平局留较早 epoch。M-Hyper固定200epoch；其余10次不改进停止（50epoch），cap1000。训练正例/负例排除只用 TRAIN；历史 DEV evaluator filter 使用 canonical TRAIN∪DEV∪TEST 已知事实，当前 gold 保留。`run_test=false` 不等于训练时从未打开 TEST 文件。

上游锁：NativE `c72291c3d571b36c754897935d21718aa4568fe7`；AdaMF-MAT `50930e9b28aed57133bedfc281391185ac9b68a4`；M-Hyper `cb390b6a54e08a6ac76d664cf2c2f6e1ff13ff76`。三个 checkout HEAD 及所比对脚本与对应 Git blob 一致。脚本的字节快照随小型审计附件保存。

本地可取得的 reconstruction commit 是 **`f2e8198d24d5dde56476fc768a56436adf72e2ba`**。66 个训练包 Python 源文件与该树一致，差异仅允许 CRLF/LF，Git blob / working bytes / normalized SHA 分开记录。这不是旧训练执行 receipt。旧 state_dict 没有执行 commit、epoch、optimizer/RNG state 或完整运行环境；不能以该新核验快照回填未知历史。18 个 checkpoint 的 epoch 仍依赖此前“严格保存代码＋日志最大值＋导出DEV误差≤1e-7＋后续缓存hash”的组合证据。

模态数据来源沿用已确认的 MMRNS Google Drive 发布 HDF5：实际四个文件和两份 alignment inputs 的 checksum 已回传并匹配。原始编码器版本和抽取参数未知，重建范围以精确预计算特征为起点；不承诺从原始图文重提特征，也不假装有公开 checkpoint 下载链接。

## 服务器重建入口（本轮无需运行）

先取得仓库、固定上游映射和已校验的 processed 数据。已有原始 HDF5 时，可按 `docs/EXTERNAL_DATASETS.md` 的预处理流程重建；对应源文件必须匹配已公开 manifest。下面默认只读取配置/源码并打印计划：

```powershell
Set-Location 'G:\mmkg-project-research'
git pull --ff-only
python scripts/rebuild_paper_a_base_model.py --dataset mkg_w --architecture native --seed 1
```

如独立重建确有需要，在 CUDA 服务器显式加入 `--train`：

```powershell
python scripts/rebuild_paper_a_base_model.py --dataset mkg_w --architecture native --seed 1 --train
```

dataset 可取 `mkg_w`/`db15k`；architecture 可取 `mhyper`/`native`/`adamf_mat`；seed=1/2/3。输出限定到新 `outputs/base_model_rebuilds`，不会覆盖原模型。执行前检查66文件source inventory和canonical manifest；CUDA缺失直接失败，不回落到本机CPU训练。新运行记录 commit、配置、Python/Torch/CUDA/GPU。它重建已报告设置，不运行新TEST选择，也不保证历史checkpoint逐字节相同。默认 plan 路径已本机验证；未启动服务器训练。

取得任一声称对应的 checkpoint 后，用完整 inventory digest 验证，例如：

```powershell
$records = Get-Content 'outputs/paper_a_safe_correction/alignment_training_v1/checkpoint_inventory.json' -Raw | ConvertFrom-Json
foreach ($record in $records) {
    $checkpointPath = Join-Path $record.run 'best.ckpt'
    $actualDigest = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualDigest -ne $record.checkpoint_sha256) { throw "Checkpoint mismatch: $checkpointPath" }
}
```

审计、表格生成、论文依赖核验均绑定本轮文件。提交只含代码、协议、报告、约1.6MB的映射/配置/来源清单和论文源码；更新PDF在本地生成，沿用现有忽略规则。不提交checkpoint、HDF5或原始score矩阵。
