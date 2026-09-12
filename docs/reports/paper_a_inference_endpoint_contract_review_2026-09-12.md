# B08/B10 核验：预测前回退、anchor 并列与端点约定

日期：2026-09-12。范围：六个已锁定 pair；不重新训练，不在本机运行基础模型评分，不用 TEST 选参或替换历史结果。

后续服务器结果已完成核验，见 [B10 回传报告](paper_a_endpoint_contract_return_review_2026-09-12.md)。以下保留回传前的检查记录，不代表当前仍等待服务器结果。

## 回传前结论

- **B08：新防护推理入口已修复，原锁定结果已复现。** 非有限特征先分流，缩放后溢出先于 classifier 分流，非有限 logit 先于概率预测分流。异常行在阈值为零时仍回到 anchor。
- **B10：anchor tie 和共享 evaluator 端点单测已完成；实际 checkpoint 的 normalized/raw 对照待服务器结果。** 不能无条件证明端点标准化保持排名：已复现有限 float32 反例。当前不会标记 B10 完全关闭。

## B08：旧路径确有失败反例

原 `scripts/lock_apply_anchored_dynamic.py` / `scripts/ablate_anchored_dynamic.py` 路径把非有限坐标显式改为 NaN，再调用完整 imputer/scaler/logistic pipeline，最后才回退。显式转换能解释中位数插补如何接收 ±∞，但不能保证后续数值运算不溢出。实际 sklearn 单测使用 `[NaN, 1e308]`：第一维成功插补，另一维缩放成 infinity，`predict_proba` 在 fallback 前抛出错误。

新增 `router/anchored_inference.py` 与 CLI `scripts/apply_anchored_safe.py`。只将有限行送到下一阶段；原始特征、变换后特征、logit、probability 各有检查。异常原因写入输出，`predicted=false`；占位 logit=0 / probability=0.5 不是预测结果。格式不匹配、拟合状态损坏仍明确报错。只对已暴露在特征矩阵中的异常提供防护，不能发现已被上游 geometry sanitizer 隐藏的原始分数异常。

这是新的受支持推理入口，复用原 DEV model/lock。历史实验脚本保留原字节及调用顺序，供已有 source manifest 复现使用；不能继续将旧脚本描述为已具备同一异常行防护。论文 Algorithm 1 已按新入口改为先初始化 anchor，再逐阶段检查，最后仅对有效行计算 action。

## 本地真实结果复核

`scripts/audit_paper_a_inference_contract.py` 只读取原小型拟合模型和已导出的行，运行新的 `apply_frame` 入口。

| 检查 | 结果 |
| --- | --- |
| 六 pair × DEV/TEST | 12 个完整分区，474,732 行 |
| 新增异常回退 | 0 行 |
| ADC / Query-soft 网格权重、fallback flag | 与原导出逐行完全一致 |
| decision、probability、连续 ADC 权重、缓存 RR | 绝对误差均不超过 1e-12；最大 decision 误差约 2.00e-15 |
| 全 DEV anchor | 6/6 复现 |
| OOF 训练部分 anchor | 30/30 复现 |
| selector 拟合 / 基础模型评分 | 0 / 0 |

sklearn 运行时为 1.7.1，原 pickle 为 1.7.2；版本警告保存在审计 JSON。数值复现是上述容差和精确动作检查，不是跨版本原始字节相同的主张。

证据位于 `outputs/paper_a_safe_correction/inference_contract_review_v1/{audit.json,selector_replay.csv,anchor_replay.csv}`，由 `paper_a_draft/inference_contract_source_manifest.json` 绑定。两个约 603 MB 的既有脏文件及无关计时结果不纳入本次提交。

## B10：完整 tie 规则及等价性边界

Global 的实际两个实现均最大化 `(MRR, -abs(alpha-0.5), -alpha)`：MRR 精确并列时先靠近 0.5，再选较小 alpha，没有容差。即使另一个权重只高一个浮点单位，也首先按较高 MRR 选。该规则在全 DEV 与 OOF 训练部分一致。β/τ 并列依次偏好较小 β、较大 τ；投影依次最小化到 proposal 的距离、到 anchor 的距离、alpha 本身。

共享 evaluator 对 gold 保留、其他已知答案过滤后，使用 `1 + count(candidate > gold)`。相等分数不增加 rank。部署的 alpha=0/1 专门路径返回对应 raw standalone rank，忽略未启用模型。单测逐行比较实际 head/tail evaluator、导出端点、dense/sparse mask，覆盖 tie、常数、候选非有限值及未启用模型全 NaN。

须分清以下两件事：

1. **部署端点与共享 raw standalone evaluator 一致**：已有逐行单测支持，实际数据全量比对将在服务器执行。
2. **未特判的 normalized endpoint 与 raw endpoint 一致**：仅有条件化的实数推导，不存在对所有 float32 输入的保证。`[0, 1, 1e8]` 的第一个分数作为 gold，raw rank=3，标准化后前两项并列、rank=2；包含活动模型 infinity 也有反例。缓存端点早已走过特判，不能拿其相等证明第二件事。

论文已将端点列为明确的分段执行约定，并限定候选分数扰动性质适用于定义的 normalized score family。B01 的全量分数有限性审计不能排除有限精度引入的 tie，B09 的旧 rank 复现也不能消除这一问题。

## 验证及下一步

`tests/test_guarded_inference_endpoints.py` 与 `tests/test_endpoint_audit_runner.py` 共 **19 项通过**。服务器 runner 用微型 CPU 假 scorer 验证了 72 cell 的计数、全部 normalized/raw 差异保留、无重评分续跑及被修改的回传行拒绝；不加载实际 checkpoint。真实范围的 `--plan-only` 确认 18 checkpoint、72 cell、474,732 行。

论文编译无 overfull box 或未解析引用；全稿核验通过 2,790 项来源快照、44 张表和 4 张图。当前 PDF 为 51 页，已目视检查第 5–8、23–24 页的修改布局。

完整协议见 `docs/protocols/paper_a_inference_endpoint_contract_review.md`。服务器 PowerShell：

```powershell
git pull --ff-only
python scripts/audit_paper_a_endpoint_contract.py --device cuda
```

回传 `outputs/paper_a_safe_correction/endpoint_contract_audit_v1_return.zip`。审计固定原 checkpoint、数据、批次和过滤约定；报告 raw export/shared/deployed/normalized 的全部差异，异常及 RR 和，不调参、不替换历史结果。若有差异，后续量化影响并判断是否须重评；本轮不推断实际数据差异为零。
