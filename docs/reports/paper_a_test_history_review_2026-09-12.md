# C07/C08：历史 TEST 使用与确认性地位核查（2026-09-12）

本轮最低关闭条件通过“补证据并降调”满足。MKG-W 改为主要后验评估；DB15K 改为次要外部评估（后验），不再声称两者是独立确认性 holdout。这里“外部”只指第二套数据，不代表独立于整个研究开发过程。没有运行训练、重新选参或重评 TEST。

核查源版本：`da34b6632b810dea884d37db0339c4055c9b2be6`。历史文档、锁和结果保持原样；本报告及当前 PDF 取代它们的 confirmatory / secondary replication 证据等级解释，不回写或伪造旧冻结记录。

## 已查清与仍不可查清的事

- 初版 ADC 的 β 上限为 0.20，P3 扩展为 0.50。13 特征、OOF、P3 和锁定代码可按 Git 版本追溯。
- 9 月 3 日是本次定位到的四主 pair ADC TEST 数值及其锁摘要首次入库记录。DEV 锁和 TEST 摘要在同一提交的 source manifest 中出现，不能据此构造一个更早的独立冻结时间。
- 同日历史报告明确承认 DB15K TEST 已用于更早的 score-ensemble 实验。9 月 4 日 AACPI 协议进一步明确两套数据的 TEST 结果都已查看；六 pair 暴露清单也包括当时尚未有 pair-specific TEST 输出的 NativE+AdaMF。该协议的禁止条款针对后续 AACPI，不能倒推它证明了早期 ADC 首次查看的精确时间。
- 当前修复特征边界后的结果、健康 DynaSemble、半径分析和 matched alternatives 均晚于已记录的历史曝光。9 月 9 日 DEV-only primary 审计无法追溯性地预注册 9 月 3 日已报告的四 pair。
- 4 个旧策略的锁与 TEST receipt 对照当时已提交的 SHA-256 完全一致，10 个旧/新策略的完整 policy 和 model hash 均匹配。锁定执行身份有证据；方法、feature family、β 上限与 pair frame 在首次查看 TEST 之前已完整冻结，没有足够证据。
- 两套数据首次被人查看的准确时间，以及旧 TEST 是否实际影响方法/特征/pair 选择，均不能从现有日志确定。历史报告写过“没有影响”，但缺少完整、不可变的设计决策/访问记录来独立支持此断言。因此不把“不能排除”写成“已证明发生 TEST 调参”，也不再保证“完全没有影响”。

这里分开三种边界：训练和显式参数选择用哪些标签；filtered evaluation 用哪些已知事实；研究人员是否已见 TEST 结果。默认历史 DEV evaluator 的 TRAIN∪DEV∪TEST filter 是已知事实访问，不能写成“DEV 完全不打开 TEST”；它与 TEST MRR 是否影响设计是不同问题。当前推理特征不接收 filter/gold 的证据由 B07/C04 审计负责。

77 项历史检查通过只说明其检查的执行条件，并不能恢复 holdout 独立性；后续 B07/C04 修复也表明不能把那个 PASS 当作总体无泄漏证明。现有 bootstrap 为固定模型/设计的条件性、逐项描述区间，不覆盖历史 TEST 复用和自适应设计带来的不确定性。

## 可追溯时间线

以下日期均是 Git 作者/提交元数据，不是经认证的运行时间。文件 mtime、目录名日期和脚本“TEST_NOT_USED”自述不被提升为首次查看记录。完整原始证据快照随 `test_history_review_v1/evidence/` 提交。

### E01 · Discovery framework

作者时间 `2026-09-01T16:09:37+08:00`；提交时间 `2026-09-01T16:09:37+08:00`。

OpenBG heterogeneous protocol records M-Hyper with NativE/AdaMF, Global and Relation grids, RRF and Oracle. It is not a complete MKG-W/DB15K ADC preregistration.

- [docs/openbg_heterogeneous_complementarity_protocol.md](https://github.com/Cccjha0/mmkg-paper/blob/e32d8577c5ffecf826423217a5ae77d1645a30ce/docs/openbg_heterogeneous_complementarity_protocol.md)；SHA-256 `53479d74df362a1089601ef0d33535d47fb0ce4ec2c70a591208c8a47dd158bd`。

### E02 · Grouped OOF

作者时间 `2026-09-01T16:58:14+08:00`；提交时间 `2026-09-01T16:58:14+08:00`。

Original-triple grouped DEV cross-fitting added; the code fixes five folds and fold seed 20260901. This documents a selection procedure, not dataset-level TEST non-exposure.

- [scripts/crossfit_heterogeneous_dev_policies.py](https://github.com/Cccjha0/mmkg-paper/blob/ccd377e849595342421898c57fb0a1481c20b58e/scripts/crossfit_heterogeneous_dev_policies.py)；SHA-256 `46936c9357e3f7f0eda506a5fa3102020f516b9930e2f30415ba88ec9232daaa`。

### E03 · Initial ADC

作者时间 `2026-09-02T16:50:58+08:00`；提交时间 `2026-09-02T16:50:58+08:00`。

ADC cross-fit implementation and 13 score-geometry fields added. Initial beta grid is 0.05, 0.10, 0.15, 0.20.

- [scripts/crossfit_anchored_dynamic.py](https://github.com/Cccjha0/mmkg-paper/blob/f45b7787efeda282c0839d4799a8c537d78d9aa8/scripts/crossfit_anchored_dynamic.py)；SHA-256 `05a652bc927006da4ca1d30848f3ce0213d8c58fb818acbd63c44759f4907f22`。
- [router/query_geometry.py](https://github.com/Cccjha0/mmkg-paper/blob/f45b7787efeda282c0839d4799a8c537d78d9aa8/router/query_geometry.py)；SHA-256 `58c9e028fdaf2676c19a0989030cd7cefd701e0cbca07bfc34710ae26371431e`。

### E04 · P3 extension

作者时间 `2026-09-02T20:44:17+08:00`；提交时间 `2026-09-02T20:44:17+08:00`。

P3 adds feature/anchor/fallback ablations and extends beta to 0.50. The later report describes DEV-led development; no independently timestamped complete design freeze establishes first-inspection ordering.

- [scripts/ablate_anchored_dynamic.py](https://github.com/Cccjha0/mmkg-paper/blob/5b43015b91541c9706d365493f57f8f7bfaaae65/scripts/ablate_anchored_dynamic.py)；SHA-256 `bd46307d4bfff55c92fc82aef05080ab55672fdd5a8cf6c824a3212155446ea5`。

### E05 · Full-DEV lock code

作者时间 `2026-09-03T17:40:38+08:00`；提交时间 `2026-09-03T17:40:38+08:00`。

Serialized full-DEV policy and immutable TEST apply path added. Source-code availability does not timestamp any run or prove no earlier TEST viewing.

- [scripts/lock_apply_anchored_dynamic.py](https://github.com/Cccjha0/mmkg-paper/blob/6c78bdc9d462df9f1ad7af1fcd8c78b3df29f314/scripts/lock_apply_anchored_dynamic.py)；SHA-256 `2abb4e195d3d34d244d49c135aed475ae30814162f5ab3cb8e18bd486ab8ca5c`。

### E06 · First located four-pair result commit

作者时间 `2026-09-03T19:40:53+08:00`；提交时间 `2026-09-03T19:40:53+08:00`。

Four ADC TEST results and hashes of P3, DEV locks and TEST receipts appear together. The report calls MKG-W confirmatory and DB15K secondary replication and acknowledges earlier DB15K score-ensemble TEST use. These are historical author claims, not independent certification; actual first human inspection times remain unknown.

- [outputs/anchored_dynamic/four_pair_summary/four_pair_summary.json](https://github.com/Cccjha0/mmkg-paper/blob/f0932c2a0f3b9391c770ed0da62fd6260200817e/outputs/anchored_dynamic/four_pair_summary/four_pair_summary.json)；SHA-256 `46dfd21a970288c8f434c0da01501c6e6248968360b72a5ebd583aa4e854393e`。
- [docs/reports/recent_experiments_report_2026-09-03.md](https://github.com/Cccjha0/mmkg-paper/blob/f0932c2a0f3b9391c770ed0da62fd6260200817e/docs/reports/recent_experiments_report_2026-09-03.md)；SHA-256 `c13c17edb2a0aed1a280774b0c89abfc1ee5ef6e1cfdf35eb6c99d70ed2fcf05`。
- [docs/reports/anchored_dynamic_four_pair_report.md](https://github.com/Cccjha0/mmkg-paper/blob/f0932c2a0f3b9391c770ed0da62fd6260200817e/docs/reports/anchored_dynamic_four_pair_report.md)；SHA-256 `9752a3bfd67383010e56335f1e4df2b7c01a1c1dea33c288af989458f6a9f6b2`。

### E07 · Explicit exposure record

作者时间 `2026-09-04T01:05:29+08:00`；提交时间 `2026-09-04T01:05:29+08:00`。

The AACPI freeze explicitly says both datasets' TEST results had been inspected. Its manifest marks all six pairs retrospective/secondary, including NativE+AdaMF at dataset level. Its prospective rule concerns AACPI, so it does not by itself prove when the earlier ADC design saw TEST.

- [docs/protocols/AACPI_V2_DEV_PROTOCOL_FREEZE.md](https://github.com/Cccjha0/mmkg-paper/blob/9a87bbe30f6117d6bcd6642ff3d4694abf7174cd/docs/protocols/AACPI_V2_DEV_PROTOCOL_FREEZE.md)；SHA-256 `28bacebcb5b0544cea64a216d76050ecfee1378e63800302c695eeb7155300c4`。
- [docs/protocols/aacpi_test_exposure_manifest.csv](https://github.com/Cccjha0/mmkg-paper/blob/9a87bbe30f6117d6bcd6642ff3d4694abf7174cd/docs/protocols/aacpi_test_exposure_manifest.csv)；SHA-256 `068d93732647356d9820cc00a39cfc04b65c69a6d5d9c8acbbabbfe8046c71f8`。

### E08 · Primary eligibility audit

作者时间 `2026-09-09T20:10:29+08:00`；提交时间 `2026-09-09T20:10:29+08:00`。

DEV-only reliable-primary protocol added after the four-pair TEST result commit. It refers to the four already-current M-Hyper pairs; it cannot retroactively preregister their selection.

- [docs/protocols/paper_a_reliable_primary_protocol.md](https://github.com/Cccjha0/mmkg-paper/blob/f909a7902115bc8ed6156b57dc82361b949a114c/docs/protocols/paper_a_reliable_primary_protocol.md)；SHA-256 `c98bea1cf57e305f963d34ddae6898a561eb9328605511709ec5f69ed9df859f`。

### E09 · Additional-pair protocol

作者时间 `2026-09-09T20:27:13+08:00`；提交时间 `2026-09-09T20:27:13+08:00`。

NativE+AdaMF workflow added after dataset-level exposure; a newly evaluated pair is not a new holdout.

- [docs/protocols/paper_a_boundary_pair_protocol.md](https://github.com/Cccjha0/mmkg-paper/blob/0dd8f18c41c3bebeab891a6d90f9d16289599d4f/docs/protocols/paper_a_boundary_pair_protocol.md)；SHA-256 `5b12082e02af1797ebd2964e50a95efe393d4b6bd27a1a9d5a619910e8239714`。

### E10 · Historical 77-check audit

作者时间 `2026-09-10T01:47:36+08:00`；提交时间 `2026-09-10T01:47:36+08:00`。

77 execution-integrity checks pass. They inspect selection sources, folds and policy/hash consistency; they do not reconstruct human exposure or certify a complete pre-TEST design freeze.

- [outputs/paper_a_safe_correction/protocol_audit/audit.json](https://github.com/Cccjha0/mmkg-paper/blob/f42342e0343052b75c23682fe047e7e2e88acd9c/outputs/paper_a_safe_correction/protocol_audit/audit.json)；SHA-256 `ceef7b5e8720a464c69f6e0a362dd5285565f2ec98886191d2a42a236dfc934b`。
- [outputs/paper_a_safe_correction/protocol_audit/final_protocol_integrity.md](https://github.com/Cccjha0/mmkg-paper/blob/f42342e0343052b75c23682fe047e7e2e88acd9c/outputs/paper_a_safe_correction/protocol_audit/final_protocol_integrity.md)；SHA-256 `ebe002aeeb52509006a98a17aba5f8d9ba8ed8287c02ad75523c4e05b14df447`。

### E11 · Information-boundary repair

作者时间 `2026-09-10T19:07:35+08:00`；提交时间 `2026-09-10T19:07:35+08:00`。

Gold-independent features/normalization replace the historical implementation after TEST inspection. Corrected evaluation retains the same already-seen splits.

- [router/query_geometry.py](https://github.com/Cccjha0/mmkg-paper/blob/84d60fa363ff6417188494eb26542c136a74155e/router/query_geometry.py)；SHA-256 `51dde608681670c4cc616af604103544bfe9c8c5de51c1eb4afa86eef354dd30`。
- [scripts/lock_apply_anchored_dynamic.py](https://github.com/Cccjha0/mmkg-paper/blob/84d60fa363ff6417188494eb26542c136a74155e/scripts/lock_apply_anchored_dynamic.py)；SHA-256 `d6328868a6d3a1d86983be22e749c2337a3e3e55ba644e4c8aeedff7b05b0d91`。

### E12 · Corrected receipts and control protocol

作者时间 `2026-09-10T22:55:35+08:00`；提交时间 `2026-09-10T22:55:35+08:00`。

Corrected six-pair rerun audit is tracked; new DynaSemble control protocol fixes transparent repairs. Both follow the historical result/exposure records.

- [outputs/paper_a_safe_correction/information_boundary_rerun_audit/audit.json](https://github.com/Cccjha0/mmkg-paper/blob/1c0dbcb2aa0e06702add55cc8bd8f7574eb92dfd/outputs/paper_a_safe_correction/information_boundary_rerun_audit/audit.json)；SHA-256 `79c8fc92500ea756bfbf1e97522ca339256b38569e3c211c9aa60df76418ebf6`。
- [docs/protocols/paper_a_dynasemble_controls.md](https://github.com/Cccjha0/mmkg-paper/blob/1c0dbcb2aa0e06702add55cc8bd8f7574eb92dfd/docs/protocols/paper_a_dynasemble_controls.md)；SHA-256 `75bf3d28b8f7e0d7741fca8ca8bd1c9525d014e0e2677a625321db9ccaca106f`。

### E13 · Healthy-control review

作者时间 `2026-09-11T16:10:53+08:00`；提交时间 `2026-09-11T16:10:53+08:00`。

Returned DynaSemble controls and cache replay are reviewed; the failed transfer remains reported. This is post-exposure supplementary evidence.

- [docs/reports/paper_a_dynasemble_controls_review_2026-09-11.md](https://github.com/Cccjha0/mmkg-paper/blob/93ddfc1b032e23dedeb6d48e32a15a12d921e49b/docs/reports/paper_a_dynasemble_controls_review_2026-09-11.md)；SHA-256 `97a2be41ded2b81c02e758f7001cb5c368b7ec0054abb1fe8effc3a2fd3bf827`。

### E14 · Radius and harm analysis

作者时间 `2026-09-11T17:08:52+08:00`；提交时间 `2026-09-11T17:08:52+08:00`。

Wider DEV radius and common risk comparisons are added after TEST inspection. They do not select a new TEST policy or establish a pre-TEST reason for cap 0.50.

- [docs/protocols/paper_a_conservative_radius_review.md](https://github.com/Cccjha0/mmkg-paper/blob/ca734a5d42a88d987fef0d5cabb3b336fa03160a/docs/protocols/paper_a_conservative_radius_review.md)；SHA-256 `a3d8f47b6c4ba97d1dc7b7926b7e8e47ba15f636303df447f2a82f9416b5e725`。

### E15 · Matched alternatives

作者时间 `2026-09-12T00:16:53+08:00`；提交时间 `2026-09-12T00:16:53+08:00`。

Matched shrinkage/linear/clip comparisons and explicit Global fallback are added after TEST inspection; their own DEV locks constrain execution, not the prior choice of families.

- [docs/protocols/paper_a_matched_alternatives.md](https://github.com/Cccjha0/mmkg-paper/blob/ce89634a5e083715f3644337d33cc5fd3d675da9/docs/protocols/paper_a_matched_alternatives.md)；SHA-256 `240eee15accf6df7e48aafea6174cd497ec449c1817b0463c78310df135424db`。
- [docs/reports/paper_a_matched_alternatives_review_2026-09-12.md](https://github.com/Cccjha0/mmkg-paper/blob/ce89634a5e083715f3644337d33cc5fd3d675da9/docs/reports/paper_a_matched_alternatives_review_2026-09-12.md)；SHA-256 `42498f588144f93ebe06e01ab5c75c539758b124fa75168160cb4470bf5a1a00`。

## 冻结对象与证据范围

- **Feature family / learner** (E03, E04, E11): 13 geometry fields and logistic implementation are versioned; P3 evaluates feature subsets; corrected geometry is versioned separately. Limit: Complete family choice before first TEST inspection is not established; corrected computation postdates exposure.
- **Action grid and radius cap** (E01, E03, E04, E06, E14): Alpha grid is 0:0.05:1; beta cap changes from 0.20 to 0.50; legacy locks carry the full grids. Limit: No independent pre-inspection cap freeze or calibrated safety rationale; wider-radius analysis is retrospective.
- **OOF grouping and selection** (E02, E03, E04, E10): Five folds group seeds and directions by original triple; code and selected P3 source hashes are recorded. Limit: Grouped DEV validation does not certify TEST-blind choice of the validation scheme or design.
- **Main pairs and primary rule** (E01, E06, E08, E09): Four main M-Hyper pairs are explicit with September 3 results; the six-pair DEV primary rule is committed September 9. Limit: Earlier OpenBG expert choices do not preregister this dataset/pair frame; a later DEV-only audit cannot prove it was chosen blind.
- **Legacy final parameters** (E05, E06): Four complete DEV policies, serialized model hashes and TEST-embedded policies match their committed result manifest. Limit: Lock receipts and outcomes were recorded together; no trusted first-view timestamp establishes complete pre-exposure freezing.
- **Corrected final parameters** (E07, E11, E12): Six current lock/model/TEST receipt bindings match the rerun source manifest; all four main and two additional pairs are retained. Limit: This repairs the feature boundary and supports reproducible execution on previously viewed TEST, not fresh confirmation.
- **Supplementary families / budgets** (E12, E13, E14, E15): DynaSemble repairs, radius and matched-action studies have their own recorded protocols and DEV choices. Limit: Their design follows TEST inspection; within-study DEV selection cannot remove program-level adaptive design uncertainty.

## 主张处理

- **withdraw** — MKG-W confirms the pre-specified complete ADC design → Primary retrospective evaluation on MKG-W
- **withdraw** — DB15K is an independent replication → Secondary external evaluation on DB15K, conducted retrospectively
- **not established** — No historical TEST outcome influenced method, features or pair choice → Recorded selection routines use DEV labels; the available record cannot exclude indirect design influence from already viewed TEST outcomes.
- **withdraw inference** — 77 passing checks restore holdout independence → The archived 77 checks establish only their recorded execution-consistency conditions.
- **retain with scope** — Current TEST application uses serialized DEV parameters → All six current complete policies and model hashes match the recorded TEST applications; this is execution identity, not pre-inspection design independence.
- **withdraw inference** — Bootstrap intervals provide confirmation after adaptive design → Pointwise conditional descriptive intervals do not account for historical TEST reuse or adaptive method/pair/family choice.

## 复核与下一次真正确认性实验

轻量核查命令（PowerShell；不训练模型）：

```powershell
python scripts/audit_paper_a_test_history.py
python scripts/build_paper_a_test_history_assets.py
python -m pytest tests/test_test_history_audit.py -q
python paper_a_draft/scripts/compile.py
python paper_a_draft/scripts/verify_draft.py
```

`audit.json` 记录 15 个历史节点、4 个旧策略和 6 个当前策略的核验、未知首次查看时间及明确降调；`policy_bindings.json` 保留每个完整锁、模型、TEST receipt 的 SHA-256。既有大规模源资产的完整检查仍由 draft verifier 负责。

本轮不需要服务器计算。再次运行同一 TEST 不会恢复确认性。若将来需要独立确认，应先将完整方法/feature contract、β/τ/α 搜索空间、候选 pair 和 primary 规则、特征来源/模型 checkpoint 规则、OOF、预算、终点和报告规则一并冻结并留存可追溯提交，再访问一套对该设计仍未暴露的评估集；当前记录不为任何候选数据集预先认证“未暴露”。
