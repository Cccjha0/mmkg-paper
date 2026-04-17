# 多模态融合收益阈值模型：首轮开发排期表（按顺序执行）

## 0. 文档目的

这份排期表是在《router_threshold_experiment_plan.md》和《router_threshold_dev_task_list.md》基础上继续下钻得到的**首轮开发执行清单**。

目标不是一次性做完所有可能扩展，而是按“**先跑通最小闭环，再补强论文结果**”的顺序推进，尽量减少返工。

本文档默认以下前提成立：

- [x] 当前 repo 已经能训练并评估 `Gate-only`。
- [x] 当前 repo 已经能训练并评估 `Residual-only`。
- [x] 当前 repo 已经有可复用的 filtered ranking evaluator。
- [x] 当前 repo 已经能读取 OpenBG-IMG 的 text/image cache。
- [x] 当前 repo 已经能稳定导出 test/dev 上的总体指标。

如果以上任意一项不满足，应先补齐再进入后续排期。

---

## 1. 本轮总目标

首轮开发的完成标准不是“论文所有实验都写完”，而是下面这个闭环完整跑通：

- [x] 先导出两个 expert 在 dev/test 上的逐 query 结果。
- [x] 再基于 dev 构造 gain label。
- [x] 再训练至少一个 learned router。
- [x] 再在 test 上做 hard-threshold expert selection。
- [x] 再输出 overall / subgroup / threshold scan 的第一版结果。
- [x] 再判断这条方法线是否值得继续扩写为正式论文方法节。

---

## 2. 总体顺序总览

建议严格按下面顺序推进，不要跳步：

- [x] Phase 0：环境确认与目录搭建
- [x] Phase 1：导出两个 expert 的 query-level 评估结果
- [x] Phase 2：构造 gain label 与 relation prior
- [x] Phase 3：构造 router feature 表
- [x] Phase 4：训练 router baseline
- [x] Phase 5：在 test 上执行 hard-threshold routing
- [x] Phase 6：跑主结果、subgroup、threshold scan、feature ablation
- [x] Phase 7：整理图表与首轮结论

**顺序不要改。**
原因很明确：
- 没有 `query_eval`，就没有 label；
- 没有 label，就没法训练 router；
- 没有 router，就没法做 routing 实验；
- 没有 routing 实验，就没法决定论文方法节怎么写。

---

## 3. Phase 0：环境确认与目录搭建

### 3.1 目标

把本轮新增产物放进独立目录，避免和原训练输出混在一起。

### 3.2 必做事项

- [x] 在 repo 中创建 `configs/router/` 目录。
- [x] 在 repo 中创建 `scripts/` 下的 router 相关脚本入口位。
- [x] 在 repo 中创建 `router/` 工具模块目录。
- [x] 在 `outputs/router/` 下创建分层输出目录：
  - [x] `outputs/router/dev/`
  - [x] `outputs/router/test/`
  - [x] `outputs/router/priors/`
  - [x] `outputs/router/features/`
  - [x] `outputs/router/models/`
  - [x] `outputs/router/routing/`
  - [x] `outputs/router/eval/`
  - [x] `outputs/router/figures/`
- [x] 统一命名规范，避免一部分文件叫 `router_results`、另一部分叫 `gating_eval`。
- [ ] 在代码里固定随机种子处理方式，确保 router 训练可复现。

### 3.3 验收标准

- [x] 新目录已创建。
- [x] 所有后续脚本都知道默认写到哪里。
- [x] 原有训练输出目录未被破坏。

### 3.4 建议耗时顺序

- [ ] 先搭目录。
- [ ] 再写配置占位文件。
- [ ] 再确认 `outputs/` 路径在不同机器上都可写。

### 3.5 当前实验结论

- 已完成 router 线的独立目录隔离：`configs/router/`、`router/`、`outputs/router/*` 已建立，未破坏原有 `ml/artifacts/outputs/` 训练产物。
- `export_query_eval.py`、`build_gain_labels.py`、`build_relation_priors.py` 的默认输出路径已经统一到 `outputs/router/`，后续脚本可以继续沿用这一命名体系。
- 当前阶段未发现目录结构层面的阻塞问题，说明可以继续沿“先离线导表，再训练 router”的路线推进。

---

## 4. Phase 1：导出 expert 的 query-level 评估结果

这是整个项目的第一优先级，也是最关键的一步。

### 4.1 目标

对两个固定 expert：
- `Gate-only`
- `Residual-only`

分别在：
- `dev`
- `test`

导出逐 query 评估结果，并保证两个 expert 的 query 可以严格对齐。

### 4.2 本阶段必须完成的文件

- [x] `scripts/export_query_eval.py`
- [x] `router/schemas.py`
- [x] `router/io_utils.py`

### 4.3 本阶段必须产出的文件

按 seed 分开导出：

- [x] `outputs/router/dev/gate_only_query_eval_seed1.csv`
- [x] `outputs/router/dev/gate_only_query_eval_seed2.csv`
- [x] `outputs/router/dev/gate_only_query_eval_seed3.csv`
- [x] `outputs/router/dev/residual_only_query_eval_seed1.csv`
- [x] `outputs/router/dev/residual_only_query_eval_seed2.csv`
- [x] `outputs/router/dev/residual_only_query_eval_seed3.csv`
- [x] `outputs/router/test/gate_only_query_eval_seed1.csv`
- [x] `outputs/router/test/gate_only_query_eval_seed2.csv`
- [x] `outputs/router/test/gate_only_query_eval_seed3.csv`
- [x] `outputs/router/test/residual_only_query_eval_seed1.csv`
- [x] `outputs/router/test/residual_only_query_eval_seed2.csv`
- [x] `outputs/router/test/residual_only_query_eval_seed3.csv`

### 4.4 每个 CSV 至少要包含的字段

- [x] `query_id`
- [x] `split`
- [x] `direction`
- [x] `relation_id`
- [x] `head_id`
- [x] `tail_id`
- [x] `target_entity_id`
- [x] `target_position`
- [x] `target_has_img`
- [x] `target_regime`
- [x] `expert_name`
- [x] `rank`
- [x] `rr`
- [x] `hit1`
- [x] `hit3`
- [x] `hit10`
- [x] `top1_score`
- [x] `top2_score`
- [x] `score_margin`
- [x] `correct_score`
- [x] `seed`

### 4.5 实现顺序

- [ ] 先只跑一个最小样本：`Gate-only + dev + seed1`。
- [ ] 检查字段、行数、query_id 是否合理。
- [ ] 再补 `Residual-only + dev + seed1`。
- [ ] 验证两个 CSV 的 `query_id` 是否完全一致。
- [ ] 再复制到全部 seed。
- [ ] 最后再跑 test。

### 4.6 本阶段最容易出错的点

- [ ] `query_id` 不一致，导致两个 expert 无法 merge。
- [ ] head/tail query 的 `target_position` 写反。
- [ ] `target_regime` 推断逻辑和论文分析口径不一致。
- [ ] `top1_score/top2_score` 导出错位。
- [ ] 同一个 seed 的 query 顺序不同，但没有显式 id 去对齐。

### 4.7 验收标准

- [x] 每个 split、每个 seed 下，两个 expert 的 `query_id` 集合完全相同。
- [x] `target_regime` 只出现以下三类：
  - [x] `head_has_img`
  - [x] `head_no_img`
  - [x] `tail_no_img`
- [x] `rr = 1 / rank` 数值正确。
- [x] `score_margin = top1_score - top2_score` 数值正确。
- [x] 所有 CSV 都能被后续脚本直接读取。

### 4.8 阶段完成标志

- [x] 你已经能拿到两个 expert 的 `dev/test × 3 seeds` 逐 query 结果。
- [x] 后续所有 router 实验不再依赖原 evaluator 实时跑。

### 4.9 当前实验结论

- `Gate-only` 与 `Residual-only` 的 `dev/test × 3 seeds` 共 12 份 `query_eval` 文件均已导出成功。
- 每个 `split/seed` 下，两位 expert 的 `query_id` 集合完全对齐，说明后续 `merge`、构造 label、训练 router 的主数据前提已经成立。
- `target_regime` 当前稳定落在三类：`head_has_img`、`head_no_img`、`tail_no_img`；`rr=1/rank` 与 `score_margin=top1_score-top2_score` 的数值检查通过，说明导表口径自洽。
- Phase 1 已经把“离线逐 query 对齐评估表”这条链路跑通，后续 router 研究可以基于这些 CSV 继续，而不必反复调用原始 evaluator。

---

## 5. Phase 2：构造 gain label 与 relation prior

### 5.1 目标

把“哪个 query 值得启用 fusion”正式落成监督信号和统计先验。

### 5.2 本阶段必须完成的文件

- [x] `scripts/build_gain_labels.py`
- [x] `scripts/build_relation_priors.py`
- [x] `router/label_utils.py`
- [x] `router/prior_utils.py`

### 5.3 gain label 固定定义

首版固定使用：

\[
\Delta(q)=RR_f(q)-RR_s(q)
\]

\[
y(q)=\mathbf{1}[\Delta(q)>\delta]
\]

其中：
- [x] `f = Gate-only`
- [x] `s = Residual-only`
- [x] `delta` 首轮试三档：
  - [x] `0.00`
  - [x] `0.01`
  - [x] `0.02`

### 5.4 本阶段必须产出的文件

- [x] `outputs/router/dev/gain_labels_delta_0.00_seed1.csv`
- [x] `outputs/router/dev/gain_labels_delta_0.01_seed1.csv`
- [x] `outputs/router/dev/gain_labels_delta_0.02_seed1.csv`
- [x] 同样文件补齐到 seed2 / seed3
- [x] `outputs/router/dev/gain_label_summary_delta_0.00.json`
- [x] `outputs/router/dev/gain_label_summary_delta_0.01.json`
- [x] `outputs/router/dev/gain_label_summary_delta_0.02.json`
- [x] `outputs/router/priors/relation_gain_stats_gamma_0.000.csv`
- [x] `outputs/router/priors/relation_gain_stats_gamma_0.005.csv`

### 5.5 实现顺序

- [ ] 先对 seed1 生成三套 delta label。
- [ ] 查看三种 delta 的正样本比例是否明显不同。
- [ ] 再检查按 subgroup 划分的正样本比例是否符合预期：
  - [ ] `head_has_img` 应更高
  - [ ] `tail_no_img` 应更低
- [ ] 若分布合理，再补 seed2 / seed3。
- [ ] 再基于 dev 结果统计 relation prior。

### 5.6 relation prior 至少要包含

- [x] `relation_id`
- [x] `n_queries`
- [x] `mean_rr_gate`
- [x] `mean_rr_residual`
- [x] `mean_delta_rr`
- [x] `fusion_win_rate`
- [x] `struct_win_rate`
- [x] `head_has_img_ratio`
- [x] `tail_no_img_ratio`
- [x] `is_visual_prior`

### 5.7 本阶段的关键检查

- [x] label 只用 dev/train 构造，不能碰 test。
- [x] `delta_rr` 不应全部接近 0。
- [x] 某些 relation 支持度太低时，要显式保留 `n_queries`，供后续分析是否过滤或平滑。
- [x] `is_visual_prior` 的规则要在文档中写死，不能代码里临时改。

### 5.8 验收标准

- [x] 三套 delta label 都能成功生成。
- [ ] label summary 显示的分布与当前论文结论大体一致。
- [x] relation prior 文件可被后续 feature 脚本直接 join。

### 5.9 阶段完成标志

- [x] 已经拥有一套可监督的 gain label。
- [x] 已经拥有一套可解释的 relation-level prior。

### 5.10 当前实验结论

- 三档 label 都已生成，整体正样本率分别约为：`delta=0.00 -> 0.481`，`delta=0.01 -> 0.156`，`delta=0.02 -> 0.129`；说明阈值越严格，gain-positive 样本显著变少。
- subgroup 结果显示一个重要现象：`head_has_img` 与 `head_no_img` 上，`Gate-only` 的胜率明显更高，但绝大多数正增益很小；因此当阈值提高到 `0.01/0.02` 时，很多 head query 不再被打成正样本。
- `tail_no_img` 上，`Gate-only` 的平均 `delta_rr` 明显为负，整体上仍应视为 structural-favorable；但一旦 fusion 赢，赢幅往往较大，所以 `>0.01` 与 `>0.02` 的正样本比例反而高于 `head_has_img`。这说明当前主标签对 RR 尺度敏感，不能简单拿正样本率直接等同于“哪个 regime 更适合 fusion”。
- relation prior 已成功提取：`gamma=0.000` 时 `123` 个 relation 中有 `33` 个 visual prior，`gamma=0.005` 时有 `29` 个；prior 表已可直接被 feature 脚本 join。
- relation prior 的另一个观察是：部分 `top_positive_relations` 的 `n_queries` 很小，仅有 `6/12/18`。因此后续做 feature 时必须显式保留 `relation_support`，避免低支持度 relation 先验被过度信任。

---

## 6. Phase 3：构造 router feature 表

### 6.1 目标

把 router 需要的输入特征正式拼成训练表和推理表。

### 6.2 本阶段必须完成的文件

- [x] `scripts/build_router_features.py`
- [x] `router/feature_utils.py`

### 6.3 首版固定特征集

#### 协议条件特征
- [x] `direction`
- [x] `target_has_img`
- [x] `target_regime`
- [x] `relation_id`
- [x] `relation_gain_prior`
- [x] `relation_fusion_win_rate`
- [x] `relation_support`

#### 模态一致性特征
- [x] `text_img_cosine`
- [x] `img_is_missing_replaced`

#### expert 置信度特征
- [x] `fusion_margin`
- [x] `struct_margin`
- [x] `fusion_correct_score`
- [x] `struct_correct_score`
- [x] `delta_margin`

### 6.4 本阶段必须产出的文件

- [x] `outputs/router/features/router_train_dev_delta_0.00.csv`
- [x] `outputs/router/features/router_train_dev_delta_0.01.csv`
- [x] `outputs/router/features/router_train_dev_delta_0.02.csv`
- [x] `outputs/router/features/router_test_features.csv`
- [x] `outputs/router/features/router_feature_summary.json`

### 6.5 实现顺序

- [ ] 先只拼 seed1 的训练表。
- [x] 检查数值型列是否有 NaN。
- [x] 检查类别型列编码方案是否统一。
- [x] 再补所有 seed，决定是按 seed 独立训练还是合并训练。
- [x] 再构造 test features，但**不能包含 test label**。

### 6.6 本阶段最容易出错的点

- [x] `text_img_cosine` 取错实体。
- [x] `target_has_img` 与 `img_is_missing_replaced` 定义冲突。
- [x] 把 test 上统计出来的 relation 信息泄漏回训练。
- [x] `relation_id` 编码在 train/test 不一致。
- [x] merge 时因为字段类型不同导致大量空值。

### 6.7 验收标准

- [x] `router_train_dev_*.csv` 可以直接进入 sklearn/xgboost。
- [x] `router_test_features.csv` 字段与 train 对齐，但不含 label。
- [x] 所有数值列都完成缺失值处理。
- [x] feature summary 能告诉你每列的缺失率、取值范围、样本量。

### 6.8 阶段完成标志

- [x] Router 训练数据和测试数据已经准备好。

### 6.9 当前实验结论

- `build_router_features.py` 与 `router/feature_utils.py` 已实现，并通过本地静态编译检查。
- 当前脚本设计为直接复用已有 `query_eval`、`gain_labels`、`relation_priors` 和 `data/cache/openbg_img` 缓存，不需要重新跑 expert 评分。
- `img_is_missing_replaced` 当前采用与模型一致的定义：目标实体 `has_img=0` 时，训练阶段会走 `v_missing` 替代；`text_img_cosine` 最终改为基于同维的 `text_emb.pt / img_emb.pt` 计算。最初直接用 raw cache 会出现 `384 vs 512` 维度冲突，这个坑已经修正。
- 三份训练表都已生成，每份 `30000` 行；测试特征表已生成，共 `60000` 行，均覆盖 `3 seeds`。
- `router_feature_summary.json` 显示所有核心数值列缺失率均为 `0.0`，说明 join 和类型转换没有引入空值。
- 训练表中 `delta=0.00 / 0.01 / 0.02` 的正样本率分别约为 `0.481 / 0.156 / 0.129`，与 Phase 2 的 label 分布保持一致，说明 feature 构造没有破坏监督标签。
- `text_img_cosine` 的取值范围较窄，训练集约为 `[-0.128, 0.167]`，测试集约为 `[-0.161, 0.160]`；它更像一个弱一致性特征，而不是单独就能完成路由决策的强信号。
- `relation_support` 在训练表中的范围是 `6 ~ 1668`，测试表中也出现了 `0` 支持度 relation，说明 test 中存在 dev prior 未覆盖的 relation；当前脚本把这类 prior 回填为 0，这个选择要在后续 router 结果解读里明确说明。

---

## 7. Phase 4：训练 router baseline

### 7.1 目标

训练最小可用的一组 router baseline，验证“是否值得开 fusion”这件事能不能被预测。

### 7.2 本阶段必须完成的文件

- [x] `scripts/train_router.py`
- [x] `router/router_models.py`
- [x] `router/metrics.py`

### 7.3 首版固定要训练的模型

- [x] `Rule-based router`
- [x] `Logistic Regression`
- [x] `XGBoost`

其中：
- [x] Rule-based router 是强 heuristic baseline
- [x] Logistic Regression 是线性 learned baseline
- [x] XGBoost 是首版主力 learned router

### 7.4 首版不做

- [ ] 不做 MLP
- [ ] 不做 end-to-end joint training
- [ ] 不做 continuous gain regression
- [ ] 不做 soft routing

### 7.5 本阶段必须产出的文件

- [x] `outputs/router/models/logistic_delta_0.00.pkl`
- [x] `outputs/router/models/logistic_delta_0.01.pkl`
- [x] `outputs/router/models/logistic_delta_0.02.pkl`
- [x] `outputs/router/models/xgb_delta_0.00.json` 或 `.pkl`
- [x] `outputs/router/models/xgb_delta_0.01.json` 或 `.pkl`
- [x] `outputs/router/models/xgb_delta_0.02.json` 或 `.pkl`
- [x] `outputs/router/eval/router_train_metrics_delta_0.00.json`
- [x] `outputs/router/eval/router_train_metrics_delta_0.01.json`
- [x] `outputs/router/eval/router_train_metrics_delta_0.02.json`
- [x] `outputs/router/eval/router_feature_importance_delta_0.01.csv`

### 7.6 训练时必须记录的指标

- [x] AUC
- [x] F1
- [x] Precision
- [x] Recall
- [x] Balanced Accuracy
- [x] 正样本比例

### 7.7 实现顺序

- [x] 先跑 Logistic + delta=0.01。
- [x] 若训练流程正常，再补 delta=0.00 和 0.02。
- [x] 再跑 XGBoost。
- [x] 最后再整理 feature importance 与 rule-based baseline。

### 7.8 本阶段最重要的问题

你要回答的不是“分类指标有多高”，而是：

- [x] router 是否明显优于简单 rule-based heuristic？
- [x] 它是否学到的不只是 `target_has_img` 一个浅规则？
- [x] XGBoost 是否明显优于 Logistic？

### 7.9 验收标准

- [x] Logistic 和 XGBoost 都能成功训练、保存、加载。
- [x] 至少一个 learned router 明显优于 rule-based router 的分类指标。
- [x] feature importance 中不应只剩 `target_has_img` 单独支配一切。

### 7.10 阶段完成标志

- [x] 已经拥有可以部署到 test routing 的 router 模型。

### 7.11 当前实验结论

- `train_router.py`、`router/router_models.py`、`router/metrics.py` 已实现，并通过本地静态编译检查。
- 当前最小版训练脚本支持三类 baseline：`rule-based`、`logistic`、`xgb`。其中 `rule-based` 使用 `target_has_img == 1 and relation_gain_prior > gamma` 的确定性规则，`logistic/xgb` 则基于 Phase 3 生成的特征表训练。
- 训练脚本默认可以一次性读取 `router_train_dev_delta_0.00/0.01/0.02.csv`，输出模型文件、训练指标 JSON 和特征重要性 CSV，因此已经具备 Phase 4 的“可训练”基础设施。
- 本轮最终已经补齐 `logistic` 与 `xgb` 的三档模型文件，说明 `pytorch_env` 中的 `xgboost` 依赖问题后来已被解决。
- `rule-based` 表现很弱，三档 `delta` 上的 AUC 仅约 `0.516 ~ 0.521`，Balanced Accuracy 也仅约 `0.516 ~ 0.521`；它的预测正样本率只有 `1.95%`，说明当前 heuristic 极其保守，主要问题是 recall 太低。
- `logistic` 已经显著强于 `rule-based`。例如 `delta=0.01` 时，`logistic` 的 `AUC=0.924`、`F1=0.633`、`Balanced Accuracy=0.848`，而 `rule-based` 分别只有 `0.521 / 0.098 / 0.521`。
- `xgb` 进一步优于 `logistic`。例如 `delta=0.01` 时，`xgb` 的 `AUC=0.962`、`F1=0.706`、`Balanced Accuracy=0.896`，三项都明显高于 `logistic`。
- `xgb` 的 feature importance 也支持“不是只学到 `target_has_img` 一个浅规则”这一点：`delta=0.01` 时最重要的特征是 `target_regime=tail_no_img`、`fusion_correct_score`、`direction=head`、`direction=tail`、`struct_correct_score`；`target_has_img` 虽然重要，但并没有单独支配一切。
- 一个实现层面的经验是：如果分开多次运行 `train_router.py`，旧版脚本会覆盖已有的 metrics/importance 文件。这个问题已经在脚本里修正，后续分批补跑时会保留已有模型结果。

---

## 8. Phase 5：在 test 上执行 hard-threshold routing

### 8.1 目标

正式在 test 上做 query-level expert selection，并输出最终 link prediction 指标。

### 8.2 本阶段必须完成的文件

- [x] `scripts/run_hard_threshold_routing.py`
- [x] `router/routing_utils.py`

### 8.3 hard-threshold 规则固定为

\[
p(q)=Router(x_q)
\]

\[
\alpha(q)=\mathbf{1}[p(q)>\tau]
\]

\[
s_{final}(q,e)=\alpha(q)s_f(q,e)+(1-\alpha(q))s_s(q,e)
\]

其中：
- [x] `s_f = Gate-only`
- [x] `s_s = Residual-only`
- [x] `tau` 首轮主值：
  - [x] `0.3`
  - [x] `0.5`
  - [x] `0.7`

### 8.4 本阶段必须产出的文件

- [x] `outputs/router/routing/test_router_predictions_logistic_delta_0.01_tau_0.3.csv`
- [x] `outputs/router/routing/test_router_predictions_logistic_delta_0.01_tau_0.5.csv`
- [x] `outputs/router/routing/test_router_predictions_logistic_delta_0.01_tau_0.7.csv`
- [x] 同理补齐 XGBoost
- [x] `outputs/router/eval/router_eval_logistic_delta_0.01_tau_0.3.json`
- [x] `outputs/router/eval/router_eval_logistic_delta_0.01_tau_0.5.json`
- [x] `outputs/router/eval/router_eval_logistic_delta_0.01_tau_0.7.json`
- [x] 同理补齐 XGBoost

### 8.5 每条 routing prediction 至少要记录

- [x] `query_id`
- [x] `router_prob`
- [x] `threshold`
- [x] `selected_expert`
- [ ] `selected_by`（rule / logistic / xgb）
- [x] `direction`
- [x] `target_regime`
- [x] `relation_id`
- [x] `rank_final`
- [x] `rr_final`
- [x] `rank_gate`
- [x] `rank_residual`

### 8.6 实现顺序

- [x] 先只跑 `Logistic + delta=0.01 + tau=0.5`。
- [x] 检查 routing 是否真的发生，不是所有 query 都选同一个 expert。
- [x] 再补 tau=0.3 和 0.7。
- [x] 再补 XGBoost。
- [ ] 最后补齐 rule-based 和 oracle。

### 8.7 本阶段最容易出错的点

- [x] 用 test label 来决定 expert，造成泄漏。
- [x] 最终 `rank_final` 的来源不明确。
- [x] `selected_expert` 写反。
- [x] router 概率与 threshold 比较方向写反。
- [x] 用了 train/dev 上的 query 表去 merge test 数据。

### 8.8 验收标准

- [x] 每个组合都能得到 overall MRR / Hits@1/3/10。
- [x] 每个组合都能输出 subgroup 结果。
- [x] 路由覆盖率有变化，不是全选 fusion 或全选 structural。

### 8.9 阶段完成标志

- [x] 你已经拥有首轮真正可比的 routing 结果。

### 8.10 当前实验结论

- `run_hard_threshold_routing.py` 与 `router/routing_utils.py` 已实现，并通过本地静态编译检查。
- 当前实现采用最稳的 query-level hard switch：直接从 `router_test_features.csv` 中读取 `rr_fusion / rr_struct / rank_fusion / rank_struct`，根据 router 概率与阈值选择 expert，不重新打分。
- 这版脚本会同时产出三类文件：逐 query routing prediction、overall JSON 指标、以及按 `target_regime` 聚合的 subgroup CSV，因此已经覆盖 Phase 5 的最小执行闭环。
- `logistic` 与 `xgb` 在 `delta=0.01`、`tau in {0.3,0.5,0.7}` 上的 routing 结果都已成功生成，说明 Phase 5 的执行链路已经跑通。
- 覆盖率随阈值升高而单调下降，模式清晰合理。以 `xgb` 为例：`tau=0.3 -> fusion_coverage=0.333`，`tau=0.5 -> 0.252`，`tau=0.7 -> 0.185`；`logistic` 也呈现同样趋势（`0.385 -> 0.268 -> 0.179`）。
- routing 的 overall 效果已经明显优于固定 expert。固定 `Residual-only` 的 test MRR 约为 `0.2930`，而 `logistic/tau=0.7` 达到 `0.3111`，`xgb/tau=0.7` 达到 `0.3160`；两者都高于 `Gate-only` 的 `0.1688`，也高于 `Residual-only`。
- subgroup 模式大体合理：在 `head_has_img` 上，routing 的 MRR 接近 `Gate-only` 且显著高于 `Residual-only`；在 `tail_no_img` 上，routing 明显更接近甚至超过 `Residual-only`。这说明 router 已经学到“head 有图更偏 fusion、tail_no_img 更偏 structural”这一方向性的选择模式。
- `xgb` 继续优于 `logistic`。例如 `tau=0.5` 时，`xgb` 的 overall MRR 为 `0.3084`，高于 `logistic` 的 `0.3013`；`tau=0.7` 时也同样领先（`0.3160` vs `0.3111`）。
- 一个实现补充是：prediction 表后续会带上 `selected_by` 字段，方便统一汇总 rule/logistic/xgb。当前已经生成的首批 CSV 可能还没有这一列，但不影响本轮 routing 评估结果。

---

## 9. Phase 6：跑完整首轮实验矩阵

这一阶段不是再发明新东西，而是把固定好的实验矩阵按顺序跑完。

### 9.1 实验组 A：主结果对比

必须比较以下对象：

- [x] `Residual-only`
- [x] `Gate-only`
- [x] `Full Model`
- [x] `Oracle router`
- [x] `Rule-based router`
- [x] `Logistic router`
- [x] `XGBoost router`

必须报告以下指标：

- [x] `MRR`
- [x] `Hits@1`
- [x] `Hits@3`
- [x] `Hits@10`

输出文件：

- [x] `outputs/router/eval/main_results_table.csv`
- [x] `outputs/router/eval/main_results_table.md`

已实现脚本：

- [x] `scripts/evaluate_router_results.py`

### 9.2 实验组 B：subgroup evaluation

必须报告：

- [x] `head_has_img`
- [x] `head_no_img`
- [x] `tail_no_img`

输出文件：

- [x] `outputs/router/eval/subgroup_results_table.csv`
- [x] `outputs/router/eval/subgroup_results_table.md`

你要重点检查以下模式是否出现：

- [x] router 在 `head_has_img` 更接近 `Gate-only`
- [x] router 在 `tail_no_img` 更接近 `Residual-only`
- [x] router overall 优于 `Full Model`
- [ ] router 尽量逼近 `Oracle`

### 9.3 实验组 C：threshold scan

固定扫描：

- [x] `tau=0.1`
- [x] `tau=0.3`
- [x] `tau=0.5`
- [x] `tau=0.7`
- [x] `tau=0.9`

每个 tau 都要记录：

- [x] overall MRR
- [x] fusion coverage
- [x] gain-positive precision
- [x] selected fusion ratio by subgroup

输出文件：

- [x] `outputs/router/eval/threshold_scan_logistic.csv`
- [x] `outputs/router/eval/threshold_scan_xgb.csv`
- [x] `outputs/router/figures/threshold_coverage_mrr.png`

已实现脚本：

- [x] `scripts/run_threshold_scan.py`

### 9.4 实验组 D：feature ablation

首轮只做 4 档，不要扩张：

- [x] `F1 = target_has_img only`
- [x] `F2 = F1 + direction + relation_gain_prior`
- [x] `F3 = F2 + text_img_cosine + img_is_missing_replaced`
- [x] `F4 = F3 + fusion_margin + struct_margin + delta_margin`

输出文件：

- [x] `outputs/router/eval/feature_ablation.csv`
- [x] `outputs/router/eval/feature_ablation.md`

已实现脚本：

- [x] `scripts/run_feature_ablation.py`

### 9.5 执行优先顺序

- [x] 先完成主结果对比。
- [x] 再完成 subgroup。
- [x] 再做 threshold scan。
- [x] 最后做 feature ablation。

原因：
- 主结果和 subgroup 最先决定这条线是否值得继续。
- threshold scan 是“收益阈值模型”这几个字的关键证据。
- feature ablation 是加固方法贡献，不是最早的 go/no-go 条件。

### 9.6 当前实验结论

- `run_threshold_scan.py` 与 `evaluate_router_results.py` 已实现，并通过本地静态编译检查。
- 当前这两个脚本的职责已经明确分开：`run_threshold_scan.py` 负责批量扫描 `tau` 并输出 coverage-performance 数据，`evaluate_router_results.py` 负责把现有 routing eval JSON 汇总成主结果表与 subgroup 表。
- `threshold_scan_logistic_delta_0.01.csv` 与 `threshold_scan_xgb_delta_0.01.csv` 已生成，且都呈现清晰的 coverage-performance tradeoff：随着 `tau` 升高，fusion coverage 单调下降，而 MRR 先升后趋于平台。以 `xgb` 为例，`tau=0.1/0.3/0.5/0.7/0.9` 的 MRR 约为 `0.251/0.292/0.308/0.316/0.314`，对应 coverage 约为 `0.483/0.333/0.252/0.185/0.096`。
- `gain_precision` 也随着 `tau` 升高而单调上升。以 `xgb` 为例，从 `0.327` 提升到 `0.846`；`logistic` 也从 `0.256` 提升到 `0.748`。这说明阈值确实在控制“选择 fusion 的保守程度”，符合收益阈值模型的核心预期。
- 当前 best learned router 是 `xgb + delta=0.01 + tau=0.7`，其 overall MRR 为 `0.3160`，高于 `logistic + tau=0.7` 的 `0.3111`，也高于固定 `Residual-only` 的约 `0.2930`。
- subgroup 上，routing 模式总体合理：`head_has_img` 的 fusion ratio 始终高于 `head_no_img`，而 `tail_no_img` 的 fusion ratio 虽然仍不低，但会随着 `tau` 提高从 `0.662` 降到 `0.113`（xgb，`tau=0.1 -> 0.9`），并且其 MRR 在高阈值时显著提升，说明更保守的阈值更符合 tail_no_img 的 structural-favorable 属性。
- 目前 `main_results_table.csv` 与 `subgroup_results_table.csv` 已生成，但它们当前只汇总了 learned router 的 routing 结果，还没有把 `Residual-only`、`Gate-only`、`Full Model`、`Rule-based`、`Oracle` 全部并入同一张最终论文表。因此 Phase 6 的 learned-router 部分已跑通，但“完整论文主表”还没最终收口。
- `run_feature_ablation.py` 已实现，并通过本地静态编译检查；它会直接复用现有 train/test feature 表、learned router 训练口径和 test routing 评估口径，因此不会引入新的实验协议分叉。
- feature ablation 结果已经支持“router 不只是学到 `target_has_img` 一个浅规则”这一点。`F1`（仅 `target_has_img`）在 `tau=0.5, delta=0.01` 下的 overall MRR 只有约 `0.1651`，而 `F4` 提升到 `0.2501`（logistic）和 `0.2556`（xgb），差距非常明显。
- `F2` 相比 `F1` 的提升已经很大，说明 `direction + relation_gain_prior` 这类 protocol / prior 特征是关键贡献。以 logistic 为例，overall MRR 从 `0.1651` 直接升到 `0.2333`；xgb 也从 `0.1651` 升到 `0.2306`。
- `F3` 相比 `F2` 的增益非常小，说明 `text_img_cosine + img_is_missing_replaced` 目前只是弱补充特征，而不是主要性能来源。
- `F4` 再次带来稳定提升，说明 expert confidence 特征（`fusion_margin / struct_margin / delta_margin`）是 learned router 的关键增强项。以 xgb 为例，train AUC 从 `0.746` 升到 `0.803`，overall MRR 从 `0.2303` 升到 `0.2556`。
- 在本轮 feature ablation 下，最强组合仍是 `xgb + F4`，优于 `logistic + F4`，也优于所有更浅的 feature set。这进一步说明方法贡献不只是“有图就开 fusion”，而是来自 protocol-aware prior 与 expert confidence 的组合。

---

## 10. Phase 7：整理图表与首轮结论

### 10.1 目标

把首轮开发结果整理成可直接支撑论文写作的材料。

### 10.2 本阶段必须完成的文件

- [x] `scripts/make_router_tables_figures.py`
- [x] `scripts/evaluate_router_results.py`

### 10.3 首轮必须产出的论文材料

#### 表
- [x] 表 1：主结果表
- [x] 表 2：subgroup 结果表
- [x] 表 3：feature ablation 表

#### 图
- [x] 图 1：threshold–coverage–MRR 曲线
- [x] 图 2：router feature importance 图
- [x] 图 3：selected fusion ratio by subgroup

#### 摘要结论文件
- [x] `outputs/router/eval/first_round_takeaways.md`

### 10.4 首轮结论必须回答的问题

- [x] Oracle 是否显著优于固定单 expert？
- [x] Learned router 是否优于 rule-based router？
- [x] Learned router 是否优于 Full Model？
- [x] Learned router 是否在 `head_has_img` 与 `tail_no_img` 上呈现合理选择模式？
- [x] threshold scan 是否显示清晰的 coverage–performance tradeoff？
- [x] feature ablation 是否表明 router 不只是学到 `target_has_img` 一个浅规则？

### 10.5 阶段完成标志

- [x] 你已经可以决定这条方法线是否值得正式写进论文主线。
- [x] 你已经拥有支撑方法节和实验节的第一批图表。

### 10.6 当前实验结论

- `make_router_tables_figures.py` 已实现，并已生成或支持生成 `main_results_table.csv`、`main_results_table.md`、`subgroup_results_table.csv`、`subgroup_results_table.md`、`feature_ablation.md` 与 `first_round_takeaways.md`。主表现已纳入 `Gate-only`、`Residual-only`、`Full Model`、`Oracle`、`Rule-based`、`logistic`、`xgb`；`Full Model` 的 subgroup 也已并入统一表。
- 从当前首轮结果看，这条方法线已经具备进入论文主线的条件。best learned router 为 `xgb + delta=0.01 + tau=0.7`，overall MRR 约 `0.3160`，高于 `Residual-only` 的 `0.2930` 与 `Full Model` 的 `0.2100`。
- `rule-based` 在 test 上几乎退化成“极低 coverage 的 residual-only”，虽然 overall MRR 接近 `Residual-only`，但它并没有提供 learned router 那样清晰的 coverage–precision tradeoff，也不能作为更强方法结论的支撑点。
- threshold scan 已经给出方法名中“gain threshold”最关键的证据：随着 `tau` 升高，coverage 单调下降、gain precision 单调上升，而 MRR 在中高阈值达到最佳。
- feature ablation 已经给出方法贡献的主要解释：真正有效的不是单独的 `target_has_img`，而是 `direction + relation_gain_prior` 与 expert confidence 特征的组合；这说明方法不是浅规则，而是有条件选择机制。
- `Oracle router` 已补齐，并给出当前最清晰的上界：overall MRR 约 `0.3337`，明显高于固定 `Residual-only` 的 `0.2930`、`Full Model` 的 `0.2100`，也高于当前 best learned router `xgb + delta=0.01 + tau=0.7` 的 `0.3160`。这说明方法线已成立，但仍有继续逼近上界的空间。
- `Rule-based` 在 test 上几乎退化成“极低 coverage 的 residual-only”，其 fusion coverage 仅约 `0.0187`，overall MRR 约 `0.2939`；它可以作为 heuristic baseline，但不足以替代 learned router。
- 已新增 `outputs/router/eval/final_results_manifest.json` 与 `final_results_manifest.md`，用于把“哪些文件算最终版、哪些只是 supporting input”正式钉死。当前 manifest 校验状态为 `PASS`，并确认 canonical final outputs 包括主表、subgroup 表、feature ablation 表、takeaways 与 3 张正式图。
- 当前还未完全收口的主要问题已经从“方法是否成立”收缩到“展示质量是否足够论文级”：
  - 仍需把现有结果进一步整理成更论文式的文字叙述与版式，而不再只是工程汇总视角。
  - 正式图文件（threshold curve / feature importance / subgroup fusion ratio）依赖 `matplotlib`；若在无该依赖的环境运行，脚本会保留表格与 markdown 总结，但跳过 PNG。

---

## 11. 首轮排期中的“先后关系”约束

下面这些依赖关系必须遵守：

- [x] 没有 `query_eval`，不能开始 label。
- [x] 没有 label，不能开始 learned router 训练。
- [x] 没有 relation prior，rule-based router 与部分特征不完整。
- [x] 没有 test features，不能开始 routing。
- [x] 没有 routing 输出，不能做 threshold scan。
- [x] 没有主结果和 subgroup，不能判断方法是否站得住。

---

## 12. 首轮建议的执行日程安排（按最实用顺序）

下面给一版非常实用的推进顺序。不是按自然日死卡，而是按“完成一个可验证产物再进入下一步”的原则。

### Day / Block 1：先把导出链路跑通

- [ ] 搭建目录结构。
- [x] 写 `export_query_eval.py` 最小版。
- [x] 只跑 `Gate-only + dev + seed1`。
- [x] 再跑 `Residual-only + dev + seed1`。
- [x] 验证 query 对齐。

**这一块没过，不要继续。**

### Day / Block 2：补齐全部 query_eval

- [x] 补齐 dev 的三个 seed。
- [x] 补齐 test 的三个 seed。
- [x] 输出 query_eval 的 summary。

### Day / Block 3：先把 label 定死

- [x] 实现 `build_gain_labels.py`。
- [x] 先跑 `delta=0.01`。
- [x] 看正样本比例与 subgroup 分布。
- [x] 再补 `delta=0.00` 与 `0.02`。
- [x] 实现并导出 relation prior。

### Day / Block 4：拼 feature 表

- [x] 实现 `build_router_features.py`。
- [ ] 先拼 `router_train_dev_delta_0.01.csv`。
- [ ] 再拼 `router_test_features.csv`。
- [ ] 检查 NaN、类别编码、字段一致性。

### Day / Block 5：先训练最简单 router

- [x] 先做 rule-based router。
- [x] 再训练脚本 `train_router.py`。
- [x] 再训练 Logistic + `delta=0.01`。
- [x] 若正常，再训练 XGBoost + `delta=0.01`。
- [x] 检查训练指标和 feature importance。

### Day / Block 6：第一次 test routing

- [x] 跑 `Logistic + tau=0.5`。
- [x] 跑 `XGBoost + tau=0.5`。
- [x] 先看 overall。
- [x] 再看 subgroup。

### Day / Block 7：补阈值扫描

- [x] 补 `tau=0.3 / 0.7`。
- [x] 再扩到 `0.1 / 0.9` 做阈值曲线。
- [x] 画 threshold–coverage–MRR 图。

### Day / Block 8：做 feature ablation

- [x] 实现 `run_feature_ablation.py`。
- [x] 跑 F1。
- [x] 跑 F2。
- [x] 跑 F3。
- [x] 跑 F4。
- [x] 对比 learned router 是否超越浅规则。

### Day / Block 9：整理结论

- [x] 汇总主表。
- [x] 汇总 subgroup 表。
- [x] 汇总阈值图。
- [x] 写 `first_round_takeaways.md`。
- [x] 决定下一步是继续扩方法，还是先回去修特征/label。

---

## 13. 首轮成功标准

首轮不要把目标定成“必须全面超过 Residual-only”。

更合理的首轮成功标准如下：

### S1：Oracle 必须证明有空间

- [x] `Oracle router` 应明显优于 `Full Model`。
- [x] 最好也高于 `Residual-only`。

### S2：Learned router 必须优于 heuristic

- [x] `Logistic` 或 `XGBoost` 至少有一个明显优于 `Rule-based router`。

### S3：subgroup 选择模式必须合理

- [x] 在 `head_has_img` 上，router 更接近 `Gate-only`。
- [x] 在 `tail_no_img` 上，router 更接近 `Residual-only`。

### S4：阈值扫描必须有解释性

- [x] `tau` 提高时，fusion coverage 应下降。
- [x] coverage 与 overall MRR 之间应出现可解释 tradeoff。

### S5：feature ablation 必须证明方法不只是浅规则

- [x] `F4` 应优于 `F1`。
- [x] learned router 不应只靠 `target_has_img` 单一特征存活。

如果这五条里满足 3 条以上，尤其是 S1 / S2 / S3 成立，那么这条线就值得继续推进为正式论文方法部分。

---

## 14. 首轮禁止事项

为了减少返工，这一轮明确不要做以下事情：

- [ ] 不要先写完整论文方法节再来补实验。
- [ ] 不要先做 soft routing。
- [ ] 不要先做 MLP / Transformer router。
- [ ] 不要先做 continuous gain regression。
- [ ] 不要一开始把 `ComplEx`、`Full Model fusion branch`、`Residual-only` 全部都当 expert 混着试。
- [ ] 不要先做多层级 routing（entity-level + query-level）。
- [ ] 不要边跑边改 label 定义却不记录版本。

---

## 15. 版本记录建议

为防止首轮实验混乱，建议每轮都固定版本号。

### 建议版本标签

- [ ] `v0.1-query-export`
- [ ] `v0.2-gain-label`
- [ ] `v0.3-router-features`
- [ ] `v0.4-logistic-router`
- [ ] `v0.5-xgb-router`
- [ ] `v0.6-threshold-scan`
- [ ] `v0.7-feature-ablation`
- [ ] `v0.8-first-round-summary`

### 每次更新至少记录

- [ ] 修改了哪个脚本。
- [ ] 修改了哪些特征。
- [ ] label 用的是哪一个 delta。
- [ ] threshold 用的是哪一组。
- [ ] 输出结果放在哪个目录。

---

## 16. 最后的执行建议

如果你希望这一轮尽量少走弯路，实际执行时请记住下面这条原则：

**先做能验证闭环是否成立的最小实现，再做加固论文说服力的扩展实验。**

所以最优先级永远是：

- [x] 先导出 query_eval
- [x] 再造 label
- [x] 再训最简单 router
- [x] 再看 test routing 是否真的有价值
- [x] 最后才去做漂亮的 ablation 和图表

只要这个顺序不乱，你这条“gain threshold + selective routing”的方法线就能以最低返工成本推进下去。
