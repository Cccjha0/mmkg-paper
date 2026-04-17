# 多模态融合收益阈值模型：开发任务清单（按代码文件 / 脚本 / 输出文件拆分）

## 0. 文档目的

本文档用于把上一份《router baseline + hard-threshold 实验方案》继续下钻成一份**可直接进入实现阶段**的开发任务清单。

这份文档重点回答以下问题：

1. 这轮代码实现的最小可行闭环到底是什么。
2. 需要新增哪些文件，哪些文件可以复用，哪些文件尽量不要动。
3. 每个脚本的职责、输入、输出、中间产物分别是什么。
4. 各阶段应如何运行，如何串起来，如何避免数据泄漏。
5. 每个阶段跑完后，应该验收什么结果，什么现象说明是正常的。
6. 首轮论文需要的表和图分别从哪些输出文件生成。

本文档默认你当前 repo 已经具备以下基础：

- OpenBG-IMG 训练与评估管线已经跑通。
- `Gate-only` 与 `Residual-only` 已能稳定训练并输出测试结果。
- 当前 repo 至少已有：数据读取、训练、评估、缓存 embedding 读取、filtered ranking 等功能。

如果当前 repo 的真实文件名与本文档建议不一致，**以功能对齐为准，不必强行逐字照抄文件名**。但为了减少歧义，本文档仍会给出一套建议的目录结构和文件命名。

---

## 1. 首轮实现目标：先跑通最小闭环

这一轮不要一开始就追求“完整论文所有实验全部自动化”，而要先实现一个最小可行闭环：

1. 训练或加载两个固定 expert：
   - Fusion expert = `Gate-only`
   - Structural expert = `Residual-only`
2. 在 **dev 集** 上导出两个 expert 的逐 query 评估结果。
3. 在 dev 上构造 gain label。
4. 从 dev 结果中构造 router 训练特征。
5. 训练一个简单 router：
   - Logistic Regression
   - XGBoost
6. 在 **test 集** 上导出两个 expert 的逐 query 结果与候选打分摘要。
7. 用训练好的 router + hard threshold 在 test 上执行 query-level expert 选择。
8. 输出 overall / subgroup / threshold scan / feature ablation 所需结果。

只有这个闭环跑通，后续再考虑：

- soft routing
- end-to-end finetune
- 更多 expert 组合
- continuous gain regression
- representation-level routing

---

## 2. 建议目录结构

建议在现有 repo 下新增一个相对独立的 router 目录，避免污染原本 trainer / model / eval 主逻辑。

建议结构如下：

```text
repo/
├─ configs/
│  ├─ router/
│  │  ├─ router_base.yaml
│  │  ├─ router_logistic.yaml
│  │  ├─ router_xgb.yaml
│  │  ├─ threshold_scan.yaml
│  │  └─ feature_ablation.yaml
│
├─ scripts/
│  ├─ export_query_eval.py
│  ├─ build_gain_labels.py
│  ├─ build_relation_priors.py
│  ├─ build_router_features.py
│  ├─ train_router.py
│  ├─ run_hard_threshold_routing.py
│  ├─ evaluate_router_results.py
│  ├─ run_threshold_scan.py
│  ├─ run_feature_ablation.py
│  └─ make_router_tables_figures.py
│
├─ router/
│  ├─ __init__.py
│  ├─ schemas.py
│  ├─ feature_utils.py
│  ├─ label_utils.py
│  ├─ prior_utils.py
│  ├─ router_models.py
│  ├─ routing_utils.py
│  ├─ metrics.py
│  └─ io_utils.py
│
├─ outputs/
│  ├─ router/
│  │  ├─ dev/
│  │  ├─ test/
│  │  ├─ priors/
│  │  ├─ features/
│  │  ├─ models/
│  │  ├─ routing/
│  │  ├─ eval/
│  │  └─ figures/
│
└─ existing_training_code/
```

说明：

- `scripts/` 放可直接运行的入口。
- `router/` 放纯函数、数据结构、工具逻辑。
- `outputs/router/` 放所有中间产物和结果，避免和原训练输出混在一起。

---

## 3. 建议不要大改的现有模块

为了降低风险，这一轮建议**尽量不动**下面几类现有核心逻辑：

1. 原始 `Gate-only` / `Residual-only` 模型定义。
2. 原训练主循环。
3. 原 filtered ranking evaluator 主逻辑。
4. 原数据切分与缓存读取逻辑。

本轮最好的做法是：

- 通过“导出评估结果”的方式与现有代码解耦；
- 在 router 层单独处理 label、feature、选择逻辑；
- 最后在 inference 阶段组合两个 expert 的结果。

也就是说，本轮不是去重写主模型，而是新增一个**分析-决策层**。

---

## 4. 数据契约：必须先统一的中间表结构

这一轮实现最容易出错的地方不是模型，而是**中间文件字段不统一**。因此，先把几个关键 CSV / JSONL 的字段定义写死。

---

### 4.1 `query_eval` 基础表

文件用途：

- 保存某个 expert 在某个 split 上的逐 query 评估结果。
- 后续 label 构造、feature 构造、router 训练、routing 决策都依赖它。

建议文件名：

```text
outputs/router/dev/gate_only_query_eval.csv
outputs/router/dev/residual_only_query_eval.csv
outputs/router/test/gate_only_query_eval.csv
outputs/router/test/residual_only_query_eval.csv
```

建议字段：

| 字段名 | 类型 | 含义 |
|---|---:|---|
| `query_id` | str | 唯一 query 标识 |
| `split` | str | `dev` / `test` |
| `direction` | str | `head` / `tail` |
| `relation_id` | str/int | relation 编号 |
| `relation_name` | str | 若可取到则保留 |
| `head_id` | str/int | query 中 head 实体 |
| `tail_id` | str/int | query 中 tail 实体 |
| `target_entity_id` | str/int | 当前 query 的正确 target |
| `target_position` | str | `head` / `tail` |
| `target_has_img` | int | 0/1 |
| `target_regime` | str | `head_has_img` / `head_no_img` / `tail_no_img` |
| `expert_name` | str | `gate_only` / `residual_only` |
| `rank` | int | filtered rank |
| `rr` | float | reciprocal rank = 1/rank |
| `hit1` | int | rank<=1 |
| `hit3` | int | rank<=3 |
| `hit10` | int | rank<=10 |
| `top1_score` | float | 当前 expert top1 候选分数 |
| `top2_score` | float | 当前 expert top2 候选分数 |
| `score_margin` | float | top1_score - top2_score |
| `correct_score` | float | 正确 target 分数 |
| `seed` | int | seed 编号 |

注意：

- `query_id` 必须在两个 expert 间一一对齐。
- `target_regime` 强烈建议导出时直接写好，不要后处理时再推断。
- 如果你当前 evaluator 不能直接拿到 `top1_score/top2_score`，那就至少先导出 `correct_score` 和 `rank`，但最好补出来。

---

### 4.2 `relation_gain_stats` 表

用途：

- 为 rule-based router 与 learned router 提供 relation prior。

建议文件名：

```text
outputs/router/priors/relation_gain_stats_dev.csv
```

建议字段：

| 字段名 | 类型 | 含义 |
|---|---:|---|
| `relation_id` | str/int | relation 编号 |
| `relation_name` | str | 若可取到则保留 |
| `n_queries` | int | 该 relation 在 dev 中的 query 数 |
| `mean_rr_gate` | float | Gate-only 平均 RR |
| `mean_rr_residual` | float | Residual-only 平均 RR |
| `mean_delta_rr` | float | mean(rr_gate - rr_residual) |
| `fusion_win_rate` | float | gate_only 胜出比例 |
| `struct_win_rate` | float | residual_only 胜出比例 |
| `head_has_img_ratio` | float | 该 relation 中 head_has_img 比例 |
| `tail_no_img_ratio` | float | 该 relation 中 tail_no_img 比例 |
| `is_visual_prior` | int | 依据阈值判断是否是 fusion-favorable relation |

---

### 4.3 `router_features` 表

用途：

- router 训练与推理直接使用的特征表。

建议文件名：

```text
outputs/router/features/router_train_dev.csv
outputs/router/features/router_test_features.csv
```

建议字段：

| 字段名 | 类型 | 含义 |
|---|---:|---|
| `query_id` | str | 与 query_eval 对齐 |
| `seed` | int | seed |
| `direction` | int/str | head/tail，可编码 |
| `target_has_img` | int | 0/1 |
| `target_regime` | str | subgroup |
| `relation_id` | str/int | relation |
| `relation_gain_prior` | float | 从 dev/train 统计得到 |
| `relation_fusion_win_rate` | float | relation 级 fusion 胜率 |
| `relation_support` | int | relation 支持度 |
| `text_img_cosine` | float | target 的 text/image cosine |
| `img_is_missing_replaced` | int | 0/1 |
| `fusion_margin` | float | gate_only top1-top2 |
| `struct_margin` | float | residual_only top1-top2 |
| `fusion_correct_score` | float | gate_only 正确 target 分数 |
| `struct_correct_score` | float | residual_only 正确 target 分数 |
| `delta_margin` | float | fusion_margin - struct_margin |
| `label_gain` | int | y，训练表才有 |

说明：

- `router_train_dev.csv` 有 label。
- `router_test_features.csv` 不能有 test 构造出来的 gain label。

---

## 5. 新增代码文件与职责拆分

下面按“文件 → 作用 → 输入 → 输出 → 核心检查点”的方式展开。

---

## 5.1 `scripts/export_query_eval.py`

### 作用

从已有 expert 模型检查点出发，在指定 split 上导出逐 query 评估结果。

### 为什么它是第一优先级

因为整个 router 项目都建立在两个 expert 的 query-level 输出之上。没有这一步，后续所有事情都无法开始。

### 输入

- expert 名称：`gate_only` / `residual_only`
- split：`dev` / `test`
- checkpoint 路径
- 数据集路径
- evaluator 配置
- seed

### 输出

- `gate_only_query_eval.csv`
- `residual_only_query_eval.csv`

### 建议命令形式

```bash
python scripts/export_query_eval.py \
  --expert gate_only \
  --split dev \
  --ckpt path/to/gate_only_seed1.ckpt \
  --seed 1 \
  --out outputs/router/dev/gate_only_query_eval_seed1.csv
```

### 关键实现点

1. 需要确保 query 对齐。
2. 最好按当前 evaluator 的 query 顺序生成 `query_id`。
3. `query_id` 的生成规则必须稳定。

建议：

```text
query_id = f"{split}|{seed}|{direction}|{relation_id}|{head_id}|{tail_id}|{target_entity_id}"
```

如果 head/tail query 形式不同，建议显式写：

```text
head_query_id = f"dev|1|head|r=12|t=398|target=102"
tail_query_id = f"dev|1|tail|h=21|r=8|target=901"
```

### 最低验收标准

- Gate-only 与 Residual-only 在同一 split 同一 seed 上的 `query_id` 集合完全一致。
- `rank` 和 `rr` 字段非空。
- `target_regime` 只出现这三个值：
  - `head_has_img`
  - `head_no_img`
  - `tail_no_img`
- 每个 CSV 行数与该 split 的 query 总数一致。

---

## 5.2 `router/schemas.py`

### 作用

统一本轮项目中所有中间数据结构，避免不同脚本字段名漂移。

### 建议内容

可定义：

- `QueryEvalRecord`
- `RelationGainStatRecord`
- `RouterFeatureRecord`
- `RouterPredictionRecord`

如果你不想引入 dataclass / pydantic，也至少把字段常量集中放这里。

### 关键价值

不是“高级封装”，而是防止：

- 一个脚本用 `rel_id`
- 一个脚本用 `relation`
- 一个脚本用 `relation_id`

最后 merge 直接炸掉。

---

## 5.3 `scripts/build_gain_labels.py`

### 作用

基于 dev 集两个 expert 的 query_eval 输出，构造 gain label。

### 输入

- `gate_only_query_eval.csv`
- `residual_only_query_eval.csv`
- `delta` 阈值

### 输出

建议输出：

```text
outputs/router/dev/gain_labels_delta_0.00.csv
outputs/router/dev/gain_labels_delta_0.01.csv
outputs/router/dev/gain_labels_delta_0.02.csv
```

### label 定义（首版固定）


a. 对每个 query：

\[
\Delta(q)=RR_f(q)-RR_s(q)
\]

b. 构造二分类标签：

\[
y(q)=\mathbf{1}[\Delta(q)>\delta]
\]

其中：

- `f` = Gate-only
- `s` = Residual-only
- `delta ∈ {0.00, 0.01, 0.02}`

### 输出字段建议

| 字段名 | 含义 |
|---|---|
| `query_id` | query 唯一标识 |
| `rr_fusion` | Gate-only RR |
| `rr_struct` | Residual-only RR |
| `delta_rr` | rr_fusion - rr_struct |
| `delta_threshold` | 当前 delta |
| `label_gain` | 0/1 |
| `direction` | head/tail |
| `target_regime` | subgroup |
| `relation_id` | relation |
| `seed` | seed |

### 最低验收标准

- `delta_rr` 数值合理，不应全为 0。
- 不同 delta 下，正样本比例发生变化。
- 三个 delta 的 label 分布都要记录下来。

建议额外输出一个 summary json：

```text
outputs/router/dev/gain_label_summary_delta_0.01.json
```

内容示例：

```json
{
  "n_total": 15000,
  "n_positive": 4120,
  "positive_rate": 0.2747,
  "by_regime": {
    "head_has_img": 0.53,
    "head_no_img": 0.21,
    "tail_no_img": 0.08
  }
}
```

这一步非常重要，因为它会直接告诉你：

- label 是否与旧稿分析一致；
- 若 `head_has_img` 的正样本比例也不高，那说明 Gate-only 可能没你想象中稳；
- 若 `tail_no_img` 正样本比例反而很高，说明前面某处实现或对齐可能出了问题。

---

## 5.4 `scripts/build_relation_priors.py`

### 作用

从 dev query_eval / gain_label 中统计 relation 级先验，供：

- rule-based router
- learned router
- feature ablation

使用。

### 输入

- dev 上的 Gate-only query_eval
- dev 上的 Residual-only query_eval
- 或 gain label 文件

### 输出

```text
outputs/router/priors/relation_gain_stats_delta_0.00.csv
outputs/router/priors/relation_gain_stats_delta_0.01.csv
outputs/router/priors/relation_gain_stats_delta_0.02.csv
```

### 建议核心统计

对每个 relation：

1. `mean_delta_rr`
2. `fusion_win_rate`
3. `struct_win_rate`
4. `n_queries`
5. `head_has_img_ratio`
6. `tail_no_img_ratio`
7. `is_visual_prior`

其中：

```text
is_visual_prior = 1 if mean_delta_rr > gamma else 0
```

首版直接试：

- `gamma = 0.0`
- `gamma = 0.005`

### 最低验收标准

- 支持度过低的 relation 要保留 `n_queries`，后续决定是否过滤。
- relation prior 只能用 dev/train 统计，不能从 test 反推。
- 对于 support 太低的 relation，建议加平滑，例如：

```text
shrunk_prior = mean_delta_rr * n / (n + k)
```

首版若想简单，可先不加，但至少保留 `n_queries` 供后续分析。

---

## 5.5 `router/prior_utils.py`

### 作用

封装 relation prior 的计算与 merge 逻辑。

### 推荐函数

- `compute_relation_gain_stats(df_fusion, df_struct, delta)`
- `mark_visual_prior(df_stats, gamma)`
- `merge_relation_priors(df_features, df_stats)`

### 价值

避免把统计逻辑散落在 notebook 或多个脚本里，后面实验改阈值时很难追。

---

## 5.6 `router/feature_utils.py`

### 作用

集中管理 router 特征构造。

### 首版必须实现的特征

#### 协议条件特征

1. `direction`
2. `target_has_img`
3. `target_regime`
4. `relation_id`
5. `relation_gain_prior`
6. `relation_fusion_win_rate`
7. `relation_support`

#### 模态一致性特征

8. `text_img_cosine`
9. `img_is_missing_replaced`

#### expert 置信度特征

10. `fusion_margin`
11. `struct_margin`
12. `fusion_correct_score`
13. `struct_correct_score`
14. `delta_margin`

### 暂时不要做的特征

以下特征可留作后续增强，但首轮不建议强依赖：

- gate entropy
- projection gradient statistics
- mix weight diagnostics
- per-query branch internal activations
- graph topology 复杂统计
- 候选分布高阶不确定性指标

原因：

- 实现复杂度高
- 与现有 router 首版目标不匹配
- 不利于先验证路线是否成立

### 建议函数

- `compute_text_img_cosine(...)`
- `compute_margin(top1, top2)`
- `merge_expert_features(df_gate, df_residual)`
- `attach_relation_features(df, df_prior)`
- `encode_direction(...)`

---

## 5.7 `scripts/build_router_features.py`

### 作用

从 query_eval + relation priors + entity embedding cache 中构建训练 / 推理特征表。

### 输入

- Gate-only query_eval
- Residual-only query_eval
- relation priors
- text / image embedding cache
- gain labels（仅训练表需要）

### 输出

```text
outputs/router/features/router_train_dev_delta_0.00.csv
outputs/router/features/router_train_dev_delta_0.01.csv
outputs/router/features/router_train_dev_delta_0.02.csv
outputs/router/features/router_test_features.csv
```

### 关键实现点

1. **test features 不能带 label**。
2. `text_img_cosine` 若 target 无图：
   - 可设为 0；
   - 同时保留 `img_is_missing_replaced=1`；
   - 不建议直接 NaN。
3. `relation_id` 若喂给 Logistic：
   - 可 one-hot / target encoding；
   - 首版更简单是只保留 relation prior，不强行 one-hot。
4. `relation_id` 若喂给 XGBoost：
   - 可 label encode；
   - 但要注意不要误导为有序数值，可先主要依赖 relation prior。

### 最低验收标准

- 训练表与 label 一一对应。
- 测试表与 test query_eval 一一对应。
- 数值列没有异常 NaN / inf。
- `fusion_margin` 与 `struct_margin` 分布合理，不是常数。

---

## 5.8 `router/label_utils.py`

### 作用

统一 gain label 的计算逻辑，避免 build script 和后续分析使用两套定义。

### 必须包含的内容

- `compute_delta_rr(rr_fusion, rr_struct)`
- `build_binary_gain_label(delta_rr, delta)`
- `summarize_gain_distribution(df)`

### 原则

**label 定义只在这里写一次。**

否则后期你很容易出现：

- train_router.py 用 `>`
- analysis notebook 用 `>=`
- 论文里又写成 margin ranking

最后结果会自相矛盾。

---

## 5.9 `router/router_models.py`

### 作用

封装首版 router 模型。

### 首版必须实现

1. `LogisticRouter`
2. `XGBRouter`

### 建议接口

```python
class BaseRouter:
    def fit(self, X, y): ...
    def predict_proba(self, X): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

### Logistic 首版建议

- 可使用 sklearn
- 类别不平衡时考虑：
  - `class_weight='balanced'`
- 标准化：
  - numeric feature 建议加 scaler

### XGBoost 首版建议

- 二分类目标
- 指标：`logloss` / `auc`
- 不要过度调参，首版先稳住

建议只调少量参数：

- `max_depth`
- `learning_rate`
- `n_estimators`
- `subsample`
- `colsample_bytree`

### 最低验收标准

- 在 dev 训练集上能稳定输出概率，不是全 0.5。
- 至少能计算 AUC / F1 / precision / recall。
- 不同 delta 下结果有变化。

---

## 5.10 `scripts/train_router.py`

### 作用

训练 router，并输出预测概率与训练摘要。

### 输入

- `router_train_dev_delta_xx.csv`
- router config
- model type: logistic / xgb

### 输出

```text
outputs/router/models/logistic_delta_0.01.pkl
outputs/router/models/xgb_delta_0.01.pkl
outputs/router/eval/router_dev_pred_logistic_delta_0.01.csv
outputs/router/eval/router_train_summary_logistic_delta_0.01.json
```

### summary json 建议字段

```json
{
  "model": "logistic",
  "delta": 0.01,
  "n_train": 15000,
  "positive_rate": 0.27,
  "auc": 0.76,
  "f1": 0.61,
  "precision": 0.58,
  "recall": 0.64,
  "features": ["target_has_img", "relation_gain_prior", "fusion_margin"]
}
```

### 最低验收标准

- 训练脚本可重复跑。
- 模型文件可保存可加载。
- 输出概率文件与输入 query 一一对应。

---

## 5.11 `scripts/run_hard_threshold_routing.py`

### 作用

这是整轮方法的核心执行脚本。

它负责：

1. 读取 test 上两个 expert 的 query_eval / candidate score 摘要。
2. 读取训练好的 router 概率或直接在线推理。
3. 对每个 query 应用 hard threshold。
4. 决定该 query 使用哪个 expert。
5. 生成最终的 routed 结果。

### 输入

- `router_test_features.csv`
- `gate_only_query_eval.csv`（test）
- `residual_only_query_eval.csv`（test）
- 训练好的 router model
- threshold `tau`

### 输出

```text
outputs/router/routing/routed_test_logistic_delta_0.01_tau_0.50.csv
outputs/router/routing/routed_test_xgb_delta_0.01_tau_0.50.csv
```

### 输出字段建议

| 字段名 | 含义 |
|---|---|
| `query_id` | query id |
| `router_prob` | 预测为 gain-positive 的概率 |
| `threshold` | 当前 tau |
| `use_fusion` | 0/1 |
| `selected_expert` | `gate_only` / `residual_only` |
| `rank_final` | 最终选中 expert 的 rank |
| `rr_final` | 最终 RR |
| `rank_gate` | gate rank |
| `rank_residual` | residual rank |
| `target_regime` | subgroup |
| `direction` | head/tail |
| `relation_id` | relation |
| `seed` | seed |

### 注意

首版只做 **query-level hard switch**：

- 整个 query 只选一个 expert；
- 不做 candidate-level 混合；
- 不做 score interpolation with alpha in (0,1)。

### 最低验收标准

- `use_fusion=1` 的 query 数量会随 tau 变化。
- tau 越高，fusion coverage 越低。
- routed 结果的 `rank_final` 等于被选中 expert 的 rank。

---

## 5.12 `router/routing_utils.py`

### 作用

统一 hard threshold 规则与路由选择逻辑。

### 推荐函数

- `hard_route(prob, tau)`
- `select_expert_row(row_gate, row_struct, use_fusion)`
- `compute_coverage(df_routed)`
- `compute_gain_precision(df_routed)`

### 说明

`compute_gain_precision` 的用途是：

- 在路由为 fusion 的 query 中，真正 `label_gain=1` 的比例是多少。
- 这会成为 threshold scan 的一个关键轴。

---

## 5.13 `scripts/evaluate_router_results.py`

### 作用

对 routed 结果进行整体与 subgroup 评估。

### 输入

- `routed_test_*.csv`

### 输出

```text
outputs/router/eval/router_eval_logistic_delta_0.01_tau_0.50.json
outputs/router/eval/router_eval_logistic_delta_0.01_tau_0.50_by_regime.csv
```

### 必须输出的指标

#### overall

- MRR
- Hits@1
- Hits@3
- Hits@10
- fusion coverage

#### subgroup

- `head_has_img`
- `head_no_img`
- `tail_no_img`

各 subgroup 都输出：

- MRR
- Hits@1
- Hits@3
- Hits@10
- 该 subgroup 的 fusion coverage

### 最低验收标准

- subgroup 汇总后的样本数与 overall 一致。
- `head_has_img + head_no_img + tail_no_img = total`。

---

## 5.14 `scripts/run_threshold_scan.py`

### 作用

批量扫描不同 threshold，生成 threshold–coverage–performance 曲线数据。

### 首版阈值集合

```text
tau ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
```

### 输入

- 固定 model
- 固定 delta
- 固定 feature set
- test features
- 两个 expert test 结果

### 输出

```text
outputs/router/eval/threshold_scan_xgb_delta_0.01.csv
```

### 建议字段

| 字段名 | 含义 |
|---|---|
| `tau` | threshold |
| `overall_mrr` | overall MRR |
| `hits1` | overall Hits@1 |
| `fusion_coverage` | 路由到 fusion 的比例 |
| `gain_precision` | 路由到 fusion 的样本中，真正 gain-positive 的比例 |
| `mrr_head_has_img` | subgroup |
| `mrr_head_no_img` | subgroup |
| `mrr_tail_no_img` | subgroup |

### 最低验收标准

- `fusion_coverage` 随 tau 上升单调下降或近似下降。
- 曲线形态合理，不应出现 tau 越高 coverage 越高的情况。

---

## 5.15 `scripts/run_feature_ablation.py`

### 作用

运行首版 feature ablation。

### 首版固定四组

- **F1**: `target_has_img`
- **F2**: `target_has_img + direction + relation_gain_prior`
- **F3**: `F2 + text_img_cosine + img_is_missing_replaced`
- **F4**: `F3 + fusion_margin + struct_margin + delta_margin`

### 输出

```text
outputs/router/eval/feature_ablation_logistic_delta_0.01.csv
outputs/router/eval/feature_ablation_xgb_delta_0.01.csv
```

### 输出字段建议

| 字段名 | 含义 |
|---|---|
| `feature_set` | F1/F2/F3/F4 |
| `model` | logistic/xgb |
| `delta` | label threshold |
| `tau` | routing threshold |
| `overall_mrr` | overall MRR |
| `fusion_coverage` | coverage |
| `auc` | router 分类质量 |
| `f1` | router F1 |

### 预期解释方向

- 若 F4 明显优于 F1，说明 router 不是只在学“有图就开融合”。
- 若 F2 已经很强，说明 protocol + relation prior 非常关键。

---

## 5.16 `scripts/make_router_tables_figures.py`

### 作用

从 eval 输出统一生成论文需要的表格和图。

### 首轮至少生成 4 个对象

#### 表 1：主结果表

列：

- Residual-only
- Gate-only
- Full Model
- Oracle router
- Rule-based router
- Logistic router
- XGBoost router

指标：

- MRR
- Hits@1
- Hits@3
- Hits@10

#### 表 2：subgroup 结果表

列：

- `head_has_img`
- `head_no_img`
- `tail_no_img`

行：

- Residual-only
- Gate-only
- Rule-based router
- Best learned router

#### 图 1：threshold–coverage–MRR 曲线

至少包含：

- x 轴：threshold tau
- 左 y 轴：overall MRR
- 右 y 轴：fusion coverage

#### 表 3：feature ablation

行：F1-F4
列：AUC, F1, overall MRR, coverage

### 输出路径建议

```text
outputs/router/figures/
outputs/router/tables/
```

---

## 6. Rule-based router 的单独实现要求

因为它既是 baseline，也是 sanity check，所以单独说明。

### 文件建议

- `router/router_models.py` 中增加 `RuleBasedRouter`
- 或者单独 `router/rule_based.py`

### 首版规则固定为

```text
use_fusion = 1
iff
(target_has_img == 1) AND (relation_gain_prior > gamma)
```

其中：

- `gamma ∈ {0.0, 0.005}`

### 为什么必须实现它

因为它是对“旧稿分析结论是否已经足够转化为有效 heuristic”的直接检验。

如果 Rule-based router 已经接近 learned router，说明：

- learned 部分贡献有限；
- 方法写法要更收敛，不能夸大。

如果 learned router 明显超过 rule-based，则说明：

- 学到的不只是 protocol heuristic；
- 方法贡献更稳。

---

## 7. Oracle router：必须做的上界实验

### 文件建议

可以作为 `scripts/run_hard_threshold_routing.py` 的一个特殊模式：

```text
--router oracle
```

### Oracle 定义

对每个 query：

- 若 `rr_gate > rr_residual`，选 Gate-only
- 否则选 Residual-only

### 为什么必须做

它回答的是最根本的问题：

> “即使 selector 完美，这条路到底有没有足够空间？”

若 Oracle 只比 Residual-only 高一点点，说明这条研究线空间有限。  
若 Oracle 明显更好，说明选择机制本身是值得做的。

### 输出

```text
outputs/router/eval/oracle_router_eval.json
outputs/router/eval/oracle_router_by_regime.csv
```

---

## 8. 配置文件建议

建议新增这些配置文件。

---

### 8.1 `configs/router/router_base.yaml`

建议包含：

```yaml
output_dir: outputs/router
split_train: dev
split_test: test
fusion_expert: gate_only
struct_expert: residual_only
label_delta: 0.01
routing_threshold: 0.5
seed_list: [1, 2, 3]
```

---

### 8.2 `configs/router/router_logistic.yaml`

```yaml
model_type: logistic
class_weight: balanced
use_scaler: true
features:
  - target_has_img
  - direction
  - relation_gain_prior
  - text_img_cosine
  - img_is_missing_replaced
  - fusion_margin
  - struct_margin
  - delta_margin
```

---

### 8.3 `configs/router/router_xgb.yaml`

```yaml
model_type: xgb
max_depth: 4
learning_rate: 0.05
n_estimators: 300
subsample: 0.9
colsample_bytree: 0.9
objective: binary:logistic
eval_metric: auc
features:
  - target_has_img
  - direction
  - relation_gain_prior
  - text_img_cosine
  - img_is_missing_replaced
  - fusion_margin
  - struct_margin
  - delta_margin
```

---

### 8.4 `configs/router/threshold_scan.yaml`

```yaml
tau_list: [0.1, 0.3, 0.5, 0.7, 0.9]
```

---

### 8.5 `configs/router/feature_ablation.yaml`

```yaml
feature_sets:
  F1: [target_has_img]
  F2: [target_has_img, direction, relation_gain_prior]
  F3: [target_has_img, direction, relation_gain_prior, text_img_cosine, img_is_missing_replaced]
  F4: [target_has_img, direction, relation_gain_prior, text_img_cosine, img_is_missing_replaced, fusion_margin, struct_margin, delta_margin]
```

---

## 9. 执行顺序：必须按这个顺序推进

首轮建议严格按下面顺序做，不要跳步骤。

---

### 阶段 A：导出 expert query-level 结果

先完成：

- `export_query_eval.py`

交付物：

- 4 份 CSV（dev/test × gate/residual）

验收点：

- query 完整对齐
- `target_regime` 分布正确
- rank/rr 合法

---

### 阶段 B：构造 gain label 与 relation priors

再完成：

- `build_gain_labels.py`
- `build_relation_priors.py`

交付物：

- 3 个 delta 对应的 gain label 文件
- relation prior 文件
- label summary json

验收点：

- 正样本比例合理
- 与旧稿的 subgroup 结论方向一致

---

### 阶段 C：构造 router features

完成：

- `build_router_features.py`

交付物：

- train/dev feature 表
- test feature 表

验收点：

- 无泄漏
- 无 NaN
- 特征分布合理

---

### 阶段 D：训练 router

完成：

- `train_router.py`
- Logistic
- XGBoost

交付物：

- 模型文件
- dev 概率输出
- summary json

验收点：

- 预测概率不塌缩
- AUC/F1 可报告

---

### 阶段 E：运行 hard-threshold routing

完成：

- `run_hard_threshold_routing.py`
- `evaluate_router_results.py`

交付物：

- routed test CSV
- overall eval json
- subgroup csv

验收点：

- tau 提高时 coverage 下降
- subgroup 模式符合预期

---

### 阶段 F：批量扫描与消融

完成：

- `run_threshold_scan.py`
- `run_feature_ablation.py`

交付物：

- threshold scan csv
- feature ablation csv

验收点：

- 曲线可解释
- F4 相对 F1 是否有提升

---

### 阶段 G：产出论文表图

完成：

- `make_router_tables_figures.py`

交付物：

- 主结果表
- subgroup 表
- threshold 图
- feature ablation 表

---

## 10. 数据泄漏与实现风险控制

这是本轮必须死守的部分。

### 10.1 绝对不能做的事

1. **不能用 test 集构造 gain label。**
2. **不能用 test 集统计 relation priors。**
3. **不能先看 test 表现再回过头调 label 规则并宣称这是固定方案。**
4. **不能让 router 直接看到 test 的“哪个 expert 更优”真值。**

### 10.2 建议做法

- 所有 label/prior 一律来自 dev 或 train。
- test 只用于最后推理与报告。
- 若之后需要更稳，可升级为：
  - train 上做 prior
  - dev 上选 delta/tau
  - test 上报告最终结果

### 10.3 relation prior 的一个现实问题

若 relation 支持度很低，`mean_delta_rr` 会非常不稳定。

首版做法：

- 保留 `relation_support`
- 结果分析时检查 low-support relation 是否导致 router 不稳

增强版做法：

- 加 shrinkage / smoothing

首轮先不强行上增强版，但文档里必须记录这个风险。

---

## 11. 首轮成功标准：跑完之后怎么判断这条路值不值得继续

### S1：Oracle 必须体现空间

- Oracle router 应明显优于 Full Model
- 最好也优于 Residual-only

若 Oracle 本身几乎没有提升，说明这条路理论空间有限。

### S2：Learned router 应优于 Rule-based router

否则说明你学到的东西不多，首版方法更像 heuristic。

### S3：subgroup 模式必须对

期望模式：

- `head_has_img`：路由结果更接近 Gate-only
- `tail_no_img`：路由结果更接近 Residual-only
- `overall`：优于 Full Model，并尽量接近 Oracle

### S4：threshold scan 需要有可解释趋势

- tau 越高，coverage 越低
- 存在一个相对合理的性能-覆盖率折中点

### S5：feature ablation 能说明 router 不只是“看到有图就开融合”

即：

- F4 应该比 F1 更有说服力
- 至少 learned router 应该显著超过仅 `target_has_img` 规则

---

## 12. 首轮不做事项清单

为防止发散，这些内容明确列为 **暂不做**：

1. 不做 representation-level routing。
2. 不做 alpha 连续插值加权。
3. 不做 end-to-end 联训。
4. 不做 Full Model 作为 fusion expert。
5. 不做 ComplEx 作为首版 structural expert。
6. 不做 continuous gain regression。
7. 不做复杂神经 router。
8. 不做 candidate-level routing。
9. 不做额外 benchmark 扩展。

这不是说它们没价值，而是为了保证首轮能回答一个最关键的问题：

> “基于当前已有 gain-boundary 发现，简单的 gain-aware hard-threshold routing 是否已经能形成一个成立的方法扩展？”

---

## 13. 建议的开发优先级（按工作量与收益排序）

### P0：必须先做

1. `export_query_eval.py`
2. `build_gain_labels.py`
3. `build_relation_priors.py`
4. `build_router_features.py`
5. `train_router.py`
6. `run_hard_threshold_routing.py`
7. `evaluate_router_results.py`

### P1：首轮论文最好有

8. `run_threshold_scan.py`
9. `run_feature_ablation.py`
10. `make_router_tables_figures.py`

### P2：后续增强

11. more priors / smoothing
12. ComplEx 扩展
13. soft routing
14. stronger router architectures

---

## 14. 一份最小命令流示例

以下是一套首轮建议命令流，仅作参考。

### Step 1：导出 expert 结果

```bash
python scripts/export_query_eval.py --expert gate_only --split dev --seed 1
python scripts/export_query_eval.py --expert residual_only --split dev --seed 1
python scripts/export_query_eval.py --expert gate_only --split test --seed 1
python scripts/export_query_eval.py --expert residual_only --split test --seed 1
```

### Step 2：构造 gain label

```bash
python scripts/build_gain_labels.py --delta 0.01
```

### Step 3：构造 relation prior

```bash
python scripts/build_relation_priors.py --delta 0.01 --gamma 0.0
```

### Step 4：构造 feature

```bash
python scripts/build_router_features.py --delta 0.01
```

### Step 5：训练 logistic router

```bash
python scripts/train_router.py --config configs/router/router_logistic.yaml --delta 0.01
```

### Step 6：训练 xgb router

```bash
python scripts/train_router.py --config configs/router/router_xgb.yaml --delta 0.01
```

### Step 7：在 test 上做 routing

```bash
python scripts/run_hard_threshold_routing.py --model logistic --delta 0.01 --tau 0.5
python scripts/run_hard_threshold_routing.py --model xgb --delta 0.01 --tau 0.5
```

### Step 8：评估

```bash
python scripts/evaluate_router_results.py --model logistic --delta 0.01 --tau 0.5
python scripts/evaluate_router_results.py --model xgb --delta 0.01 --tau 0.5
```

### Step 9：threshold scan

```bash
python scripts/run_threshold_scan.py --model xgb --delta 0.01
```

### Step 10：feature ablation

```bash
python scripts/run_feature_ablation.py --model xgb --delta 0.01
```

### Step 11：画图表

```bash
python scripts/make_router_tables_figures.py
```

---

## 15. 论文映射：这些代码结果最终会写到哪里

最后，把代码产物与论文章节映射起来，避免做了实验却不知道放哪。

### 15.1 方法部分

对应：

- `build_gain_labels.py` → Gain label construction
- `router_models.py` → Router definition
- `run_hard_threshold_routing.py` → Hard-threshold inference rule

### 15.2 主实验部分

对应：

- `evaluate_router_results.py` → main overall comparison

### 15.3 protocol-aware analysis

对应：

- subgroup results from `evaluate_router_results.py`

### 15.4 threshold model 可视化

对应：

- `run_threshold_scan.py`
- `make_router_tables_figures.py`

### 15.5 消融实验

对应：

- `run_feature_ablation.py`

---

## 16. 最终建议：先写代码，不要先扩论文正文

在这份开发清单下，最合理的下一步不是继续空写方法节，而是：

1. 先把 `export_query_eval.py` 和 `build_gain_labels.py` 跑通；
2. 看 label 分布是否符合当前论文分析；
3. 再推进 router feature / train / routing。

因为只要前两步一跑，很多问题会立刻变清楚：

- gain label 是否真的可学；
- Oracle 空间是否足够；
- `head_has_img` 是否真的是主要 gain-positive 区域；
- `tail_no_img` 是否仍然是 structural-dominant。

这些信息会直接决定后续论文写法是否需要收缩或加强。

---

## 17. 一句话总结本轮开发目标

本轮不是去“发明一个更复杂的多模态模型”，而是把当前论文已经得到的 gain-boundary 结论转化为一个**可执行、可评估、可写入论文的方法闭环**：

> 用 query-level gain-aware hard-threshold router，在 multimodal expected gain 足够高时启用 Gate-only，否则退回 Residual-only，从而实现 protocol-aware selective multimodal activation。

