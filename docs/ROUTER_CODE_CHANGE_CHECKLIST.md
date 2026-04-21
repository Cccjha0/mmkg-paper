# Router 代码改动清单（按文件逐项写到函数级别）

## 0. 文档目的

本文档将当前仓库中的 router 修复工作落成一份可直接执行的代码改动清单，目标是把现有体系拆分为两条互不混淆的结果线：

- `clean`：主文可使用的、query-time 合法的 deployable router
- `posthoc`：附录或分析使用的 target-aware / confidence-aware selector

核心原则：

1. `clean` 与 `posthoc` 必须在代码、表格、模型目录、输出文件名上彻底分离。
2. 主文只使用 `clean` 结果。
3. `posthoc` 结果保留，但明确降级为 analysis-only / upper-bound-style selector。
4. 不允许再出现同名表格混合不同 feature family 或不同 routing legality 的情况。

---

## 1. 全局命名与常量约定

建议新增两个全局字符串常量：

```python
ROUTER_MODE_CLEAN = "clean"
ROUTER_MODE_POSTHOC = "posthoc"
```

并在所有脚本输出中显式记录：

- `router_mode`
- `feature_set`
- `is_query_time_legal`

推荐统一目录结构：

```text
outputs/router/
├─ features/
│  ├─ router_train_dev_clean_delta_0.00.csv
│  ├─ router_train_dev_clean_delta_0.01.csv
│  ├─ router_train_dev_clean_delta_0.02.csv
│  ├─ router_test_clean_features.csv
│  ├─ router_feature_summary_clean.json
│  ├─ router_train_dev_posthoc_delta_0.00.csv
│  ├─ router_train_dev_posthoc_delta_0.01.csv
│  ├─ router_train_dev_posthoc_delta_0.02.csv
│  ├─ router_test_posthoc_features.csv
│  └─ router_feature_summary_posthoc.json
├─ models/
│  ├─ clean/
│  └─ posthoc/
├─ eval/
│  ├─ clean/
│  └─ posthoc/
└─ figures/
   ├─ clean/
   └─ posthoc/
```

---

## 2. `router/schemas.py`

当前问题：`RouterFeatureRecord` 同时包含 clean 与 posthoc 特征，导致 target-aware 字段和 correct-score 字段进入同一份 router feature schema。

### 2.1 保留 `QueryEvalRecord`

`QueryEvalRecord` 暂时不改。原因：

- query-level evaluator 输出本来就允许保留 `target_entity_id`
- 它属于评估导出表，而不是 deployable router 的输入表

### 2.2 拆分 `RouterFeatureRecord`

新增两个 dataclass。

#### `CleanRouterFeatureRecord`

建议字段：

```python
@dataclass(frozen=True)
class CleanRouterFeatureRecord:
    query_id: str
    split: str
    seed: int
    direction: str
    relation_id: int
    relation_name: str
    head_id: int
    tail_id: int

    observed_entity_id: int
    observed_has_img: int
    observed_text_img_cosine: float
    observed_img_missing_replaced: int

    relation_gain_prior: float
    relation_fusion_win_rate: float
    relation_support: int
    relation_is_visual_prior: int

    label_gain: int | None = None
    delta_threshold: float | None = None
```

#### `PosthocRouterFeatureRecord`

把当前旧 `RouterFeatureRecord` 的字段整体迁移到这里，不改语义，只改名字。

### 2.3 新增 header 常量

新增：

- `CLEAN_ROUTER_FEATURE_HEADER`
- `POSTHOC_ROUTER_FEATURE_HEADER`

### 2.4 验收标准

- clean feature csv 只使用 `CleanRouterFeatureRecord`
- posthoc feature csv 只使用 `PosthocRouterFeatureRecord`
- clean 表头中不再出现 `target_*`、`fusion_correct_score`、`struct_correct_score`

---

## 3. `router/feature_utils.py`

当前问题：`build_feature_rows()` 直接使用 `target_entity_id` 构造 `text_img_cosine` / `img_is_missing_replaced`，并且把 `fusion_correct_score`、`struct_correct_score`、`target_has_img`、`target_regime` 一起写入 router feature 表，导致主线结果不合法。

### 3.1 保留的函数

以下函数可以原样保留：

- `load_cache_bundle`
- `load_run_config`
- `infer_cache_dir`
- `load_relation_prior_map`

### 3.2 新增 observed-side helper

新增三个函数。

#### `get_observed_entity_id(row: dict) -> int`

建议规则：

- 若 `direction == "tail"`，query 为 `(h, r, ?)`，observed entity = `head_id`
- 若 `direction == "head"`，query 为 `(?, r, t)`，observed entity = `tail_id`

不要依赖 `target_position`，直接依赖 `direction`，因为这更符合 query-time 语义。

#### `cosine_for_observed_entity(cache_bundle: dict, entity_id: int) -> float`

基于当前 `cosine_for_entity()` 改名并复用逻辑。

#### `missing_replaced_flag_for_observed_entity(cache_bundle: dict, entity_id: int) -> int`

基于当前 `missing_replaced_flag()` 改名并复用逻辑。

### 3.3 拆分 `build_feature_rows`

将当前函数拆成两个。

#### A. `build_clean_feature_rows(...)`

输入仍可保持：

- `gate_rows`
- `residual_rows`
- `relation_prior_map`
- `cache_bundle`
- `label_by_query_id=None`

但注意：

**该函数中严禁使用以下字段作为特征来源：**

- `target_entity_id`
- `target_has_img`
- `target_regime`
- `correct_score`
- `score_margin`
- `rr_fusion`
- `rr_struct`
- `rank_fusion`
- `rank_struct`

它只负责构造 clean router 的训练/测试输入表。

建议内部主逻辑：

1. 从 `direction` 推导 `observed_entity_id`
2. 从 `cache_bundle` 计算：
   - `observed_has_img`
   - `observed_text_img_cosine`
   - `observed_img_missing_replaced`
3. 从 `relation_prior_map` 读取：
   - `relation_gain_prior`
   - `relation_fusion_win_rate`
   - `relation_support`
   - `relation_is_visual_prior`
4. 若给定 `label_by_query_id`，则只追加：
   - `label_gain`
   - `delta_threshold`

#### B. `build_posthoc_feature_rows(...)`

把当前 `build_feature_rows()` 的旧逻辑整体搬到这里，保持兼容：

- `target_has_img`
- `target_regime`
- `text_img_cosine`
- `img_is_missing_replaced`
- `fusion_margin`
- `struct_margin`
- `fusion_correct_score`
- `struct_correct_score`
- `delta_margin`
- `rr_fusion`
- `rr_struct`
- `rank_fusion`
- `rank_struct`

### 3.4 拆分 summary 函数

当前 `summarize_feature_rows()` 会把 `fusion_correct_score`、`struct_correct_score`、`rr_fusion` 等字段一起统计，clean 模式下不再适用。

新增两个 summary 函数：

- `summarize_clean_feature_rows(...)`
- `summarize_posthoc_feature_rows(...)`

#### `summarize_clean_feature_rows(...)` 建议 numeric columns

- `relation_gain_prior`
- `relation_fusion_win_rate`
- `relation_support`
- `observed_text_img_cosine`

#### `summarize_posthoc_feature_rows(...)`

保留当前旧版全部 numeric columns。

### 3.5 验收标准

- clean summary 中不再出现 `target_*`, `correct_score`, `rr_*`, `rank_*`
- posthoc summary 保留旧版字段，便于附录复用

---

## 4. `scripts/build_router_features.py`

当前问题：无论 train/test 都统一走 `build_feature_rows()`，因此 clean 与 posthoc 结果混在同一套命名和输出下。

### 4.1 修改 `parse_args()`

新增参数：

- `--router-mode`，取值 `clean | posthoc`
- `--out-prefix`，可选，若不提供则按 mode 自动命名

### 4.2 修改 `main()`

根据 `args.router_mode` 选择：

- `build_clean_feature_rows`
- `build_posthoc_feature_rows`

并根据 mode 选择：

- `CLEAN_ROUTER_FEATURE_HEADER`
- `POSTHOC_ROUTER_FEATURE_HEADER`

### 4.3 输出文件名彻底分开

不要再写统一文件名：

- `router_train_dev_delta_*.csv`
- `router_test_features.csv`

改为：

#### clean

- `outputs/router/features/router_train_dev_clean_delta_0.00.csv`
- `outputs/router/features/router_train_dev_clean_delta_0.01.csv`
- `outputs/router/features/router_train_dev_clean_delta_0.02.csv`
- `outputs/router/features/router_test_clean_features.csv`
- `outputs/router/features/router_feature_summary_clean.json`

#### posthoc

- `outputs/router/features/router_train_dev_posthoc_delta_0.00.csv`
- `outputs/router/features/router_train_dev_posthoc_delta_0.01.csv`
- `outputs/router/features/router_train_dev_posthoc_delta_0.02.csv`
- `outputs/router/features/router_test_posthoc_features.csv`
- `outputs/router/features/router_feature_summary_posthoc.json`

### 4.4 summary 增加元信息

建议在 json 中新增：

- `router_mode`
- `feature_family`
- `is_query_time_legal`

### 4.5 验收标准

- clean / posthoc 两套表不会相互覆盖
- clean 表头不含 target-aware 字段
- posthoc 表头与旧版兼容

---

## 5. `router/router_models.py`

当前问题：`FEATURE_SETS` 只有一套，F1–F4/FULL 全部偏向 target-aware / posthoc selector，无法支持合法的 clean routing 主线。

### 5.1 拆分 `FEATURE_SETS`

改成两套：

- `CLEAN_FEATURE_SETS`
- `POSTHOC_FEATURE_SETS`

### 5.2 clean feature sets 建议

```python
CLEAN_FEATURE_SETS = {
    "C1": ["direction"],
    "C2": [
        "direction",
        "relation_gain_prior",
        "relation_fusion_win_rate",
        "relation_support",
        "relation_is_visual_prior",
    ],
    "C3": [
        "direction",
        "relation_gain_prior",
        "relation_fusion_win_rate",
        "relation_support",
        "relation_is_visual_prior",
        "observed_has_img",
        "observed_text_img_cosine",
        "observed_img_missing_replaced",
    ],
    "C4": [
        "direction",
        "relation_id",
        "relation_gain_prior",
        "relation_fusion_win_rate",
        "relation_support",
        "relation_is_visual_prior",
        "observed_has_img",
        "observed_text_img_cosine",
        "observed_img_missing_replaced",
    ],
}
```

### 5.3 posthoc feature sets 建议

将现有逻辑整体迁移：

```python
POSTHOC_FEATURE_SETS = {
    "PH1": ["target_has_img"],
    "PH2": ["target_has_img", "direction", "relation_gain_prior"],
    "PH3": ["target_has_img", "direction", "relation_gain_prior", "text_img_cosine", "img_is_missing_replaced"],
    "PH4": [
        "target_has_img",
        "direction",
        "relation_gain_prior",
        "text_img_cosine",
        "img_is_missing_replaced",
        "fusion_margin",
        "struct_margin",
        "delta_margin",
    ],
    "PH_FULL": [
        "direction",
        "target_has_img",
        "target_regime",
        "relation_id",
        "relation_gain_prior",
        "relation_fusion_win_rate",
        "relation_support",
        "relation_is_visual_prior",
        "text_img_cosine",
        "img_is_missing_replaced",
        "fusion_margin",
        "struct_margin",
        "fusion_correct_score",
        "struct_correct_score",
        "delta_margin",
    ],
}
```

### 5.4 修改 `cast_feature_value()`

需要支持 clean 模式下的新字段：

- `observed_has_img`
- `observed_img_missing_replaced`
- `observed_text_img_cosine`

### 5.5 新增 `get_feature_sets(router_mode: str)`

统一返回：

- clean → `CLEAN_FEATURE_SETS`
- posthoc → `POSTHOC_FEATURE_SETS`

避免 train/eval 脚本各自重复写条件判断。

### 5.6 修改 rule-based router

当前 `RuleBasedRouter` 规则为：

```text
(target_has_img == 1) AND (relation_gain_prior > gamma)
```

这对 clean 主线不合法。

建议改成两类：

#### `CleanRuleBasedRouter`

```text
(observed_has_img == 1) AND (relation_gain_prior > gamma)
```

#### `PosthocRuleBasedRouter`

保留当前旧规则，用于附录对照。

### 5.7 验收标准

- clean 模型不能引用 `target_*`
- posthoc 模型完整复现旧逻辑
- rule-based baseline 同样分 clean / posthoc 两条线

---

## 6. `scripts/train_router.py`

当前问题：脚本结构可复用，但它默认只存在一套 `FEATURE_SETS`，且模型输出目录和 summary 文件无法区分 clean/posthoc。

### 6.1 修改 `parse_args()`

新增参数：

- `--router-mode clean|posthoc`

并让 `--feature-set` 的合法取值根据 `router_mode` 动态校验。

### 6.2 修改 `contract_mode()`

当前流程保留：

- 读 train table
- normalize labels
- 训练 model
- 输出 `model.pkl`
- 输出 `feature_columns.json`
- 输出 `train_summary.json`

但需要新增元信息：

#### 在 `train_summary.json` 中新增

- `router_mode`
- `is_query_time_legal`
- `feature_family`

其中：

- clean → `is_query_time_legal = true`
- posthoc → `is_query_time_legal = false`

### 6.3 修改模型目录命名

不要再平铺到同一层。

建议改成：

- `outputs/router/models/clean/logistic_delta_0.01_C2/`
- `outputs/router/models/clean/xgb_delta_0.01_C4/`
- `outputs/router/models/posthoc/xgb_delta_0.01_PH_FULL/`

### 6.4 处理 `legacy_mode()`

建议：

- 保留，但标记为 deprecated
- 在函数开头打印 warning
- 主线不再依赖 legacy mode

### 6.5 验收标准

- 任一模型目录都能一眼看出：clean 还是 posthoc
- 不再出现“xgb delta=0.01 但不知道是 C4、PH4 还是 FULL”的情况

---

## 7. `router/routing_utils.py`

当前问题：`select_expert_row()` 假设 `feature_row` 内部直接带有 `target_regime`，这会让 clean feature row 继续混入 evaluation-only 信息。

### 7.1 修改 `select_expert_row()` 签名

当前签名：

```python
select_expert_row(feature_row, use_fusion, selected_by=None)
```

建议改成：

```python
select_expert_row(
    route_row: dict,
    eval_meta_row: dict,
    use_fusion: int,
    selected_by: str | None = None,
) -> dict
```

#### 其中：

- `route_row`：router 输入相关字段 + `router_prob`
- `eval_meta_row`：
  - `target_regime`
  - `rank_gate`
  - `rr_gate`
  - `rank_residual`
  - `rr_residual`

### 7.2 新增/推荐函数

#### `compute_eval_summary_with_ci(rows: list[dict]) -> dict`

即使第一版只按 seed 输出 mean/std，也建议预留该函数，方便后续加 bootstrap。

### 7.3 保留但边界化 `compute_gain_precision()`

保留此函数，但在注释与调用侧明确：

- 它是 evaluation-side diagnostic metric
- 不是 router 输入特征
- clean 主文中只把它当辅助解释指标

### 7.4 验收标准

- clean routing 与 evaluation metadata 明确分离
- `target_regime` 不再被误认为 router 输入字段

---

## 8. `scripts/eval_router_threshold_scan.py`

当前问题：脚本主逻辑可用，但输出文件命名与字段不够自描述，容易导致 clean/posthoc 混线。当前仓库里 xgb threshold scan 与 main results table 不一致，很可能就与这类命名和读写约束不严有关。

### 8.1 修改 `SCAN_COLUMNS`

建议至少新增：

- `router_mode`
- `feature_set`
- `is_query_time_legal`
- `model`
- `delta`

### 8.2 修改 `infer_model_stub(summary)`

不要只返回：

- `model_type`
- `delta_tag`

改为同时返回：

- `model_type`
- `delta_tag`
- `router_mode`
- `feature_set`
- `is_query_time_legal`

### 8.3 修改输出文件名

不要再输出：

- `threshold_scan_xgb_delta_0.01.csv`

改为：

- `threshold_scan_clean_xgb_delta_0.01_C4.csv`
- `threshold_scan_posthoc_xgb_delta_0.01_PH4.csv`

### 8.4 可选：新增 subgroup threshold scan

建议额外输出一个 subgroup-level threshold scan csv，便于主文直接画 clean Figure 2。

### 8.5 验收标准

- scan csv 文件名本身就能看出 legality 与 feature family
- clean / posthoc scan 永不覆盖

---

## 9. `scripts/evaluate_router_results.py`

当前问题：主表与 subgroup 表缺少 `router_mode`、`feature_set`、`is_query_time_legal` 等关键区分字段，容易把 clean 与 posthoc 结果拼到同一个结果表里。

### 9.1 修改 `MAIN_HEADER`

新增：

- `router_mode`
- `feature_set`
- `is_query_time_legal`
- `source_eval_json`

### 9.2 修改 `SUBGROUP_HEADER`

同步新增以上字段。

### 9.3 修改 `load_eval_jsons()`

建议不要再在一个目录下全扫：

```python
glob("router_eval_*_delta_*_tau_*.json")
```

而是按目录分两条：

- `outputs/router/eval/clean/`
- `outputs/router/eval/posthoc/`

### 9.4 修改排序逻辑

当前排序：

- `(delta, model, tau)`

建议改为：

- `(router_mode, feature_set, delta, model, tau)`

### 9.5 验收标准

- 主表一眼可以区分 clean 与 posthoc
- 子群表同样可区分
- 不再需要人工记忆“某个 xgb 是哪条线的结果”

---

## 10. `scripts/make_router_tables_figures.py`

当前问题：从现有产物冲突来看，这个脚本高度可能在拼表或画图时混用了不同 feature family 或不同 routing legality 的输入文件，因此必须重构。

### 10.1 建议拆脚本

建议拆成：

- `scripts/make_clean_router_tables_figures.py`
- `scripts/make_posthoc_router_tables_figures.py`

如果暂时不想拆两个文件，至少增加：

- `--router-mode clean|posthoc`

### 10.2 主文只读取 clean

clean 只生成：

- `main_results_table_clean.csv/md`
- `subgroup_results_table_clean.csv/md`
- `feature_ablation_clean.csv/md`
- `threshold_scan_clean_*`
- `router_feature_importance_clean_*`

### 10.3 posthoc 单独生成

posthoc 只生成：

- `main_results_table_posthoc.csv/md`
- `subgroup_results_table_posthoc.csv/md`
- `feature_ablation_posthoc.csv/md`
- `threshold_scan_posthoc_*`
- `router_feature_importance_posthoc_*`

### 10.4 Figure 2 只读 clean scan

论文主图必须只从 clean threshold scan 生成。

### 10.5 验收标准

- 主文图表与附录图表目录完全分离
- xgb clean threshold scan 与 clean main table 数值一致

---

## 11. `scripts/verify_router_result_consistency.py`

当前问题：一致性检查默认只有一条 learned line，并且硬编码假设 `best_learned_is_xgb_delta_0.01_tau_0.7`，这在 clean/posthoc 双线结构下已经不适用。

### 11.1 拆检查项

新增两组检查：

#### clean checks

- `all_clean_required_files_exist`
- `clean_main_table_has_expected_models`
- `clean_feature_ablation_has_C1_to_C4`
- `clean_scan_best_model_consistent_with_clean_main_table`

#### posthoc checks

- `all_posthoc_required_files_exist`
- `posthoc_feature_ablation_has_PH1_to_PH4`
- `posthoc_best_ge_clean_best`

### 11.2 删除硬编码 best learned 假设

去掉：

- `best_learned_is_xgb_delta_0.01_tau_0.7`

因为 clean 线重跑后最优点很可能变化。

### 11.3 新增 legality 回归检查

新增一个非常重要的自动检查：

- `no_illegal_features_in_clean_feature_columns`

实现思路：

1. 读取 clean 模型目录下的 `feature_columns.json`
2. 检查不包含：
   - `target_has_img`
   - `target_regime`
   - `fusion_correct_score`
   - `struct_correct_score`
   - `fusion_margin`
   - `struct_margin`
   - `delta_margin`

### 11.4 验收标准

- clean consistency checks 能自动发现 legality 回归
- posthoc consistency checks 能维持旧分析线可复现

---

## 12. `scripts/build_relation_priors.py` 与 `router/prior_utils.py`

这两处总体问题不大，因为当前逻辑本来就是 dev-only prior 构建链路，可直接保留为 clean 主线的一部分。

### 12.1 保留 `compute_relation_gain_stats()`

现有逻辑可保留，不必重写。

### 12.2 建议新增 shrinkage prior

在 `router/prior_utils.py` 中新增：

```python
def shrink_mean_delta_rr(mean_delta_rr: float, n_queries: int, k: float = 20.0) -> float:
    return float(mean_delta_rr) * float(n_queries / (n_queries + k))
```

然后可扩展：

```python
def compute_relation_gain_stats(..., use_shrinkage: bool = False, shrink_k: float = 20.0)
```

### 12.3 增加输出字段

建议 relation prior csv 额外增加：

- `mean_delta_rr_shrunk`

这样 clean router 在 relation support 较低时更稳。

### 12.4 验收标准

- clean 主线默认仍使用 dev-only prior
- 若启用 shrinkage，也必须只基于 dev 统计

---

## 13. 建议新增脚本

### 13.1 `scripts/run_router_predictions.py`

职责：

- 给任一 clean/posthoc router 在 test 上产出统一 prediction file
- 供 main eval / subgroup eval / threshold scan 共用

作用：减少多脚本重复生成概率导致的不一致。

### 13.2 `scripts/run_router_significance.py`

职责：

- 输入 routed rows
- 输出：
  - seed-wise MRR
  - mean ± std
  - paired bootstrap CI vs residual-only
  - paired bootstrap CI vs clean rule-based

这不是第一优先级必须项，但由于当前仓库已经有三 seed query-level outcome，后续补统计显著性时非常值得加。

---

## 14. 推荐执行顺序

### 第一阶段：先完成 clean/posthoc 分家

1. 改 `router/schemas.py`
2. 改 `router/feature_utils.py`
3. 改 `scripts/build_router_features.py`
4. 改 `router/router_models.py`

### 第二阶段：让训练与评估链路能识别双线结构

5. 改 `scripts/train_router.py`
6. 改 `router/routing_utils.py`
7. 改 `scripts/eval_router_threshold_scan.py`
8. 改 `scripts/evaluate_router_results.py`

### 第三阶段：重建 paper-facing outputs

9. 改 `scripts/make_router_tables_figures.py`
10. 改 `scripts/verify_router_result_consistency.py`
11. 重跑 clean C1–C4
12. 再把旧结果归到 posthoc

---

## 15. 最先该动的 3 个文件

如果不想一次改太多，最优先的三个文件是：

1. `router/feature_utils.py`
2. `router/router_models.py`
3. `scripts/build_router_features.py`

原因：

- 只要这三处改完，clean 与 posthoc 在 feature 层面就真正分家了
- 后续 train/eval/table 只是顺着新 schema 与 feature family 往下接

---

## 16. 一句话总结

本轮代码修复的本质不是“继续把当前 router 打磨得更强”，而是：

> 先把 deployable clean router 与 analysis-only posthoc selector 从 schema、feature、model、eval、tables 全链路上彻底拆开，再分别生成主文与附录所需结果。

只有这一步完成后，论文主线的 routing claim 才会重新站稳。
