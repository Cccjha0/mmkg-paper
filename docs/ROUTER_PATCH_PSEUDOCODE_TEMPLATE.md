# Router Patch 伪代码模板（按文件与函数展开）

## 0. 文档目的

本文档是 `docs/ROUTER_CODE_CHANGE_CHECKLIST.md` 的下一层执行稿。

上一份文档回答的是：

- 哪些文件要改
- 为什么要改
- 改完后的 clean / posthoc 双线结构应该是什么样

本文件回答的是：

- 每个关键文件应该怎样改到函数级别
- 新增哪些函数
- 函数签名建议长什么样
- 旧逻辑迁移到哪里
- 主流程脚本如何分流 clean 与 posthoc

注意：

1. 本文档是 **patch 伪代码模板**，不是最终可直接运行的代码。
2. 变量名、导入位置、异常处理风格可以按你当前 repo 风格微调。
3. 目标不是一次性重构所有脚本，而是让 clean 与 posthoc 两条线真正分家，并保证主文只依赖 clean 线。

---

## 1. `router/schemas.py`

### 1.1 当前问题

当前 `RouterFeatureRecord` 混合了：

- clean query-time 可用特征
- target-aware 特征
- correct-score / confidence 特征
- routing evaluation 中间变量

这会让后续任何 train/eval/table 脚本默认读到污染后的 schema。

### 1.2 建议 patch 结构

#### 保留 `QueryEvalRecord`

```python
# keep as is
@dataclass(frozen=True)
class QueryEvalRecord:
    ...
```

#### 替换 `RouterFeatureRecord`

把当前单一 `RouterFeatureRecord` 删除，改为两个 dataclass。

```python
from dataclasses import asdict, dataclass


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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PosthocRouterFeatureRecord:
    query_id: str
    split: str
    seed: int
    direction: str
    relation_id: int
    relation_name: str
    head_id: int
    tail_id: int
    target_entity_id: int
    target_position: str
    target_has_img: int
    target_regime: str

    relation_gain_prior: float
    relation_fusion_win_rate: float
    relation_support: int
    relation_is_visual_prior: int

    text_img_cosine: float
    img_is_missing_replaced: int
    fusion_margin: float
    struct_margin: float
    fusion_correct_score: float
    struct_correct_score: float
    delta_margin: float

    rr_fusion: float
    rr_struct: float
    rank_fusion: int
    rank_struct: int

    label_gain: int | None = None
    delta_threshold: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)
```

#### 新增 header 常量

```python
QUERY_EVAL_HEADER = list(QueryEvalRecord.__dataclass_fields__.keys())
CLEAN_ROUTER_FEATURE_HEADER = list(CleanRouterFeatureRecord.__dataclass_fields__.keys())
POSTHOC_ROUTER_FEATURE_HEADER = list(PosthocRouterFeatureRecord.__dataclass_fields__.keys())
```

### 1.3 最低验收点

- clean feature csv 表头中不再出现 `target_*`
- posthoc feature csv 保留旧版字段全集

---

## 2. `router/feature_utils.py`

### 2.1 当前问题

当前 `build_feature_rows()` 有三个问题：

1. 它统一生成一份 feature row，clean/posthoc 不分。
2. 它直接依赖 `target_entity_id` 计算 `text_img_cosine` 与 missing flag。
3. 它把 `fusion_correct_score`、`struct_correct_score`、`rr_fusion` 等后验字段也一并写入 router 输入表。

### 2.2 patch 目标

- 保留基础 IO / cache / prior helper
- 把 observed-side feature 构造抽出来
- 把 feature builder 拆成 clean 与 posthoc 两个版本
- 把 summary 也拆成两套

### 2.3 建议 patch 伪代码

#### 保留部分

```python
def load_cache_bundle(cache_dir: str | Path) -> dict:
    ...


def load_run_config(run_dir: str | Path) -> dict:
    ...


def infer_cache_dir(cache_dir: str | None, run_dir: str | None) -> str:
    ...


def load_relation_prior_map(rows: list[dict]) -> dict[int, dict]:
    ...
```

#### 新增：observed-side helper

```python
def get_observed_entity_id(row: dict) -> int:
    direction = str(row["direction"])
    if direction == "tail":
        # query = (h, r, ?)
        return int(row["head_id"])
    if direction == "head":
        # query = (?, r, t)
        return int(row["tail_id"])
    raise ValueError(f"Unknown direction: {direction}")
```

```python
def cosine_for_observed_entity(cache_bundle: dict, entity_id: int) -> float:
    text_vec = cache_bundle["text_feat"][entity_id]
    img_vec = cache_bundle["img_feat"][entity_id]
    if not bool(cache_bundle["has_img"][entity_id].item()):
        return 0.0
    text_norm = float(text_vec.norm().item())
    img_norm = float(img_vec.norm().item())
    if text_norm <= 0.0 or img_norm <= 0.0:
        return 0.0
    return float(F.cosine_similarity(text_vec.unsqueeze(0), img_vec.unsqueeze(0), dim=1).item())
```

```python
def missing_replaced_flag_for_observed_entity(cache_bundle: dict, entity_id: int) -> int:
    return int(not bool(cache_bundle["has_img"][entity_id].item()))
```

```python
def observed_has_img_flag(cache_bundle: dict, entity_id: int) -> int:
    return int(bool(cache_bundle["has_img"][entity_id].item()))
```

#### 保留：query 对齐逻辑

```python
def _merge_query_eval_pair(gate_rows: list[dict], residual_rows: list[dict]) -> list[tuple[dict, dict]]:
    ...
```

#### 新增：clean builder

```python
from router.schemas import CleanRouterFeatureRecord, PosthocRouterFeatureRecord


def build_clean_feature_rows(
    gate_rows: list[dict],
    residual_rows: list[dict],
    relation_prior_map: dict[int, dict],
    cache_bundle: dict,
    label_by_query_id: dict[str, dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []

    for gate, residual in _merge_query_eval_pair(gate_rows, residual_rows):
        relation_id = int(gate["relation_id"])
        observed_entity_id = get_observed_entity_id(gate)
        prior = relation_prior_map.get(
            relation_id,
            {
                "relation_gain_prior": 0.0,
                "relation_fusion_win_rate": 0.0,
                "relation_support": 0,
                "relation_is_visual_prior": 0,
            },
        )
        label_row = label_by_query_id.get(gate["query_id"]) if label_by_query_id is not None else None

        record = CleanRouterFeatureRecord(
            query_id=str(gate["query_id"]),
            split=str(gate["split"]),
            seed=int(gate["seed"]),
            direction=str(gate["direction"]),
            relation_id=relation_id,
            relation_name=str(gate.get("relation_name", f"rel_{relation_id:04d}")),
            head_id=int(gate["head_id"]),
            tail_id=int(gate["tail_id"]),
            observed_entity_id=observed_entity_id,
            observed_has_img=observed_has_img_flag(cache_bundle, observed_entity_id),
            observed_text_img_cosine=cosine_for_observed_entity(cache_bundle, observed_entity_id),
            observed_img_missing_replaced=missing_replaced_flag_for_observed_entity(cache_bundle, observed_entity_id),
            relation_gain_prior=float(prior["relation_gain_prior"]),
            relation_fusion_win_rate=float(prior["relation_fusion_win_rate"]),
            relation_support=int(prior["relation_support"]),
            relation_is_visual_prior=int(prior["relation_is_visual_prior"]),
            label_gain=int(label_row["label_gain"]) if label_row is not None else None,
            delta_threshold=float(label_row["delta_threshold"]) if label_row is not None else None,
        )
        rows.append(record.to_dict())

    return rows
```

#### 新增：posthoc builder

```python
def build_posthoc_feature_rows(
    gate_rows: list[dict],
    residual_rows: list[dict],
    relation_prior_map: dict[int, dict],
    cache_bundle: dict,
    label_by_query_id: dict[str, dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []

    for gate, residual in _merge_query_eval_pair(gate_rows, residual_rows):
        relation_id = int(gate["relation_id"])
        target_entity_id = int(gate["target_entity_id"])
        prior = relation_prior_map.get(...)
        label_row = ...

        record = PosthocRouterFeatureRecord(
            query_id=str(gate["query_id"]),
            split=str(gate["split"]),
            seed=int(gate["seed"]),
            direction=str(gate["direction"]),
            relation_id=relation_id,
            relation_name=str(gate.get("relation_name", f"rel_{relation_id:04d}")),
            head_id=int(gate["head_id"]),
            tail_id=int(gate["tail_id"]),
            target_entity_id=target_entity_id,
            target_position=str(gate["target_position"]),
            target_has_img=int(gate["target_has_img"]),
            target_regime=str(gate["target_regime"]),
            relation_gain_prior=float(prior["relation_gain_prior"]),
            relation_fusion_win_rate=float(prior["relation_fusion_win_rate"]),
            relation_support=int(prior["relation_support"]),
            relation_is_visual_prior=int(prior["relation_is_visual_prior"]),
            text_img_cosine=cosine_for_entity(cache_bundle, target_entity_id),
            img_is_missing_replaced=missing_replaced_flag(cache_bundle, target_entity_id),
            fusion_margin=float(gate["score_margin"]),
            struct_margin=float(residual["score_margin"]),
            fusion_correct_score=float(gate["correct_score"]),
            struct_correct_score=float(residual["correct_score"]),
            delta_margin=float(gate["score_margin"]) - float(residual["score_margin"]),
            rr_fusion=float(gate["rr"]),
            rr_struct=float(residual["rr"]),
            rank_fusion=int(gate["rank"]),
            rank_struct=int(residual["rank"]),
            label_gain=int(label_row["label_gain"]) if label_row is not None else None,
            delta_threshold=float(label_row["delta_threshold"]) if label_row is not None else None,
        )
        rows.append(record.to_dict())

    return rows
```

#### 新增：clean summary

```python
def summarize_clean_feature_rows(train_rows_by_delta: dict[str, list[dict]], test_rows: list[dict]) -> dict:
    numeric_cols = [
        "relation_gain_prior",
        "relation_fusion_win_rate",
        "relation_support",
        "observed_text_img_cosine",
    ]
    return _summarize_feature_rows_impl(train_rows_by_delta, test_rows, numeric_cols)
```

#### 新增：posthoc summary

```python
def summarize_posthoc_feature_rows(train_rows_by_delta: dict[str, list[dict]], test_rows: list[dict]) -> dict:
    numeric_cols = [
        "relation_gain_prior",
        "relation_fusion_win_rate",
        "relation_support",
        "text_img_cosine",
        "fusion_margin",
        "struct_margin",
        "fusion_correct_score",
        "struct_correct_score",
        "delta_margin",
        "rr_fusion",
        "rr_struct",
        "rank_fusion",
        "rank_struct",
    ]
    return _summarize_feature_rows_impl(train_rows_by_delta, test_rows, numeric_cols)
```

#### 可选：抽出通用 summary impl

```python
def _summarize_feature_rows_impl(train_rows_by_delta, test_rows, numeric_cols) -> dict:
    ...
```

### 2.4 最低验收点

- clean feature summary 里不再出现 `fusion_correct_score`、`rr_fusion`、`rank_fusion`
- posthoc feature summary 保留旧版全部统计

---

## 3. `scripts/build_router_features.py`

### 3.1 当前问题

当前脚本统一调用 `build_feature_rows()`，导致：

- clean/posthoc 共用同一份 feature 表
- train/test 文件名也共用同一套命名
- 后续训练脚本无法只读 clean 表

### 3.2 patch 目标

- 增加 `router_mode`
- 分流 clean/posthoc feature builder
- 分流输出文件命名与 summary json

### 3.3 建议 patch 伪代码

#### 修改 imports

```python
from router.feature_utils import (
    build_clean_feature_rows,
    build_posthoc_feature_rows,
    infer_cache_dir,
    load_cache_bundle,
    load_relation_prior_map,
    summarize_clean_feature_rows,
    summarize_posthoc_feature_rows,
)
from router.schemas import (
    CLEAN_ROUTER_FEATURE_HEADER,
    POSTHOC_ROUTER_FEATURE_HEADER,
)
```

#### 修改 `parse_args()`

```python
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ...
    ap.add_argument("--router-mode", choices=["clean", "posthoc"], required=True)
    ap.add_argument("--out-prefix", default=None)
    return ap.parse_args()
```

#### 新增 helper：输出命名

```python
def resolve_output_names(router_mode: str, delta_tag: str) -> tuple[str, str]:
    if router_mode == "clean":
        return (
            f"router_train_dev_clean_delta_{delta_tag}.csv",
            "router_test_clean_features.csv",
        )
    if router_mode == "posthoc":
        return (
            f"router_train_dev_posthoc_delta_{delta_tag}.csv",
            "router_test_posthoc_features.csv",
        )
    raise ValueError(f"Unknown router mode: {router_mode}")
```

#### 修改 `main()` 的核心分流

```python
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    cache_dir = infer_cache_dir(args.cache_dir, args.run_dir)
    cache_bundle = load_cache_bundle(cache_dir)
    prior_map = load_relation_prior_map(read_csv(args.prior_csv))

    if args.router_mode == "clean":
        build_rows_fn = build_clean_feature_rows
        summary_fn = summarize_clean_feature_rows
        feature_header = CLEAN_ROUTER_FEATURE_HEADER
        summary_name = "router_feature_summary_clean.json"
    else:
        build_rows_fn = build_posthoc_feature_rows
        summary_fn = summarize_posthoc_feature_rows
        feature_header = POSTHOC_ROUTER_FEATURE_HEADER
        summary_name = "router_feature_summary_posthoc.json"

    train_rows_by_delta: dict[str, list[dict]] = {}
    for delta_tag in args.deltas:
        all_rows = []
        ...
        rows = build_rows_fn(
            gate_rows,
            residual_rows,
            prior_map,
            cache_bundle,
            label_by_query_id=label_map,
        )
        all_rows.extend(rows)

        train_name, _ = resolve_output_names(args.router_mode, delta_tag)
        out_path = out_dir / train_name
        write_csv(out_path, all_rows, feature_header)
        train_rows_by_delta[delta_tag] = all_rows

    test_rows = []
    ...
    rows = build_rows_fn(
        gate_rows,
        residual_rows,
        prior_map,
        cache_bundle,
        label_by_query_id=None,
    )
    test_rows.extend(rows)

    _, test_name = resolve_output_names(args.router_mode, "unused")
    test_out = out_dir / test_name
    write_csv(test_out, test_rows, feature_header)

    summary = summary_fn(train_rows_by_delta, test_rows)
    summary["router_mode"] = args.router_mode
    summary["is_query_time_legal"] = args.router_mode == "clean"
    summary["cache_dir"] = cache_bundle["cache_dir"]
    summary["prior_csv"] = str(Path(args.prior_csv).as_posix())
    summary_path = out_dir / summary_name
    write_json(summary_path, summary)
```

### 3.4 最低验收点

- 生成四套 clean csv/json 与 posthoc csv/json，不再覆盖
- clean 表头中不含 `target_*`

---

## 4. `router/router_models.py`

### 4.1 当前问题

当前 `FEATURE_SETS` 只有一套：

- F1–F4
- FULL

这意味着 clean/posthoc 无法在 model definition 层面分开。

### 4.2 patch 目标

- 将 feature set 显式拆成 clean 与 posthoc
- rule-based router 同样拆成 clean 与 posthoc
- 提供统一 helper 给 train/eval 脚本调用

### 4.3 建议 patch 伪代码

#### 替换 `FEATURE_SETS`

```python
CLEAN_FEATURE_SETS: dict[str, list[str]] = {
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

POSTHOC_FEATURE_SETS: dict[str, list[str]] = {
    "PH1": ["target_has_img"],
    "PH2": ["target_has_img", "direction", "relation_gain_prior"],
    "PH3": [
        "target_has_img",
        "direction",
        "relation_gain_prior",
        "text_img_cosine",
        "img_is_missing_replaced",
    ],
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

#### 新增 helper

```python
def get_feature_sets(router_mode: str) -> dict[str, list[str]]:
    if router_mode == "clean":
        return CLEAN_FEATURE_SETS
    if router_mode == "posthoc":
        return POSTHOC_FEATURE_SETS
    raise ValueError(f"Unknown router mode: {router_mode}")
```

#### 修改 `cast_feature_value()`

```python
def cast_feature_value(key: str, value: str | int | float | None) -> Any:
    if value in ("", None):
        return 0.0

    if key in {"direction"}:
        return str(value)

    if key in {"relation_id"}:
        return f"rel_{int(value)}"

    if key in {
        "target_has_img",
        "img_is_missing_replaced",
        "relation_is_visual_prior",
        "observed_has_img",
        "observed_img_missing_replaced",
    }:
        return int(value)

    return float(value)
```

#### 替换 rule-based router

```python
@dataclass
class CleanRuleBasedRouter:
    gamma: float = 0.0

    def predict_proba_from_rows(self, rows: list[dict]) -> list[float]:
        probs = []
        for row in rows:
            use_fusion = (
                int(row["observed_has_img"]) == 1
                and float(row["relation_gain_prior"]) > float(self.gamma)
            )
            probs.append(1.0 if use_fusion else 0.0)
        return probs
```

```python
@dataclass
class PosthocRuleBasedRouter:
    gamma: float = 0.0

    def predict_proba_from_rows(self, rows: list[dict]) -> list[float]:
        probs = []
        for row in rows:
            use_fusion = (
                int(row["target_has_img"]) == 1
                and float(row["relation_gain_prior"]) > float(self.gamma)
            )
            probs.append(1.0 if use_fusion else 0.0)
        return probs
```

#### 修改训练入口函数签名

```python
def train_logistic_router(
    rows: list[dict],
    feature_set: str,
    router_mode: str,
    random_state: int = 42,
) -> TrainedRouterArtifact:
    feature_names = get_feature_sets(router_mode)[feature_set]
    ...
```

```python
def train_xgb_router(
    rows: list[dict],
    feature_set: str,
    router_mode: str,
    random_state: int = 42,
) -> TrainedRouterArtifact:
    feature_names = get_feature_sets(router_mode)[feature_set]
    ...
```

### 4.4 最低验收点

- clean 模型永远不会引用 `target_has_img`
- posthoc 模型完整保留旧 F1/F2/F3/F4/FULL 逻辑

---

## 5. `scripts/train_router.py`

### 5.1 当前问题

当前训练脚本默认只有一套 feature_set，输出目录和 train summary 也无法区分 clean/posthoc。

### 5.2 patch 目标

- 加 `router_mode`
- 根据 mode 选择 feature set family
- 模型输出目录按 clean/posthoc 分层
- train summary 明确 legality

### 5.3 建议 patch 伪代码

#### 修改 imports

```python
from router.router_models import (
    CLEAN_FEATURE_SETS,
    POSTHOC_FEATURE_SETS,
    CleanRuleBasedRouter,
    PosthocRuleBasedRouter,
    compute_feature_importance_rows,
    get_feature_sets,
    train_logistic_router,
    train_xgb_router,
)
```

#### 修改 `parse_args()`

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(...)
    ...
    parser.add_argument("--router-mode", choices=["clean", "posthoc"], required=True)
    return parser.parse_args()
```

#### 新增 helper

```python
def is_query_time_legal(router_mode: str) -> bool:
    return router_mode == "clean"
```

#### 修改 `train_one_model()`

```python
def train_one_model(
    model_name: str,
    rows: list[dict],
    feature_set: str,
    router_mode: str,
    rule_gamma: float,
    random_state: int,
):
    if model_name == "rule":
        if router_mode == "clean":
            return CleanRuleBasedRouter(gamma=rule_gamma)
        return PosthocRuleBasedRouter(gamma=rule_gamma)

    if model_name == "logistic":
        return train_logistic_router(
            rows,
            feature_set=feature_set,
            router_mode=router_mode,
            random_state=random_state,
        )

    if model_name == "xgb":
        return train_xgb_router(
            rows,
            feature_set=feature_set,
            router_mode=router_mode,
            random_state=random_state,
        )

    raise ValueError(...)
```

#### 修改 `contract_mode()`

```python
def contract_mode(args: argparse.Namespace) -> None:
    ...
    feature_sets = get_feature_sets(args.router_mode)
    if args.feature_set not in feature_sets:
        raise ValueError(...)

    rows = normalize_contract_rows(...)
    artifact = train_one_model(
        model_name=args.model_type,
        rows=rows,
        feature_set=args.feature_set,
        router_mode=args.router_mode,
        rule_gamma=args.rule_gamma,
        random_state=args.random_state,
    )
    ...
    feature_columns = list(feature_sets[args.feature_set])
    ...
    summary = {
        "model_type": args.model_type,
        "router_mode": args.router_mode,
        "is_query_time_legal": is_query_time_legal(args.router_mode),
        "feature_set": args.feature_set,
        "delta": float(delta_str),
        ...
    }
```

#### 建议输出目录命名模板

```python
def resolve_model_out_dir(base_out_dir: Path, router_mode: str, model_type: str, delta_str: str, feature_set: str) -> Path:
    return base_out_dir / router_mode / f"{model_type}_delta_{delta_str}_{feature_set}"
```

### 5.4 最低验收点

- `train_summary.json` 能一眼看出 clean/posthoc
- `feature_columns.json` 与 `router_mode` 一致

---

## 6. `router/routing_utils.py`

### 6.1 当前问题

当前 `select_expert_row()` 直接从 `feature_row` 中取：

- `target_regime`
- `rank_fusion`
- `rr_fusion`
- `rank_struct`
- `rr_struct`

这会让 clean routing 在结构上也继续依赖污染后的 feature row。

### 6.2 patch 目标

- 把 router 输入与 evaluation metadata 分离
- clean row 不再强制包含 `target_regime`

### 6.3 建议 patch 伪代码

#### 修改签名

```python
def select_expert_row(
    route_row: dict,
    eval_meta_row: dict,
    use_fusion: int,
    selected_by: str | None = None,
) -> dict:
    use_fusion = int(use_fusion)

    if use_fusion:
        selected_expert = "gate_only"
        rank_final = int(eval_meta_row["rank_gate"])
        rr_final = float(eval_meta_row["rr_gate"])
    else:
        selected_expert = "residual_only"
        rank_final = int(eval_meta_row["rank_residual"])
        rr_final = float(eval_meta_row["rr_residual"])

    return {
        "query_id": route_row["query_id"],
        "split": route_row["split"],
        "seed": int(route_row["seed"]),
        "direction": route_row["direction"],
        "target_regime": eval_meta_row["target_regime"],
        "relation_id": int(route_row["relation_id"]),
        "selected_by": selected_by or "",
        "router_prob": float(route_row["router_prob"]),
        "threshold": float(route_row["threshold"]),
        "use_fusion": use_fusion,
        "selected_expert": selected_expert,
        "rank_final": rank_final,
        "rr_final": rr_final,
        "rank_gate": int(eval_meta_row["rank_gate"]),
        "rr_gate": float(eval_meta_row["rr_gate"]),
        "rank_residual": int(eval_meta_row["rank_residual"]),
        "rr_residual": float(eval_meta_row["rr_residual"]),
    }
```

#### 可选新增：seed summary with std

```python
def compute_eval_summary_with_std(rows: list[dict]) -> dict:
    summary = compute_eval_summary(rows)
    seed_mrrs = [stats["mrr"] for stats in summary["by_seed"].values()]
    if seed_mrrs:
        mean_mrr = float(sum(seed_mrrs) / len(seed_mrrs))
        var = float(sum((x - mean_mrr) ** 2 for x in seed_mrrs) / len(seed_mrrs))
        summary["seed_mean_mrr"] = mean_mrr
        summary["seed_std_mrr"] = var ** 0.5
    else:
        summary["seed_mean_mrr"] = 0.0
        summary["seed_std_mrr"] = 0.0
    return summary
```

#### 保留 `compute_gain_precision()`，但加注释

```python
def compute_gain_precision(rows: list[dict], delta: float = 0.0) -> float:
    """
    Evaluation-side diagnostic only.
    Not a router input feature.
    """
    ...
```

### 6.4 最低验收点

- clean route row 不包含 `target_regime`
- evaluation metadata 单独来源于 query_eval merge / eval targets

---

## 7. `scripts/eval_router_threshold_scan.py`

### 7.1 当前问题

当前 threshold scan：

- 文件名不区分 clean/posthoc
- 列字段太少
- 容易让不同 feature family 的 scan 覆盖彼此

### 7.2 patch 目标

- scan csv 自带 legality / feature family / model identity
- 输出文件名显式带 clean/posthoc

### 7.3 建议 patch 伪代码

#### 修改 `SCAN_COLUMNS`

```python
SCAN_COLUMNS = [
    "router_mode",
    "feature_set",
    "is_query_time_legal",
    "model",
    "delta",
    "tau",
    "overall_mrr",
    "fusion_coverage",
    "gain_precision",
    "gain_recall",
    "n_selected_fusion",
    "n_total",
]
```

#### 修改 `infer_model_stub(summary)`

```python
def infer_model_stub(summary: dict) -> tuple[str, str, str, str, bool]:
    model_type = str(summary["model_type"])
    delta_value = float(summary["delta"])
    delta_tag = f"{delta_value:.2f}"
    router_mode = str(summary["router_mode"])
    feature_set = str(summary["feature_set"])
    legal = bool(summary.get("is_query_time_legal", router_mode == "clean"))
    return model_type, delta_tag, router_mode, feature_set, legal
```

#### 修改输出文件命名

```python
def resolve_scan_path(out_dir: Path, router_mode: str, model_type: str, delta_tag: str, feature_set: str) -> Path:
    return out_dir / router_mode / f"threshold_scan_{router_mode}_{model_type}_delta_{delta_tag}_{feature_set}.csv"
```

#### 修改 `main()` 写 scan rows

```python
def main() -> None:
    ...
    model_type, delta_tag, router_mode, feature_set, legal = infer_model_stub(summary)
    ...
    for tau in args.taus:
        ...
        scan_rows.append(
            {
                "router_mode": router_mode,
                "feature_set": feature_set,
                "is_query_time_legal": legal,
                "model": model_type,
                "delta": delta_tag,
                "tau": float(tau),
                "overall_mrr": mrr,
                "fusion_coverage": safe_ratio(n_selected, n_total),
                "gain_precision": gain_precision,
                "gain_recall": gain_recall,
                "n_selected_fusion": int(n_selected),
                "n_total": int(n_total),
            }
        )
```

#### 可选新增：subgroup threshold scan

```python
def subgroup_threshold_stats(joined_rows: list[dict], tau: float) -> list[dict]:
    # optional: compute mrr / coverage by target_regime
    ...
```

### 7.4 最低验收点

- clean xgb scan 与 clean main table 使用同一条 feature family
- 文件名中能直接看出 C4 / PH4 / PH_FULL

---

## 8. `scripts/evaluate_router_results.py`

### 8.1 当前问题

当前主表与 subgroup 表列太少：

- 没有 `router_mode`
- 没有 `feature_set`
- 没有 `is_query_time_legal`

容易让 clean/posthoc 结果混在一张表里。

### 8.2 patch 目标

- 在 CSV level 就彻底区分 clean 与 posthoc
- 输出表本身可直接被 manuscript side 引用

### 8.3 建议 patch 伪代码

#### 修改 `MAIN_HEADER`

```python
MAIN_HEADER = [
    "router_mode",
    "feature_set",
    "is_query_time_legal",
    "model",
    "delta",
    "tau",
    "n_queries",
    "mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
    "source_eval_json",
]
```

#### 修改 `SUBGROUP_HEADER`

```python
SUBGROUP_HEADER = [
    "router_mode",
    "feature_set",
    "is_query_time_legal",
    "model",
    "delta",
    "tau",
    "target_regime",
    "n_queries",
    "mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
    "source_eval_json",
]
```

#### 修改 `load_eval_jsons()`

```python
def load_eval_jsons(eval_dir: Path, router_mode: str) -> list[dict]:
    payload = []
    target_dir = eval_dir / router_mode
    for path in sorted(target_dir.glob("router_eval_*_delta_*_tau_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = path.as_posix()
        payload.append(data)
    return payload
```

#### 修改 `main()`

```python
def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)

    payloads = []
    for mode in ["clean", "posthoc"]:
        payloads.extend(load_eval_jsons(eval_dir, mode))

    if not payloads:
        raise RuntimeError(...)

    main_rows = []
    subgroup_rows = []
    for data in payloads:
        overall = data["overall"]
        main_rows.append(
            {
                "router_mode": data["router_mode"],
                "feature_set": data["feature_set"],
                "is_query_time_legal": data["is_query_time_legal"],
                "model": data["model"],
                "delta": data["delta"],
                "tau": data["tau"],
                "n_queries": overall["n_queries"],
                "mrr": overall["mrr"],
                "hits1": overall["hits1"],
                "hits3": overall["hits3"],
                "hits10": overall["hits10"],
                "fusion_coverage": overall["fusion_coverage"],
                "source_eval_json": data["_path"],
            }
        )

        for regime, stats in sorted(data.get("by_regime", {}).items()):
            subgroup_rows.append(
                {
                    "router_mode": data["router_mode"],
                    "feature_set": data["feature_set"],
                    "is_query_time_legal": data["is_query_time_legal"],
                    "model": data["model"],
                    "delta": data["delta"],
                    "tau": data["tau"],
                    "target_regime": regime,
                    "n_queries": stats["n_queries"],
                    "mrr": stats["mrr"],
                    "hits1": stats["hits1"],
                    "hits3": stats["hits3"],
                    "hits10": stats["hits10"],
                    "fusion_coverage": stats["fusion_coverage"],
                    "source_eval_json": data["_path"],
                }
            )
```

#### 修改排序

```python
main_rows.sort(key=lambda row: (row["router_mode"], row["feature_set"], row["delta"], row["model"], float(row["tau"])))
subgroup_rows.sort(key=lambda row: (row["router_mode"], row["feature_set"], row["delta"], row["model"], float(row["tau"]), row["target_regime"]))
```

### 8.4 最低验收点

- clean 与 posthoc 结果在 CSV 层就不可混淆
- 主文只需要过滤 `router_mode == clean`

---

## 9. `scripts/make_router_tables_figures.py`

### 9.1 当前问题

现有仓库产物显示：

- `main_results_table` 与 `threshold_scan_xgb_delta_0.01.csv` 存在明显不一致
- xgb threshold scan 很可能串到了 F4 结果

因此该脚本需要重构，避免 clean/posthoc 与不同 feature set 混用。

### 9.2 patch 目标

- clean 与 posthoc 分开出表
- 主文图表仅从 clean 输入生成

### 9.3 建议 patch 方案

#### 方案 A：拆脚本（推荐）

新增两个脚本：

- `scripts/make_clean_router_tables_figures.py`
- `scripts/make_posthoc_router_tables_figures.py`

#### 方案 B：单脚本增加 mode 参数

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ...
    parser.add_argument("--router-mode", choices=["clean", "posthoc"], required=True)
    return parser.parse_args()
```

#### 读取表格时只过滤对应 mode

```python
def load_main_rows(path: Path, router_mode: str) -> list[dict]:
    rows = read_csv(path)
    return [row for row in rows if row["router_mode"] == router_mode]
```

#### 生成主文表名

```python
def resolve_table_name(router_mode: str, stem: str) -> str:
    return f"{stem}_{router_mode}.md"
```

例如：

- `main_results_table_clean.md`
- `subgroup_results_table_clean.md`
- `feature_ablation_clean.md`
- `threshold_coverage_mrr_clean.png`

### 9.4 最低验收点

- clean 图表目录与 posthoc 图表目录彻底分开
- Figure 2 只来自 clean threshold scan

---

## 10. `scripts/verify_router_result_consistency.py`

### 10.1 当前问题

当前 consistency script 默认认为只有一条 learned router 线，并且硬编码：

- `best_learned_is_xgb_delta_0.01_tau_0.7`

clean/posthoc 分家后这不再成立。

### 10.2 patch 目标

- clean checks 与 posthoc checks 分开
- 新增 legality 回归检查

### 10.3 建议 patch 伪代码

#### 新增 legality helper

```python
ILLEGAL_CLEAN_FEATURES = {
    "target_has_img",
    "target_regime",
    "fusion_correct_score",
    "struct_correct_score",
    "fusion_margin",
    "struct_margin",
    "delta_margin",
}


def clean_feature_columns_are_legal(path: str) -> bool:
    cols = json.loads(Path(path).read_text(encoding="utf-8"))
    return all(col not in ILLEGAL_CLEAN_FEATURES for col in cols)
```

#### 拆 checks

```python
def build_clean_checks(...) -> dict:
    return {
        "all_clean_required_files_exist": ...,
        "clean_main_table_has_expected_models": ...,
        "clean_feature_ablation_has_C1_to_C4": ...,
        "no_illegal_features_in_clean_feature_columns": clean_feature_columns_are_legal(...),
    }
```

```python
def build_posthoc_checks(...) -> dict:
    return {
        "all_posthoc_required_files_exist": ...,
        "posthoc_feature_ablation_has_PH1_to_PH4": ...,
        "posthoc_best_ge_clean_best": ...,
    }
```

#### 删除硬编码最优点

不要再检查：

```python
best_learned_is_xgb_delta_0.01_tau_0.7
```

改成动态检查：

```python
def best_row(rows: list[dict], router_mode: str) -> dict:
    candidates = [row for row in rows if row["router_mode"] == router_mode and row["model"] in {"logistic", "xgb"}]
    return max(candidates, key=lambda row: float(row["mrr"]))
```

### 10.4 最低验收点

- 一致性脚本能自动阻止 clean 结果重新混入非法特征
- posthoc 仍能维持旧分析线完整可追溯

---

## 11. `router/prior_utils.py`

### 11.1 当前问题

当前 relation prior 本身没大问题，但若 `relation_support` 很低，直接用 `mean_delta_rr` 可能会不稳。

### 11.2 patch 目标

- 保留 dev-only prior 逻辑
- 可选加 shrinkage 版 prior，作为 clean router 的增强版输入

### 11.3 建议 patch 伪代码

#### 新增 shrinkage helper

```python
def shrink_mean_delta_rr(mean_delta_rr: float, n_queries: int, k: float = 20.0) -> float:
    return float(mean_delta_rr) * float(n_queries / (n_queries + k))
```

#### 修改 `compute_relation_gain_stats()`

```python
def compute_relation_gain_stats(
    gate_rows: list[dict],
    residual_rows: list[dict],
    gamma: float,
    use_shrinkage: bool = False,
    shrink_k: float = 20.0,
) -> list[dict]:
    ...
    raw_mean_delta_rr = bucket["sum_delta_rr"] / n if n else 0.0
    mean_delta_rr = (
        shrink_mean_delta_rr(raw_mean_delta_rr, n, shrink_k)
        if use_shrinkage else raw_mean_delta_rr
    )
    ...
    rows.append(
        {
            ...,
            "mean_delta_rr": mean_delta_rr,
            "mean_delta_rr_raw": raw_mean_delta_rr,
            ...,
        }
    )
```

### 11.4 最低验收点

- clean 主线即使使用 shrinkage，也仍只基于 dev set 统计

---

## 12. 建议新增脚本：`scripts/run_router_predictions.py`

### 12.1 目标

当前多个脚本都可能单独读取 test table 并重新计算概率，容易出不一致。

新增统一预测脚本后：

- `evaluate_router_results.py`
- `eval_router_threshold_scan.py`
- `make_router_tables_figures.py`

都可以基于同一份 prediction csv 继续处理。

### 12.2 建议伪代码

```python
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-table", required=True)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def read_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    return pd.read_csv(path).to_dict(orient="records")


def load_router(model_dir: Path):
    with (model_dir / "model.pkl").open("rb") as handle:
        return pickle.load(handle)


def load_train_summary(model_dir: Path) -> dict:
    return json.loads((model_dir / "train_summary.json").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    rows = read_table(Path(args.test_table))
    router = load_router(model_dir)
    summary = load_train_summary(model_dir)

    probs = router.predict_proba_from_rows(rows)

    out_rows = []
    for row, prob in zip(rows, probs):
        out_rows.append(
            {
                "query_id": row["query_id"],
                "split": row["split"],
                "seed": row["seed"],
                "direction": row["direction"],
                "relation_id": row["relation_id"],
                "router_mode": summary["router_mode"],
                "feature_set": summary["feature_set"],
                "is_query_time_legal": summary["is_query_time_legal"],
                "router_prob": float(prob),
            }
        )

    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
```

---

## 13. 建议新增脚本：`scripts/run_router_significance.py`

### 13.1 目标

给 clean 主文补：

- seed-wise MRR
- mean ± std
- paired bootstrap CI vs residual-only
- paired bootstrap CI vs clean rule-based

### 13.2 最小伪代码骨架

```python
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routed-csv", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def seedwise_mrr(rows: pd.DataFrame) -> dict:
    out = {}
    for seed, group in rows.groupby("seed"):
        out[int(seed)] = float(group["rr_final"].mean())
    return out


def main() -> None:
    args = parse_args()
    routed = pd.read_csv(args.routed_csv)
    baseline = pd.read_csv(args.baseline_csv)

    payload = {
        "seedwise_routed_mrr": seedwise_mrr(routed),
        "seedwise_baseline_mrr": seedwise_mrr(baseline),
        # TODO: add paired bootstrap
    }

    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

---

## 14. 最后推荐的落地顺序

### 第一轮：只做 clean/posthoc 分家

1. 改 `router/schemas.py`
2. 改 `router/feature_utils.py`
3. 改 `scripts/build_router_features.py`
4. 改 `router/router_models.py`

### 第二轮：把训练与评估链路改通

5. 改 `scripts/train_router.py`
6. 改 `router/routing_utils.py`
7. 改 `scripts/eval_router_threshold_scan.py`
8. 改 `scripts/evaluate_router_results.py`

### 第三轮：重建 paper-facing outputs

9. 改 `scripts/make_router_tables_figures.py`
10. 改 `scripts/verify_router_result_consistency.py`
11. 可选新增 `run_router_predictions.py`
12. 可选新增 `run_router_significance.py`

---

## 15. 一句话总结

这份 patch 伪代码模板的核心目标是：

> 在不推翻现有 query-level expert outcome、dev-only priors 与三-seed 结果基础的前提下，把 feature schema、model definition、training I/O、evaluation tables 和 figure generation 全链路改造成 clean / posthoc 双线结构，并保证主文只依赖 clean 线。