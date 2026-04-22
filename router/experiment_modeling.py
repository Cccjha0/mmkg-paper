from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedShuffleSplit

from router.constants import ROUTER_MODE_CLEAN
from router.router_models import get_feature_sets, rows_to_feature_dicts


VALID_DELTAS = {"0.00": "gain_label_d0", "0.01": "gain_label_d001", "0.02": "gain_label_d002"}


def read_table(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported file type: {path.as_posix()}")


def normalize_binary_train_rows(rows: list[dict], delta_str: str) -> list[dict]:
    label_name = VALID_DELTAS[delta_str]
    payload = []
    for row in rows:
        out = dict(row)
        if "label_gain" in row and row["label_gain"] not in ("", None):
            out["label_gain"] = int(row["label_gain"])
        elif label_name in row:
            out["label_gain"] = int(row[label_name])
        else:
            raise KeyError(f"Missing both label_gain and {label_name}")
        payload.append(out)
    return payload


def join_eval_fields(test_rows: list[dict], eval_rows: list[dict], delta_str: str | None = None) -> list[dict]:
    eval_by_id = {str(row["query_id"]): row for row in eval_rows}
    payload = []
    label_name = VALID_DELTAS.get(delta_str) if delta_str else None
    for row in test_rows:
        query_id = str(row["query_id"])
        if query_id not in eval_by_id:
            raise RuntimeError(f"Missing eval target for query_id={query_id}")
        target = eval_by_id[query_id]
        merged = dict(row)
        merged["rr_gate"] = float(target["rr_gate"])
        merged["rr_residual"] = float(target["rr_residual"])
        merged["target_regime"] = str(target["target_regime"])
        merged["direction"] = str(target.get("direction", row.get("direction", "")))
        merged["relation_id"] = int(target.get("relation_id", row.get("relation_id", 0)))
        if label_name:
            merged["label_gain"] = int(target[label_name])
        payload.append(merged)
    return payload


def merge_delta_rr_targets(train_rows: list[dict], gain_label_csvs: list[str]) -> list[dict]:
    delta_rr_by_id = {}
    for path in gain_label_csvs:
        for row in read_table(path):
            delta_rr_by_id[str(row["query_id"])] = {
                "delta_rr": float(row["delta_rr"]),
                "rr_gate": float(row.get("rr_fusion", row.get("rr_gate", 0.0))),
                "rr_residual": float(row.get("rr_struct", row.get("rr_residual", 0.0))),
            }

    payload = []
    for row in train_rows:
        query_id = str(row["query_id"])
        if query_id not in delta_rr_by_id:
            raise RuntimeError(f"Missing delta_rr target for query_id={query_id}")
        merged = dict(row)
        merged.update(delta_rr_by_id[query_id])
        payload.append(merged)
    return payload


def stratified_calibration_split(rows: list[dict], label_key: str, test_size: float, random_state: int) -> tuple[list[dict], list[dict]]:
    if not rows:
        raise RuntimeError("Cannot split empty rows.")
    y = [int(row[label_key]) for row in rows]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, calib_idx = next(splitter.split([[0]] * len(rows), y))
    train_rows = [rows[idx] for idx in train_idx]
    calib_rows = [rows[idx] for idx in calib_idx]
    return train_rows, calib_rows


def vectorize_rows(rows: list[dict], feature_set: str, router_mode: str = ROUTER_MODE_CLEAN) -> tuple[DictVectorizer, Any]:
    feature_names = get_feature_sets(router_mode)[feature_set]
    payload = rows_to_feature_dicts(rows, feature_names)
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform(payload)
    return vectorizer, x


def transform_rows(vectorizer: DictVectorizer, rows: list[dict], feature_names: list[str]) -> Any:
    return vectorizer.transform(rows_to_feature_dicts(rows, feature_names))


@dataclass
class ExperimentArtifact:
    task: str
    model_name: str
    feature_set: str
    feature_names: list[str]
    vectorizer: DictVectorizer
    estimator: Any
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return path

    def predict_scores(self, rows: list[dict]) -> list[float]:
        x = transform_rows(self.vectorizer, rows, self.feature_names)
        if self.task == "regression":
            return [float(value) for value in self.estimator.predict(x)]
        if self.task == "classification":
            if hasattr(self.estimator, "predict_proba"):
                return [float(prob[1]) for prob in self.estimator.predict_proba(x)]
        raise RuntimeError(f"Artifact task {self.task} does not support predict_scores.")

    def predict_class_proba(self, rows: list[dict]) -> list[list[float]]:
        x = transform_rows(self.vectorizer, rows, self.feature_names)
        if not hasattr(self.estimator, "predict_proba"):
            raise RuntimeError(f"Estimator {type(self.estimator).__name__} has no predict_proba.")
        return [[float(value) for value in probs] for probs in self.estimator.predict_proba(x)]

    def predict_class(self, rows: list[dict]) -> list[int]:
        x = transform_rows(self.vectorizer, rows, self.feature_names)
        return [int(value) for value in self.estimator.predict(x)]


def train_regression_artifact(
    rows: list[dict],
    feature_set: str,
    model_type: str,
    random_state: int,
) -> ExperimentArtifact:
    feature_names = get_feature_sets(ROUTER_MODE_CLEAN)[feature_set]
    vectorizer, x = vectorize_rows(rows, feature_set, router_mode=ROUTER_MODE_CLEAN)
    y = [float(row["delta_rr"]) for row in rows]

    if model_type == "linear":
        estimator = Ridge(alpha=1.0, random_state=random_state)
    elif model_type == "xgb":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        )
    elif model_type == "lgbm":
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported regression model_type: {model_type}")

    estimator.fit(x, y)
    return ExperimentArtifact(
        task="regression",
        model_name=model_type,
        feature_set=feature_set,
        feature_names=feature_names,
        vectorizer=vectorizer,
        estimator=estimator,
        metadata={"random_state": random_state},
    )


def assign_ordinal_bucket(delta_rr: float, thresholds: list[float]) -> int:
    if len(thresholds) != 3:
        raise ValueError("Ordinal thresholds must contain exactly three cut points.")
    if delta_rr < thresholds[0]:
        return 0
    if delta_rr < thresholds[1]:
        return 1
    if delta_rr < thresholds[2]:
        return 2
    return 3


def train_ordinal_artifact(
    rows: list[dict],
    feature_set: str,
    model_type: str,
    random_state: int,
    thresholds: list[float],
) -> ExperimentArtifact:
    feature_names = get_feature_sets(ROUTER_MODE_CLEAN)[feature_set]
    vectorizer, x = vectorize_rows(rows, feature_set, router_mode=ROUTER_MODE_CLEAN)
    y = [assign_ordinal_bucket(float(row["delta_rr"]), thresholds) for row in rows]

    if model_type == "logistic":
        estimator = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        )
    elif model_type == "xgb":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported ordinal model_type: {model_type}")

    estimator.fit(x, y)
    class_counts = {str(label): int(y.count(label)) for label in sorted(set(y))}
    return ExperimentArtifact(
        task="ordinal",
        model_name=model_type,
        feature_set=feature_set,
        feature_names=feature_names,
        vectorizer=vectorizer,
        estimator=estimator,
        metadata={"random_state": random_state, "thresholds": thresholds, "class_counts": class_counts},
    )


def write_summary(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
