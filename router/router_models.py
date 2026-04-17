from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


FEATURE_SETS: dict[str, list[str]] = {
    "F1": ["target_has_img"],
    "F2": ["target_has_img", "direction", "relation_gain_prior"],
    "F3": ["target_has_img", "direction", "relation_gain_prior", "text_img_cosine", "img_is_missing_replaced"],
    "F4": [
        "target_has_img",
        "direction",
        "relation_gain_prior",
        "text_img_cosine",
        "img_is_missing_replaced",
        "fusion_margin",
        "struct_margin",
        "delta_margin",
    ],
    "FULL": [
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


def cast_feature_value(key: str, value: str | int | float | None) -> Any:
    if value in ("", None):
        return 0.0
    if key in {"direction", "target_regime"}:
        return str(value)
    if key in {"relation_id"}:
        return f"rel_{int(value)}"
    if key in {"target_has_img", "img_is_missing_replaced", "relation_is_visual_prior"}:
        return int(value)
    return float(value)


def rows_to_feature_dicts(rows: list[dict], feature_names: list[str]) -> list[dict]:
    payload = []
    for row in rows:
        payload.append({name: cast_feature_value(name, row.get(name)) for name in feature_names})
    return payload


def extract_labels(rows: list[dict]) -> list[int]:
    return [int(row["label_gain"]) for row in rows]


@dataclass
class RuleBasedRouter:
    gamma: float = 0.0

    def predict_proba_from_rows(self, rows: list[dict]) -> list[float]:
        probs = []
        for row in rows:
            use_fusion = int(row["target_has_img"]) == 1 and float(row["relation_gain_prior"]) > float(self.gamma)
            probs.append(1.0 if use_fusion else 0.0)
        return probs

    def predict_from_rows(self, rows: list[dict], threshold: float = 0.5) -> list[int]:
        return [int(p >= threshold) for p in self.predict_proba_from_rows(rows)]


@dataclass
class TrainedRouterArtifact:
    model_name: str
    feature_set: str
    feature_names: list[str]
    vectorizer: DictVectorizer | None
    estimator: Any

    def predict_proba_from_rows(self, rows: list[dict]) -> list[float]:
        if self.vectorizer is None:
            raise RuntimeError("This artifact does not support vectorized inference.")
        payload = rows_to_feature_dicts(rows, self.feature_names)
        x = self.vectorizer.transform(payload)
        if hasattr(self.estimator, "predict_proba"):
            return [float(p[1]) for p in self.estimator.predict_proba(x)]
        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(x)
            return [float(1.0 / (1.0 + pow(2.718281828459045, -float(s)))) for s in scores]
        raise RuntimeError(f"Estimator {type(self.estimator).__name__} does not expose probability-like outputs.")

    def predict_from_rows(self, rows: list[dict], threshold: float = 0.5) -> list[int]:
        return [int(p >= threshold) for p in self.predict_proba_from_rows(rows)]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        return path


def train_logistic_router(rows: list[dict], feature_set: str, random_state: int = 42) -> TrainedRouterArtifact:
    feature_names = FEATURE_SETS[feature_set]
    payload = rows_to_feature_dicts(rows, feature_names)
    y = extract_labels(rows)

    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform(payload)
    estimator = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
        solver="liblinear",
    )
    estimator.fit(x, y)
    return TrainedRouterArtifact(
        model_name="logistic",
        feature_set=feature_set,
        feature_names=feature_names,
        vectorizer=vectorizer,
        estimator=estimator,
    )


def train_xgb_router(rows: list[dict], feature_set: str, random_state: int = 42) -> TrainedRouterArtifact:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is not installed in the current environment.") from exc

    feature_names = FEATURE_SETS[feature_set]
    payload = rows_to_feature_dicts(rows, feature_names)
    y = extract_labels(rows)
    positive = sum(y)
    negative = len(y) - positive
    scale_pos_weight = float(negative / positive) if positive > 0 else 1.0

    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform(payload)
    estimator = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
    )
    estimator.fit(x, y)
    return TrainedRouterArtifact(
        model_name="xgb",
        feature_set=feature_set,
        feature_names=feature_names,
        vectorizer=vectorizer,
        estimator=estimator,
    )


def compute_feature_importance_rows(artifact: TrainedRouterArtifact) -> list[dict]:
    if artifact.vectorizer is None:
        return []
    names = artifact.vectorizer.get_feature_names_out()
    if artifact.model_name == "logistic":
        weights = artifact.estimator.coef_[0]
        importance = [abs(float(w)) for w in weights]
    elif artifact.model_name == "xgb":
        importance = [float(w) for w in artifact.estimator.feature_importances_]
    else:
        return []
    rows = [
        {
            "model": artifact.model_name,
            "feature_set": artifact.feature_set,
            "feature_name": name,
            "importance": score,
        }
        for name, score in zip(names, importance)
    ]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows
