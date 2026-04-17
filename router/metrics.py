from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true, y_pred, y_prob) -> dict[str, Any]:
    y_true = list(int(x) for x in y_true)
    y_pred = list(int(x) for x in y_pred)
    y_prob = list(float(x) for x in y_prob)

    positive_rate = float(sum(y_true) / len(y_true)) if y_true else 0.0
    pred_positive_rate = float(sum(y_pred) / len(y_pred)) if y_pred else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None

    return {
        "n_samples": len(y_true),
        "positive_rate": positive_rate,
        "pred_positive_rate": pred_positive_rate,
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
