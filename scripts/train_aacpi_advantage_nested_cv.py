from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.constants import QUERY_GEOMETRY_FIELDS
from scripts.crossfit_heterogeneous_dev_policies import assign_grouped_folds, triple_key


ZERO_TOLERANCE = 1e-15
CALIBRATION_BUCKETS = (
    ("lowest_10pct", 0.00, 0.10),
    ("10_to_30pct", 0.10, 0.30),
    ("30_to_50pct", 0.30, 0.50),
    ("50_to_70pct", 0.50, 0.70),
    ("70_to_90pct", 0.70, 0.90),
    ("highest_10pct", 0.90, 1.00),
)


@dataclass(frozen=True)
class ModelConfig:
    hidden_width: int
    learning_rate: float
    negative_weight: float

    @property
    def config_id(self) -> str:
        lr = f"{self.learning_rate:.0e}".replace("-0", "-")
        weight = str(self.negative_weight).replace(".", "p")
        return f"w{self.hidden_width}_lr{lr}_neg{weight}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the AACPI V2 Phase 2B advantage predictor with original-triple "
            "grouped nested CV. This script produces DEV OOF predictions only."
        )
    )
    parser.add_argument("--utility-table", required=True)
    parser.add_argument(
        "--search-space",
        default="docs/protocols/AACPI_V2_PHASE2_SEARCH_SPACE.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the frozen configuration, input, and fold isolation without training.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the frozen Phase 2 search space") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def validate_search_space(space: dict) -> None:
    if space.get("status") != "frozen_before_systematic_training":
        raise ValueError("Phase 2 search space must be frozen before training")
    if space.get("data", {}).get("split") != "dev":
        raise ValueError("Phase 2B accepts DEV only")
    if space.get("data", {}).get("target_clipping") != "none":
        raise ValueError("The primary Phase 2B run must not clip targets")
    if space.get("architecture", {}).get("family") != "two_hidden_layer_mlp":
        raise ValueError("Only the frozen two-hidden-layer MLP is allowed")
    if space.get("architecture", {}).get("dropout") != 0.0:
        raise ValueError("Dropout is not in the frozen search space")
    if space.get("loss", {}).get("family") != "smooth_l1":
        raise ValueError("Only frozen Smooth-L1 is allowed")
    if space.get("training", {}).get("early_stopping") is not False:
        raise ValueError("Phase 2B uses the frozen fixed-epoch training rule")
    expected_features = [*QUERY_GEOMETRY_FIELDS, "delta_alpha", "abs_delta_alpha"]
    if list(space.get("data", {}).get("feature_fields", [])) != expected_features:
        raise ValueError("Frozen feature fields do not match the AACPI V2 feature contract")


def model_configs(space: dict) -> list[ModelConfig]:
    configs = [
        ModelConfig(int(width), float(lr), float(weight))
        for width, lr, weight in itertools.product(
            space["architecture"]["hidden_width"],
            space["optimizer"]["learning_rate"],
            space["loss"]["negative_advantage_weight"],
        )
    ]
    ids = [config.config_id for config in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("Frozen model configurations do not have unique IDs")
    return configs


def read_utility_table(path: Path, feature_fields: list[str]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for Phase 2B data loading") from exc
    required = [
        "dataset",
        "pair_id",
        "split",
        "original_triple_id",
        "query_id",
        "seed",
        "direction",
        "head",
        "relation",
        "tail",
        "alpha0",
        "alpha",
        "delta_alpha",
        "abs_delta_alpha",
        "rr_anchor",
        "rr_action",
        "advantage",
        *QUERY_GEOMETRY_FIELDS,
    ]
    frame = pd.read_csv(path, compression="infer", usecols=required)
    if frame.empty:
        raise ValueError("Utility table is empty")
    if set(frame["split"].astype(str)) != {"dev"}:
        raise RuntimeError("AACPI Phase 2B is DEV-only and rejects non-DEV rows")
    if frame[required].isna().any().any():
        missing = frame[required].columns[frame[required].isna().any()].tolist()
        raise ValueError(f"Utility table contains missing values: {missing}")
    if frame["query_id"].astype(str).str.len().eq(0).any():
        raise ValueError("Utility table contains empty query IDs")
    if frame["original_triple_id"].astype(str).str.len().eq(0).any():
        raise ValueError("Utility table contains empty original-triple IDs")
    features = frame[feature_fields].to_numpy(dtype=np.float64)
    target = frame["advantage"].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Utility features and targets must be finite")
    if not np.allclose(
        frame["abs_delta_alpha"].to_numpy(dtype=np.float64),
        np.abs(frame["delta_alpha"].to_numpy(dtype=np.float64)),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("abs_delta_alpha is inconsistent")
    reference = np.isclose(
        frame["alpha"].to_numpy(dtype=np.float64),
        frame["alpha0"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    if not np.all(np.abs(target[reference]) <= ZERO_TOLERANCE):
        raise ValueError("Reference actions must have zero advantage")
    return frame, features, target, reference


def representative_group_rows(frame) -> list[dict[str, str]]:
    representatives = frame.drop_duplicates("original_triple_id")
    rows = []
    for item in representatives.itertuples(index=False):
        row = {
            "head_id": str(int(item.head)),
            "relation_id": str(int(item.relation)),
            "tail_id": str(int(item.tail)),
        }
        if triple_key(row) != str(item.original_triple_id):
            raise ValueError(
                "original_triple_id does not match head/relation/tail: "
                f"{item.original_triple_id!r}"
            )
        rows.append(row)
    return rows


def grouped_fold_vector(frame, *, folds: int, fold_seed: int) -> tuple[np.ndarray, dict]:
    representatives = representative_group_rows(frame)
    assignment, audit = assign_grouped_folds(representatives, folds=folds, fold_seed=fold_seed)
    vector = frame["original_triple_id"].map(assignment).to_numpy(dtype=np.int64)
    if np.any(vector < 0) or np.any(vector >= folds):
        raise AssertionError("Grouped fold assignment is outside the expected range")
    for group, group_rows in frame.assign(_fold=vector).groupby("original_triple_id"):
        if group_rows["_fold"].nunique() != 1:
            raise AssertionError(f"Original triple crosses folds: {group}")
    utility_rows_per_fold = {str(fold): int((vector == fold).sum()) for fold in range(folds)}
    query_instances_per_fold = {
        str(fold): int(frame.loc[vector == fold, "query_id"].nunique()) for fold in range(folds)
    }
    if any(count == 0 for count in utility_rows_per_fold.values()):
        raise ValueError("Grouped fold assignment produced an empty fold")
    audit["utility_rows_per_fold"] = utility_rows_per_fold
    audit["query_instances_per_fold"] = query_instances_per_fold
    return vector, audit


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2:
        return float("nan")
    actual_rank = average_ranks(actual)
    predicted_rank = average_ranks(predicted)
    if np.std(actual_rank) == 0.0 or np.std(predicted_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(actual_rank, predicted_rank)[0, 1])


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    total = np.arange(1, len(labels) + 1)
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(labels) - 1]
    precision = cumulative_tp[ends] / total[ends]
    recall = cumulative_tp[ends] / positives
    previous = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous) * precision))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def smooth_l1_mean(actual: np.ndarray, predicted: np.ndarray, beta: float) -> float:
    error = np.abs(predicted - actual)
    values = np.where(error < beta, 0.5 * error * error / beta, error - 0.5 * beta)
    return float(values.mean())


def evaluate_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    beta: float,
) -> dict[str, float | int]:
    if len(actual) != len(predicted) or not len(actual):
        raise ValueError("Prediction evaluation requires aligned non-empty arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Prediction evaluation requires finite arrays")
    positive = actual > ZERO_TOLERANCE
    harmful = actual < -ZERO_TOLERANCE
    positive_prevalence = float(positive.mean())
    harmful_prevalence = float(harmful.mean())
    positive_auprc = average_precision(positive, predicted)
    harmful_auprc = average_precision(harmful, -predicted)
    return {
        "count": len(actual),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "smooth_l1": smooth_l1_mean(actual, predicted, beta),
        "spearman": spearman_correlation(actual, predicted),
        "positive_prevalence": positive_prevalence,
        "positive_auprc": positive_auprc,
        "positive_auprc_lift": positive_auprc - positive_prevalence,
        "positive_auroc": roc_auc(positive, predicted),
        "harmful_prevalence": harmful_prevalence,
        "harmful_auprc": harmful_auprc,
        "harmful_auprc_lift": harmful_auprc - harmful_prevalence,
        "harmful_auroc": roc_auc(harmful, -predicted),
    }


def calibration_rows(actual: np.ndarray, predicted: np.ndarray) -> list[dict]:
    order = np.argsort(predicted, kind="mergesort")
    rows = []
    for label, lower, upper in CALIBRATION_BUCKETS:
        start = int(round(lower * len(order)))
        end = int(round(upper * len(order)))
        indices = order[start:end]
        if len(indices) == 0:
            raise ValueError(f"Calibration bucket is empty: {label}")
        rows.append(
            {
                "bucket": label,
                "lower_fraction": lower,
                "upper_fraction": upper,
                "count": len(indices),
                "predicted_mean_u": float(predicted[indices].mean()),
                "predicted_min_u": float(predicted[indices].min()),
                "predicted_max_u": float(predicted[indices].max()),
                "actual_mean_u": float(actual[indices].mean()),
                "positive_rate": float((actual[indices] > ZERO_TOLERANCE).mean()),
                "harmful_rate": float((actual[indices] < -ZERO_TOLERANCE).mean()),
                "zero_rate": float((np.abs(actual[indices]) <= ZERO_TOLERANCE).mean()),
            }
        )
    return rows


def write_calibration_svg(path: Path, rows: list[dict], *, dataset: str, pair_id: str) -> None:
    width, height = 1040, 460
    labels = [row["bucket"] for row in rows]
    predicted = [float(row["predicted_mean_u"]) for row in rows]
    actual = [float(row["actual_mean_u"]) for row in rows]
    utility_scale = max([abs(value) for value in predicted + actual] + [1e-6])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.subtitle{font-size:14px;font-weight:700}.axis{font-size:10px}.legend{font-size:12px}</style>',
        '<text x="24" y="30" class="title">AACPI Phase 2B OOF advantage calibration</text>',
        f'<text x="24" y="52" class="subtitle">{html.escape(dataset)} / {html.escape(pair_id)} — non-reference DEV actions</text>',
    ]
    chart_y, chart_h = 105, 250
    left_x, chart_w = 70, 410
    zero_y = chart_y + chart_h / 2
    parts.append(f'<text x="{left_x}" y="{chart_y - 20}" class="subtitle">Predicted vs actual mean U</text>')
    parts.append(f'<line x1="{left_x}" y1="{zero_y}" x2="{left_x + chart_w}" y2="{zero_y}" stroke="#a0aec0" stroke-dasharray="4 3"/>')
    for values, color, name in ((predicted, "#2b6cb0", "Predicted"), (actual, "#2f855a", "Actual")):
        points = []
        for index, value in enumerate(values):
            x = left_x + (index + 0.5) * chart_w / len(rows)
            y = zero_y - value / utility_scale * chart_h * 0.44
            points.append(f"{x:.2f},{y:.2f}")
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_x = left_x + (0 if name == "Predicted" else 110)
        parts.append(f'<line x1="{legend_x}" y1="{chart_y + chart_h + 62}" x2="{legend_x + 24}" y2="{chart_y + chart_h + 62}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{chart_y + chart_h + 66}" class="legend">{name}</text>')
    for index, label in enumerate(labels):
        x = left_x + (index + 0.5) * chart_w / len(rows)
        parts.append(f'<text x="{x:.2f}" y="{chart_y + chart_h + 18}" text-anchor="middle" class="axis">{html.escape(label.replace("pct", "%"))}</text>')
    parts.append(f'<text x="{left_x - 8}" y="{chart_y + 3}" text-anchor="end" class="axis">+{utility_scale:.4f}</text>')
    parts.append(f'<text x="{left_x - 8}" y="{zero_y + 3}" text-anchor="end" class="axis">0</text>')
    parts.append(f'<text x="{left_x - 8}" y="{chart_y + chart_h}" text-anchor="end" class="axis">-{utility_scale:.4f}</text>')

    right_x = 580
    parts.append(f'<text x="{right_x}" y="{chart_y - 20}" class="subtitle">Actual outcome rates</text>')
    parts.append(f'<line x1="{right_x}" y1="{chart_y + chart_h}" x2="{right_x + chart_w}" y2="{chart_y + chart_h}" stroke="#5f6368"/>')
    bar_group = chart_w / len(rows)
    for index, row in enumerate(rows):
        x = right_x + index * bar_group + 8
        bar_w = (bar_group - 18) / 2
        for offset, field, color in ((0, "positive_rate", "#2f855a"), (bar_w, "harmful_rate", "#c53030")):
            bar_h = chart_h * float(row[field])
            parts.append(f'<rect x="{x + offset:.2f}" y="{chart_y + chart_h - bar_h:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{right_x + (index + 0.5)*bar_group:.2f}" y="{chart_y + chart_h + 18}" text-anchor="middle" class="axis">{html.escape(labels[index].replace("pct", "%"))}</text>')
    for fraction in (0.0, 0.5, 1.0):
        y = chart_y + chart_h * (1.0 - fraction)
        parts.append(f'<text x="{right_x - 8}" y="{y + 3}" text-anchor="end" class="axis">{int(100*fraction)}%</text>')
    parts.append(f'<rect x="{right_x}" y="{chart_y + chart_h + 48}" width="12" height="12" fill="#2f855a"/><text x="{right_x + 18}" y="{chart_y + chart_h + 59}" class="legend">P(U &gt; 0)</text>')
    parts.append(f'<rect x="{right_x + 110}" y="{chart_y + chart_h + 48}" width="12" height="12" fill="#c53030"/><text x="{right_x + 128}" y="{chart_y + chart_h + 59}" class="legend">P(U &lt; 0)</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_oof_report(
    path: Path,
    *,
    dataset: str,
    pair_id: str,
    primary: dict,
    all_actions: dict,
    calibration: list[dict],
) -> None:
    lines = [
        "# AACPI V2 Phase 2B OOF Advantage Learnability",
        "",
        f"Dataset/pair: `{dataset} / {pair_id}`",
        "",
        "All primary metrics use genuinely outer-fold OOF predictions on non-reference DEV actions. The alpha0 rows are reported only as a supplemental scope.",
        "",
        "| Scope | MAE | Smooth-L1 | Spearman | Positive AUPRC / prevalence | Harmful AUPRC / prevalence |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, metrics in (("Non-reference", primary), ("All actions", all_actions)):
        lines.append(
            f"| {scope} | {metrics['mae']:.6f} | {metrics['smooth_l1']:.6f} | "
            f"{metrics['spearman']:.6f} | {metrics['positive_auprc']:.6f} / "
            f"{metrics['positive_prevalence']:.6f} | {metrics['harmful_auprc']:.6f} / "
            f"{metrics['harmful_prevalence']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Predicted-U bucket | Predicted mean U | Actual mean U | P(U>0) | P(U<0) | P(U=0) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in calibration:
        lines.append(
            f"| {row['bucket']} | {row['predicted_mean_u']:+.6f} | "
            f"{row['actual_mean_u']:+.6f} | {row['positive_rate']:.6f} | "
            f"{row['harmful_rate']:.6f} | {row['zero_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "No kappa, lambda, tau, uncertainty penalty, fallback threshold, or policy evaluation is used in Phase 2B.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inner_selection_key(metrics: dict, config: ModelConfig) -> tuple:
    positive_lift = float(metrics["positive_auprc_lift"])
    harmful_lift = float(metrics["harmful_auprc_lift"])
    return (
        min(positive_lift, harmful_lift),
        0.5 * (positive_lift + harmful_lift),
        float(metrics["spearman"]),
        -float(metrics["mae"]),
        -config.hidden_width,
        -abs(config.negative_weight - 1.0),
        -config.learning_rate,
    )


def resolve_device(requested: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Phase 2B training") from exc
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def train_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    config: ModelConfig,
    space: dict,
    device: str,
    seed: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_x = torch.from_numpy(((x_train - mean) / scale).astype(np.float32)).to(device)
    train_y = torch.from_numpy(y_train.astype(np.float32)).to(device)
    eval_x = torch.from_numpy(((x_eval - mean) / scale).astype(np.float32)).to(device)

    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], config.hidden_width),
        torch.nn.ReLU(),
        torch.nn.Linear(config.hidden_width, config.hidden_width),
        torch.nn.ReLU(),
        torch.nn.Linear(config.hidden_width, 1),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=float(space["optimizer"]["weight_decay"]),
    )
    epochs = int(space["training"]["epochs"])
    batch_size = int(space["training"]["batch_size"])
    beta = float(space["loss"]["beta"])
    max_norm = float(space["optimizer"]["max_gradient_norm"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(train_x), generator=generator, device=device)
        for start in range(0, len(train_x), batch_size):
            indices = permutation[start : start + batch_size]
            prediction = model(train_x[indices]).squeeze(1)
            loss = functional.smooth_l1_loss(
                prediction,
                train_y[indices],
                reduction="none",
                beta=beta,
            )
            weights = torch.where(
                train_y[indices] < 0.0,
                torch.full_like(loss, config.negative_weight),
                torch.ones_like(loss),
            )
            objective = (loss * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(eval_x), batch_size):
            predictions.append(model(eval_x[start : start + batch_size]).squeeze(1).cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_nested_cv(
    utility_path: Path,
    search_path: Path,
    output_dir: Path,
    *,
    requested_device: str,
    dry_run: bool,
    overwrite: bool,
) -> dict:
    space = load_yaml(search_path)
    validate_search_space(space)
    configs = model_configs(space)
    features_fields = list(space["data"]["feature_fields"])
    frame, features, target, reference = read_utility_table(utility_path, features_fields)
    outer_folds = int(space["nested_cv"]["outer_folds"])
    outer_vector, outer_audit = grouped_fold_vector(
        frame,
        folds=outer_folds,
        fold_seed=int(space["nested_cv"]["outer_fold_seed"]),
    )
    pair_values = set(frame["pair_id"].astype(str))
    dataset_values = set(frame["dataset"].astype(str))
    if len(pair_values) != 1 or len(dataset_values) != 1:
        raise ValueError("Each nested-CV run must contain one dataset/expert pair")
    pair_id = next(iter(pair_values))
    dataset = next(iter(dataset_values))

    input_audit = {
        "schema_version": 1,
        "phase": "AACPI V2 Phase 2B",
        "split": "dev",
        "dataset": dataset,
        "pair_id": pair_id,
        "utility_table": portable_path(utility_path),
        "utility_sha256": sha256_file(utility_path),
        "search_space": portable_path(search_path),
        "search_space_sha256": sha256_file(search_path),
        "n_rows": len(frame),
        "n_query_instances": int(frame["query_id"].nunique()),
        "n_original_triples": int(frame["original_triple_id"].nunique()),
        "n_reference_rows": int(reference.sum()),
        "n_nonreference_rows": int((~reference).sum()),
        "n_configs": len(configs),
        "outer_fold_audit": outer_audit,
        "dry_run": dry_run,
        "test_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dry_path = output_dir / "phase2b_input_audit.json"
    if dry_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {dry_path} without --overwrite")
    dry_path.write_text(json.dumps(input_audit, indent=2) + "\n", encoding="utf-8")
    if dry_run:
        print(
            f"[DRY-RUN OK] pair={pair_id} rows={len(frame)} groups={input_audit['n_original_triples']} "
            f"configs={len(configs)} outer_folds={outer_folds}"
        )
        return input_audit

    expected_outputs = (
        output_dir / "dev_oof_predictions.csv.gz",
        output_dir / "outer_fold_selections.csv",
        output_dir / "inner_search_results.csv",
        output_dir / "dev_oof_metrics.json",
        output_dir / "dev_oof_metrics.csv",
        output_dir / "dev_oof_calibration.csv",
        output_dir / "dev_oof_calibration.svg",
        output_dir / "dev_oof_report.md",
    )
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Phase 2B outputs: {existing}")

    device = resolve_device(requested_device)
    inner_folds = int(space["nested_cv"]["inner_folds"])
    inner_seed = int(space["nested_cv"]["inner_fold_seed"])
    model_seed = int(space["training"]["model_seed"])
    beta = float(space["loss"]["beta"])
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    selected_config_ids = np.asarray([None] * len(frame), dtype=object)
    outer_selection_rows = []
    inner_result_rows = []

    for outer_fold in range(outer_folds):
        outer_train = outer_vector != outer_fold
        outer_holdout = outer_vector == outer_fold
        inner_frame = frame.loc[outer_train].copy()
        inner_vector, inner_audit = grouped_fold_vector(
            inner_frame,
            folds=inner_folds,
            fold_seed=inner_seed + outer_fold,
        )
        outer_train_indices = np.flatnonzero(outer_train)
        candidates = []
        for config_index, config in enumerate(configs):
            inner_oof = np.full(len(outer_train_indices), np.nan, dtype=np.float64)
            for inner_fold in range(inner_folds):
                inner_train_local = inner_vector != inner_fold
                inner_hold_local = inner_vector == inner_fold
                seed = model_seed + outer_fold * 100_000 + config_index * 100 + inner_fold
                inner_oof[inner_hold_local] = train_predict(
                    features[outer_train_indices[inner_train_local]],
                    target[outer_train_indices[inner_train_local]],
                    features[outer_train_indices[inner_hold_local]],
                    config=config,
                    space=space,
                    device=device,
                    seed=seed,
                )
            if not np.isfinite(inner_oof).all():
                raise AssertionError("Inner OOF predictions are incomplete")
            inner_nonreference = ~reference[outer_train_indices]
            metrics = evaluate_predictions(
                target[outer_train_indices][inner_nonreference],
                inner_oof[inner_nonreference],
                beta=beta,
            )
            selection_key = inner_selection_key(metrics, config)
            candidates.append((selection_key, config, metrics))
            inner_result_rows.append(
                {
                    "outer_fold": outer_fold + 1,
                    "config_id": config.config_id,
                    "hidden_width": config.hidden_width,
                    "learning_rate": config.learning_rate,
                    "negative_weight": config.negative_weight,
                    **metrics,
                }
            )
            print(
                f"[INNER] pair={pair_id} outer={outer_fold + 1}/{outer_folds} "
                f"config={config.config_id} min_ap_lift={selection_key[0]:.6f}",
                flush=True,
            )
        _, selected, selected_metrics = max(candidates, key=lambda item: item[0])
        final_seed = model_seed + outer_fold * 100_000 + 99_999
        oof[outer_holdout] = train_predict(
            features[outer_train],
            target[outer_train],
            features[outer_holdout],
            config=selected,
            space=space,
            device=device,
            seed=final_seed,
        )
        selected_config_ids[outer_holdout] = selected.config_id
        outer_selection_rows.append(
            {
                "outer_fold": outer_fold + 1,
                "selected_config_id": selected.config_id,
                "hidden_width": selected.hidden_width,
                "learning_rate": selected.learning_rate,
                "negative_weight": selected.negative_weight,
                "outer_train_rows": int(outer_train.sum()),
                "outer_holdout_rows": int(outer_holdout.sum()),
                "outer_train_groups": int(frame.loc[outer_train, "original_triple_id"].nunique()),
                "outer_holdout_groups": int(frame.loc[outer_holdout, "original_triple_id"].nunique()),
                "inner_fold_seed": inner_seed + outer_fold,
                "inner_group_audit": json.dumps(inner_audit, sort_keys=True),
                **{f"selected_inner_{key}": value for key, value in selected_metrics.items()},
            }
        )
        print(
            f"[OUTER] pair={pair_id} fold={outer_fold + 1}/{outer_folds} "
            f"selected={selected.config_id}",
            flush=True,
        )

    if not np.isfinite(oof).all() or any(value is None for value in selected_config_ids):
        raise AssertionError("Outer OOF predictions are incomplete")
    nonreference = ~reference
    primary_metrics = evaluate_predictions(target[nonreference], oof[nonreference], beta=beta)
    all_metrics = evaluate_predictions(target, oof, beta=beta)
    calibration = calibration_rows(target[nonreference], oof[nonreference])
    for row in calibration:
        row.update({"dataset": dataset, "pair_id": pair_id, "evaluation_rows": "nonreference"})

    frame["outer_fold"] = outer_vector + 1
    frame["selected_config_id"] = selected_config_ids
    frame["predicted_advantage_oof"] = oof
    frame.to_csv(output_dir / "dev_oof_predictions.csv.gz", index=False, compression="gzip")
    write_csv(output_dir / "outer_fold_selections.csv", outer_selection_rows)
    write_csv(output_dir / "inner_search_results.csv", inner_result_rows)
    metrics_payload = {
        **input_audit,
        "dry_run": False,
        "device": device,
        "primary_evaluation_rows": "nonreference_actions_only",
        "primary_metrics": primary_metrics,
        "supplemental_all_action_metrics": all_metrics,
        "calibration": calibration,
        "go_signal_components": {
            "nonreference_spearman_gt_0": primary_metrics["spearman"] > 0.0,
            "positive_auprc_lift_gt_0": primary_metrics["positive_auprc_lift"] > 0.0,
            "harmful_auprc_lift_gt_0": primary_metrics["harmful_auprc_lift"] > 0.0,
            "highest_10pct_actual_mean_advantage_gt_0": calibration[-1]["actual_mean_u"] > 0.0,
        },
        "policy_evaluation_performed": False,
        "kappa_lambda_tau_used": False,
    }
    (output_dir / "dev_oof_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8"
    )
    metric_rows = []
    for scope, values in (
        ("nonreference", primary_metrics),
        ("all_actions", all_metrics),
    ):
        metric_rows.append({"dataset": dataset, "pair_id": pair_id, "scope": scope, **values})
    write_csv(output_dir / "dev_oof_metrics.csv", metric_rows)
    write_csv(output_dir / "dev_oof_calibration.csv", calibration)
    write_calibration_svg(
        output_dir / "dev_oof_calibration.svg",
        calibration,
        dataset=dataset,
        pair_id=pair_id,
    )
    write_oof_report(
        output_dir / "dev_oof_report.md",
        dataset=dataset,
        pair_id=pair_id,
        primary=primary_metrics,
        all_actions=all_metrics,
        calibration=calibration,
    )
    print(
        f"[OK] pair={pair_id} OOF spearman={primary_metrics['spearman']:.6f} "
        f"positive_ap_lift={primary_metrics['positive_auprc_lift']:.6f} "
        f"harmful_ap_lift={primary_metrics['harmful_auprc_lift']:.6f}",
        flush=True,
    )
    return metrics_payload


def main() -> None:
    args = parse_args()
    run_nested_cv(
        Path(args.utility_table),
        Path(args.search_space),
        Path(args.output_dir),
        requested_device=args.device,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
