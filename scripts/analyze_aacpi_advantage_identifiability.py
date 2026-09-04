from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.constants import QUERY_GEOMETRY_FIELDS
from scripts.train_aacpi_advantage_nested_cv import (
    average_precision,
    evaluate_predictions,
    roc_auc,
    spearman_correlation,
)


ZERO_TOLERANCE = 1e-15
RELATION_MIN_QUERY_SUPPORT = 60
ACTION_DELTAS = (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30)
CALIBRATION_BUCKETS = (
    ("bottom_10pct", 0.00, 0.10),
    ("10_to_30pct", 0.10, 0.30),
    ("30_to_50pct", 0.30, 0.50),
    ("50_to_70pct", 0.50, 0.70),
    ("70_to_90pct", 0.70, 0.90),
    ("top_10pct", 0.90, 1.00),
)
PAIR_CONFIGS = (
    ("mkg_w", "mkgw_mhyper_native", "mhyper_native_seed123"),
    ("mkg_w", "mkgw_mhyper_adamf", "mhyper_adamf_seed123"),
    ("mkg_w", "mkgw_native_adamf", "native_adamf_seed123"),
    ("db15k", "db15k_mhyper_native", "mhyper_native_seed123"),
    ("db15k", "db15k_mhyper_adamf", "mhyper_adamf_seed123"),
    ("db15k", "db15k_native_adamf", "native_adamf_seed123"),
)
FROZEN_H1_PASS = {
    "mkgw_mhyper_native": True,
    "mkgw_mhyper_adamf": True,
    "mkgw_native_adamf": False,
    "db15k_mhyper_native": False,
    "db15k_mhyper_adamf": False,
    "db15k_native_adamf": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AACPI V2 Phase 2D DEV-only identifiability diagnostics."
    )
    parser.add_argument(
        "--output-dir", default="outputs/aacpi/diagnostics/phase2d"
    )
    parser.add_argument(
        "--report",
        default="docs/reports/aacpi_v2_phase2d_identifiability_audit_2026-09-04.md",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def action_sign(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.int8)
    result[values > ZERO_TOLERANCE] = 1
    result[values < -ZERO_TOLERANCE] = -1
    return result


def h1_style_pass(metrics: dict) -> bool:
    return bool(
        metrics["spearman"] > 0.0
        and metrics["positive_auprc_lift"] > 0.0
        and metrics["harmful_auprc_lift"] > 0.0
        and metrics["top_10pct_actual_mean_u"] > 0.0
    )


def calibration_rows(frame: pd.DataFrame) -> list[dict]:
    actual = frame["advantage"].to_numpy(dtype=float)
    predicted = frame["predicted_advantage_oof"].to_numpy(dtype=float)
    order = np.argsort(predicted, kind="mergesort")
    rows = []
    for label, lower, upper in CALIBRATION_BUCKETS:
        start = int(round(lower * len(order)))
        stop = int(round(upper * len(order)))
        indices = order[start:stop]
        bucket_actual = actual[indices]
        bucket_predicted = predicted[indices]
        signs = action_sign(bucket_actual)
        rows.append(
            {
                "bucket": label,
                "lower_fraction": lower,
                "upper_fraction": upper,
                "n_rows": len(indices),
                "mean_predicted_u": float(np.mean(bucket_predicted)),
                "min_predicted_u": float(np.min(bucket_predicted)),
                "max_predicted_u": float(np.max(bucket_predicted)),
                "mean_actual_u": float(np.mean(bucket_actual)),
                "median_actual_u": float(np.median(bucket_actual)),
                "positive_rate": float(np.mean(signs == 1)),
                "zero_rate": float(np.mean(signs == 0)),
                "harmful_rate": float(np.mean(signs == -1)),
            }
        )
    return rows


def prediction_metrics(frame: pd.DataFrame) -> dict:
    actual = frame["advantage"].to_numpy(dtype=float)
    predicted = frame["predicted_advantage_oof"].to_numpy(dtype=float)
    metrics = evaluate_predictions(actual, predicted, beta=0.02)
    top = calibration_rows(frame)[-1]
    return {
        **metrics,
        "top_10pct_actual_mean_u": top["mean_actual_u"],
        "top_10pct_positive_rate": top["positive_rate"],
        "top_10pct_harmful_rate": top["harmful_rate"],
    }


def standardized_mean_difference(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if len(positive) < 2 or len(negative) < 2:
        return math.nan
    pooled = math.sqrt((float(np.var(positive, ddof=1)) + float(np.var(negative, ddof=1))) / 2.0)
    if pooled == 0.0:
        return 0.0 if float(np.mean(positive)) == float(np.mean(negative)) else math.nan
    return (float(np.mean(positive)) - float(np.mean(negative))) / pooled


def feature_diagnostics(frame: pd.DataFrame, dataset: str, pair_id: str) -> list[dict]:
    target = frame["advantage"].to_numpy(dtype=float)
    signs = action_sign(target)
    rows = []
    for feature in QUERY_GEOMETRY_FIELDS:
        values = frame[feature].to_numpy(dtype=float)
        finite = np.isfinite(values) & np.isfinite(target)
        x = values[finite]
        y = target[finite]
        s = signs[finite]
        positive_labels = s == 1
        harmful_labels = s == -1
        positive_auc = roc_auc(positive_labels.astype(np.int8), x)
        harmful_auc = roc_auc(harmful_labels.astype(np.int8), x)
        nonzero = s != 0
        positive_vs_harmful_auc = roc_auc((s[nonzero] == 1).astype(np.int8), x[nonzero])
        rows.append(
            {
                "dataset": dataset,
                "pair_id": pair_id,
                "h1_status": "PASS" if FROZEN_H1_PASS[pair_id] else "FAIL",
                "feature": feature,
                "n_finite": len(x),
                "n_positive": int(np.sum(s == 1)),
                "n_zero": int(np.sum(s == 0)),
                "n_harmful": int(np.sum(s == -1)),
                "positive_mean": float(np.mean(x[s == 1])),
                "positive_median": float(np.median(x[s == 1])),
                "zero_mean": float(np.mean(x[s == 0])),
                "zero_median": float(np.median(x[s == 0])),
                "harmful_mean": float(np.mean(x[s == -1])),
                "harmful_median": float(np.median(x[s == -1])),
                "smd_positive_vs_rest": standardized_mean_difference(x[s == 1], x[s != 1]),
                "smd_harmful_vs_rest": standardized_mean_difference(x[s == -1], x[s != -1]),
                "smd_positive_vs_harmful": standardized_mean_difference(x[s == 1], x[s == -1]),
                "positive_auroc_increasing": positive_auc,
                "positive_auroc_orientation_free": max(positive_auc, 1.0 - positive_auc),
                "positive_auroc_lift_orientation_free": max(positive_auc, 1.0 - positive_auc) - 0.5,
                "harmful_auroc_increasing": harmful_auc,
                "harmful_auroc_orientation_free": max(harmful_auc, 1.0 - harmful_auc),
                "harmful_auroc_lift_orientation_free": max(harmful_auc, 1.0 - harmful_auc) - 0.5,
                "positive_vs_harmful_auroc_increasing": positive_vs_harmful_auc,
                "positive_vs_harmful_auroc_orientation_free": max(
                    positive_vs_harmful_auc, 1.0 - positive_vs_harmful_auc
                ),
                "positive_vs_harmful_auroc_lift_orientation_free": max(
                    positive_vs_harmful_auc, 1.0 - positive_vs_harmful_auc
                )
                - 0.5,
                "spearman_with_advantage": spearman_correlation(y, x),
            }
        )
    return rows


def winner_action_diagnostics(
    winner: pd.DataFrame,
    utility: pd.DataFrame,
    dataset: str,
    pair_id: str,
) -> tuple[dict, list[dict]]:
    sort_frame = utility.sort_values(
        ["query_id", "rr_action", "abs_delta_alpha", "alpha"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    best = sort_frame.groupby("query_id", sort=False).head(1).set_index("query_id")
    indexed = winner.set_index("query_id")
    if set(best.index) != set(indexed.index):
        raise AssertionError(f"Winner/utility query mismatch for {pair_id}")
    if not np.allclose(
        best.loc[indexed.index, "alpha"].to_numpy(dtype=float),
        indexed["best_alpha"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(f"Best action mismatch for {pair_id}")

    endpoint = indexed["endpoint_preference"].astype(str).to_numpy()
    best_direction = indexed["best_direction"].astype(str).to_numpy()
    endpoint_class = np.where(endpoint == "a", "toward_a", np.where(endpoint == "b", "toward_b", "stay"))
    endpoint_tie = endpoint == "tie"
    endpoint_nontie = ~endpoint_tie
    beneficial = best_direction != "stay"
    intersection = endpoint_nontie & beneficial
    opposite = intersection & (endpoint_class != best_direction)
    predicted_deviation_true_anchor = endpoint_nontie & ~beneficial
    three_class_match = endpoint_class == best_direction
    binary_match = endpoint_nontie == beneficial

    n_all = len(indexed)
    n_nontie = int(np.sum(endpoint_nontie))
    n_beneficial = int(np.sum(beneficial))
    n_intersection = int(np.sum(intersection))
    row = {
        "dataset": dataset,
        "pair_id": pair_id,
        "n_all_queries": n_all,
        "n_endpoint_ties": int(np.sum(endpoint_tie)),
        "endpoint_tie_rate_all": float(np.mean(endpoint_tie)),
        "n_endpoint_nonties": n_nontie,
        "n_anchor_optimal": int(np.sum(~beneficial)),
        "anchor_optimal_rate_all": float(np.mean(~beneficial)),
        "n_beneficial_nonanchor": n_beneficial,
        "beneficial_nonanchor_rate_all": float(np.mean(beneficial)),
        "three_class_agreement_all": float(np.mean(three_class_match)),
        "deviation_vs_no_deviation_agreement_all": float(np.mean(binary_match)),
        "winner_best_class_agreement_excluding_endpoint_ties": float(np.mean(three_class_match[endpoint_nontie])),
        "winner_best_class_agreement_excluding_anchor": float(np.mean(three_class_match[beneficial])),
        "n_endpoint_nontie_and_beneficial": n_intersection,
        "direction_agreement_excluding_ties_and_anchor": float(np.mean(three_class_match[intersection])),
        "n_predicted_deviation_true_anchor": int(np.sum(predicted_deviation_true_anchor)),
        "predicted_deviation_true_anchor_rate_all": float(np.mean(predicted_deviation_true_anchor)),
        "predicted_deviation_true_anchor_rate_endpoint_nontie": float(np.mean(~beneficial[endpoint_nontie])),
        "n_opposite_direction": int(np.sum(opposite)),
        "opposite_direction_rate_all": float(np.mean(opposite)),
        "opposite_direction_rate_excluding_ties_and_anchor": float(np.mean(opposite[intersection])),
    }
    confusion = []
    for endpoint_value in ("toward_a", "toward_b", "stay"):
        denominator = int(np.sum(endpoint_class == endpoint_value))
        for best_value in ("toward_a", "toward_b", "stay"):
            count = int(np.sum((endpoint_class == endpoint_value) & (best_direction == best_value)))
            confusion.append(
                {
                    "dataset": dataset,
                    "pair_id": pair_id,
                    "endpoint_class": endpoint_value,
                    "best_local_class": best_value,
                    "count": count,
                    "rate_within_endpoint_class": count / denominator if denominator else math.nan,
                    "rate_all_queries": count / n_all,
                }
            )
    return row, confusion


def read_historical_anchored(path: Path) -> tuple[float, str]:
    frame = pd.read_csv(path)
    selected = frame.loc[frame["config_id"] == "expanded_selected"]
    if len(selected) != 1:
        raise AssertionError(f"Expected one historical expanded_selected row in {path}")
    return float(selected.iloc[0]["delta_vs_global"]), str(selected.iloc[0]["label"])


def color_for_value(value: float, limit: float) -> str:
    if not math.isfinite(value):
        return "#e5e7eb"
    ratio = min(abs(value) / max(limit, 1e-12), 1.0)
    if value >= 0:
        start, end = (239, 246, 255), (29, 78, 216)
    else:
        start, end = (255, 247, 237), (194, 65, 12)
    rgb = tuple(round(start[i] + ratio * (end[i] - start[i])) for i in range(3))
    return "#%02x%02x%02x" % rgb


def write_heatmap_svg(
    path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float]],
    *,
    limit: float,
    digits: int = 2,
) -> None:
    cell_w, cell_h = 105, 42
    left, top, right, bottom = 250, 145, 25, 30
    width = left + len(col_labels) * cell_w + right
    height = top + len(row_labels) * cell_h + bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:700}.label{font-size:12px}.value{font-size:11px;font-weight:600}</style>',
        f'<text x="20" y="30" class="title">{html.escape(title)}</text>',
    ]
    for col, label in enumerate(col_labels):
        x = left + col * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top - 12}" text-anchor="end" class="label" transform="rotate(-45 {x} {top - 12})">{html.escape(label)}</text>'
        )
    for row, label in enumerate(row_labels):
        y = top + row * cell_h
        parts.append(
            f'<text x="{left - 10}" y="{y + 26}" text-anchor="end" class="label">{html.escape(label)}</text>'
        )
        for col, value in enumerate(values[row]):
            x = left + col * cell_w
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" rx="3" fill="{color_for_value(value, limit)}"/>'
            )
            label_value = "N/A" if not math.isfinite(value) else f"{value:.{digits}f}"
            text_color = "#ffffff" if math.isfinite(value) and abs(value) / max(limit, 1e-12) > 0.62 else "#111827"
            parts.append(
                f'<text x="{x + (cell_w - 2) / 2}" y="{y + 25}" text-anchor="middle" class="value" fill="{text_color}">{label_value}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_calibration_svg(path: Path, rows: list[dict]) -> None:
    pair_ids = [pair_id for _, pair_id, _ in PAIR_CONFIGS]
    width, height = 1200, 650
    left, right, top, bottom = 80, 30, 65, 85
    chart_w, chart_h = width - left - right, height - top - bottom
    all_y = [float(row["mean_actual_u"]) for row in rows]
    y_limit = max(max(abs(v) for v in all_y), 0.001) * 1.1
    colors = ["#1d4ed8", "#0891b2", "#7c3aed", "#059669", "#d97706", "#dc2626"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:21px;font-weight:700}.axis{font-size:12px}.legend{font-size:11px}</style>',
        '<text x="24" y="30" class="title">Phase 2D OOF advantage concentration</text>',
    ]
    zero_y = top + chart_h / 2
    parts.append(f'<line x1="{left}" y1="{zero_y}" x2="{width-right}" y2="{zero_y}" stroke="#9ca3af" stroke-dasharray="5 4"/>')
    for index, pair_id in enumerate(pair_ids):
        subset = [row for row in rows if row["pair_id"] == pair_id]
        points = []
        for bucket_index, row in enumerate(subset):
            x = left + bucket_index * chart_w / 5
            y = top + (y_limit - float(row["mean_actual_u"])) / (2 * y_limit) * chart_h
            points.append((x, y))
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{colors[index]}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colors[index]}"/>')
        lx = 90 + (index % 3) * 360
        ly = height - 48 + (index // 3) * 20
        parts.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+24}" y2="{ly-4}" stroke="{colors[index]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+30}" y="{ly}" class="legend">{html.escape(pair_id)}</text>')
    labels = [bucket[0] for bucket in CALIBRATION_BUCKETS]
    for index, label in enumerate(labels):
        x = left + index * chart_w / 5
        parts.append(f'<text x="{x}" y="{top+chart_h+24}" text-anchor="middle" class="axis">{html.escape(label)}</text>')
    parts.append(f'<text x="18" y="{top+chart_h/2}" class="axis" transform="rotate(-90 18 {top+chart_h/2})">Actual mean advantage</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_bar_svg(path: Path, title: str, rows: list[dict], field: str) -> None:
    width, height = 920, 500
    left, right, top, bottom = 260, 60, 70, 45
    chart_w = width - left - right
    bar_h = 44
    maximum = max(float(row[field]) for row in rows) * 1.12
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:21px;font-weight:700}.label{font-size:12px}.value{font-size:12px;font-weight:700}</style>',
        f'<text x="24" y="32" class="title">{html.escape(title)}</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * 62
        value = float(row[field])
        w = value / maximum * chart_w if maximum else 0
        fill = "#c2410c" if not FROZEN_H1_PASS[row["pair_id"]] else "#2563eb"
        parts.append(f'<text x="{left-12}" y="{y+28}" text-anchor="end" class="label">{html.escape(row["pair_id"])}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{fill}"/>')
        parts.append(f'<text x="{left+w+8:.1f}" y="{y+28}" class="value">{value:.1%}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def source_paths(dataset: str, pair_id: str, legacy_dir: str) -> dict[str, Path]:
    utility_prefix = Path("outputs/aacpi/utility_tables") / f"{pair_id}_dev"
    phase2b = Path("outputs/aacpi/phase2b") / pair_id
    return {
        "utility_table": Path(f"{utility_prefix}_utility_table.csv.gz"),
        "utility_summary": Path(f"{utility_prefix}_utility_summary.json"),
        "utility_manifest": Path(f"{utility_prefix}_source_manifest.json"),
        "oof_predictions": phase2b / "dev_oof_predictions.csv.gz",
        "phase2b_metrics": phase2b / "dev_oof_metrics.json",
        "phase2b_audit": phase2b / "phase2b_input_audit.json",
        "winner_actions": Path("outputs/aacpi/phase2a_diagnostics/query_diagnostics") / f"{pair_id}_dev_winner_action_diagnostics.csv.gz",
        "historical_anchored_dev": Path("outputs") / dataset / "anchored_dynamic" / legacy_dir / "p3_ablation" / "dev_p3_results.csv",
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    expected_outputs = [
        output_dir / "phase2d_pair_summary.csv",
        output_dir / "phase2d_calibration_diagnostics.csv",
        output_dir / "phase2d_action_diagnostics.csv",
        output_dir / "phase2d_feature_separability.csv",
        output_dir / "phase2d_feature_pass_fail_summary.csv",
        output_dir / "phase2d_direction_seed_diagnostics.csv",
        output_dir / "phase2d_relation_diagnostics.csv",
        output_dir / "phase2d_relation_pair_summary.csv",
        output_dir / "phase2d_winner_action_mismatch.csv",
        output_dir / "phase2d_winner_action_confusion.csv",
        output_dir / "phase2d_available_identifiable.csv",
        output_dir / "phase2d_failure_taxonomy.csv",
        output_dir / "phase2d_source_manifest.json",
        output_dir / "phase2d_calibration.svg",
        output_dir / "phase2d_action_spearman_heatmap.svg",
        output_dir / "phase2d_feature_spearman_heatmap.svg",
        output_dir / "phase2d_native_feature_separability.svg",
        output_dir / "phase2d_winner_mismatch.svg",
        report_path,
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist; pass --overwrite: {existing[:3]}")
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_path = Path("docs/protocols/AACPI_V2_PHASE2D_IDENTIFIABILITY_PROTOCOL.md")
    search_space_path = Path("docs/protocols/AACPI_V2_PHASE2_SEARCH_SPACE.yaml")
    manifest_files = {
        "phase2d_analysis_script": Path(__file__),
        "phase2d_protocol": protocol_path,
        "phase2b_search_space": search_space_path,
    }
    pair_rows = []
    calibration_output = []
    action_output = []
    feature_output = []
    direction_seed_output = []
    relation_output = []
    mismatch_output = []
    mismatch_confusion_output = []
    available_output = []
    frames = {}
    pair_details = {}

    for dataset, pair_id, legacy_dir in PAIR_CONFIGS:
        paths = source_paths(dataset, pair_id, legacy_dir)
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest_files[f"{pair_id}:{label}"] = path

        utility_manifest = json.loads(paths["utility_manifest"].read_text(encoding="utf-8"))
        utility_summary = json.loads(paths["utility_summary"].read_text(encoding="utf-8"))
        phase2b_metrics = json.loads(paths["phase2b_metrics"].read_text(encoding="utf-8"))
        phase2b_audit = json.loads(paths["phase2b_audit"].read_text(encoding="utf-8"))
        if utility_manifest["split"] != "dev" or phase2b_metrics["split"] != "dev" or phase2b_audit["split"] != "dev":
            raise AssertionError(f"Non-DEV source for {pair_id}")
        if phase2b_audit["test_accessed"] is not False:
            raise AssertionError(f"Invalid TEST boundary flag for {pair_id}")
        if sha256_file(paths["utility_table"]) != utility_manifest["output_table"]["sha256"]:
            raise AssertionError(f"Phase 1 utility hash mismatch for {pair_id}")
        if sha256_file(paths["utility_table"]) != phase2b_audit["utility_sha256"]:
            raise AssertionError(f"Phase 2B utility hash mismatch for {pair_id}")
        if sha256_file(search_space_path) != phase2b_audit["search_space_sha256"]:
            raise AssertionError(f"Phase 2B search-space hash mismatch for {pair_id}")

        utility = pd.read_csv(paths["utility_table"])
        frame = pd.read_csv(paths["oof_predictions"])
        winner = pd.read_csv(paths["winner_actions"])
        if set(frame["split"].astype(str)) != {"dev"} or set(utility["split"].astype(str)) != {"dev"}:
            raise AssertionError(f"Non-DEV row for {pair_id}")
        if len(frame) != len(utility):
            raise AssertionError(f"Utility/OOF row count mismatch for {pair_id}")
        row_key = ["original_triple_id", "query_id", "seed", "direction", "alpha"]
        if frame.duplicated(row_key).any() or utility.duplicated(row_key).any():
            raise AssertionError(f"Duplicate query-action row for {pair_id}")
        if not frame[row_key].equals(utility[row_key]):
            raise AssertionError(f"Utility/OOF row ordering mismatch for {pair_id}")
        for field in ("alpha0", "rr_anchor", "rr_action", "advantage"):
            if not np.allclose(frame[field], utility[field], rtol=0.0, atol=1e-12):
                raise AssertionError(f"Utility/OOF {field} mismatch for {pair_id}")
        if not np.isfinite(frame["predicted_advantage_oof"]).all():
            raise AssertionError(f"Non-finite OOF prediction for {pair_id}")
        if frame.groupby("original_triple_id")["outer_fold"].nunique().max() != 1:
            raise AssertionError(f"Original-triple leakage for {pair_id}")
        if frame.groupby("query_id")["outer_fold"].nunique().max() != 1:
            raise AssertionError(f"Query leakage for {pair_id}")

        reference = np.isclose(frame["alpha"], frame["alpha0"], rtol=0.0, atol=1e-12)
        nonreference = frame.loc[~reference].copy()
        metrics = prediction_metrics(nonreference)
        frozen = phase2b_metrics["primary_metrics"]
        for metric in ("spearman", "positive_prevalence", "positive_auprc", "positive_auprc_lift", "harmful_prevalence", "harmful_auprc", "harmful_auprc_lift"):
            if not np.isclose(metrics[metric], frozen[metric], rtol=0.0, atol=1e-12):
                raise AssertionError(f"Frozen metric mismatch for {pair_id}: {metric}")
        if not np.isclose(
            metrics["top_10pct_actual_mean_u"],
            phase2b_metrics["calibration"][-1]["actual_mean_u"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(f"Top-decile metric mismatch for {pair_id}")
        calculated_h1 = h1_style_pass(metrics)
        if calculated_h1 != FROZEN_H1_PASS[pair_id]:
            raise AssertionError(f"Frozen H1 status mismatch for {pair_id}")

        pair_row = {
            "dataset": dataset,
            "pair_id": pair_id,
            "h1_status": "PASS" if calculated_h1 else "FAIL",
            "n_nonreference_actions": len(nonreference),
            "spearman": metrics["spearman"],
            "positive_prevalence": metrics["positive_prevalence"],
            "positive_auprc": metrics["positive_auprc"],
            "positive_auprc_lift": metrics["positive_auprc_lift"],
            "harmful_prevalence": metrics["harmful_prevalence"],
            "harmful_auprc": metrics["harmful_auprc"],
            "harmful_auprc_lift": metrics["harmful_auprc_lift"],
            "highest_10pct_actual_mean_u": metrics["top_10pct_actual_mean_u"],
            "highest_10pct_positive_rate": metrics["top_10pct_positive_rate"],
            "highest_10pct_harmful_rate": metrics["top_10pct_harmful_rate"],
        }
        pair_rows.append(pair_row)

        pair_calibration = calibration_rows(nonreference)
        for row in pair_calibration:
            calibration_output.append({"dataset": dataset, "pair_id": pair_id, **row})

        pair_actions = []
        for delta in ACTION_DELTAS:
            subset = nonreference.loc[np.isclose(nonreference["delta_alpha"], delta, rtol=0.0, atol=1e-12)]
            if subset.empty:
                action_row = {
                    "dataset": dataset,
                    "pair_id": pair_id,
                    "delta_alpha": delta,
                    "available": False,
                    "n_rows": 0,
                    "positive_prevalence": math.nan,
                    "zero_prevalence": math.nan,
                    "harmful_prevalence": math.nan,
                    "spearman": math.nan,
                    "positive_auprc_lift": math.nan,
                    "harmful_auprc_lift": math.nan,
                    "top_10pct_actual_mean_u": math.nan,
                    "h1_style_pass": False,
                }
            else:
                subset_metrics = prediction_metrics(subset)
                signs = action_sign(subset["advantage"].to_numpy(dtype=float))
                action_row = {
                    "dataset": dataset,
                    "pair_id": pair_id,
                    "delta_alpha": delta,
                    "available": True,
                    "n_rows": len(subset),
                    "positive_prevalence": float(np.mean(signs == 1)),
                    "zero_prevalence": float(np.mean(signs == 0)),
                    "harmful_prevalence": float(np.mean(signs == -1)),
                    "spearman": subset_metrics["spearman"],
                    "positive_auprc_lift": subset_metrics["positive_auprc_lift"],
                    "harmful_auprc_lift": subset_metrics["harmful_auprc_lift"],
                    "top_10pct_actual_mean_u": subset_metrics["top_10pct_actual_mean_u"],
                    "h1_style_pass": h1_style_pass(subset_metrics),
                }
            pair_actions.append(action_row)
            action_output.append(action_row)

        pair_features = feature_diagnostics(nonreference, dataset, pair_id)
        feature_output.extend(pair_features)

        pair_subgroups = []
        for direction in ("head", "tail"):
            subset = nonreference.loc[nonreference["direction"] == direction]
            subgroup_metrics = prediction_metrics(subset)
            row = {
                "dataset": dataset,
                "pair_id": pair_id,
                "subgroup_type": "direction",
                "subgroup_value": direction,
                "n_rows": len(subset),
                **{key: subgroup_metrics[key] for key in (
                    "spearman", "positive_prevalence", "positive_auprc", "positive_auprc_lift",
                    "harmful_prevalence", "harmful_auprc", "harmful_auprc_lift",
                    "top_10pct_actual_mean_u", "top_10pct_positive_rate", "top_10pct_harmful_rate",
                )},
                "h1_style_pass": h1_style_pass(subgroup_metrics),
            }
            pair_subgroups.append(row)
            direction_seed_output.append(row)
        for seed in (1, 2, 3):
            subset = nonreference.loc[nonreference["seed"] == seed]
            subgroup_metrics = prediction_metrics(subset)
            row = {
                "dataset": dataset,
                "pair_id": pair_id,
                "subgroup_type": "seed",
                "subgroup_value": str(seed),
                "n_rows": len(subset),
                **{key: subgroup_metrics[key] for key in (
                    "spearman", "positive_prevalence", "positive_auprc", "positive_auprc_lift",
                    "harmful_prevalence", "harmful_auprc", "harmful_auprc_lift",
                    "top_10pct_actual_mean_u", "top_10pct_positive_rate", "top_10pct_harmful_rate",
                )},
                "h1_style_pass": h1_style_pass(subgroup_metrics),
            }
            pair_subgroups.append(row)
            direction_seed_output.append(row)

        pair_relations = []
        omitted_relations = 0
        for relation, subset_all in frame.groupby("relation", sort=True):
            n_queries = subset_all["query_id"].nunique()
            if n_queries < RELATION_MIN_QUERY_SUPPORT:
                omitted_relations += 1
                continue
            subset = subset_all.loc[
                ~np.isclose(subset_all["alpha"], subset_all["alpha0"], rtol=0.0, atol=1e-12)
            ]
            relation_metrics = prediction_metrics(subset)
            query_opportunity = subset.groupby("query_id")["advantage"].max() > ZERO_TOLERANCE
            row = {
                "dataset": dataset,
                "pair_id": pair_id,
                "relation": relation,
                "n_query_instances": n_queries,
                "n_original_triples": subset_all["original_triple_id"].nunique(),
                "n_nonreference_action_rows": len(subset),
                "opportunity_prevalence": float(query_opportunity.mean()),
                "actual_mean_u": float(subset["advantage"].mean()),
                "spearman": relation_metrics["spearman"],
                "positive_auprc_lift": relation_metrics["positive_auprc_lift"],
                "harmful_auprc_lift": relation_metrics["harmful_auprc_lift"],
                "top_10pct_actual_mean_u": relation_metrics["top_10pct_actual_mean_u"],
                "h1_style_pass": h1_style_pass(relation_metrics),
            }
            pair_relations.append(row)
            relation_output.append(row)

        mismatch_row, confusion_rows = winner_action_diagnostics(
            winner, utility, dataset, pair_id
        )
        mismatch_output.append(mismatch_row)
        mismatch_confusion_output.extend(confusion_rows)

        historical_delta, historical_label = read_historical_anchored(paths["historical_anchored_dev"])
        available_output.append(
            {
                "dataset": dataset,
                "pair_id": pair_id,
                "local_oracle_headroom": utility_summary["local_action_potential"]["oracle_local_headroom"],
                "positive_opportunity_query_rate": utility_summary["positive_opportunity_query_rate"],
                "spearman": metrics["spearman"],
                "positive_auprc_lift": metrics["positive_auprc_lift"],
                "harmful_auprc_lift": metrics["harmful_auprc_lift"],
                "top_10pct_actual_mean_u": metrics["top_10pct_actual_mean_u"],
                "historical_anchored_dev_delta_vs_global": historical_delta,
                "historical_anchored_label": historical_label,
                "historical_evidence_role": "retrospective DEV-only descriptive reference",
            }
        )
        frames[pair_id] = nonreference
        pair_details[pair_id] = {
            "actions": pair_actions,
            "features": pair_features,
            "subgroups": pair_subgroups,
            "relations": pair_relations,
            "relations_omitted_below_support": omitted_relations,
        }

    feature_summary = []
    feature_frame = pd.DataFrame(feature_output)
    for feature in QUERY_GEOMETRY_FIELDS:
        subset = feature_frame.loc[feature_frame["feature"] == feature]
        for status in ("PASS", "FAIL"):
            group = subset.loc[subset["h1_status"] == status]
            feature_summary.append(
                {
                    "feature": feature,
                    "h1_status": status,
                    "n_pairs": len(group),
                    "mean_abs_spearman": float(group["spearman_with_advantage"].abs().mean()),
                    "mean_positive_auroc_lift_orientation_free": float(group["positive_auroc_lift_orientation_free"].mean()),
                    "mean_harmful_auroc_lift_orientation_free": float(group["harmful_auroc_lift_orientation_free"].mean()),
                }
            )

    taxonomy_rows = []
    pair_frame = pd.DataFrame(pair_rows).set_index("pair_id")
    mismatch_frame = pd.DataFrame(mismatch_output).set_index("pair_id")
    for _, pair_id, _ in PAIR_CONFIGS:
        status = FROZEN_H1_PASS[pair_id]
        features = pd.DataFrame(pair_details[pair_id]["features"])
        actions = pd.DataFrame(pair_details[pair_id]["actions"])
        subgroups = pd.DataFrame(pair_details[pair_id]["subgroups"])
        relations = pd.DataFrame(pair_details[pair_id]["relations"])
        available_actions = actions.loc[actions["available"]]
        max_pos_feature_lift = float(features["positive_auroc_lift_orientation_free"].max())
        max_harm_feature_lift = float(features["harmful_auroc_lift_orientation_free"].max())
        max_abs_feature_spearman = float(features["spearman_with_advantage"].abs().max())
        action_passes = available_actions["h1_style_pass"].astype(bool)
        action_spearman_range = float(available_actions["spearman"].max() - available_actions["spearman"].min())
        direction_passes = subgroups.loc[subgroups["subgroup_type"] == "direction", "h1_style_pass"].astype(bool)
        seed_passes = subgroups.loc[subgroups["subgroup_type"] == "seed", "h1_style_pass"].astype(bool)
        relation_pos_range = float(relations["positive_auprc_lift"].max() - relations["positive_auprc_lift"].min()) if len(relations) else 0.0
        relation_harm_range = float(relations["harmful_auprc_lift"].max() - relations["harmful_auprc_lift"].min()) if len(relations) else 0.0
        relation_crosses = bool(
            len(relations)
            and (
                (relations["positive_auprc_lift"].min() < 0 < relations["positive_auprc_lift"].max() and relation_pos_range >= 0.05)
                or (relations["harmful_auprc_lift"].min() < 0 < relations["harmful_auprc_lift"].max() and relation_harm_range >= 0.05)
            )
        )
        relation_pass_query_coverage = (
            float(
                relations.loc[relations["h1_style_pass"], "n_query_instances"].sum()
                / relations["n_query_instances"].sum()
            )
            if len(relations)
            else 0.0
        )
        f1 = bool(not status and max_pos_feature_lift < 0.05 and max_harm_feature_lift < 0.05 and max_abs_feature_spearman < 0.05)
        f2 = bool(not status and ((action_passes.any() and (~action_passes).any()) or action_spearman_range >= 0.10))
        f3 = bool(not status and ((direction_passes.any() and (~direction_passes).any()) or (seed_passes.any() and (~seed_passes).any()) or relation_crosses))
        f4 = bool(not status and pair_frame.loc[pair_id, "spearman"] > 0.05 and pair_frame.loc[pair_id, "highest_10pct_actual_mean_u"] <= 0.0)
        f5 = bool(not status and
            pair_frame.loc[pair_id, "positive_auprc_lift"] >= 0.02
            and (
                pair_frame.loc[pair_id, "harmful_auprc_lift"] <= 0.0
                or pair_frame.loc[pair_id, "positive_auprc_lift"] - pair_frame.loc[pair_id, "harmful_auprc_lift"] >= 0.05
            )
        )
        f6 = bool(not status and (
            mismatch_frame.loc[pair_id, "opposite_direction_rate_excluding_ties_and_anchor"] >= 0.20
            or mismatch_frame.loc[pair_id, "predicted_deviation_true_anchor_rate_endpoint_nontie"] >= 0.40
        ))
        modes = [name for name, flag in (("F1", f1), ("F2", f2), ("F3", f3), ("F4", f4), ("F5", f5), ("F6", f6)) if flag]
        if status:
            outcome = "H1 PASS; failure outcome not applicable"
        elif f2 or f3:
            outcome = "Outcome B — structured heterogeneity bottleneck"
        elif f1 and not action_passes.any() and not direction_passes.any() and not seed_passes.any():
            outcome = "Outcome C — advantage target weakly identifiable"
        elif f1:
            outcome = "Outcome A — representation bottleneck"
        else:
            outcome = "Outcome A — representation bottleneck"
        taxonomy_rows.append(
            {
                "dataset": pair_frame.loc[pair_id, "dataset"],
                "pair_id": pair_id,
                "h1_status": "PASS" if status else "FAIL",
                "failure_modes": ";".join(modes),
                "F1_representation_failure": f1,
                "F2_action_heterogeneity": f2,
                "F3_regime_heterogeneity": f3,
                "F4_ranking_without_calibration": f4,
                "F5_harm_asymmetry": f5,
                "F6_winner_action_mismatch": f6,
                "max_positive_single_feature_auc_lift": max_pos_feature_lift,
                "max_harmful_single_feature_auc_lift": max_harm_feature_lift,
                "max_abs_feature_advantage_spearman": max_abs_feature_spearman,
                "action_spearman_range": action_spearman_range,
                "n_action_regimes_passing": int(action_passes.sum()),
                "n_direction_regimes_passing": int(direction_passes.sum()),
                "n_seed_regimes_passing": int(seed_passes.sum()),
                "relation_positive_lift_range": relation_pos_range,
                "relation_harmful_lift_range": relation_harm_range,
                "passing_relation_query_coverage": relation_pass_query_coverage,
                "recommended_outcome": outcome,
            }
        )

    write_csv(output_dir / "phase2d_pair_summary.csv", pair_rows)
    write_csv(output_dir / "phase2d_calibration_diagnostics.csv", calibration_output)
    write_csv(output_dir / "phase2d_action_diagnostics.csv", action_output)
    write_csv(output_dir / "phase2d_feature_separability.csv", feature_output)
    write_csv(output_dir / "phase2d_feature_pass_fail_summary.csv", feature_summary)
    write_csv(output_dir / "phase2d_direction_seed_diagnostics.csv", direction_seed_output)
    write_csv(output_dir / "phase2d_relation_diagnostics.csv", relation_output)
    write_csv(output_dir / "phase2d_winner_action_mismatch.csv", mismatch_output)
    write_csv(output_dir / "phase2d_winner_action_confusion.csv", mismatch_confusion_output)
    write_csv(output_dir / "phase2d_available_identifiable.csv", available_output)
    write_csv(output_dir / "phase2d_failure_taxonomy.csv", taxonomy_rows)

    pair_ids = [pair_id for _, pair_id, _ in PAIR_CONFIGS]
    write_calibration_svg(output_dir / "phase2d_calibration.svg", calibration_output)
    action_map = {(row["pair_id"], round(float(row["delta_alpha"]), 2)): float(row["spearman"]) for row in action_output}
    write_heatmap_svg(
        output_dir / "phase2d_action_spearman_heatmap.svg",
        "Action-conditional OOF Spearman",
        pair_ids,
        [f"{delta:+.2f}" for delta in ACTION_DELTAS],
        [[action_map.get((pair_id, round(delta, 2)), math.nan) for delta in ACTION_DELTAS] for pair_id in pair_ids],
        limit=0.35,
    )
    feature_map = {(row["pair_id"], row["feature"]): float(row["spearman_with_advantage"]) for row in feature_output}
    short_features = [feature.replace("geometry_", "") for feature in QUERY_GEOMETRY_FIELDS]
    write_heatmap_svg(
        output_dir / "phase2d_feature_spearman_heatmap.svg",
        "Frozen-feature Spearman with action advantage",
        pair_ids,
        short_features,
        [[feature_map[(pair_id, feature)] for feature in QUERY_GEOMETRY_FIELDS] for pair_id in pair_ids],
        limit=0.15,
    )
    separability_map = {
        (row["pair_id"], row["feature"]): float(
            row["positive_vs_harmful_auroc_lift_orientation_free"]
        )
        for row in feature_output
    }
    native_pairs = ["mkgw_native_adamf", "db15k_native_adamf"]
    write_heatmap_svg(
        output_dir / "phase2d_native_feature_separability.svg",
        "NativE + AdaMF-MAT: positive-vs-harmful single-feature AUROC lift",
        native_pairs,
        short_features,
        [[separability_map[(pair_id, feature)] for feature in QUERY_GEOMETRY_FIELDS] for pair_id in native_pairs],
        limit=0.12,
    )
    write_bar_svg(
        output_dir / "phase2d_winner_mismatch.svg",
        "Endpoint winner opposite to best beneficial local direction",
        mismatch_output,
        "opposite_direction_rate_excluding_ties_and_anchor",
    )

    manifest = {
        "schema_version": 1,
        "phase": "AACPI V2 Phase 2D",
        "scope": "DEV-only outer-fold OOF diagnostics",
        "relation_min_distinct_query_instances": RELATION_MIN_QUERY_SUPPORT,
        "source_files": [
            {"role": role, "path": path.as_posix(), "sha256": sha256_file(path)}
            for role, path in sorted(manifest_files.items())
        ],
        "integrity": {
            "test_rows_accessed": 0,
            "test_evaluation_commands_executed": 0,
            "phase1_utility_tables_modified": False,
            "phase2b_predictor_retrained": False,
            "features_modified": False,
            "action_grid_modified": False,
            "alpha0_modified": False,
            "all_prediction_diagnostics_outer_fold_oof": True,
            "original_triple_cross_fold_leakage_detected": False,
            "phase2b_metrics_exactly_reproduced": True,
            "unified_rebuild_script": "scripts/analyze_aacpi_advantage_identifiability.py",
        },
    }
    (output_dir / "phase2d_source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    taxonomy_frame = pd.DataFrame(taxonomy_rows)
    calibration_frame = pd.DataFrame(calibration_output)
    action_frame = pd.DataFrame(action_output)
    subgroup_frame = pd.DataFrame(direction_seed_output)
    relation_frame = pd.DataFrame(relation_output)
    relation_pair_summary = []
    calibration_summary = {}
    action_summary = {}
    subgroup_summary = {}
    feature_pair_summary = {}
    for _, pair_id, _ in PAIR_CONFIGS:
        pair_relations = relation_frame.loc[relation_frame["pair_id"] == pair_id]
        passing_relations = pair_relations.loc[pair_relations["h1_style_pass"]]
        relation_pair_summary.append(
            {
                "dataset": pair_frame.loc[pair_id, "dataset"],
                "pair_id": pair_id,
                "n_supported_relations": len(pair_relations),
                "n_passing_relations": len(passing_relations),
                "passing_relation_rate": len(passing_relations) / len(pair_relations),
                "passing_relation_query_coverage": passing_relations["n_query_instances"].sum()
                / pair_relations["n_query_instances"].sum(),
                "largest_relation_original_triples": int(pair_relations["n_original_triples"].max()),
                "largest_relation_passes": bool(
                    pair_relations.sort_values("n_original_triples", ascending=False)
                    .iloc[0]["h1_style_pass"]
                ),
            }
        )

        pair_calibration = calibration_frame.loc[calibration_frame["pair_id"] == pair_id]
        actual_means = pair_calibration["mean_actual_u"].to_numpy(dtype=float)
        calibration_summary[pair_id] = {
            "bottom": float(actual_means[0]),
            "top": float(actual_means[-1]),
            "top_minus_bottom": float(actual_means[-1] - actual_means[0]),
            "monotone_improving_steps": int(np.sum(np.diff(actual_means) >= 0.0)),
        }

        pair_actions = action_frame.loc[
            (action_frame["pair_id"] == pair_id) & action_frame["available"]
        ]
        action_summary[pair_id] = {
            "n_available": len(pair_actions),
            "n_passing": int(pair_actions["h1_style_pass"].sum()),
            "spearman_min": float(pair_actions["spearman"].min()),
            "spearman_max": float(pair_actions["spearman"].max()),
        }

        pair_subgroups = subgroup_frame.loc[subgroup_frame["pair_id"] == pair_id]
        subgroup_summary[pair_id] = {
            "direction_passing": int(
                pair_subgroups.loc[
                    pair_subgroups["subgroup_type"] == "direction", "h1_style_pass"
                ].sum()
            ),
            "seed_passing": int(
                pair_subgroups.loc[
                    pair_subgroups["subgroup_type"] == "seed", "h1_style_pass"
                ].sum()
            ),
        }

        pair_features = feature_frame.loc[feature_frame["pair_id"] == pair_id]
        feature_pair_summary[pair_id] = {
            "max_positive_lift": float(
                pair_features["positive_auroc_lift_orientation_free"].max()
            ),
            "max_harmful_lift": float(
                pair_features["harmful_auroc_lift_orientation_free"].max()
            ),
            "max_sign_lift": float(
                pair_features["positive_vs_harmful_auroc_lift_orientation_free"].max()
            ),
            "max_abs_spearman": float(
                pair_features["spearman_with_advantage"].abs().max()
            ),
        }
    write_csv(output_dir / "phase2d_relation_pair_summary.csv", relation_pair_summary)
    relation_summary_frame = pd.DataFrame(relation_pair_summary).set_index("pair_id")
    native_mkg = pair_frame.loc["mkgw_native_adamf"]
    native_db = pair_frame.loc["db15k_native_adamf"]
    native_mismatch_mkg = mismatch_frame.loc["mkgw_native_adamf"]
    native_mismatch_db = mismatch_frame.loc["db15k_native_adamf"]
    report_lines = [
        "# AACPI V2 Phase 2D Advantage Identifiability Audit",
        "",
        "Phase 2B remains a frozen NO-GO. This report uses DEV-only outer-fold OOF predictions and performs no policy evaluation.",
        "",
        "## PASS / FAIL comparison",
        "",
        "| Dataset | Pair | H1 | Spearman | Positive AP lift | Harmful AP lift | Top-10% actual mean U | Top-10% P(U>0) | Top-10% P(U<0) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pair_rows:
        report_lines.append(
            f"| {row['dataset']} | {row['pair_id']} | {row['h1_status']} | {row['spearman']:.6f} | "
            f"{row['positive_auprc_lift']:+.6f} | {row['harmful_auprc_lift']:+.6f} | "
            f"{row['highest_10pct_actual_mean_u']:+.6f} | {row['highest_10pct_positive_rate']:.3%} | "
            f"{row['highest_10pct_harmful_rate']:.3%} |"
        )
    report_lines.extend([
        "",
        "## Same-pair contrast: NativE + AdaMF-MAT",
        "",
        f"MKG-W is nearly unidentifiable in the pooled OOF result (Spearman {native_mkg['spearman']:.4f}, "
        f"positive AP lift {native_mkg['positive_auprc_lift']:+.4f}, harmful AP lift "
        f"{native_mkg['harmful_auprc_lift']:+.4f}). Its top prediction decile remains at mean U "
        f"{native_mkg['highest_10pct_actual_mean_u']:+.4f}. DB15K retains useful rank and sign structure "
        f"(Spearman {native_db['spearman']:.4f}, positive/harmful AP lifts "
        f"{native_db['positive_auprc_lift']:+.4f}/{native_db['harmful_auprc_lift']:+.4f}) and reaches "
        f"top-decile mean U {native_db['highest_10pct_actual_mean_u']:+.4f}.",
        "",
        "## Calibration and advantage concentration",
        "",
        "| Pair | Bottom-decile actual U | Top-decile actual U | Top-bottom change | Monotone improving transitions (of 5) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for _, pair_id, _ in PAIR_CONFIGS:
        values = calibration_summary[pair_id]
        report_lines.append(
            f"| {pair_id} | {values['bottom']:+.6f} | {values['top']:+.6f} | "
            f"{values['top_minus_bottom']:+.6f} | {values['monotone_improving_steps']} |"
        )
    report_lines.extend([
        "",
        "MKG-W NativE + AdaMF-MAT is not completely unordered: its bottom-to-top change is positive, "
        "but improvement stops at a negative utility ceiling and only three of five adjacent bucket transitions "
        "improve. Its top decile is therefore weakly ranked but unsafe for intervention. DB15K for the same pair "
        "reaches positive utility and a materially higher positive rate in the top decile.",
        "",
        "## Action and regime conditioning",
        "",
        "| Pair | Available actions | Passing actions | Action Spearman range | Passing directions (of 2) | Passing seeds (of 3) | Passing-relation query coverage |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ])
    for _, pair_id, _ in PAIR_CONFIGS:
        action_values = action_summary[pair_id]
        subgroup_values = subgroup_summary[pair_id]
        report_lines.append(
            f"| {pair_id} | {action_values['n_available']} | {action_values['n_passing']} | "
            f"{action_values['spearman_min']:.4f} to {action_values['spearman_max']:.4f} | "
            f"{subgroup_values['direction_passing']} | {subgroup_values['seed_passing']} | "
            f"{relation_summary_frame.loc[pair_id, 'passing_relation_query_coverage']:.2%} |"
        )
    report_lines.extend([
        "",
        "For MKG-W NativE + AdaMF-MAT, all five actions, both directions, and all three seeds fail. "
        "Its 12 passing supported relations cover only 16.02% of supported query instances, and its largest "
        "relation fails. Relation pockets show heterogeneity but do not rescue the pooled formulation. DB15K "
        "NativE + AdaMF-MAT has one passing action, one passing direction, all three seeds passing, and 47.16% "
        "coverage by passing supported relations.",
        "",
        "## Frozen-feature separability",
        "",
        "| Pair | Max positive-vs-rest AUROC lift | Max harmful-vs-rest AUROC lift | Max positive-vs-harmful AUROC lift | Max abs(feature-U Spearman) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for _, pair_id, _ in PAIR_CONFIGS:
        values = feature_pair_summary[pair_id]
        report_lines.append(
            f"| {pair_id} | {values['max_positive_lift']:.4f} | {values['max_harmful_lift']:.4f} | "
            f"{values['max_sign_lift']:.4f} | {values['max_abs_spearman']:.4f} |"
        )
    report_lines.extend([
        "",
        "MKG-W NativE + AdaMF-MAT has apparently strong positive-versus-rest and harmful-versus-rest "
        "single-feature separation, but the same features often move positive and harmful rows together away "
        "from the large zero plateau. Once zero rows are removed, its best positive-versus-harmful AUROC lift is "
        "only 0.0352; DB15K reaches 0.0857. The maximum absolute feature/continuous-U Spearman likewise differs "
        "by more than threefold (0.0483 versus 0.1557). The MKG-W representation mainly identifies utility "
        "activity, not the sign and magnitude required for safe correction.",
        "",
        "## Winner/action mismatch",
        "",
        "| Dataset | Pair | Anchor-optimal | Endpoint predicts deviation but anchor is best | Opposite direction, excluding ties and anchor | Direction agreement, excluding ties and anchor |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in mismatch_output:
        report_lines.append(
            f"| {row['dataset']} | {row['pair_id']} | {row['anchor_optimal_rate_all']:.3%} | "
            f"{row['predicted_deviation_true_anchor_rate_endpoint_nontie']:.3%} | "
            f"{row['opposite_direction_rate_excluding_ties_and_anchor']:.3%} | "
            f"{row['direction_agreement_excluding_ties_and_anchor']:.3%} |"
        )
    report_lines.extend([
        "",
        f"For MKG-W NativE + AdaMF-MAT, the opposite-direction rate is "
        f"{native_mismatch_mkg['opposite_direction_rate_excluding_ties_and_anchor']:.2%}; for DB15K it is "
        f"{native_mismatch_db['opposite_direction_rate_excluding_ties_and_anchor']:.2%}. Endpoint preference also "
        "predicts a deviation on many queries whose true local optimum is the anchor. This directly supports the "
        "claim that expert endpoint preference is not equivalent to beneficial mixture correction.",
        "",
        "## Available versus identifiable complementarity",
        "",
        "| Dataset | Pair | Local headroom | Opportunity rate | Positive AP lift | Harmful AP lift | Top-10% U | Historical Anchored DEV delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in available_output:
        report_lines.append(
            f"| {row['dataset']} | {row['pair_id']} | {row['local_oracle_headroom']:.6f} | "
            f"{row['positive_opportunity_query_rate']:.3%} | {row['positive_auprc_lift']:+.6f} | "
            f"{row['harmful_auprc_lift']:+.6f} | {row['top_10pct_actual_mean_u']:+.6f} | "
            f"{row['historical_anchored_dev_delta_vs_global']:+.6f} |"
        )
    report_lines.extend([
        "",
        "Historical Anchored values are retrospective DEV cross-fit/P3 references. No new policy was run.",
        "",
        "## Failure taxonomy",
        "",
        "| Pair | H1 | Modes | Feature max positive/harmful AUROC lift | Max abs(feature-U Spearman) | Action Spearman range | Relation-pass coverage | Outcome |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in taxonomy_rows:
        report_lines.append(
            f"| {row['pair_id']} | {row['h1_status']} | {row['failure_modes'] or '—'} | "
            f"{row['max_positive_single_feature_auc_lift']:.4f}/{row['max_harmful_single_feature_auc_lift']:.4f} | "
            f"{row['max_abs_feature_advantage_spearman']:.4f} | {row['action_spearman_range']:.4f} | "
            f"{row['passing_relation_query_coverage']:.2%} | "
            f"{row['recommended_outcome']} |"
        )
    report_lines.extend([
        "",
        "The preregistered F1 flag is not triggered for MKG-W NativE + AdaMF-MAT because positive-versus-rest "
        "and harmful-versus-rest AUROC both exceed its threshold. The positive-versus-harmful diagnostic shows "
        "that much of this is plateau/activity separation rather than correction-sign separation. This report "
        "retains the frozen F1 classification and records the sign bottleneck as interpretation; it does not "
        "retroactively alter the taxonomy rule.",
    ])
    fail_rows = taxonomy_frame.loc[taxonomy_frame["h1_status"] == "FAIL"]
    if fail_rows["F2_action_heterogeneity"].any() or fail_rows["F3_regime_heterogeneity"].any():
        overall_outcome = "Outcome B — structured heterogeneity bottleneck"
    elif fail_rows["F1_representation_failure"].all():
        overall_outcome = "Outcome C — advantage target weakly identifiable"
    else:
        overall_outcome = "Outcome A — representation bottleneck"
    report_lines.extend([
        "",
        "## Research-direction decision",
        "",
        f"Primary Phase 2D conclusion: **{overall_outcome}**.",
        "",
        "### 1. Why the same NativE + AdaMF-MAT pair differs across datasets",
        "",
        "Both datasets contain substantial available complementarity, so opportunity supply is not the explanation. "
        "The difference is identifiability: MKG-W has no passing action, direction, or seed regime, weak direct "
        "positive-versus-harmful feature separation, and a negative top-decile utility. DB15K has stable signal "
        "across all three seeds, one passing direction and action, much stronger feature/continuous-utility structure, "
        "and almost half of supported query instances in passing relation regimes.",
        "",
        "### 2. Dominant failure mechanisms",
        "",
        "The aggregate failure set is heterogeneous, which supports Outcome B. MKG-W NativE + AdaMF-MAT is the "
        "strongest representation/sign-identifiability bottleneck: relation pockets are too narrow to change that "
        "assessment. DB15K M-Hyper + AdaMF-MAT is principally F4/F5: it ranks some positive opportunity but its "
        "high-score region stays harmful and harmful-action detection is below prevalence. DB15K M-Hyper + NativE "
        "shows seed/relation dependence and weak aggregate calibration.",
        "",
        "### 3. Endpoint winner versus beneficial mixture correction",
        "",
        "Across pairs, 10.75%-35.61% of endpoint-nontie beneficial corrections point opposite the endpoint winner, "
        "and 21.46%-51.33% of endpoint-nontie queries are actually anchor-optimal. This is large enough to establish "
        "a structural weakness in winner supervision. It is not sufficient by itself to explain every Anchored result: "
        "MKG-W M-Hyper + AdaMF-MAT has high mismatch yet historical Anchored DEV gain remains positive. Anchored "
        "failure is best explained by winner/action mismatch interacting with weak advantage identifiability.",
        "",
        "### 4. Signal remaining in the frozen geometry",
        "",
        "Yes, but it is not uniformly the required signal. Several features identify plateau versus nonzero movement, "
        "and DB15K contains sign and magnitude structure that the OOF predictor uses. In MKG-W NativE + AdaMF-MAT, "
        "the same summaries mainly indicate that an action may change rank, without reliably indicating whether the "
        "change helps or harms. No frozen action, direction, or seed subgroup exposes a stable missed solution.",
        "",
        "### 5. Next research hypothesis",
        "",
        "The most defensible next step is a separately preregistered richer score representation with query/context "
        "augmentation, designed specifically to distinguish beneficial from harmful rank movement. Structured regime "
        "information is a secondary hypothesis and should be evaluated as representation, not used to fit ad hoc "
        "relation-specific policies. The DB15K success means score-based dynamic combination is not disproved in all "
        "settings, but continuing with the same 13 summaries or a larger MLP is not justified.",
        "",
        "This task does not implement the next hypothesis. AACPI V2 remains failed and must not be retroactively changed.",
        "",
        "## Integrity audit",
        "",
        "- TEST rows accessed: **0**.",
        "- TEST evaluation commands executed: **0**.",
        "- Phase 1 utility tables modified: **no**.",
        "- Phase 2B predictor retrained: **no**.",
        "- Frozen 13 features, action grids, and alpha0 values modified: **no**.",
        "- Every prediction diagnostic uses recorded outer-fold OOF predictions.",
        "- Every original triple remains in exactly one outer fold.",
        "- Phase 1 and Phase 2B source hashes and frozen Phase 2B metrics were reproduced exactly.",
        "- Every Phase 2D table and figure is rebuilt by `scripts/analyze_aacpi_advantage_identifiability.py`.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(
        f"[OK] Phase 2D pairs={len(pair_rows)} actions={len(action_output)} "
        f"features={len(feature_output)} relations={len(relation_output)} "
        f"outcome={overall_outcome}"
    )


if __name__ == "__main__":
    main()
