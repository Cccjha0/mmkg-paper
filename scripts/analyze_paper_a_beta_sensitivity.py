from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.constants import QUERY_GEOMETRY_FIELDS  # noqa: E402
from scripts.analyze_paper_a_risk_coverage import (  # noqa: E402
    PAIR_SPECS,
    alpha_column,
    load_pair_split,
    map_to_grid,
    portable_path,
    rr_for_alpha,
)


FORMAL_BETA_GRID = tuple(value / 100.0 for value in range(5, 51, 5))
DIAGNOSTIC_BETA = 1.0
BETA_GRID = (*FORMAL_BETA_GRID, DIAGNOSTIC_BETA)
UNCHANGED_TOLERANCE = 1e-12

SUMMARY_COLUMNS = (
    "split",
    "dataset",
    "pair",
    "beta",
    "beta_role",
    "is_dev_locked_beta",
    "n_observations",
    "mrr",
    "global_mrr",
    "delta_mrr_vs_global",
    "harmful_query_rate",
    "mean_harm",
    "beneficial_query_rate",
    "changed_alpha_rate",
    "fallback_rate",
    "mean_abs_alpha_deviation",
    "p95_abs_alpha_deviation",
    "saturation_rate",
    "n_harmful",
    "n_beneficial",
    "n_unchanged",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the fixed Paper A Anchored model over a predeclared beta "
            "trust-region sensitivity grid. No model is trained or selected."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/beta_sensitivity",
    )
    parser.add_argument(
        "--report-path",
        default="docs/reports/paper_a_beta_trust_region_report.md",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_root(repo_root: Path, spec: dict[str, str]) -> Path:
    return (
        repo_root
        / "outputs"
        / spec["dataset_dir"]
        / "anchored_dynamic"
        / f"{spec['pair_dir']}_seed123"
    )


def rows_path(repo_root: Path, spec: dict[str, str], split: str) -> Path:
    root = pair_root(repo_root, spec)
    if split == "dev":
        return root / "anchored_crossfit/dev_anchored_query_rows.csv"
    return root / "test_anchored/test_locked_query_rows.csv"


def verify_theoretical_bound() -> dict[str, Any]:
    rng = np.random.default_rng(20260910)
    n = 100_000
    score_primary = rng.normal(size=n)
    score_secondary = rng.normal(size=n)
    alpha0 = rng.uniform(0.0, 1.0, size=n)
    beta = rng.uniform(0.0, 1.0, size=n)
    decision = rng.normal(size=n)
    alpha = np.clip(alpha0 + beta * np.tanh(decision), 0.0, 1.0)
    score_alpha = (1.0 - alpha) * score_primary + alpha * score_secondary
    score_anchor = (
        (1.0 - alpha0) * score_primary + alpha0 * score_secondary
    )
    identity_rhs = (alpha - alpha0) * (score_secondary - score_primary)
    bound_rhs = beta * np.abs(score_secondary - score_primary)
    identity_error = np.abs((score_alpha - score_anchor) - identity_rhs)
    bound_violation = np.abs(score_alpha - score_anchor) - bound_rhs
    trust_region_violation = np.abs(alpha - alpha0) - beta
    tolerance = 1e-12
    passed = (
        float(identity_error.max()) <= tolerance
        and float(bound_violation.max()) <= tolerance
        and float(trust_region_violation.max()) <= tolerance
    )
    if not passed:
        raise RuntimeError("Numerical check of the beta trust-region bound failed")
    return {
        "deterministic_seed": 20260910,
        "n_random_checks": n,
        "identity_max_abs_error": float(identity_error.max()),
        "bound_max_signed_violation": float(bound_violation.max()),
        "bound_max_positive_violation": max(0.0, float(bound_violation.max())),
        "trust_region_max_signed_violation": float(trust_region_violation.max()),
        "trust_region_max_positive_violation": max(
            0.0, float(trust_region_violation.max())
        ),
        "tolerance": tolerance,
        "passed": True,
    }


def load_state(
    repo_root: Path,
    spec: dict[str, str],
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Reuse the risk-coverage loader to verify feature schema, confidence,
    # stored fallback/policy, exact alpha-grid RR, lock identity, and split.
    _, source_audit = load_pair_split(repo_root, spec, split)
    root = pair_root(repo_root, spec)
    path = rows_path(repo_root, spec, split)
    lock_path = root / "dev_lock/anchored_dev_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(path)
    alphas = tuple(float(value) for value in lock["alpha_grid"])
    required = {
        "query_id",
        "seed",
        "direction",
        "split",
        "anchored_decision",
        "anchored_probability_a",
        *QUERY_GEOMETRY_FIELDS,
        *(alpha_column(alpha) for alpha in alphas),
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing beta-sensitivity columns in {path}: {sorted(missing)}")
    decision = frame["anchored_decision"].to_numpy(dtype=np.float64)
    probability = frame["anchored_probability_a"].to_numpy(dtype=np.float64)
    geometry = frame[list(QUERY_GEOMETRY_FIELDS)].to_numpy(dtype=np.float64)
    nonfinite = (
        ~np.isfinite(geometry).all(axis=1)
        | ~np.isfinite(decision)
        | ~np.isfinite(probability)
    )
    confidence = np.abs(2.0 * probability - 1.0)
    alpha0 = float(lock["alpha0"])
    threshold = float(lock["confidence_threshold"])
    if not any(math.isclose(alpha0, alpha, abs_tol=1e-12) for alpha in alphas):
        raise RuntimeError(f"alpha0 is absent from the exact grid: {lock_path}")
    frame = frame.copy()
    frame["decision_for_sensitivity"] = decision
    frame["confidence_for_sensitivity"] = confidence
    frame["nonfinite_for_sensitivity"] = nonfinite
    frame["rr_global_for_sensitivity"] = frame[alpha_column(alpha0)].to_numpy(
        dtype=np.float64
    )
    audit = {
        **source_audit,
        "formal_beta_grid": list(FORMAL_BETA_GRID),
        "diagnostic_beta": DIAGNOSTIC_BETA,
        "dev_locked_beta": float(lock["beta"]),
        "dev_locked_alpha0": alpha0,
        "dev_locked_confidence_threshold": threshold,
        "feature_schema": list(lock["query_geometry_fields"]),
        "beta_is_only_varied_component": True,
        "test_used_for_selection": False,
    }
    return frame, audit


def apply_beta(
    frame: pd.DataFrame,
    *,
    alpha0: float,
    beta: float,
    threshold: float,
    alphas: tuple[float, ...],
) -> pd.DataFrame:
    decision = frame["decision_for_sensitivity"].to_numpy(dtype=np.float64)
    confidence = frame["confidence_for_sensitivity"].to_numpy(dtype=np.float64)
    nonfinite = frame["nonfinite_for_sensitivity"].to_numpy(dtype=bool)
    fallback = nonfinite | (confidence < threshold)
    unconstrained = alpha0 + beta * np.tanh(decision)
    continuous = np.clip(unconstrained, 0.0, 1.0)
    continuous = np.where(fallback, alpha0, continuous)
    anchors = np.full(len(frame), alpha0, dtype=np.float64)
    applied = map_to_grid(continuous, anchors, alphas)
    rr = rr_for_alpha(frame, applied, alphas)
    rr_global = frame["rr_global_for_sensitivity"].to_numpy(dtype=np.float64)
    output = pd.DataFrame(
        {
            "query_id": frame["query_id"].astype(str),
            "seed": frame["seed"].astype(int),
            "direction": frame["direction"].astype(str),
            "rr_global": rr_global,
            "rr_method": rr,
            "delta_rr": rr - rr_global,
            "fallback": fallback,
            "alpha_applied": applied,
            "abs_alpha_deviation": np.abs(applied - alpha0),
            "saturated": (unconstrained <= 0.0) | (unconstrained >= 1.0),
        }
    )
    return output


def summarize(
    state: pd.DataFrame,
    *,
    split: str,
    spec: dict[str, str],
    beta: float,
    locked_beta: float,
    seed: str | int = "pooled",
    direction: str = "pooled",
) -> dict[str, Any]:
    delta = state["delta_rr"].to_numpy(dtype=np.float64)
    harmful = delta < 0.0
    beneficial = delta > 0.0
    unchanged = ~(harmful | beneficial)
    harm = -delta[harmful]
    return {
        "split": split,
        "dataset": spec["dataset"],
        "pair": spec["pair"],
        "beta": beta,
        "beta_role": "no_local_bound_diagnostic"
        if math.isclose(beta, DIAGNOSTIC_BETA)
        else "formal_dev_candidate",
        "is_dev_locked_beta": math.isclose(beta, locked_beta, abs_tol=1e-12),
        "seed": seed,
        "direction": direction,
        "n_observations": int(len(state)),
        "mrr": float(state["rr_method"].mean()),
        "global_mrr": float(state["rr_global"].mean()),
        "delta_mrr_vs_global": float(delta.mean()),
        "harmful_query_rate": float(harmful.mean()),
        "mean_harm": float(harm.mean()) if harm.size else None,
        "beneficial_query_rate": float(beneficial.mean()),
        "changed_alpha_rate": float(
            (state["abs_alpha_deviation"] > UNCHANGED_TOLERANCE).mean()
        ),
        "fallback_rate": float(state["fallback"].mean()),
        "mean_abs_alpha_deviation": float(
            state["abs_alpha_deviation"].mean()
        ),
        "p95_abs_alpha_deviation": float(
            state["abs_alpha_deviation"].quantile(0.95)
        ),
        "saturation_rate": float(state["saturated"].mean()),
        "n_harmful": int(harmful.sum()),
        "n_beneficial": int(beneficial.sum()),
        "n_unchanged": int(unchanged.sum()),
    }


def plot_with_matplotlib(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for axis, spec in zip(axes.ravel(), PAIR_SPECS, strict=True):
        subset = summary[
            summary["dataset"].eq(spec["dataset"])
            & summary["pair"].eq(spec["pair"])
        ]
        for split, marker, style in (("dev", "o", "-"), ("test", "s", "--")):
            curve = subset[subset["split"].eq(split)].sort_values("beta")
            values = curve[metric].to_numpy(dtype=np.float64)
            if metric.endswith("rate"):
                values = 100.0 * values
            axis.plot(curve["beta"], values, marker=marker, linestyle=style, linewidth=1.5, markersize=3.5, label="Grouped OOF DEV" if split == "dev" else "Locked TEST")
        locked_beta = float(subset.loc[subset["is_dev_locked_beta"], "beta"].iloc[0])
        axis.axvline(locked_beta, linewidth=0.8, linestyle=":", label="DEV-locked beta")
        if metric == "delta_mrr_vs_global":
            axis.axhline(0.0, linewidth=0.6, color="black")
        axis.set_title(f"{spec['dataset']}: {spec['pair']}", fontsize=9)
        axis.set_xlabel("beta (1.0 = no local bound)")
        axis.set_ylabel(ylabel)
        axis.grid(True, linewidth=0.4, alpha=0.35)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_with_reportlab(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    try:
        from reportlab.graphics import renderPDF
        from reportlab.graphics.shapes import Circle, Drawing, Line, String
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError as error:
        raise RuntimeError("Figure rendering requires matplotlib or reportlab") from error

    width, height = 756.0, 518.4
    drawing = Drawing(width, height)
    dark = HexColor("#202124")
    grid = HexColor("#d5d8dc")
    dev_color = HexColor("#3568a8")
    test_color = HexColor("#c45a32")
    drawing.add(String(width / 2, height - 24, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=12, fillColor=dark))
    panel_width, panel_height = 330.0, 190.0
    origins = ((52.0, 274.0), (410.0, 274.0), (52.0, 57.0), (410.0, 57.0))
    x_ticks = (0.05, 0.20, 0.35, 0.50, 1.00)

    for (origin_x, origin_y), spec in zip(origins, PAIR_SPECS, strict=True):
        subset = summary[summary["dataset"].eq(spec["dataset"]) & summary["pair"].eq(spec["pair"])]
        metric_values = subset[metric].to_numpy(dtype=np.float64)
        if metric.endswith("rate"):
            metric_values = 100.0 * metric_values
        value_min = float(metric_values.min())
        value_max = float(metric_values.max())
        if metric == "delta_mrr_vs_global":
            value_min = min(value_min, 0.0)
            value_max = max(value_max, 0.0)
        padding = max((value_max - value_min) * 0.12, 1e-5)
        if metric == "delta_mrr_vs_global":
            y_min, y_max = value_min - padding, value_max + padding
        else:
            y_min, y_max = 0.0, value_max + padding
        plot_left = origin_x + 38.0
        plot_bottom = origin_y + 44.0
        plot_width = panel_width - 48.0
        plot_height = panel_height - 70.0
        drawing.add(String(origin_x + panel_width / 2, origin_y + panel_height - 12.0, f"{spec['dataset']}: {spec['pair']}", textAnchor="middle", fontName="Helvetica-Bold", fontSize=9, fillColor=dark))
        drawing.add(String(plot_left, plot_bottom + plot_height + 5.0, ylabel, fontName="Helvetica", fontSize=7.0, fillColor=dark))
        for tick in np.linspace(y_min, y_max, 5):
            y = plot_bottom + plot_height * (tick - y_min) / (y_max - y_min)
            drawing.add(Line(plot_left, y, plot_left + plot_width, y, strokeColor=grid, strokeWidth=0.45))
            if metric == "delta_mrr_vs_global":
                label = f"{tick:.4f}"
            elif metric == "mean_abs_alpha_deviation":
                label = f"{tick:.3f}"
            else:
                label = f"{tick:.1f}"
            drawing.add(String(plot_left - 6.0, y - 2.5, label, textAnchor="end", fontName="Helvetica", fontSize=6.5, fillColor=dark))
        drawing.add(Line(plot_left, plot_bottom, plot_left, plot_bottom + plot_height, strokeColor=dark, strokeWidth=0.7))
        drawing.add(Line(plot_left, plot_bottom, plot_left + plot_width, plot_bottom, strokeColor=dark, strokeWidth=0.7))
        if metric == "delta_mrr_vs_global" and y_min <= 0.0 <= y_max:
            zero_y = plot_bottom + plot_height * (0.0 - y_min) / (y_max - y_min)
            drawing.add(Line(plot_left, zero_y, plot_left + plot_width, zero_y, strokeColor=dark, strokeWidth=0.65))
        def x_for(beta: float) -> float:
            return plot_left + plot_width * (float(beta) - 0.05) / 0.95

        for beta in x_ticks:
            x = x_for(beta)
            drawing.add(String(x, plot_bottom - 12.0, f"{beta:.2f}" if beta < 1.0 else "1.0", textAnchor="middle", fontName="Helvetica", fontSize=6.5, fillColor=dark))
        drawing.add(String(origin_x + panel_width / 2, origin_y + 8.0, "beta (1.0 = no local bound)", textAnchor="middle", fontName="Helvetica", fontSize=7.2, fillColor=dark))
        locked_beta = float(subset.loc[subset["is_dev_locked_beta"], "beta"].iloc[0])
        locked_x = x_for(locked_beta)
        drawing.add(Line(locked_x, plot_bottom, locked_x, plot_bottom + plot_height, strokeColor=HexColor("#777777"), strokeWidth=0.8, strokeDashArray=[1.5, 2]))
        for split, color, filled in (("dev", dev_color, True), ("test", test_color, False)):
            curve = subset[subset["split"].eq(split)].sort_values("beta")
            values = curve[metric].to_numpy(dtype=np.float64)
            if metric.endswith("rate"):
                values = 100.0 * values
            points = [(x_for(beta), plot_bottom + plot_height * (value - y_min) / (y_max - y_min)) for beta, value in zip(curve["beta"], values, strict=True)]
            for (x1, y1), (x2, y2) in zip(points[:-1], points[1:], strict=True):
                drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.35, strokeDashArray=None if filled else [4, 2]))
            for x, y in points:
                drawing.add(Circle(x, y, 2.4, strokeColor=color, fillColor=color if filled else None, strokeWidth=1.0))
    legend_y = 25.0
    legend_items = (("Grouped OOF DEV", dev_color, True), ("Locked TEST", test_color, False), ("DEV-locked beta", HexColor("#777777"), False))
    total_width = sum(stringWidth(label, "Helvetica", 8) + 40.0 for label, _, _ in legend_items)
    cursor = (width - total_width) / 2.0
    for label, color, filled in legend_items:
        drawing.add(Line(cursor, legend_y, cursor + 22.0, legend_y, strokeColor=color, strokeWidth=1.35, strokeDashArray=None if filled else [3, 2]))
        if label != "DEV-locked beta":
            drawing.add(Circle(cursor + 11.0, legend_y, 2.4, strokeColor=color, fillColor=color if filled else None, strokeWidth=1.0))
        drawing.add(String(cursor + 28.0, legend_y - 2.8, label, fontName="Helvetica", fontSize=8, fillColor=dark))
        cursor += stringWidth(label, "Helvetica", 8) + 40.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(drawing, str(output_path))


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    if not plot_with_matplotlib(summary, metric, ylabel, output_path):
        plot_with_reportlab(summary, metric, ylabel, title, output_path)


def format_metric(value: Any, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def monotonic_non_decreasing(values: np.ndarray, tolerance: float = 1e-15) -> bool:
    return bool(np.all(np.diff(values) >= -tolerance))


def validate_core_test_reference(
    summary: pd.DataFrame,
    repo_root: Path,
) -> dict[str, Any]:
    path = (
        repo_root
        / "outputs/paper_a_safe_correction/core_ablation/core_ablation_summary.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    core = pd.read_csv(path)
    variant_beta = {"A_full_anchored": 0.5, "C_no_bound": 1.0}
    reference = core[
        core["split"].eq("test")
        & core["variant_id"].isin(variant_beta)
    ].copy()
    reference["beta"] = reference["variant_id"].map(variant_beta)
    current = summary[
        summary["split"].eq("test") & summary["beta"].isin((0.5, 1.0))
    ]
    merged = current.merge(
        reference,
        on=["split", "dataset", "pair", "beta"],
        suffixes=("_sensitivity", "_core"),
        validate="one_to_one",
    )
    columns = (
        "mrr",
        "delta_mrr_vs_global",
        "harmful_query_rate",
        "mean_harm",
        "beneficial_query_rate",
        "fallback_rate",
        "mean_abs_alpha_deviation",
        "p95_abs_alpha_deviation",
        "saturation_rate",
    )
    errors = {
        column: float(
            np.max(
                np.abs(
                    merged[f"{column}_sensitivity"].to_numpy(dtype=np.float64)
                    - merged[f"{column}_core"].to_numpy(dtype=np.float64)
                )
            )
        )
        for column in columns
    }
    changed_error = float(
        np.max(
            np.abs(
                merged["changed_alpha_rate"].to_numpy(dtype=np.float64)
                - merged["changed_from_anchor_rate"].to_numpy(dtype=np.float64)
            )
        )
    )
    errors["changed_alpha_rate"] = changed_error
    tolerance = 1e-12
    if len(merged) != 8 or max(errors.values()) > tolerance:
        raise RuntimeError(
            "beta=0.5/1.0 TEST sensitivity does not reproduce core ablation"
        )
    return {
        "source": portable_path(path, repo_root),
        "source_sha256": sha256_file(path),
        "matched_rows": int(len(merged)),
        "matched_variants": {
            "beta=0.5": "A_full_anchored",
            "beta=1.0": "C_no_bound",
        },
        "max_abs_errors": errors,
        "tolerance": tolerance,
        "passed": True,
    }


def write_report(
    path: Path,
    summary: pd.DataFrame,
    audits: list[dict[str, Any]],
    theory: dict[str, Any],
    core_reference: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
) -> None:
    formal = summary[summary["beta_role"].eq("formal_dev_candidate")]
    formal_groups = [
        group.sort_values("beta")
        for _, group in formal.groupby(["split", "dataset", "pair"], sort=False)
    ]
    sign_stable_groups = sum(
        bool((group["delta_mrr_vs_global"] > 0.0).all())
        for group in formal_groups
    )
    monotone_harm_groups = sum(
        monotonic_non_decreasing(
            group["harmful_query_rate"].to_numpy(dtype=np.float64)
        )
        for group in formal_groups
    )
    monotone_deviation_groups = sum(
        monotonic_non_decreasing(
            group["mean_abs_alpha_deviation"].to_numpy(dtype=np.float64)
        )
        for group in formal_groups
    )
    selected = summary[summary["is_dev_locked_beta"]].set_index(
        ["split", "dataset", "pair"]
    )
    no_bound = summary[
        summary["beta_role"].eq("no_local_bound_diagnostic")
    ].set_index(["split", "dataset", "pair"])
    common = selected.index.intersection(no_bound.index)
    no_bound_mrr_higher = sum(
        no_bound.loc[key, "delta_mrr_vs_global"]
        > selected.loc[key, "delta_mrr_vs_global"]
        for key in common
    )
    no_bound_harm_higher = sum(
        no_bound.loc[key, "harmful_query_rate"]
        > selected.loc[key, "harmful_query_rate"]
        for key in common
    )
    no_bound_severity_higher = sum(
        no_bound.loc[key, "mean_harm"] > selected.loc[key, "mean_harm"]
        for key in common
    )
    lines = [
        "# Paper A Bounded-Correction / Beta Trust-Region Report",
        "",
        "## Main answer",
        "",
        "Beta has a direct operational meaning: it bounds how far the fused score of "
        "every candidate can move away from the DEV-selected Global anchor, scaled by "
        "the candidate's inter-expert score disagreement. Empirically, all 10 formal "
        f"beta values remain net-positive in {sign_stable_groups}/8 pair/split curves, "
        "which is a wide sign-stable performance region. This stability does not imply "
        f"constant risk: Harm Rate is monotone non-decreasing in {monotone_harm_groups}/8 "
        f"curves and mean alpha deviation in {monotone_deviation_groups}/8. Relative to "
        f"the locked beta, `beta=1.0` raises Delta MRR in {no_bound_mrr_higher}/8 cases, "
        f"but also raises harm frequency in {no_bound_harm_higher}/8 and conditional Mean "
        f"Harm in {no_bound_severity_higher}/8. The local bound therefore acts as a risk "
        "budget even when a wider correction range improves mean MRR.",
        "",
        "## Theoretical derivation",
        "",
        "For primary score `s_p(q,e)`, secondary score `s_s(q,e)`, and secondary weight "
        "`alpha`, define",
        "",
        "```text",
        "s_alpha(q,e) = (1 - alpha) s_p(q,e) + alpha s_s(q,e).",
        "```",
        "",
        "Subtracting the anchor score at `alpha0` gives the exact identity",
        "",
        "```text",
        "s_alpha(q,e) - s_alpha0(q,e)",
        "  = (alpha - alpha0) [s_s(q,e) - s_p(q,e)].",
        "```",
        "",
        "The Anchored policy first proposes `alpha0 + beta*tanh(g(phi(q)))` and projects "
        "it to `[0,1]`. Because `alpha0` is itself in `[0,1]`, this projection cannot "
        "increase its distance from `alpha0`; hence `|alpha-alpha0| <= beta`. Therefore",
        "",
        "```text",
        "|s_alpha(q,e) - s_alpha0(q,e)|",
        "  <= beta |s_s(q,e) - s_p(q,e)|.",
        "```",
        "",
        "Thus beta is a score-perturbation trust-region radius, not merely a generic "
        "hyperparameter. It limits candidate-score movement when the experts disagree, "
        "while producing little movement when their scores agree. For two candidates "
        "`e1,e2`, the anchor pairwise margin changes by at most "
        "`beta * |d(e1)-d(e2)|`, where `d(e)=s_s(q,e)-s_p(q,e)`. Consequently, an anchor "
        "top candidate is guaranteed unchanged whenever every anchor margin exceeds that "
        "pairwise perturbation bound.",
        "",
        f"A deterministic numerical check over {theory['n_random_checks']:,} random score "
        f"configurations passed at tolerance {theory['tolerance']:.0e}; maximum identity "
        f"error was {theory['identity_max_abs_error']:.3e}; maximum positive score-bound "
        f"violation and trust-region violation were both "
        f"{max(theory['bound_max_positive_violation'], theory['trust_region_max_positive_violation']):.3e}.",
        "",
        "## Protocol boundary",
        "",
        "The feature model, grouped folds, alpha0, confidence threshold, exact alpha grid, "
        "and all model outputs are frozen. Only beta is varied over the predeclared formal "
        "grid `0.05, 0.10, ..., 0.50`; `beta=1.0` is labeled no-local-bound diagnostic and "
        "is excluded from formal selection. The selected beta remains the existing DEV "
        "lock. TEST curves are descriptive only and cannot select beta.",
        "",
        "## Full sensitivity table",
        "",
        "| Split | Dataset/Pair | beta | Role | Locked? | Delta MRR | Harm % | Mean harm | Benefit % | Changed % | Fallback % | Mean abs(alpha-alpha0) | P95 | Saturation % |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {str(row.split).upper()} | {row.dataset} / {row.pair} | {row.beta:.2f} | "
            f"{'no bound' if row.beta_role == 'no_local_bound_diagnostic' else 'formal grid'} | "
            f"{'yes' if row.is_dev_locked_beta else 'no'} | {row.delta_mrr_vs_global:+.6f} | "
            f"{100.0 * row.harmful_query_rate:.2f} | {format_metric(row.mean_harm)} | "
            f"{100.0 * row.beneficial_query_rate:.2f} | {100.0 * row.changed_alpha_rate:.2f} | "
            f"{100.0 * row.fallback_rate:.2f} | {row.mean_abs_alpha_deviation:.4f} | "
            f"{row.p95_abs_alpha_deviation:.4f} | {100.0 * row.saturation_rate:.2f} |"
        )
    lines += ["", "## Stability and risk interpretation", ""]
    for split in ("dev", "test"):
        for spec in PAIR_SPECS:
            curve = formal[
                formal["split"].eq(split)
                & formal["dataset"].eq(spec["dataset"])
                & formal["pair"].eq(spec["pair"])
            ].sort_values("beta")
            deltas = curve["delta_mrr_vs_global"].to_numpy(dtype=np.float64)
            harms = curve["harmful_query_rate"].to_numpy(dtype=np.float64)
            deviations = curve["mean_abs_alpha_deviation"].to_numpy(dtype=np.float64)
            positive_count = int((deltas > 0.0).sum())
            lines.append(
                f"- {split.upper()} {spec['dataset']} / {spec['pair']}: Delta MRR spans "
                f"{deltas.min():+.6f} to {deltas.max():+.6f} across the 10 formal beta "
                f"values ({positive_count}/10 positive). Harm rate is "
                f"{'monotone non-decreasing' if monotonic_non_decreasing(harms) else 'not strictly monotone'} "
                f"and moves from {100.0 * harms[0]:.2f}% to {100.0 * harms[-1]:.2f}%; "
                f"mean alpha deviation is "
                f"{'monotone non-decreasing' if monotonic_non_decreasing(deviations) else 'not strictly monotone'} "
                f"from {deviations[0]:.4f} to {deviations[-1]:.4f}."
            )
    lines += [
        "",
        "A wide sign-stable region means net MRR remains above Global over many beta "
        "values; it does not mean risk is unchanged. Harm frequency, conditional harm "
        "magnitude, and score-weight displacement must be read alongside Delta MRR. Grid "
        "rounding can create short plateaus or local non-monotonicities even though the "
        "continuous trust-region radius increases monotonically.",
        "",
        "## Figures",
        "",
        f"- `{portable_path(output_dir / 'beta_vs_delta_mrr.pdf', repo_root)}`",
        f"- `{portable_path(output_dir / 'beta_vs_harm_rate.pdf', repo_root)}`",
        f"- `{portable_path(output_dir / 'beta_vs_mean_alpha_deviation.pdf', repo_root)}`",
        "",
        "Each figure contains the four main pairs, grouped OOF DEV and locked TEST, the "
        "DEV-locked beta marker, and the separated `beta=1.0` no-local-bound endpoint.",
        "",
        "## Source and lock audit",
        "",
        "| Split | Dataset/Pair | alpha0 | Locked beta | Threshold | Observations | Source SHA-256 | Lock SHA-256 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for item in audits:
        lines.append(
            f"| {item['split'].upper()} | {item['dataset']} / {item['pair']} | "
            f"{item['dev_locked_alpha0']:.2f} | {item['dev_locked_beta']:.2f} | "
            f"{item['dev_locked_confidence_threshold']:.2f} | {item['n_observations']} | "
            f"`{item['source_rows_sha256']}` | `{item['lock_sha256']}` |"
        )
    lines += [
        "",
        "No base model or combiner was trained. Stored confidence and the original policy "
        "were reconstructed and checked before sensitivity evaluation. On locked TEST, "
        "the `beta=0.50` and `beta=1.00` rows exactly reproduce the existing Full and "
        f"No Bound core-ablation outputs across {core_reference['matched_rows']} matched "
        "pair/configuration rows.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if not report_path.is_absolute():
        report_path = repo_root / report_path
    output_dir.mkdir(parents=True, exist_ok=True)

    theory = verify_theoretical_bound()
    summaries: list[dict[str, Any]] = []
    by_seed: list[dict[str, Any]] = []
    by_direction: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        for spec in PAIR_SPECS:
            frame, source_audit = load_state(repo_root, spec, split)
            audits.append(source_audit)
            root = pair_root(repo_root, spec)
            lock = json.loads((root / "dev_lock/anchored_dev_lock.json").read_text(encoding="utf-8"))
            alpha0 = float(lock["alpha0"])
            locked_beta = float(lock["beta"])
            threshold = float(lock["confidence_threshold"])
            alphas = tuple(float(value) for value in lock["alpha_grid"])
            if tuple(float(value) for value in lock["beta_grid"]) != FORMAL_BETA_GRID:
                raise RuntimeError(f"Locked formal beta grid differs for {spec['dataset']} / {spec['pair']}")
            for beta in BETA_GRID:
                state = apply_beta(frame, alpha0=alpha0, beta=beta, threshold=threshold, alphas=alphas)
                summaries.append(summarize(state, split=split, spec=spec, beta=beta, locked_beta=locked_beta))
                for seed, subset in state.groupby("seed", sort=True):
                    by_seed.append(summarize(subset, split=split, spec=spec, beta=beta, locked_beta=locked_beta, seed=int(seed)))
                for direction, subset in state.groupby("direction", sort=True):
                    by_direction.append(summarize(subset, split=split, spec=spec, beta=beta, locked_beta=locked_beta, direction=str(direction)))

    summary = pd.DataFrame(summaries)[list(SUMMARY_COLUMNS)]
    seed_frame = pd.DataFrame(by_seed)
    direction_frame = pd.DataFrame(by_direction)
    core_reference = validate_core_test_reference(summary, repo_root)
    summary.to_csv(output_dir / "beta_sensitivity_summary.csv", index=False, na_rep="")
    seed_frame.to_csv(output_dir / "beta_sensitivity_by_seed.csv", index=False, na_rep="")
    direction_frame.to_csv(output_dir / "beta_sensitivity_by_direction.csv", index=False, na_rep="")
    plot_metric(summary, "delta_mrr_vs_global", "Delta MRR vs Global", "Beta sensitivity: net performance", output_dir / "beta_vs_delta_mrr.pdf")
    plot_metric(summary, "harmful_query_rate", "Harmful query rate (%)", "Beta sensitivity: harmful-correction frequency", output_dir / "beta_vs_harm_rate.pdf")
    plot_metric(summary, "mean_abs_alpha_deviation", "Mean |alpha-alpha0|", "Beta sensitivity: intervention magnitude", output_dir / "beta_vs_mean_alpha_deviation.pdf")
    write_report(
        report_path,
        summary,
        audits,
        theory,
        core_reference,
        output_dir,
        repo_root,
    )

    outputs = [
        "beta_sensitivity_summary.csv",
        "beta_sensitivity_by_seed.csv",
        "beta_sensitivity_by_direction.csv",
        "beta_vs_delta_mrr.pdf",
        "beta_vs_harm_rate.pdf",
        "beta_vs_mean_alpha_deviation.pdf",
    ]
    payload = {
        "schema_version": 1,
        "analysis": "Paper A bounded-correction / beta trust-region sensitivity",
        "formal_beta_grid": list(FORMAL_BETA_GRID),
        "diagnostic_beta": DIAGNOSTIC_BETA,
        "diagnostic_beta_participates_in_selection": False,
        "selected_beta_source": "existing immutable DEV lock only",
        "test_used_for_beta_or_threshold_selection": False,
        "fixed_components": ["features", "model", "alpha0", "grouped folds", "confidence rule", "alpha grid"],
        "theoretical_check": theory,
        "core_ablation_test_reproduction": core_reference,
        "source_audit": audits,
        "output_hashes": {name: sha256_file(output_dir / name) for name in outputs},
    }
    (output_dir / "beta_sensitivity_audit.json").write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote beta sensitivity analysis to {output_dir}")
    print(f"[OK] wrote report to {report_path}")


if __name__ == "__main__":
    main()
