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
    map_to_grid,
    portable_path,
    rr_for_alpha,
)


TOLERANCE = 1e-12
COUNTERFACTUAL_COLUMNS = (
    "split",
    "dataset",
    "pair",
    "query_id",
    "query_key",
    "head_id",
    "relation_id",
    "tail_id",
    "target_entity_id",
    "seed",
    "direction",
    "alpha0",
    "beta",
    "confidence_threshold",
    "decision",
    "probability_a",
    "confidence",
    "nonfinite_fallback",
    "alpha_final",
    "rr_final",
    "alpha_raw_continuous",
    "alpha_raw_grid",
    "rr_raw",
    "rr_global",
    "delta_rr_raw_vs_global",
    "fallback_class",
    "rr_loss_avoided",
    "potential_gain_sacrificed",
    "net_fallback_utility",
    "source_rows_sha256",
    "dev_lock_sha256",
)

SUMMARY_COLUMNS = (
    "split",
    "dataset",
    "pair",
    "scope",
    "seed",
    "direction",
    "n_observations",
    "n_fallback",
    "fallback_rate",
    "n_changed_from_anchor",
    "changed_from_anchor_rate",
    "n_effective_adaptation",
    "effective_adaptation_rate",
    "mrr_final",
    "global_mrr",
    "delta_mrr_vs_global",
    "n_avoided_harm",
    "avoided_harm_rate_among_fallback",
    "n_missed_benefit",
    "missed_benefit_rate_among_fallback",
    "n_neutral",
    "neutral_rate_among_fallback",
    "total_rr_loss_avoided",
    "total_potential_gain_sacrificed",
    "net_fallback_utility",
    "net_fallback_utility_per_observation",
    "net_fallback_utility_per_fallback",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit final Paper A Anchored fallback behavior using a no-confidence-"
            "fallback raw-policy counterfactual. No model or threshold is changed."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/fallback_audit",
    )
    parser.add_argument(
        "--report-path",
        default="docs/reports/paper_a_fallback_behavior_report.md",
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


def source_path(repo_root: Path, spec: dict[str, str], split: str) -> Path:
    root = pair_root(repo_root, spec)
    if split == "dev":
        return root / "anchored_crossfit/dev_anchored_query_rows.csv"
    return root / "test_anchored/test_locked_query_rows.csv"


def columns_for_split(split: str) -> dict[str, str]:
    if split == "dev":
        return {
            "alpha0": "alpha0_crossfit",
            "beta": "anchored_beta",
            "threshold": "anchored_confidence_threshold",
            "alpha_final": "alpha_anchored_crossfit",
            "rr_final": "rr_anchored_crossfit",
            "rr_global": "rr_global_crossfit",
        }
    return {
        "alpha0": "alpha0_locked",
        "beta": "anchored_beta_locked",
        "threshold": "anchored_confidence_threshold_locked",
        "alpha_final": "alpha_anchored_locked",
        "rr_final": "rr_anchored_locked",
        "rr_global": "rr_global",
    }


def assert_close(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if not np.allclose(left, right, rtol=0.0, atol=TOLERANCE, equal_nan=True):
        error = float(np.nanmax(np.abs(left - right)))
        raise RuntimeError(f"{label} mismatch; max absolute error={error}")


def load_pair_split(
    repo_root: Path,
    spec: dict[str, str],
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = pair_root(repo_root, spec)
    path = source_path(repo_root, spec, split)
    lock_path = root / "dev_lock/anchored_dev_lock.json"
    if not path.exists() or not lock_path.exists():
        raise FileNotFoundError(path if not path.exists() else lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    alphas = tuple(float(value) for value in lock["alpha_grid"])
    frame = pd.read_csv(path)
    columns = columns_for_split(split)
    required = {
        "query_id",
        "query_key",
        "head_id",
        "relation_id",
        "tail_id",
        "target_entity_id",
        "seed",
        "direction",
        "split",
        "anchored_decision",
        "anchored_probability_a",
        "anchored_confidence",
        "anchored_fallback",
        *QUERY_GEOMETRY_FIELDS,
        *(alpha_column(alpha) for alpha in alphas),
        *columns.values(),
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing fallback-audit columns in {path}: {sorted(missing)}")
    if frame.empty or set(frame["split"].astype(str)) != {split}:
        raise RuntimeError(f"Unexpected split content in {path}")
    if frame["query_id"].duplicated().any():
        raise RuntimeError(f"Duplicate query_id in {path}")

    decision = frame["anchored_decision"].to_numpy(dtype=np.float64)
    probability = frame["anchored_probability_a"].to_numpy(dtype=np.float64)
    confidence = np.abs(2.0 * probability - 1.0)
    stored_confidence = frame["anchored_confidence"].to_numpy(dtype=np.float64)
    assert_close(confidence, stored_confidence, f"{split} confidence")
    geometry = frame[list(QUERY_GEOMETRY_FIELDS)].to_numpy(dtype=np.float64)
    nonfinite = (
        ~np.isfinite(geometry).all(axis=1)
        | ~np.isfinite(decision)
        | ~np.isfinite(probability)
    )
    alpha0 = frame[columns["alpha0"]].to_numpy(dtype=np.float64)
    beta = frame[columns["beta"]].to_numpy(dtype=np.float64)
    threshold = frame[columns["threshold"]].to_numpy(dtype=np.float64)
    final_fallback = nonfinite | (confidence < threshold)
    if not np.array_equal(
        final_fallback,
        frame["anchored_fallback"].to_numpy(dtype=np.int64).astype(bool),
    ):
        raise RuntimeError(f"Stored fallback mask differs from final policy in {path}")

    unconstrained = alpha0 + beta * np.tanh(decision)
    bounded = np.clip(unconstrained, 0.0, 1.0)
    final_continuous = np.where(final_fallback, alpha0, bounded)
    final_alpha = map_to_grid(final_continuous, alpha0, alphas)
    stored_alpha = frame[columns["alpha_final"]].to_numpy(dtype=np.float64)
    assert_close(final_alpha, stored_alpha, f"{split} final alpha")
    final_rr = rr_for_alpha(frame, final_alpha, alphas)
    stored_rr = frame[columns["rr_final"]].to_numpy(dtype=np.float64)
    assert_close(final_rr, stored_rr, f"{split} final RR")

    raw_continuous = np.where(nonfinite, alpha0, bounded)
    raw_alpha = map_to_grid(raw_continuous, alpha0, alphas)
    raw_rr = rr_for_alpha(frame, raw_alpha, alphas)
    global_rr = frame[columns["rr_global"]].to_numpy(dtype=np.float64)
    grid_global_rr = rr_for_alpha(frame, alpha0, alphas)
    assert_close(grid_global_rr, global_rr, f"{split} Global RR")

    delta_raw = raw_rr - global_rr
    fallback_class = np.full(len(frame), "not_fallback", dtype=object)
    fallback_class[final_fallback & (delta_raw < 0.0)] = "avoided_harm"
    fallback_class[final_fallback & (delta_raw > 0.0)] = "missed_benefit"
    fallback_class[final_fallback & (delta_raw == 0.0)] = "neutral"
    changed = np.abs(final_alpha - alpha0) > TOLERANCE
    effective = (~final_fallback) & changed
    if not np.array_equal(changed, effective):
        raise RuntimeError(
            "Final fallback did not return every fallback query to alpha0"
        )

    state = pd.DataFrame(
        {
            "split": split,
            "dataset": spec["dataset"],
            "pair": spec["pair"],
            "query_id": frame["query_id"].astype(str),
            "query_key": frame["query_key"].astype(str),
            "head_id": frame["head_id"].astype(int),
            "relation_id": frame["relation_id"].astype(int),
            "tail_id": frame["tail_id"].astype(int),
            "target_entity_id": frame["target_entity_id"].astype(int),
            "seed": frame["seed"].astype(int),
            "direction": frame["direction"].astype(str),
            "alpha0": alpha0,
            "beta": beta,
            "confidence_threshold": threshold,
            "decision": decision,
            "probability_a": probability,
            "confidence": confidence,
            "nonfinite_fallback": nonfinite,
            "fallback": final_fallback,
            "changed_from_anchor": changed,
            "effective_adaptation": effective,
            "alpha_final": final_alpha,
            "rr_final": final_rr,
            "alpha_raw_continuous": raw_continuous,
            "alpha_raw_grid": raw_alpha,
            "rr_raw": raw_rr,
            "rr_global": global_rr,
            "delta_rr_raw_vs_global": delta_raw,
            "fallback_class": fallback_class,
            "rr_loss_avoided": np.where(
                final_fallback & (delta_raw < 0.0), -delta_raw, 0.0
            ),
            "potential_gain_sacrificed": np.where(
                final_fallback & (delta_raw > 0.0), delta_raw, 0.0
            ),
            "net_fallback_utility": np.where(
                final_fallback, -delta_raw, 0.0
            ),
        }
    )
    fallback_rows = state.loc[state["fallback"]].copy()
    source_hash = sha256_file(path)
    lock_hash = sha256_file(lock_path)
    fallback_rows["source_rows_sha256"] = source_hash
    fallback_rows["dev_lock_sha256"] = lock_hash
    fallback_rows = fallback_rows[list(COUNTERFACTUAL_COLUMNS)]
    audit = {
        "split": split,
        "dataset": spec["dataset"],
        "pair": spec["pair"],
        "source_rows": portable_path(path, repo_root),
        "source_rows_sha256": source_hash,
        "dev_lock": portable_path(lock_path, repo_root),
        "dev_lock_sha256": lock_hash,
        "n_observations": int(len(frame)),
        "n_fallback": int(final_fallback.sum()),
        "n_nonfinite_fallback": int(nonfinite.sum()),
        "stored_policy_reconstruction_passed": True,
        "counterfactual_confidence_threshold_applied": False,
        "counterfactual_nonfinite_fallback_only": True,
        "formal_threshold_modified": False,
        "test_is_posthoc_diagnostic_only": split == "test",
    }
    return state, fallback_rows, audit


def summarize(
    state: pd.DataFrame,
    *,
    scope: str,
    seed: str | int = "pooled",
    direction: str = "pooled",
) -> dict[str, Any]:
    fallback = state["fallback"].astype(bool)
    fallback_rows = state.loc[fallback]
    count = int(fallback.sum())
    avoided = fallback_rows["fallback_class"].eq("avoided_harm")
    missed = fallback_rows["fallback_class"].eq("missed_benefit")
    neutral = fallback_rows["fallback_class"].eq("neutral")

    def rate(value: int) -> float | None:
        return float(value / count) if count else None

    avoided_count = int(avoided.sum())
    missed_count = int(missed.sum())
    neutral_count = int(neutral.sum())
    loss_avoided = float(fallback_rows["rr_loss_avoided"].sum())
    gain_sacrificed = float(fallback_rows["potential_gain_sacrificed"].sum())
    net_utility = float(fallback_rows["net_fallback_utility"].sum())
    return {
        "split": str(state.iloc[0]["split"]),
        "dataset": str(state.iloc[0]["dataset"]),
        "pair": str(state.iloc[0]["pair"]),
        "scope": scope,
        "seed": seed,
        "direction": direction,
        "n_observations": int(len(state)),
        "n_fallback": count,
        "fallback_rate": float(fallback.mean()),
        "n_changed_from_anchor": int(state["changed_from_anchor"].sum()),
        "changed_from_anchor_rate": float(state["changed_from_anchor"].mean()),
        "n_effective_adaptation": int(state["effective_adaptation"].sum()),
        "effective_adaptation_rate": float(
            state["effective_adaptation"].mean()
        ),
        "mrr_final": float(state["rr_final"].mean()),
        "global_mrr": float(state["rr_global"].mean()),
        "delta_mrr_vs_global": float(
            (state["rr_final"] - state["rr_global"]).mean()
        ),
        "n_avoided_harm": avoided_count,
        "avoided_harm_rate_among_fallback": rate(avoided_count),
        "n_missed_benefit": missed_count,
        "missed_benefit_rate_among_fallback": rate(missed_count),
        "n_neutral": neutral_count,
        "neutral_rate_among_fallback": rate(neutral_count),
        "total_rr_loss_avoided": loss_avoided,
        "total_potential_gain_sacrificed": gain_sacrificed,
        "net_fallback_utility": net_utility,
        "net_fallback_utility_per_observation": net_utility / len(state),
        "net_fallback_utility_per_fallback": net_utility / count
        if count
        else None,
    }


def plot_with_matplotlib(summary: pd.DataFrame, output_dir: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    markers = {"dev": "o", "test": "s"}
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    for split in ("dev", "test"):
        subset = summary[summary["split"].eq(split)]
        axis.scatter(
            100.0 * subset["fallback_rate"],
            subset["delta_mrr_vs_global"],
            marker=markers[split],
            label="Grouped OOF DEV" if split == "dev" else "Locked TEST",
        )
        for row in subset.itertuples(index=False):
            axis.annotate(
                f"{row.dataset}/{row.pair.replace('M-Hyper + ', 'MH+').replace('AdaMF-MAT', 'Ada')}",
                (100.0 * row.fallback_rate, row.delta_mrr_vs_global),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("Fallback rate (%)")
    axis.set_ylabel("Delta MRR vs Global")
    axis.grid(True, linewidth=0.4, alpha=0.35)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "fallback_rate_vs_delta_mrr.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for axis, split in zip(axes, ("dev", "test"), strict=True):
        subset = summary[summary["split"].eq(split)]
        x = np.arange(len(subset))
        avoided = 100.0 * subset["avoided_harm_rate_among_fallback"].fillna(0.0)
        missed = 100.0 * subset["missed_benefit_rate_among_fallback"].fillna(0.0)
        neutral = 100.0 * subset["neutral_rate_among_fallback"].fillna(0.0)
        axis.bar(x, avoided, label="Avoided harm")
        axis.bar(x, missed, bottom=avoided, label="Missed benefit")
        axis.bar(x, neutral, bottom=avoided + missed, label="Neutral")
        axis.set_xticks(x, [f"{row.dataset}\n{row.pair.replace('M-Hyper + ', 'MH+').replace('AdaMF-MAT', 'Ada')}" for row in subset.itertuples(index=False)], rotation=20, ha="right")
        axis.set_title("Grouped OOF DEV" if split == "dev" else "Locked TEST")
        axis.set_ylabel("Composition among fallback queries (%)")
        axis.set_ylim(0.0, 100.0)
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    figure.savefig(output_dir / "fallback_composition.pdf", bbox_inches="tight")
    plt.close(figure)
    return True


def plot_with_reportlab(summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        from reportlab.graphics import renderPDF
        from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
        from reportlab.lib.colors import HexColor
    except ImportError as error:
        raise RuntimeError("Figure rendering requires matplotlib or reportlab") from error
    dark = HexColor("#202124")
    grid = HexColor("#d5d8dc")
    dev_color = HexColor("#3568a8")
    test_color = HexColor("#c45a32")

    width, height = 540.0, 360.0
    drawing = Drawing(width, height)
    drawing.add(String(width / 2, height - 23, "Fallback exposure vs final Delta MRR", textAnchor="middle", fontName="Helvetica-Bold", fontSize=12, fillColor=dark))
    left, bottom, plot_width, plot_height = 64.0, 62.0, 430.0, 240.0
    x_max = max(5.0, 100.0 * float(summary["fallback_rate"].max()) * 1.12)
    y_values = summary["delta_mrr_vs_global"].to_numpy(dtype=np.float64)
    y_min = min(0.0, float(y_values.min()))
    y_max = max(0.0, float(y_values.max()))
    y_pad = max((y_max - y_min) * 0.12, 1e-4)
    y_min -= y_pad
    y_max += y_pad
    for tick in np.linspace(y_min, y_max, 6):
        y = bottom + plot_height * (tick - y_min) / (y_max - y_min)
        drawing.add(Line(left, y, left + plot_width, y, strokeColor=grid, strokeWidth=0.45))
        drawing.add(String(left - 7, y - 2.5, f"{tick:.4f}", textAnchor="end", fontName="Helvetica", fontSize=7, fillColor=dark))
    for tick in np.linspace(0.0, x_max, 6):
        x = left + plot_width * tick / x_max
        drawing.add(String(x, bottom - 13, f"{tick:.0f}", textAnchor="middle", fontName="Helvetica", fontSize=7, fillColor=dark))
    drawing.add(Line(left, bottom, left, bottom + plot_height, strokeColor=dark, strokeWidth=0.7))
    drawing.add(Line(left, bottom, left + plot_width, bottom, strokeColor=dark, strokeWidth=0.7))
    zero_y = bottom + plot_height * (0.0 - y_min) / (y_max - y_min)
    drawing.add(Line(left, zero_y, left + plot_width, zero_y, strokeColor=dark, strokeWidth=0.65))
    for row in summary.itertuples(index=False):
        x = left + plot_width * (100.0 * row.fallback_rate) / x_max
        y = bottom + plot_height * (row.delta_mrr_vs_global - y_min) / (y_max - y_min)
        color = dev_color if row.split == "dev" else test_color
        drawing.add(Circle(x, y, 3.0, strokeColor=color, fillColor=color if row.split == "dev" else None, strokeWidth=1.2))
        short_pair = row.pair.replace("M-Hyper + ", "MH+").replace("AdaMF-MAT", "Ada")
        dx, dy = (5.0, 7.0) if row.split == "dev" else (5.0, -11.0)
        if row.dataset == "DB15K" and row.pair == "M-Hyper + NativE":
            dy = -13.0 if row.split == "dev" else 5.0
        drawing.add(String(x + dx, y + dy, f"{row.dataset}/{short_pair}", fontName="Helvetica", fontSize=6.5, fillColor=dark))
    drawing.add(String(left + plot_width / 2, 28.0, "Fallback rate (%)", textAnchor="middle", fontName="Helvetica", fontSize=8, fillColor=dark))
    drawing.add(String(left, bottom + plot_height + 8.0, "Delta MRR vs Global", fontName="Helvetica", fontSize=8, fillColor=dark))
    drawing.add(Circle(187, 13, 3, strokeColor=dev_color, fillColor=dev_color))
    drawing.add(String(196, 10, "Grouped OOF DEV", fontName="Helvetica", fontSize=8, fillColor=dark))
    drawing.add(Circle(310, 13, 3, strokeColor=test_color, fillColor=None))
    drawing.add(String(319, 10, "Locked TEST", fontName="Helvetica", fontSize=8, fillColor=dark))
    renderPDF.drawToFile(drawing, str(output_dir / "fallback_rate_vs_delta_mrr.pdf"))

    width, height = 756.0, 390.0
    drawing = Drawing(width, height)
    drawing.add(String(width / 2, height - 23, "Counterfactual composition of final fallback queries", textAnchor="middle", fontName="Helvetica-Bold", fontSize=12, fillColor=dark))
    colors = {
        "avoided": HexColor("#3a7d5b"),
        "missed": HexColor("#c45a32"),
        "neutral": HexColor("#aeb4ba"),
    }
    for panel_index, split in enumerate(("dev", "test")):
        subset = summary[summary["split"].eq(split)].reset_index(drop=True)
        panel_left = 48.0 + panel_index * 365.0
        plot_bottom = 90.0
        plot_height = 220.0
        drawing.add(String(panel_left + 155.0, 330.0, "Grouped OOF DEV" if split == "dev" else "Locked TEST", textAnchor="middle", fontName="Helvetica-Bold", fontSize=10, fillColor=dark))
        for tick in range(0, 101, 25):
            y = plot_bottom + plot_height * tick / 100.0
            drawing.add(Line(panel_left, y, panel_left + 310.0, y, strokeColor=grid, strokeWidth=0.45))
            drawing.add(String(panel_left - 6, y - 2.5, str(tick), textAnchor="end", fontName="Helvetica", fontSize=7, fillColor=dark))
        bar_width = 46.0
        gap = 28.0
        for index, row in subset.iterrows():
            x = panel_left + 15.0 + index * (bar_width + gap)
            if int(row["n_fallback"]) == 0:
                drawing.add(String(x + bar_width / 2, plot_bottom + 4, "n=0", textAnchor="middle", fontName="Helvetica", fontSize=7, fillColor=dark))
            else:
                base = plot_bottom
                values = (
                    (100.0 * float(row["avoided_harm_rate_among_fallback"]), colors["avoided"]),
                    (100.0 * float(row["missed_benefit_rate_among_fallback"]), colors["missed"]),
                    (100.0 * float(row["neutral_rate_among_fallback"]), colors["neutral"]),
                )
                for value, color in values:
                    segment = plot_height * value / 100.0
                    drawing.add(Rect(x, base, bar_width, segment, strokeColor=None, fillColor=color))
                    base += segment
                drawing.add(String(x + bar_width / 2, plot_bottom + plot_height + 7, f"n={int(row['n_fallback'])}", textAnchor="middle", fontName="Helvetica", fontSize=6.5, fillColor=dark))
            short_pair = str(row["pair"]).replace("M-Hyper + ", "MH+").replace("AdaMF-MAT", "Ada")
            drawing.add(String(x + bar_width / 2, 75.0, str(row["dataset"]), textAnchor="middle", fontName="Helvetica", fontSize=6.5, fillColor=dark))
            drawing.add(String(x + bar_width / 2, 65.0, short_pair, textAnchor="middle", fontName="Helvetica", fontSize=6.5, fillColor=dark))
    legend = (("Avoided harm", colors["avoided"]), ("Missed benefit", colors["missed"]), ("Neutral", colors["neutral"]))
    x = 250.0
    for label, color in legend:
        drawing.add(Rect(x, 25, 12, 8, strokeColor=None, fillColor=color))
        drawing.add(String(x + 17, 25, label, fontName="Helvetica", fontSize=8, fillColor=dark))
        x += 110.0
    renderPDF.drawToFile(drawing, str(output_dir / "fallback_composition.pdf"))


def plot_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    if not plot_with_matplotlib(summary, output_dir):
        plot_with_reportlab(summary, output_dir)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    summary: pd.DataFrame,
    audits: list[dict[str, Any]],
    output_dir: Path,
    repo_root: Path,
) -> None:
    stress = summary[
        summary["dataset"].eq("DB15K")
        & summary["pair"].eq("M-Hyper + AdaMF-MAT")
    ].set_index("split")
    active_pairs = int((summary.groupby(["dataset", "pair"])["n_fallback"].max() > 0).sum())
    locked_test_zero_pairs = int(
        (summary.loc[summary["split"].eq("test"), "n_fallback"] == 0).sum()
    )
    net_positive = int((summary.loc[summary["n_fallback"] > 0, "net_fallback_utility"] > 0.0).sum())
    lines = [
        "# Paper A Anchored Dynamic Fallback Behavior Audit",
        "",
        "## Main answer",
        "",
        f"Across grouped OOF DEV and locked TEST, the final confidence rule activates "
        f"fallback in {active_pairs}/4 main pairs. On locked TEST, "
        f"{locked_test_zero_pairs}/4 pairs have a zero threshold and therefore no "
        "finite-query fallback. Across the "
        f"active pair/split cases, fallback has positive aggregate counterfactual utility "
        f"in {net_positive}/{int((summary['n_fallback'] > 0).sum())} cases: avoided raw "
        "RR loss exceeds sacrificed raw RR gain. This does not imply that every fallback "
        "decision is correct; avoided harm, missed benefit, and neutral outcomes are "
        "reported separately below.",
        "",
        "## Counterfactual definition",
        "",
        "For each query, the final stored policy is reconstructed from its actual alpha0, "
        "beta, decision, confidence threshold, and exact alpha grid. The counterfactual "
        "then removes only confidence fallback:",
        "",
        "```text",
        "alpha_raw = clip(alpha0 + beta * tanh(decision), 0, 1).",
        "```",
        "",
        "Non-finite queries still fall back. A final fallback query is `avoided_harm` if "
        "`rr_raw < rr_global`, `missed_benefit` if `rr_raw > rr_global`, and `neutral` "
        "under exact equality. Net fallback utility is total avoided RR loss minus total "
        "potential RR gain sacrificed, equivalently `sum(rr_global - rr_raw)` over "
        "fallback queries.",
        "",
        "## Pair-level summary",
        "",
        "| Split | Dataset/Pair | Fallback n (%) | Changed % | Effective adaptation % | Delta MRR | Avoided n (%) | Missed n (%) | Neutral n (%) | RR loss avoided | Gain sacrificed | Net fallback utility |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {str(row.split).upper()} | {row.dataset} / {row.pair} | "
            f"{row.n_fallback} ({100.0 * row.fallback_rate:.2f}) | "
            f"{100.0 * row.changed_from_anchor_rate:.2f} | "
            f"{100.0 * row.effective_adaptation_rate:.2f} | "
            f"{row.delta_mrr_vs_global:+.6f} | "
            f"{row.n_avoided_harm} ({fmt(None if pd.isna(row.avoided_harm_rate_among_fallback) else 100.0 * row.avoided_harm_rate_among_fallback, 2)}) | "
            f"{row.n_missed_benefit} ({fmt(None if pd.isna(row.missed_benefit_rate_among_fallback) else 100.0 * row.missed_benefit_rate_among_fallback, 2)}) | "
            f"{row.n_neutral} ({fmt(None if pd.isna(row.neutral_rate_among_fallback) else 100.0 * row.neutral_rate_among_fallback, 2)}) | "
            f"{row.total_rr_loss_avoided:.6f} | "
            f"{row.total_potential_gain_sacrificed:.6f} | "
            f"{row.net_fallback_utility:+.6f} |"
        )
    lines += [
        "",
        "Rates in the avoided/missed/neutral columns use final fallback queries as the "
        "denominator. Changed-from-anchor and effective-adaptation rates use all queries. "
        "They are equal here because every fallback query is verified to return exactly "
        "to alpha0.",
        "",
        "## DB15K / M-Hyper + AdaMF-MAT stress case",
        "",
    ]
    for split in ("dev", "test"):
        row = stress.loc[split]
        lines.append(
            f"- {split.upper()}: fallback covers {100.0 * row['fallback_rate']:.2f}% "
            f"({int(row['n_fallback'])}/{int(row['n_observations'])}) of observations. "
            f"Among fallback queries, {100.0 * row['avoided_harm_rate_among_fallback']:.2f}% "
            f"avoid harm, {100.0 * row['missed_benefit_rate_among_fallback']:.2f}% miss a "
            f"benefit, and {100.0 * row['neutral_rate_among_fallback']:.2f}% are neutral. "
            f"Total RR loss avoided is {row['total_rr_loss_avoided']:.6f}, potential gain "
            f"sacrificed is {row['total_potential_gain_sacrificed']:.6f}, and net fallback "
            f"utility is {row['net_fallback_utility']:+.6f} "
            f"({row['net_fallback_utility_per_observation']:+.6f} MRR contribution over "
            "all observations)."
        )
    lines += [
        "",
        "This stress case quantifies the mechanism's tradeoff rather than using it to "
        "retune the threshold. TEST composition is post-hoc diagnostic evidence only.",
        "",
        "## Figures",
        "",
        f"- `{portable_path(output_dir / 'fallback_rate_vs_delta_mrr.pdf', repo_root)}`",
        f"- `{portable_path(output_dir / 'fallback_composition.pdf', repo_root)}`",
        "",
        "## Integrity and information boundary",
        "",
        "| Split | Dataset/Pair | Rows | Fallback | Non-finite | Stored reconstruction | Source SHA-256 | DEV lock SHA-256 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for item in audits:
        lines.append(
            f"| {item['split'].upper()} | {item['dataset']} / {item['pair']} | "
            f"{item['n_observations']} | {item['n_fallback']} | "
            f"{item['n_nonfinite_fallback']} | yes | "
            f"`{item['source_rows_sha256']}` | `{item['dev_lock_sha256']}` |"
        )
    lines += [
        "",
        "This script trains no model, changes no threshold, and performs no TEST-based "
        "selection. Grouped OOF DEV and immutable locked TEST are reported separately. "
        "Every stored fallback flag, final alpha, final RR, and Global RR is reconstructed "
        "from exact-ranking assets before counterfactual classification.",
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

    states: list[pd.DataFrame] = []
    counterfactuals: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        for spec in PAIR_SPECS:
            state, fallback_rows, source_audit = load_pair_split(
                repo_root, spec, split
            )
            states.append(state)
            counterfactuals.append(fallback_rows)
            audits.append(source_audit)
            summary_rows.append(summarize(state, scope="pair"))
            for seed, subset in state.groupby("seed", sort=True):
                seed_rows.append(
                    summarize(subset, scope="seed", seed=int(seed))
                )
            for direction, subset in state.groupby("direction", sort=True):
                direction_rows.append(
                    summarize(
                        subset,
                        scope="direction",
                        direction=str(direction),
                    )
                )
    summary = pd.DataFrame(summary_rows)[list(SUMMARY_COLUMNS)]
    by_seed = pd.DataFrame(seed_rows)[list(SUMMARY_COLUMNS)]
    by_direction = pd.DataFrame(direction_rows)[list(SUMMARY_COLUMNS)]
    fallback_rows = pd.concat(counterfactuals, ignore_index=True)
    summary.to_csv(output_dir / "fallback_summary.csv", index=False, na_rep="")
    fallback_rows.to_csv(
        output_dir / "fallback_counterfactual.csv", index=False, na_rep=""
    )
    by_seed.to_csv(output_dir / "fallback_by_seed.csv", index=False, na_rep="")
    by_direction.to_csv(
        output_dir / "fallback_by_direction.csv", index=False, na_rep=""
    )
    plot_figures(summary, output_dir)
    write_report(report_path, summary, audits, output_dir, repo_root)

    outputs = [
        "fallback_summary.csv",
        "fallback_counterfactual.csv",
        "fallback_by_seed.csv",
        "fallback_by_direction.csv",
        "fallback_rate_vs_delta_mrr.pdf",
        "fallback_composition.pdf",
    ]
    payload = {
        "schema_version": 1,
        "analysis": "Paper A final Anchored fallback behavior audit",
        "counterfactual": "raw bounded policy with no confidence fallback; non-finite fallback only",
        "classification": {
            "avoided_harm": "rr_raw < rr_global",
            "missed_benefit": "rr_raw > rr_global",
            "neutral": "rr_raw == rr_global",
        },
        "pure_posthoc_mechanism_analysis": True,
        "formal_fallback_threshold_modified": False,
        "test_is_diagnostic_only": True,
        "test_used_for_selection": False,
        "stored_policy_reconstruction_passed": all(
            item["stored_policy_reconstruction_passed"] for item in audits
        ),
        "source_audit": audits,
        "output_hashes": {
            name: sha256_file(output_dir / name) for name in outputs
        },
    }
    (output_dir / "fallback_audit.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote fallback behavior audit to {output_dir}")
    print(f"[OK] wrote report to {report_path}")


if __name__ == "__main__":
    main()
