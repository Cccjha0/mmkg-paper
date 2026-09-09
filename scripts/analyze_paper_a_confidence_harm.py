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

from scripts.analyze_paper_a_risk_coverage import (  # noqa: E402
    PAIR_SPECS,
    load_pair_split,
    portable_path,
)


BIN_LABELS = (
    "top_20pct",
    "20_to_40pct",
    "40_to_60pct",
    "60_to_80pct",
    "bottom_20pct",
)

SUMMARY_COLUMNS = (
    "split",
    "scope",
    "dataset",
    "pair",
    "n_observations",
    "n_finite_confidence",
    "n_nonfinite_fallback",
    "n_harmful",
    "harm_prevalence",
    "auroc_risk_to_harm",
    "auprc_risk_to_harm",
    "auprc_lift_over_prevalence",
    "spearman_confidence_vs_delta_rr_raw",
    "mean_delta_rr_raw",
)

BIN_COLUMNS = (
    "split",
    "scope",
    "dataset",
    "pair",
    "confidence_bin",
    "confidence_rank_start_percent",
    "confidence_rank_end_percent",
    "n",
    "n_harmful",
    "harm_rate",
    "mean_delta_rr",
    "mean_harm",
    "mean_confidence",
    "min_confidence",
    "max_confidence",
)

QUERY_COLUMNS = (
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
    "beta_locked",
    "anchored_decision",
    "anchored_probability_a",
    "confidence",
    "risk_score",
    "nonfinite_fallback",
    "alpha_raw_continuous",
    "alpha_raw_grid",
    "rr_global",
    "rr_raw_bounded",
    "delta_rr_raw",
    "harm_label",
    "source_rows_sha256",
    "dev_lock_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether frozen Anchored Dynamic confidence identifies harmful "
            "counterfactual raw bounded corrections. No model is trained or tuned."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/confidence_harm",
    )
    parser.add_argument(
        "--report-path",
        default="docs/reports/paper_a_confidence_harm_report.md",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum = float(ranks[labels == 1].sum())
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware non-interpolated average precision."""
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    total = np.arange(1, len(labels) + 1)
    ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]),
        len(labels) - 1,
    ]
    precision = cumulative_tp[ends] / total[ends]
    recall = cumulative_tp[ends] / positives
    previous = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous) * precision))


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def source_path(repo_root: Path, spec: dict[str, str], split: str) -> Path:
    root = (
        repo_root
        / "outputs"
        / spec["dataset_dir"]
        / "anchored_dynamic"
        / f"{spec['pair_dir']}_seed123"
    )
    if split == "dev":
        return root / "anchored_crossfit/dev_anchored_query_rows.csv"
    return root / "test_anchored/test_locked_query_rows.csv"


def prepare_pair_split(
    repo_root: Path,
    spec: dict[str, str],
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # This is the same audited reconstruction used by the risk-coverage analysis:
    # final locked alpha0/beta, no confidence threshold, non-finite fallback only,
    # and the existing exact-ranking grid/tie rule.
    reconstructed, source_audit = load_pair_split(repo_root, spec, split)
    rows_path = source_path(repo_root, spec, split)
    source = pd.read_csv(
        rows_path,
        usecols=[
            "query_id",
            "anchored_decision",
            "anchored_probability_a",
        ],
    )
    frame = reconstructed.merge(
        source,
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    if frame[["anchored_decision", "anchored_probability_a"]].isna().any().any():
        raise RuntimeError(f"Missing decision/probability after identity join: {rows_path}")

    frame = frame.rename(
        columns={
            "beta": "beta_locked",
            "rr_raw": "rr_raw_bounded",
        }
    )
    frame["risk_score"] = 1.0 - frame["confidence"]
    frame["delta_rr_raw"] = frame["rr_raw_bounded"] - frame["rr_global"]
    frame["harm_label"] = (
        frame["delta_rr_raw"].to_numpy(dtype=np.float64) < 0.0
    ).astype(np.int8)
    frame["source_rows_sha256"] = source_audit["source_rows_sha256"]
    frame["dev_lock_sha256"] = source_audit["lock_sha256"]

    nonfinite = frame["nonfinite_fallback"].to_numpy(dtype=bool)
    if np.any(nonfinite & (frame["harm_label"].to_numpy(dtype=bool))):
        raise RuntimeError("A non-finite fallback observation was labeled harmful")
    finite = ~nonfinite
    if not np.isfinite(
        frame.loc[finite, ["confidence", "risk_score", "delta_rr_raw"]].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise RuntimeError(f"Non-finite diagnostic values remain in {rows_path}")
    if not np.allclose(
        frame.loc[finite, "confidence"].to_numpy(dtype=np.float64),
        np.abs(
            2.0
            * frame.loc[finite, "anchored_probability_a"].to_numpy(
                dtype=np.float64
            )
            - 1.0
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"Confidence formula mismatch in {rows_path}")
    if split == "dev" and "anchored_fold" not in pd.read_csv(rows_path, nrows=0).columns:
        raise RuntimeError("DEV predictions are not grouped OOF rows")
    return frame[list(QUERY_COLUMNS)], source_audit


def summarize(
    frame: pd.DataFrame,
    *,
    scope: str,
    dataset: str,
    pair: str,
) -> dict[str, Any]:
    finite = ~frame["nonfinite_fallback"].to_numpy(dtype=bool)
    eval_frame = frame.loc[finite]
    labels = eval_frame["harm_label"].to_numpy(dtype=np.int64)
    risk = eval_frame["risk_score"].to_numpy(dtype=np.float64)
    confidence = eval_frame["confidence"].to_numpy(dtype=np.float64)
    delta = eval_frame["delta_rr_raw"].to_numpy(dtype=np.float64)
    prevalence = float(frame["harm_label"].mean())
    ap = average_precision(labels, risk)
    return {
        "split": str(frame.iloc[0]["split"]),
        "scope": scope,
        "dataset": dataset,
        "pair": pair,
        "n_observations": int(len(frame)),
        "n_finite_confidence": int(finite.sum()),
        "n_nonfinite_fallback": int((~finite).sum()),
        "n_harmful": int(frame["harm_label"].sum()),
        "harm_prevalence": prevalence,
        "auroc_risk_to_harm": roc_auc(labels, risk),
        "auprc_risk_to_harm": ap,
        "auprc_lift_over_prevalence": ap - prevalence,
        "spearman_confidence_vs_delta_rr_raw": spearman_correlation(
            confidence, delta
        ),
        "mean_delta_rr_raw": float(frame["delta_rr_raw"].mean()),
    }


def confidence_bins(
    frame: pd.DataFrame,
    *,
    scope: str,
    dataset: str,
    pair: str,
) -> list[dict[str, Any]]:
    finite = frame.loc[~frame["nonfinite_fallback"].astype(bool)].copy()
    finite = finite.sort_values(
        ["confidence", "query_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if finite.empty:
        return []
    # Fixed-count rank quintiles avoid qcut silently dropping bins under ties.
    positions = np.arange(len(finite), dtype=np.int64)
    finite["bin_index"] = np.minimum(4, (5 * positions) // len(finite))
    output = []
    for index, label in enumerate(BIN_LABELS):
        subset = finite.loc[finite["bin_index"].eq(index)]
        harmful = subset["harm_label"].astype(bool)
        harm_values = -subset.loc[harmful, "delta_rr_raw"]
        output.append(
            {
                "split": str(frame.iloc[0]["split"]),
                "scope": scope,
                "dataset": dataset,
                "pair": pair,
                "confidence_bin": label,
                "confidence_rank_start_percent": index * 20,
                "confidence_rank_end_percent": (index + 1) * 20,
                "n": int(len(subset)),
                "n_harmful": int(harmful.sum()),
                "harm_rate": float(harmful.mean()),
                "mean_delta_rr": float(subset["delta_rr_raw"].mean()),
                "mean_harm": float(harm_values.mean())
                if not harm_values.empty
                else None,
                "mean_confidence": float(subset["confidence"].mean()),
                "min_confidence": float(subset["confidence"].min()),
                "max_confidence": float(subset["confidence"].max()),
            }
        )
    return output


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="")


def plot_bins(bins: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plot_bins_reportlab(bins, output_path)
        return

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    x = np.arange(5)
    short_labels = ["Top 20%", "20-40%", "40-60%", "60-80%", "Bottom 20%"]
    for axis, spec in zip(axes.ravel(), PAIR_SPECS, strict=True):
        subset = bins[
            bins["scope"].eq("pair")
            & bins["dataset"].eq(spec["dataset"])
            & bins["pair"].eq(spec["pair"])
        ]
        for split, marker, style in (("dev", "o", "-"), ("test", "s", "--")):
            curve = subset[subset["split"].eq(split)].copy()
            curve["confidence_bin"] = pd.Categorical(
                curve["confidence_bin"], categories=BIN_LABELS, ordered=True
            )
            curve = curve.sort_values("confidence_bin")
            axis.plot(
                x,
                100.0 * curve["harm_rate"].to_numpy(dtype=np.float64),
                marker=marker,
                linestyle=style,
                linewidth=1.5,
                markersize=4.0,
                label="Grouped OOF DEV" if split == "dev" else "Locked TEST",
            )
        axis.set_title(f"{spec['dataset']}: {spec['pair']}", fontsize=10)
        axis.set_xticks(x, short_labels, rotation=20, ha="right")
        axis.set_ylabel("Harmful-correction rate (%)")
        axis.set_xlabel("Confidence quintile (high to low)")
        axis.grid(True, linewidth=0.4, alpha=0.35)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(
        "Does lower Anchored confidence identify harmful raw bounded corrections?",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_bins_reportlab(bins: pd.DataFrame, output_path: Path) -> None:
    """Vector-PDF fallback for environments without matplotlib."""
    try:
        from reportlab.graphics import renderPDF
        from reportlab.graphics.shapes import Circle, Drawing, Line, String
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError as error:
        raise RuntimeError(
            "Rendering confidence_vs_harm.pdf requires matplotlib or reportlab"
        ) from error

    width, height = 756.0, 518.4
    drawing = Drawing(width, height)
    dark = HexColor("#202124")
    grid = HexColor("#d5d8dc")
    dev_color = HexColor("#3568a8")
    test_color = HexColor("#c45a32")
    drawing.add(
        String(
            width / 2,
            height - 24,
            "Does lower Anchored confidence identify harmful raw bounded corrections?",
            textAnchor="middle",
            fontName="Helvetica-Bold",
            fontSize=12,
            fillColor=dark,
        )
    )
    panel_width, panel_height = 330.0, 190.0
    origins = ((52.0, 274.0), (410.0, 274.0), (52.0, 57.0), (410.0, 57.0))
    short_labels = ("Top 20%", "20-40%", "40-60%", "60-80%", "Bottom 20%")
    max_harm = float(
        100.0
        * bins.loc[bins["scope"].eq("pair"), "harm_rate"].max()
    )
    y_max = max(5.0, math.ceil(max_harm / 5.0) * 5.0)

    for (origin_x, origin_y), spec in zip(origins, PAIR_SPECS, strict=True):
        subset = bins[
            bins["scope"].eq("pair")
            & bins["dataset"].eq(spec["dataset"])
            & bins["pair"].eq(spec["pair"])
        ]
        plot_left = origin_x + 36.0
        plot_bottom = origin_y + 44.0
        plot_width = panel_width - 46.0
        plot_height = panel_height - 70.0
        title = f"{spec['dataset']}: {spec['pair']}"
        drawing.add(
            String(
                origin_x + panel_width / 2,
                origin_y + panel_height - 12.0,
                title,
                textAnchor="middle",
                fontName="Helvetica-Bold",
                fontSize=9,
                fillColor=dark,
            )
        )
        for tick in np.linspace(0.0, y_max, 5):
            y = plot_bottom + plot_height * tick / y_max
            drawing.add(Line(plot_left, y, plot_left + plot_width, y, strokeColor=grid, strokeWidth=0.45))
            label = f"{tick:.0f}"
            drawing.add(String(plot_left - 6.0, y - 2.5, label, textAnchor="end", fontName="Helvetica", fontSize=7, fillColor=dark))
        drawing.add(Line(plot_left, plot_bottom, plot_left, plot_bottom + plot_height, strokeColor=dark, strokeWidth=0.7))
        drawing.add(Line(plot_left, plot_bottom, plot_left + plot_width, plot_bottom, strokeColor=dark, strokeWidth=0.7))
        x_values = [plot_left + i * plot_width / 4.0 for i in range(5)]
        for x, label in zip(x_values, short_labels, strict=True):
            drawing.add(String(x, plot_bottom - 12.0, label, textAnchor="middle", fontName="Helvetica", fontSize=6.3, fillColor=dark))
        drawing.add(String(origin_x + panel_width / 2, origin_y + 8.0, "Confidence quintile (high to low)", textAnchor="middle", fontName="Helvetica", fontSize=7.2, fillColor=dark))
        drawing.add(
            String(
                plot_left,
                plot_bottom + plot_height + 5.0,
                "Harm rate (%)",
                fontName="Helvetica",
                fontSize=7.0,
                fillColor=dark,
            )
        )

        for split, color, filled in (("dev", dev_color, True), ("test", test_color, False)):
            curve = subset[subset["split"].eq(split)].copy()
            curve["confidence_bin"] = pd.Categorical(curve["confidence_bin"], categories=BIN_LABELS, ordered=True)
            curve = curve.sort_values("confidence_bin")
            values = 100.0 * curve["harm_rate"].to_numpy(dtype=np.float64)
            points = [(x, plot_bottom + plot_height * value / y_max) for x, value in zip(x_values, values, strict=True)]
            for (x1, y1), (x2, y2) in zip(points[:-1], points[1:], strict=True):
                drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.35, strokeDashArray=None if filled else [4, 2]))
            for x, y in points:
                drawing.add(Circle(x, y, 2.6, strokeColor=color, fillColor=color if filled else None, strokeWidth=1.0))

    legend_y = 25.0
    legend_items = (("Grouped OOF DEV", dev_color, True), ("Locked TEST", test_color, False))
    total_width = sum(stringWidth(label, "Helvetica", 8) + 40.0 for label, _, _ in legend_items)
    cursor = (width - total_width) / 2.0
    for label, color, filled in legend_items:
        drawing.add(Line(cursor, legend_y, cursor + 22.0, legend_y, strokeColor=color, strokeWidth=1.4, strokeDashArray=None if filled else [4, 2]))
        drawing.add(Circle(cursor + 11.0, legend_y, 2.6, strokeColor=color, fillColor=color if filled else None, strokeWidth=1.0))
        drawing.add(String(cursor + 28.0, legend_y - 2.8, label, fontName="Helvetica", fontSize=8, fillColor=dark))
        cursor += stringWidth(label, "Helvetica", 8) + 40.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(drawing, str(output_path))


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def qualitative_answer(row: pd.Series) -> str:
    auc = float(row["auroc_risk_to_harm"])
    ap_lift = float(row["auprc_lift_over_prevalence"])
    if not math.isfinite(auc):
        return "not identifiable because only one harm class is present"
    if auc >= 0.60 and ap_lift > 0.0:
        return "meaningful positive discrimination"
    if auc >= 0.55 and ap_lift > 0.0:
        return "weak positive discrimination"
    if auc > 0.50 and ap_lift > 0.0:
        return "very weak positive discrimination"
    if auc > 0.50 and ap_lift <= 0.0:
        return "weak AUROC; no precision lift"
    if math.isclose(auc, 0.50, abs_tol=0.01):
        return "approximately chance-level discrimination"
    return "inverse or non-useful discrimination"


def write_report(
    path: Path,
    summary: pd.DataFrame,
    bins: pd.DataFrame,
    audits: list[dict[str, Any]],
    output_dir: Path,
    repo_root: Path,
) -> None:
    pair_rows = summary[summary["scope"].eq("pair")]
    dev_rows = pair_rows[pair_rows["split"].eq("dev")]
    test_rows = pair_rows[pair_rows["split"].eq("test")]
    dev_positive = int((dev_rows["auroc_risk_to_harm"] > 0.5).sum())
    test_positive = int((test_rows["auroc_risk_to_harm"] > 0.5).sum())
    dev_ap_positive = int((dev_rows["auprc_lift_over_prevalence"] > 0.0).sum())
    test_ap_positive = int((test_rows["auprc_lift_over_prevalence"] > 0.0).sum())
    lines = [
        "# Paper A Confidence-to-Harm Diagnostic Report",
        "",
        "## Main answer",
        "",
        "Current confidence provides limited, pair-sensitive harm discrimination rather "
        "than a reliably calibrated harmful-correction probability. Risk-score AUROC is "
        f"above 0.5 in {dev_positive}/4 pairs on grouped OOF DEV and {test_positive}/4 "
        "pairs on locked TEST, but AUPRC exceeds the pair's harm prevalence in only "
        f"{dev_ap_positive}/4 DEV and {test_ap_positive}/4 TEST cases. The highest-confidence "
        "quintile is consistently the safest, yet harm is generally concentrated in "
        "middle-confidence quintiles rather than increasing monotonically toward the "
        "lowest-confidence quintile. Confidence can therefore support selective retention "
        "of especially safe queries, but `1 - confidence` is not a strong standalone harm "
        "detector across all four pairs.",
        "",
        "## Information boundary and counterfactual",
        "",
        "This analysis never defines harm from the final fallback-filtered Anchored result. "
        "For every query it reconstructs the counterfactual raw bounded policy "
        "`clip(alpha0 + beta_locked * tanh(decision), 0, 1)`, applies only non-finite "
        "fallback, maps to the existing exact-ranking alpha grid, and reads the resulting "
        "reciprocal rank from the precomputed full-ranking grid. Harm is "
        "`rr_raw_bounded < rr_global`.",
        "",
        "Confidence is `abs(2 * p_A - 1)` and harm risk is `1 - confidence`. Grouped OOF "
        "DEV predictions are evaluated separately from locked TEST. TEST results are "
        "diagnostic only and were not used to alter beta, the confidence threshold, the "
        "feature schema, or any operating point.",
        "",
        "## Discrimination metrics",
        "",
        "| Split | Dataset/Pair | Harm % | AUROC | AUPRC | AP lift | Spearman(confidence, delta RR) | Assessment |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in pair_rows.itertuples(index=False):
        series = pd.Series(row._asdict())
        lines.append(
            f"| {str(row.split).upper()} | {row.dataset} / {row.pair} | "
            f"{100.0 * row.harm_prevalence:.2f} | {fmt(row.auroc_risk_to_harm)} | "
            f"{fmt(row.auprc_risk_to_harm)} | "
            f"{float(row.auprc_lift_over_prevalence):+.4f} | "
            f"{float(row.spearman_confidence_vs_delta_rr_raw):+.4f} | "
            f"{qualitative_answer(series)} |"
        )
    lines += [
        "",
        "AUROC measures ranking discrimination of `1 - confidence` for harm. AUPRC must "
        "be read against harm prevalence; AP lift is AUPRC minus prevalence. Spearman "
        "tests whether higher confidence accompanies better raw correction outcomes, but "
        "it measures magnitude ordering rather than binary harm detection.",
        "",
        "The coexistence of AUROC slightly above 0.5 and AUPRC below prevalence on DB15K "
        "is not contradictory: high-confidence queries contain very few harmful cases, "
        "while the risk ranking becomes non-monotonic through the middle and bottom of "
        "the distribution.",
        "",
        "## Confidence quintiles",
        "",
        "Quintiles are fixed-count rank buckets within each pair and split, sorted from "
        "highest to lowest confidence with `query_id` as deterministic tie-break. The "
        "bottom 20% is therefore the lowest-confidence bucket.",
        "",
        "| Split | Dataset/Pair | Confidence bucket | n | Harm % | Mean delta RR | Mean harm | Mean confidence |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in bins[bins["scope"].eq("pair")].itertuples(index=False):
        lines.append(
            f"| {str(row.split).upper()} | {row.dataset} / {row.pair} | "
            f"{row.confidence_bin} | {row.n} | {100.0 * row.harm_rate:.2f} | "
            f"{float(row.mean_delta_rr):+.6f} | {fmt(row.mean_harm, 6)} | "
            f"{float(row.mean_confidence):.4f} |"
        )
    lines += [
        "",
        "## Source integrity",
        "",
        "| Split | Dataset/Pair | Observations | Non-finite fallback | alpha0 | beta | Source rows SHA-256 | DEV lock SHA-256 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for item in audits:
        lines.append(
            f"| {item['split'].upper()} | {item['dataset']} / {item['pair']} | "
            f"{item['n_observations']} | {item['n_nonfinite']} | "
            f"{item['alpha0']:.2f} | {item['beta']:.2f} | "
            f"`{item['source_rows_sha256']}` | `{item['lock_sha256']}` |"
        )
    lines += [
        "",
        "The stored confidence and original fallback-filtered policy are reconstructed "
        "and verified before the counterfactual is evaluated. No model or combiner is "
        "trained by this script.",
        "",
        "## Figure",
        "",
        f"- `{portable_path(output_dir / 'confidence_vs_harm.pdf', repo_root)}`",
        "",
        "The four panels show harmful-correction rates from high- to low-confidence "
        "quintiles for grouped OOF DEV and locked TEST.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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

    all_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        split_frames = []
        for spec in PAIR_SPECS:
            frame, source_audit = prepare_pair_split(repo_root, spec, split)
            split_frames.append(frame)
            all_frames.append(frame)
            audits.append(source_audit)
            summary_rows.append(
                summarize(
                    frame,
                    scope="pair",
                    dataset=spec["dataset"],
                    pair=spec["pair"],
                )
            )
            bin_rows.extend(
                confidence_bins(
                    frame,
                    scope="pair",
                    dataset=spec["dataset"],
                    pair=spec["pair"],
                )
            )
        pooled = pd.concat(split_frames, ignore_index=True)
        summary_rows.append(
            summarize(
                pooled,
                scope="pooled",
                dataset="Pooled",
                pair="Pooled",
            )
        )
        bin_rows.extend(
            confidence_bins(
                pooled,
                scope="pooled",
                dataset="Pooled",
                pair="Pooled",
            )
        )

    query_rows = pd.concat(all_frames, ignore_index=True)[list(QUERY_COLUMNS)]
    summary = pd.DataFrame(summary_rows)[list(SUMMARY_COLUMNS)]
    bins = pd.DataFrame(bin_rows)[list(BIN_COLUMNS)]
    write_csv(output_dir / "query_rows.csv", query_rows)
    write_csv(output_dir / "confidence_harm_summary.csv", summary)
    write_csv(output_dir / "confidence_bins.csv", bins)
    plot_bins(bins, output_dir / "confidence_vs_harm.pdf")

    payload = {
        "schema_version": 1,
        "analysis": "Paper A confidence-to-harm counterfactual diagnostic",
        "counterfactual_policy": "clip(alpha0 + beta_locked * tanh(decision), 0, 1)",
        "confidence_threshold_applied": False,
        "nonfinite_fallback_only": True,
        "exact_alpha_grid_mapping": "nearest grid; ties prefer alpha nearer alpha0, then smaller alpha",
        "harm_label": "1 iff rr_raw_bounded < rr_global (strict comparison)",
        "confidence": "abs(2 * anchored_probability_a - 1)",
        "risk_score": "1 - confidence",
        "dev_predictions": "grouped held-out-original-triple OOF",
        "test_is_diagnostic_only": True,
        "test_used_to_modify_threshold_or_method": False,
        "metrics": json_value(summary.to_dict(orient="records")),
        "source_audit": audits,
        "output_hashes": {
            "confidence_harm_summary.csv": sha256_file(
                output_dir / "confidence_harm_summary.csv"
            ),
            "confidence_bins.csv": sha256_file(output_dir / "confidence_bins.csv"),
            "query_rows.csv": sha256_file(output_dir / "query_rows.csv"),
            "confidence_vs_harm.pdf": sha256_file(
                output_dir / "confidence_vs_harm.pdf"
            ),
        },
    }
    (output_dir / "auroc_auprc.json").write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(report_path, summary, bins, audits, output_dir, repo_root)
    print(f"[OK] wrote confidence-to-harm analysis to {output_dir}")
    print(f"[OK] wrote report to {report_path}")


if __name__ == "__main__":
    main()
