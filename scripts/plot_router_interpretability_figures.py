import argparse
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for this script. Install it with `pip install matplotlib`.") from exc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-ready interpretability figures for second-round routing analysis.")
    parser.add_argument("--xgb-grouped", default="outputs/router/analysis/xgb_feature_importance_grouped.csv")
    parser.add_argument("--delta-summary", default="outputs/router/analysis/delta_scan_summary.csv")
    parser.add_argument("--logistic-coef", default="outputs/router/analysis/logistic_top_coefficients.csv")
    parser.add_argument("--out-dir", default="outputs/router/figures")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_both(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote PDF -> {pdf_path.as_posix()}")
    print(f"[OK] wrote PNG -> {png_path.as_posix()}")


def plot_xgb_grouped(rows: list[dict]) -> plt.Figure:
    picked = sorted(rows, key=lambda row: float(row["aggregated_gain"]), reverse=True)[:12]
    labels = [row["feature_family"] for row in picked][::-1]
    values = [float(row["aggregated_gain"]) for row in picked][::-1]

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.barh(labels, values, color="#2ca02c")
    ax.set_xlabel("Aggregated gain importance")
    ax.set_title("Grouped XGBoost feature importance")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.22, color="#999999")
    fig.tight_layout()
    return fig


def plot_delta_summary(rows: list[dict]) -> plt.Figure:
    picked = [row for row in rows if row["model_type"] == "xgb"]
    picked.sort(key=lambda row: float(row["delta"]))
    deltas = [float(row["delta"]) for row in picked]
    best_mrr = [float(row["best_mrr"]) for row in picked]
    label_rate = [float(row["positive_label_rate_dev"]) for row in picked if row["positive_label_rate_dev"] != ""]

    fig, ax1 = plt.subplots(figsize=(8.4, 4.9))
    ax2 = ax1.twinx()

    ax1.plot(deltas, best_mrr, marker="o", linewidth=2.0, color="#d62728", label="Best MRR")
    ax1.set_xlabel("Gain margin $\\delta$")
    ax1.set_ylabel("Best MRR")
    ax1.set_xticks(deltas)
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.22, color="#999999")

    if len(label_rate) == len(deltas):
        ax2.plot(deltas, label_rate, marker="s", linestyle="--", linewidth=1.8, color="#1f77b4", label="Positive label rate")
        ax2.set_ylabel("Dev positive-label rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")
    ax1.set_title("Delta scan summary (XGBoost)")
    fig.tight_layout()
    return fig


def plot_logistic_coefficients(rows: list[dict]) -> plt.Figure:
    positive = [row for row in rows if row["direction"] == "positive_for_fusion"]
    negative = [row for row in rows if row["direction"] == "negative_for_fusion"]
    positive.sort(key=lambda row: float(row["coefficient"]), reverse=True)
    negative.sort(key=lambda row: float(row["coefficient"]))
    picked = positive[:8] + negative[:8]
    picked.sort(key=lambda row: float(row["coefficient"]))

    labels = [row["feature_name"] for row in picked]
    values = [float(row["coefficient"]) for row in picked]
    colors = ["#d62728" if value > 0 else "#1f77b4" for value in values]

    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    ax.barh(labels, values, color=colors)
    ax.axvline(0.0, color="#555555", linewidth=1.0)
    ax.set_xlabel("Coefficient")
    ax.set_title("Logistic top coefficients")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.22, color="#999999")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    set_style()

    xgb_rows = read_csv(args.xgb_grouped)
    delta_rows = read_csv(args.delta_summary)
    logistic_rows = read_csv(args.logistic_coef)
    out_dir = Path(args.out_dir)

    save_both(plot_xgb_grouped(xgb_rows), out_dir, "xgb_feature_importance_grouped", args.dpi)
    save_both(plot_delta_summary(delta_rows), out_dir, "delta_scan_summary", args.dpi)
    save_both(plot_logistic_coefficients(logistic_rows), out_dir, "logistic_top_coefficients", args.dpi)


if __name__ == "__main__":
    main()
