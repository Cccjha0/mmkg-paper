import argparse
import csv
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "matplotlib is required for this script. Install it with `pip install matplotlib`."
    ) from exc


DEFAULT_LOGISTIC_CSV = Path("outputs/router/eval/threshold_scan_logistic_delta_0.01.csv")
DEFAULT_XGB_CSV = Path("outputs/router/eval/threshold_scan_xgb_delta_0.01.csv")
DEFAULT_PDF = Path("docs/paper/figures/threshold_scan_router.pdf")
DEFAULT_PNG = Path("docs/paper/figures/threshold_scan_router.png")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sorted_metric(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    ordered = sorted(rows, key=lambda row: float(row["tau"]))
    taus = [float(row["tau"]) for row in ordered]
    values = [float(row[key]) for row in ordered]
    return taus, values


def best_point(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: float(row["overall_mrr"]))


def build_figure(logistic_rows: list[dict], xgb_rows: list[dict]) -> plt.Figure:
    taus_log, logistic_mrr = sorted_metric(logistic_rows, "overall_mrr")
    taus_xgb, xgb_mrr = sorted_metric(xgb_rows, "overall_mrr")
    _, xgb_coverage = sorted_metric(xgb_rows, "fusion_coverage")
    _, xgb_precision = sorted_metric(xgb_rows, "gain_precision")

    best = best_point(xgb_rows)
    best_tau = float(best["tau"])
    best_mrr = float(best["overall_mrr"])

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

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    ax = axes[0]
    ax.plot(
        taus_log,
        logistic_mrr,
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        color="#1f77b4",
        label="Logistic",
    )
    ax.plot(
        taus_xgb,
        xgb_mrr,
        marker="s",
        linewidth=2.0,
        markersize=5.5,
        color="#d62728",
        label="XGBoost",
    )
    ax.scatter([best_tau], [best_mrr], s=42, color="#d62728", zorder=4)
    ax.annotate(
        f"Peak = {best_mrr:.4f}\nat $\\tau$ = {best_tau:.1f}",
        xy=(best_tau, best_mrr),
        xytext=(best_tau - 0.31, best_mrr - 0.028),
        fontsize=9.0,
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#555555"},
        bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "#c8c8c8", "lw": 0.8},
    )
    ax.set_xlabel("Routing threshold $\\tau$")
    ax.set_ylabel("MRR")
    ax.set_ylim(0.23, 0.323)
    ax.set_xticks(taus_xgb)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.22, color="#999999")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(a) MRR versus routing threshold")

    ax = axes[1]
    ax.plot(
        taus_xgb,
        xgb_coverage,
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        color="#2ca02c",
        label="Fusion coverage",
    )
    ax.plot(
        taus_xgb,
        xgb_precision,
        marker="^",
        linewidth=1.9,
        linestyle="--",
        markersize=5.8,
        color="#9467bd",
        label="Gain precision",
    )
    ax.set_xlabel("Routing threshold $\\tau$")
    ax.set_ylabel("Coverage / precision")
    ax.set_ylim(0.0, 0.9)
    ax.set_xticks(taus_xgb)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.22, color="#999999")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(b) Coverage-precision tradeoff (XGBoost)")

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the paper-ready threshold-scan figure from router evaluation CSV files."
    )
    parser.add_argument("--logistic-csv", type=Path, default=DEFAULT_LOGISTIC_CSV)
    parser.add_argument("--xgb-csv", type=Path, default=DEFAULT_XGB_CSV)
    parser.add_argument("--pdf-out", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png-out", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logistic_rows = load_rows(args.logistic_csv)
    xgb_rows = load_rows(args.xgb_csv)

    fig = build_figure(logistic_rows, xgb_rows)

    args.pdf_out.parent.mkdir(parents=True, exist_ok=True)
    args.png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf_out, bbox_inches="tight")
    fig.savefig(args.png_out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote PDF -> {args.pdf_out.as_posix()}")
    print(f"[OK] wrote PNG -> {args.png_out.as_posix()}")


if __name__ == "__main__":
    main()
