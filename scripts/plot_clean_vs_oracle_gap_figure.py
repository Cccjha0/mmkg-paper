import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "matplotlib is required for this script. Install it with `pip install matplotlib`."
    ) from exc


DEFAULT_OUT_DIR = Path("docs/paper/figures")
DEFAULT_STEM = "Figure_clean_vs_oracle_gap"


METHODS = [
    "Residual-only",
    "Clean rule",
    "Naive global\nclean",
    "Direction-specific\nthreshold",
    "Regression\nclean",
    "Hard-selection\nOracle",
]


MRR_VALUES = [
    0.29304943039096304,
    0.29428194121180534,
    0.293863770631994,
    0.29743590499633776,
    0.29818238528001295,
    0.3337373406732906,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the clean-routing progression versus Oracle for the paper figure."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def build_figure() -> plt.Figure:
    delta_e1_rule = MRR_VALUES[3] - MRR_VALUES[1]
    delta_e5_rule = MRR_VALUES[4] - MRR_VALUES[1]
    delta_oracle_e5 = MRR_VALUES[5] - MRR_VALUES[4]

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = list(range(len(METHODS)))

    ax.plot(
        x[:5],
        MRR_VALUES[:5],
        marker="o",
        linewidth=1.2,
        markersize=4.5,
        color="#1f77b4",
        label="Clean routing progression",
    )

    ax.scatter(
        x[5],
        MRR_VALUES[5],
        marker="D",
        s=36,
        color="#d62728",
        label="Hard-selection Oracle",
        zorder=3,
    )

    ax.axhline(MRR_VALUES[1], linestyle="--", linewidth=0.8, alpha=0.6, color="#777777")
    ax.axhline(MRR_VALUES[5], linestyle="--", linewidth=0.8, alpha=0.6, color="#777777")

    for i, value in enumerate(MRR_VALUES):
        ax.text(
            i,
            value + 0.0009,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.annotate(
        f"+{delta_e1_rule:.4f} vs. clean rule",
        xy=(3, MRR_VALUES[3]),
        xytext=(2.1, MRR_VALUES[3] + 0.009),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8.5,
    )

    ax.annotate(
        f"+{delta_e5_rule:.4f} vs. clean rule",
        xy=(4, MRR_VALUES[4]),
        xytext=(3.25, MRR_VALUES[4] + 0.006),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8.5,
    )

    ax.annotate(
        f"remaining gap: {delta_oracle_e5:.4f}",
        xy=(5, MRR_VALUES[5]),
        xytext=(3.85, MRR_VALUES[5] - 0.006),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8.5,
    )

    ax.set_ylabel("MRR")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(0.288, 0.338)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    set_style()

    fig = build_figure()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    svg_path = args.out_dir / f"{args.stem}.svg"
    png_path = args.out_dir / f"{args.stem}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote PDF -> {pdf_path.as_posix()}")
    print(f"[OK] wrote SVG -> {svg_path.as_posix()}")
    print(f"[OK] wrote PNG -> {png_path.as_posix()}")


if __name__ == "__main__":
    main()
