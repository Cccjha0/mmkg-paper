import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready tables for second-round routing analysis.")
    parser.add_argument("--analysis-dir", default="outputs/router/analysis")
    parser.add_argument("--out-dir", default="outputs/router/paper_tables")
    parser.add_argument("--emit-tex", action="store_true")
    return parser.parse_args()


def format_float(value, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def write_tex_table(rows: list[dict], columns: list[str], out_path: Path) -> None:
    header = " & ".join(columns) + r" \\"
    body = [" & ".join(str(row.get(col, "")) for col in columns) + r" \\" for row in rows]
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule", header, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_feature_importance_table(analysis_dir: Path, out_dir: Path, emit_tex: bool) -> None:
    rows = read_csv(analysis_dir / "xgb_feature_importance_grouped.csv")
    picked = sorted(rows, key=lambda row: float(row["aggregated_gain"]), reverse=True)
    out_rows = [
        {
            "feature_family": row["feature_family"],
            "feature_count": row["feature_count"],
            "aggregated_gain": format_float(row["aggregated_gain"]),
            "aggregated_weight": format_float(row["aggregated_weight"]),
            "aggregated_cover": format_float(row["aggregated_cover"]),
        }
        for row in picked
    ]
    csv_path = out_dir / "table_feature_importance_grouped.csv"
    write_csv(csv_path, out_rows, ["feature_family", "feature_count", "aggregated_gain", "aggregated_weight", "aggregated_cover"])
    if emit_tex:
        write_tex_table(out_rows, ["feature_family", "feature_count", "aggregated_gain"], out_dir / "table_feature_importance_grouped.tex")


def build_delta_summary_table(analysis_dir: Path, out_dir: Path, emit_tex: bool) -> None:
    rows = read_csv(analysis_dir / "delta_scan_summary.csv")
    out_rows = [
        {
            "model_type": row["model_type"],
            "delta": format_float(row["delta"], 2),
            "selected_tau": format_float(row["selected_tau"], 1),
            "best_mrr": format_float(row["best_mrr"]),
            "fusion_coverage": format_float(row["fusion_coverage"]),
            "gain_precision": format_float(row["gain_precision"]),
            "positive_label_rate_dev": format_float(row["positive_label_rate_dev"]) if row["positive_label_rate_dev"] != "" else "",
            "oracle_gap_at_best_tau": format_float(row["oracle_gap_at_best_tau"]) if row["oracle_gap_at_best_tau"] != "" else "",
        }
        for row in rows
    ]
    csv_path = out_dir / "table_delta_scan_summary.csv"
    header = [
        "model_type",
        "delta",
        "selected_tau",
        "best_mrr",
        "fusion_coverage",
        "gain_precision",
        "positive_label_rate_dev",
        "oracle_gap_at_best_tau",
    ]
    write_csv(csv_path, out_rows, header)
    if emit_tex:
        write_tex_table(out_rows, ["model_type", "delta", "selected_tau", "best_mrr", "gain_precision"], out_dir / "table_delta_scan_summary.tex")


def build_logistic_table(analysis_dir: Path, out_dir: Path, emit_tex: bool) -> None:
    rows = read_csv(analysis_dir / "logistic_top_coefficients.csv")
    positive = [row for row in rows if row["direction"] == "positive_for_fusion"]
    negative = [row for row in rows if row["direction"] == "negative_for_fusion"]
    positive.sort(key=lambda row: int(row["rank_within_direction"]))
    negative.sort(key=lambda row: int(row["rank_within_direction"]))
    picked = positive[:8] + negative[:8]
    out_rows = [
        {
            "feature_name": row["feature_name"],
            "coefficient": format_float(row["coefficient"]),
            "direction": row["direction"],
            "rank_abs": row["rank_abs"],
        }
        for row in picked
    ]
    csv_path = out_dir / "table_logistic_coefficients.csv"
    write_csv(csv_path, out_rows, ["feature_name", "coefficient", "direction", "rank_abs"])
    if emit_tex:
        write_tex_table(out_rows, ["feature_name", "coefficient", "direction"], out_dir / "table_logistic_coefficients.tex")


def write_caption_notes(out_dir: Path) -> None:
    lines = [
        "# Router V2 Caption Notes",
        "",
        "- `table_feature_importance_grouped.csv`: grouped XGBoost importance under the routing-compatible line; use as interpretability support rather than as a causal claim.",
        "- `table_delta_scan_summary.csv`: protocol-specific configuration summary for gain-margin selection; do not mix with the official model-comparison line.",
        "- `table_logistic_coefficients.csv`: auxiliary linear interpretability evidence; use as a supporting table rather than a main result table.",
    ]
    (out_dir / "caption_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_feature_importance_table(analysis_dir, out_dir, args.emit_tex)
    build_delta_summary_table(analysis_dir, out_dir, args.emit_tex)
    build_logistic_table(analysis_dir, out_dir, args.emit_tex)
    write_caption_notes(out_dir)

    print(f"[OK] wrote paper tables -> {out_dir.as_posix()}")


if __name__ == "__main__":
    main()
