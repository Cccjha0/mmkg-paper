from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate-router paper-ready result tables.")
    parser.add_argument("--baseline-summary", default="outputs/router/eval/clean/baseline_locked_summary.csv")
    parser.add_argument("--candidate-eval-glob", default="outputs/candidate_router/eval/ca_s3_full_ranking_seed*_eval.csv")
    parser.add_argument(
        "--candidate-by-regime-glob",
        default="outputs/candidate_router/eval/ca_s3_full_ranking_seed*_by_regime.csv",
    )
    parser.add_argument("--significance-dir", default="outputs/candidate_router/eval")
    parser.add_argument("--out-dir", default="outputs/candidate_router/eval/tables")
    parser.add_argument("--paper-table-dir", default="docs/paper_tables")
    return parser.parse_args()


def mean_std(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4f}"


def fmt_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(col) for col in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False)]
    widths = [
        max(len(columns[i]), *(len(row[i]) for row in rows)) if rows else len(columns[i])
        for i in range(len(columns))
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    lines = [
        fmt_row(columns),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def write_latex_table(
    path: Path,
    frame: pd.DataFrame,
    caption: str,
    label: str,
    column_spec: str,
    note: str | None = None,
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(latex_escape(col) for col in frame.columns) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if note:
        lines.append(rf"\vspace{{0.4ex}}\caption*{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_candidate_eval(pattern: str) -> pd.DataFrame:
    paths = sorted(Path().glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No candidate eval files matched: {pattern}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_main_table(baseline: pd.DataFrame, candidate_eval: pd.DataFrame) -> pd.DataFrame:
    by_method = {row["method"]: row for _, row in baseline.iterrows()}
    residual = float(by_method["Residual-only"]["mrr"])
    e5 = float(by_method["Regression-based clean router"]["mrr"])
    oracle = float(by_method["Oracle routing"]["mrr"])
    oracle_gap = oracle - residual

    ca_mean, ca_std = mean_std(candidate_eval["full_ranking_mrr"])
    ca_hits1, ca_hits1_std = mean_std(candidate_eval["full_ranking_hits1"])
    ca_hits3, ca_hits3_std = mean_std(candidate_eval["full_ranking_hits3"])
    ca_hits10, ca_hits10_std = mean_std(candidate_eval["full_ranking_hits10"])

    rows = []

    def add_baseline(method: str, level: str, feature_type: str, objective: str) -> None:
        mrr = float(by_method[method]["mrr"])
        hits1 = float(by_method[method]["hits1"])
        hits3 = float(by_method[method]["hits3"])
        hits10 = float(by_method[method]["hits10"])
        recovery = (mrr - residual) / oracle_gap if oracle_gap > 0 else 0.0
        rows.append(
            {
                "Method": method,
                "Routing level": level,
                "Feature type": feature_type,
                "Objective": objective,
                "MRR": fmt(mrr),
                "Hits@1": fmt(hits1),
                "Hits@3": fmt(hits3),
                "Hits@10": fmt(hits10),
                "Delta vs Residual": "0.0000" if method == "Residual-only" else fmt_delta(mrr - residual),
                "Delta vs E5": "0.0000" if method == "Regression-based clean router" else fmt_delta(mrr - e5),
                "Oracle gap recovered": fmt_percent(recovery),
            }
        )

    add_baseline("Residual-only", "fixed", "structural", "none")
    add_baseline("Clean rule", "query", "strict clean", "rule")
    add_baseline("Direction-specific threshold", "query", "strict clean", "threshold")
    add_baseline("Regression-based clean router", "query", "strict clean", "delta-RR regression")
    rows.append(
        {
            "Method": "CA-S3 full-ranking",
            "Routing level": "candidate",
            "Feature type": "clean + score-aware",
            "Objective": "pairwise ranking",
            "MRR": f"{fmt(ca_mean)} $\\pm$ {fmt(ca_std)}",
            "Hits@1": f"{fmt(ca_hits1)} $\\pm$ {fmt(ca_hits1_std)}",
            "Hits@3": f"{fmt(ca_hits3)} $\\pm$ {fmt(ca_hits3_std)}",
            "Hits@10": f"{fmt(ca_hits10)} $\\pm$ {fmt(ca_hits10_std)}",
            "Delta vs Residual": fmt_delta(ca_mean - residual),
            "Delta vs E5": fmt_delta(ca_mean - e5),
            "Oracle gap recovered": fmt_percent((ca_mean - residual) / oracle_gap),
        }
    )
    add_baseline("Oracle routing", "oracle", "answer-aware", "post-hoc max")
    return pd.DataFrame(rows)


def build_significance_table(significance_dir: Path) -> pd.DataFrame:
    order = [
        "significance_ca_s3_vs_residual.csv",
        "significance_ca_s3_vs_clean_rule.csv",
        "significance_ca_s3_vs_direction_specific.csv",
        "significance_ca_s3_vs_e5.csv",
    ]
    rows = []
    for name in order:
        path = significance_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        row = pd.read_csv(path).iloc[0]
        rows.append(
            {
                "Comparison": str(row["comparison"]).replace("CA-S3 full-ranking vs ", "vs "),
                "Baseline MRR": fmt(float(row["baseline_mrr"])),
                "CA-S3 MRR": fmt(float(row["candidate_mrr"])),
                "Delta MRR": fmt_delta(float(row["mean_delta_mrr_querywise"])),
                "95% bootstrap CI": (
                    f"[{fmt_delta(float(row['bootstrap_ci_low']))}, "
                    f"{fmt_delta(float(row['bootstrap_ci_high']))}]"
                ),
                "Paired queries": int(row["n_paired_queries"]),
            }
        )
    return pd.DataFrame(rows)


def build_subgroup_table(by_regime: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, group in by_regime.groupby("scope", sort=False):
        ca_mean, ca_std = mean_std(group["full_ranking_mrr"])
        residual_mean, residual_std = mean_std(group["residual_full_mrr"])
        delta_mean, delta_std = mean_std(group["delta_vs_residual_full"])
        gate_mean, _ = mean_std(group["gate_full_mrr"])
        count = int(group["count"].iloc[0])
        rows.append(
            {
                "Regime": regime,
                "Count/seed": count,
                "Gate MRR": fmt(gate_mean),
                "Residual MRR": f"{fmt(residual_mean)} $\\pm$ {fmt(residual_std)}",
                "CA-S3 MRR": f"{fmt(ca_mean)} $\\pm$ {fmt(ca_std)}",
                "Delta vs Residual": f"{fmt_delta(delta_mean)} $\\pm$ {fmt(delta_std)}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    paper_dir = Path(args.paper_table_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(args.baseline_summary)
    candidate_eval = load_candidate_eval(args.candidate_eval_glob)
    by_regime = load_candidate_eval(args.candidate_by_regime_glob)

    main_table = build_main_table(baseline, candidate_eval)
    significance_table = build_significance_table(Path(args.significance_dir))
    subgroup_table = build_subgroup_table(by_regime)

    outputs = {
        "candidate_router_main_results": main_table,
        "candidate_router_significance": significance_table,
        "candidate_router_subgroup_results": subgroup_table,
    }
    for stem, frame in outputs.items():
        frame.to_csv(out_dir / f"{stem}.csv", index=False)
        (out_dir / f"{stem}.md").write_text(markdown_table(frame) + "\n", encoding="utf-8")

    write_latex_table(
        paper_dir / "table_candidate_router_main_results.tex",
        main_table,
        caption=(
            "Candidate-aware full-ranking router results on OpenBG-IMG. "
            "CA-S3 is evaluated by full filtered ranking over three seeds."
        ),
        label="tab:candidate_router_main_results",
        column_spec="p{0.20\\textwidth}p{0.10\\textwidth}p{0.13\\textwidth}p{0.14\\textwidth}ccccccc",
        note=(
            "E5 denotes the regression-based clean router. "
            "Oracle routing is answer-aware and is shown only as an upper bound."
        ),
    )
    write_latex_table(
        paper_dir / "table_candidate_router_significance.tex",
        significance_table,
        caption="Paired bootstrap significance for CA-S3 full-ranking.",
        label="tab:candidate_router_significance",
        column_spec="p{0.28\\textwidth}cccp{0.22\\textwidth}c",
        note="Confidence intervals use 2,000 paired bootstrap resamples over matched query-level reciprocal-rank deltas.",
    )
    write_latex_table(
        paper_dir / "table_candidate_router_subgroup_results.tex",
        subgroup_table,
        caption="CA-S3 full-ranking results by target-side regime.",
        label="tab:candidate_router_subgroup_results",
        column_spec="p{0.18\\textwidth}ccccc",
        note="The overall gain is primarily driven by the tail-side regime, while head-side gains remain small in absolute MRR.",
    )

    print(f"[OK] wrote tables under {out_dir.as_posix()}")
    print(f"[OK] wrote LaTeX tables under {paper_dir.as_posix()}")


if __name__ == "__main__":
    main()
