from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate-router paper-ready result tables.")
    parser.add_argument("--baseline-summary", default="outputs/router/eval/clean/baseline_locked_summary.csv")
    parser.add_argument("--candidate-eval-glob", default="outputs/candidate_router/eval/ca_s*_full_ranking_seed*_eval.csv")
    parser.add_argument(
        "--candidate-by-regime-glob",
        default="outputs/candidate_router/eval/ca_s*_full_ranking_seed*_by_regime.csv",
    )
    parser.add_argument(
        "--candidate-query-row-glob",
        default="outputs/candidate_router/eval/ca_s*_full_ranking_seed*_query_rows.csv",
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


def load_candidate_query_rows(pattern: str) -> pd.DataFrame:
    paths = sorted(Path().glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No candidate query-row files matched: {pattern}")
    frames = []
    usecols = [
        "seed",
        "target_regime",
        "mixed_rr",
        "residual_rr",
        "gate_rr",
        "target_alpha",
        "mean_alpha",
    ]
    for path in paths:
        frame = pd.read_csv(path, usecols=usecols)
        if "ca_s1_" in path.name:
            frame["feature_set"] = "CA-S1"
        elif "ca_s2_" in path.name:
            frame["feature_set"] = "CA-S2"
        elif "ca_s3_" in path.name:
            frame["feature_set"] = "CA-S3"
        else:
            raise ValueError(f"Cannot infer feature set from filename: {path.name}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def method_label(feature_set: str) -> str:
    return {
        "CA-S1": "CA-S1 clean candidate",
        "CA-S2": "CA-S2 score-aware",
        "CA-S3": "CA-S3 clean + score",
    }.get(feature_set, feature_set)


def compact_method_label(text: str) -> str:
    return (
        text.replace(" clean candidate full-ranking", "")
        .replace(" score-aware full-ranking", "")
        .replace(" clean + score-aware full-ranking", "")
        .replace("clean + score-aware", "")
        .strip()
    )


def method_feature_type(feature_set: str) -> str:
    return {
        "CA-S1": "clean candidate",
        "CA-S2": "score-aware",
        "CA-S3": "clean + score-aware",
    }.get(feature_set, "candidate")


def build_main_table(baseline: pd.DataFrame, candidate_eval: pd.DataFrame) -> pd.DataFrame:
    by_method = {row["method"]: row for _, row in baseline.iterrows()}
    residual = float(by_method["Residual-only"]["mrr"])
    e5 = float(by_method["Regression-based clean router"]["mrr"])
    oracle = float(by_method["Oracle routing"]["mrr"])
    oracle_gap = oracle - residual

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
    for feature_set in ["CA-S1", "CA-S2", "CA-S3"]:
        group = candidate_eval[candidate_eval["feature_set"].eq(feature_set)]
        if group.empty:
            continue
        ca_mean, ca_std = mean_std(group["full_ranking_mrr"])
        ca_hits1, ca_hits1_std = mean_std(group["full_ranking_hits1"])
        ca_hits3, ca_hits3_std = mean_std(group["full_ranking_hits3"])
        ca_hits10, ca_hits10_std = mean_std(group["full_ranking_hits10"])
        rows.append(
            {
                "Method": method_label(feature_set),
                "Routing level": "candidate",
                "Feature type": method_feature_type(feature_set),
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
        "significance_ca_s1_vs_residual.csv",
        "significance_ca_s1_vs_e5.csv",
        "significance_ca_s2_vs_residual.csv",
        "significance_ca_s2_vs_e5.csv",
        "significance_ca_s3_vs_residual.csv",
        "significance_ca_s3_vs_e5.csv",
        "significance_ca_s2_vs_ca_s3.csv",
    ]
    rows = []
    for name in order:
        path = significance_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        row = pd.read_csv(path).iloc[0]
        rows.append(
            {
                "Comparison": str(row["comparison"])
                .replace(str(row["left_label"]), compact_method_label(str(row["left_label"])))
                .replace(str(row["right_label"]), compact_method_label(str(row["right_label"]))),
                "Baseline MRR": fmt(float(row["baseline_mrr"])),
                "Candidate MRR": fmt(float(row["candidate_mrr"])),
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
    for (feature_set, regime), group in by_regime.groupby(["feature_set", "scope"], sort=False):
        ca_mean, ca_std = mean_std(group["full_ranking_mrr"])
        residual_mean, residual_std = mean_std(group["residual_full_mrr"])
        delta_mean, delta_std = mean_std(group["delta_vs_residual_full"])
        gate_mean, _ = mean_std(group["gate_full_mrr"])
        count = int(group["count"].iloc[0])
        rows.append(
            {
                "Method": method_label(feature_set),
                "Regime": regime,
                "Count/seed": count,
                "Gate MRR": fmt(gate_mean),
                "Residual MRR": f"{fmt(residual_mean)} $\\pm$ {fmt(residual_std)}",
                "Method MRR": f"{fmt(ca_mean)} $\\pm$ {fmt(ca_std)}",
                "Delta vs Residual": f"{fmt_delta(delta_mean)} $\\pm$ {fmt(delta_std)}",
            }
        )
    return pd.DataFrame(rows)


def build_alpha_behavior_table(query_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = query_rows.groupby(["feature_set", "target_regime"], sort=False)
    for (feature_set, regime), group in grouped:
        by_seed = (
            group.groupby("seed")
            .agg(
                mean_alpha=("mean_alpha", "mean"),
                target_alpha=("target_alpha", "mean"),
                method_mrr=("mixed_rr", "mean"),
                residual_mrr=("residual_rr", "mean"),
                gate_mrr=("gate_rr", "mean"),
            )
            .reset_index()
        )
        mean_alpha, mean_alpha_std = mean_std(by_seed["mean_alpha"])
        target_alpha, target_alpha_std = mean_std(by_seed["target_alpha"])
        gain = by_seed["method_mrr"] - by_seed["residual_mrr"]
        gain_mean, gain_std = mean_std(gain)
        gate_delta = by_seed["method_mrr"] - by_seed["gate_mrr"]
        gate_delta_mean, _ = mean_std(gate_delta)
        rows.append(
            {
                "Method": method_label(feature_set),
                "Regime": regime,
                "Mean alpha": f"{fmt(mean_alpha)} $\\pm$ {fmt(mean_alpha_std)}",
                "Target alpha": f"{fmt(target_alpha)} $\\pm$ {fmt(target_alpha_std)}",
                "Target/mean alpha": fmt(target_alpha / mean_alpha) if mean_alpha > 0 else "n/a",
                "MRR gain vs Residual": f"{fmt_delta(gain_mean)} $\\pm$ {fmt(gain_std)}",
                "MRR delta vs Gate": fmt_delta(gate_delta_mean),
            }
        )
    return pd.DataFrame(rows)


def build_alpha_overall_table(query_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set, group in query_rows.groupby("feature_set", sort=False):
        by_seed = (
            group.groupby("seed")
            .agg(
                mean_alpha=("mean_alpha", "mean"),
                target_alpha=("target_alpha", "mean"),
                method_mrr=("mixed_rr", "mean"),
                residual_mrr=("residual_rr", "mean"),
            )
            .reset_index()
        )
        mean_alpha, mean_alpha_std = mean_std(by_seed["mean_alpha"])
        target_alpha, target_alpha_std = mean_std(by_seed["target_alpha"])
        gain = by_seed["method_mrr"] - by_seed["residual_mrr"]
        gain_mean, gain_std = mean_std(gain)
        rows.append(
            {
                "Method": method_label(feature_set),
                "Mean alpha": f"{fmt(mean_alpha)} $\\pm$ {fmt(mean_alpha_std)}",
                "Target alpha": f"{fmt(target_alpha)} $\\pm$ {fmt(target_alpha_std)}",
                "Target/mean alpha": fmt(target_alpha / mean_alpha) if mean_alpha > 0 else "n/a",
                "MRR gain vs Residual": f"{fmt_delta(gain_mean)} $\\pm$ {fmt(gain_std)}",
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
    query_rows = load_candidate_query_rows(args.candidate_query_row_glob)

    main_table = build_main_table(baseline, candidate_eval)
    significance_table = build_significance_table(Path(args.significance_dir))
    subgroup_table = build_subgroup_table(by_regime)
    alpha_behavior_table = build_alpha_behavior_table(query_rows)
    alpha_overall_table = build_alpha_overall_table(query_rows)

    outputs = {
        "candidate_router_main_results": main_table,
        "candidate_router_significance": significance_table,
        "candidate_router_subgroup_results": subgroup_table,
        "candidate_router_alpha_behavior": alpha_behavior_table,
        "candidate_router_alpha_overall": alpha_overall_table,
    }
    for stem, frame in outputs.items():
        frame.to_csv(out_dir / f"{stem}.csv", index=False)
        (out_dir / f"{stem}.md").write_text(markdown_table(frame) + "\n", encoding="utf-8")

    write_latex_table(
        paper_dir / "table_candidate_router_main_results.tex",
        main_table,
        caption=(
            "Candidate-aware full-ranking router results on OpenBG-IMG. "
            "All candidate-aware variants are evaluated by full filtered ranking over three seeds."
        ),
        label="tab:candidate_router_main_results",
        column_spec="p{0.20\\textwidth}p{0.10\\textwidth}p{0.13\\textwidth}p{0.14\\textwidth}ccccccc",
        note=(
            "E5 denotes the regression-based clean router. "
            "Oracle routing is answer-aware and is shown only as an upper bound. "
            "CA-S2 and CA-S3 show that score-aware features are the dominant source of improvement."
        ),
    )
    write_latex_table(
        paper_dir / "table_candidate_router_significance.tex",
        significance_table,
        caption="Paired bootstrap significance for candidate-aware full-ranking routers.",
        label="tab:candidate_router_significance",
        column_spec="p{0.30\\textwidth}cccp{0.22\\textwidth}c",
        note="Confidence intervals use 2,000 paired bootstrap resamples over matched query-level reciprocal-rank deltas.",
    )
    write_latex_table(
        paper_dir / "table_candidate_router_subgroup_results.tex",
        subgroup_table,
        caption="Candidate-aware full-ranking results by target-side regime.",
        label="tab:candidate_router_subgroup_results",
        column_spec="p{0.18\\textwidth}p{0.14\\textwidth}ccccc",
        note="The overall gain is primarily driven by the tail-side regime, while head-side gains remain small in absolute MRR.",
    )
    write_latex_table(
        paper_dir / "table_candidate_router_alpha_behavior.tex",
        alpha_behavior_table,
        caption="Alpha behavior of candidate-aware full-ranking routers by target-side regime.",
        label="tab:candidate_router_alpha_behavior",
        column_spec="p{0.17\\textwidth}p{0.13\\textwidth}ccccc",
        note=(
            "Mean alpha averages the router weight over all ranked candidates, whereas target alpha is the weight assigned "
            "to the correct entity. Values show that score-aware routers remain mostly structural over the candidate set "
            "but assign larger fusion weight to target candidates."
        ),
    )
    write_latex_table(
        paper_dir / "table_candidate_router_alpha_overall.tex",
        alpha_overall_table,
        caption="Overall alpha behavior of candidate-aware full-ranking routers.",
        label="tab:candidate_router_alpha_overall",
        column_spec="p{0.22\\textwidth}cccc",
        note="The target/mean alpha ratio summarizes how selectively the router raises fusion weight on the correct entity.",
    )

    print(f"[OK] wrote tables under {out_dir.as_posix()}")
    print(f"[OK] wrote LaTeX tables under {paper_dir.as_posix()}")


if __name__ == "__main__":
    main()
