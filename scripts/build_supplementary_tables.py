from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUT_DIR = Path("docs/paper_tables/supplementary")


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace(">", r"$>$")
    )


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}\\%"


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_s1() -> None:
    source = Path("docs/relation_type_summary_min20.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    groups = [
        ("visual_relations", "Visual relations"),
        ("weak_visual_relations", "Weak visual relations"),
        ("ambiguous_material_relations", "Ambiguous/material relations"),
    ]
    model_keys = ["Residual-only", "Full Model", "Gate-only"]
    rows = []
    for group_key, label in groups:
        group = data["groups"][group_key]
        metrics = {}
        for model in model_keys:
            stats = data["models"][model]["group_stats"][group_key]["stats"]
            metrics[model] = float(stats["mrr"]["mean"])
        best = max(model_keys, key=lambda item: metrics[item])
        rows.append(
            {
                "group": label,
                "queries": int(group["triple_count_in_test"]) * 2,
                "residual": metrics["Residual-only"],
                "full": metrics["Full Model"],
                "gate": metrics["Gate-only"],
                "best": best,
                "interpretation": "ordering preserved; structural expert remains strongest",
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "table_s1_relation_group_sanity_check.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Relation-group sanity check for fixed experts.}",
        r"\label{tab:supp_relation_group_sanity}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrp{0.28\textwidth}}",
        r"\toprule",
        r"Relation group & \#Queries & Residual-only & Full Model & Gate-only & Interpretation \\",
        r"\midrule",
    ]
    for row in rows:
        values = {
            "Residual-only": fmt4(row["residual"]),
            "Full Model": fmt4(row["full"]),
            "Gate-only": fmt4(row["gate"]),
        }
        values[row["best"]] = r"\textbf{" + values[row["best"]] + "}"
        lines.append(
            f"{tex_escape(row['group'])} & {row['queries']:,} & {values['Residual-only']} & "
            f"{values['Full Model']} & {values['Gate-only']} & {row['interpretation']} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4ex}",
            (
                r"\caption*{\footnotesize \textit{Note:} Query counts are bidirectional counts "
                r"derived from the test triples in each relation group. MRR values are means over "
                r"three seed-specific evaluations from the locked fixed-expert runs.}"
            ),
            r"\end{table}",
        ]
    )
    write(OUT_DIR / "table_s1_relation_group_sanity_check.tex", lines)


def build_s2() -> None:
    source_csv = Path("docs/paper_tables/table_degree_bucket_sanity_check.csv")
    source_tex = Path("docs/paper_tables/table_degree_bucket_sanity_check.tex")
    if not source_csv.exists() or not source_tex.exists():
        raise FileNotFoundError("Run scripts/build_degree_bucket_sanity_table.py before building S2.")
    (OUT_DIR / "table_s2_degree_bucket_sanity_check.csv").write_text(
        source_csv.read_text(encoding="utf-8"), encoding="utf-8"
    )
    tex = source_tex.read_text(encoding="utf-8")
    tex = tex.replace(
        r"\label{tab:degree_bucket_sanity_check}",
        r"\label{tab:supp_degree_bucket_sanity}",
    )
    write(OUT_DIR / "table_s2_degree_bucket_sanity_check.tex", tex.rstrip("\n").splitlines())


def build_s3() -> None:
    clean = pd.read_csv("outputs/router/eval/clean/baseline_locked_summary.csv")
    ordinal = pd.read_csv("outputs/router/eval/clean/ordinal_router_scan_xgb_C4.csv")
    regression = pd.read_csv("outputs/router/eval/clean/regression_router_scan_xgb_C4.csv")

    clean_rule = clean.loc[clean["method"] == "Clean rule"].iloc[0]
    direction = clean.loc[clean["method"] == "Direction-specific threshold"].iloc[0]
    regression_best = regression.loc[regression["theta"] == 0.0].iloc[0]
    ordinal_weak = ordinal.loc[ordinal["decision_rule"] == "weak_or_strong_positive"].iloc[0]
    ordinal_strong = ordinal.loc[ordinal["decision_rule"] == "strong_only"].iloc[0]
    baseline_mrr = float(clean_rule["mrr"])

    rows = [
        {
            "target": "Binary clean gain",
            "policy": "Clean rule",
            "mrr": float(clean_rule["mrr"]),
            "coverage": float(clean_rule["fusion_coverage"]),
            "interpretation": "sparse conservative baseline",
        },
        {
            "target": "Binary clean gain",
            "policy": "Direction-specific threshold",
            "mrr": float(direction["mrr"]),
            "coverage": float(direction["fusion_coverage"]),
            "interpretation": "structured threshold improves coverage",
        },
        {
            "target": "Ordinal gain label",
            "policy": "XGB, weak-or-strong positive",
            "mrr": float(ordinal_weak["overall_mrr"]),
            "coverage": float(ordinal_weak["fusion_coverage"]),
            "interpretation": "ordinal supervision preserves useful magnitude signal",
        },
        {
            "target": "Ordinal gain label",
            "policy": "XGB, strong only",
            "mrr": float(ordinal_strong["overall_mrr"]),
            "coverage": float(ordinal_strong["fusion_coverage"]),
            "interpretation": "stricter ordinal activation is more conservative",
        },
        {
            "target": "Scalar gain regression",
            "policy": "XGB, theta=0",
            "mrr": float(regression_best["overall_mrr"]),
            "coverage": float(regression_best["fusion_coverage"]),
            "interpretation": "best clean router; scalar target helps most",
        },
    ]
    for row in rows:
        row["delta_vs_clean_rule"] = row["mrr"] - baseline_mrr
    pd.DataFrame(rows).to_csv(OUT_DIR / "table_s3_ordinal_gain_modeling.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ordinal and scalar gain modeling for clean routing.}",
        r"\label{tab:supp_ordinal_gain_modeling}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.18\textwidth}p{0.25\textwidth}cccp{0.24\textwidth}}",
        r"\toprule",
        r"Router target & Policy & MRR & $\Delta$ vs. clean rule & Selection rate & Interpretation \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['target'])} & {tex_escape(row['policy'])} & {fmt4(row['mrr'])} & "
            f"{fmt_delta(row['delta_vs_clean_rule'])} & {pct(row['coverage'])} & "
            f"{tex_escape(row['interpretation'])} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4ex}",
            (
                r"\caption*{\footnotesize \textit{Note:} All policies use the locked clean-routing "
                r"evaluation rows. Selection rate is the fraction of seed-query evaluations assigned "
                r"to the fusion expert.}"
            ),
            r"\end{table}",
        ]
    )
    write(OUT_DIR / "table_s3_ordinal_gain_modeling.tex", lines)


def build_s4() -> None:
    delta = pd.read_csv("outputs/router/eval/clean/delta_sensitivity_clean.csv")
    clean = pd.read_csv("outputs/router/eval/clean/baseline_locked_summary.csv")
    baseline_mrr = float(clean.loc[clean["method"] == "Clean rule", "mrr"].iloc[0])

    delta = delta.copy()
    delta["delta_vs_clean_rule"] = delta["best_overall_mrr"] - baseline_mrr
    delta["interpretation"] = delta["delta"].map(
        {
            0.0: "non-margin label gives broad activation",
            0.01: "strict positive margin becomes very conservative",
            0.02: "larger margin remains conservative",
        }
    )
    delta.to_csv(OUT_DIR / "table_s4_delta_sensitivity.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Delta sensitivity of clean learned routing.}",
        r"\label{tab:supp_delta_sensitivity}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccp{0.31\textwidth}}",
        r"\toprule",
        r"Model & Gain margin $\delta$ & Best $\tau$ & MRR & Selection rate & Interpretation \\",
        r"\midrule",
    ]
    for _, row in delta.iterrows():
        lines.append(
            f"{tex_escape(row['model'])} & {row['delta']:.2f} & {row['best_tau']:.1f} & "
            f"{fmt4(row['best_overall_mrr'])} ({fmt_delta(row['delta_vs_clean_rule'])}) & "
            f"{pct(row['best_fusion_coverage'])} & {tex_escape(row['interpretation'])} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4ex}",
            (
                r"\caption*{\footnotesize \textit{Note:} Parentheses report $\Delta$ MRR relative "
                r"to the clean rule. Larger gain margins sharply reduce fusion selection, showing "
                r"that binary clean labels are sensitive to the margin definition.}"
            ),
            r"\end{table}",
        ]
    )
    write(OUT_DIR / "table_s4_delta_sensitivity.tex", lines)


def build_s5() -> None:
    source = Path("docs/paper_tables/table_score_interpolation_alpha_stability.tex")
    if not source.exists():
        raise FileNotFoundError(source)
    tex = source.read_text(encoding="utf-8")
    tex = tex.replace(
        r"\label{tab:score_interpolation_alpha_stability}",
        r"\label{tab:supp_score_interpolation_alpha_stability}",
    )
    write(OUT_DIR / "table_s5_alpha_sweep_stability.tex", tex.rstrip("\n").splitlines())
    best = Path("outputs/score_ensemble/eval/score_ensemble_alpha_curve_best.csv")
    if best.exists():
        (OUT_DIR / "table_s5_alpha_sweep_best.csv").write_text(best.read_text(encoding="utf-8"), encoding="utf-8")


def build_manifest() -> None:
    lines = [
        "# Supplementary Table Bundle",
        "",
        "| Table | TeX file | Source data | Status |",
        "|---|---|---|---|",
        "| S1 | `table_s1_relation_group_sanity_check.tex` | `docs/relation_type_summary_min20.json` | generated |",
        "| S2 | `table_s2_degree_bucket_sanity_check.tex` | `docs/paper_tables/table_degree_bucket_sanity_check.csv` | generated |",
        "| S3 | `table_s3_ordinal_gain_modeling.tex` | clean router ordinal/regression CSVs | generated |",
        "| S4 | `table_s4_delta_sensitivity.tex` | `outputs/router/eval/clean/delta_sensitivity_clean.csv` | generated |",
        "| S5 | `table_s5_alpha_sweep_stability.tex` | `docs/paper_tables/table_score_interpolation_alpha_stability.tex` | generated |",
        "",
        "S4 should be described as margin sensitivity rather than as full robustness across many deltas; the existing data covers delta values 0.00, 0.01, and 0.02.",
    ]
    write(OUT_DIR / "README.md", lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_s1()
    build_s2()
    build_s3()
    build_s4()
    build_s5()
    build_manifest()
    print(f"[OK] wrote supplementary tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
