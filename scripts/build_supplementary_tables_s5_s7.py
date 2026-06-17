from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


OUT_DIR = Path("docs/paper_tables/supplementary")


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
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


def seed_from_name(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot infer seed from {path}")
    return int(match.group(1))


def build_s5() -> None:
    model_files = {
        "Gate-only": "gate_only_query_eval_seed*.csv",
        "Full Model": "full_model_query_eval_seed*.csv",
        "Residual-only": "residual_only_query_eval_seed*.csv",
    }
    rows = []
    for model, pattern in model_files.items():
        for path in sorted(Path("outputs/router/test").glob(pattern)):
            frame = pd.read_csv(path, usecols=["rr", "hit1", "hit3", "hit10"])
            rows.append(
                {
                    "seed": seed_from_name(path),
                    "model": model,
                    "n_queries": len(frame),
                    "mrr": float(frame["rr"].mean()),
                    "hits1": float(frame["hit1"].mean()),
                    "hits3": float(frame["hit3"].mean()),
                    "hits10": float(frame["hit10"].mean()),
                    "source": str(path),
                }
            )
    long = pd.DataFrame(rows).sort_values(["seed", "model"])
    long.to_csv(OUT_DIR / "table_s5_per_seed_model_comparison_long.csv", index=False)

    pivot = long.pivot(index="seed", columns="model", values="mrr").reset_index()
    pivot["delta_residual_vs_full"] = pivot["Residual-only"] - pivot["Full Model"]
    pivot["best_model"] = pivot[["Gate-only", "Full Model", "Residual-only"]].idxmax(axis=1)

    mean_row = {
        "seed": "Mean",
        "Gate-only": long.loc[long["model"] == "Gate-only", "mrr"].mean(),
        "Full Model": long.loc[long["model"] == "Full Model", "mrr"].mean(),
        "Residual-only": long.loc[long["model"] == "Residual-only", "mrr"].mean(),
    }
    mean_row["delta_residual_vs_full"] = mean_row["Residual-only"] - mean_row["Full Model"]
    mean_row["best_model"] = max(["Gate-only", "Full Model", "Residual-only"], key=lambda name: mean_row[name])
    summary = pd.concat([pivot, pd.DataFrame([mean_row])], ignore_index=True)
    summary.to_csv(OUT_DIR / "table_s5_per_seed_model_comparison.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Per-seed fixed-expert model comparison on the test split.}",
        r"\label{tab:supp_per_seed_model_comparison}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Seed & Gate-only & Full Model & Residual-only & $\Delta$ Res.-Full & Best model \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        best = row["best_model"]
        vals = {model: fmt4(float(row[model])) for model in ["Gate-only", "Full Model", "Residual-only"]}
        vals[best] = r"\textbf{" + vals[best] + "}"
        lines.append(
            f"{row['seed']} & {vals['Gate-only']} & {vals['Full Model']} & {vals['Residual-only']} & "
            f"{fmt_delta(float(row['delta_residual_vs_full']))} & {tex_escape(best)} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4ex}",
            (
                r"\caption*{\footnotesize \textit{Note:} Each seed row averages 20,000 bidirectional test queries. "
                r"The mean row is an unweighted mean over the three seed-specific MRR values.}"
            ),
            r"\end{table}",
        ]
    )
    write(OUT_DIR / "table_s5_per_seed_model_comparison.tex", lines)


def build_s6() -> None:
    ci = pd.read_csv("outputs/score_ensemble/eval/score_aware_bootstrap_ci.csv")
    n_bootstrap = int(ci["n_bootstrap"].iloc[0])
    n_queries = int(ci["n_queries"].iloc[0])
    unit = str(ci["bootstrap_unit"].iloc[0])
    seeds = ", ".join(str(int(seed)) for seed in ci["seed"].tolist())
    rows = [
        ("Bootstrap target", "Matched reciprocal-rank differences for each comparison"),
        ("Bootstrap unit", f"{unit}; three seed-specific records are averaged per original query before resampling"),
        ("Number of units", f"{n_queries:,} original bidirectional test queries"),
        ("Resamples", f"{n_bootstrap:,} paired bootstrap resamples per comparison"),
        ("Interval", "2.5th and 97.5th percentiles of bootstrapped mean differences"),
        ("Random seeds", seeds),
        ("Implementation", "scripts/run_score_aware_bootstrap_ci.py"),
        ("Reported file", "docs/paper_tables/table_score_aware_bootstrap_ci.tex"),
    ]
    pd.DataFrame(rows, columns=["item", "detail"]).to_csv(
        OUT_DIR / "table_s6_bootstrap_implementation_details.csv", index=False
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Bootstrap implementation details for score-aware comparisons.}",
        r"\label{tab:supp_bootstrap_implementation}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{0.25\textwidth}p{0.62\textwidth}}",
        r"\toprule",
        r"Item & Detail \\",
        r"\midrule",
    ]
    for item, detail in rows:
        lines.append(f"{tex_escape(item)} & {tex_escape(detail)} " + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4ex}",
            (
                r"\caption*{\footnotesize \textit{Note:} The paired design preserves matched query-level "
                r"differences between two methods before resampling, so the interval measures uncertainty in "
                r"the mean MRR difference rather than uncertainty in two independent MRR estimates.}"
            ),
            r"\end{table}",
        ]
    )
    write(OUT_DIR / "table_s6_bootstrap_implementation_details.tex", lines)


def config_value(path: str, *keys: str, default: object = None) -> object:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def build_s7() -> None:
    fixed_configs = {
        "ComplEx": "ml/artifacts/outputs/openbg_img_complex/20260328_004926_seed1/config_merged.json",
        "TuckER": "ml/artifacts/outputs/openbg_img_tucker/20260328_011550_seed1/config_merged.json",
        "Text-only": "ml/artifacts/outputs/openbg_img_text_only/20260326_184048_seed1/config_merged.json",
        "Early fusion": "ml/artifacts/outputs/openbg_img_early/20260326_235410_seed1/config_merged.json",
        "Gate-only": "ml/artifacts/outputs/openbg_img_gate_only/20260329_202546_seed1/config_merged.json",
        "Residual-only": "ml/artifacts/outputs/openbg_img_residual_only/20260329_202016_seed1/config_merged.json",
        "Full model": "ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260329_205816_seed1/config_merged.json",
    }
    fixed_lrs = []
    fixed_epochs = []
    fixed_dropout = []
    for name, path in fixed_configs.items():
        if not Path(path).exists():
            continue
        fixed_lrs.append(f"{name}: {config_value(path, 'training', 'lr')}")
        fixed_epochs.append(int(config_value(path, "training", "epochs", default=0)))
        dropout = config_value(path, "training", "img_dropout", default=None)
        if dropout is not None:
            fixed_dropout.append(f"{name}: {dropout}")
    fixed_epoch_summary = f"{min(fixed_epochs)}-{max(fixed_epochs)} epochs" if fixed_epochs else "config-specific"

    xgb_summary = json.loads(Path("outputs/router/models/clean/xgb_delta_0.01_C4/train_summary.json").read_text())
    ordinal_summary = json.loads(Path("outputs/router/models/clean_ordinal/xgb_bins_C4/train_summary.json").read_text())
    regression_summary = json.loads(
        Path("outputs/router/models/clean_regression/xgb_delta_rr_C4/train_summary.json").read_text()
    )
    ca_s2 = json.loads(Path("outputs/candidate_router/models/ca_s2_score_soft_pairwise/config.json").read_text())

    rows = [
        {
            "component": "Fixed KGC experts",
            "data": "OpenBG-IMG paper_split train/dev/test; seeds 1/2/3",
            "model_or_features": "ComplEx, TuckER, Text-only, Early fusion, Gate-only, Residual-only, Full model",
            "key_hyperparameters": (
                f"d=256; batch=4096; neg=10; adv_temperature=2.0; {fixed_epoch_summary}; "
                f"lr by model ({'; '.join(fixed_lrs)})"
            ),
            "selection_or_eval": "best checkpoint on dev; filtered bidirectional test ranking",
        },
        {
            "component": "Clean binary routers",
            "data": f"{xgb_summary['train_table']}; n={xgb_summary['n_train']:,}",
            "model_or_features": "C4 legal features; logistic and XGB",
            "key_hyperparameters": (
                "delta in {0.00, 0.01, 0.02}; tau grid {0.1,0.3,0.5,0.7,0.9}; "
                "XGB: 300 trees, depth 4, lr 0.05, subsample/colsample 0.8, random_state 42"
            ),
            "selection_or_eval": "locked clean test rows; query-time legal features only",
        },
        {
            "component": "Ordinal/scalar clean routers",
            "data": f"{ordinal_summary['train_table']}; n={ordinal_summary['n_train']:,}",
            "model_or_features": "C4 legal features; ordinal XGB and delta-RR XGB regression",
            "key_hyperparameters": (
                f"ordinal bins {ordinal_summary['bin_thresholds']}; regression target "
                f"{regression_summary['target_field']}; random_state 42"
            ),
            "selection_or_eval": "test policies selected from locked clean evaluation outputs",
        },
        {
            "component": "Candidate-level CA-S2 router",
            "data": f"{ca_s2['train_table']}; train pairs={ca_s2['train_pairs']:,}",
            "model_or_features": "score/rank CA-S2 features; pairwise loss",
            "key_hyperparameters": (
                f"top-100 candidates; negatives/query={ca_s2['negatives_per_query']}; hard ratio={ca_s2['hard_ratio']}; "
                f"epochs={ca_s2['epochs']}; batch={ca_s2['batch_size']}; lr={ca_s2['lr']}; wd={ca_s2['weight_decay']}; seed={ca_s2['seed']}"
            ),
            "selection_or_eval": "full-ranking CA-S2 evaluation on test seeds",
        },
        {
            "component": "Score interpolation",
            "data": "dev split for alpha selection; test split for final reporting",
            "model_or_features": "fixed Gate-only and Residual-only score spaces",
            "key_hyperparameters": "alpha grid 0.00:0.05:1.00; relation min support=20; fallback=global alpha",
            "selection_or_eval": "alpha selected by dev MRR; global/direction/relation policies reported on test",
        },
        {
            "component": "Bootstrap CIs",
            "data": "score_aware_per_query_rr_seed_averaged.csv",
            "model_or_features": "matched reciprocal-rank differences",
            "key_hyperparameters": "10,000 paired resamples; query-level seed-averaged unit; seeds 42-45",
            "selection_or_eval": "95% percentile confidence intervals",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "table_s7_hyperparameter_training_config.csv", index=False)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Hyperparameter and training configuration summary.}",
        r"\label{tab:supp_hyperparameter_training_config}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{p{0.13\textwidth}p{0.20\textwidth}p{0.20\textwidth}p{0.30\textwidth}p{0.12\textwidth}}",
        r"\toprule",
        r"Component & Data & Model/features & Key hyperparameters & Selection/evaluation \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['component'])} & {tex_escape(row['data'])} & "
            f"{tex_escape(row['model_or_features'])} & {tex_escape(row['key_hyperparameters'])} & "
            f"{tex_escape(row['selection_or_eval'])} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4ex}",
            (
                r"\caption*{\footnotesize \textit{Note:} Fixed-expert hyperparameters are read from the locked "
                r"seed-1 merged configs for the reported runs; the seed value itself is varied over 1/2/3. "
                r"Router and candidate-router settings are read from their saved training summaries/configs.}"
            ),
            r"\end{table*}",
        ]
    )
    write(OUT_DIR / "table_s7_hyperparameter_training_config.tex", lines)


def update_readme() -> None:
    readme = OUT_DIR / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# Supplementary Table Bundle\n"
    lines = [
        "# Supplementary Table Bundle",
        "",
        "| Table | TeX file | Source data | Status |",
        "|---|---|---|---|",
        "| S1 | `table_s1_relation_group_sanity_check.tex` | `docs/relation_type_summary_min20.json` | generated |",
        "| S2 | `table_s2_degree_bucket_sanity_check.tex` | `docs/paper_tables/table_degree_bucket_sanity_check.csv` | generated |",
        "| S3 | `table_s3_ordinal_gain_modeling.tex` | clean router ordinal/regression CSVs | generated |",
        "| S4 | `table_s4_delta_sensitivity.tex` | `outputs/router/eval/clean/delta_sensitivity_clean.csv` | generated |",
        "| S5 | `table_s5_per_seed_model_comparison.tex` | `outputs/router/test/*_query_eval_seed*.csv` | generated |",
        "| S6 | `table_s6_bootstrap_implementation_details.tex` | `outputs/score_ensemble/eval/score_aware_bootstrap_ci.csv` | generated |",
        "| S7 | `table_s7_hyperparameter_training_config.tex` | merged configs and router/candidate-router summaries | generated |",
        "",
        "Optional alpha-stability table retained as `table_s5_alpha_sweep_stability.tex`; renumber it only if you decide to include it after S7.",
        "",
        "S4 should be described as margin sensitivity rather than as full robustness across many deltas; the existing data covers delta values 0.00, 0.01, and 0.02.",
    ]
    if "table_s5_per_seed_model_comparison.tex" not in existing:
        write(readme, lines)
    else:
        write(readme, lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_s5()
    build_s6()
    build_s7()
    update_readme()
    print(f"[OK] wrote S5-S7 supplementary tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
