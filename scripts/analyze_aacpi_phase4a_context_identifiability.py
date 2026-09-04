from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aacpi_phase4a_common import REPRESENTATIONS, feature_contract, portable_path, reject_test_path, sha256_file
from scripts.run_aacpi_phase3a_representation_oof import sign_metrics
from scripts.train_aacpi_advantage_nested_cv import evaluate_predictions


PAIRS = {
    "mkgw_mhyper_native", "mkgw_mhyper_adamf", "mkgw_native_adamf",
    "db15k_mhyper_native", "db15k_mhyper_adamf", "db15k_native_adamf",
}
NATIVE_PAIRS = {"mkgw_native_adamf", "db15k_native_adamf"}
PRIOR_PASS = {"mkgw_mhyper_native", "mkgw_mhyper_adamf", "db15k_native_adamf"}
METRIC_FIELDS = (
    "spearman", "positive_auprc_lift", "harmful_auprc_lift",
    "positive_vs_harmful_auroc", "positive_vs_harmful_auroc_lift",
    "highest_10pct_actual_mean_advantage", "highest_10pct_positive_rate",
    "highest_10pct_harmful_rate", "nonzero_activity_auprc_lift",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen DEV-only Phase 4A contextual identifiability.")
    parser.add_argument("--phase4a-root", default="outputs/aacpi/phase4a")
    parser.add_argument("--phase3a-root", default="outputs/aacpi/phase3a")
    parser.add_argument("--report", default="docs/reports/aacpi_phase4a_contextual_identifiability_audit_2026-09-05.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def top_decile(actual, predicted):
    order = np.argsort(predicted, kind="mergesort")
    chosen = order[int(round(0.9 * len(order))):]
    return {
        "highest_10pct_actual_mean_advantage": float(actual[chosen].mean()),
        "highest_10pct_positive_rate": float((actual[chosen] > 1e-12).mean()),
        "highest_10pct_harmful_rate": float((actual[chosen] < -1e-12).mean()),
    }


def metrics(actual, predicted):
    result = evaluate_predictions(actual, predicted, beta=0.02)
    result.update(sign_metrics(actual, predicted)); result.update(top_decile(actual, predicted))
    return result


def pass_h1(row):
    return all(float(row[key]) > 0 for key in ("spearman", "positive_auprc_lift", "harmful_auprc_lift", "highest_10pct_actual_mean_advantage"))


def load_predictions(root: Path, phase3a_root: Path):
    import pandas as pd

    paths = sorted((root / "oof_raw").glob("*/c?/dev_oof_predictions.csv.gz"))
    observed = {(path.parent.parent.name, path.parent.name.upper()) for path in paths}
    expected = {(pair, rep) for pair in PAIRS for rep in REPRESENTATIONS}
    if observed != expected:
        raise RuntimeError(f"Incomplete Phase 4A 6x5 grid; missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    frames, sources, c0_max_delta = [], [], 0.0
    for path in paths:
        reject_test_path(path)
        frame = pd.read_csv(path, compression="gzip")
        pair, rep = path.parent.parent.name, path.parent.name.upper()
        if frame.empty or set(frame.split.astype(str)) != {"dev"} or set(frame.representation.astype(str).str.upper()) != {rep}:
            raise RuntimeError(f"Invalid DEV prediction: {path}")
        group_folds = frame.groupby("original_triple_id").outer_fold.nunique()
        if int(group_folds.max()) != 1 or not np.isfinite(frame.predicted_advantage_oof).all():
            raise AssertionError(f"OOF leakage/nonfinite prediction: {path}")
        if rep == "C0":
            r3_path = phase3a_root / pair / "r3" / "dev_oof_predictions.csv.gz"
            r3 = pd.read_csv(r3_path, compression="gzip", usecols=["query_id", "alpha", "predicted_advantage_oof"])
            if not frame[["query_id", "alpha"]].reset_index(drop=True).equals(r3[["query_id", "alpha"]].reset_index(drop=True)):
                raise AssertionError(f"C0/R3 row mismatch for {pair}")
            delta = np.max(np.abs(frame.predicted_advantage_oof.to_numpy(float) - r3.predicted_advantage_oof.to_numpy(float)))
            c0_max_delta = max(c0_max_delta, float(delta))
        frames.append(frame); sources.append(path)
    if c0_max_delta != 0.0:
        raise AssertionError(f"C0 does not exactly reproduce Phase 3A R3: max delta={c0_max_delta}")
    return pd.concat(frames, ignore_index=True), sources, c0_max_delta


def summarize(combined):
    pair_rows, action_rows, direction_rows, seed_rows, relation_rows = [], [], [], [], []
    nonref = combined[~np.isclose(combined.alpha, combined.alpha0, atol=1e-12, rtol=0.0)].copy()
    specs = [
        (pair_rows, ["dataset", "pair_id", "representation"]),
        (action_rows, ["dataset", "pair_id", "representation", "delta_alpha"]),
        (direction_rows, ["dataset", "pair_id", "representation", "direction"]),
        (seed_rows, ["dataset", "pair_id", "representation", "seed"]),
        (relation_rows, ["dataset", "pair_id", "representation", "relation"]),
    ]
    for destination, keys in specs:
        for values, frame in nonref.groupby(keys, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            if len(frame) < 20:
                continue
            result = metrics(frame.advantage.to_numpy(float), frame.predicted_advantage_oof.to_numpy(float))
            row = {key: value for key, value in zip(keys, values)}
            destination.append({**row, **result, "h1_style_pass": pass_h1(result), "n_rows": len(frame), "n_queries": int(frame.query_id.nunique())})
    return pair_rows, action_rows, direction_rows, seed_rows, relation_rows


def increments_and_gates(pair_rows, direction_rows, seed_rows):
    by_key = {(row["pair_id"], row["representation"]): row for row in pair_rows}
    increments, gates = [], []
    for pair in sorted(PAIRS):
        base = by_key[(pair, "C0")]
        for rep in REPRESENTATIONS[1:]:
            current = by_key[(pair, rep)]
            increments.append({
                "dataset": current["dataset"], "pair_id": pair, "representation": rep,
                **{f"delta_{field}": float(current[field]) - float(base[field]) for field in METRIC_FIELDS},
            })
    for rep in REPRESENTATIONS[1:]:
        pass_pairs = {pair for pair in PAIRS if pass_h1(by_key[(pair, rep)])}
        critical = by_key[("mkgw_native_adamf", rep)]; control = by_key[("mkgw_native_adamf", "C0")]
        direction_pass = sum(1 for row in direction_rows if row["pair_id"] == "mkgw_native_adamf" and row["representation"] == rep and row["h1_style_pass"])
        seed_pass = sum(1 for row in seed_rows if row["pair_id"] == "mkgw_native_adamf" and row["representation"] == rep and row["h1_style_pass"])
        stable_count = len(PRIOR_PASS & pass_pairs)
        no_double_collapse = all(not (float(by_key[(pair, rep)]["highest_10pct_actual_mean_advantage"]) <= 0 and float(by_key[(pair, rep)]["positive_vs_harmful_auroc_lift"]) <= 0) for pair in PRIOR_PASS)
        stability = stable_count >= 2 and no_double_collapse
        primary = (len(pass_pairs) >= 4 and NATIVE_PAIRS <= pass_pairs and float(critical["positive_auprc_lift"]) > 0 and float(critical["harmful_auprc_lift"]) > 0 and float(critical["highest_10pct_actual_mean_advantage"]) > 0 and float(critical["positive_vs_harmful_auroc"]) > float(control["positive_vs_harmful_auroc"]))
        strong = (float(critical["positive_vs_harmful_auroc"]) >= 0.55 and float(critical["highest_10pct_actual_mean_advantage"]) > 0 and direction_pass >= 1 and seed_pass >= 2)
        contribution = (float(critical["positive_vs_harmful_auroc"]) > float(control["positive_vs_harmful_auroc"]) and float(critical["highest_10pct_actual_mean_advantage"]) > float(control["highest_10pct_actual_mean_advantage"]))
        gates.append({
            "representation": rep, "h1_pass_count": len(pass_pairs), "h1_pass_pairs": ";".join(sorted(pass_pairs)),
            "both_native_adamf_pass": NATIVE_PAIRS <= pass_pairs, "critical_direction_pass_count": direction_pass,
            "critical_seed_pass_count": seed_pass, "prior_pass_stable_count": stable_count,
            "prior_pass_no_double_collapse": no_double_collapse, "primary_gate": primary,
            "strong_recovery_gate": strong, "context_contribution_gate": contribution,
            "prior_pass_stability_gate": stability, "phase4a_go": primary and strong and contribution and stability,
        })
    return increments, gates


def svg_bars(path: Path, rows, field: str, title: str):
    width, height, margin = 1260, 560, 80
    ordered = sorted(rows, key=lambda row: (row["pair_id"], row["representation"])); values = [float(row[field]) for row in ordered]
    limit, zero = max(max(abs(v) for v in values), 1e-6), height / 2; usable = height / 2 - margin
    bar_width = (width - 2 * margin) / len(ordered)
    colors = {"C0":"#777777","C1":"#377eb8","C2":"#ff7f00","C3":"#4daf4a","C4":"#984ea3"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',f'<line x1="{margin}" y1="{zero}" x2="{width-margin}" y2="{zero}" stroke="#222"/>']
    for index, (row, value) in enumerate(zip(ordered, values)):
        x = margin + index * bar_width + 1; size = abs(value) / limit * usable; y = zero - size if value >= 0 else zero
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width-2,1):.2f}" height="{size:.2f}" fill="{colors[row["representation"]]}"/>')
    parts.append("</svg>"); path.write_text("\n".join(parts), encoding="utf-8")


def report_text(pair_rows, increments, gates, action_rows, direction_rows, seed_rows, relation_rows):
    by_key = {(row["pair_id"], row["representation"]): row for row in pair_rows}
    go = [row["representation"] for row in gates if row["phase4a_go"]]; outcome = "GO" if go else "NO-GO"
    lines = ["# AACPI Phase 4A Contextual Identifiability Audit", "", f"**Frozen decision:** {outcome}", "", "This audit uses DEV-only outer-held-out predictions and runs no policy or TEST evaluation.", "", "## Pair results", "", "| Dataset | Pair | Rep | H1 | Spearman | Positive AP lift | Harmful AP lift | Sign AUROC | Top-10% U | Top-10% P+ | Top-10% P- |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(pair_rows, key=lambda x:(x["dataset"],x["pair_id"],x["representation"])):
        lines.append(f"| {row['dataset']} | {row['pair_id']} | {row['representation']} | {'PASS' if row['h1_style_pass'] else 'FAIL'} | {row['spearman']:.6f} | {row['positive_auprc_lift']:+.6f} | {row['harmful_auprc_lift']:+.6f} | {row['positive_vs_harmful_auroc']:.6f} | {row['highest_10pct_actual_mean_advantage']:+.6f} | {row['highest_10pct_positive_rate']:.3%} | {row['highest_10pct_harmful_rate']:.3%} |")
    lines += ["", "## Frozen gates", "", "| Rep | H1 pairs | Native pairs | Critical dirs | Critical seeds | Primary | Strong | Contribution | Stability | Decision |", "| --- | ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |"]
    for row in gates:
        lines.append(f"| {row['representation']} | {row['h1_pass_count']}/6 | {row['both_native_adamf_pass']} | {row['critical_direction_pass_count']}/2 | {row['critical_seed_pass_count']}/3 | {row['primary_gate']} | {row['strong_recovery_gate']} | {row['context_contribution_gate']} | {row['prior_pass_stability_gate']} | {'GO' if row['phase4a_go'] else 'NO-GO'} |")
    lines += ["", "## MKG-W / NativE + AdaMF-MAT", ""]
    for rep in REPRESENTATIONS:
        row = by_key[("mkgw_native_adamf", rep)]
        actions = sum(1 for x in action_rows if x["pair_id"]=="mkgw_native_adamf" and x["representation"]==rep and x["h1_style_pass"])
        directions = sum(1 for x in direction_rows if x["pair_id"]=="mkgw_native_adamf" and x["representation"]==rep and x["h1_style_pass"])
        seeds = sum(1 for x in seed_rows if x["pair_id"]=="mkgw_native_adamf" and x["representation"]==rep and x["h1_style_pass"])
        rel = [x for x in relation_rows if x["pair_id"]=="mkgw_native_adamf" and x["representation"]==rep and int(x["n_queries"])>=60]
        coverage = sum(int(x["n_queries"]) for x in rel if x["h1_style_pass"])/max(sum(int(x["n_queries"]) for x in rel),1)
        lines.append(f"- {rep}: Spearman {row['spearman']:.6f}; AP lifts {row['positive_auprc_lift']:+.6f}/{row['harmful_auprc_lift']:+.6f}; sign AUROC {row['positive_vs_harmful_auroc']:.6f}; top-10% U {row['highest_10pct_actual_mean_advantage']:+.6f}; P+/P- {row['highest_10pct_positive_rate']:.3%}/{row['highest_10pct_harmful_rate']:.3%}; passing actions {actions}/5; directions {directions}/2; seeds {seeds}/3; supported-relation coverage {coverage:.2%}.")
    mean_delta = lambda rep, field: float(np.mean([x[f"delta_{field}"] for x in increments if x["representation"]==rep]))
    best_sign = max(REPRESENTATIONS[1:], key=lambda rep: mean_delta(rep,"positive_vs_harmful_auroc"))
    best_top = max(REPRESENTATIONS[1:], key=lambda rep: mean_delta(rep,"highest_10pct_actual_mean_advantage"))
    c3, c4 = by_key[("mkgw_native_adamf","C3")], by_key[("mkgw_native_adamf","C4")]
    lines += ["", "## Core audit answers", "", f"1. Structural context: C1 mean sign-AUROC increment is {mean_delta('C1','positive_vs_harmful_auroc'):+.6f}; interpret together with its gate row above.", f"2. Modality context: C2 mean sign-AUROC increment is {mean_delta('C2','positive_vs_harmful_auroc'):+.6f}; activity lift is reported separately in `context_increments.csv`.", f"3. Frozen latent context: C3 mean sign-AUROC increment is {mean_delta('C3','positive_vs_harmful_auroc'):+.6f}; the strongest mean sign increment is {best_sign}.", f"4. C3 critical-pair top-confidence utility is {c3['highest_10pct_actual_mean_advantage']:+.6f} ({'positive' if c3['highest_10pct_actual_mean_advantage']>0 else 'nonpositive'}).", f"5. C4 critical-pair top-confidence utility is {c4['highest_10pct_actual_mean_advantage']:+.6f}; the strongest mean top-decile increment is {best_top}.", "6. Beneficial-vs-harmful and nonzero-activity increments are separate columns; the frozen gate uses sign separation and actual top-decile utility.", f"7. Prior-PASS stability: " + ", ".join(f"{x['representation']}={x['prior_pass_stability_gate']}" for x in gates) + ".", f"8. Phase 4A GO representations: {', '.join(go) if go else 'none'}.", f"9. Evidence for Context-Conditioned Conservative Policy: {'eligible for a separately preregistered Phase 4B, without TEST access' if go else 'insufficient; policy remains prohibited'}.", f"10. Frozen conclusion: {'contextual identifiability recovered on DEV' if go else 'the frozen contextual families did not clear the information-sufficiency gates; stop feature expansion and treat an answer-agnostic identifiability ceiling as the active explanation'}.", "", "## Integrity audit", "", "- TEST rows accessed: **0**; TEST commands: **0**; policy evaluations: **0**.", "- Expert retraining: **0**; checkpoint reselection: **0**.", "- AACPI V2, Phase 3A results, alpha0, action grid, utility target, and historical feature contracts modified: **no**.", "- Structural/modality statistics are TRAIN-only; latent states are frozen and target-independent.", "- Fold-fitted PCA and scaling use current training groups only; outer original-triple leakage is zero.", "- C0 exactly reproduces Phase 3A R3; all source and generated artifacts are hash-auditable."]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args(); root, p3root, report = Path(args.phase4a_root), Path(args.phase3a_root), Path(args.report)
    outputs = [root/"pair_summaries.csv",root/"action_summaries.csv",root/"direction_summaries.csv",root/"seed_summaries.csv",root/"relation_summaries.csv",root/"context_increments.csv",root/"phase4a_gate_summary.csv",root/"context_feature_manifest.json",root/"phase4a_analysis_manifest.json",report]
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("Refusing to overwrite Phase 4A analysis outputs")
    combined, sources, c0_delta = load_predictions(root, p3root)
    pair_rows, action_rows, direction_rows, seed_rows, relation_rows = summarize(combined)
    increments, gates = increments_and_gates(pair_rows, direction_rows, seed_rows)
    for path, rows in zip(outputs[:7], (pair_rows,action_rows,direction_rows,seed_rows,relation_rows,increments,gates)):
        write_csv(path, rows)
    outputs[7].write_text(json.dumps(feature_contract(), indent=2)+"\n", encoding="utf-8")
    figures=root/"figures"; figures.mkdir(parents=True, exist_ok=True)
    svg_bars(figures/"context_sign_auroc_lift.svg", pair_rows, "positive_vs_harmful_auroc_lift", "Phase 4A sign-AUROC lift")
    svg_bars(figures/"context_top_decile_utility.svg", pair_rows, "highest_10pct_actual_mean_advantage", "Phase 4A top-decile actual advantage")
    svg_bars(figures/"mkgw_native_adamf_context_recovery.svg", [x for x in pair_rows if x["pair_id"]=="mkgw_native_adamf"], "highest_10pct_actual_mean_advantage", "MKG-W NativE + AdaMF-MAT contextual recovery")
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(report_text(pair_rows,increments,gates,action_rows,direction_rows,seed_rows,relation_rows), encoding="utf-8")
    manifest = {"schema_version":1,"phase":"AACPI Phase 4A","split":"dev","source_oof_predictions":[{"path":portable_path(p),"sha256":sha256_file(p)} for p in sources],"outputs":[],"c0_phase3a_r3_max_abs_prediction_delta":c0_delta,"outer_original_triple_leakage":0,"test_rows_accessed":0,"test_evaluation_commands":0,"policy_evaluations":0,"phase4a_go_representations":[x["representation"] for x in gates if x["phase4a_go"]]}
    manifest["outputs"]=[{"path":portable_path(p),"sha256":sha256_file(p)} for p in [*outputs[:8],report,*sorted(figures.glob("*.svg"))]]
    outputs[8].write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"[OK] Phase 4A frozen decision={'GO' if manifest['phase4a_go_representations'] else 'NO-GO'} -> {report}")


if __name__ == "__main__":
    main()
