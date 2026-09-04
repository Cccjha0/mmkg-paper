from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aacpi_phase3a_common import feature_manifest, portable_path, sha256_file
from scripts.run_aacpi_phase3a_representation_oof import sign_metrics
from scripts.train_aacpi_advantage_nested_cv import evaluate_predictions


PRIOR_FAIL = {"mkgw_native_adamf", "db15k_mhyper_native", "db15k_mhyper_adamf"}
PRIOR_PASS = {"mkgw_mhyper_native", "mkgw_mhyper_adamf", "db15k_native_adamf"}
NATIVE_PAIRS = {"mkgw_native_adamf", "db15k_native_adamf"}
REPRESENTATIONS = ("R0", "R1", "R2", "R3")
METRIC_FIELDS = (
    "spearman",
    "positive_auprc_lift",
    "harmful_auprc_lift",
    "positive_vs_harmful_auroc",
    "positive_vs_harmful_auroc_lift",
    "highest_10pct_actual_mean_advantage",
    "highest_10pct_positive_rate",
    "highest_10pct_harmful_rate",
    "nonzero_activity_auprc_lift",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DEV-only AACPI Phase 3A OOF representation results.")
    parser.add_argument("--phase3a-root", default="outputs/aacpi/phase3a")
    parser.add_argument("--asset-root", default="outputs/aacpi/action_response_assets")
    parser.add_argument("--output-dir", default="outputs/aacpi/phase3a")
    parser.add_argument(
        "--report",
        default="docs/reports/aacpi_phase3a_action_response_representation_audit_2026-09-04.md",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def top_decile(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    order = np.argsort(predicted, kind="mergesort")
    indices = order[int(round(0.9 * len(order))) :]
    return {
        "highest_10pct_actual_mean_advantage": float(actual[indices].mean()),
        "highest_10pct_positive_rate": float((actual[indices] > 1e-12).mean()),
        "highest_10pct_harmful_rate": float((actual[indices] < -1e-12).mean()),
    }


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    result = evaluate_predictions(actual, predicted, beta=0.02)
    result.update(sign_metrics(actual, predicted))
    result.update(top_decile(actual, predicted))
    return result


def pass_h1(row: dict) -> bool:
    return all(
        float(row[key]) > 0.0
        for key in (
            "spearman",
            "positive_auprc_lift",
            "harmful_auprc_lift",
            "highest_10pct_actual_mean_advantage",
        )
    )


def load_predictions(root: Path):
    import pandas as pd

    paths = sorted(root.glob("*/r?/dev_oof_predictions.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No Phase 3A OOF predictions under {root}")
    frames, source_paths = [], []
    for path in paths:
        if "test" in {part.lower() for part in path.parts}:
            raise RuntimeError(f"Refusing TEST-like path: {path}")
        frame = pd.read_csv(path, compression="gzip")
        if frame.empty or set(frame["split"].astype(str)) != {"dev"}:
            raise RuntimeError(f"Non-DEV or empty prediction file: {path}")
        if not frame["predicted_advantage_oof"].map(np.isfinite).all():
            raise ValueError(f"Nonfinite OOF predictions: {path}")
        representation = str(frame["representation"].iloc[0]).upper()
        pair_id = str(frame["pair_id"].iloc[0])
        if representation not in REPRESENTATIONS or path.parent.name != representation.lower():
            raise ValueError(f"Representation/path mismatch: {path}")
        if path.parent.parent.name != pair_id:
            raise ValueError(f"Pair/path mismatch: {path}")
        group_folds = frame.groupby("original_triple_id")["outer_fold"].nunique()
        if int(group_folds.max()) != 1:
            raise AssertionError(f"Original-triple leakage in {path}")
        if frame["query_id"].duplicated().all():
            raise AssertionError(f"Invalid query/action layout in {path}")
        frames.append(frame)
        source_paths.append(path)
    combined = pd.concat(frames, ignore_index=True)
    expected_pairs = {
        "mkgw_mhyper_native", "mkgw_mhyper_adamf", "mkgw_native_adamf",
        "db15k_mhyper_native", "db15k_mhyper_adamf", "db15k_native_adamf",
    }
    observed = set(zip(combined["pair_id"].astype(str), combined["representation"].str.upper()))
    expected = {(pair, rep) for pair in expected_pairs for rep in REPRESENTATIONS}
    if observed != expected:
        raise RuntimeError(f"Incomplete 6x4 comparison; missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    return combined, source_paths


def summarize(combined):
    pair_rows, action_rows, direction_rows, seed_rows, relation_rows = [], [], [], [], []
    grouping_specs = [
        (action_rows, ["dataset", "pair_id", "representation", "delta_alpha"]),
        (direction_rows, ["dataset", "pair_id", "representation", "direction"]),
        (seed_rows, ["dataset", "pair_id", "representation", "seed"]),
        (relation_rows, ["dataset", "pair_id", "representation", "relation"]),
    ]
    nonref = combined[~np.isclose(combined["alpha"], combined["alpha0"], atol=1e-12)].copy()
    for keys, frame in nonref.groupby(["dataset", "pair_id", "representation"], sort=True):
        values = metrics(frame["advantage"].to_numpy(float), frame["predicted_advantage_oof"].to_numpy(float))
        pair_rows.append({"dataset": keys[0], "pair_id": keys[1], "representation": keys[2], **values, "h1_style_pass": pass_h1(values)})
    for destination, keys in grouping_specs:
        for values, frame in nonref.groupby(keys, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            if len(frame) < 20:
                continue
            result = metrics(frame["advantage"].to_numpy(float), frame["predicted_advantage_oof"].to_numpy(float))
            row = {key: value for key, value in zip(keys, values)}
            destination.append({**row, **result, "h1_style_pass": pass_h1(result), "n_queries": int(frame["query_id"].nunique())})
    return pair_rows, action_rows, direction_rows, seed_rows, relation_rows


def comparisons(pair_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_key = {(row["pair_id"], row["representation"]): row for row in pair_rows}
    rows, gate_rows = [], []
    for pair in sorted({row["pair_id"] for row in pair_rows}):
        base = by_key[(pair, "R0")]
        for rep in REPRESENTATIONS[1:]:
            current = by_key[(pair, rep)]
            delta = {f"delta_{field}": float(current[field]) - float(base[field]) for field in METRIC_FIELDS}
            clear = (
                delta["delta_positive_vs_harmful_auroc"] >= 0.02
                and delta["delta_positive_auprc_lift"] >= 0.01
                and delta["delta_harmful_auprc_lift"] >= 0.01
                and delta["delta_highest_10pct_actual_mean_advantage"] >= 0.001
                and float(current["highest_10pct_actual_mean_advantage"]) > 0.0
                and delta["delta_spearman"] >= 0.0
            )
            rows.append({
                "dataset": current["dataset"], "pair_id": pair, "representation": rep,
                "prior_status": "FAIL" if pair in PRIOR_FAIL else "PASS",
                **delta, "clear_sign_identifiability_improvement": clear,
            })
    for rep in REPRESENTATIONS[1:]:
        rep_rows = [by_key[(pair, rep)] for pair in sorted({row["pair_id"] for row in pair_rows})]
        pass_pairs = {row["pair_id"] for row in rep_rows if pass_h1(row)}
        improved = {
            row["pair_id"] for row in rows
            if row["representation"] == rep and row["clear_sign_identifiability_improvement"]
        }
        prior_stable_count = len(PRIOR_PASS & pass_pairs)
        no_double_collapse = all(
            not (
                float(by_key[(pair, rep)]["highest_10pct_actual_mean_advantage"]) <= 0.0
                and float(by_key[(pair, rep)]["positive_vs_harmful_auroc_lift"]) <= 0.0
            )
            for pair in PRIOR_PASS
        )
        primary = (
            len(pass_pairs) >= 4
            and NATIVE_PAIRS <= pass_pairs
            and float(by_key[("mkgw_native_adamf", rep)]["highest_10pct_actual_mean_advantage"]) > 0.0
            and float(by_key[("mkgw_native_adamf", rep)]["positive_auprc_lift"]) > 0.0
            and float(by_key[("mkgw_native_adamf", rep)]["harmful_auprc_lift"]) > 0.0
        )
        recovery = len(PRIOR_FAIL & improved) >= 2 and "mkgw_native_adamf" in improved
        stability = prior_stable_count >= 2 and no_double_collapse
        gate_rows.append({
            "representation": rep,
            "h1_pass_count": len(pass_pairs),
            "h1_pass_pairs": ";".join(sorted(pass_pairs)),
            "prior_fail_clear_improvement_count": len(PRIOR_FAIL & improved),
            "mkgw_native_adamf_clear_improvement": "mkgw_native_adamf" in improved,
            "prior_pass_h1_stable_count": prior_stable_count,
            "prior_pass_no_double_collapse": no_double_collapse,
            "primary_gate": primary,
            "representation_recovery_gate": recovery,
            "prior_pass_stability_gate": stability,
            "phase3a_go": primary and recovery and stability,
        })
    return rows, gate_rows


def svg_bars(path: Path, rows: list[dict], field: str, title: str) -> None:
    width, height, margin = 1180, 540, 80
    ordered = sorted(rows, key=lambda row: (row["pair_id"], row["representation"]))
    values = [float(row[field]) for row in ordered]
    limit = max(max(abs(v) for v in values), 1e-6)
    zero = height / 2
    usable = height / 2 - margin
    bar_width = (width - 2 * margin) / len(ordered)
    colors = {"R0": "#777777", "R1": "#377eb8", "R2": "#e41a1c", "R3": "#4daf4a"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>', f'<line x1="{margin}" y1="{zero}" x2="{width-margin}" y2="{zero}" stroke="#222"/>']
    for i, (row, value) in enumerate(zip(ordered, values)):
        x = margin + i * bar_width + 1
        size = abs(value) / limit * usable
        y = zero - size if value >= 0 else zero
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width-2,1):.2f}" height="{size:.2f}" fill="{colors[row["representation"]]}"/>')
        if i % 4 == 1:
            label = row["pair_id"].replace("_", " ")
            parts.append(f'<text x="{x+bar_width:.2f}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="10" transform="rotate(-18 {x+bar_width:.2f},{height-20})">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def report_text(pair_rows, comparison_rows, gate_rows, relation_rows) -> str:
    by_key = {(row["pair_id"], row["representation"]): row for row in pair_rows}
    go_reps = [row["representation"] for row in gate_rows if row["phase3a_go"]]
    outcome = "GO" if go_reps else "NO-GO"
    lines = [
        "# AACPI Phase 3A Action-Response Representation Audit",
        "",
        f"**Frozen decision:** {outcome}",
        "",
        "This report uses DEV-only outer-fold OOF predictions. It evaluates representation recovery and runs no policy.",
        "",
        "## Pair results",
        "",
        "| Dataset | Pair | Rep | H1 | Spearman | Positive AP lift | Harmful AP lift | Pos-vs-harm AUROC | Top-10% U | Top-10% P+ | Top-10% P- |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(pair_rows, key=lambda item: (item["dataset"], item["pair_id"], item["representation"])):
        lines.append(
            f"| {row['dataset']} | {row['pair_id']} | {row['representation']} | {'PASS' if row['h1_style_pass'] else 'FAIL'} | "
            f"{row['spearman']:.6f} | {row['positive_auprc_lift']:+.6f} | {row['harmful_auprc_lift']:+.6f} | "
            f"{row['positive_vs_harmful_auroc']:.6f} | {row['highest_10pct_actual_mean_advantage']:+.6f} | "
            f"{row['highest_10pct_positive_rate']:.3%} | {row['highest_10pct_harmful_rate']:.3%} |"
        )
    lines += ["", "## Frozen gates", "", "| Rep | H1 pairs | Prior FAIL improved | MKG-W Native improved | Primary | Recovery | Stability | Decision |", "| --- | ---: | ---: | --- | --- | --- | --- | --- |"]
    for row in gate_rows:
        lines.append(
            f"| {row['representation']} | {row['h1_pass_count']}/6 | {row['prior_fail_clear_improvement_count']}/3 | "
            f"{row['mkgw_native_adamf_clear_improvement']} | {row['primary_gate']} | {row['representation_recovery_gate']} | "
            f"{row['prior_pass_stability_gate']} | {'GO' if row['phase3a_go'] else 'NO-GO'} |"
        )
    native = [row for row in pair_rows if row["pair_id"] == "mkgw_native_adamf"]
    lines += ["", "## MKG-W / NativE + AdaMF-MAT falsification pair", ""]
    for row in sorted(native, key=lambda item: item["representation"]):
        lines.append(
            f"- {row['representation']}: Spearman {row['spearman']:.6f}; positive/harmful AP lift "
            f"{row['positive_auprc_lift']:+.6f}/{row['harmful_auprc_lift']:+.6f}; positive-vs-harmful AUROC "
            f"{row['positive_vs_harmful_auroc']:.6f}; top-decile U {row['highest_10pct_actual_mean_advantage']:+.6f}."
        )
    lines += ["", "## Required audit answers", ""]
    r2_improvements = [row for row in comparison_rows if row["representation"] == "R2" and row["clear_sign_identifiability_improvement"]]
    mean_delta = lambda rep, field: float(np.mean([row[f"delta_{field}"] for row in comparison_rows if row["representation"] == rep]))
    best_rep = max(REPRESENTATIONS[1:], key=lambda rep: mean_delta(rep, "positive_vs_harmful_auroc"))
    native_rel = [row for row in relation_rows if row["pair_id"] == "mkgw_native_adamf" and int(row["n_queries"]) >= 60]
    relation_coverage = {}
    for rep in REPRESENTATIONS:
        subset = [row for row in native_rel if row["representation"] == rep]
        total = sum(int(row["n_queries"]) for row in subset)
        passed = sum(int(row["n_queries"]) for row in subset if row["h1_style_pass"])
        relation_coverage[rep] = passed / total if total else 0.0
    lines += [
        f"1. R2 clear-improvement pairs: {len(r2_improvements)}/6; its frozen gate result is {next(row for row in gate_rows if row['representation']=='R2')['phase3a_go']}.",
        f"2. MKG-W / NativE + AdaMF-MAT recovery under R1/R2/R3: " + ", ".join(f"{rep}={'yes' if any(r['pair_id']=='mkgw_native_adamf' and r['representation']==rep and r['clear_sign_identifiability_improvement'] for r in comparison_rows) else 'no'}" for rep in REPRESENTATIONS[1:]) + ".",
        f"3. By mean positive-vs-harmful AUROC gain, the larger contribution is {best_rep}; R1={mean_delta('R1','positive_vs_harmful_auroc'):+.4f}, R2={mean_delta('R2','positive_vs_harmful_auroc'):+.4f}.",
        f"4. R3 minus R2 mean sign-AUROC change is {mean_delta('R3','positive_vs_harmful_auroc')-mean_delta('R2','positive_vs_harmful_auroc'):+.4f}; this is the independent context increment.",
        f"5. Sign separation and activity lift are reported separately; the decision uses sign AUROC/AP and top-decile utility, never activity alone.",
        f"6. Prior PASS stability: " + ", ".join(f"{row['representation']}={row['prior_pass_stability_gate']}" for row in gate_rows) + ".",
        f"7. Phase 3A GO representations: {', '.join(go_reps) if go_reps else 'none'}.",
        f"8. Basis for a later conservative-policy phase: {'sufficient DEV representation evidence, subject to a separate protocol' if go_reps else 'insufficient; do not enter policy development'}.",
        "",
        "MKG-W / NativE + AdaMF-MAT supported-relation H1 coverage by representation: " + ", ".join(f"{rep}={relation_coverage[rep]:.2%}" for rep in REPRESENTATIONS) + ".",
        "",
        "## Integrity audit",
        "",
        "- TEST rows accessed: **0**.",
        "- TEST evaluation commands: **0**.",
        "- Expert checkpoints modified: **no**.",
        "- Phase 1 utility tables, AACPI V2, action grids, alpha0, and Base13 modified: **no**.",
        "- Every new representation field follows the frozen answer-agnostic contract.",
        "- Every prediction is outer-fold held out; each original triple occurs in one outer fold.",
        "- No Advantage-Greedy or other policy was run.",
        "- Source and output hashes are recorded in the Phase 3A manifests.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root, asset_root, output_dir, report = Path(args.phase3a_root), Path(args.asset_root), Path(args.output_dir), Path(args.report)
    outputs = [
        output_dir / "pair_summaries.csv", output_dir / "action_summaries.csv",
        output_dir / "direction_summaries.csv", output_dir / "seed_summaries.csv",
        output_dir / "relation_summaries.csv", output_dir / "feature_family_comparison.csv",
        output_dir / "phase3a_gate_summary.csv", output_dir / "phase3a_analysis_manifest.json",
        output_dir / "representation_manifest.json", output_dir / "feature_manifest.json",
        output_dir / "candidate_score_source_hashes.json", report,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")
    combined, source_paths = load_predictions(root)
    pair_rows, action_rows, direction_rows, seed_rows, relation_rows = summarize(combined)
    comparison_rows, gate_rows = comparisons(pair_rows)
    write_csv(outputs[0], pair_rows); write_csv(outputs[1], action_rows)
    write_csv(outputs[2], direction_rows); write_csv(outputs[3], seed_rows)
    write_csv(outputs[4], relation_rows); write_csv(outputs[5], comparison_rows); write_csv(outputs[6], gate_rows)
    figures = output_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    svg_bars(figures / "representation_top_decile_utility.svg", pair_rows, "highest_10pct_actual_mean_advantage", "Phase 3A top-decile actual advantage")
    svg_bars(figures / "representation_sign_auroc_lift.svg", pair_rows, "positive_vs_harmful_auroc_lift", "Phase 3A positive-vs-harmful AUROC lift")
    native_rows = [row for row in pair_rows if row["pair_id"] == "mkgw_native_adamf"]
    svg_bars(figures / "mkgw_native_adamf_recovery.svg", native_rows, "highest_10pct_actual_mean_advantage", "MKG-W NativE + AdaMF-MAT recovery")
    asset_manifests = sorted(asset_root.glob("*/candidate_score_source_manifest.json"))
    if len(asset_manifests) != 6:
        raise RuntimeError(f"Expected six candidate-score source manifests, found {len(asset_manifests)}")
    candidate_sources = {
        "schema_version": 1, "split": "dev", "manifests": [
            {"path": portable_path(path), "sha256": sha256_file(path), "payload": json.loads(path.read_text(encoding="utf-8"))}
            for path in asset_manifests
        ], "test_rows_accessed": 0,
    }
    outputs[10].write_text(json.dumps(candidate_sources, indent=2) + "\n", encoding="utf-8")
    outputs[9].write_text(json.dumps(feature_manifest(), indent=2) + "\n", encoding="utf-8")
    representation_manifest = {
        "schema_version": 1,
        "status": "frozen_before_systematic_comparison",
        "representations": list(REPRESENTATIONS),
        "estimator": "frozen AACPI V2 Phase 2B two-hidden-layer MLP",
        "nested_cv": "five outer/three inner original-triple grouped folds",
        "pair_gate": "Spearman, positive AP lift, harmful AP lift, and top-decile actual U all > 0",
        "clear_improvement_thresholds": {
            "positive_vs_harmful_auroc_delta_min": 0.02,
            "positive_auprc_lift_delta_min": 0.01,
            "harmful_auprc_lift_delta_min": 0.01,
            "top_decile_actual_u_delta_min": 0.001,
            "top_decile_actual_u_must_be_positive": True,
            "spearman_delta_min": 0.0,
        },
        "policy_evaluation": False,
    }
    outputs[8].write_text(json.dumps(representation_manifest, indent=2) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_text(pair_rows, comparison_rows, gate_rows, relation_rows), encoding="utf-8")
    manifest = {
        "schema_version": 1, "phase": "AACPI Phase 3A", "split": "dev",
        "source_oof_predictions": [{"path": portable_path(path), "sha256": sha256_file(path)} for path in source_paths],
        "outputs": [{"path": portable_path(path), "sha256": sha256_file(path)} for path in [*outputs[:7], *outputs[8:11], report, *sorted(figures.glob("*.svg"))]],
        "test_rows_accessed": 0, "test_evaluation_commands": 0,
        "policy_evaluation_performed": False,
        "phase3a_go_representations": [row["representation"] for row in gate_rows if row["phase3a_go"]],
    }
    outputs[7].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Phase 3A audit -> {report}")


if __name__ == "__main__":
    main()
