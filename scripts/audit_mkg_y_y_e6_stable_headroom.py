from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.audit_stable_headroom as core
from scripts.audit_mkg_y_y_e3_resolution import append_code_inventory, direct_source_inventory
from scripts.audit_mkg_y_y_e4_cross_seed_transfer import PAIR_IDS, PAIR_LABELS, load_pair
from scripts.exp2_information_common import ALPHAS, portable_path, reject_test_path, sha256_file


OUTCOME = "Y_E6_CLOSURE_REPLICATION_REPORTED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen MKG-Y Y-E6 DEV closure replication.")
    parser.add_argument("--contract", default="docs/protocols/MKG_Y_Y_E6_STABLE_HEADROOM_CONTRACT.json")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/mkg_y_y_e6_stable_headroom")
    parser.add_argument("--report", default="docs/reports/mkg_y_stable_headroom_gain_concentration_audit_2026-09-08.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def load_contract(path: Path) -> dict:
    reject_test_path(path)
    contract = load_json(path)
    if contract.get("status") != "frozen_before_systematic_run" or contract.get("split") != "dev":
        raise RuntimeError("MKG-Y Y-E6 contract is not frozen DEV-only")
    if tuple(contract.get("pair_ids", [])) != PAIR_IDS:
        raise RuntimeError("MKG-Y Y-E6 pair inventory changed")
    if tuple(contract.get("seeds", [])) != (1, 2, 3):
        raise RuntimeError("MKG-Y Y-E6 seed inventory changed")
    if tuple(contract.get("directions", [])) != ("head", "tail"):
        raise RuntimeError("MKG-Y Y-E6 direction inventory changed")
    if not np.allclose(contract.get("alpha_grid", []), ALPHAS):
        raise RuntimeError("MKG-Y Y-E6 alpha grid changed")
    if contract.get("assessment", {}).get("classification") is not False:
        raise RuntimeError("MKG-Y Y-E6 must remain a descriptive replication")
    prohibited = (
        "test_access", "test_commands", "checkpoint_retraining", "checkpoint_reselection",
        "new_selector", "new_feature", "new_representation", "policy_tuning",
        "action_grid_modification", "prior_result_modification",
    )
    if any(int(contract.get(field, -1)) != 0 for field in prohibited):
        raise RuntimeError("MKG-Y Y-E6 prohibited-operation boundary changed")
    return contract


def write_concentration_curves(concentration: pd.DataFrame, path: Path) -> None:
    width, height = 900, 610
    left, right, top, bottom = 90, 830, 80, 520
    colors = ("#28778e", "#48a08a", "#c45d3c")
    parts = core.svg_start(width, height, "MKG-Y Cumulative Oracle-Gain Concentration")
    parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#ddd"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{top}" stroke="#aaa" stroke-dasharray="5 5"/>')
    for pair_id, color in zip(PAIR_IDS, colors):
        group = concentration.loc[concentration.pair_id == pair_id].sort_values("descending_gain_rank")
        xs = left + group.cumulative_triple_fraction.to_numpy() * (right - left)
        ys = bottom - group.cumulative_gain_fraction.to_numpy() * (bottom - top)
        points = f"{left},{bottom} " + " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + value * (right - left)
        y = bottom - value * (bottom - top)
        core.svg_text(parts, x, bottom + 24, f"{value:.0%}", 10, "middle")
        core.svg_text(parts, left - 10, y + 4, f"{value:.0%}", 10, "end")
    for index, (pair_id, color) in enumerate(zip(PAIR_IDS, colors)):
        y = 548 + index * 18
        core.svg_text(parts, 110, y, PAIR_LABELS[pair_id], 10, fill=color)
    core.svg_text(parts, (left + right) / 2, 590, "Top original-triple fraction", 11, "middle")
    core.svg_text(parts, 18, 300, "Cumulative gain fraction", 11)
    core.svg_text(parts, 560, 566, "Diagonal: uniform contribution reference", 10, fill="#666")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def format_ci(row, stem: str) -> str:
    ci_stem = {
        "raw_oracle_headroom": "raw",
        "consensus_headroom": "consensus",
        "stable2_headroom": "stable2",
        "stable3_headroom": "stable3",
        "loso_stable_headroom": "loso",
        "frozen_x4_oof_gain": "frozen_x4",
    }[stem]
    return f"{getattr(row, stem):+.6f} [{getattr(row, ci_stem + '_ci95_low'):+.6f}, {getattr(row, ci_stem + '_ci95_high'):+.6f}]"


def write_report(summary: pd.DataFrame, y_e5: dict, assessment: dict, path: Path, contract: dict) -> None:
    lines = [
        "# MKG-Y Y-E6 Stable Headroom and Gain Concentration Audit",
        "", f"Date: {contract['effective_date']}", "Split: DEV only", "",
        "## Outcome", "",
        f"**{OUTCOME}** (descriptive external replication; no progression gate or route selection).", "",
        "## Stable headroom decomposition", "",
        "| Pair | Raw Oracle | Consensus | 2-of-3 | 3-of-3 | LOSO | Frozen X4 OOF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {format_ci(row, 'raw_oracle_headroom')} "
            f"| {format_ci(row, 'consensus_headroom')} | {format_ci(row, 'stable2_headroom')} "
            f"| {format_ci(row, 'stable3_headroom')} | {format_ci(row, 'loso_stable_headroom')} "
            f"| {format_ci(row, 'frozen_x4_oof_gain')} |"
        )
    lines += [
        "", "Consensus and two-/three-of-three quantities are all-seed ex-post upper diagnostics. LOSO is a held-out-seed diagnostic. Frozen X4 is the strict-OOF inference-time observable probe. They are not equivalent deployable policies.", "",
        "## Recoveries and stable opportunities", "",
        "| Pair | Consensus recovery | 2-of-3 recovery | 3-of-3 recovery | LOSO recovery | P(A2) | P(A3) | P(consensus != alpha0) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {row.consensus_recovery:.1%} | {row.stable2_recovery:.1%} "
            f"| {row.stable3_recovery:.1%} | {row.loso_recovery:.1%} | {row.stable2_opportunity_rate:.1%} "
            f"| {row.stable3_opportunity_rate:.1%} | {row.consensus_changed_rate:.1%} |"
        )
    lines += [
        "", "## Oracle gain concentration", "",
        "| Pair | Top1 | Top5 | Top10 | Top20 | Q50 | Gini | Effective support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {row.top1_gain_share:.1%} | {row.top5_gain_share:.1%} "
            f"| {row.top10_gain_share:.1%} | {row.top20_gain_share:.1%} | {row.q50:.1%} "
            f"| {row.gini:.3f} | {row.effective_support:.1%} |"
        )
    local_count = int(y_e5.get("evidence", {}).get("local_signal_pairs", 0))
    lines += [
        "", "The concentration curve orders non-negative original-triple Oracle gains from largest to smallest. Its diagonal denotes uniform contribution, not a random baseline.", "",
        "## Descriptive closure evidence", "",
        f"- 2-of-3 headroom CI lower > 0: {assessment['stable2_ci_lower_positive_pairs']}/3 pairs.",
        f"- Median 2-of-3 recovery: {assessment['median_stable2_recovery']:.1%}.",
        f"- 3-of-3 headroom CI lower > 0: {assessment['stable3_ci_lower_positive_pairs']}/3 pairs.",
        f"- Median 3-of-3 recovery: {assessment['median_stable3_recovery']:.1%}.",
        f"- Top10 contributes at least 50%: {assessment['top10_share_gte_50pct_pairs']}/3 pairs.",
        f"- Q50 is at most 10%: {assessment['q50_lte_10pct_pairs']}/3 pairs.",
        f"- Frozen Y-E5 LOCAL_SIGNAL_PAIR count: {local_count}/3.", "",
        "These counts are external-replication diagnostics. The six-pair Experiment 6 classification thresholds are not reapplied or rescaled to three MKG-Y pairs.", "",
        "## Evidence funnel", "",
        "Raw Available → Seed-Stable / Transferable → Locally Observable → Empirically Deployable remains the interpretation order. Raw Oracle is not treated as headroom that merely awaits a better selector.", "",
        "## Figures", "",
        "1. figure1_stable_headroom_decomposition.svg",
        "2. figure2_stable_recovery_forest.svg",
        "3. figure3_oracle_gain_concentration_curves.svg",
        "4. figure4_gain_concentration_summary.svg", "",
        "## Integrity audit", "",
        "- TEST access = 0", "- checkpoint inference/retraining/reselection = 0",
        "- new selector / feature / representation = 0", "- policy tuning = 0",
        "- action-grid modification = 0", "- Y-E1–Y-E5 result modification = 0",
        f"- original-triple clustered bootstrap = {contract['bootstrap']['samples']} replicates",
        "- all direct source/output hashes recorded = yes", "- MKG-Y TEST remains locked",
        "- subsequent stage started = 0", "", OUTCOME,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    for path in (contract_path, output_dir, report_path):
        reject_test_path(path)
    contract = load_contract(contract_path)
    source_inventory = direct_source_inventory(contract_path, contract)
    source_inventory = append_code_inventory(source_inventory, [
        Path(__file__),
        Path("scripts/run_mkg_y_y_e6_stable_headroom.ps1"),
        Path("scripts/audit_stable_headroom.py"),
        Path("scripts/audit_mkg_y_y_e4_cross_seed_transfer.py"),
        Path("scripts/audit_cross_seed_transfer.py"),
        Path("scripts/exp2_information_common.py"),
    ])
    exp1_stats = pd.read_csv(contract["source_boundaries"]["available_headroom"]["path"])
    exp4_summary = pd.read_csv(contract["source_boundaries"]["y_e4_pair_summary"]["path"])
    y_e1 = load_json(Path(contract["source_boundaries"]["y_e1_audit"]["path"]))
    y_e4 = load_json(Path(contract["source_boundaries"]["y_e4_audit"]["path"]))
    y_e5 = load_json(Path(contract["source_boundaries"]["y_e5_assessment"]["path"]))
    if y_e1.get("assessment", {}).get("outcome") != "Y_E1_AVAILABLE_COMPLEMENTARITY_PRESENT":
        raise RuntimeError("MKG-Y Y-E1 is not frozen and complete")
    if y_e4.get("assessment", {}).get("outcome") != "Y_E4_CROSS_SEED_REPLICATION_REPORTED":
        raise RuntimeError("MKG-Y Y-E4 is not frozen and complete")
    if y_e5.get("final_classification") != "Y_E5_LOCAL_IDENTIFIABILITY_REPLICATION_REPORTED":
        raise RuntimeError("MKG-Y Y-E5 is not frozen and complete")
    if set(exp1_stats.pair_id) != set(PAIR_IDS) or set(exp4_summary.pair_id) != set(PAIR_IDS):
        raise RuntimeError("MKG-Y frozen pair inventory mismatch")

    core.PAIR_IDS = PAIR_IDS
    core.PAIR_LABELS = PAIR_LABELS
    loaded = []
    preflight = []
    for pair_id in PAIR_IDS:
        frame, rr, alpha0, _, _ = load_pair(pair_id, contract, exp1_stats)
        loaded.append((pair_id, frame, rr, alpha0))
        preflight.append({
            "pair_id": pair_id,
            "rows": int(len(frame)),
            "query_identities": int(len(frame) // 3),
            "original_triples": int(frame.original_triple_id.nunique()),
            "alpha0": alpha0,
        })
    if args.dry_run:
        print(json.dumps({
            "status": "PRECHECK_OK", "split": "dev", "pairs": preflight,
            "verified_source_count": len(source_inventory),
            "test_access": 0, "checkpoint_execution": 0,
        }, indent=2))
        return

    if not args.overwrite and ((output_dir.exists() and any(output_dir.iterdir())) or report_path.exists()):
        raise FileExistsError("Refusing to overwrite existing MKG-Y Y-E6 audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_query_frames = []
    concentration_frames = []
    summaries = []
    bootstrap_rows = []
    for pair_id, frame, rr, alpha0 in loaded:
        per_query, concentration, summary, boot = core.analyze_pair(
            pair_id, frame, rr, alpha0,
            exp1_stats.loc[exp1_stats.pair_id == pair_id].iloc[0],
            exp4_summary.loc[exp4_summary.pair_id == pair_id].iloc[0],
            contract,
        )
        per_query_frames.append(per_query)
        concentration_frames.append(concentration)
        summaries.append(summary)
        bootstrap_rows.extend(boot)
        print(
            f"[OK] {pair_id}: raw={summary['raw_oracle_headroom']:.6f}, "
            f"stable2={summary['stable2_headroom']:.6f}, stable3={summary['stable3_headroom']:.6f}"
        )

    per_query = pd.concat(per_query_frames, ignore_index=True)
    concentration = pd.concat(concentration_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    bootstrap = pd.DataFrame(bootstrap_rows)
    assessment = {
        "outcome": OUTCOME,
        "is_progression_gate": False,
        "classification": False,
        "route_selection": False,
        "stable2_ci_lower_positive_pairs": int((summary.stable2_ci95_low > 0).sum()),
        "median_stable2_recovery": float(summary.stable2_recovery.median()),
        "stable3_ci_lower_positive_pairs": int((summary.stable3_ci95_low > 0).sum()),
        "median_stable3_recovery": float(summary.stable3_recovery.median()),
        "top10_share_gte_50pct_pairs": int((summary.top10_gain_share >= 0.50).sum()),
        "q50_lte_10pct_pairs": int((summary.q50 <= 0.10).sum()),
        "y_e5_local_signal_pairs": int(y_e5.get("evidence", {}).get("local_signal_pairs", 0)),
        "next_step": "cross_dataset_closure_synthesis",
        "test_status": "LOCKED",
    }

    per_query_path = output_dir / "stable_headroom_per_query.csv.gz"
    stable_summary_path = output_dir / "stable_headroom_pair_summary.csv"
    concentration_path = output_dir / "concentration_per_triple.csv.gz"
    concentration_summary_path = output_dir / "concentration_pair_summary.csv"
    bootstrap_path = output_dir / "bootstrap_ci.csv"
    assessment_path = output_dir / "replication_assessment.json"
    figure_paths = [
        output_dir / "figure1_stable_headroom_decomposition.svg",
        output_dir / "figure2_stable_recovery_forest.svg",
        output_dir / "figure3_oracle_gain_concentration_curves.svg",
        output_dir / "figure4_gain_concentration_summary.svg",
    ]
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    per_query.to_csv(per_query_path, index=False, compression=compression, lineterminator="\n")
    concentration.to_csv(concentration_path, index=False, compression=compression, lineterminator="\n")
    stable_columns = [
        column for column in summary.columns
        if not column.startswith(("top1_", "top5_", "top10_", "top20_", "q50", "gini", "effective_support", "total_gain"))
    ]
    concentration_columns = [
        "dataset", "pair_id", "pair_label", "n_original_triples", "raw_oracle_headroom",
        "total_gain", "top1_gain_share", "top1_triple_count", "top5_gain_share", "top5_triple_count",
        "top10_gain_share", "top10_triple_count", "top20_gain_share", "top20_triple_count",
        "q50", "q50_triple_count", "gini", "effective_support",
    ]
    summary[stable_columns].to_csv(stable_summary_path, index=False, lineterminator="\n")
    summary[concentration_columns].to_csv(concentration_summary_path, index=False, lineterminator="\n")
    bootstrap.to_csv(bootstrap_path, index=False, lineterminator="\n")
    assessment_path.write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")

    core.write_headroom_figure(summary, figure_paths[0])
    core.write_recovery_figure(summary, figure_paths[1])
    write_concentration_curves(concentration, figure_paths[2])
    core.write_concentration_summary(summary, figure_paths[3])
    write_report(summary, y_e5, assessment, report_path, contract)

    audit_path = output_dir / "audit_manifest.json"
    inventory = {row["path"]: row for row in source_inventory}
    output_paths = [
        per_query_path, stable_summary_path, concentration_path, concentration_summary_path,
        bootstrap_path, assessment_path, *figure_paths, report_path,
    ]
    for path in output_paths:
        inventory[portable_path(path)] = {
            "path": portable_path(path), "sha256": sha256_file(path), "role": "y_e6_output",
        }
    audit = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "split": "dev",
        "dataset": "mkg_y",
        "assessment": assessment,
        "preflight": preflight,
        "source_and_output_hash_count": len(inventory),
        "sources_and_outputs": [inventory[path] for path in sorted(inventory)],
        "hash_inventory_note": "audit_manifest.json self-hash excluded to avoid recursive content",
        "operational_audit": {
            "test_access": 0, "test_commands": 0, "checkpoint_execution": 0,
            "checkpoint_retraining": 0, "checkpoint_reselection": 0,
            "new_selector": 0, "new_feature": 0, "new_representation": 0,
            "policy_tuning": 0, "action_grid_modification": 0,
            "prior_result_modification": 0, "original_triple_bootstrap_intact": True,
        },
        "decision_memo_generated": False,
        "test_status": "LOCKED",
        "next_step_started": 0,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
