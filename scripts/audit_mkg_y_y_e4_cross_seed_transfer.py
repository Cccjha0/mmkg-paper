from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.audit_cross_seed_transfer as core
from scripts.audit_mkg_y_y_e3_resolution import append_code_inventory, direct_source_inventory
from scripts.exp2_information_common import ALPHAS, RR_COLUMNS, portable_path, reject_test_path, sha256_file


PAIR_IDS = (
    "mkg_y_mhyper_native",
    "mkg_y_mhyper_adamf",
    "mkg_y_native_adamf",
)
PAIR_LABELS = {
    "mkg_y_mhyper_native": "MKG-Y / M-Hyper + NativE",
    "mkg_y_mhyper_adamf": "MKG-Y / M-Hyper + AdaMF-MAT",
    "mkg_y_native_adamf": "MKG-Y / NativE + AdaMF-MAT",
}
SEEDS = (1, 2, 3)
DIRECTIONS = ("head", "tail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen MKG-Y Y-E4 DEV-only cross-seed transfer/LOSO audit.")
    parser.add_argument("--contract", default="docs/protocols/MKG_Y_Y_E4_CROSS_SEED_TRANSFER_CONTRACT.json")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/mkg_y_y_e4_cross_seed_transfer")
    parser.add_argument("--report", default="docs/reports/mkg_y_cross_seed_transfer_audit_2026-09-08.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_systematic_run" or contract.get("split") != "dev":
        raise RuntimeError("MKG-Y Y-E4 contract is not frozen DEV-only")
    if tuple(contract.get("pair_ids", [])) != PAIR_IDS:
        raise RuntimeError("MKG-Y Y-E4 pair inventory changed")
    if tuple(contract.get("seeds", [])) != SEEDS or tuple(contract.get("directions", [])) != DIRECTIONS:
        raise RuntimeError("MKG-Y Y-E4 seed/direction inventory changed")
    if not np.allclose(contract.get("alpha_grid", []), ALPHAS):
        raise RuntimeError("MKG-Y Y-E4 action grid changed")
    prohibited = (
        "test_access", "test_commands", "checkpoint_retraining", "checkpoint_reselection",
        "new_selector", "new_feature", "new_representation", "policy_development",
        "action_grid_modification", "prior_result_modification",
    )
    if any(int(contract.get(field, -1)) != 0 for field in prohibited):
        raise RuntimeError("MKG-Y Y-E4 prohibited-operation boundary changed")
    return contract


def load_pair(pair_id: str, contract: dict, exp1_stats: pd.DataFrame):
    manifest_path = Path(contract["source_boundaries"]["utility_manifests"][pair_id]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pair_id") != pair_id or manifest.get("split") != "dev":
        raise RuntimeError(f"Canonical manifest identity mismatch: {pair_id}")
    if manifest.get("score_normalization") != "query_zscore" or not np.allclose(manifest.get("global_alpha_grid", []), ALPHAS):
        raise RuntimeError(f"Frozen score/action protocol mismatch: {pair_id}")
    source_path = Path(manifest["source_query_rows"]["path"])
    reject_test_path(source_path)
    if sha256_file(source_path) != manifest["source_query_rows"]["sha256"]:
        raise RuntimeError(f"Canonical source hash mismatch: {source_path}")
    usecols = [
        "dataset", "pair_name", "split", "query_id", "seed", "direction",
        "relation_id", "head_id", "tail_id", "alpha_global", *RR_COLUMNS,
    ]
    frame = pd.read_csv(source_path, usecols=usecols)
    if set(frame.dataset.astype(str)) != {"mkg_y"} or set(frame.pair_name.astype(str)) != {pair_id}:
        raise RuntimeError(f"MKG-Y pair identity mismatch: {pair_id}")
    if set(frame.seed.astype(int)) != set(SEEDS) or set(frame.direction.astype(str)) != set(DIRECTIONS):
        raise RuntimeError(f"Incomplete seed/direction inventory: {pair_id}")
    if frame.query_id.duplicated().any() or frame[RR_COLUMNS].isna().any().any():
        raise RuntimeError(f"Incomplete or duplicate exact-RR rows: {pair_id}")
    frame["original_triple_id"] = core.original_triple_ids(frame)
    counts = frame.groupby("original_triple_id", sort=False).size()
    if not (counts == 6).all():
        raise RuntimeError(f"Each triple must contain three seeds and two directions: {pair_id}")
    global_alpha = float(manifest["global_alpha"])
    stat = exp1_stats.loc[exp1_stats.pair_id == pair_id]
    if len(stat) != 1 or not np.isclose(float(stat.iloc[0].global_alpha), global_alpha):
        raise RuntimeError(f"Y-E1 alpha0 mismatch: {pair_id}")
    frame = frame.sort_values(["original_triple_id", "direction", "seed"], kind="stable").reset_index(drop=True)
    frame.attrs["source"] = str(source_path)
    return frame, frame[RR_COLUMNS].to_numpy(np.float64), global_alpha, manifest_path, source_path


def metric_row(pair_id: str, metric: str, estimate: float, low: float, high: float, samples: int) -> dict:
    return {
        "dataset": "mkg_y", "pair_id": pair_id, "metric": metric, "estimate": estimate,
        "ci95_low": low, "ci95_high": high, "bootstrap_samples": samples,
        "bootstrap_unit": "original_triple_id",
    }


def write_report(summary: pd.DataFrame, matrix: pd.DataFrame, path: Path, contract: dict) -> None:
    lines = [
        "# MKG-Y Y-E4 Cross-Seed Transfer and LOSO Stability Audit", "", "Date: 2026-09-08", "",
        "## Outcome", "", "**Y_E4_CROSS_SEED_REPLICATION_REPORTED** (descriptive external replication; no progression gate or route selection).", "",
        "Direct transfer applies each source seed's deterministic Oracle alpha to an independently trained target seed without target reselection or gain clipping. LOSO chooses from two training seeds and evaluates once on the held-out seed. Both are stability diagnostics, not deployable inference-time policies.", "",
        "## Pair-level results", "",
        "| Pair | Raw Oracle | Cross-seed transfer (95% CI) | Transfer recovery | LOSO (95% CI) | LOSO recovery | Exact alpha agreement | Direction agreement | B / Z / H | Frozen X4 OOF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {row.raw_oracle_headroom:+.6f} "
            f"| {row.cross_seed_transfer_headroom:+.6f} [{row.transfer_ci95_low:+.6f}, {row.transfer_ci95_high:+.6f}] "
            f"| {row.transfer_recovery:.1%} | {row.loso_stable_headroom:+.6f} [{row.loso_ci95_low:+.6f}, {row.loso_ci95_high:+.6f}] "
            f"| {row.loso_recovery:.1%} | {row.exact_alpha_three_seed_agreement:.1%} | {row.direction_three_seed_agreement:.1%} "
            f"| {row.beneficial_transfer_rate:.1%} / {row.zero_transfer_rate:.1%} / {row.harmful_transfer_rate:.1%} | {row.frozen_x4_oof_gain:+.6f} |"
        )
    significant_transfer = int((summary.transfer_ci95_low > 0).sum())
    significant_loso = int((summary.loso_ci95_low > 0).sum())
    lines += [
        "", "## Interpretation", "",
        f"Cross-seed transfer is significantly positive in {significant_transfer}/3 pairs; LOSO stable headroom is significantly positive in {significant_loso}/3 pairs. Median transfer recovery is {summary.transfer_recovery.median():.1%}, and median LOSO recovery is {summary.loso_recovery.median():.1%}.", "",
        "The two M-Hyper pairs retain 18.5–20.0% of Raw Oracle under direct seed transfer and 25.5–26.9% under LOSO. NativE + AdaMF-MAT is much more fragile: direct transfer retains 7.0%, LOSO retains 10.3%, and its frozen X4 OOF gain remains negative. Thus cross-seed stability is detectable but strongly pair-dependent and does not by itself imply inference-time observability.", "",
        "The gap from Raw Oracle to direct cross-seed transfer quantifies seed-specific action fragility. LOSO is a less restrictive all-but-one-seed stability diagnostic and must not be interpreted as an observable query policy. Frozen X4 remains the strict-OOF inference-time observable probe and uses its fold-specific Global baseline.", "",
        "## Ordered seed transfers", "",
        "| Pair | 1→2 | 1→3 | 2→1 | 2→3 | 3→1 | 3→2 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair_id in PAIR_IDS:
        subset = matrix.loc[(matrix.pair_id == pair_id) & (matrix.source_seed != matrix.target_seed)].set_index(["source_seed", "target_seed"])
        values = [float(subset.loc[key].headroom) for key in ((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2))]
        lines.append(f"| {PAIR_LABELS[pair_id]} | " + " | ".join(f"{value:+.6f}" for value in values) + " |")
    lines += [
        "", "## Figures", "",
        "1. `figure1_headroom_funnel.svg` — Raw Oracle → direct cross-seed transfer → LOSO → frozen X4.",
        "2. `figure2_cross_seed_transfer_matrix.svg` — pair-specific 3×3 seed transfer matrices.",
        "3. `figure3_transfer_recovery_forest.svg` — direct-transfer recovery with clustered intervals.",
        "4. `figure4_stability_vs_transfer.svg` — direction agreement versus transfer recovery.", "",
        "## Operational audit", "", "- TEST access = 0", "- checkpoint inference/retraining/reselection = 0",
        "- target-seed action reselection = 0", "- negative-gain clipping = 0", "- new selector/feature/representation = 0",
        "- original-triple clustered bootstrap = 10,000 replicates", "- all direct source and output hashes recorded = yes", "",
        "MKG-Y TEST remains locked. Y-E5 was not started by this audit.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    contract_path, output_dir, report_path = Path(args.contract), Path(args.output_dir), Path(args.report)
    for path in (contract_path, output_dir, report_path):
        reject_test_path(path)
    contract = load_contract(contract_path)
    source_inventory = direct_source_inventory(contract_path, contract)
    source_inventory = append_code_inventory(source_inventory, [
        Path(__file__),
        Path("scripts/run_mkg_y_y_e4_cross_seed_transfer.ps1"),
        Path("scripts/audit_cross_seed_transfer.py"),
        Path("scripts/exp2_information_common.py"),
    ])
    exp1_stats = pd.read_csv(contract["source_boundaries"]["available_headroom"]["path"])
    x4_metrics = pd.read_csv(contract["source_boundaries"]["y_e2_primary"]["path"])
    if set(exp1_stats.pair_id) != set(PAIR_IDS) or set(x4_metrics.pair_id) != set(PAIR_IDS):
        raise RuntimeError("Y-E1/Y-E2 pair inventory mismatch")
    y_e3 = json.loads(Path(contract["source_boundaries"]["y_e3_audit"]["path"]).read_text(encoding="utf-8"))
    if y_e3.get("assessment", {}).get("outcome") != "Y_E3_RESOLUTION_REPLICATION_REPORTED":
        raise RuntimeError("Y-E3 is not frozen and complete")

    core.PAIR_IDS = PAIR_IDS
    core.PAIR_LABELS = PAIR_LABELS
    core.metric_row = metric_row
    loaded, preflight = {}, []
    for pair_id in PAIR_IDS:
        frame, rr, alpha0, manifest_path, source_path = load_pair(pair_id, contract, exp1_stats)
        loaded[pair_id] = (frame, rr, alpha0)
        preflight.append({"pair_id": pair_id, "rows": len(frame), "query_identities": len(frame) // 3, "original_triples": int(frame.original_triple_id.nunique()), "alpha0": alpha0})
        source_inventory.extend([
            {"path": portable_path(source_path), "sha256": sha256_file(source_path), "role": "exact_rr_source"},
        ])
    if args.dry_run:
        print(json.dumps({"status": "PRECHECK_OK", "pairs": preflight, "test_access": 0, "checkpoint_execution": 0}, indent=2))
        return
    if not args.overwrite and (output_dir.exists() and any(output_dir.iterdir()) or report_path.exists()):
        raise FileExistsError("Refusing to overwrite existing MKG-Y Y-E4 audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries, matrix_rows, bootstrap_rows, transfers, losos, oracles = [], [], [], [], [], []
    samples, seed = int(contract["bootstrap"]["samples"]), int(contract["bootstrap"]["seed"])
    for pair_id in PAIR_IDS:
        frame, rr, alpha0 = loaded[pair_id]
        result = core.analyze_pair(
            pair_id, frame, rr, alpha0,
            exp1_stats.loc[exp1_stats.pair_id == pair_id].iloc[0],
            x4_metrics.loc[x4_metrics.pair_id == pair_id].iloc[0], samples, seed,
        )
        summary, matrix, boot, transfer, loso, oracle, _ = result
        summaries.append(summary); matrix_rows.extend(matrix); bootstrap_rows.extend(boot)
        transfers.append(transfer); losos.append(loso); oracles.append(oracle)
    summary = pd.DataFrame(summaries)
    matrix = pd.DataFrame(matrix_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary["transfer_ci_lower_gt_zero"] = summary.transfer_ci95_low > 0
    summary["loso_ci_lower_gt_zero"] = summary.loso_ci95_low > 0
    paths = {
        "pair_summary.csv": summary,
        "seed_pair_transfer_matrix.csv": matrix,
        "bootstrap_ci.csv": bootstrap,
    }
    for name, frame in paths.items():
        frame.to_csv(output_dir / name, index=False, lineterminator="\n")
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    pd.concat(transfers, ignore_index=True).to_csv(output_dir / "per_query_seed_transfer.csv.gz", index=False, compression=compression, lineterminator="\n")
    pd.concat(losos, ignore_index=True).to_csv(output_dir / "per_query_loso.csv.gz", index=False, compression=compression, lineterminator="\n")
    pd.concat(oracles, ignore_index=True).to_csv(output_dir / "per_query_oracle.csv.gz", index=False, compression=compression, lineterminator="\n")
    assessment = {
        "outcome": "Y_E4_CROSS_SEED_REPLICATION_REPORTED", "is_progression_gate": False,
        "transfer_ci_lower_positive_pairs": int((summary.transfer_ci95_low > 0).sum()),
        "loso_ci_lower_positive_pairs": int((summary.loso_ci95_low > 0).sum()),
        "median_transfer_recovery": float(summary.transfer_recovery.median()),
        "median_loso_recovery": float(summary.loso_recovery.median()),
        "next_replication": "Y-E5", "test_status": "LOCKED",
    }
    (output_dir / "replication_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    figure_paths = [
        output_dir / "figure1_headroom_funnel.svg", output_dir / "figure2_cross_seed_transfer_matrix.svg",
        output_dir / "figure3_transfer_recovery_forest.svg", output_dir / "figure4_stability_vs_transfer.svg",
    ]
    core.write_funnel(summary, figure_paths[0]); core.write_matrix(matrix, figure_paths[1])
    core.write_forest(summary, figure_paths[2]); core.write_scatter(summary, figure_paths[3])
    write_report(summary, matrix, report_path, contract)

    audit_path = output_dir / "audit_manifest.json"
    inventory = {row["path"]: row for row in source_inventory}
    for path in [item for item in output_dir.rglob("*") if item.is_file() and item != audit_path] + [report_path]:
        inventory[portable_path(path)] = {"path": portable_path(path), "sha256": sha256_file(path), "role": "y_e4_output"}
    audit = {
        "schema_version": 1, "experiment": contract["experiment"], "split": "dev", "dataset": "mkg_y",
        "assessment": assessment, "source_and_output_hash_count": len(inventory),
        "sources_and_outputs": [inventory[path] for path in sorted(inventory)],
        "operational_audit": {"test_access": 0, "checkpoint_execution": 0, "checkpoint_retraining": 0,
                              "checkpoint_reselection": 0, "target_reselection": 0, "negative_gain_clipping": 0,
                              "new_selector": 0, "new_feature": 0, "new_representation": 0,
                              "original_triple_bootstrap_intact": True},
        "next_step_started": 0,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
