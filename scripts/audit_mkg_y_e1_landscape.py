from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.audit_complementarity_landscape as landscape


ROOT = Path(__file__).resolve().parents[1]
PAIR_ORDER = [
    "mkg_y_mhyper_native",
    "mkg_y_mhyper_adamf",
    "mkg_y_native_adamf",
]
PAIR_DISPLAY = {
    "mkg_y_mhyper_native": "MKG-Y / M-Hyper + NativE",
    "mkg_y_mhyper_adamf": "MKG-Y / M-Hyper + AdaMF-MAT",
    "mkg_y_native_adamf": "MKG-Y / NativE + AdaMF-MAT",
}
PAIR_SHORT = {
    "mkg_y_mhyper_native": "MKG-Y\nM-Hyper+NativE",
    "mkg_y_mhyper_adamf": "MKG-Y\nM-Hyper+AdaMF",
    "mkg_y_native_adamf": "MKG-Y\nNativE+AdaMF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen MKG-Y Y-E1 DEV action-landscape audit.")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    parser.add_argument("--support-min", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def require_absent_or_overwrite(paths: list[Path], overwrite: bool) -> None:
    existing = [portable(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")


def source_file(manifest: dict, suffix: str) -> tuple[Path, str]:
    matches = [(path, digest) for path, digest in manifest["files"].items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one manifest file ending in {suffix!r}; found {len(matches)}")
    path, digest = matches[0]
    return repo_path(path), str(digest)


def write_report(
    path: Path,
    stats: pd.DataFrame,
    assessment: dict,
    output_dir: Path,
    source_records: list[dict],
    support_min: int,
) -> None:
    ordered = stats.set_index("pair_id").loc[PAIR_ORDER].reset_index()
    headroom_low = ordered["available_headroom"].min()
    headroom_high = ordered["available_headroom"].max()
    opportunity_low = ordered["positive_opportunity_rate"].min()
    opportunity_high = ordered["positive_opportunity_rate"].max()
    lines = [
        "# MKG-Y Y-E1 Available Complementarity Landscape Audit",
        "",
        "Date: 2026-09-07",
        "",
        "## Outcome",
        "",
        f"Descriptive replication outcome: **{assessment['outcome']}**.",
        "",
        "Y-E1 has no method-development or progression gate. It describes the complete frozen DEV action geometry and does not train a selector or start Y-E2.",
        "",
        "## Main findings",
        "",
        f"- All {assessment['significant_headroom_pairs']}/3 pairs have positive action-grid Oracle headroom with an original-triple clustered 95% CI lower bound above zero.",
        f"- Available headroom ranges from {headroom_low:.6f} to {headroom_high:.6f} MRR.",
        f"- Positive-opportunity prevalence ranges from {100*opportunity_low:.2f}% to {100*opportunity_high:.2f}% across pairs.",
        f"- The frozen Global alphas are {', '.join(f'{row.pair_id}={row.global_alpha:.2f}' for row in ordered.itertuples())}.",
        "- These results establish available complementarity only. They do not establish cross-seed stability, inference-time observability, or deployability.",
        "",
        "## Frozen protocol and boundary",
        "",
        "- Dataset: MKG-Y; pairs: M-Hyper + NativE, M-Hyper + AdaMF-MAT, and NativE + AdaMF-MAT.",
        "- Evidence: Y2 exact per-query filtered RR for alpha `0.00:0.05:1.00`, three seeds, and head/tail directions.",
        "- Score normalization: `query_zscore`; Global alpha is taken unchanged from each Y2 DEV selection.",
        "- MKG-Y DEV filtering uses TRAIN+DEV known facts only so no TEST row is opened or used as a filter fact.",
        "- Bootstrap resamples original triples while retaining all three seeds and both directions.",
        f"- Supported relation-direction groups require at least {support_min} seed-direction observations.",
        "- TEST access = 0; checkpoint evaluation = 0; retraining = 0; reselection = 0; policy training = 0.",
        "",
        "## Pair-level results",
        "",
        "| Pair | alpha0 | Global MRR | Oracle MRR | Headroom | Clustered 95% CI | Positive opportunities | G median | W median | D median* | Plateau mean | Fragmented positive | Direction consistency** |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered.itertuples():
        lines.append(
            f"| {PAIR_DISPLAY[row.pair_id]} | {row.global_alpha:.2f} | {row.global_mrr:.6f} | "
            f"{row.oracle_mrr:.6f} | {row.available_headroom:.6f} | "
            f"[{row.headroom_ci95_low:.6f}, {row.headroom_ci95_high:.6f}] | "
            f"{100*row.positive_opportunity_rate:.2f}% | {row.gain_median:.6f} | "
            f"{row.width_median:.3f} | {row.distance_median_positive_only:.2f} | "
            f"{row.plateau_ratio_mean:.3f} | {100*row.fragmented_positive_opportunity_rate:.2f}% | "
            f"{row.direction_consistency_macro_supported_relation_x_direction:.3f} |"
        )
    lines.extend(
        [
            "",
            "\* `D` is summarized over positive-opportunity queries only. ** Macro average over supported relation × direction groups.",
            "",
            "## Figures",
            "",
            f"1. [Global-to-Oracle dumbbell]({landscape.document_relative_link(output_dir / 'figure1_global_to_oracle.svg', path)})",
            f"2. [Highlighted action landscapes]({landscape.document_relative_link(output_dir / 'figure2_action_landscape_heatmaps.svg', path)})",
            f"3. [G W D distributions]({landscape.document_relative_link(output_dir / 'figure3_gwd_distributions.svg', path)})",
            f"4. [Relation by direction consistency]({landscape.document_relative_link(output_dir / 'figure4_relation_direction_consistency.svg', path)})",
            "",
            "## Reproducibility outputs",
            "",
            f"- Per-query geometry: `{portable(output_dir / 'per_query_action_geometry.csv.gz')}`",
            f"- Pair statistics: `{portable(output_dir / 'pair_statistics.csv')}`",
            f"- Machine-readable assessment: `{portable(output_dir / 'replication_assessment.json')}`",
            f"- Audit manifest: `{portable(output_dir / 'audit_manifest.json')}`",
            "",
            "## Source hashes",
            "",
            "| Pair | Role | Path | SHA256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in source_records:
        lines.append(f"| {record['pair_id']} | {record['role']} | `{record['path']}` | `{record['sha256']}` |")
    lines.extend(
        [
            "",
            "All source hashes were verified before analysis. The analytical input contains DEV rows only.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0 or args.support_min <= 0:
        raise ValueError("bootstrap-samples and support-min must be positive")
    contract_path = repo_path(args.contract)
    contract = read_json(contract_path)
    if contract.get("status") != "frozen_before_y_e1_run" or contract.get("split") != "dev":
        raise RuntimeError("MKG-Y Y-E1 contract is not frozen for DEV")
    if contract.get("test_access") != 0 or contract.get("filter_fact_scope") != "train_dev":
        raise RuntimeError("MKG-Y Y-E1 TEST boundary drift")
    if [pair["pair_id"] for pair in contract["pairs"]] != PAIR_ORDER:
        raise RuntimeError("MKG-Y Y-E1 pair order/inventory drift")
    for name, record in contract["inputs"].items():
        source_path = repo_path(record["path"])
        landscape.validate_hash(source_path, record["sha256"], f"contract_input:{name}")
        payload = read_json(source_path)
        if payload.get("status") not in {"PASS", "Y2_DEV_EXPORT_VERIFIED"}:
            raise RuntimeError(f"Y2 input did not pass: {name}")

    output_dir = repo_path(args.output_dir)
    report_path = repo_path(args.report)
    expected_outputs = [
        output_dir / "per_query_action_geometry.csv.gz",
        output_dir / "pair_statistics.csv",
        output_dir / "pair_statistics.json",
        output_dir / "gwd_plateau_distribution_summary.csv",
        output_dir / "fragmentation_statistics.csv",
        output_dir / "direction_consistency.csv",
        output_dir / "action_level_summary.csv",
        output_dir / "replication_assessment.json",
        output_dir / "audit_manifest.json",
        report_path,
    ]
    require_absent_or_overwrite(expected_outputs, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometries: dict[str, pd.DataFrame] = {}
    stats_rows, distributions, consistencies, actions = [], [], [], []
    source_records = [
        {
            "pair_id": "all",
            "role": name,
            "path": portable(repo_path(record["path"])),
            "sha256": record["sha256"],
        }
        for name, record in contract["inputs"].items()
    ]
    for role, path in (
        ("y_e1_contract", contract_path),
        ("y_e1_implementation", repo_path("scripts/audit_mkg_y_e1_landscape.py")),
        ("shared_landscape_implementation", repo_path("scripts/audit_complementarity_landscape.py")),
    ):
        source_records.append(
            {"pair_id": "all", "role": role, "path": portable(path), "sha256": landscape.sha256_file(path)}
        )
    for offset, pair in enumerate(contract["pairs"]):
        pair_id = pair["pair_id"]
        manifest_path = repo_path(pair["source_manifest"])
        if landscape.sha256_file(manifest_path) != pair["source_manifest_sha256"]:
            raise RuntimeError(f"Y2 source manifest hash mismatch: {pair_id}")
        manifest = read_json(manifest_path)
        pair_meta = manifest["pair"]
        if (
            pair_meta.get("status") != "PASS"
            or pair_meta.get("test_rows_opened") != 0
            or pair_meta.get("filter_fact_scope") != "train_dev"
            or pair_meta.get("alpha_grid") != landscape.ALPHA_GRID.tolist()
        ):
            raise RuntimeError(f"Invalid Y2 source boundary: {pair_id}")
        query_path, query_hash = source_file(manifest, "/dev_query_rows.csv")
        selection_path, selection_hash = source_file(manifest, "/selection.json")
        summary_path, summary_hash = source_file(manifest, "/dev_summary.json")
        for role, path, digest in (
            ("y2_source_manifest", manifest_path, pair["source_manifest_sha256"]),
            ("source_query_rows", query_path, query_hash),
            ("source_selection", selection_path, selection_hash),
            ("source_summary", summary_path, summary_hash),
        ):
            landscape.validate_hash(path, digest, f"{pair_id}:{role}")
            source_records.append({"pair_id": pair_id, "role": role, "path": portable(path), "sha256": digest})

        selection = read_json(selection_path)
        summary = read_json(summary_path)
        alpha0 = float(selection["global_alpha"])
        if (
            selection.get("pair_name") != pair_id
            or selection.get("dataset") != "mkg_y"
            or selection.get("alpha_grid") != landscape.ALPHA_GRID.tolist()
            or selection.get("score_normalization") != "query_zscore"
            or summary.get("test_rows_opened") != 0
            or summary.get("filter_fact_scope") != "train_dev"
        ):
            raise RuntimeError(f"Selection/summary contract mismatch: {pair_id}")

        use_columns = [
            "pair_name", "dataset", "expert_a_name", "expert_b_name", "query_id", "split",
            "seed", "direction", "relation_id", "head_id", "tail_id", "filter_fact_scope",
            *landscape.ALPHA_COLUMNS,
        ]
        source = pd.read_csv(query_path, usecols=use_columns)
        if set(source["split"].astype(str)) != {"dev"} or set(source["filter_fact_scope"].astype(str)) != {"train_dev"}:
            raise RuntimeError(f"Non-DEV or invalid filter-scope rows: {pair_id}")
        if set(source["pair_name"].astype(str)) != {pair_id}:
            raise RuntimeError(f"Pair id mismatch: {pair_id}")
        if sorted(source["seed"].astype(int).unique()) != [1, 2, 3] or set(source["direction"]) != {"head", "tail"}:
            raise RuntimeError(f"Seed/direction coverage mismatch: {pair_id}")
        cluster_sizes = source.groupby(["head_id", "relation_id", "tail_id"]).size()
        if set(cluster_sizes.unique()) != {6} or len(cluster_sizes) != 2665:
            raise RuntimeError(f"Original-triple cluster coverage mismatch: {pair_id}")

        geometry = landscape.compute_query_geometry(source, pair_id=pair_id, alpha0=alpha0)
        geometries[pair_id] = geometry
        stats_rows.append(
            landscape.pair_statistics(
                geometry,
                n_bootstrap=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + offset,
                support_min=args.support_min,
            )
        )
        distributions.extend(landscape.distribution_rows(geometry))
        consistencies.extend(landscape.direction_consistency_rows(geometry, args.support_min))
        for alpha in landscape.ALPHA_GRID:
            column = f"u_alpha_{alpha:.2f}".replace(".", "_")
            values = geometry[column]
            actions.append(
                {
                    "dataset": "mkg_y",
                    "pair_id": pair_id,
                    "global_alpha": alpha0,
                    "alpha": float(alpha),
                    "delta_alpha": float(alpha - alpha0),
                    "mean_utility": float(values.mean()),
                    "positive_rate": float((values > landscape.ZERO_TOLERANCE).mean()),
                    "zero_rate": float((np.abs(values) <= landscape.ZERO_TOLERANCE).mean()),
                    "negative_rate": float((values < -landscape.ZERO_TOLERANCE).mean()),
                }
            )

    all_geometry = pd.concat([geometries[pair] for pair in PAIR_ORDER], ignore_index=True)
    stats = pd.DataFrame(stats_rows).set_index("pair_id").loc[PAIR_ORDER].reset_index()
    distribution_frame = pd.DataFrame(distributions)
    consistency_frame = pd.DataFrame(consistencies)
    action_frame = pd.DataFrame(actions)
    all_geometry.to_csv(output_dir / "per_query_action_geometry.csv.gz", index=False)
    stats.to_csv(output_dir / "pair_statistics.csv", index=False)
    (output_dir / "pair_statistics.json").write_text(json.dumps(stats.to_dict(orient="records"), indent=2) + "\n", encoding="utf-8")
    distribution_frame.to_csv(output_dir / "gwd_plateau_distribution_summary.csv", index=False)
    stats[["dataset", "pair_id", "fragmented_all_query_rate", "fragmented_positive_opportunity_rate", "beneficial_components_mean_positive_only", "beneficial_components_max"]].to_csv(output_dir / "fragmentation_statistics.csv", index=False)
    consistency_frame.to_csv(output_dir / "direction_consistency.csv", index=False)
    action_frame.to_csv(output_dir / "action_level_summary.csv", index=False)

    landscape.PAIR_ORDER = PAIR_ORDER
    landscape.PAIR_DISPLAY = PAIR_DISPLAY
    landscape.PAIR_SHORT = PAIR_SHORT
    figure_paths = []
    figure_paths.extend(landscape.plot_global_oracle(stats, output_dir))
    figure_paths.extend(landscape.plot_action_heatmaps(geometries, output_dir, [PAIR_ORDER[0], PAIR_ORDER[2]]))
    figure_paths.extend(landscape.plot_gwd(geometries, output_dir))
    figure_paths.extend(landscape.plot_relation_direction_consistency(consistency_frame, output_dir, [("mkg_y", "MKG-Y")]))

    significant = (stats["available_headroom"] > 0) & (stats["headroom_ci95_low"] > 0)
    prevalent = stats["positive_opportunity_rate"] >= 0.25
    assessment = {
        "schema_version": 1,
        "outcome": "Y_E1_AVAILABLE_COMPLEMENTARITY_PRESENT" if significant.all() else "Y_E1_MIXED_AVAILABLE_COMPLEMENTARITY",
        "is_progression_gate": False,
        "significant_headroom_pairs": int(significant.sum()),
        "positive_opportunity_at_least_25pct_pairs": int(prevalent.sum()),
        "pair_results": [
            {
                "pair_id": row.pair_id,
                "available_headroom": row.available_headroom,
                "ci95_low": row.headroom_ci95_low,
                "ci95_high": row.headroom_ci95_high,
                "positive_opportunity_rate": row.positive_opportunity_rate,
            }
            for row in stats.itertuples()
        ],
        "next_stage": "Y_E2_PROTOCOL_FREEZE_ELIGIBLE",
        "test_status": "LOCKED",
    }
    assessment_path = output_dir / "replication_assessment.json"
    assessment_path.write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, stats, assessment, output_dir, source_records, args.support_min)

    generated = [
        output_dir / "per_query_action_geometry.csv.gz", output_dir / "pair_statistics.csv",
        output_dir / "pair_statistics.json", output_dir / "gwd_plateau_distribution_summary.csv",
        output_dir / "fragmentation_statistics.csv", output_dir / "direction_consistency.csv",
        output_dir / "action_level_summary.csv", assessment_path, *figure_paths, report_path,
    ]
    audit = {
        "schema_version": 1,
        "experiment": "MKG-Y Y-E1 Available Complementarity Landscape Audit",
        "date": "2026-09-07",
        "branch": "m1/recent-mmkgc-baselines",
        "split": "dev",
        "dataset": "mkg_y",
        "pairs": PAIR_ORDER,
        "alpha_grid": landscape.ALPHA_GRID.tolist(),
        "score_normalization": "query_zscore",
        "filter_fact_scope": "train_dev",
        "seeds": [1, 2, 3],
        "directions": ["head", "tail"],
        "bootstrap_unit": "original_triple_id",
        "bootstrap_samples": args.bootstrap_samples,
        "assessment": assessment,
        "operational_audit": {
            "test_access": 0, "checkpoint_evaluation": 0, "checkpoint_retraining": 0,
            "checkpoint_reselection": 0, "historical_result_modification": 0,
            "selector_training": 0, "policy_runs": 0, "y_e2_started": 0,
        },
        "sources": source_records,
        "outputs": [{"path": portable(path), "size_bytes": path.stat().st_size, "sha256": landscape.sha256_file(path)} for path in generated],
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"assessment": assessment, "pair_statistics": stats.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
