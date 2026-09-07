from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp2_information_common import (
    ALPHAS,
    RR_COLUMNS,
    load_contract,
    policy_metrics,
    portable_path,
    reject_test_path,
    sha256_file,
    sign_metrics,
)


PAIR_LABELS = {
    "mkg_y_mhyper_native": "MKG-Y / M-Hyper + NativE",
    "mkg_y_mhyper_adamf": "MKG-Y / M-Hyper + AdaMF-MAT",
    "mkg_y_native_adamf": "MKG-Y / NativE + AdaMF-MAT",
}

TEXT_HASH_SUFFIXES = {
    ".bib", ".csv", ".json", ".md", ".ps1", ".py", ".svg", ".tex", ".tsv", ".txt", ".yaml", ".yml"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the frozen MKG-Y Y-E2 X4 strict nested-OOF replication.")
    parser.add_argument("--contract", default="docs/protocols/MKG_Y_Y_E2_X4_OOF_CONTRACT.json")
    parser.add_argument("--run-root", default="outputs/complementarity_identifiability/mkg_y_y_e2_information/runs")
    parser.add_argument("--asset-dir", default="outputs/complementarity_identifiability/mkg_y_y_e2_information/assets")
    parser.add_argument("--utility-manifest-dir", default="outputs/complementarity_identifiability/mkg_y_y_e2_information/utility_tables")
    parser.add_argument("--exp1-stats", default="outputs/complementarity_identifiability/mkg_y_y_e1_landscape/pair_statistics.csv")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/mkg_y_y_e2_information")
    parser.add_argument("--report", default="docs/reports/mkg_y_information_identifiability_audit_2026-09-08.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def relative_link(target: Path, document: Path) -> str:
    return os.path.relpath(target.resolve(), document.resolve().parent).replace("\\", "/")


def declared_hash_records(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value["path"], value["sha256"]
        for field in ("source_files", "files", "source_and_output_hashes"):
            mapping = value.get(field)
            if isinstance(mapping, dict):
                for path, digest in mapping.items():
                    if isinstance(path, str) and isinstance(digest, str):
                        yield path, digest
        accepted = value.get("accepted_runs")
        if isinstance(accepted, dict):
            for runs in accepted.values():
                if not isinstance(runs, dict):
                    continue
                for record in runs.values():
                    if isinstance(record, dict) and isinstance(record.get("run_dir"), str) and isinstance(record.get("checkpoint_sha256"), str):
                        yield str(Path(record["run_dir"]) / "best.ckpt"), record["checkpoint_sha256"]
        for key, nested in value.items():
            if key not in {"source_files", "files", "source_and_output_hashes", "accepted_runs"}:
                yield from declared_hash_records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from declared_hash_records(nested)


def complete_hash_inventory(seed_paths: list[Path], excluded: set[Path]) -> list[dict]:
    records: dict[str, str] = {}
    verification_modes: dict[str, str] = {}
    queue = [Path(path) for path in seed_paths]
    parsed_json: set[Path] = set()
    excluded_resolved = {path.resolve() for path in excluded}
    while queue:
        path = queue.pop().resolve()
        if path in excluded_resolved or "preflight" in {part.lower() for part in path.parts}:
            continue
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Hash inventory source is missing: {path}")
        actual = sha256_file(path)
        portable = portable_path(path)
        previous = records.get(portable)
        if previous is not None and previous != actual:
            raise RuntimeError(f"Inconsistent hash inventory for {portable}")
        records[portable] = actual
        if path.suffix.lower() != ".json" or path in parsed_json:
            continue
        parsed_json.add(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for declared_path, expected in declared_hash_records(payload):
            source = Path(declared_path).resolve()
            if source in excluded_resolved:
                continue
            if not source.exists():
                raise RuntimeError(f"Declared source hash mismatch: {declared_path}")
            actual = sha256_file(source)
            if actual == expected:
                verification_modes.setdefault(portable_path(source), "raw_bytes")
            elif source.suffix.lower() in TEXT_HASH_SUFFIXES:
                normalized = source.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                normalized_hash = hashlib.sha256(normalized).hexdigest()
                if normalized_hash != expected:
                    raise RuntimeError(f"Declared source hash mismatch: {declared_path}")
                verification_modes[portable_path(source)] = "lf_normalized_text"
            else:
                raise RuntimeError(f"Declared source hash mismatch: {declared_path}")
            queue.append(source)
    return [
        {
            "path": path,
            "sha256": records[path],
            **({"declared_hash_verification": verification_modes[path]} if path in verification_modes else {}),
        }
        for path in sorted(records)
    ]


def load_pair_data(pair_id: str, asset_dir: Path, utility_dir: Path):
    asset_manifest_path = asset_dir / f"{pair_id}_query_information_manifest.json"
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    asset_path = Path(asset_manifest["output"]["path"])
    if sha256_file(asset_path) != asset_manifest["output"]["sha256"]:
        raise RuntimeError(f"Asset hash mismatch for {pair_id}")
    with np.load(asset_path, allow_pickle=False) as asset:
        query_ids = asset["query_id"].astype(str)
        frame = pd.DataFrame({
            "query_id": query_ids,
            "seed": asset["seed"].astype(int),
            "direction": asset["direction"].astype(str),
            "head_id": asset["head_id"].astype(int),
            "relation_id": asset["relation_id"].astype(int),
            "tail_id": asset["tail_id"].astype(int),
        })
    frame["original_triple_id"] = "h=" + frame.head_id.astype(str) + "|r=" + frame.relation_id.astype(str) + "|t=" + frame.tail_id.astype(str)
    utility_manifest_path = utility_dir / f"{pair_id}_dev_source_manifest.json"
    utility_manifest = json.loads(utility_manifest_path.read_text(encoding="utf-8"))
    query_path = Path(utility_manifest["source_query_rows"]["path"])
    if sha256_file(query_path) != utility_manifest["source_query_rows"]["sha256"]:
        raise RuntimeError(f"Exact RR source hash mismatch for {pair_id}")
    exact = pd.read_csv(query_path, usecols=["query_id", *RR_COLUMNS]).set_index("query_id")
    if set(exact.index.astype(str)) != set(query_ids):
        raise RuntimeError(f"Exact RR query inventory mismatch for {pair_id}")
    rr = exact.loc[query_ids, RR_COLUMNS].to_numpy(np.float64)
    return frame, rr, asset_manifest_path, asset_path, utility_manifest_path, query_path


def nested_select(pair_id: str, learners: list[str], args, contract: dict, available: float):
    frame, rr, asset_manifest_path, asset_path, utility_manifest_path, query_path = load_pair_data(
        pair_id, Path(args.asset_dir), Path(args.utility_manifest_dir)
    )
    learner_data = {}
    direct_sources = [asset_manifest_path, asset_path, utility_manifest_path, query_path]
    for learner in learners:
        root = Path(args.run_root) / pair_id / "x4" / learner
        required = [
            root / "metrics.json", root / "oof_action_predictions.npz", root / "outer_fold_selections.csv",
            root / "training_fold_metrics.csv", root / "inner_cv_results.csv", root / "seed_direction_metrics.csv",
            root / "input_audit.json",
        ]
        if any(not path.exists() for path in required):
            raise FileNotFoundError(f"Incomplete Y-E2 systematic run under {root}")
        direct_sources.extend(required)
        payload = json.loads(required[0].read_text(encoding="utf-8"))
        if payload.get("contract_sha256") != sha256_file(Path(args.contract)):
            raise RuntimeError(f"Run was built under a different contract: {root}")
        with np.load(required[1], allow_pickle=False) as prediction:
            if list(prediction["query_id"].astype(str)) != list(frame.query_id.astype(str)):
                raise RuntimeError(f"Prediction alignment mismatch: {root}")
            learner_data[learner] = {
                "payload": payload,
                "predicted": prediction["predicted_u"].astype(np.float64),
                "fold": prediction["outer_fold"].astype(int),
                "global": prediction["fold_global_index"].astype(int),
                "selection": pd.read_csv(required[2]),
                "training": pd.read_csv(required[3]),
            }
    fold_reference = next(iter(learner_data.values()))["fold"]
    global_reference = next(iter(learner_data.values()))["global"]
    predicted = np.empty_like(next(iter(learner_data.values()))["predicted"])
    learner_order = {learner: index for index, learner in enumerate(learners)}
    selection_rows, train_gains, train_counts = [], [], []
    outer_folds = int(contract["nested_cv"]["outer_folds"])
    for fold in range(1, outer_folds + 1):
        candidates = []
        for learner in learners:
            data = learner_data[learner]
            if not np.array_equal(data["fold"], fold_reference) or not np.array_equal(data["global"], global_reference):
                raise RuntimeError("Learner OOF fold/global assignments differ")
            row = data["selection"].loc[lambda value: value.outer_fold == fold].iloc[0]
            candidates.append((float(row.inner_probe_delta_mrr), float(row.inner_spearman), -learner_order[learner], learner, row))
        winner = max(candidates)
        learner, row = winner[3], winner[4]
        mask = fold_reference == fold
        predicted[mask] = learner_data[learner]["predicted"][mask]
        train_row = learner_data[learner]["training"].loc[lambda value: value.outer_fold == fold].iloc[0]
        train_gains.append(float(train_row.training_gain))
        train_counts.append(int(train_row["count"]))
        selection_rows.append({
            "dataset": "mkg_y", "pair_id": pair_id, "representation": "X4", "outer_fold": fold,
            "selected_learner": learner, "selected_config": row.selected_config,
            "inner_probe_delta_mrr": row.inner_probe_delta_mrr, "inner_spearman": row.inner_spearman,
            "fold_global_alpha": row.fold_global_alpha,
        })
    bootstrap = contract["bootstrap"]
    metrics, chosen = policy_metrics(
        rr, predicted, global_reference, frame.original_triple_id.to_numpy(str), available,
        int(bootstrap["samples"]), int(bootstrap["seed"]),
    )
    actual_u = rr - rr[np.arange(len(rr)), global_reference][:, None]
    non_global = np.arange(len(ALPHAS))[None, :] != global_reference[:, None]
    metrics.update(sign_metrics(actual_u[non_global], predicted[non_global]))
    training_gain = float(np.average(train_gains, weights=train_counts))
    metrics.update({"training_gain": training_gain, "oof_gain": metrics["delta_mrr"], "train_oof_generalization_gap": training_gain - metrics["delta_mrr"]})

    root = Path(args.run_root) / pair_id / "x4" / "nested_selected"
    root.mkdir(parents=True, exist_ok=True)
    prediction_path = root / "oof_action_predictions.npz"
    selection_path = root / "outer_fold_learner_selections.csv"
    slice_path = root / "seed_direction_metrics.csv"
    metrics_path = root / "metrics.json"
    np.savez_compressed(
        prediction_path, query_id=frame.query_id.to_numpy(str), outer_fold=fold_reference,
        fold_global_index=global_reference, predicted_u=predicted.astype(np.float32), chosen_action_index=chosen,
    )
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    slice_rows = []
    scopes = [("direction", value, frame.direction == value) for value in ("head", "tail")]
    scopes += [("seed", str(seed), frame.seed == seed) for seed in (1, 2, 3)]
    scopes += [
        ("seed_x_direction", f"{seed}_{direction}", (frame.seed == seed) & (frame.direction == direction))
        for seed in (1, 2, 3) for direction in ("head", "tail")
    ]
    for offset, (scope, value, mask) in enumerate(scopes):
        result, _ = policy_metrics(
            rr[mask], predicted[mask], global_reference[mask], frame.loc[mask, "original_triple_id"].to_numpy(str),
            available, int(bootstrap["samples"]), int(bootstrap["seed"]) + offset + 101,
        )
        slice_actual = actual_u[mask]
        slice_non_global = np.arange(len(ALPHAS))[None, :] != global_reference[mask, None]
        result.update(sign_metrics(slice_actual[slice_non_global], predicted[mask][slice_non_global]))
        slice_rows.append({"dataset": "mkg_y", "pair_id": pair_id, "representation": "X4", "learner": "nested_selected", "scope": scope, "value": value, **result})
    pd.DataFrame(slice_rows).to_csv(slice_path, index=False)
    nested_payload = {
        "schema_version": 1, "experiment": contract["experiment"], "probe": "nested_selected",
        "dataset": "mkg_y", "pair_id": pair_id, "representation": "X4", "metrics": metrics,
        "contract_sha256": sha256_file(Path(args.contract)), "test_access": 0, "final_policy_development": 0,
    }
    metrics_path.write_text(json.dumps(nested_payload, indent=2) + "\n", encoding="utf-8")
    direct_sources.extend([prediction_path, selection_path, slice_path, metrics_path])
    row = {"dataset": "mkg_y", "pair_id": pair_id, "representation": "X4", "learner": "nested_selected", **metrics}
    return row, selection_rows, direct_sources


def write_figures(all_metrics: pd.DataFrame, primary: pd.DataFrame, available: pd.DataFrame, output_dir: Path) -> list[Path]:
    colors = {"linear_huber": "#4c78a8", "hist_gbdt": "#f58518", "mlp_low": "#54a24b", "mlp_high": "#e45756", "nested_selected": "#6f4e7c"}
    figures = []
    width, height = 1000, 590
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.label{font-size:11px}.panel{font-size:13px;font-weight:700}</style>', '<text x="24" y="32" class="title">MKG-Y Frozen X4 Strict-OOF Replication</text>']
    values = np.r_[all_metrics.delta_mrr.to_numpy(), all_metrics.training_gain.to_numpy(), primary.delta_mrr.to_numpy()]
    low, high = min(-0.003, float(values.min())), max(0.003, float(values.max()))
    sy = lambda value: 500 - (float(value) - low) / (high - low) * 400
    parts.append(f'<line x1="80" y1="{sy(0):.1f}" x2="950" y2="{sy(0):.1f}" stroke="#999"/>')
    learners = list(colors)[:-1]
    for pair_index, pair_id in enumerate(PAIR_LABELS):
        center = 190 + pair_index * 285
        parts.append(f'<text x="{center}" y="545" text-anchor="middle" class="panel">{html.escape(PAIR_LABELS[pair_id].replace("MKG-Y / ", ""))}</text>')
        for learner_index, learner in enumerate(learners):
            row = all_metrics[(all_metrics.pair_id == pair_id) & (all_metrics.learner == learner)].iloc[0]
            x = center - 75 + learner_index * 45
            parts.append(f'<line x1="{x}" y1="{sy(row.delta_mrr):.1f}" x2="{x}" y2="{sy(row.training_gain):.1f}" stroke="{colors[learner]}" stroke-dasharray="4 3"/>')
            parts.append(f'<circle cx="{x}" cy="{sy(row.delta_mrr):.1f}" r="5" fill="{colors[learner]}"/>')
            parts.append(f'<circle cx="{x}" cy="{sy(row.training_gain):.1f}" r="4" fill="white" stroke="{colors[learner]}" stroke-width="2"/>')
        selected = primary[primary.pair_id == pair_id].iloc[0]
        x = center + 110
        parts.append(f'<circle cx="{x}" cy="{sy(selected.delta_mrr):.1f}" r="7" fill="{colors["nested_selected"]}"/>')
    parts += ['<text x="25" y="300" transform="rotate(-90 25 300)" text-anchor="middle" class="panel">MRR gain vs fold-specific Global</text>', '<text x="90" y="575" class="label">filled = OOF; open = training; purple = fold-wise nested-selected primary</text>', '</svg>']
    path = output_dir / "figure_y_e2_x4_learner_comparison.svg"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    figures.append(path)

    merged = primary.merge(available[["pair_id", "available_headroom"]], on="pair_id")
    maximum = max(float(merged.available_headroom.max()), float(merged.delta_mrr.max()), 0.001)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="600" viewBox="0 0 760 600">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.label{font-size:11px}</style>', '<text x="24" y="32" class="title">Available vs Empirically Identifiable Headroom — MKG-Y</text>']
    sx = lambda value: 90 + float(value) / maximum * 580
    sy2 = lambda value: 510 - float(value) / maximum * 420
    parts.append(f'<line x1="90" y1="510" x2="670" y2="90" stroke="#aaa" stroke-dasharray="5 4"/>')
    for index, row in enumerate(merged.itertuples()):
        color = ("#4c78a8", "#f58518", "#54a24b")[index]
        parts.append(f'<circle cx="{sx(row.available_headroom):.1f}" cy="{sy2(row.delta_mrr):.1f}" r="7" fill="{color}"/>')
        parts.append(f'<text x="{sx(row.available_headroom)+10:.1f}" y="{sy2(row.delta_mrr)-8:.1f}" class="label">{html.escape(PAIR_LABELS[row.pair_id].replace("MKG-Y / ", ""))}</text>')
    parts += ['<text x="380" y="570" text-anchor="middle" class="label">Y-E1 Available Headroom</text>', '<text x="25" y="300" transform="rotate(-90 25 300)" text-anchor="middle" class="label">Y-E2 X4 OOF gain</text>', '</svg>']
    path = output_dir / "figure_y_e2_available_vs_identifiable.svg"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    figures.append(path)
    return figures


def main() -> None:
    args = parse_args()
    contract_path, output_dir, report_path = Path(args.contract), Path(args.output_dir), Path(args.report)
    for path in (contract_path, output_dir, report_path):
        reject_test_path(path)
    contract = load_contract(contract_path)
    if contract.get("protocol_profile") != "mkg_y_frozen_x4_replication":
        raise ValueError("This analyzer accepts only the frozen MKG-Y X4 replication")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit_manifest.json"
    if not args.overwrite and (audit_path.exists() or report_path.exists()):
        raise FileExistsError("Refusing to overwrite an existing Y-E2 audit")
    available_path = Path(args.exp1_stats)
    boundary = contract["source_boundaries"]["available_headroom"]
    if portable_path(available_path) != boundary["path"] or sha256_file(available_path) != boundary["sha256"]:
        raise RuntimeError("Y-E1 available-headroom source differs from the frozen contract")
    available = pd.read_csv(available_path)
    pair_ids = list(contract["pair_ids"])
    learners = list(contract["learner_compatibility"]["X4"])
    all_rows, primary_rows, selection_rows, inventory_paths = [], [], [], [contract_path, available_path]
    for pair_id in pair_ids:
        for learner in learners:
            metrics_path = Path(args.run_root) / pair_id / "x4" / learner / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            all_rows.append({"dataset": "mkg_y", "pair_id": pair_id, "representation": "X4", "learner": learner, **payload["metrics"]})
        headroom = float(available.loc[available.pair_id == pair_id, "available_headroom"].iloc[0])
        primary, selections, sources = nested_select(pair_id, learners, args, contract, headroom)
        primary_rows.append(primary)
        selection_rows.extend(selections)
        inventory_paths.extend(sources)
    all_metrics, primary = pd.DataFrame(all_rows), pd.DataFrame(primary_rows)
    metrics_table = output_dir / "metrics_by_learner.csv"
    primary_table = output_dir / "primary_nested_x4_probe_metrics.csv"
    selections_table = output_dir / "nested_learner_selections.csv"
    all_metrics.to_csv(metrics_table, index=False)
    primary.to_csv(primary_table, index=False)
    pd.DataFrame(selection_rows).to_csv(selections_table, index=False)
    figure_paths = write_figures(all_metrics, primary, available, output_dir)

    summary = []
    for row in primary.itertuples():
        seed_rows = pd.read_csv(Path(args.run_root) / row.pair_id / "x4" / "nested_selected" / "seed_direction_metrics.csv")
        seed_positive = int((seed_rows.loc[seed_rows.scope == "seed", "delta_mrr"] > 0).sum())
        summary.append({
            "pair_id": row.pair_id,
            "robust_positive": bool(row.delta_mrr > 0 and row.clustered_ci95_low > 0),
            "recovery_ge_10pct": bool(row.headroom_recovery >= 0.10),
            "positive_seeds": seed_positive,
        })
    assessment = {
        "outcome": "Y_E2_X4_REPLICATION_REPORTED",
        "is_progression_gate": False,
        "robust_positive_pairs": int(sum(row["robust_positive"] for row in summary)),
        "recovery_ge_10pct_pairs": int(sum(row["recovery_ge_10pct"] for row in summary)),
        "pair_diagnostics": summary,
        "x6_run": False,
        "method_development_started": False,
    }
    assessment_path = output_dir / "replication_assessment.json"
    assessment_path.write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# MKG-Y Y-E2 Frozen X4 Information–Identifiability Audit", "", "Date: 2026-09-08", "",
        "## Outcome", "", "**Y_E2_X4_REPLICATION_REPORTED** (descriptive replication; not a progression gate).", "",
        "This is the frozen finite-model strict-OOF identifiability probe. It is reported as Empirical Identifiable Headroom, not theoretical identifiability and not a final deployable policy.", "",
        "## Frozen protocol", "",
        "- DEV only; exact filtered RR on alpha `0.00:0.05:1.00`; query-zscore score normalization.",
        "- Five outer folds and three inner folds, grouped by original triple.",
        "- Each held-out fold uses an alpha0 selected only from its outer-training groups.",
        "- X4 is unchanged from Experiment 2: score geometry, cross-expert disagreement, TRAIN-only structural context, and TRAIN-only modality context.",
        "- Four frozen learners/config grids are compared inside each outer fold; learner selection never sees that fold's held-out queries.",
        "- X6 was not run because the pre-specified conditional justification was absent.", "",
        "## Primary fold-wise nested-selected X4 probe", "",
        "| Pair | OOF Global MRR | OOF MRR | OOF gain | 95% clustered CI | Available recovery | Positive gain | Negative transfer | Changed | Train gain | Train–OOF gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary.itertuples():
        report_lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {row.fold_specific_global_mrr:.6f} | {row.oof_mrr:.6f} | {row.delta_mrr:+.6f} | "
            f"[{row.clustered_ci95_low:+.6f}, {row.clustered_ci95_high:+.6f}] | {100*row.headroom_recovery:.1f}% | "
            f"{100*row.positive_gain_rate:.1f}% | {100*row.negative_transfer_rate:.1f}% | {100*row.changed_rate:.1f}% | "
            f"{row.training_gain:+.6f} | {row.train_oof_generalization_gap:+.6f} |"
        )
    report_lines += ["", "## Utility-identifiability diagnostics", "", "| Pair | Spearman(pred U, U) | Positive AP lift | Harmful AP lift | Positive-vs-harmful AUROC |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in primary.itertuples():
        report_lines.append(f"| {PAIR_LABELS[row.pair_id]} | {row.spearman_pred_u_actual_u:+.4f} | {row.positive_ap_lift:+.4f} | {row.harmful_ap_lift:+.4f} | {row.positive_vs_harmful_auroc:.4f} |")
    report_lines += ["", "## Descriptive replication summary", "", f"- Robust-positive pairs (OOF gain > 0 and clustered CI lower > 0): {assessment['robust_positive_pairs']}/3.", f"- Pairs recovering at least 10% of Y-E1 available headroom: {assessment['recovery_ge_10pct_pairs']}/3.", "- No GO/NO-GO gate is applied at Y-E2, and no selector or representation was developed.", "", "## Figures", ""]
    for index, path in enumerate(figure_paths, 1):
        report_lines.append(f"{index}. [{path.stem}]({relative_link(path, report_path)})")
    report_lines += ["", "## Operational audit", "", "- TEST access = 0", "- outer-triple leakage = 0", "- full-DEV alpha0 used for held-out fold = 0", "- checkpoint retraining/reselection = 0", "- new selector/representation = 0", "- X6 run = 0", "- final policy development = 0", "", "All direct source and output hashes are recorded in `audit_manifest.json`.", ""]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    inventory_paths.extend([metrics_table, primary_table, selections_table, assessment_path, *figure_paths, report_path])
    inventory = complete_hash_inventory(inventory_paths, {audit_path})
    audit = {
        "schema_version": 1, "experiment": contract["experiment"], "split": "dev", "dataset": "mkg_y",
        "assessment": assessment, "source_and_output_hash_count": len(inventory), "sources_and_outputs": inventory,
        "operational_audit": {
            "test_access": 0, "outer_triple_leakage": 0, "full_dev_global_for_heldout": 0,
            "checkpoint_training": 0, "checkpoint_reselection": 0, "new_selector": 0,
            "new_representation": 0, "x6_run": 0, "final_policy_development": 0,
        },
        "next_step_started": 0,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
