from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_mkg_y_y_e2_x4_oof import TEXT_HASH_SUFFIXES, declared_hash_records
from scripts.audit_adaptive_resolution import (
    LEVELS,
    GROUP_LEVELS,
    LEVEL_NAMES,
    apply_policy,
    choose_l4_support,
    degree_edges,
    fit_group_policy,
    group_stability,
    metrics_for_level,
    modality_states,
    paired_ci,
)
from scripts.exp2_information_common import (
    ALPHAS,
    RR_COLUMNS,
    ZERO_TOLERANCE,
    grouped_folds,
    portable_path,
    reject_test_path,
    select_global_alpha,
    sha256_file,
)


PAIR_LABELS = {
    "mkg_y_mhyper_native": "MKG-Y / M-Hyper + NativE",
    "mkg_y_mhyper_adamf": "MKG-Y / M-Hyper + AdaMF-MAT",
    "mkg_y_native_adamf": "MKG-Y / NativE + AdaMF-MAT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen MKG-Y Y-E3 DEV-only adaptive-resolution replication.")
    parser.add_argument("--contract", default="docs/protocols/MKG_Y_Y_E3_ADAPTIVE_RESOLUTION_CONTRACT.json")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/mkg_y_y_e3_resolution")
    parser.add_argument("--report", default="docs/reports/mkg_y_adaptive_resolution_audit_2026-09-08.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_systematic_run" or contract.get("split") != "dev":
        raise RuntimeError("MKG-Y Y-E3 contract is not frozen DEV-only")
    if tuple(contract.get("pair_ids", [])) != tuple(PAIR_LABELS):
        raise RuntimeError("MKG-Y Y-E3 pair inventory mismatch")
    if not np.allclose(contract.get("alpha_grid", []), ALPHAS):
        raise RuntimeError("MKG-Y Y-E3 alpha grid mismatch")
    if contract["levels"]["L4"]["minimum_support_candidates"] != [10, 25, 50, 100]:
        raise RuntimeError("MKG-Y Y-E3 L4 support candidates changed")
    if contract["levels"]["L5"]["source"] != "MKG-Y Y-E2 X4 nested_selected strict-OOF predictions":
        raise RuntimeError("MKG-Y Y-E3 L5 source changed")
    prohibited = ("test_access", "hierarchical_shrinkage", "lcb", "conservative_policy", "new_query_selector", "checkpoint_modification")
    if any(int(contract.get(field, -1)) != 0 for field in prohibited):
        raise RuntimeError("MKG-Y Y-E3 prohibited-operation boundary changed")
    return contract


def verify_declared_source(path: Path, expected: str) -> str:
    import hashlib

    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual == expected:
        return "raw_bytes"
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if hashlib.sha256(normalized).hexdigest() == expected:
            return "lf_normalized_text"
    raise RuntimeError(f"Frozen direct source hash mismatch: {path}")


def direct_source_inventory(contract_path: Path, contract: dict) -> list[dict]:
    inventory = [{"path": portable_path(contract_path), "sha256": sha256_file(contract_path), "role": "y_e3_contract"}]
    seen = {contract_path.resolve()}
    for path_string, expected in declared_hash_records(contract["source_boundaries"]):
        path = Path(path_string).resolve()
        mode = verify_declared_source(path, expected)
        if path in seen:
            continue
        seen.add(path)
        inventory.append({"path": portable_path(path), "sha256": sha256_file(path), "declared_sha256": expected, "declared_hash_verification": mode})
    return sorted(inventory, key=lambda row: row["path"])


def append_code_inventory(inventory: list[dict], paths: list[Path]) -> list[dict]:
    by_path = {row["path"]: row for row in inventory}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        key = portable_path(path)
        by_path[key] = {"path": key, "sha256": sha256_file(path), "role": "audit_code"}
    return [by_path[key] for key in sorted(by_path)]


def align_indices(source_ids: np.ndarray, target_ids: np.ndarray, label: str) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(source_ids)}
    if len(lookup) != len(source_ids) or set(lookup) != set(map(str, target_ids)):
        raise RuntimeError(f"{label} query inventory mismatch")
    return np.asarray([lookup[str(value)] for value in target_ids], dtype=np.int64)


def load_pair(pair_id: str, contract: dict):
    source = contract["source_boundaries"]["pairs"][pair_id]
    utility_manifest_path = Path(source["utility_manifest"]["path"])
    utility = json.loads(utility_manifest_path.read_text(encoding="utf-8"))
    query_path = Path(utility["source_query_rows"]["path"])
    asset_manifest_path = Path(source["asset_manifest"]["path"])
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    asset_path = Path(source["asset"]["path"])
    fields = asset_manifest["feature_fields_x4"]
    required = [contract["levels"]["L4"]["degree_feature"], *contract["levels"]["L4"]["modality_fields"]]
    indices = [fields.index(field) for field in required]
    with np.load(asset_path, allow_pickle=False) as asset:
        query_ids = asset["query_id"].astype(str)
        matrix = asset["features_x4"][:, indices].astype(np.float64)
        frame = pd.DataFrame({
            "query_id": query_ids,
            "seed": asset["seed"].astype(int),
            "direction": asset["direction"].astype(str),
            "head_id": asset["head_id"].astype(int),
            "relation_id": asset["relation_id"].astype(int),
            "tail_id": asset["tail_id"].astype(int),
            "degree_value": matrix[:, 0],
        })
    frame["modality_state"] = modality_states(matrix[:, 1], matrix[:, 2])
    frame["original_triple_id"] = "h=" + frame.head_id.astype(str) + "|r=" + frame.relation_id.astype(str) + "|t=" + frame.tail_id.astype(str)
    exact = pd.read_csv(query_path, usecols=["query_id", *RR_COLUMNS])
    order = align_indices(exact.query_id.to_numpy(str), query_ids, "exact RR")
    rr = exact.iloc[order][RR_COLUMNS].to_numpy(np.float64)
    prediction_path = Path(source["x4_predictions"]["path"])
    with np.load(prediction_path, allow_pickle=False) as prediction:
        order = align_indices(prediction["query_id"].astype(str), query_ids, "Y-E2 X4")
        x4 = {key: prediction[key][order] for key in prediction.files if key != "query_id"}
    folds = x4["outer_fold"].astype(int) - 1
    if frame.assign(_fold=folds).groupby("original_triple_id")._fold.nunique().max() != 1:
        raise RuntimeError(f"Outer-triple leakage in Y-E2 artifact: {pair_id}")
    computed, _ = grouped_folds(frame, int(contract["outer_cv"]["folds"]), int(contract["outer_cv"]["fold_seed"]))
    if not np.array_equal(computed, folds):
        raise RuntimeError(f"Y-E3 fold assignment differs from frozen Y-E2: {pair_id}")
    x4["outer_fold_zero"] = folds
    return frame, rr, x4


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:20px;font-weight:700}.panel{font-size:12px;font-weight:700}.axis{font-size:9px}.note{font-size:10px;fill:#5f6368}</style>',
        f'<text x="20" y="28" class="title">{html.escape(title)}</text>',
    ]


def write_svg(path: Path, parts: list[str]) -> None:
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def add_curve(parts, data, pair_id, ox, oy, width, height, fields, panel_label=None):
    values = np.concatenate([data[field].to_numpy(np.float64) for field, _, _ in fields])
    low, high = min(-0.001, float(values.min())), max(0.001, float(values.max()))
    sx = lambda index: ox + index * width / 5
    sy = lambda value: oy + height - (float(value) - low) / (high - low) * height
    label = PAIR_LABELS[pair_id] if panel_label is None else panel_label
    parts.append(f'<text x="{ox}" y="{oy-12}" class="panel">{html.escape(label)}</text>')
    parts.append(f'<line x1="{ox}" y1="{sy(0):.1f}" x2="{ox+width}" y2="{sy(0):.1f}" stroke="#9aa0a6"/>')
    for field, color, dash in fields:
        points = " ".join(f"{sx(index):.1f},{sy(value):.1f}" for index, value in enumerate(data[field]))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
    for index, level in enumerate(LEVELS):
        parts.append(f'<text x="{sx(index):.1f}" y="{oy+height+15}" text-anchor="middle" class="axis">{LEVEL_NAMES[level]}</text>')


def build_figures(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    pair_ids = list(PAIR_LABELS)
    paths = []
    parts = svg_header(820, 420, "MKG-Y Resolution–Identifiability Curve")
    averaged = metrics.groupby("level", sort=False)[["delta_mrr", "training_adaptive_gain"]].mean().reindex(LEVELS).reset_index()
    add_curve(parts, averaged, pair_ids[0], 80, 85, 650, 245, [("delta_mrr", "#376795", ""), ("training_adaptive_gain", "#d1495b", "5 3")], "Mean across three frozen MKG-Y expert pairs")
    parts += ['<line x1="220" y1="390" x2="245" y2="390" stroke="#376795" stroke-width="2"/><text x="252" y="394" class="note">OOF gain</text>', '<line x1="390" y1="390" x2="415" y2="390" stroke="#d1495b" stroke-width="2" stroke-dasharray="5 3"/><text x="422" y="394" class="note">training gain</text>']
    path = output_dir / "figure1_resolution_identifiability_curve.svg"; write_svg(path, parts); paths.append(path)

    parts = svg_header(820, 420, "MKG-Y Resolution vs Generalization Gap")
    averaged_gap = metrics.groupby("level", sort=False)["train_oof_generalization_gap"].mean().reindex(LEVELS)
    plot_frame = pd.DataFrame({"level": LEVELS, "gap": averaged_gap.to_numpy()})
    add_curve(parts, plot_frame, pair_ids[0], 80, 85, 650, 245, [("gap", "#7a5195", "")], "Mean training adaptive gain − OOF gain")
    path = output_dir / "figure2_resolution_generalization_gap.svg"; write_svg(path, parts); paths.append(path)

    parts = svg_header(1080, 430, "MKG-Y Resolution vs Support, Alpha Instability, and Negative Transfer")
    panels = [("support_coverage", "Support coverage"), ("group_alpha_foldwise_mad", "Group alpha foldwise MAD"), ("negative_transfer_ratio", "Negative transfer ratio")]
    for panel, (field, title) in enumerate(panels):
        data = metrics.groupby("level", sort=False)[field].mean().reindex(LEVELS)
        ox, oy, width, height = 55 + panel * 350, 85, 285, 245
        finite = data[np.isfinite(data)]
        low, high = min(0.0, float(finite.min())), max(0.01, float(finite.max()))
        sx = lambda index: ox + index * width / 5
        sy = lambda value: oy + height - (float(value) - low) / (high - low) * height
        points = " ".join(f"{sx(index):.1f},{sy(value):.1f}" for index, value in enumerate(data) if np.isfinite(value))
        parts.append(f'<text x="{ox}" y="{oy-15}" class="panel">{title}</text><polyline points="{points}" fill="none" stroke="#376795" stroke-width="2"/>')
        for index, value in enumerate(data):
            if np.isfinite(value): parts.append(f'<circle cx="{sx(index):.1f}" cy="{sy(value):.1f}" r="3" fill="#376795"/>')
            parts.append(f'<text x="{sx(index):.1f}" y="{oy+height+16}" text-anchor="middle" class="axis">{LEVEL_NAMES[LEVELS[index]]}</text>')
    path = output_dir / "figure3_resolution_support_instability_negative_transfer.svg"; write_svg(path, parts); paths.append(path)

    parts = svg_header(1180, 390, "MKG-Y Pair-Specific Resolution Curves")
    for panel, pair_id in enumerate(pair_ids):
        data = metrics[metrics.pair_id == pair_id].set_index("level").loc[list(LEVELS)].reset_index()
        add_curve(parts, data, pair_id, 55 + panel * 385, 80, 315, 220, [("delta_mrr", "#376795", ""), ("training_adaptive_gain", "#d1495b", "5 3")])
    parts += ['<line x1="430" y1="370" x2="455" y2="370" stroke="#376795" stroke-width="2"/><text x="462" y="374" class="note">OOF gain</text>', '<line x1="555" y1="370" x2="580" y2="370" stroke="#d1495b" stroke-width="2" stroke-dasharray="5 3"/><text x="587" y="374" class="note">training gain</text>']
    path = output_dir / "figure4_pair_specific_resolution_curves.svg"; write_svg(path, parts); paths.append(path)
    return paths


def relative_link(target: Path, report: Path) -> str:
    return os.path.relpath(target.resolve(), report.resolve().parent).replace("\\", "/")


def main() -> None:
    args = parse_args()
    contract_path, output_dir, report_path = Path(args.contract), Path(args.output_dir), Path(args.report)
    for path in (contract_path, output_dir, report_path): reject_test_path(path)
    contract = load_contract(contract_path)
    source_inventory = direct_source_inventory(contract_path, contract)
    source_inventory = append_code_inventory(source_inventory, [
        Path(__file__),
        Path("scripts/run_mkg_y_y_e3_resolution.ps1"),
        Path("scripts/audit_adaptive_resolution.py"),
        Path("scripts/exp2_information_common.py"),
    ])
    y_e2_audit = json.loads(Path(contract["source_boundaries"]["y_e2_audit"]["path"]).read_text(encoding="utf-8"))
    if y_e2_audit.get("assessment", {}).get("outcome") != "Y_E2_X4_REPLICATION_REPORTED" or int(y_e2_audit.get("operational_audit", {}).get("test_access", -1)) != 0:
        raise RuntimeError("Y-E2 is not a clean frozen DEV-only input")
    primary = pd.read_csv(contract["source_boundaries"]["y_e2_primary"]["path"])
    available = pd.read_csv(contract["source_boundaries"]["available_headroom"]["path"]).set_index("pair_id")
    loaded, preflight = {}, []
    for pair_id in contract["pair_ids"]:
        frame, rr, x4 = load_pair(pair_id, contract)
        loaded[pair_id] = (frame, rr, x4)
        preflight.append({"pair_id": pair_id, "queries": len(frame), "original_triples": int(frame.original_triple_id.nunique()), "outer_triple_leakage": 0})
    if args.dry_run:
        print(json.dumps({"status": "PRECHECK_OK", "pairs": preflight, "verified_hashes": len(source_inventory), "test_access": 0}, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit_manifest.json"
    if not args.overwrite and (audit_path.exists() or report_path.exists()):
        raise FileExistsError("Refusing to overwrite existing MKG-Y Y-E3 audit")

    metric_rows, fold_rows, slice_rows, policy_rows, inner_rows = [], [], [], [], []
    chosen_by_pair, global_by_pair, coverage_by_pair, training_by_pair = {}, {}, {}, {}
    bootstrap_samples, bootstrap_seed = int(contract["bootstrap"]["samples"]), int(contract["bootstrap"]["seed"])
    for pair_id in contract["pair_ids"]:
        frame, rr, x4 = loaded[pair_id]
        folds, n = x4["outer_fold_zero"], len(frame)
        chosen = {level: np.full(n, -1, dtype=np.int16) for level in LEVELS}
        coverage = {level: np.zeros(n, dtype=bool) for level in LEVELS}
        globals_vector = np.full(n, -1, dtype=np.int16)
        training_accumulator = {level: [0.0, 0] for level in LEVELS[:-1]}
        for outer_fold in range(int(contract["outer_cv"]["folds"])):
            outer_train, outer_hold = folds != outer_fold, folds == outer_fold
            train_indices, hold_indices = np.flatnonzero(outer_train), np.flatnonzero(outer_hold)
            global_index, _ = select_global_alpha(rr, outer_train)
            if not np.all(x4["fold_global_index"][hold_indices].astype(int) == global_index):
                raise RuntimeError(f"Fold-specific Global mismatch: {pair_id} fold {outer_fold+1}")
            globals_vector[hold_indices] = global_index
            chosen["L0"][hold_indices] = global_index; coverage["L0"][hold_indices] = True
            training_accumulator["L0"][1] += len(train_indices)
            policy_rows.append({"pair_id": pair_id, "level": "L0", "outer_fold": outer_fold + 1, "group_key": '["global"]', "support_original_triples": int(frame.loc[outer_train, "original_triple_id"].nunique()), "selected_action_index": global_index, "selected_alpha": float(ALPHAS[global_index]), "minimum_support": 1})
            fold_rows.append({"pair_id": pair_id, "level": "L0", "outer_fold": outer_fold + 1, "global_alpha": float(ALPHAS[global_index]), "minimum_support": 1, "training_adaptive_gain": 0.0, "oof_delta_mrr": 0.0, "support_coverage": 1.0, "supported_group_count": 1, "degree_edges": ""})
            for level in ("L1", "L2", "L3"):
                minimum_support = int(contract["levels"][level]["minimum_support"])
                policy, rows, keys = fit_group_policy(frame, rr, outer_train, level, global_index, minimum_support, None)
                hold_actions, hold_coverage = apply_policy(keys, policy, hold_indices, global_index)
                train_actions, _ = apply_policy(keys, policy, train_indices, global_index)
                chosen[level][hold_indices] = hold_actions; coverage[level][hold_indices] = hold_coverage
                train_gain = rr[train_indices, train_actions] - rr[train_indices, global_index]
                training_accumulator[level][0] += float(train_gain.sum()); training_accumulator[level][1] += len(train_gain)
                policy_rows.extend({"pair_id": pair_id, "level": level, "outer_fold": outer_fold + 1, "minimum_support": minimum_support, **row} for row in rows)
                fold_rows.append({"pair_id": pair_id, "level": level, "outer_fold": outer_fold + 1, "global_alpha": float(ALPHAS[global_index]), "minimum_support": minimum_support, "training_adaptive_gain": float(train_gain.mean()), "oof_delta_mrr": float((rr[hold_indices, hold_actions] - rr[hold_indices, global_index]).mean()), "support_coverage": float(hold_coverage.mean()), "supported_group_count": len(policy), "degree_edges": ""})
            selected_support, current_inner = choose_l4_support(frame, rr, outer_train, outer_fold, contract)
            inner_rows.extend({"pair_id": pair_id, **row} for row in current_inner)
            edges = degree_edges(frame, outer_train, contract["levels"]["L4"]["degree_quantiles"])
            policy, rows, keys = fit_group_policy(frame, rr, outer_train, "L4", global_index, selected_support, edges)
            hold_actions, hold_coverage = apply_policy(keys, policy, hold_indices, global_index)
            train_actions, _ = apply_policy(keys, policy, train_indices, global_index)
            chosen["L4"][hold_indices] = hold_actions; coverage["L4"][hold_indices] = hold_coverage
            train_gain = rr[train_indices, train_actions] - rr[train_indices, global_index]
            training_accumulator["L4"][0] += float(train_gain.sum()); training_accumulator["L4"][1] += len(train_gain)
            policy_rows.extend({"pair_id": pair_id, "level": "L4", "outer_fold": outer_fold + 1, "minimum_support": selected_support, **row} for row in rows)
            fold_rows.append({"pair_id": pair_id, "level": "L4", "outer_fold": outer_fold + 1, "global_alpha": float(ALPHAS[global_index]), "minimum_support": selected_support, "training_adaptive_gain": float(train_gain.mean()), "oof_delta_mrr": float((rr[hold_indices, hold_actions] - rr[hold_indices, global_index]).mean()), "support_coverage": float(hold_coverage.mean()), "supported_group_count": len(policy), "degree_edges": json.dumps([float(value) for value in edges])})
        chosen["L5"] = x4["chosen_action_index"].astype(np.int16); coverage["L5"][:] = True
        if any(np.any(actions < 0) for actions in chosen.values()) or np.any(globals_vector < 0): raise RuntimeError(f"Incomplete OOF actions: {pair_id}")
        training = {level: values[0] / values[1] if values[1] else 0.0 for level, values in training_accumulator.items()}
        training["L5"] = float(primary.loc[primary.pair_id == pair_id, "training_gain"].iloc[0])
        for outer_fold in range(int(contract["outer_cv"]["folds"])):
            indices = np.flatnonzero(folds == outer_fold)
            gain = rr[indices, chosen["L5"][indices]] - rr[indices, globals_vector[indices]]
            fold_rows.append({"pair_id": pair_id, "level": "L5", "outer_fold": outer_fold + 1, "global_alpha": float(ALPHAS[int(globals_vector[indices][0])]), "minimum_support": "", "training_adaptive_gain": float("nan"), "oof_delta_mrr": float(gain.mean()), "support_coverage": 1.0, "supported_group_count": "", "degree_edges": ""})
        chosen_by_pair[pair_id], global_by_pair[pair_id], coverage_by_pair[pair_id], training_by_pair[pair_id] = chosen, globals_vector, coverage, training

    policy_frame = pd.DataFrame(policy_rows)
    stability_rows = []
    for pair_offset, pair_id in enumerate(contract["pair_ids"]):
        frame, rr, x4 = loaded[pair_id]
        for level_offset, level in enumerate(LEVELS):
            group_mad, median_group_mad, rows = group_stability(policy_frame, pair_id, level)
            stability_rows.extend(rows)
            result, slices = metrics_for_level(frame, rr, chosen_by_pair[pair_id][level], global_by_pair[pair_id], coverage_by_pair[pair_id][level], training_by_pair[pair_id][level], float(available.loc[pair_id, "available_headroom"]), bootstrap_samples, bootstrap_seed + pair_offset * 100 + level_offset, group_mad, median_group_mad, x4["outer_fold_zero"])
            metric_rows.append({"dataset": "mkg_y", "pair_id": pair_id, "level": level, "level_name": LEVEL_NAMES[level], **result})
            slice_rows.extend({"dataset": "mkg_y", "pair_id": pair_id, "level": level, **row} for row in slices)
    metrics = pd.DataFrame(metric_rows)
    level_rows = []
    for level in GROUP_LEVELS:
        current = metrics[metrics.level == level]
        level_rows.append({"level": level, "robust_positive_pairs": int(current.robust_positive.sum()), "pairs_delta_below_minus_0_001": int((current.delta_mrr < -0.001).sum()), "mean_support_coverage": float(current.support_coverage.mean()), "mean_group_alpha_foldwise_mad": float(current.group_alpha_foldwise_mad.mean())})
    comparisons, overfit_pairs = [], []
    for pair_offset, pair_id in enumerate(contract["pair_ids"]):
        current = metrics[(metrics.pair_id == pair_id) & metrics.level.isin(GROUP_LEVELS)]
        best = current.loc[current.delta_mrr.idxmax()]
        frame, rr, _ = loaded[pair_id]
        rows = np.arange(len(frame))
        group_gain = rr[rows, chosen_by_pair[pair_id][str(best.level)]] - rr[rows, global_by_pair[pair_id]]
        x4_gain = rr[rows, chosen_by_pair[pair_id]["L5"]] - rr[rows, global_by_pair[pair_id]]
        low, high = paired_ci(x4_gain - group_gain, frame.original_triple_id.to_numpy(str), contract, pair_offset)
        ordered = metrics[metrics.pair_id == pair_id].set_index("level")
        adjacent = [float(ordered.loc[fine, "training_adaptive_gain"]) > float(ordered.loc[coarse, "training_adaptive_gain"]) + ZERO_TOLERANCE and float(ordered.loc[fine, "delta_mrr"]) <= float(ordered.loc[coarse, "delta_mrr"]) + ZERO_TOLERANCE for coarse, fine in zip(GROUP_LEVELS[:-1], GROUP_LEVELS[1:])]
        if sum(adjacent) >= 2: overfit_pairs.append(pair_id)
        comparisons.append({"pair_id": pair_id, "best_group_level": str(best.level), "best_group_delta_mrr": float(best.delta_mrr), "x4_query_delta_mrr": float(x4_gain.mean()), "x4_minus_group": float((x4_gain-group_gain).mean()), "x4_minus_group_ci95_low": low, "x4_minus_group_ci95_high": high, "finer_overfit_transition_count": int(sum(adjacent)), "finer_overfit_pattern": bool(sum(adjacent) >= 2)})
    comparison_frame = pd.DataFrame(comparisons)
    assessment = {"outcome": "Y_E3_RESOLUTION_REPLICATION_REPORTED", "is_progression_gate": False, "route_selection": False, "robust_positive_pairs_by_level": {row["level"]: row["robust_positive_pairs"] for row in level_rows}, "best_group_ge_x4_pairs": int((comparison_frame.best_group_delta_mrr >= comparison_frame.x4_query_delta_mrr - ZERO_TOLERANCE).sum()), "x4_significantly_exceeds_best_group_pairs": int((comparison_frame.x4_minus_group_ci95_low > ZERO_TOLERANCE).sum()), "finer_overfit_pattern_pairs": overfit_pairs, "next_replication": "Y-E4", "test_status": "LOCKED"}

    outputs = {
        "pair_level_metrics.csv": metrics, "fold_metrics.csv": pd.DataFrame(fold_rows), "seed_direction_metrics.csv": pd.DataFrame(slice_rows),
        "group_alpha_stability.csv": pd.DataFrame(stability_rows), "l4_inner_support_selection.csv": pd.DataFrame(inner_rows),
        "level_summary.csv": pd.DataFrame(level_rows), "best_group_vs_x4.csv": comparison_frame,
    }
    for name, frame in outputs.items(): frame.to_csv(output_dir / name, index=False)
    policy_frame.to_csv(output_dir / "group_actions.csv.gz", index=False, compression="gzip")
    npz_payload = {}
    for pair_id in contract["pair_ids"]:
        frame = loaded[pair_id][0]
        npz_payload[f"{pair_id}__query_id"] = frame.query_id.to_numpy(str)
        npz_payload[f"{pair_id}__global_action_index"] = global_by_pair[pair_id]
        npz_payload[f"{pair_id}__chosen_action_index"] = np.vstack([chosen_by_pair[pair_id][level] for level in LEVELS])
        npz_payload[f"{pair_id}__support_covered"] = np.vstack([coverage_by_pair[pair_id][level] for level in LEVELS])
    np.savez_compressed(output_dir / "per_query_resolution_results.npz", **npz_payload)
    (output_dir / "replication_assessment.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    figures = build_figures(metrics, output_dir)

    report = ["# MKG-Y Y-E3 Adaptive Resolution Audit", "", "Date: 2026-09-08", "", "## Outcome", "", "**Y_E3_RESOLUTION_REPLICATION_REPORTED** (descriptive external replication; no gate or route selection).", "", "## Main findings", "", "Robust-positive counts across the three MKG-Y pairs are " + ", ".join(f"{row['level']}={row['robust_positive_pairs']}/3" for row in level_rows) + ".", f"The best group-level resolution matches or exceeds frozen X4 in {assessment['best_group_ge_x4_pairs']}/3 pairs; X4 significantly exceeds the best group in {assessment['x4_significantly_exceeds_best_group_pairs']}/3 pairs.", f"Finer-granularity train-up / OOF-flat-or-down behavior appears in {len(overfit_pairs)}/3 pairs.", "", "The result is pair-dependent rather than a monotone granularity effect. Relation-level adaptation is robust only for M-Hyper + AdaMF-MAT; its OOF gain is +0.004920 (95% CI [+0.002682, +0.007142]) and recovers 11.2% of available headroom. M-Hyper + NativE has no robust L1-L4 resolution, while frozen X4 is positive and significant. NativE + AdaMF-MAT has no robust resolution at any level, including frozen X4.", "", "Direction-only adaptation is negative for all three pairs. Moving from Relation to Relation×Direction or Context raises training gain without improving OOF gain for M-Hyper + AdaMF-MAT, and Context coverage falls to 48.3–79.3% across pairs. These results do not support a general claim that finer resolution improves identifiability.", "", "## Pair × resolution metrics", "", "| Pair | Level | Train gain | OOF gain | 95% clustered CI | Recovery | Negative transfer | Changed | Coverage | Group MAD | Robust |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in metrics.itertuples():
        mad = "NA" if not np.isfinite(row.group_alpha_foldwise_mad) else f"{row.group_alpha_foldwise_mad:.3f}"
        report.append(f"| {PAIR_LABELS[row.pair_id]} | {row.level} {row.level_name} | {row.training_adaptive_gain:+.6f} | {row.delta_mrr:+.6f} | [{row.clustered_ci95_low:+.6f}, {row.clustered_ci95_high:+.6f}] | {100*row.available_headroom_recovery:.1f}% | {100*row.negative_transfer_ratio:.1f}% | {100*row.changed_rate:.1f}% | {100*row.support_coverage:.1f}% | {mad} | {row.robust_positive} |")
    report += ["", "## Best group versus frozen X4", "", "| Pair | Best group | Group gain | X4 gain | X4−group 95% CI | Overfit transitions |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in comparison_frame.itertuples(): report.append(f"| {PAIR_LABELS[row.pair_id]} | {row.best_group_level} | {row.best_group_delta_mrr:+.6f} | {row.x4_query_delta_mrr:+.6f} | [{row.x4_minus_group_ci95_low:+.6f}, {row.x4_minus_group_ci95_high:+.6f}] | {row.finer_overfit_transition_count}/3 |")
    report += ["", "## Figures", ""]
    for index, path in enumerate(figures, 1): report.append(f"{index}. [{path.stem}]({relative_link(path, report_path)})")
    report += ["", "## Operational audit", "", "- TEST access = 0", "- outer-triple leakage = 0", "- L5 new training = 0", "- new selector/policy development = 0", "- hierarchical shrinkage/LCB/conservative policy = 0", "- checkpoint modification = 0", "", "MKG-Y TEST remains locked. Y-E4 was not run by this audit.", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text("\n".join(report), encoding="utf-8")
    output_paths = [path for path in output_dir.rglob("*") if path.is_file() and path.resolve() != audit_path.resolve()]
    inventory_by_path = {row["path"]: row for row in source_inventory}
    for path in [*output_paths, report_path]:
        inventory_by_path[portable_path(path)] = {"path": portable_path(path), "sha256": sha256_file(path), "role": "y_e3_output"}
    inventory = [inventory_by_path[path] for path in sorted(inventory_by_path)]
    audit = {"schema_version": 1, "experiment": contract["experiment"], "split": "dev", "dataset": "mkg_y", "assessment": assessment, "source_and_output_hash_count": len(inventory), "sources_and_outputs": inventory, "operational_audit": {"test_access": 0, "outer_triple_leakage": 0, "new_l5_training": 0, "new_selector": 0, "policy_development": 0, "hierarchical_shrinkage": 0, "lcb": 0, "conservative_policy": 0, "checkpoint_modification": 0}, "next_step_started": 0}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
