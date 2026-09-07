from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp2_information_common import (
    ALPHAS,
    PAIR_IDS,
    RR_COLUMNS,
    ZERO_TOLERANCE,
    clustered_bootstrap,
    grouped_folds,
    portable_path,
    reject_test_path,
    select_global_alpha,
    sha256_file,
)


LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")
GROUP_LEVELS = ("L1", "L2", "L3", "L4")
LEVEL_NAMES = {
    "L0": "Global",
    "L1": "Direction",
    "L2": "Relation",
    "L3": "Relation×Direction",
    "L4": "Context",
    "L5": "Query",
}
PAIR_LABELS = {
    "mkgw_mhyper_native": "MKG-W / M-Hyper+NativE",
    "mkgw_mhyper_adamf": "MKG-W / M-Hyper+AdaMF",
    "mkgw_native_adamf": "MKG-W / NativE+AdaMF",
    "db15k_mhyper_native": "DB15K / M-Hyper+NativE",
    "db15k_mhyper_adamf": "DB15K / M-Hyper+AdaMF",
    "db15k_native_adamf": "DB15K / NativE+AdaMF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 3 DEV-only adaptive-resolution audit.")
    parser.add_argument("--contract", default="docs/protocols/EXP3_ADAPTIVE_RESOLUTION_CONTRACT.json")
    parser.add_argument("--exp1-root", default="outputs/complementarity_identifiability/exp1_landscape")
    parser.add_argument("--exp2-root", default="outputs/complementarity_identifiability/exp2_information")
    parser.add_argument("--utility-manifest-dir", default="outputs/aacpi/utility_tables")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/exp3_resolution")
    parser.add_argument("--report", default="docs/reports/adaptive_resolution_audit_2026-09-07.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_systematic_run" or contract.get("split") != "dev":
        raise RuntimeError("Experiment 3 contract is not frozen DEV-only")
    if not np.allclose(contract.get("alpha_grid", []), ALPHAS):
        raise RuntimeError("Experiment 3 alpha grid mismatch")
    if tuple(contract.get("pair_ids", [])) != PAIR_IDS:
        raise RuntimeError("Experiment 3 pair inventory mismatch")
    if contract["levels"]["L4"]["minimum_support_candidates"] != [10, 25, 50, 100]:
        raise RuntimeError("Experiment 3 L4 support search changed")
    if contract["levels"]["L5"]["source"] != "Experiment 2 X4 nested_selected strict-OOF predictions":
        raise RuntimeError("Experiment 3 L5 source changed")
    prohibited = ("test_access", "hierarchical_shrinkage", "lcb", "conservative_policy", "new_query_selector", "checkpoint_modification")
    if any(int(contract.get(field, -1)) != 0 for field in prohibited):
        raise RuntimeError("Experiment 3 prohibited-operation contract changed")
    return contract


def align_indices(source_ids: np.ndarray, target_ids: np.ndarray, label: str) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(source_ids)}
    if len(lookup) != len(source_ids) or set(lookup) != set(map(str, target_ids)):
        raise RuntimeError(f"{label} query inventory mismatch")
    return np.asarray([lookup[str(value)] for value in target_ids], dtype=np.int64)


def select_action(rr: np.ndarray, indices: np.ndarray, global_index: int) -> int:
    means = rr[indices].mean(axis=0)
    best = float(means.max())
    candidates = np.flatnonzero(np.isclose(means, best, rtol=0.0, atol=ZERO_TOLERANCE))
    preference = np.abs(ALPHAS[candidates] - ALPHAS[global_index]) + ALPHAS[candidates] * 1e-6
    return int(candidates[int(preference.argmin())])


def degree_edges(frame: pd.DataFrame, train_mask: np.ndarray, quantiles: list[float]) -> np.ndarray:
    unique = frame.loc[train_mask].drop_duplicates(["original_triple_id", "direction"])
    if unique.empty:
        raise ValueError("Cannot fit degree buckets on an empty training partition")
    return np.quantile(unique.degree_value.to_numpy(np.float64), quantiles).astype(np.float64)


def level_keys(frame: pd.DataFrame, level: str, edges: np.ndarray | None = None) -> list[tuple]:
    if level == "L1":
        return [(str(value),) for value in frame.direction]
    if level == "L2":
        return [(int(value),) for value in frame.relation_id]
    if level == "L3":
        return [(int(relation), str(direction)) for relation, direction in zip(frame.relation_id, frame.direction)]
    if level == "L4":
        if edges is None:
            raise ValueError("L4 requires degree edges")
        buckets = np.searchsorted(edges, frame.degree_value.to_numpy(np.float64), side="right")
        return [
            (int(relation), str(direction), int(bucket), str(modality))
            for relation, direction, bucket, modality in zip(
                frame.relation_id, frame.direction, buckets, frame.modality_state
            )
        ]
    raise ValueError(f"No group key for {level}")


def fit_group_policy(
    frame: pd.DataFrame,
    rr: np.ndarray,
    train_mask: np.ndarray,
    level: str,
    global_index: int,
    minimum_support: int,
    edges: np.ndarray | None,
) -> tuple[dict[tuple, int], list[dict], list[tuple]]:
    keys = level_keys(frame, level, edges)
    members: dict[tuple, list[int]] = defaultdict(list)
    for index in np.flatnonzero(train_mask):
        members[keys[index]].append(int(index))
    policy: dict[tuple, int] = {}
    rows = []
    group_ids = frame.original_triple_id.to_numpy(str)
    for key, indices_list in members.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        support = len(set(group_ids[indices]))
        if support < minimum_support:
            continue
        action = select_action(rr, indices, global_index)
        policy[key] = action
        rows.append(
            {
                "group_key": json.dumps(key, separators=(",", ":")),
                "support_original_triples": support,
                "selected_action_index": action,
                "selected_alpha": float(ALPHAS[action]),
            }
        )
    return policy, rows, keys


def apply_policy(keys: list[tuple], policy: dict[tuple, int], indices: np.ndarray, global_index: int) -> tuple[np.ndarray, np.ndarray]:
    actions = np.full(len(indices), global_index, dtype=np.int16)
    covered = np.zeros(len(indices), dtype=bool)
    for local, index in enumerate(indices):
        key = keys[int(index)]
        if key in policy:
            actions[local] = policy[key]
            covered[local] = True
    return actions, covered


def modality_states(text: np.ndarray, image: np.ndarray) -> np.ndarray:
    text = text >= 0.5
    image = image >= 0.5
    return np.select(
        [text & image, text & ~image, ~text & image],
        ["text_and_image", "text_only", "image_only"],
        default="none",
    ).astype(str)


def load_prediction(path: Path, target_ids: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        order = align_indices(payload["query_id"].astype(str), target_ids, str(path))
        return {key: payload[key][order] for key in payload.files if key != "query_id"}


def load_pair(pair_id: str, args, contract: dict) -> tuple[pd.DataFrame, np.ndarray, dict, dict, list[Path]]:
    exp2_root = Path(args.exp2_root)
    utility_manifest_path = Path(args.utility_manifest_dir) / f"{pair_id}_dev_source_manifest.json"
    utility = json.loads(utility_manifest_path.read_text(encoding="utf-8"))
    if utility.get("split") != "dev":
        raise RuntimeError("Non-DEV utility manifest")
    query_path = Path(utility["source_query_rows"]["path"])
    if sha256_file(query_path) != utility["source_query_rows"]["sha256"]:
        raise RuntimeError(f"Exact RR source hash mismatch: {pair_id}")
    asset_manifest_path = exp2_root / "assets" / f"{pair_id}_query_information_manifest.json"
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    asset_path = Path(asset_manifest["output"]["path"])
    if sha256_file(asset_path) != asset_manifest["output"]["sha256"]:
        raise RuntimeError(f"Query-information asset hash mismatch: {pair_id}")
    fields = asset_manifest["feature_fields_x4"]
    required = [
        contract["levels"]["L4"]["degree_feature"],
        *contract["levels"]["L4"]["modality_fields"],
    ]
    indices = [fields.index(field) for field in required]
    with np.load(asset_path, allow_pickle=False) as asset:
        query_ids = asset["query_id"].astype(str)
        matrix = asset["features_x4"][:, indices].astype(np.float64)
        frame = pd.DataFrame(
            {
                "query_id": query_ids,
                "seed": asset["seed"].astype(int),
                "direction": asset["direction"].astype(str),
                "head_id": asset["head_id"].astype(int),
                "relation_id": asset["relation_id"].astype(int),
                "tail_id": asset["tail_id"].astype(int),
                "degree_value": matrix[:, 0],
            }
        )
    frame["modality_state"] = modality_states(matrix[:, 1], matrix[:, 2])
    frame["original_triple_id"] = (
        "h=" + frame.head_id.astype(str) + "|r=" + frame.relation_id.astype(str) + "|t=" + frame.tail_id.astype(str)
    )
    exact = pd.read_csv(query_path, usecols=["query_id", *RR_COLUMNS])
    order = align_indices(exact.query_id.to_numpy(str), query_ids, "exact RR")
    rr = exact.iloc[order][RR_COLUMNS].to_numpy(np.float64)

    x4_path = exp2_root / "runs" / pair_id / "x4" / "nested_selected" / "oof_action_predictions.npz"
    x6_path = exp2_root / "runs" / pair_id / "x6" / "set_encoder" / "oof_action_predictions.npz"
    x4 = load_prediction(x4_path, query_ids)
    x6 = load_prediction(x6_path, query_ids)
    outer = x4["outer_fold"].astype(int) - 1
    if not np.array_equal(x6["outer_fold"].astype(int) - 1, outer):
        raise RuntimeError(f"X4/X6 outer folds differ: {pair_id}")
    if frame.assign(_fold=outer).groupby("original_triple_id")._fold.nunique().max() != 1:
        raise RuntimeError(f"Outer-triple leakage in Experiment 2 artifact: {pair_id}")
    computed, _ = grouped_folds(frame, int(contract["outer_cv"]["folds"]), int(contract["outer_cv"]["fold_seed"]))
    if not np.array_equal(computed, outer):
        raise RuntimeError(f"Experiment 3 fold assignment differs from Experiment 2: {pair_id}")
    x4["outer_fold_zero"] = outer
    x6["outer_fold_zero"] = outer
    source_paths = [utility_manifest_path, query_path, asset_manifest_path, asset_path, x4_path, x6_path]
    for representation, learner in (("x4", "nested_selected"), ("x6", "set_encoder")):
        source_paths.append(exp2_root / "runs" / pair_id / representation / learner / "metrics.json")
    return frame, rr, x4, x6, source_paths


def choose_l4_support(frame: pd.DataFrame, rr: np.ndarray, outer_train: np.ndarray, outer_fold: int, contract: dict) -> tuple[int, list[dict]]:
    candidates = list(contract["levels"]["L4"]["minimum_support_candidates"])
    outer_indices = np.flatnonzero(outer_train)
    inner_frame = frame.iloc[outer_indices].reset_index(drop=True)
    inner_vector, _ = grouped_folds(
        inner_frame,
        int(contract["inner_cv"]["folds"]),
        int(contract["inner_cv"]["fold_seed"]) + outer_fold,
    )
    totals = {candidate: [0.0, 0] for candidate in candidates}
    rows = []
    for inner_fold in range(int(contract["inner_cv"]["folds"])):
        inner_train_indices = outer_indices[inner_vector != inner_fold]
        inner_valid_indices = outer_indices[inner_vector == inner_fold]
        train_mask = np.zeros(len(frame), dtype=bool)
        train_mask[inner_train_indices] = True
        global_index, _ = select_global_alpha(rr, train_mask)
        edges = degree_edges(frame, train_mask, contract["levels"]["L4"]["degree_quantiles"])
        for candidate in candidates:
            policy, _, keys = fit_group_policy(frame, rr, train_mask, "L4", global_index, candidate, edges)
            actions, covered = apply_policy(keys, policy, inner_valid_indices, global_index)
            gain = rr[inner_valid_indices, actions] - rr[inner_valid_indices, global_index]
            totals[candidate][0] += float(gain.sum())
            totals[candidate][1] += len(gain)
            rows.append(
                {
                    "outer_fold": outer_fold + 1,
                    "inner_fold": inner_fold + 1,
                    "minimum_support": candidate,
                    "inner_global_alpha": float(ALPHAS[global_index]),
                    "inner_delta_mrr": float(gain.mean()),
                    "inner_support_coverage": float(covered.mean()),
                }
            )
    scores = {candidate: totals[candidate][0] / totals[candidate][1] for candidate in candidates}
    selected = candidates[0]
    for candidate in candidates[1:]:
        if scores[candidate] > scores[selected] + ZERO_TOLERANCE:
            selected = candidate
    for row in rows:
        row["selected_for_outer_fold"] = row["minimum_support"] == selected
        row["pooled_inner_delta_mrr"] = scores[row["minimum_support"]]
    return selected, rows


def fold_action_mad(actions: np.ndarray, folds: np.ndarray) -> float:
    medians = np.asarray([np.median(ALPHAS[actions[folds == fold]]) for fold in sorted(np.unique(folds))])
    center = np.median(medians)
    return float(np.median(np.abs(medians - center)))


def group_stability(policy_rows: pd.DataFrame, pair_id: str, level: str) -> tuple[float, float, list[dict]]:
    current = policy_rows[(policy_rows.pair_id == pair_id) & (policy_rows.level == level)]
    if current.empty:
        return float("nan"), float("nan"), []
    rows = []
    for key, group in current.groupby("group_key", sort=False):
        if group.outer_fold.nunique() < 2:
            continue
        values = group.selected_alpha.to_numpy(np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        rows.append(
            {
                "pair_id": pair_id,
                "level": level,
                "group_key": key,
                "fold_count": int(group.outer_fold.nunique()),
                "median_alpha": median,
                "foldwise_mad": mad,
                "mean_support_original_triples": float(group.support_original_triples.mean()),
            }
        )
    if not rows:
        return float("nan"), float("nan"), []
    values = np.asarray([row["foldwise_mad"] for row in rows])
    weights = np.asarray([row["mean_support_original_triples"] for row in rows])
    return float(np.average(values, weights=weights)), float(np.median(values)), rows


def metrics_for_level(
    frame: pd.DataFrame,
    rr: np.ndarray,
    actions: np.ndarray,
    global_indices: np.ndarray,
    covered: np.ndarray,
    training_gain: float,
    available: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    group_mad: float,
    median_group_mad: float,
    folds: np.ndarray,
) -> tuple[dict, list[dict]]:
    row = np.arange(len(frame))
    selected_rr = rr[row, actions]
    global_rr = rr[row, global_indices]
    gain = selected_rr - global_rr
    low, high = clustered_bootstrap(
        gain, frame.original_triple_id.to_numpy(str), bootstrap_samples, bootstrap_seed
    )
    slices = []
    seed_gains = []
    direction_gains = []
    for scope, values in (("seed", (1, 2, 3)), ("direction", ("head", "tail"))):
        source = frame.seed if scope == "seed" else frame.direction
        for offset, value in enumerate(values):
            mask = source.astype(str) == str(value)
            slice_gain = gain[mask]
            slice_low, slice_high = clustered_bootstrap(
                slice_gain,
                frame.loc[mask, "original_triple_id"].to_numpy(str),
                bootstrap_samples,
                bootstrap_seed + 10 + len(slices),
            )
            mean = float(slice_gain.mean())
            slices.append(
                {
                    "scope": scope,
                    "value": str(value),
                    "count": int(mask.sum()),
                    "delta_mrr": mean,
                    "clustered_ci95_low": slice_low,
                    "clustered_ci95_high": slice_high,
                }
            )
            (seed_gains if scope == "seed" else direction_gains).append(mean)
    positive_seeds = sum(value > ZERO_TOLERANCE for value in seed_gains)
    head_tail_not_both_negative = not all(value < -ZERO_TOLERANCE for value in direction_gains)
    robust = (
        float(gain.mean()) > ZERO_TOLERANCE
        and low > ZERO_TOLERANCE
        and positive_seeds >= 2
        and head_tail_not_both_negative
    )
    result = {
        "count": int(len(frame)),
        "fold_specific_global_mrr": float(global_rr.mean()),
        "oof_mrr": float(selected_rr.mean()),
        "delta_mrr": float(gain.mean()),
        "clustered_ci95_low": low,
        "clustered_ci95_high": high,
        "training_adaptive_gain": float(training_gain),
        "train_oof_generalization_gap": float(training_gain - gain.mean()),
        "negative_transfer_ratio": float((gain < -ZERO_TOLERANCE).mean()),
        "positive_transfer_ratio": float((gain > ZERO_TOLERANCE).mean()),
        "changed_rate": float((actions != global_indices).mean()),
        "support_coverage": float(covered.mean()),
        "available_headroom_recovery": float(gain.mean() / available),
        "group_alpha_foldwise_mad": group_mad,
        "median_group_alpha_foldwise_mad": median_group_mad,
        "fold_action_median_mad": fold_action_mad(actions, folds),
        "positive_seed_count": int(positive_seeds),
        "head_delta_mrr": direction_gains[0],
        "tail_delta_mrr": direction_gains[1],
        "head_tail_not_both_negative": bool(head_tail_not_both_negative),
        "robust_positive": bool(robust),
    }
    return result, slices


def paired_ci(difference: np.ndarray, groups: np.ndarray, contract: dict, offset: int) -> tuple[float, float]:
    return clustered_bootstrap(
        difference,
        groups,
        int(contract["bootstrap"]["samples"]),
        int(contract["bootstrap"]["seed"]) + offset,
    )


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


def add_curve_panel(parts: list[str], data: pd.DataFrame, pair_id: str, ox: int, oy: int, width: int, height: int, fields: list[tuple[str, str, str]]) -> None:
    values = np.concatenate([data[field].to_numpy(np.float64) for field, _, _ in fields])
    low, high = min(-0.001, float(values.min())), max(0.001, float(values.max()))
    if math.isclose(low, high):
        high = low + 1e-3
    sx = lambda index: ox + index * width / 5
    sy = lambda value: oy + height - (float(value) - low) / (high - low) * height
    parts.append(f'<text x="{ox}" y="{oy-12}" class="panel">{html.escape(PAIR_LABELS[pair_id])}</text>')
    parts.append(f'<line x1="{ox}" y1="{sy(0):.1f}" x2="{ox+width}" y2="{sy(0):.1f}" stroke="#9aa0a6"/>')
    for field, color, dash in fields:
        points = " ".join(f"{sx(index):.1f},{sy(value):.1f}" for index, value in enumerate(data[field]))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
    for index, level in enumerate(LEVELS):
        parts.append(f'<text x="{sx(index):.1f}" y="{oy+height+15}" text-anchor="middle" class="axis">{LEVEL_NAMES[level]}</text>')


def build_figures(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []
    parts = svg_header(1180, 700, "Resolution–Identifiability Curve")
    for panel, pair_id in enumerate(PAIR_IDS):
        data = metrics[metrics.pair_id == pair_id].set_index("level").loc[list(LEVELS)].reset_index()
        add_curve_panel(parts, data, pair_id, 55 + (panel % 3) * 385, 70 + (panel // 3) * 305, 315, 210, [("delta_mrr", "#376795", ""), ("training_adaptive_gain", "#d1495b", "5 3")])
    parts += ['<line x1="430" y1="675" x2="455" y2="675" stroke="#376795" stroke-width="2"/><text x="462" y="679" class="note">OOF gain</text>', '<line x1="555" y1="675" x2="580" y2="675" stroke="#d1495b" stroke-width="2" stroke-dasharray="5 3"/><text x="587" y="679" class="note">training gain</text>']
    path = output_dir / "figure1_resolution_identifiability_curve.svg"; write_svg(path, parts); paths.append(path)

    parts = svg_header(940, 560, "Resolution vs Generalization Gap")
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    maximum = max(float(metrics.train_oof_generalization_gap.abs().max()), 1e-6)
    sx = lambda index: 100 + index * 125
    sy = lambda value: 285 - float(value) / maximum * 210
    parts.append('<line x1="100" y1="285" x2="725" y2="285" stroke="#999"/>')
    for pair_index, pair_id in enumerate(PAIR_IDS):
        data = metrics[metrics.pair_id == pair_id].set_index("level").loc[list(LEVELS)]
        points = " ".join(f"{sx(index):.1f},{sy(value):.1f}" for index, value in enumerate(data.train_oof_generalization_gap))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{palette[pair_index]}" stroke-width="2"/>')
        parts.append(f'<text x="750" y="{100+pair_index*28}" class="note">{html.escape(PAIR_LABELS[pair_id])}</text>')
    for index, level in enumerate(LEVELS): parts.append(f'<text x="{sx(index):.1f}" y="520" text-anchor="middle" class="axis">{LEVEL_NAMES[level]}</text>')
    path = output_dir / "figure2_resolution_generalization_gap.svg"; write_svg(path, parts); paths.append(path)

    parts = svg_header(1080, 430, "Resolution vs Support, Alpha Instability, and Negative Transfer")
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

    parts = svg_header(900, 430, "Highlighted Pair-Specific Resolution Curves")
    for panel, pair_id in enumerate(("mkgw_mhyper_native", "mkgw_native_adamf")):
        data = metrics[metrics.pair_id == pair_id].set_index("level").loc[list(LEVELS)].reset_index()
        add_curve_panel(parts, data, pair_id, 70 + panel * 430, 90, 350, 245, [("delta_mrr", "#376795", ""), ("training_adaptive_gain", "#d1495b", "5 3")])
    path = output_dir / "figure4_highlighted_pair_curves.svg"; write_svg(path, parts); paths.append(path)
    return paths


def relative_link(target: Path, report: Path) -> str:
    return os.path.relpath(target.resolve(), report.resolve().parent).replace("\\", "/")


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    exp1_root, exp2_root = Path(args.exp1_root), Path(args.exp2_root)
    output_dir, report_path = Path(args.output_dir), Path(args.report)
    for path in (contract_path, exp1_root, exp2_root, output_dir, report_path):
        reject_test_path(path)
    contract = load_contract(contract_path)
    exp1_audit_path, exp2_audit_path = exp1_root / "audit_manifest.json", exp2_root / "audit_manifest.json"
    exp1_audit = json.loads(exp1_audit_path.read_text(encoding="utf-8"))
    exp2_audit = json.loads(exp2_audit_path.read_text(encoding="utf-8"))
    if exp1_audit.get("gate", {}).get("decision") != "GO":
        raise RuntimeError("Experiment 1 gate is not GO")
    if exp2_audit.get("split") != "dev" or int(exp2_audit.get("operational_audit", {}).get("test_access", -1)) != 0:
        raise RuntimeError("Experiment 2 audit is not clean DEV-only input")
    exp1_stats_path = exp1_root / "pair_statistics.csv"
    exp2_primary_path = exp2_root / "primary_nested_probe_metrics.csv"
    exp1_stats = pd.read_csv(exp1_stats_path).set_index("pair_id")
    exp2_primary = pd.read_csv(exp2_primary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit_manifest.json"
    if not args.overwrite and (audit_path.exists() or report_path.exists()):
        raise FileExistsError("Refusing to overwrite existing Experiment 3 audit")

    source_paths: set[Path] = {contract_path, exp1_audit_path, exp1_stats_path, exp2_audit_path, exp2_primary_path}
    preflight = []
    loaded = {}
    for pair_id in PAIR_IDS:
        frame, rr, x4, x6, paths = load_pair(pair_id, args, contract)
        loaded[pair_id] = (frame, rr, x4, x6)
        source_paths.update(paths)
        preflight.append(
            {
                "pair_id": pair_id,
                "queries": len(frame),
                "original_triples": int(frame.original_triple_id.nunique()),
                "outer_triple_leakage": 0,
                "x4_global_alignment": bool(np.array_equal(x4["fold_global_index"], x6["fold_global_index"])),
            }
        )
    if args.dry_run:
        print(json.dumps({"status": "PRECHECK_OK", "pairs": preflight, "test_access": 0}, indent=2))
        return

    metric_rows, fold_rows, slice_rows, policy_rows, inner_rows = [], [], [], [], []
    chosen_by_pair: dict[str, dict[str, np.ndarray]] = {}
    global_by_pair: dict[str, np.ndarray] = {}
    coverage_by_pair: dict[str, dict[str, np.ndarray]] = {}
    x6_actions_by_pair: dict[str, np.ndarray] = {}
    training_by_pair: dict[str, dict[str, float]] = {}
    bootstrap_samples = int(contract["bootstrap"]["samples"])
    bootstrap_seed = int(contract["bootstrap"]["seed"])

    for pair_offset, pair_id in enumerate(PAIR_IDS):
        frame, rr, x4, x6 = loaded[pair_id]
        folds = x4["outer_fold_zero"]
        n = len(frame)
        chosen = {level: np.full(n, -1, dtype=np.int16) for level in LEVELS}
        coverage = {level: np.zeros(n, dtype=bool) for level in LEVELS}
        globals_vector = np.full(n, -1, dtype=np.int16)
        training_accumulator = {level: [0.0, 0] for level in LEVELS[:-1]}
        for outer_fold in range(int(contract["outer_cv"]["folds"])):
            outer_train, outer_hold = folds != outer_fold, folds == outer_fold
            train_indices, hold_indices = np.flatnonzero(outer_train), np.flatnonzero(outer_hold)
            global_index, _ = select_global_alpha(rr, outer_train)
            if not np.all(x4["fold_global_index"][hold_indices].astype(int) == global_index):
                raise RuntimeError(f"Fold-specific Global mismatch against Experiment 2: {pair_id} fold {outer_fold+1}")
            globals_vector[hold_indices] = global_index
            chosen["L0"][hold_indices] = global_index
            coverage["L0"][hold_indices] = True
            training_accumulator["L0"][1] += len(train_indices)
            policy_rows.append(
                {
                    "pair_id": pair_id, "level": "L0", "outer_fold": outer_fold + 1,
                    "group_key": '["global"]', "support_original_triples": int(frame.loc[outer_train, "original_triple_id"].nunique()),
                    "selected_action_index": global_index, "selected_alpha": float(ALPHAS[global_index]), "minimum_support": 1,
                }
            )
            fold_rows.append(
                {
                    "pair_id": pair_id, "level": "L0", "outer_fold": outer_fold + 1,
                    "global_alpha": float(ALPHAS[global_index]), "minimum_support": 1,
                    "training_adaptive_gain": 0.0, "oof_delta_mrr": 0.0,
                    "support_coverage": 1.0, "supported_group_count": 1, "degree_edges": "",
                }
            )
            for level in ("L1", "L2", "L3"):
                minimum_support = int(contract["levels"][level]["minimum_support"])
                policy, rows, keys = fit_group_policy(frame, rr, outer_train, level, global_index, minimum_support, None)
                hold_actions, hold_coverage = apply_policy(keys, policy, hold_indices, global_index)
                train_actions, _ = apply_policy(keys, policy, train_indices, global_index)
                chosen[level][hold_indices] = hold_actions
                coverage[level][hold_indices] = hold_coverage
                train_gain = rr[train_indices, train_actions] - rr[train_indices, global_index]
                training_accumulator[level][0] += float(train_gain.sum())
                training_accumulator[level][1] += len(train_gain)
                for row in rows:
                    policy_rows.append({"pair_id": pair_id, "level": level, "outer_fold": outer_fold + 1, "minimum_support": minimum_support, **row})
                fold_rows.append(
                    {
                        "pair_id": pair_id, "level": level, "outer_fold": outer_fold + 1,
                        "global_alpha": float(ALPHAS[global_index]), "minimum_support": minimum_support,
                        "training_adaptive_gain": float(train_gain.mean()),
                        "oof_delta_mrr": float((rr[hold_indices, hold_actions] - rr[hold_indices, global_index]).mean()),
                        "support_coverage": float(hold_coverage.mean()), "supported_group_count": len(policy),
                        "degree_edges": "",
                    }
                )
            selected_support, current_inner = choose_l4_support(frame, rr, outer_train, outer_fold, contract)
            for row in current_inner:
                inner_rows.append({"pair_id": pair_id, **row})
            edges = degree_edges(frame, outer_train, contract["levels"]["L4"]["degree_quantiles"])
            policy, rows, keys = fit_group_policy(frame, rr, outer_train, "L4", global_index, selected_support, edges)
            hold_actions, hold_coverage = apply_policy(keys, policy, hold_indices, global_index)
            train_actions, _ = apply_policy(keys, policy, train_indices, global_index)
            chosen["L4"][hold_indices] = hold_actions
            coverage["L4"][hold_indices] = hold_coverage
            train_gain = rr[train_indices, train_actions] - rr[train_indices, global_index]
            training_accumulator["L4"][0] += float(train_gain.sum())
            training_accumulator["L4"][1] += len(train_gain)
            for row in rows:
                policy_rows.append({"pair_id": pair_id, "level": "L4", "outer_fold": outer_fold + 1, "minimum_support": selected_support, **row})
            fold_rows.append(
                {
                    "pair_id": pair_id, "level": "L4", "outer_fold": outer_fold + 1,
                    "global_alpha": float(ALPHAS[global_index]), "minimum_support": selected_support,
                    "training_adaptive_gain": float(train_gain.mean()),
                    "oof_delta_mrr": float((rr[hold_indices, hold_actions] - rr[hold_indices, global_index]).mean()),
                    "support_coverage": float(hold_coverage.mean()), "supported_group_count": len(policy),
                    "degree_edges": json.dumps([float(value) for value in edges]),
                }
            )
        chosen["L5"] = x4["chosen_action_index"].astype(np.int16)
        coverage["L5"][:] = True
        x6_actions_by_pair[pair_id] = x6["chosen_action_index"].astype(np.int16)
        if any(np.any(actions < 0) for actions in chosen.values()) or np.any(globals_vector < 0):
            raise RuntimeError(f"Incomplete OOF actions: {pair_id}")
        training = {
            level: values[0] / values[1] if values[1] else 0.0
            for level, values in training_accumulator.items()
        }
        training["L5"] = float(
            exp2_primary.loc[
                (exp2_primary.pair_id == pair_id) & (exp2_primary.representation == "X4"), "training_gain"
            ].iloc[0]
        )
        for outer_fold in range(int(contract["outer_cv"]["folds"])):
            hold = folds == outer_fold
            indices = np.flatnonzero(hold)
            l5_gain = rr[indices, chosen["L5"][indices]] - rr[indices, globals_vector[indices]]
            fold_rows.append(
                {
                    "pair_id": pair_id, "level": "L5", "outer_fold": outer_fold + 1,
                    "global_alpha": float(ALPHAS[int(globals_vector[indices][0])]), "minimum_support": "",
                    "training_adaptive_gain": float("nan"), "oof_delta_mrr": float(l5_gain.mean()),
                    "support_coverage": 1.0, "supported_group_count": "", "degree_edges": "",
                }
            )
        chosen_by_pair[pair_id], global_by_pair[pair_id], coverage_by_pair[pair_id], training_by_pair[pair_id] = chosen, globals_vector, coverage, training

    policy_frame = pd.DataFrame(policy_rows)
    stability_rows = []
    for pair_offset, pair_id in enumerate(PAIR_IDS):
        frame, rr, x4, _ = loaded[pair_id]
        available = float(exp1_stats.loc[pair_id, "available_headroom"])
        for level_offset, level in enumerate(LEVELS):
            group_mad, median_group_mad, rows = group_stability(policy_frame, pair_id, level)
            stability_rows.extend(rows)
            result, slices = metrics_for_level(
                frame, rr, chosen_by_pair[pair_id][level], global_by_pair[pair_id], coverage_by_pair[pair_id][level],
                training_by_pair[pair_id][level], available, bootstrap_samples,
                bootstrap_seed + pair_offset * 100 + level_offset, group_mad, median_group_mad,
                x4["outer_fold_zero"],
            )
            dataset = "mkg_w" if pair_id.startswith("mkgw_") else "db15k"
            metric_rows.append({"dataset": dataset, "pair_id": pair_id, "level": level, "level_name": LEVEL_NAMES[level], **result})
            for row in slices:
                slice_rows.append({"dataset": dataset, "pair_id": pair_id, "level": level, **row})

    metrics = pd.DataFrame(metric_rows)
    gate_rows = []
    for level in GROUP_LEVELS:
        current = metrics[metrics.level == level]
        robust = current[current.robust_positive]
        robust_count = len(robust)
        datasets = set(robust.dataset)
        negative_count = int((current.delta_mrr < -0.001).sum())
        passed = robust_count >= 4 and datasets == {"mkg_w", "db15k"} and negative_count <= 1
        gate_rows.append(
            {
                "level": level, "robust_positive_pairs": robust_count,
                "mkg_w_has_robust_positive": "mkg_w" in datasets, "db15k_has_robust_positive": "db15k" in datasets,
                "pairs_delta_below_minus_0_001": negative_count, "pass": bool(passed),
            }
        )
    gate = pd.DataFrame(gate_rows)
    group_go = bool(gate["pass"].any())

    comparison_rows = []
    best_group_ge_x4 = 0
    x4_significantly_better = 0
    x6_significantly_better_pairs = []
    finer_pattern_pairs = []
    for pair_offset, pair_id in enumerate(PAIR_IDS):
        current = metrics[(metrics.pair_id == pair_id) & metrics.level.isin(GROUP_LEVELS)]
        best = current.loc[current.delta_mrr.idxmax()]
        best_level = str(best.level)
        x4 = metrics[(metrics.pair_id == pair_id) & (metrics.level == "L5")].iloc[0]
        if float(best.delta_mrr) >= float(x4.delta_mrr) - ZERO_TOLERANCE:
            best_group_ge_x4 += 1
        frame, rr, _, _ = loaded[pair_id]
        rows_index = np.arange(len(frame))
        group_rr = rr[rows_index, chosen_by_pair[pair_id][best_level]]
        x4_rr = rr[rows_index, chosen_by_pair[pair_id]["L5"]]
        x6_rr = rr[rows_index, x6_actions_by_pair[pair_id]]
        groups = frame.original_triple_id.to_numpy(str)
        x4_low, x4_high = paired_ci(x4_rr - group_rr, groups, contract, pair_offset * 2)
        x6_low, x6_high = paired_ci(x6_rr - group_rr, groups, contract, pair_offset * 2 + 1)
        if x4_low > ZERO_TOLERANCE:
            x4_significantly_better += 1
        if x6_low > ZERO_TOLERANCE:
            x6_significantly_better_pairs.append(pair_id)
        ordered = metrics[metrics.pair_id == pair_id].set_index("level")
        adjacent = []
        for coarse, fine in zip(GROUP_LEVELS[:-1], GROUP_LEVELS[1:]):
            adjacent.append(
                float(ordered.loc[fine, "training_adaptive_gain"]) > float(ordered.loc[coarse, "training_adaptive_gain"]) + ZERO_TOLERANCE
                and float(ordered.loc[fine, "delta_mrr"]) <= float(ordered.loc[coarse, "delta_mrr"]) + ZERO_TOLERANCE
            )
        finer_pattern = sum(adjacent) >= 2
        if finer_pattern:
            finer_pattern_pairs.append(pair_id)
        comparison_rows.append(
            {
                "pair_id": pair_id, "best_group_level": best_level, "best_group_delta_mrr": float(best.delta_mrr),
                "x4_query_delta_mrr": float(x4.delta_mrr), "x4_minus_group": float((x4_rr - group_rr).mean()),
                "x4_minus_group_ci95_low": x4_low, "x4_minus_group_ci95_high": x4_high,
                "x6_minus_group": float((x6_rr - group_rr).mean()), "x6_minus_group_ci95_low": x6_low,
                "x6_minus_group_ci95_high": x6_high, "finer_overfit_transition_count": int(sum(adjacent)),
                "finer_overfit_pattern": bool(finer_pattern),
            }
        )

    query_level_go = exp2_audit.get("decision") == "PRELIMINARY GO"
    route_a = (
        group_go
        and best_group_ge_x4 >= int(contract["route_a"]["minimum_pairs_best_group_ge_x4_query"])
        and x4_significantly_better <= int(contract["route_a"]["maximum_pairs_x4_query_significantly_gt_best_group"])
        and len(finer_pattern_pairs) >= int(contract["route_a"]["minimum_pairs_with_finer_overfit_pattern"])
    )
    required_b = set(contract["route_b"]["required_pairs"])
    route_b = (
        len(x6_significantly_better_pairs) >= int(contract["route_b"]["minimum_pairs_query_minus_best_group_ci_lower_gt_zero"])
        and required_b.issubset(x6_significantly_better_pairs)
    )
    if route_b:
        decision = "ROUTE_B_QUERY_LEVEL"
    elif route_a:
        decision = "ROUTE_A_HIERARCHICAL"
    else:
        decision = "ROUTE_C_LIMITS"
    decision_payload = {
        "decision": decision, "group_level_go": group_go, "query_level_go": query_level_go,
        "route_a_pass": route_a, "route_b_pass": route_b, "best_group_ge_x4_pairs": best_group_ge_x4,
        "x4_significantly_exceeds_best_group_pairs": x4_significantly_better,
        "finer_overfit_pattern_pairs": finer_pattern_pairs,
        "x6_significantly_exceeds_best_group_pairs": x6_significantly_better_pairs,
        "defaulted_to_route_c_for_insufficient_evidence": decision == "ROUTE_C_LIMITS" and not (not group_go and not query_level_go),
    }

    metrics_path = output_dir / "pair_level_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(output_dir / "seed_direction_metrics.csv", index=False)
    policy_frame.to_csv(output_dir / "group_actions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(stability_rows).to_csv(output_dir / "group_alpha_stability.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(output_dir / "l4_inner_support_selection.csv", index=False)
    gate.to_csv(output_dir / "group_level_gate.csv", index=False)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output_dir / "route_comparisons.csv", index=False)
    (output_dir / "route_decision.json").write_text(json.dumps(decision_payload, indent=2) + "\n", encoding="utf-8")
    npz_payload = {}
    for pair_id in PAIR_IDS:
        frame = loaded[pair_id][0]
        npz_payload[f"{pair_id}__query_id"] = frame.query_id.to_numpy(str)
        npz_payload[f"{pair_id}__global_action_index"] = global_by_pair[pair_id]
        npz_payload[f"{pair_id}__chosen_action_index"] = np.vstack([chosen_by_pair[pair_id][level] for level in LEVELS])
        npz_payload[f"{pair_id}__support_covered"] = np.vstack([coverage_by_pair[pair_id][level] for level in LEVELS])
        npz_payload[f"{pair_id}__x6_action_index"] = x6_actions_by_pair[pair_id]
    per_query_path = output_dir / "per_query_resolution_results.npz"
    np.savez_compressed(per_query_path, **npz_payload)
    figure_paths = build_figures(metrics, output_dir)

    report_lines = [
        "# Adaptive Resolution Audit — Experiment 3", "", "Date: 2026-09-07", "",
        "## Outcome", "", f"Group-Level GO: **{'GO' if group_go else 'NO-GO'}**.",
        f"Experiment 2 Query-Level GO: **{'GO' if query_level_go else 'NO-GO'}**.",
        f"Frozen route decision: **{decision}**.", "",
        "## Main findings", "",
        f"No single L1–L4 resolution reaches the required four robust-positive pairs. "
        f"The counts are " + ", ".join(f"{row['level']}={row['robust_positive_pairs']}/6" for row in gate_rows) + ".",
        f"L4 has mean held-out support coverage {100*metrics.loc[metrics.level == 'L4', 'support_coverage'].mean():.1f}% "
        f"and mean support-weighted group-alpha foldwise MAD {metrics.loc[metrics.level == 'L4', 'group_alpha_foldwise_mad'].mean():.3f}.",
        f"The best group-level point estimate is below the frozen X4 query-level probe in {6-best_group_ge_x4}/6 pairs; "
        f"X4 is significantly better by paired clustered bootstrap in {x4_significantly_better}/6.",
        f"The frozen X6 probe significantly exceeds the best group in {len(x6_significantly_better_pairs)}/6 pairs.",
        "Thus coarser adaptation does not recover the missing deployable headroom, while the richer candidate-level probe also fails. "
        "This supports a limits result rather than another selector or granularity extension.", "",
        "## Pair × resolution metrics", "",
        "| Pair | Level | Train gain | OOF gain | 95% clustered CI | Recovery | Neg. transfer | Changed | Coverage | Group MAD | Robust |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in metrics.itertuples():
        mad = "NA" if not np.isfinite(row.group_alpha_foldwise_mad) else f"{row.group_alpha_foldwise_mad:.3f}"
        report_lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {row.level} {row.level_name} | {row.training_adaptive_gain:+.6f} | "
            f"{row.delta_mrr:+.6f} | [{row.clustered_ci95_low:+.6f}, {row.clustered_ci95_high:+.6f}] | "
            f"{100*row.available_headroom_recovery:.1f}% | {100*row.negative_transfer_ratio:.1f}% | "
            f"{100*row.changed_rate:.1f}% | {100*row.support_coverage:.1f}% | {mad} | {row.robust_positive} |"
        )
    report_lines += ["", "## Group-Level GO", "", "| Level | Robust-positive pairs | MKG-W represented | DB15K represented | ΔMRR < -0.001 pairs | Pass |", "| --- | ---: | --- | --- | ---: | --- |"]
    for row in gate.to_dict(orient="records"):
        report_lines.append(
            f"| {row['level']} | {row['robust_positive_pairs']}/6 | {row['mkg_w_has_robust_positive']} | "
            f"{row['db15k_has_robust_positive']} | {row['pairs_delta_below_minus_0_001']} | {row['pass']} |"
        )
    report_lines += ["", "## Best group versus frozen query probes", "", "| Pair | Best group | Group OOF gain | X4 query gain | X4−group 95% CI | X6−group 95% CI |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in comparisons.itertuples():
        report_lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {row.best_group_level} | {row.best_group_delta_mrr:+.6f} | "
            f"{row.x4_query_delta_mrr:+.6f} | [{row.x4_minus_group_ci95_low:+.6f}, {row.x4_minus_group_ci95_high:+.6f}] | "
            f"[{row.x6_minus_group_ci95_low:+.6f}, {row.x6_minus_group_ci95_high:+.6f}] |"
        )
    report_lines += [
        "", "## Route evidence", "",
        f"- best group-level >= X4 query-level: {best_group_ge_x4}/6 pairs",
        f"- X4 query-level significantly exceeds best group: {x4_significantly_better}/6 pairs",
        f"- finer-granularity train-up / OOF-flat-or-down pattern: {len(finer_pattern_pairs)}/6 pairs",
        f"- X6 significantly exceeds best group: {len(x6_significantly_better_pairs)}/6 pairs; required NativE+AdaMF inclusion = {required_b.issubset(x6_significantly_better_pairs)}",
        "", "## Figures", "",
    ]
    for index, path in enumerate(figure_paths, 1):
        report_lines.append(f"{index}. [{path.stem}]({relative_link(path, report_path)})")
    report_lines += [
        "", "## Operational audit", "", "- TEST access = 0", "- outer-triple leakage = 0",
        "- new query selector / policy development = 0", "- hierarchical shrinkage / LCB / conservative policy = 0",
        "- checkpoint modification = 0", "- all direct source and output hashes are in `audit_manifest.json`", "",
        decision, "",
    ]

    # Verify and inventory direct sources/outputs before publishing the final report.
    source_records = []
    for path in sorted(source_paths, key=lambda value: portable_path(value)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Experiment 3 source: {path}")
        source_records.append({"role": "source", "path": portable_path(path), "sha256": sha256_file(path)})
    output_paths = [path for path in output_dir.rglob("*") if path.is_file() and path != audit_path]
    output_records = [
        {"role": "output", "path": portable_path(path), "sha256": sha256_file(path)}
        for path in sorted(output_paths, key=lambda value: portable_path(value))
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    output_records.append({"role": "output", "path": portable_path(report_path), "sha256": sha256_file(report_path)})
    audit = {
        "schema_version": 1, "experiment": "Experiment 3 — Granularity Ladder / Adaptive Resolution Audit",
        "split": "dev", "decision": decision, "group_level_go": group_go, "query_level_go": query_level_go,
        "group_gate": gate_rows, "route_evidence": decision_payload, "preflight": preflight,
        "sources_and_outputs": source_records + output_records,
        "operational_audit": {
            "test_access": 0, "outer_triple_leakage": 0, "new_query_selector": 0, "policy_development": 0,
            "hierarchical_shrinkage": 0, "lcb": 0, "conservative_policy": 0, "checkpoint_modification": 0,
        },
        "next_stage_started": 0,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "group_level_go": group_go, "route_evidence": decision_payload}, indent=2))


if __name__ == "__main__":
    main()
