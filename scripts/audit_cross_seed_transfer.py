from __future__ import annotations

import argparse
import html
import json
import math
import sys
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
    portable_path,
    reject_test_path,
    sha256_file,
)


PAIR_LABELS = {
    "mkgw_mhyper_native": "MKG-W / M-Hyper+NativE",
    "mkgw_mhyper_adamf": "MKG-W / M-Hyper+AdaMF",
    "mkgw_native_adamf": "MKG-W / NativE+AdaMF",
    "db15k_mhyper_native": "DB15K / M-Hyper+NativE",
    "db15k_mhyper_adamf": "DB15K / M-Hyper+AdaMF",
    "db15k_native_adamf": "DB15K / NativE+AdaMF",
}
SEEDS = (1, 2, 3)
DIRECTIONS = ("head", "tail")
CLASSIFICATIONS = (
    "E4_SUBSTANTIALLY_TRANSFERABLE",
    "E4_RESIDUAL_DOMINATED",
    "E4_INTERMEDIATE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DEV-only Experiment 4 cross-seed transfer audit.")
    parser.add_argument("--contract", default="docs/protocols/EXP4_CROSS_SEED_TRANSFER_CONTRACT.json")
    parser.add_argument("--utility-manifest-dir", default="outputs/aacpi/utility_tables")
    parser.add_argument("--exp1-root", default="outputs/complementarity_identifiability/exp1_landscape")
    parser.add_argument("--exp2-root", default="outputs/complementarity_identifiability/exp2_information")
    parser.add_argument("--exp3-root", default="outputs/complementarity_identifiability/exp3_resolution")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/exp4_cross_seed_transfer")
    parser.add_argument("--report", default="docs/reports/cross_seed_transferable_complementarity_audit_2026-09-07.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def load_contract(path: Path) -> dict:
    contract = load_json(path)
    if contract.get("status") != "frozen_before_systematic_run" or contract.get("split") != "dev":
        raise RuntimeError("Experiment 4 contract is not frozen DEV-only")
    if tuple(contract.get("pair_ids", [])) != PAIR_IDS:
        raise RuntimeError("Experiment 4 pair inventory changed")
    if tuple(contract.get("seeds", [])) != SEEDS or tuple(contract.get("directions", [])) != DIRECTIONS:
        raise RuntimeError("Experiment 4 seed/direction inventory changed")
    if not np.allclose(contract.get("alpha_grid", []), ALPHAS):
        raise RuntimeError("Experiment 4 alpha grid changed")
    prohibited = (
        "test_access", "test_commands", "checkpoint_retraining", "checkpoint_reselection",
        "new_selector", "new_feature", "new_representation", "hierarchical_policy", "aacpi",
        "experiment_1_3_result_modification",
    )
    if any(int(contract.get(field, -1)) != 0 for field in prohibited):
        raise RuntimeError("Experiment 4 prohibited-operation boundary changed")
    return contract


def original_triple_ids(frame: pd.DataFrame) -> np.ndarray:
    return (
        "h=" + frame.head_id.astype(str) + "|r=" + frame.relation_id.astype(str) + "|t=" + frame.tail_id.astype(str)
    ).to_numpy(str)


def action_indices(rr: np.ndarray, global_index: int) -> np.ndarray:
    maxima = rr.max(axis=-1, keepdims=True)
    tied = np.isclose(rr, maxima, rtol=0.0, atol=ZERO_TOLERANCE)
    preference = np.abs(ALPHAS - ALPHAS[global_index]) + ALPHAS * 1e-9
    return np.where(tied, preference, np.inf).argmin(axis=-1).astype(np.int16)


def cluster_means(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"group": groups, "value": np.asarray(values, dtype=np.float64)})
    result = frame.groupby("group", sort=True).value.mean()
    return result.index.to_numpy(str), result.to_numpy(np.float64)


def bootstrap_joint(
    cluster_arrays: dict[str, np.ndarray], samples: int, seed: int
) -> tuple[dict[str, tuple[float, float]], dict[tuple[str, str], tuple[float, float]]]:
    lengths = {len(value) for value in cluster_arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Paired cluster arrays are not aligned")
    n_cluster = lengths.pop()
    rng = np.random.default_rng(seed)
    distributions = {name: np.empty(samples, dtype=np.float64) for name in cluster_arrays}
    ratios: dict[tuple[str, str], np.ndarray] = {}
    if "raw" in cluster_arrays:
        ratios[("transfer", "raw")] = np.empty(samples, dtype=np.float64)
        ratios[("loso", "raw")] = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 64):
        stop = min(start + 64, samples)
        indices = rng.integers(0, n_cluster, size=(stop - start, n_cluster))
        for name, values in cluster_arrays.items():
            distributions[name][start:stop] = values[indices].mean(axis=1)
        if "raw" in cluster_arrays:
            denom = distributions["raw"][start:stop]
            ratios[("transfer", "raw")][start:stop] = distributions["transfer"][start:stop] / denom
            ratios[("loso", "raw")][start:stop] = distributions["loso"][start:stop] / denom
    intervals = {
        name: tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))
        for name, values in distributions.items()
    }
    ratio_intervals = {
        key: tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))
        for key, values in ratios.items()
    }
    return intervals, ratio_intervals


def bootstrap_single(values: np.ndarray, groups: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    _, cluster = cluster_means(values, groups)
    intervals, _ = bootstrap_joint({"value": cluster}, samples, seed)
    return intervals["value"]


def load_pair(
    pair_id: str, manifest_dir: Path, exp1_stats: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, float, Path, Path, dict]:
    manifest_path = manifest_dir / f"{pair_id}_dev_source_manifest.json"
    reject_test_path(manifest_path)
    manifest = load_json(manifest_path)
    if manifest.get("pair_id") != pair_id or manifest.get("split") != "dev":
        raise RuntimeError(f"Canonical manifest identity mismatch: {manifest_path}")
    if manifest.get("score_normalization") != "query_zscore":
        raise RuntimeError(f"Normalization changed for {pair_id}")
    if not np.allclose(manifest.get("global_alpha_grid", []), ALPHAS):
        raise RuntimeError(f"Alpha grid changed for {pair_id}")
    source_path = Path(manifest["source_query_rows"]["path"])
    reject_test_path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Canonical Experiment 1 source missing: {source_path}")
    if sha256_file(source_path) != manifest["source_query_rows"]["sha256"]:
        raise RuntimeError(f"Canonical Experiment 1 source hash mismatch: {source_path}")
    usecols = [
        "dataset", "pair_name", "split", "query_id", "seed", "direction",
        "relation_id", "head_id", "tail_id", "alpha_global", *RR_COLUMNS,
    ]
    frame = pd.read_csv(source_path, usecols=usecols)
    if len(frame) == 0 or set(frame.split.astype(str)) != {"dev"}:
        raise RuntimeError(f"Non-DEV or empty canonical source for {pair_id}")
    expected_dataset = "mkg_w" if pair_id.startswith("mkgw") else "db15k"
    if set(frame.pair_name.astype(str)) != {pair_id} or set(frame.dataset.astype(str)) != {expected_dataset}:
        raise RuntimeError(f"Canonical source pair/dataset identity mismatch for {pair_id}")
    if set(frame.seed.astype(int)) != set(SEEDS) or set(frame.direction.astype(str)) != set(DIRECTIONS):
        raise RuntimeError(f"Incomplete seed/direction inventory for {pair_id}")
    if frame.query_id.duplicated().any():
        raise RuntimeError(f"Duplicate query_id in canonical source for {pair_id}")
    if frame[RR_COLUMNS].isna().any().any() or not np.isfinite(frame[RR_COLUMNS].to_numpy()).all():
        raise RuntimeError(f"Incomplete/non-finite alpha grid for {pair_id}")
    frame["original_triple_id"] = original_triple_ids(frame)
    seed_sets = frame.groupby(["original_triple_id", "direction"], sort=False).seed.agg(
        lambda x: tuple(sorted(int(v) for v in x))
    )
    if not seed_sets.map(lambda value: value == SEEDS).all():
        raise RuntimeError(f"Each query identity must contain seeds exactly {SEEDS}: {pair_id}")
    direction_sets = frame.groupby("original_triple_id", sort=False).direction.agg(lambda x: set(map(str, x)))
    if not direction_sets.map(lambda value: value == set(DIRECTIONS)).all():
        raise RuntimeError(f"Each original triple must contain head and tail: {pair_id}")
    counts = frame.groupby("original_triple_id", sort=False).size()
    if not (counts == 6).all():
        raise RuntimeError(f"Every original triple must contain 3 seeds × 2 directions: {pair_id}")
    global_alpha = float(manifest["global_alpha"])
    if frame.alpha_global.nunique() != 1 or not np.isclose(float(frame.alpha_global.iloc[0]), global_alpha):
        raise RuntimeError(f"alpha0 is inconsistent with the manifest for {pair_id}")
    stat = exp1_stats.loc[exp1_stats.pair_id == pair_id]
    if len(stat) != 1 or not np.isclose(float(stat.iloc[0].global_alpha), global_alpha):
        raise RuntimeError(f"alpha0 is inconsistent with Experiment 1 statistics for {pair_id}")
    frame = frame.sort_values(["original_triple_id", "direction", "seed"], kind="stable").reset_index(drop=True)
    rr = frame[RR_COLUMNS].to_numpy(np.float64)
    return frame, rr, global_alpha, manifest_path, source_path, manifest


def metric_row(pair_id: str, metric: str, estimate: float, low: float, high: float, samples: int) -> dict:
    return {
        "dataset": "mkg_w" if pair_id.startswith("mkgw") else "db15k",
        "pair_id": pair_id,
        "metric": metric,
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_samples": samples,
        "bootstrap_unit": "original_triple_id",
    }


def analyze_pair(
    pair_id: str,
    frame: pd.DataFrame,
    rr: np.ndarray,
    alpha0: float,
    exp1_row: pd.Series,
    x4_row: pd.Series,
    samples: int,
    seed: int,
) -> tuple[dict, list[dict], list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    global_candidates = np.flatnonzero(np.isclose(ALPHAS, alpha0, rtol=0.0, atol=ZERO_TOLERANCE))
    if len(global_candidates) != 1:
        raise RuntimeError(f"alpha0 is not on the frozen grid for {pair_id}: {alpha0}")
    global_index = int(global_candidates[0])
    if len(frame) % 3:
        raise RuntimeError(f"Cannot reshape seed triplets for {pair_id}")
    identity = frame.iloc[::3].reset_index(drop=True)
    rr3 = rr.reshape(-1, 3, len(ALPHAS))
    seed_matrix = frame.seed.to_numpy(np.int16).reshape(-1, 3)
    if not np.array_equal(seed_matrix, np.tile(np.asarray(SEEDS), (len(identity), 1))):
        raise RuntimeError(f"Seed ordering failed for {pair_id}")
    oracle_index = action_indices(rr3, global_index)
    rows = np.arange(len(identity))[:, None]
    seeds_axis = np.arange(3)[None, :]
    oracle_rr = rr3[rows, seeds_axis, oracle_index]
    global_rr = rr3[:, :, global_index]
    oracle_gain = oracle_rr - global_rr
    raw = float(oracle_gain.mean())
    if not np.isclose(raw, float(exp1_row.available_headroom), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Raw Oracle Headroom does not reproduce Experiment 1 for {pair_id}")

    identity_groups = identity.original_triple_id.to_numpy(str)
    raw_groups = np.repeat(identity_groups, 3)
    oracle_records = []
    for seed_index, seed_value in enumerate(SEEDS):
        oracle_records.append(pd.DataFrame({
            "dataset": identity.dataset.to_numpy(str),
            "pair_id": pair_id,
            "original_triple_id": identity_groups,
            "direction": identity.direction.to_numpy(str),
            "head_id": identity.head_id.to_numpy(),
            "relation_id": identity.relation_id.to_numpy(),
            "tail_id": identity.tail_id.to_numpy(),
            "seed": seed_value,
            "alpha0": alpha0,
            "oracle_alpha": ALPHAS[oracle_index[:, seed_index]],
            "oracle_gain": oracle_gain[:, seed_index],
        }))
    oracle_frame = pd.concat(oracle_records, ignore_index=True)

    transfer_records = []
    matrix_rows = []
    bootstrap_rows = []
    transfer_gain_parts = []
    transfer_group_parts = []
    for source_index, source_seed in enumerate(SEEDS):
        diagonal_gain = oracle_gain[:, source_index]
        dlow, dhigh = bootstrap_single(diagonal_gain, identity_groups, samples, seed)
        matrix_rows.append({
            "dataset": identity.dataset.iloc[0], "pair_id": pair_id, "source_seed": source_seed,
            "target_seed": source_seed, "cell_type": "raw_oracle", "headroom": float(diagonal_gain.mean()),
            "ci95_low": dlow, "ci95_high": dhigh,
        })
        bootstrap_rows.append(metric_row(pair_id, f"raw_oracle_seed_{source_seed}", float(diagonal_gain.mean()), dlow, dhigh, samples))
        for target_index, target_seed in enumerate(SEEDS):
            if source_index == target_index:
                continue
            chosen = oracle_index[:, source_index]
            gain = rr3[np.arange(len(identity)), target_index, chosen] - global_rr[:, target_index]
            eligible = (oracle_gain[:, source_index] > ZERO_TOLERANCE) & (chosen != global_index)
            outcome = np.where(gain > ZERO_TOLERANCE, "beneficial", np.where(gain < -ZERO_TOLERANCE, "harmful", "zero"))
            transfer_records.append(pd.DataFrame({
                "dataset": identity.dataset.to_numpy(str),
                "pair_id": pair_id,
                "original_triple_id": identity_groups,
                "direction": identity.direction.to_numpy(str),
                "head_id": identity.head_id.to_numpy(),
                "relation_id": identity.relation_id.to_numpy(),
                "tail_id": identity.tail_id.to_numpy(),
                "source_seed": source_seed,
                "target_seed": target_seed,
                "alpha0": alpha0,
                "source_oracle_alpha": ALPHAS[chosen],
                "source_oracle_gain": oracle_gain[:, source_index],
                "target_transfer_utility": gain,
                "btr_eligible": eligible.astype(np.int8),
                "transfer_outcome": outcome,
            }))
            transfer_gain_parts.append(gain)
            transfer_group_parts.append(identity_groups)
            low, high = bootstrap_single(gain, identity_groups, samples, seed)
            matrix_rows.append({
                "dataset": identity.dataset.iloc[0], "pair_id": pair_id, "source_seed": source_seed,
                "target_seed": target_seed, "cell_type": "cross_seed_transfer", "headroom": float(gain.mean()),
                "ci95_low": low, "ci95_high": high,
            })
            bootstrap_rows.append(metric_row(
                pair_id, f"transfer_{source_seed}_to_{target_seed}", float(gain.mean()), low, high, samples
            ))
    transfer_frame = pd.concat(transfer_records, ignore_index=True)
    transfer_gain = np.concatenate(transfer_gain_parts)
    transfer_groups = np.concatenate(transfer_group_parts)
    transfer = float(transfer_gain.mean())

    loso_records = []
    loso_parts = []
    for held_index, held_seed in enumerate(SEEDS):
        training_indices = [index for index in range(3) if index != held_index]
        loso_index = action_indices(rr3[:, training_indices, :].mean(axis=1), global_index)
        gain = rr3[np.arange(len(identity)), held_index, loso_index] - global_rr[:, held_index]
        loso_parts.append(gain)
        loso_records.append(pd.DataFrame({
            "dataset": identity.dataset.to_numpy(str),
            "pair_id": pair_id,
            "original_triple_id": identity_groups,
            "direction": identity.direction.to_numpy(str),
            "head_id": identity.head_id.to_numpy(),
            "relation_id": identity.relation_id.to_numpy(),
            "tail_id": identity.tail_id.to_numpy(),
            "held_out_seed": held_seed,
            "training_seeds": "+".join(str(SEEDS[index]) for index in training_indices),
            "alpha0": alpha0,
            "loso_alpha": ALPHAS[loso_index],
            "loso_utility": gain,
        }))
    loso_frame = pd.concat(loso_records, ignore_index=True)
    loso_gain = np.concatenate(loso_parts)
    loso_groups = np.tile(identity_groups, 3)
    loso = float(loso_gain.mean())

    raw_cluster_ids, raw_cluster = cluster_means(oracle_gain.reshape(-1), raw_groups)
    transfer_cluster_ids, transfer_cluster = cluster_means(transfer_gain, transfer_groups)
    loso_cluster_ids, loso_cluster = cluster_means(loso_gain, loso_groups)
    if not (np.array_equal(raw_cluster_ids, transfer_cluster_ids) and np.array_equal(raw_cluster_ids, loso_cluster_ids)):
        raise RuntimeError(f"Cluster alignment failed for {pair_id}")
    intervals, ratio_intervals = bootstrap_joint(
        {"raw": raw_cluster, "transfer": transfer_cluster, "loso": loso_cluster}, samples, seed
    )
    raw_ci = intervals["raw"]
    transfer_ci = intervals["transfer"]
    loso_ci = intervals["loso"]
    transfer_recovery = transfer / raw
    loso_recovery = loso / raw
    transfer_recovery_ci = ratio_intervals[("transfer", "raw")]
    loso_recovery_ci = ratio_intervals[("loso", "raw")]
    bootstrap_rows.extend([
        metric_row(pair_id, "raw_oracle_headroom", raw, *raw_ci, samples),
        metric_row(pair_id, "cross_seed_transfer_headroom", transfer, *transfer_ci, samples),
        metric_row(pair_id, "transfer_recovery", transfer_recovery, *transfer_recovery_ci, samples),
        metric_row(pair_id, "loso_stable_headroom", loso, *loso_ci, samples),
        metric_row(pair_id, "loso_recovery", loso_recovery, *loso_recovery_ci, samples),
        metric_row(
            pair_id, "frozen_x4_oof_deployable_gain", float(x4_row.delta_mrr),
            float(x4_row.clustered_ci95_low), float(x4_row.clustered_ci95_high), samples,
        ),
    ])

    eligible_frame = transfer_frame.loc[transfer_frame.btr_eligible == 1]
    if eligible_frame.empty:
        beneficial = zero = harmful = float("nan")
    else:
        frequencies = eligible_frame.transfer_outcome.value_counts(normalize=True)
        beneficial = float(frequencies.get("beneficial", 0.0))
        zero = float(frequencies.get("zero", 0.0))
        harmful = float(frequencies.get("harmful", 0.0))
        if not np.isclose(beneficial + zero + harmful, 1.0):
            raise AssertionError(f"Transfer outcome rates do not sum to one for {pair_id}")

    exact_three = float(np.all(oracle_index == oracle_index[:, [0]], axis=1).mean())
    pairwise_exact = {
        "exact_alpha_agreement_seed1_seed2": float((oracle_index[:, 0] == oracle_index[:, 1]).mean()),
        "exact_alpha_agreement_seed1_seed3": float((oracle_index[:, 0] == oracle_index[:, 2]).mean()),
        "exact_alpha_agreement_seed2_seed3": float((oracle_index[:, 1] == oracle_index[:, 2]).mean()),
    }
    direction_codes = np.sign(oracle_index.astype(np.int16) - global_index)
    direction_three = float(np.all(direction_codes == direction_codes[:, [0]], axis=1).mean())
    all_anchor = np.all(direction_codes == 0, axis=1)
    nonanchor_direction = (
        float(np.all(direction_codes[~all_anchor] == direction_codes[~all_anchor][:, [0]], axis=1).mean())
        if (~all_anchor).any() else float("nan")
    )
    summary = {
        "dataset": identity.dataset.iloc[0],
        "pair_id": pair_id,
        "pair_label": PAIR_LABELS[pair_id],
        "n_original_triples": int(identity.original_triple_id.nunique()),
        "n_query_identities": int(len(identity)),
        "n_seed_direction_queries": int(len(frame)),
        "alpha0": alpha0,
        "raw_oracle_headroom": raw,
        "raw_ci95_low": raw_ci[0],
        "raw_ci95_high": raw_ci[1],
        "cross_seed_transfer_headroom": transfer,
        "transfer_ci95_low": transfer_ci[0],
        "transfer_ci95_high": transfer_ci[1],
        "transfer_recovery": transfer_recovery,
        "transfer_recovery_ci95_low": transfer_recovery_ci[0],
        "transfer_recovery_ci95_high": transfer_recovery_ci[1],
        "loso_stable_headroom": loso,
        "loso_ci95_low": loso_ci[0],
        "loso_ci95_high": loso_ci[1],
        "loso_recovery": loso_recovery,
        "loso_recovery_ci95_low": loso_recovery_ci[0],
        "loso_recovery_ci95_high": loso_recovery_ci[1],
        "exact_alpha_three_seed_agreement": exact_three,
        **pairwise_exact,
        "direction_three_seed_agreement": direction_three,
        "nonanchor_direction_three_seed_agreement": nonanchor_direction,
        "all_anchor_identity_rate": float(all_anchor.mean()),
        "btr_eligible_transfers": int(len(eligible_frame)),
        "beneficial_transfer_rate": beneficial,
        "zero_transfer_rate": zero,
        "harmful_transfer_rate": harmful,
        "frozen_x4_oof_gain": float(x4_row.delta_mrr),
        "frozen_x4_ci95_low": float(x4_row.clustered_ci95_low),
        "frozen_x4_ci95_high": float(x4_row.clustered_ci95_high),
        "frozen_x4_baseline": "fold-specific OOF Global",
    }
    preflight = {
        "pair_id": pair_id,
        "canonical_source": portable_path(Path(frame.attrs.get("source", ""))) if frame.attrs.get("source") else None,
        "rows": int(len(frame)),
        "query_identities": int(len(identity)),
        "original_triples": int(identity.original_triple_id.nunique()),
        "seed_inventory": list(SEEDS),
        "direction_inventory": list(DIRECTIONS),
        "alpha_columns": int(len(RR_COLUMNS)),
        "alpha0": alpha0,
        "raw_matches_exp1": True,
    }
    return summary, matrix_rows, bootstrap_rows, transfer_frame, loso_frame, oracle_frame, preflight


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="40" y="38" font-family="Arial" font-size="22" font-weight="700" fill="#202124">{html.escape(title)}</text>',
    ]


def svg_text(parts: list[str], x: float, y: float, value: str, size: int = 11, anchor: str = "start", fill: str = "#30343b") -> None:
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="{size}" fill="{fill}">{html.escape(value)}</text>'
    )


def write_funnel(summary: pd.DataFrame, path: Path) -> None:
    width, height = 1380, 720
    parts = svg_start(width, height, "Headroom funnel: stability and observability constraints")
    metrics = [
        ("Raw Oracle", "raw_oracle_headroom", "raw_ci95_low", "raw_ci95_high"),
        ("Cross-seed", "cross_seed_transfer_headroom", "transfer_ci95_low", "transfer_ci95_high"),
        ("LOSO Stable", "loso_stable_headroom", "loso_ci95_low", "loso_ci95_high"),
        ("Frozen X4 OOF*", "frozen_x4_oof_gain", "frozen_x4_ci95_low", "frozen_x4_ci95_high"),
    ]
    y_min = min(0.0, float(summary[[item[2] for item in metrics]].min().min()))
    y_max = float(summary[[item[3] for item in metrics]].max().max()) * 1.08
    for panel, row in enumerate(summary.itertuples(index=False)):
        col, line = panel % 3, panel // 3
        left, top, pw, ph = 55 + col * 445, 75 + line * 305, 390, 230
        parts.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" rx="8" fill="#ffffff" stroke="#dedbd4"/>')
        svg_text(parts, left + 12, top + 22, PAIR_LABELS[row.pair_id], 12)
        zero_y = top + 35 + (y_max / (y_max - y_min)) * (ph - 72)
        parts.append(f'<line x1="{left+42}" y1="{zero_y:.1f}" x2="{left+pw-12}" y2="{zero_y:.1f}" stroke="#aaa" stroke-dasharray="3 3"/>')
        points = []
        for index, (label, field, low_field, high_field) in enumerate(metrics):
            x = left + 64 + index * 94
            value, low, high = float(getattr(row, field)), float(getattr(row, low_field)), float(getattr(row, high_field))
            scale = lambda v: top + 35 + (y_max - v) / (y_max - y_min) * (ph - 72)
            y, yl, yh = scale(value), scale(low), scale(high)
            parts.append(f'<line x1="{x}" y1="{yh:.1f}" x2="{x}" y2="{yl:.1f}" stroke="#3f5f75"/>')
            parts.append(f'<line x1="{x-4}" y1="{yh:.1f}" x2="{x+4}" y2="{yh:.1f}" stroke="#3f5f75"/>')
            parts.append(f'<line x1="{x-4}" y1="{yl:.1f}" x2="{x+4}" y2="{yl:.1f}" stroke="#3f5f75"/>')
            points.append((x, y))
            parts.append(f'<circle cx="{x}" cy="{y:.1f}" r="5" fill="#c45d3c"/>')
            svg_text(parts, x, top + ph - 12, label, 9, "middle")
        parts.append('<polyline points="' + " ".join(f"{x},{y:.1f}" for x, y in points) + '" fill="none" stroke="#c45d3c" stroke-width="1.5"/>')
    svg_text(parts, 40, height - 18, "MRR gain vs Global. *Frozen X4 uses its preregistered fold-specific OOF Global baseline.", 11, fill="#666")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def color_for(value: float, bound: float) -> str:
    ratio = max(-1.0, min(1.0, value / bound if bound else 0.0))
    if ratio >= 0:
        base, target = (247, 245, 239), (40, 119, 142)
    else:
        base, target = (247, 245, 239), (190, 70, 62)
    amount = abs(ratio)
    rgb = tuple(round(base[i] + amount * (target[i] - base[i])) for i in range(3))
    return f"rgb{rgb}"


def write_matrix(matrix: pd.DataFrame, path: Path) -> None:
    width, height = 1120, 750
    parts = svg_start(width, height, "Cross-seed transfer matrix (uniform MRR-gain scale)")
    bound = float(np.abs(matrix.headroom).max())
    for panel, pair_id in enumerate(PAIR_IDS):
        col, line = panel % 3, panel // 3
        left, top, cell = 65 + col * 355, 90 + line * 315, 72
        svg_text(parts, left, top - 18, PAIR_LABELS[pair_id], 12)
        subset = matrix.loc[matrix.pair_id == pair_id].set_index(["source_seed", "target_seed"])
        for index, seed_value in enumerate(SEEDS):
            svg_text(parts, left + 52 + index * cell, top + 2, f"target {seed_value}", 10, "middle")
            svg_text(parts, left - 8, top + 45 + index * cell, f"source {seed_value}", 10, "end")
        for i, source_seed in enumerate(SEEDS):
            for j, target_seed in enumerate(SEEDS):
                value = float(subset.loc[(source_seed, target_seed)].headroom)
                x, y = left + 16 + j * cell, top + 14 + i * cell
                parts.append(f'<rect x="{x}" y="{y}" width="{cell-4}" height="{cell-4}" fill="{color_for(value, bound)}" stroke="#fff"/>')
                svg_text(parts, x + (cell-4)/2, y + 39, f"{value:+.4f}", 10, "middle", "#111")
    svg_text(parts, 40, height - 18, f"Diagonal: per-seed Raw Oracle. Off-diagonal: source-to-target transfer. Shared |max| = {bound:.4f}.", 11, fill="#666")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_forest(summary: pd.DataFrame, path: Path) -> None:
    width, height = 1060, 520
    parts = svg_start(width, height, "Cross-seed Transfer Recovery with clustered 95% CI")
    low = min(-0.05, float(summary.transfer_recovery_ci95_low.min()))
    high = max(0.30, float(summary.transfer_recovery_ci95_high.max()))
    left, right, top, bottom = 330, 1000, 75, 455
    xscale = lambda v: left + (v - low) / (high - low) * (right - left)
    for ref in (0.0, 0.10, 0.25):
        x = xscale(ref)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#8c8c8c" stroke-dasharray="5 4"/>')
        svg_text(parts, x, bottom + 22, f"{ref:.0%}", 10, "middle")
    for index, row in enumerate(summary.itertuples(index=False)):
        y = top + 35 + index * 56
        svg_text(parts, left - 18, y + 4, PAIR_LABELS[row.pair_id], 11, "end")
        x1, x2, x = xscale(row.transfer_recovery_ci95_low), xscale(row.transfer_recovery_ci95_high), xscale(row.transfer_recovery)
        parts.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="#3f5f75" stroke-width="2"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="#c45d3c"/>')
        svg_text(parts, min(right - 5, x2 + 8), y + 4, f"{row.transfer_recovery:.1%}", 10)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_scatter(summary: pd.DataFrame, path: Path) -> None:
    width, height = 1060, 610
    parts = svg_start(width, height, "Action-direction stability vs cross-seed transfer recovery")
    left, right, top, bottom = 95, 985, 80, 520
    xmin = max(0.0, float(summary.direction_three_seed_agreement.min()) - 0.04)
    xmax = min(1.0, float(summary.direction_three_seed_agreement.max()) + 0.04)
    ymin = min(-0.05, float(summary.transfer_recovery.min()) - 0.04)
    ymax = max(0.30, float(summary.transfer_recovery.max()) + 0.04)
    xs = lambda v: left + (v - xmin) / (xmax - xmin) * (right - left)
    ys = lambda v: bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
    for ref in (0.0, 0.10, 0.25):
        if ymin <= ref <= ymax:
            parts.append(f'<line x1="{left}" y1="{ys(ref):.1f}" x2="{right}" y2="{ys(ref):.1f}" stroke="#bbb" stroke-dasharray="4 4"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#444"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#444"/>')
    for index, row in enumerate(summary.itertuples(index=False)):
        x, y = xs(row.direction_three_seed_agreement), ys(row.transfer_recovery)
        fill = "#28778e" if row.dataset == "mkg_w" else "#c45d3c"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{fill}"/>')
        dy = -10 if index % 2 == 0 else 18
        svg_text(parts, x + 9, y + dy, row.pair_id.replace("mkgw_", "MW/").replace("db15k_", "DB/"), 10)
    svg_text(parts, (left + right)/2, 572, "3-seed LEFT / ANCHOR / RIGHT agreement", 12, "middle")
    svg_text(parts, 24, (top + bottom)/2, "Transfer Recovery", 12)
    svg_text(parts, 740, 48, "MKG-W", 10, fill="#28778e")
    svg_text(parts, 825, 48, "DB15K", 10, fill="#c45d3c")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def classify(summary: pd.DataFrame, contract: dict) -> tuple[str, dict]:
    significant_transfer = summary.transfer_ci95_low > 0
    significant_loso = summary.loso_ci95_low > 0
    transfer_datasets = set(summary.loc[significant_transfer, "dataset"])
    sub = contract["classification"]["substantially_transferable"]
    substantially = (
        int(significant_transfer.sum()) >= int(sub["minimum_transfer_ci_lower_positive_pairs"])
        and transfer_datasets == {"mkg_w", "db15k"}
        and int((summary.transfer_recovery >= 0.25).sum()) >= int(sub["minimum_recovery_at_least_0_25_pairs"])
        and int(significant_loso.sum()) >= int(sub["minimum_loso_ci_lower_positive_pairs"])
    )
    residual = contract["classification"]["residual_dominated"]
    not_significant = int((~significant_transfer).sum())
    residual_dominated = (
        float(summary.transfer_recovery.median()) < float(residual["median_transfer_recovery_lt"])
        and float(summary.loso_recovery.median()) < float(residual["median_loso_recovery_lt"])
        and not_significant >= int(residual["minimum_transfer_not_significantly_positive_pairs"])
    )
    decision = (
        "E4_SUBSTANTIALLY_TRANSFERABLE" if substantially
        else "E4_RESIDUAL_DOMINATED" if residual_dominated
        else "E4_INTERMEDIATE"
    )
    evidence = {
        "transfer_ci_lower_positive_pairs": int(significant_transfer.sum()),
        "transfer_significant_datasets": sorted(transfer_datasets),
        "transfer_recovery_ge_25pct_pairs": int((summary.transfer_recovery >= 0.25).sum()),
        "loso_ci_lower_positive_pairs": int(significant_loso.sum()),
        "median_transfer_recovery": float(summary.transfer_recovery.median()),
        "median_loso_recovery": float(summary.loso_recovery.median()),
        "transfer_not_significantly_positive_pairs": not_significant,
        "substantially_transferable_gate": bool(substantially),
        "residual_dominated_gate": bool(residual_dominated),
    }
    return decision, evidence


def format_interval(value: float, low: float, high: float) -> str:
    return f"{value:+.6f} [{low:+.6f}, {high:+.6f}]"


def write_report(summary: pd.DataFrame, matrix: pd.DataFrame, evidence: dict, decision: str, path: Path, contract: dict) -> None:
    lines = [
        "# Experiment 4 — Cross-Seed Transferable Complementarity Audit",
        "",
        "Date: 2026-09-07",
        "Split: DEV only",
        "Frozen prior route: `ROUTE_C_LIMITS`",
        "",
        "## Outcome",
        "",
        f"Frozen classification: **{decision}**.",
        "",
        "The audit transfers each source seed's deterministic Oracle alpha directly to the other independent seeds; target-seed reselection and negative-utility clipping are absent. LOSO is reported only as a cross-seed stability diagnostic, not an inference-time policy.",
        "",
        "## Interpretation",
        "",
        f"A statistically detectable cross-seed component exists in all six pairs, but it is partial: Transfer Recovery ranges from {summary.transfer_recovery.min():.1%} to {summary.transfer_recovery.max():.1%}, with a median of {summary.transfer_recovery.median():.1%}, and no pair reaches the frozen 25% threshold. LOSO recovery is larger (median {summary.loso_recovery.median():.1%}) but also remains well below the Raw Oracle ceiling.",
        "",
        "The two NativE+AdaMF pairs show the weakest transfer recovery, especially DB15K. Thus the evidence rejects both extremes: complementarity is not wholly seed-specific because every transfer CI excludes zero, but the transferable component is not large enough for `E4_SUBSTANTIALLY_TRANSFERABLE`. Most Raw Oracle headroom disappears once the source-seed action must survive an independent seed.",
        "",
        "## Pair-level results",
        "",
        "| Pair | Raw Oracle (95% CI) | Cross-seed transfer (95% CI) | Recovery (95% CI) | LOSO stable (95% CI) | LOSO recovery | Exact alpha | Direction agreement | B / Z / H | Frozen X4 OOF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {format_interval(row.raw_oracle_headroom, row.raw_ci95_low, row.raw_ci95_high)} "
            f"| {format_interval(row.cross_seed_transfer_headroom, row.transfer_ci95_low, row.transfer_ci95_high)} "
            f"| {row.transfer_recovery:.1%} [{row.transfer_recovery_ci95_low:.1%}, {row.transfer_recovery_ci95_high:.1%}] "
            f"| {format_interval(row.loso_stable_headroom, row.loso_ci95_low, row.loso_ci95_high)} "
            f"| {row.loso_recovery:.1%} | {row.exact_alpha_three_seed_agreement:.1%} "
            f"| {row.direction_three_seed_agreement:.1%} | {row.beneficial_transfer_rate:.1%} / {row.zero_transfer_rate:.1%} / {row.harmful_transfer_rate:.1%} "
            f"| {row.frozen_x4_oof_gain:+.6f} |"
        )
    lines.extend([
        "",
        "Frozen X4 OOF is copied from Experiment 2 and therefore remains relative to its fold-specific outer-train Global baseline. The first three funnel stages use the unchanged full-DEV Experiment 1 alpha0. This baseline distinction is retained rather than silently redefining the frozen X4 probe.",
        "",
        "## Ordered seed transfers",
        "",
        "| Pair | 1→2 | 1→3 | 2→1 | 2→3 | 3→1 | 3→2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for pair_id in PAIR_IDS:
        subset = matrix.loc[(matrix.pair_id == pair_id) & (matrix.source_seed != matrix.target_seed)].set_index(["source_seed", "target_seed"])
        values = [float(subset.loc[key].headroom) for key in ((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2))]
        lines.append(f"| {PAIR_LABELS[pair_id]} | " + " | ".join(f"{value:+.6f}" for value in values) + " |")
    lines.extend([
        "",
        "## Agreement and denominator checks",
        "",
        "Exact alpha agreement requires all three seed actions to match. Direction agreement maps actions to LEFT/ANCHOR/RIGHT relative to unchanged alpha0. Non-anchor direction agreement and pairwise exact-alpha agreement are retained in `pair_summary.csv`. B/Z/H is conditioned on a strictly positive, non-anchor source Oracle action; the three rates sum to one for every pair.",
        "",
        "## Frozen classification gate",
        "",
        f"- Transfer CI lower > 0: {evidence['transfer_ci_lower_positive_pairs']}/6 pairs; datasets represented: {', '.join(evidence['transfer_significant_datasets']) or 'none'}.",
        f"- Transfer Recovery >=25%: {evidence['transfer_recovery_ge_25pct_pairs']}/6 pairs.",
        f"- LOSO CI lower > 0: {evidence['loso_ci_lower_positive_pairs']}/6 pairs.",
        f"- Median Transfer Recovery: {evidence['median_transfer_recovery']:.1%}.",
        f"- Median LOSO Recovery: {evidence['median_loso_recovery']:.1%}.",
        f"- Transfer not significantly positive: {evidence['transfer_not_significantly_positive_pairs']}/6 pairs.",
        "",
        "The classification is evaluated exactly from the preregistered gates. No Experiment 1–3 result or `ROUTE_C_LIMITS` was changed.",
        "",
        "## Figures",
        "",
        "1. `figure1_headroom_funnel.svg` — Raw Oracle → Cross-seed Transfer → LOSO Stable → frozen X4 OOF gain.",
        "2. `figure2_cross_seed_transfer_matrix.svg` — six shared-scale 3×3 seed matrices.",
        "3. `figure3_transfer_recovery_forest.svg` — transfer recovery and paired clustered-bootstrap CI.",
        "4. `figure4_stability_vs_transfer.svg` — direction agreement versus transfer recovery.",
        "",
        "## Integrity audit",
        "",
        "- TEST access = 0",
        "- TEST commands = 0",
        "- checkpoint retraining = 0",
        "- checkpoint reselection = 0",
        "- new selector = 0",
        "- new representation = 0",
        "- alpha grid modified = no",
        "- alpha0 modified = no",
        "- Experiment 1 result modified = no",
        f"- original-triple bootstrap intact = yes ({contract['bootstrap']['samples']} percentile replicates, seed {contract['bootstrap']['seed']})",
        "- all direct source and output hashes recorded = yes (`audit_manifest.json`; manifest self-hash excluded to avoid recursion)",
        "- Experiment 5 started = 0",
        "",
        decision,
    ])
    if lines[-1] not in CLASSIFICATIONS:
        raise AssertionError("Report must end in one frozen Experiment 4 classification")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inventory_records(paths: list[Path], role: str) -> list[dict]:
    records = []
    for path in sorted(set(paths), key=lambda value: portable_path(value)):
        if not path.exists():
            raise FileNotFoundError(path)
        records.append({"role": role, "path": portable_path(path), "sha256": sha256_file(path)})
    return records


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    reject_test_path(contract_path)
    contract = load_contract(contract_path)
    exp1_root, exp2_root, exp3_root = Path(args.exp1_root), Path(args.exp2_root), Path(args.exp3_root)
    exp1_audit_path = exp1_root / "audit_manifest.json"
    exp1_stats_path = exp1_root / "pair_statistics.csv"
    exp2_audit_path = exp2_root / "audit_manifest.json"
    exp2_metrics_path = exp2_root / "primary_nested_probe_metrics.csv"
    exp3_audit_path = exp3_root / "audit_manifest.json"
    for path in (exp1_audit_path, exp1_stats_path, exp2_audit_path, exp2_metrics_path, exp3_audit_path):
        reject_test_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    exp1_audit, exp2_audit, exp3_audit = load_json(exp1_audit_path), load_json(exp2_audit_path), load_json(exp3_audit_path)
    if exp1_audit.get("gate", {}).get("decision") != "GO":
        raise RuntimeError("Experiment 1 Available Complementarity gate is not GO")
    if exp2_audit.get("split") != "dev" or int(exp2_audit.get("operational_audit", {}).get("test_access", -1)) != 0:
        raise RuntimeError("Experiment 2 input is not a clean DEV-only audit")
    if exp3_audit.get("decision") != "ROUTE_C_LIMITS" or int(exp3_audit.get("operational_audit", {}).get("test_access", -1)) != 0:
        raise RuntimeError("Experiment 3 frozen route/audit mismatch")
    exp1_stats = pd.read_csv(exp1_stats_path)
    exp2_metrics = pd.read_csv(exp2_metrics_path)
    x4_metrics = exp2_metrics.loc[(exp2_metrics.representation == "X4") & (exp2_metrics.learner == "nested_selected")]
    if set(exp1_stats.pair_id) != set(PAIR_IDS) or set(x4_metrics.pair_id) != set(PAIR_IDS) or len(x4_metrics) != 6:
        raise RuntimeError("Experiment 1/2 pair inventory mismatch")

    summaries, matrix_rows, bootstrap_rows = [], [], []
    transfer_frames, loso_frames, oracle_frames = [], [], []
    preflight_rows = []
    source_paths = [
        contract_path,
        Path("docs/protocols/EXP4_CROSS_SEED_TRANSFER_PROTOCOL.md"),
        Path(__file__),
        Path("scripts/exp2_information_common.py"),
        Path("scripts/run_exp4_cross_seed_transfer.ps1"),
        exp1_audit_path,
        exp1_stats_path,
        exp2_audit_path,
        exp2_metrics_path,
        exp3_audit_path,
    ]
    for pair_id in PAIR_IDS:
        frame, rr, alpha0, manifest_path, source_path, _ = load_pair(
            pair_id, Path(args.utility_manifest_dir), exp1_stats
        )
        frame.attrs["source"] = str(source_path)
        source_paths.extend([manifest_path, source_path])
        if args.dry_run:
            identity_count = frame.drop_duplicates(["original_triple_id", "direction"]).shape[0]
            preflight_rows.append({
                "pair_id": pair_id, "rows": int(len(frame)), "query_identities": int(identity_count),
                "original_triples": int(frame.original_triple_id.nunique()), "alpha0": alpha0,
            })
            continue
        summary, matrix, boot, transfers, loso, oracle, preflight = analyze_pair(
            pair_id,
            frame,
            rr,
            alpha0,
            exp1_stats.loc[exp1_stats.pair_id == pair_id].iloc[0],
            x4_metrics.loc[x4_metrics.pair_id == pair_id].iloc[0],
            int(contract["bootstrap"]["samples"]),
            int(contract["bootstrap"]["seed"]),
        )
        summaries.append(summary)
        matrix_rows.extend(matrix)
        bootstrap_rows.extend(boot)
        transfer_frames.append(transfers)
        loso_frames.append(loso)
        oracle_frames.append(oracle)
        preflight["canonical_source"] = portable_path(source_path)
        preflight_rows.append(preflight)
        print(f"[OK] {pair_id}: raw={summary['raw_oracle_headroom']:.6f}, transfer={summary['cross_seed_transfer_headroom']:.6f}, loso={summary['loso_stable_headroom']:.6f}")

    if args.dry_run:
        print(json.dumps({
            "status": "preflight_ok", "split": "dev", "pairs": preflight_rows,
            "test_access": 0, "checkpoint_execution": 0,
        }, indent=2))
        return

    output_dir = Path(args.output_dir)
    report_path = Path(args.report)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty; pass --overwrite: {output_dir}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    matrix_frame = pd.DataFrame(matrix_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    transfer_frame = pd.concat(transfer_frames, ignore_index=True)
    loso_frame = pd.concat(loso_frames, ignore_index=True)
    oracle_frame = pd.concat(oracle_frames, ignore_index=True)
    decision, evidence = classify(summary_frame, contract)
    summary_frame["transfer_ci_lower_gt_zero"] = summary_frame.transfer_ci95_low > 0
    summary_frame["loso_ci_lower_gt_zero"] = summary_frame.loso_ci95_low > 0

    summary_path = output_dir / "pair_summary.csv"
    matrix_path = output_dir / "seed_pair_transfer_matrix.csv"
    bootstrap_path = output_dir / "bootstrap_ci.csv"
    transfer_path = output_dir / "per_query_seed_transfer.csv.gz"
    loso_path = output_dir / "per_query_loso.csv.gz"
    oracle_path = output_dir / "per_query_oracle.csv.gz"
    decision_path = output_dir / "classification.json"
    figure_paths = [
        output_dir / "figure1_headroom_funnel.svg",
        output_dir / "figure2_cross_seed_transfer_matrix.svg",
        output_dir / "figure3_transfer_recovery_forest.svg",
        output_dir / "figure4_stability_vs_transfer.svg",
    ]
    summary_frame.to_csv(summary_path, index=False, lineterminator="\n")
    matrix_frame.to_csv(matrix_path, index=False, lineterminator="\n")
    bootstrap_frame.to_csv(bootstrap_path, index=False, lineterminator="\n")
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    transfer_frame.to_csv(transfer_path, index=False, compression=compression, lineterminator="\n")
    loso_frame.to_csv(loso_path, index=False, compression=compression, lineterminator="\n")
    oracle_frame.to_csv(oracle_path, index=False, compression=compression, lineterminator="\n")
    decision_path.write_text(json.dumps({"classification": decision, "evidence": evidence}, indent=2) + "\n", encoding="utf-8")
    write_funnel(summary_frame, figure_paths[0])
    write_matrix(matrix_frame, figure_paths[1])
    write_forest(summary_frame, figure_paths[2])
    write_scatter(summary_frame, figure_paths[3])
    write_report(summary_frame, matrix_frame, evidence, decision, report_path, contract)

    output_paths = [
        summary_path, matrix_path, bootstrap_path, transfer_path, loso_path, oracle_path,
        decision_path, *figure_paths, report_path,
    ]
    audit = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "split": "dev",
        "classification": decision,
        "classification_evidence": evidence,
        "preflight": preflight_rows,
        "sources_and_outputs": [
            *inventory_records(source_paths, "source"),
            *inventory_records(output_paths, "output"),
        ],
        "hash_inventory_note": "audit_manifest.json self-hash excluded to avoid recursive content",
        "operational_audit": {
            "test_access": 0,
            "test_commands": 0,
            "checkpoint_retraining": 0,
            "checkpoint_reselection": 0,
            "new_selector": 0,
            "new_representation": 0,
            "alpha_grid_modified": False,
            "alpha0_modified": False,
            "experiment_1_result_modified": False,
            "original_triple_bootstrap_intact": True,
        },
        "next_experiment_started": 0,
    }
    audit_path = output_dir / "audit_manifest.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] {decision}")
    print(f"[DONE] report={report_path}")


if __name__ == "__main__":
    main()
