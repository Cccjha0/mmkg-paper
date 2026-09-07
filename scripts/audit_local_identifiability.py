from __future__ import annotations

import argparse
import hashlib
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
    representation_features,
    sha256_file,
)


K_VALUES = (5, 10, 20, 50)
PAIR_LABELS = {
    "mkgw_mhyper_native": "MKG-W / M-Hyper+NativE",
    "mkgw_mhyper_adamf": "MKG-W / M-Hyper+AdaMF",
    "mkgw_native_adamf": "MKG-W / NativE+AdaMF",
    "db15k_mhyper_native": "DB15K / M-Hyper+NativE",
    "db15k_mhyper_adamf": "DB15K / M-Hyper+AdaMF",
    "db15k_native_adamf": "DB15K / NativE+AdaMF",
}
SUMMARY_METRICS = (
    "exact_action_agreement",
    "direction_agreement",
    "neighbor_internal_purity",
    "normalized_action_entropy",
    "beneficial_set_jaccard",
    "both_empty_rate",
    "neighbor_consensus_utility",
    "matched_exact_action_agreement",
    "matched_direction_agreement",
    "matched_neighbor_internal_purity",
    "matched_normalized_action_entropy",
    "matched_beneficial_set_jaccard",
    "matched_both_empty_rate",
    "matched_consensus_utility",
    "exact_action_agreement_lift",
    "direction_agreement_lift",
    "purity_lift",
    "jaccard_lift",
    "consensus_utility_lift",
    "matched_baseline_supported",
)
FINAL_CLASSES = (
    "E5_LOCAL_IDENTIFIABLE",
    "E5_X6_RECOVERY",
    "E5_LOCAL_AMBIGUITY",
    "E5_INTERMEDIATE",
)
POPCOUNT8 = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 5 DEV-only local-identifiability audit")
    parser.add_argument("--mode", choices=("preflight", "systematic", "analyze"), required=True)
    parser.add_argument("--contract", default="docs/protocols/EXP5_LOCAL_IDENTIFIABILITY_CONTRACT.json")
    parser.add_argument("--exp1-root", default="outputs/complementarity_identifiability/exp1_landscape")
    parser.add_argument("--exp2-root", default="outputs/complementarity_identifiability/exp2_information")
    parser.add_argument("--exp4-root", default="outputs/complementarity_identifiability/exp4_cross_seed_transfer")
    parser.add_argument("--utility-manifest-dir", default="outputs/aacpi/utility_tables")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/exp5_local_identifiability")
    parser.add_argument("--report", default="docs/reports/local_identifiability_action_ambiguity_audit_2026-09-07.md")
    parser.add_argument("--pair-id", choices=PAIR_IDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--center-batch-size", type=int, default=512)
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
        raise RuntimeError("Experiment 5 contract is not frozen DEV-only")
    if tuple(contract.get("pair_ids", [])) != PAIR_IDS:
        raise RuntimeError("Experiment 5 pair inventory changed")
    if tuple(contract.get("k_values", [])) != K_VALUES:
        raise RuntimeError("Experiment 5 k ladder changed")
    if not np.allclose(contract.get("alpha_grid", []), ALPHAS):
        raise RuntimeError("Experiment 5 alpha grid changed")
    if contract["representations"]["X6"]["status"] != "X6_FIXED_VECTOR_UNAVAILABLE":
        raise RuntimeError("Experiment 5 X6 availability decision changed")
    prohibited = (
        "test_access", "new_representation", "supervised_metric_learning", "new_pca",
        "new_embedding", "new_selector", "neighbor_policy_development", "checkpoint_execution",
    )
    if any(int(contract.get(field, -1)) != 0 for field in prohibited):
        raise RuntimeError("Experiment 5 prohibited-operation boundary changed")
    return contract


def original_triple_ids(frame: pd.DataFrame) -> np.ndarray:
    return (
        "h=" + frame.head_id.astype(str) + "|r=" + frame.relation_id.astype(str) + "|t=" + frame.tail_id.astype(str)
    ).to_numpy(str)


def action_indices(values: np.ndarray, global_index: int) -> np.ndarray:
    maxima = values.max(axis=-1, keepdims=True)
    tied = np.isclose(values, maxima, rtol=0.0, atol=ZERO_TOLERANCE)
    preference = np.abs(ALPHAS - ALPHAS[global_index]) + ALPHAS * 1e-9
    return np.where(tied, preference, np.inf).argmin(axis=-1).astype(np.int16)


def encode_positive_sets(utility: np.ndarray) -> np.ndarray:
    powers = np.left_shift(np.uint32(1), np.arange(len(ALPHAS), dtype=np.uint32))
    return ((utility > ZERO_TOLERANCE).astype(np.uint32) * powers[None, :]).sum(axis=1, dtype=np.uint32)


def popcount32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32)
    return (
        POPCOUNT8[values & np.uint32(255)]
        + POPCOUNT8[(values >> np.uint32(8)) & np.uint32(255)]
        + POPCOUNT8[(values >> np.uint32(16)) & np.uint32(255)]
        + POPCOUNT8[(values >> np.uint32(24)) & np.uint32(255)]
    )


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def align_indices(source_ids: np.ndarray, target_ids: np.ndarray, label: str) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(source_ids)}
    if len(lookup) != len(source_ids) or len(set(map(str, target_ids))) != len(target_ids):
        raise RuntimeError(f"Duplicate query id in {label}")
    if set(lookup) != set(map(str, target_ids)):
        raise RuntimeError(f"Query inventory mismatch in {label}")
    return np.asarray([lookup[str(value)] for value in target_ids], dtype=np.int64)


def exp2_hash_map(audit: dict) -> dict[str, str]:
    return {
        str(row["path"]).replace("\\", "/"): str(row["sha256"])
        for row in audit.get("sources_and_outputs", [])
        if isinstance(row, dict) and row.get("path") and row.get("sha256")
    }


def verify_declared(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    if sha256_file(path) != expected:
        raise RuntimeError(f"{label} hash mismatch: {path}")


def load_pair_assets(
    pair_id: str,
    contract: dict,
    exp1_stats: pd.DataFrame,
    exp2_audit: dict,
    manifest_dir: Path,
    exp2_root: Path,
) -> tuple[dict, list[Path]]:
    manifest_path = manifest_dir / f"{pair_id}_dev_source_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("pair_id") != pair_id or manifest.get("split") != "dev":
        raise RuntimeError(f"Experiment 1 source manifest mismatch: {pair_id}")
    if manifest.get("score_normalization") != "query_zscore" or not np.allclose(manifest.get("global_alpha_grid", []), ALPHAS):
        raise RuntimeError(f"Experiment 1 source protocol mismatch: {pair_id}")
    query_path = Path(manifest["source_query_rows"]["path"])
    reject_test_path(query_path)
    verify_declared(query_path, manifest["source_query_rows"]["sha256"], "canonical DEV query rows")
    usecols = [
        "dataset", "pair_name", "query_id", "split", "seed", "direction", "head_id",
        "relation_id", "tail_id", "alpha_global", *RR_COLUMNS,
    ]
    frame = pd.read_csv(query_path, usecols=usecols)
    expected_dataset = "mkg_w" if pair_id.startswith("mkgw") else "db15k"
    if set(frame.split.astype(str)) != {"dev"} or set(frame.pair_name.astype(str)) != {pair_id} or set(frame.dataset.astype(str)) != {expected_dataset}:
        raise RuntimeError(f"Non-DEV or wrong pair identity in {query_path}")
    if frame.query_id.duplicated().any() or frame[RR_COLUMNS].isna().any().any():
        raise RuntimeError(f"Incomplete canonical query rows: {pair_id}")
    frame["original_triple_id"] = original_triple_ids(frame)
    if frame.groupby("original_triple_id").size().ne(6).any():
        raise RuntimeError(f"Incomplete seed-direction triple groups: {pair_id}")
    alpha0 = float(manifest["global_alpha"])
    grid_match = np.flatnonzero(np.isclose(ALPHAS, alpha0, rtol=0.0, atol=ZERO_TOLERANCE))
    if len(grid_match) != 1 or frame.alpha_global.nunique() != 1 or not np.isclose(frame.alpha_global.iloc[0], alpha0):
        raise RuntimeError(f"Inconsistent alpha0: {pair_id}")
    stat = exp1_stats.loc[exp1_stats.pair_id == pair_id]
    if len(stat) != 1 or not np.isclose(stat.iloc[0].global_alpha, alpha0):
        raise RuntimeError(f"Experiment 1 alpha0 mismatch: {pair_id}")

    asset_path = exp2_root / "assets" / f"{pair_id}_query_information.npz"
    asset_manifest_path = exp2_root / "assets" / f"{pair_id}_query_information_manifest.json"
    asset_manifest = load_json(asset_manifest_path)
    verify_declared(asset_path, asset_manifest["output"]["sha256"], "Experiment 2 X4 asset")
    expected_fields = representation_features(contract=load_json(Path("docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json")))["X4"]
    if asset_manifest.get("feature_fields_x4") != expected_fields or len(expected_fields) != 40:
        raise RuntimeError(f"Experiment 2 X4 feature contract mismatch: {pair_id}")
    with np.load(asset_path, allow_pickle=False) as asset:
        order = align_indices(asset["query_id"], frame.query_id.to_numpy(str), f"{pair_id} X4")
        features = asset["features_x4"][order].astype(np.float32)
        for field, column in (("seed", "seed"), ("direction", "direction"), ("relation_id", "relation_id")):
            if not np.array_equal(asset[field][order].astype(str), frame[column].to_numpy().astype(str)):
                raise RuntimeError(f"Experiment 2 X4 {field} alignment mismatch: {pair_id}")
    if features.shape != (len(frame), 40) or not np.isfinite(features).all():
        raise RuntimeError(f"Invalid X4 feature matrix: {pair_id}")

    prediction_path = exp2_root / "runs" / pair_id / "x4" / "nested_selected" / "oof_action_predictions.npz"
    hash_map = exp2_hash_map(exp2_audit)
    portable_prediction = portable_path(prediction_path)
    expected_hash = hash_map.get(portable_prediction)
    if expected_hash is None:
        raise RuntimeError(f"Experiment 2 audit does not declare fold source: {portable_prediction}")
    verify_declared(prediction_path, expected_hash, "Experiment 2 outer-fold source")
    with np.load(prediction_path, allow_pickle=False) as prediction:
        order = align_indices(prediction["query_id"], frame.query_id.to_numpy(str), f"{pair_id} fold")
        stored_folds = prediction["outer_fold"][order].astype(np.int16)
        folds = stored_folds - 1
    if set(folds.tolist()) != {0, 1, 2, 3, 4}:
        raise RuntimeError(f"Incomplete Experiment 2 folds: {pair_id}")
    fold_counts = frame.assign(_fold=folds).groupby("original_triple_id")._fold.nunique()
    if fold_counts.max() != 1:
        raise RuntimeError(f"Experiment 2 original triple fold leakage: {pair_id}")

    rr = frame[RR_COLUMNS].to_numpy(np.float64)
    global_index = int(grid_match[0])
    utility = rr - rr[:, [global_index]]
    oracle_index = action_indices(rr, global_index)
    direction_label = np.sign(oracle_index.astype(np.int16) - global_index).astype(np.int8) + 1
    positive_mask = encode_positive_sets(utility)
    data = {
        "frame": frame,
        "features": features,
        "folds": folds,
        "rr": rr,
        "utility": utility,
        "oracle_index": oracle_index,
        "direction_label": direction_label,
        "positive_mask": positive_mask,
        "global_index": global_index,
        "alpha0": alpha0,
        "dataset": expected_dataset,
    }
    sources = [manifest_path, query_path, asset_manifest_path, asset_path, prediction_path]
    return data, sources


def exact_knn(
    reference_features: np.ndarray,
    center_features: np.ndarray,
    reference_global_indices: np.ndarray,
    k: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if device == "cpu":
        reference_norm = (reference_features.astype(np.float64) ** 2).sum(axis=1)
        tie = np.arange(len(reference_features), dtype=np.float64) * 1e-15
        neighbor_indices = np.empty((len(center_features), k), dtype=np.int64)
        neighbor_distances = np.empty((len(center_features), k), dtype=np.float32)
        for start in range(0, len(center_features), batch_size):
            stop = min(start + batch_size, len(center_features))
            center = center_features[start:stop].astype(np.float64)
            distance_sq = (center * center).sum(axis=1, keepdims=True) + reference_norm[None, :] - 2.0 * center @ reference_features.astype(np.float64).T
            np.maximum(distance_sq, 0.0, out=distance_sq)
            key = distance_sq + tie[None, :]
            candidates = np.argpartition(key, kth=k - 1, axis=1)[:, :k]
            candidate_keys = np.take_along_axis(key, candidates, axis=1)
            order = np.argsort(candidate_keys, axis=1, kind="stable")
            local = np.take_along_axis(candidates, order, axis=1)
            neighbor_indices[start:stop] = reference_global_indices[local]
            neighbor_distances[start:stop] = np.sqrt(np.take_along_axis(distance_sq, local, axis=1)).astype(np.float32)
        return neighbor_indices, neighbor_distances

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    torch_device = torch.device(device)
    reference = torch.as_tensor(reference_features, dtype=torch.float32, device=torch_device)
    reference_norm = (reference * reference).sum(dim=1)
    tie = torch.arange(len(reference), dtype=torch.float64, device=torch_device) * 1e-15
    neighbor_indices = np.empty((len(center_features), k), dtype=np.int64)
    neighbor_distances = np.empty((len(center_features), k), dtype=np.float32)
    for start in range(0, len(center_features), batch_size):
        stop = min(start + batch_size, len(center_features))
        center = torch.as_tensor(center_features[start:stop], dtype=torch.float32, device=torch_device)
        distance_sq = (center * center).sum(dim=1, keepdim=True) + reference_norm[None, :] - 2.0 * center @ reference.T
        distance_sq.clamp_(min=0.0)
        key = distance_sq.to(torch.float64) + tie[None, :]
        _, local = torch.topk(key, k=k, dim=1, largest=False, sorted=True)
        local_np = local.cpu().numpy()
        distance_np = torch.gather(distance_sq, 1, local).sqrt().cpu().numpy().astype(np.float32)
        neighbor_indices[start:stop] = reference_global_indices[local_np]
        neighbor_distances[start:stop] = distance_np
        del center, distance_sq, key, local
    del reference, reference_norm, tie
    if device == "cuda":
        torch.cuda.empty_cache()
    return neighbor_indices, neighbor_distances


def jaccard_per_center(center_masks: np.ndarray, neighbor_masks: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(len(center_masks), np.nan, dtype=np.float64)
    both_empty = np.empty(len(center_masks), dtype=np.float64)
    for start in range(0, len(center_masks), batch_size):
        stop = min(start + batch_size, len(center_masks))
        center = center_masks[start:stop, None]
        neighbors = neighbor_masks[start:stop]
        union = np.bitwise_or(center, neighbors)
        valid = union != 0
        intersection_count = popcount32(np.bitwise_and(center, neighbors)).astype(np.float64)
        union_count = popcount32(union).astype(np.float64)
        scores = np.divide(intersection_count, union_count, out=np.zeros_like(intersection_count), where=valid)
        counts = valid.sum(axis=1)
        result[start:stop] = np.divide(scores.sum(axis=1), counts, out=np.full(len(counts), np.nan), where=counts > 0)
        both_empty[start:stop] = ((center == 0) & (neighbors == 0)).mean(axis=1)
    return result, both_empty


def jaccard_against_multiset(center_masks: np.ndarray, sampled_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_centers, center_inverse = np.unique(center_masks.astype(np.uint32), return_inverse=True)
    unique_neighbors, frequencies = np.unique(sampled_masks.astype(np.uint32), return_counts=True)
    center = unique_centers[:, None]
    neighbor = unique_neighbors[None, :]
    union = np.bitwise_or(center, neighbor)
    valid = union != 0
    scores = np.divide(
        popcount32(np.bitwise_and(center, neighbor)).astype(np.float64),
        popcount32(union).astype(np.float64),
        out=np.zeros(union.shape, dtype=np.float64),
        where=valid,
    )
    weighted_valid = valid * frequencies[None, :]
    denominators = weighted_valid.sum(axis=1)
    means = np.divide(
        (scores * frequencies[None, :]).sum(axis=1),
        denominators,
        out=np.full(len(unique_centers), np.nan),
        where=denominators > 0,
    )
    zero_frequency = int(frequencies[unique_neighbors == 0][0]) if np.any(unique_neighbors == 0) else 0
    empty_rates = np.where(unique_centers == 0, zero_frequency / len(sampled_masks), 0.0)
    return means[center_inverse], empty_rates[center_inverse]


def metrics_for_neighbors(data: dict, center_indices: np.ndarray, neighbor_indices: np.ndarray) -> dict[str, np.ndarray]:
    center_action = data["oracle_index"][center_indices]
    neighbor_action = data["oracle_index"][neighbor_indices]
    center_direction = data["direction_label"][center_indices]
    neighbor_direction = data["direction_label"][neighbor_indices]
    exact = (neighbor_action == center_action[:, None]).mean(axis=1)
    direction = (neighbor_direction == center_direction[:, None]).mean(axis=1)
    counts = np.stack([(neighbor_direction == label).sum(axis=1) for label in (0, 1, 2)], axis=1)
    probabilities = counts / neighbor_indices.shape[1]
    purity = probabilities.max(axis=1)
    entropy = -(np.where(probabilities > 0, probabilities * np.log(np.maximum(probabilities, 1e-300)), 0.0).sum(axis=1)) / math.log(3)
    neighbor_masks = data["positive_mask"][neighbor_indices]
    jaccard, both_empty = jaccard_per_center(data["positive_mask"][center_indices], neighbor_masks)
    mean_utility = data["utility"][neighbor_indices].mean(axis=1)
    consensus_index = action_indices(mean_utility, data["global_index"])
    consensus_utility = data["utility"][center_indices, consensus_index]
    return {
        "exact_action_agreement": exact,
        "direction_agreement": direction,
        "neighbor_internal_purity": purity,
        "normalized_action_entropy": entropy,
        "beneficial_set_jaccard": jaccard,
        "beneficial_set_jaccard_valid_pairs": neighbor_indices.shape[1] * (1.0 - both_empty),
        "both_empty_rate": both_empty,
        "neighbor_consensus_action_index": consensus_index,
        "neighbor_consensus_utility": consensus_utility,
    }


def matched_random_metrics(
    data: dict,
    center_indices: np.ndarray,
    reference_indices: np.ndarray,
    k_values: tuple[int, ...],
    draws: int,
    base_seed: int,
    fold: int,
) -> dict[int, dict[str, np.ndarray]]:
    frame = data["frame"]
    n_center = len(center_indices)
    fields = (
        "matched_exact_action_agreement",
        "matched_direction_agreement",
        "matched_neighbor_internal_purity",
        "matched_normalized_action_entropy",
        "matched_beneficial_set_jaccard",
        "matched_both_empty_rate",
        "matched_consensus_utility",
    )
    output = {
        k: {field: np.full(n_center, np.nan, dtype=np.float64) for field in fields}
        for k in k_values
    }
    for k in k_values:
        output[k]["matched_baseline_supported"] = np.zeros(n_center, dtype=np.float64)
        output[k]["matched_beneficial_set_jaccard_valid_pairs"] = np.zeros(n_center, dtype=np.float64)

    center_local = {int(global_index): local for local, global_index in enumerate(center_indices)}
    reference_groups: dict[tuple[int, str, int], list[int]] = {}
    for global_index in reference_indices:
        row = frame.iloc[int(global_index)]
        key = (int(row.seed), str(row.direction), int(row.relation_id))
        reference_groups.setdefault(key, []).append(int(global_index))
    center_groups: dict[tuple[int, str, int], list[int]] = {}
    for global_index in center_indices:
        row = frame.iloc[int(global_index)]
        key = (int(row.seed), str(row.direction), int(row.relation_id))
        center_groups.setdefault(key, []).append(int(global_index))

    for key, group_centers_global in center_groups.items():
        pool = np.asarray(reference_groups.get(key, []), dtype=np.int64)
        supported_k = [k for k in k_values if len(pool) >= k]
        if not supported_k:
            continue
        max_k = max(supported_k)
        rng = np.random.default_rng(stable_seed(base_seed, f"fold={fold}|seed={key[0]}|direction={key[1]}|relation={key[2]}"))
        sampled = np.stack([rng.choice(pool, size=max_k, replace=False) for _ in range(draws)], axis=0)
        group_centers = np.asarray(group_centers_global, dtype=np.int64)
        local_indices = np.asarray([center_local[int(index)] for index in group_centers], dtype=np.int64)
        center_actions = data["oracle_index"][group_centers]
        center_directions = data["direction_label"][group_centers]
        center_masks = data["positive_mask"][group_centers]
        center_utility = data["utility"][group_centers]
        for k in supported_k:
            selected = sampled[:, :k]
            actions = data["oracle_index"][selected]
            directions = data["direction_label"][selected]
            masks = data["positive_mask"][selected]
            action_frequency = np.bincount(actions.reshape(-1), minlength=len(ALPHAS)) / (draws * k)
            direction_frequency = np.bincount(directions.reshape(-1), minlength=3) / (draws * k)
            output[k]["matched_exact_action_agreement"][local_indices] = action_frequency[center_actions]
            output[k]["matched_direction_agreement"][local_indices] = direction_frequency[center_directions]
            counts = np.stack([(directions == label).sum(axis=1) for label in (0, 1, 2)], axis=1)
            probabilities = counts / k
            output[k]["matched_neighbor_internal_purity"][local_indices] = probabilities.max(axis=1).mean()
            draw_entropy = -(
                np.where(probabilities > 0, probabilities * np.log(np.maximum(probabilities, 1e-300)), 0.0).sum(axis=1)
            ) / math.log(3)
            output[k]["matched_normalized_action_entropy"][local_indices] = draw_entropy.mean()
            jaccard, both_empty = jaccard_against_multiset(center_masks, masks.reshape(-1))
            output[k]["matched_beneficial_set_jaccard"][local_indices] = jaccard
            output[k]["matched_beneficial_set_jaccard_valid_pairs"][local_indices] = draws * k * (1.0 - both_empty)
            output[k]["matched_both_empty_rate"][local_indices] = both_empty
            mean_utility = data["utility"][selected].mean(axis=1)
            consensus_actions = action_indices(mean_utility, data["global_index"])
            output[k]["matched_consensus_utility"][local_indices] = center_utility[:, consensus_actions].mean(axis=1)
            output[k]["matched_baseline_supported"][local_indices] = 1.0
    return output


def make_center_rows(
    data: dict,
    center_indices: np.ndarray,
    fold: int,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    matched: dict[int, dict[str, np.ndarray]],
    pair_id: str,
) -> pd.DataFrame:
    frame = data["frame"]
    records = []
    for k in K_VALUES:
        metrics = metrics_for_neighbors(data, center_indices, neighbor_indices[:, :k])
        row = pd.DataFrame({
            "dataset": data["dataset"],
            "pair_id": pair_id,
            "representation": "X4",
            "center_row_index": center_indices,
            "center_query_id": frame.query_id.to_numpy(str)[center_indices],
            "center_original_triple_id": frame.original_triple_id.to_numpy(str)[center_indices],
            "center_seed": frame.seed.to_numpy(np.int16)[center_indices],
            "center_direction": frame.direction.to_numpy(str)[center_indices],
            "center_relation_id": frame.relation_id.to_numpy(np.int64)[center_indices],
            "outer_fold": fold,
            "k": k,
            "mean_neighbor_distance": neighbor_distances[:, :k].mean(axis=1),
            "max_neighbor_distance": neighbor_distances[:, :k].max(axis=1),
            "center_oracle_alpha": ALPHAS[data["oracle_index"][center_indices]],
            "center_action_direction": data["direction_label"][center_indices] - 1,
            "center_action_direction_label": np.asarray(["LEFT", "ANCHOR", "RIGHT"])[data["direction_label"][center_indices]],
            "center_positive_alpha_mask": data["positive_mask"][center_indices],
            **metrics,
            **matched[k],
        })
        row["neighbor_consensus_alpha"] = ALPHAS[row.neighbor_consensus_action_index.to_numpy(np.int16)]
        row["exact_action_agreement_lift"] = row.exact_action_agreement - row.matched_exact_action_agreement
        row["direction_agreement_lift"] = row.direction_agreement - row.matched_direction_agreement
        row["purity_lift"] = row.neighbor_internal_purity - row.matched_neighbor_internal_purity
        row["jaccard_lift"] = row.beneficial_set_jaccard - row.matched_beneficial_set_jaccard
        row["consensus_utility_lift"] = row.neighbor_consensus_utility - row.matched_consensus_utility
        records.append(row)
    return pd.concat(records, ignore_index=True)


def distance_bins(data: dict, neighbor_indices: np.ndarray, distances: np.ndarray, center_rows: pd.DataFrame, pair_id: str) -> pd.DataFrame:
    n_center, k = neighbor_indices.shape
    center_index = np.arange(n_center, dtype=np.int64)
    center_global = center_rows.loc[center_rows.k == max(K_VALUES)].sort_values("center_row_index").center_row_index.to_numpy(np.int64)
    if len(center_global) != n_center:
        raise RuntimeError(f"Distance-bin center alignment failed: {pair_id}")
    center_action = np.repeat(data["oracle_index"][center_global], k)
    center_direction = np.repeat(data["direction_label"][center_global], k)
    center_mask = np.repeat(data["positive_mask"][center_global], k)
    neighbor_flat = neighbor_indices.reshape(-1)
    distance_flat = distances.reshape(-1).astype(np.float64)
    exact = (center_action == data["oracle_index"][neighbor_flat]).astype(np.float64)
    direction = (center_direction == data["direction_label"][neighbor_flat]).astype(np.float64)
    union = np.bitwise_or(center_mask, data["positive_mask"][neighbor_flat])
    valid = union != 0
    jaccard = np.full(len(union), np.nan, dtype=np.float64)
    jaccard[valid] = popcount32(np.bitwise_and(center_mask[valid], data["positive_mask"][neighbor_flat[valid]])) / popcount32(union[valid])
    quantiles = np.quantile(distance_flat, np.linspace(0, 1, 11))
    bins = np.searchsorted(quantiles[1:-1], distance_flat, side="right") + 1
    k50 = center_rows.loc[center_rows.k == max(K_VALUES)]
    matched_direction = float(k50.matched_direction_agreement.mean(skipna=True))
    matched_exact = float(k50.matched_exact_action_agreement.mean(skipna=True))
    matched_valid = k50.matched_beneficial_set_jaccard_valid_pairs.to_numpy(np.float64)
    matched_values = k50.matched_beneficial_set_jaccard.to_numpy(np.float64)
    matched_mask = np.isfinite(matched_values) & (matched_valid > 0)
    matched_jaccard = float(np.average(matched_values[matched_mask], weights=matched_valid[matched_mask]))
    records = []
    for bin_index in range(1, 11):
        mask = bins == bin_index
        records.append({
            "dataset": data["dataset"], "pair_id": pair_id, "representation": "X4",
            "distance_decile": bin_index, "neighbor_pairs": int(mask.sum()),
            "distance_low": float(quantiles[bin_index - 1]), "distance_high": float(quantiles[bin_index]),
            "mean_distance": float(distance_flat[mask].mean()),
            "direction_agreement": float(direction[mask].mean()),
            "exact_action_agreement": float(exact[mask].mean()),
            "beneficial_set_jaccard": float(np.nanmean(jaccard[mask])),
            "both_empty_rate": float((~valid[mask]).mean()),
            "matched_direction_agreement_reference": matched_direction,
            "matched_exact_action_agreement_reference": matched_exact,
            "matched_beneficial_set_jaccard_reference": matched_jaccard,
        })
    return pd.DataFrame(records)


def clustered_bootstrap_wide(center_rows: pd.DataFrame, samples: int, seed: int, pair_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_by_k = {
        k: center_rows.loc[center_rows.k == k].sort_values("center_row_index").reset_index(drop=True)
        for k in K_VALUES
    }
    base = sorted_by_k[K_VALUES[0]]
    for k, frame in sorted_by_k.items():
        if not np.array_equal(base.center_row_index, frame.center_row_index):
            raise RuntimeError(f"Bootstrap center alignment failed: {pair_id}, k={k}")
    columns = [(k, metric) for k in K_VALUES for metric in SUMMARY_METRICS]
    values = np.column_stack([sorted_by_k[k][metric].to_numpy(np.float64) for k, metric in columns])
    groups = base.center_original_triple_id.to_numpy(str)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(unique_groups), len(columns)), dtype=np.float64)
    counts = np.zeros((len(unique_groups), len(columns)), dtype=np.float64)
    for column, (k, metric) in enumerate(columns):
        if metric == "beneficial_set_jaccard":
            observation_weight = sorted_by_k[k].beneficial_set_jaccard_valid_pairs.to_numpy(np.float64)
        elif metric == "matched_beneficial_set_jaccard":
            observation_weight = sorted_by_k[k].matched_beneficial_set_jaccard_valid_pairs.to_numpy(np.float64)
        else:
            observation_weight = np.ones(len(values), dtype=np.float64)
        valid = np.isfinite(values[:, column]) & (observation_weight > 0)
        np.add.at(sums[:, column], inverse[valid], values[valid, column] * observation_weight[valid])
        np.add.at(counts[:, column], inverse[valid], observation_weight[valid])
    rng = np.random.default_rng(seed)
    boot = np.empty((samples, len(columns)), dtype=np.float64)
    probabilities = np.full(len(unique_groups), 1.0 / len(unique_groups), dtype=np.float64)
    for start in range(0, samples, 16):
        stop = min(start + 16, samples)
        weights = rng.multinomial(len(unique_groups), probabilities, size=stop - start).astype(np.float64)
        numerator = weights @ sums
        denominator = weights @ counts
        boot[start:stop] = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    point_estimates = np.divide(sums.sum(axis=0), counts.sum(axis=0), out=np.full(len(columns), np.nan), where=counts.sum(axis=0) > 0)
    column_lookup = {key: index for index, key in enumerate(columns)}
    for k in K_VALUES:
        lift_column = column_lookup[(k, "jaccard_lift")]
        raw_column = column_lookup[(k, "beneficial_set_jaccard")]
        matched_column = column_lookup[(k, "matched_beneficial_set_jaccard")]
        boot[:, lift_column] = boot[:, raw_column] - boot[:, matched_column]
        point_estimates[lift_column] = point_estimates[raw_column] - point_estimates[matched_column]
    lows, highs = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    summary_records, bootstrap_records = [], []
    for column, (k, metric) in enumerate(columns):
        estimate = float(point_estimates[column])
        record = {
            "dataset": base.dataset.iloc[0], "pair_id": pair_id, "representation": "X4", "k": k,
            "metric": metric, "estimate": estimate, "ci95_low": float(lows[column]), "ci95_high": float(highs[column]),
            "bootstrap_samples": samples, "bootstrap_unit": "center original_triple_id",
        }
        bootstrap_records.append(record)
    for k in K_VALUES:
        subset = [row for row in bootstrap_records if row["k"] == k]
        wide = {"dataset": base.dataset.iloc[0], "pair_id": pair_id, "representation": "X4", "k": k}
        for row in subset:
            wide[row["metric"]] = row["estimate"]
            wide[f"{row['metric']}_ci95_low"] = row["ci95_low"]
            wide[f"{row['metric']}_ci95_high"] = row["ci95_high"]
        summary_records.append(wide)
    return pd.DataFrame(summary_records), pd.DataFrame(bootstrap_records)


def matched_coverage_preflight(data: dict) -> dict[str, float]:
    frame, folds = data["frame"], data["folds"]
    supported = {k: 0 for k in K_VALUES}
    for fold in range(5):
        train = frame.loc[folds != fold].groupby(["seed", "direction", "relation_id"]).size()
        center = frame.loc[folds == fold]
        keys = pd.MultiIndex.from_frame(center[["seed", "direction", "relation_id"]])
        sizes = train.reindex(keys, fill_value=0).to_numpy()
        for k in K_VALUES:
            supported[k] += int((sizes >= k).sum())
    return {f"matched_coverage_k{k}": supported[k] / len(frame) for k in K_VALUES}


def process_pair(
    pair_id: str,
    data: dict,
    source_paths: list[Path],
    output_dir: Path,
    contract: dict,
    device: str,
    batch_size: int,
    overwrite: bool,
) -> None:
    work_dir = output_dir / "work" / pair_id
    neighbor_dir = output_dir / "neighbors"
    work_dir.mkdir(parents=True, exist_ok=True)
    neighbor_dir.mkdir(parents=True, exist_ok=True)
    center_path = work_dir / "per_center_x4.csv.gz"
    summary_path = work_dir / "pair_k_summary.csv"
    bootstrap_path = work_dir / "bootstrap_ci.csv"
    bins_path = work_dir / "distance_purity_bins.csv"
    neighbor_path = neighbor_dir / f"{pair_id}_x4_neighbors.npz"
    manifest_path = work_dir / "systematic_manifest.json"
    targets = (center_path, summary_path, bootstrap_path, bins_path, neighbor_path, manifest_path)
    if any(path.exists() for path in targets) and not overwrite:
        raise FileExistsError(f"Experiment 5 pair outputs exist; pass --overwrite: {pair_id}")

    frame, features, folds = data["frame"], data["features"], data["folds"]
    all_neighbors = np.empty((len(frame), max(K_VALUES)), dtype=np.int64)
    all_distances = np.empty((len(frame), max(K_VALUES)), dtype=np.float32)
    center_parts = []
    for fold in range(5):
        center_indices = np.flatnonzero(folds == fold)
        reference_indices = np.flatnonzero(folds != fold)
        center_groups = set(frame.original_triple_id.iloc[center_indices])
        reference_groups = set(frame.original_triple_id.iloc[reference_indices])
        if center_groups & reference_groups:
            raise RuntimeError(f"Outer-triple leakage before kNN: {pair_id}, fold={fold}")
        mean = features[reference_indices].mean(axis=0, dtype=np.float64)
        std = features[reference_indices].std(axis=0, dtype=np.float64)
        std[std < 1e-12] = 1.0
        reference_scaled = ((features[reference_indices] - mean) / std).astype(np.float32)
        center_scaled = ((features[center_indices] - mean) / std).astype(np.float32)
        neighbors, distances = exact_knn(
            reference_scaled, center_scaled, reference_indices, max(K_VALUES), device, batch_size
        )
        leaked = frame.original_triple_id.to_numpy(str)[neighbors] == frame.original_triple_id.to_numpy(str)[center_indices, None]
        if leaked.any():
            raise RuntimeError(f"Same original triple entered neighbor set: {pair_id}, fold={fold}")
        all_neighbors[center_indices] = neighbors
        all_distances[center_indices] = distances
        matched = matched_random_metrics(
            data, center_indices, reference_indices, K_VALUES,
            int(contract["matched_random"]["draws"]), int(contract["matched_random"]["seed"]), fold,
        )
        center_parts.append(make_center_rows(data, center_indices, fold, neighbors, distances, matched, pair_id))
        print(f"[OK] {pair_id} fold {fold}: centers={len(center_indices)}, reference={len(reference_indices)}")

    center_rows = pd.concat(center_parts, ignore_index=True)
    pair_summary, bootstrap = clustered_bootstrap_wide(
        center_rows, int(contract["bootstrap"]["samples"]), int(contract["bootstrap"]["seed"]), pair_id
    )
    bins = distance_bins(data, all_neighbors, all_distances, center_rows, pair_id)
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    center_rows.to_csv(center_path, index=False, compression=compression, lineterminator="\n")
    pair_summary.to_csv(summary_path, index=False, lineterminator="\n")
    bootstrap.to_csv(bootstrap_path, index=False, lineterminator="\n")
    bins.to_csv(bins_path, index=False, lineterminator="\n")
    np.savez_compressed(
        neighbor_path,
        query_id_table=frame.query_id.to_numpy(str),
        center_outer_fold=folds,
        neighbor_row_index=all_neighbors,
        neighbor_distance=all_distances,
        neighbor_oracle_action_index=data["oracle_index"][all_neighbors],
        neighbor_action_direction=data["direction_label"][all_neighbors] - 1,
        neighbor_positive_alpha_mask=data["positive_mask"][all_neighbors],
        alpha_grid=ALPHAS,
        action_direction_codebook=np.asarray(["-1=LEFT", "0=ANCHOR", "1=RIGHT"]),
    )
    manifest = {
        "pair_id": pair_id,
        "representation": "X4",
        "rows": int(len(frame)),
        "original_triples": int(frame.original_triple_id.nunique()),
        "folds": 5,
        "k_values": list(K_VALUES),
        "device": device,
        "center_batch_size": batch_size,
        "outer_triple_leakage": 0,
        "x6_status": "X6_FIXED_VECTOR_UNAVAILABLE",
        "sources": [
            {"path": portable_path(path), "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "outputs": [
            {"path": portable_path(path), "sha256": sha256_file(path)}
            for path in (center_path, summary_path, bootstrap_path, bins_path, neighbor_path)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] {pair_id} systematic artifacts written")


def classify(summary: pd.DataFrame, contract: dict) -> tuple[list[str], str, pd.DataFrame, dict]:
    local_rows = []
    for pair_id in PAIR_IDS:
        subset = summary.loc[(summary.pair_id == pair_id) & (summary.representation == "X4")]
        satisfying = subset.loc[
            (subset.direction_agreement_lift_ci95_low > 0)
            & (subset.neighbor_consensus_utility_ci95_low > 0)
            & (subset.consensus_utility_lift_ci95_low > 0)
        ]
        local_rows.append({
            "dataset": subset.dataset.iloc[0], "pair_id": pair_id, "representation": "X4",
            "satisfying_k_count": int(len(satisfying)),
            "satisfying_k": ",".join(str(int(value)) for value in satisfying.k),
            "local_signal_pair": bool(len(satisfying) >= int(contract["local_signal_pair"]["minimum_k_satisfying_all"])),
        })
    local_frame = pd.DataFrame(local_rows)
    local_count = int(local_frame.local_signal_pair.sum())
    supported_datasets = set(local_frame.dataset)
    native_count = int(local_frame.loc[local_frame.pair_id.str.endswith("native_adamf"), "local_signal_pair"].sum())
    local_gate = contract["classification"]["local_identifiable"]
    local_identifiable = (
        local_count >= int(local_gate["minimum_local_signal_pairs"])
        and supported_datasets == {"mkg_w", "db15k"}
        and native_count >= int(local_gate["minimum_native_adamf_local_signal_pairs"])
    )
    x6_recovery = False
    primary = summary.loc[(summary.representation == "X4") & (summary.k == int(contract["primary_display_k"]))]
    zero_including = int(((primary.neighbor_consensus_utility_ci95_low <= 0) & (primary.neighbor_consensus_utility_ci95_high >= 0)).sum())
    ambiguity_gate = contract["classification"]["local_ambiguity"]
    local_ambiguity = (
        local_count <= int(ambiguity_gate["maximum_local_signal_pairs"])
        and zero_including >= int(ambiguity_gate["minimum_pairs_consensus_ci_includes_zero"])
        and not x6_recovery
    )
    recorded = []
    if x6_recovery:
        recorded.append("E5_X6_RECOVERY")
    if local_identifiable:
        recorded.append("E5_LOCAL_IDENTIFIABLE")
    if not recorded and local_ambiguity:
        recorded.append("E5_LOCAL_AMBIGUITY")
    if not recorded:
        recorded.append("E5_INTERMEDIATE")
    precedence = contract["classification"]["report_final_line_precedence"]
    final = next(value for value in precedence if value in recorded)
    evidence = {
        "local_signal_pairs": local_count,
        "supported_datasets": sorted(supported_datasets),
        "native_adamf_local_signal_pairs": native_count,
        "x6_status": "X6_FIXED_VECTOR_UNAVAILABLE",
        "x6_recovery": False,
        "primary_k_consensus_ci_includes_zero_pairs": zero_including,
        "local_identifiable_gate": bool(local_identifiable),
        "local_ambiguity_gate": bool(local_ambiguity),
    }
    return recorded, final, local_frame, evidence


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="40" y="38" font-family="Arial" font-size="22" font-weight="700" fill="#202124">{html.escape(title)}</text>',
    ]


def svg_text(parts: list[str], x: float, y: float, value: str, size: int = 11, anchor: str = "start", fill: str = "#30343b") -> None:
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="{size}" fill="{fill}">{html.escape(value)}</text>')


def write_distance_figure(bins: pd.DataFrame, path: Path) -> None:
    width, height = 1180, 520
    parts = svg_start(width, height, "Distance–Purity Curves")
    left, right, top, bottom = 85, 750, 85, 440
    parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#fff" stroke="#ddd"/>')
    colors = {"mkgw_mhyper_native": "#28778e", "mkgw_native_adamf": "#c45d3c"}
    xscale = lambda value: left + (value - 1) / 9 * (right - left)
    yscale = lambda value: bottom - value * (bottom - top)
    for pair_id, color in colors.items():
        subset = bins.loc[bins.pair_id == pair_id].sort_values("distance_decile")
        points = " ".join(f"{xscale(row.distance_decile):.1f},{yscale(row.direction_agreement):.1f}" for row in subset.itertuples(index=False))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in subset.itertuples(index=False):
            parts.append(f'<circle cx="{xscale(row.distance_decile):.1f}" cy="{yscale(row.direction_agreement):.1f}" r="3" fill="{color}"/>')
        matched = float(subset.matched_direction_agreement_reference.iloc[0])
        parts.append(f'<line x1="{left}" y1="{yscale(matched):.1f}" x2="{right}" y2="{yscale(matched):.1f}" stroke="{color}" stroke-dasharray="6 4"/>')
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = yscale(tick)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#444"/>')
        svg_text(parts, left - 10, y + 4, f"{tick:.2f}", 10, "end")
    for value in range(1, 11):
        svg_text(parts, xscale(value), bottom + 22, str(value), 10, "middle")
    svg_text(parts, (left + right) / 2, 485, "Held-out-to-training distance decile (1 = nearest)", 12, "middle")
    svg_text(parts, 18, (top + bottom) / 2, "Direction agreement", 12)
    svg_text(parts, 105, 67, "X4 solid: kNN; dashed: matched random", 11)
    svg_text(parts, 790, 115, "X6", 18, fill="#777")
    svg_text(parts, 790, 155, "X6_FIXED_VECTOR_UNAVAILABLE", 14, fill="#a04b3b")
    svg_text(parts, 790, 185, "Frozen X6 is a variable-size candidate set", 11, fill="#666")
    svg_text(parts, 790, 205, "plus set encoder, not a canonical vector.", 11, fill="#666")
    svg_text(parts, 790, 265, "MKG-W / M-Hyper+NativE", 11, fill=colors["mkgw_mhyper_native"])
    svg_text(parts, 790, 292, "MKG-W / NativE+AdaMF", 11, fill=colors["mkgw_native_adamf"])
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_utility_figure(summary: pd.DataFrame, path: Path) -> None:
    width, height = 1380, 720
    parts = svg_start(width, height, "Neighbor Consensus Utility vs k")
    ymin = min(0.0, float(summary[["neighbor_consensus_utility_ci95_low", "matched_consensus_utility_ci95_low"]].min().min()))
    ymax = max(0.001, float(summary[["neighbor_consensus_utility_ci95_high", "matched_consensus_utility_ci95_high"]].max().max())) * 1.12
    for panel, pair_id in enumerate(PAIR_IDS):
        col, line = panel % 3, panel // 3
        left, top, pw, ph = 55 + col * 445, 75 + line * 305, 390, 230
        parts.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" rx="8" fill="#fff" stroke="#ddd"/>')
        svg_text(parts, left + 12, top + 22, PAIR_LABELS[pair_id], 12)
        subset = summary.loc[summary.pair_id == pair_id].sort_values("k")
        xs = [left + 60 + index * 95 for index in range(4)]
        scale = lambda value: top + 35 + (ymax - value) / (ymax - ymin) * (ph - 72)
        zero = scale(0.0)
        parts.append(f'<line x1="{left+40}" y1="{zero:.1f}" x2="{left+pw-12}" y2="{zero:.1f}" stroke="#aaa" stroke-dasharray="3 3"/>')
        for field, color, dashed in (("neighbor_consensus_utility", "#28778e", False), ("matched_consensus_utility", "#c45d3c", True)):
            points = []
            for x, row in zip(xs, subset.itertuples(index=False)):
                value = float(getattr(row, field))
                low = float(getattr(row, f"{field}_ci95_low"))
                high = float(getattr(row, f"{field}_ci95_high"))
                parts.append(f'<line x1="{x}" y1="{scale(high):.1f}" x2="{x}" y2="{scale(low):.1f}" stroke="{color}"/>')
                parts.append(f'<circle cx="{x}" cy="{scale(value):.1f}" r="4" fill="{color}"/>')
                points.append(f"{x},{scale(value):.1f}")
            dash = ' stroke-dasharray="6 4"' if dashed else ""
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        for x, k in zip(xs, K_VALUES):
            svg_text(parts, x, top + ph - 12, str(k), 10, "middle")
    svg_text(parts, 40, height - 18, "X4 kNN (blue solid) and matched random (orange dashed); X6 fixed vector unavailable. MRR utility vs unchanged Global.", 11, fill="#666")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_forest(summary: pd.DataFrame, path: Path, k: int) -> None:
    subset = summary.loc[summary.k == k]
    width, height = 1080, 520
    parts = svg_start(width, height, f"Local Direction-Purity Lift at frozen k={k}")
    low = min(-0.05, float(subset.direction_agreement_lift_ci95_low.min()))
    high = max(0.05, float(subset.direction_agreement_lift_ci95_high.max()))
    left, right, top = 340, 1010, 85
    scale = lambda value: left + (value - low) / (high - low) * (right - left)
    parts.append(f'<line x1="{scale(0):.1f}" y1="{top}" x2="{scale(0):.1f}" y2="455" stroke="#888" stroke-dasharray="5 4"/>')
    for index, row in enumerate(subset.itertuples(index=False)):
        y = top + 35 + index * 55
        svg_text(parts, left - 18, y + 4, PAIR_LABELS[row.pair_id], 11, "end")
        x1, x2, x = scale(row.direction_agreement_lift_ci95_low), scale(row.direction_agreement_lift_ci95_high), scale(row.direction_agreement_lift)
        parts.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="#3f5f75" stroke-width="2"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="#c45d3c"/>')
        svg_text(parts, x2 + 8, y + 4, f"{row.direction_agreement_lift:+.3f}", 10)
    svg_text(parts, (left + right) / 2, 492, "kNN direction agreement − matched random", 12, "middle")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_purity_utility(summary: pd.DataFrame, path: Path, k: int) -> None:
    subset = summary.loc[summary.k == k]
    width, height = 1050, 610
    parts = svg_start(width, height, f"Local Purity vs Consensus Utility (k={k})")
    left, right, top, bottom = 100, 980, 80, 520
    xmin = min(-0.02, float(subset.direction_agreement_lift.min()) - 0.02)
    xmax = max(0.02, float(subset.direction_agreement_lift.max()) + 0.02)
    ymin = min(-0.002, float(subset.neighbor_consensus_utility.min()) - 0.001)
    ymax = max(0.002, float(subset.neighbor_consensus_utility.max()) + 0.001)
    xs = lambda value: left + (value - xmin) / (xmax - xmin) * (right - left)
    ys = lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
    parts.append(f'<line x1="{xs(0):.1f}" y1="{top}" x2="{xs(0):.1f}" y2="{bottom}" stroke="#aaa" stroke-dasharray="4 4"/>')
    parts.append(f'<line x1="{left}" y1="{ys(0):.1f}" x2="{right}" y2="{ys(0):.1f}" stroke="#aaa" stroke-dasharray="4 4"/>')
    for index, row in enumerate(subset.itertuples(index=False)):
        x, y = xs(row.direction_agreement_lift), ys(row.neighbor_consensus_utility)
        color = "#28778e" if row.dataset == "mkg_w" else "#c45d3c"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>')
        svg_text(parts, x + 9, y + (-10 if index % 2 == 0 else 18), row.pair_id, 9)
    svg_text(parts, (left + right) / 2, 570, "Direction-agreement lift vs matched", 12, "middle")
    svg_text(parts, 20, (top + bottom) / 2, "kNN consensus utility", 12)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def fmt_ci(row, field: str) -> str:
    return f"{getattr(row, field):+.5f} [{getattr(row, field + '_ci95_low'):+.5f}, {getattr(row, field + '_ci95_high'):+.5f}]"


def write_report(
    summary: pd.DataFrame,
    local_signal: pd.DataFrame,
    recorded: list[str],
    final: str,
    evidence: dict,
    contract: dict,
    path: Path,
) -> None:
    primary_k = int(contract["primary_display_k"])
    primary = summary.loc[summary.k == primary_k]
    lines = [
        "# Experiment 5 — Local Identifiability / Action Ambiguity Audit",
        "",
        "Date: 2026-09-07",
        "Split: DEV only",
        "Frozen prior route: `ROUTE_C_LIMITS`",
        "Experiment 4 status: frozen",
        "",
        "## Representation availability",
        "",
        "- X4: available as the exact 40-dimensional frozen Experiment 2 query representation.",
        "- X6: `X6_FIXED_VECTOR_UNAVAILABLE`. Experiment 2 freezes a variable-size candidate set and set encoder, not a canonical target-independent fixed vector. No new encoder, embedding, PCA, or pooling representation was created.",
        "",
        "## Frozen classification",
        "",
        f"Recorded classifications: {', '.join(f'`{value}`' for value in recorded)}.",
        f"Final report classification by frozen precedence: **{final}**.",
        "",
        "## Primary k=10 results",
        "",
        "| Pair | Direction agreement lift (95% CI) | kNN consensus utility (95% CI) | Utility lift vs matched (95% CI) | Matched coverage | Local-signal pair |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    local_lookup = local_signal.set_index("pair_id")
    for row in primary.itertuples(index=False):
        local = local_lookup.loc[row.pair_id]
        lines.append(
            f"| {PAIR_LABELS[row.pair_id]} | {fmt_ci(row, 'direction_agreement_lift')} "
            f"| {fmt_ci(row, 'neighbor_consensus_utility')} | {fmt_ci(row, 'consensus_utility_lift')} "
            f"| {row.matched_baseline_supported:.1%} | {'yes' if local.local_signal_pair else 'no'} ({local.satisfying_k_count}/4 k) |"
        )
    lines.extend([
        "",
        "A `LOCAL_SIGNAL_PAIR` requires at least two frozen k values to have positive lower confidence bounds for direction-agreement lift, center consensus utility, and consensus-utility lift. Raw purity without matched-baseline advantage does not pass this gate.",
        "",
        "## Gate evidence",
        "",
        f"- LOCAL_SIGNAL_PAIR: {evidence['local_signal_pairs']}/6.",
        f"- Supported datasets: {', '.join(evidence['supported_datasets'])}.",
        f"- NativE+AdaMF local-signal pairs: {evidence['native_adamf_local_signal_pairs']}/2.",
        f"- X4 k={primary_k} consensus CI includes zero: {evidence['primary_k_consensus_ci_includes_zero_pairs']}/6 pairs.",
        "- X6 recovery gate: unavailable and therefore false; it is never imputed from set-encoder inputs.",
        "",
        "## Protocol and diagnostics",
        "",
        "The exact Experiment 2 outer-fold assignment is reused. Every held-out center searches only outer-training rows, and the audit asserts zero overlap in original-triple IDs. X4 scaling is fit separately in each outer-training pool. Matched random pools use exact seed × direction × relation matching with no fallback and 100 frozen without-replacement draws per fold stratum.",
        "",
        "Beneficial-set Jaccard is missing when both sets are empty; `both_empty_rate` is reported separately. Neighbor-consensus actions use only outer-training neighbor utility curves. Center RR enters only after action selection, so center labels never choose the action.",
        "",
        "## Outputs and figures",
        "",
        "- `per_center_knn_metrics.csv.gz`: center-level X4 metrics at k={5,10,20,50}.",
        "- `pair_k_summary.csv`, `matched_baseline_summary.csv`, `distance_purity_bins.csv`, and `bootstrap_ci.csv`.",
        "- `neighbors/*.npz`: canonical query-ID table, top-50 row indices, distances, neighbor Oracle actions/directions, and beneficial-set masks.",
        "- Figure 5.1: distance–purity curves with matched reference and explicit X6-unavailable panel.",
        "- Figure 5.2: neighbor-consensus utility versus k.",
        "- Figure 5.3: k=10 direction-purity lift forest.",
        "- Figure 5.4: purity lift versus actual consensus utility.",
        "",
        "## Integrity audit",
        "",
        "- TEST access = 0",
        "- checkpoint execution/retraining/reselection = 0",
        "- outer-triple leakage = 0",
        "- new representation / PCA / embedding = 0",
        "- supervised metric learning = 0",
        "- new selector or neighbor-method development = 0",
        "- k tuning = 0",
        "- distance-metric selection after results = 0",
        f"- original-triple bootstrap intact = yes ({contract['bootstrap']['samples']} percentile replicates, seed {contract['bootstrap']['seed']})",
        "- all direct source/output hashes recorded = yes (`audit_manifest.json`; self-hash excluded)",
        "- subsequent experiment started = 0",
        "",
        final,
    ])
    if lines[-1] not in FINAL_CLASSES:
        raise AssertionError("Experiment 5 report must end in one frozen classification")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inventory(paths: list[Path], role: str) -> list[dict]:
    records = []
    for path in sorted(set(paths), key=portable_path):
        if not path.exists():
            raise FileNotFoundError(path)
        records.append({"role": role, "path": portable_path(path), "sha256": sha256_file(path)})
    return records


def source_paths_for_audit(args: argparse.Namespace) -> list[Path]:
    exp2_root = Path(args.exp2_root)
    paths = [
        Path(args.contract),
        Path("docs/protocols/EXP5_LOCAL_IDENTIFIABILITY_PROTOCOL.md"),
        Path("docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json"),
        Path(__file__),
        Path("scripts/exp2_information_common.py"),
        Path("scripts/run_exp5_local_identifiability.ps1"),
        Path(args.exp1_root) / "audit_manifest.json",
        Path(args.exp1_root) / "pair_statistics.csv",
        exp2_root / "audit_manifest.json",
        Path(args.exp4_root) / "audit_manifest.json",
    ]
    for pair_id in PAIR_IDS:
        source_manifest = Path(args.utility_manifest_dir) / f"{pair_id}_dev_source_manifest.json"
        manifest = load_json(source_manifest)
        paths.extend([
            source_manifest,
            Path(manifest["source_query_rows"]["path"]),
            exp2_root / "assets" / f"{pair_id}_query_information_manifest.json",
            exp2_root / "assets" / f"{pair_id}_query_information.npz",
            exp2_root / "runs" / pair_id / "x4" / "nested_selected" / "oof_action_predictions.npz",
        ])
    return paths


def analyze_outputs(args: argparse.Namespace, contract: dict) -> None:
    output_dir = Path(args.output_dir)
    summaries, bootstraps, bins, centers = [], [], [], []
    systematic_paths = []
    neighbor_paths = []
    for pair_id in PAIR_IDS:
        work = output_dir / "work" / pair_id
        manifest_path = work / "systematic_manifest.json"
        manifest = load_json(manifest_path)
        if manifest.get("pair_id") != pair_id or manifest.get("outer_triple_leakage") != 0:
            raise RuntimeError(f"Invalid Experiment 5 systematic manifest: {pair_id}")
        for row in [*manifest.get("sources", []), *manifest.get("outputs", [])]:
            verify_declared(Path(row["path"]), row["sha256"], f"{pair_id} systematic output")
        summary_path = work / "pair_k_summary.csv"
        bootstrap_path = work / "bootstrap_ci.csv"
        bins_path = work / "distance_purity_bins.csv"
        center_path = work / "per_center_x4.csv.gz"
        summaries.append(pd.read_csv(summary_path))
        bootstraps.append(pd.read_csv(bootstrap_path))
        bins.append(pd.read_csv(bins_path))
        centers.append(pd.read_csv(center_path))
        systematic_paths.extend([manifest_path, summary_path, bootstrap_path, bins_path, center_path])
        neighbor_paths.append(output_dir / "neighbors" / f"{pair_id}_x4_neighbors.npz")
    summary = pd.concat(summaries, ignore_index=True)
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    distance = pd.concat(bins, ignore_index=True)
    center = pd.concat(centers, ignore_index=True)
    if len(summary) != len(PAIR_IDS) * len(K_VALUES) or len(distance) != len(PAIR_IDS) * 10:
        raise RuntimeError("Incomplete Experiment 5 systematic inventory")
    recorded, final, local_signal, evidence = classify(summary, contract)

    summary_path = output_dir / "pair_k_summary.csv"
    matched_path = output_dir / "matched_baseline_summary.csv"
    distance_path = output_dir / "distance_purity_bins.csv"
    bootstrap_path = output_dir / "bootstrap_ci.csv"
    center_path = output_dir / "per_center_knn_metrics.csv.gz"
    local_path = output_dir / "local_signal_pairs.csv"
    classification_path = output_dir / "classification.json"
    figures = [
        output_dir / "figure1_distance_purity_curves.svg",
        output_dir / "figure2_consensus_utility_vs_k.svg",
        output_dir / "figure3_local_purity_lift_forest.svg",
        output_dir / "figure4_purity_vs_utility.svg",
    ]
    report_path = Path(args.report)
    if not args.overwrite:
        for path in (summary_path, matched_path, distance_path, bootstrap_path, center_path, local_path, classification_path, *figures, report_path):
            if path.exists():
                raise FileExistsError(f"Final output exists; pass --overwrite: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    matched_columns = [
        column for column in summary.columns
        if column in ("dataset", "pair_id", "representation", "k")
        or column.startswith("matched_") or column in ("direction_agreement_lift", "purity_lift", "jaccard_lift", "consensus_utility_lift")
    ]
    summary[matched_columns].to_csv(matched_path, index=False, lineterminator="\n")
    distance.to_csv(distance_path, index=False, lineterminator="\n")
    bootstrap.to_csv(bootstrap_path, index=False, lineterminator="\n")
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    center.to_csv(center_path, index=False, compression=compression, lineterminator="\n")
    local_signal.to_csv(local_path, index=False, lineterminator="\n")
    classification_path.write_text(json.dumps({"recorded_classifications": recorded, "final_classification": final, "evidence": evidence}, indent=2) + "\n", encoding="utf-8")
    write_distance_figure(distance, figures[0])
    write_utility_figure(summary, figures[1])
    write_forest(summary, figures[2], int(contract["primary_display_k"]))
    write_purity_utility(summary, figures[3], int(contract["primary_display_k"]))
    write_report(summary, local_signal, recorded, final, evidence, contract, report_path)

    final_outputs = [summary_path, matched_path, distance_path, bootstrap_path, center_path, local_path, classification_path, *figures, report_path]
    audit = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "split": "dev",
        "representations": {"X4": "available", "X6": "X6_FIXED_VECTOR_UNAVAILABLE"},
        "recorded_classifications": recorded,
        "final_classification": final,
        "classification_evidence": evidence,
        "sources_and_outputs": [
            *inventory(source_paths_for_audit(args), "source"),
            *inventory([*systematic_paths, *neighbor_paths, *final_outputs], "output"),
        ],
        "hash_inventory_note": "audit_manifest.json self-hash excluded to avoid recursive content",
        "operational_audit": {
            "test_access": 0,
            "checkpoint_execution": 0,
            "outer_triple_leakage": 0,
            "new_representation": 0,
            "supervised_metric_learning": 0,
            "new_pca": 0,
            "new_embedding": 0,
            "new_selector": 0,
            "neighbor_policy_development": 0,
            "k_tuning": 0,
            "post_result_distance_selection": 0,
        },
        "next_experiment_started": 0,
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Experiment 5 final classification: {final}")
    print(f"[DONE] report={report_path}")


def main() -> None:
    args = parse_args()
    contract = load_contract(Path(args.contract))
    exp1_root, exp2_root, exp4_root = Path(args.exp1_root), Path(args.exp2_root), Path(args.exp4_root)
    exp1_audit = load_json(exp1_root / "audit_manifest.json")
    exp2_audit = load_json(exp2_root / "audit_manifest.json")
    exp4_audit = load_json(exp4_root / "audit_manifest.json")
    if exp1_audit.get("gate", {}).get("decision") != "GO":
        raise RuntimeError("Experiment 1 Available Complementarity gate is not GO")
    if exp2_audit.get("split") != "dev" or int(exp2_audit.get("operational_audit", {}).get("test_access", -1)) != 0:
        raise RuntimeError("Experiment 2 is not a clean DEV-only input")
    if exp4_audit.get("split") != "dev" or int(exp4_audit.get("operational_audit", {}).get("test_access", -1)) != 0:
        raise RuntimeError("Experiment 4 is not a clean frozen DEV-only input")
    if args.mode == "analyze":
        analyze_outputs(args, contract)
        return
    exp1_stats = pd.read_csv(exp1_root / "pair_statistics.csv")
    selected_pairs = (args.pair_id,) if args.pair_id else PAIR_IDS
    preflight = []
    for pair_id in selected_pairs:
        data, pair_source_paths = load_pair_assets(pair_id, contract, exp1_stats, exp2_audit, Path(args.utility_manifest_dir), exp2_root)
        row = {
            "pair_id": pair_id,
            "rows": int(len(data["frame"])),
            "original_triples": int(data["frame"].original_triple_id.nunique()),
            "x4_dimension": int(data["features"].shape[1]),
            "folds": sorted(set(int(value) for value in data["folds"])),
            "outer_triple_leakage": 0,
            "x6_status": "X6_FIXED_VECTOR_UNAVAILABLE",
            **matched_coverage_preflight(data),
        }
        preflight.append(row)
        if args.mode == "systematic":
            systematic_sources = [
                *pair_source_paths,
                Path(args.contract),
                Path("docs/protocols/EXP5_LOCAL_IDENTIFIABILITY_PROTOCOL.md"),
                Path("docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json"),
                Path(__file__),
                Path("scripts/exp2_information_common.py"),
            ]
            process_pair(pair_id, data, systematic_sources, Path(args.output_dir), contract, args.device, args.center_batch_size, args.overwrite)
    if args.mode == "preflight":
        print(json.dumps({
            "status": "preflight_ok", "split": "dev", "pairs": preflight,
            "x6_status": "X6_FIXED_VECTOR_UNAVAILABLE", "test_access": 0, "checkpoint_execution": 0,
        }, indent=2))


if __name__ == "__main__":
    main()
