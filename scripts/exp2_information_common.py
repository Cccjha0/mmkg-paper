from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.crossfit_heterogeneous_dev_policies import assign_grouped_folds, triple_key
from scripts.train_aacpi_advantage_nested_cv import average_precision, roc_auc, spearman_correlation


ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
RR_COLUMNS = [f"rr_alpha_{alpha:.2f}".replace(".", "_") for alpha in ALPHAS]
ZERO_TOLERANCE = 1e-15
PAIR_IDS = (
    "mkgw_mhyper_native",
    "mkgw_mhyper_adamf",
    "mkgw_native_adamf",
    "db15k_mhyper_native",
    "db15k_mhyper_adamf",
    "db15k_native_adamf",
)
ACTION_FIELDS = ("alpha", "delta_alpha", "abs_delta_alpha")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def reject_test_path(path: Path) -> None:
    lowered = [part.lower() for part in path.parts]
    if "test" in lowered or path.name.lower().startswith("test"):
        raise RuntimeError(f"Experiment 2 refuses TEST-like path: {path}")


def load_contract(path: Path) -> dict:
    reject_test_path(path)
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_first_systematic_run":
        raise ValueError("Experiment 2 feature contract is not frozen")
    if contract.get("split") != "dev" or int(contract.get("test_access", -1)) != 0:
        raise RuntimeError("Experiment 2 contract is not DEV-only")
    if not np.allclose(contract.get("alpha_grid", []), ALPHAS):
        raise ValueError("Experiment 2 alpha grid mismatch")
    if tuple(contract.get("action_descriptors", [])) != ACTION_FIELDS:
        raise ValueError("Experiment 2 action descriptor mismatch")
    x6 = contract["representations"]["X6_candidate"]
    if x6.get("inherits") != "X5":
        raise ValueError("X6 must inherit the complete X5 query context")
    if int(x6.get("top_k_per_expert", -1)) != 100 or int(x6.get("maximum_union_size", -1)) != 200:
        raise ValueError("X6 candidate contract must use frozen union top-K=100 (maximum size 200)")
    expected_candidate_fields = (
        "normalized_score_a", "normalized_score_b", "normalized_rank_a", "normalized_rank_b",
        "top100_member_a", "top100_member_b", "score_a_minus_b", "abs_score_a_minus_b",
        "rank_a_minus_b", "abs_rank_a_minus_b",
    )
    if tuple(x6.get("candidate_fields", [])) != expected_candidate_fields:
        raise ValueError("X6 candidate field contract changed after freeze")
    if x6.get("candidate_identity_feature") is not False:
        raise ValueError("Candidate identity must remain audit-only")
    if x6.get("candidate_embeddings") != "excluded_before_first_systematic_run":
        raise ValueError("Candidate embedding decision changed after freeze")
    expected_compatibility = {
        **{f"X{index}": ["linear_huber", "hist_gbdt", "mlp_low", "mlp_high"] for index in range(1, 6)},
        "X6": ["set_encoder"],
    }
    if contract.get("learner_compatibility") != expected_compatibility:
        raise ValueError("Frozen learner compatibility changed")
    return contract


def representation_features(contract: dict) -> dict[str, list[str]]:
    reps = contract["representations"]
    x1 = list(reps["X1"])
    x2 = [*x1, *reps["X2_additions"]]
    x3 = [*x2, *reps["X3_additions"]]
    x4 = [*x3, *reps["X4_additions"]]
    return {"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x4}


def original_triple_ids(frame) -> np.ndarray:
    return (
        "h=" + frame.head_id.astype(str) + "|r=" + frame.relation_id.astype(str) + "|t=" + frame.tail_id.astype(str)
    ).to_numpy(str)


def grouped_folds(frame, folds: int, seed: int) -> tuple[np.ndarray, dict]:
    representatives = frame.drop_duplicates("original_triple_id")
    rows = [
        {"head_id": str(row.head_id), "relation_id": str(row.relation_id), "tail_id": str(row.tail_id)}
        for row in representatives.itertuples(index=False)
    ]
    assignment, audit = assign_grouped_folds(rows, folds=folds, fold_seed=seed)
    vector = frame.original_triple_id.map(assignment).to_numpy(np.int16)
    if np.any(vector < 0) or np.any(vector >= folds):
        raise AssertionError("Invalid grouped fold assignment")
    if frame.assign(_fold=vector).groupby("original_triple_id")._fold.nunique().max() != 1:
        raise AssertionError("Original triple leaked across folds")
    return vector, audit


def select_global_alpha(rr: np.ndarray, train_mask: np.ndarray) -> tuple[int, float]:
    if rr.ndim != 2 or rr.shape[1] != len(ALPHAS) or not train_mask.any():
        raise ValueError("Invalid RR matrix or empty training mask")
    means = rr[train_mask].mean(axis=0)
    best = float(means.max())
    candidates = np.flatnonzero(np.isclose(means, best, rtol=0.0, atol=ZERO_TOLERANCE))
    preference = np.abs(ALPHAS[candidates] - 0.5) + ALPHAS[candidates] * 1e-6
    index = int(candidates[int(preference.argmin())])
    return index, float(means[index])


def action_descriptors(global_index: int) -> np.ndarray:
    delta = ALPHAS - ALPHAS[global_index]
    return np.column_stack((ALPHAS, delta, np.abs(delta))).astype(np.float32)


def force_global_zero(predicted: np.ndarray, global_index: int) -> np.ndarray:
    result = np.asarray(predicted, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != len(ALPHAS):
        raise ValueError("Predicted utility must be query x action")
    result[:, global_index] = 0.0
    return result


def select_probe_actions(predicted: np.ndarray, global_index: int) -> np.ndarray:
    predicted = force_global_zero(predicted, global_index)
    maxima = predicted.max(axis=1, keepdims=True)
    tied = np.isclose(predicted, maxima, rtol=0.0, atol=ZERO_TOLERANCE)
    distance = np.abs(ALPHAS - ALPHAS[global_index])
    preference = distance + ALPHAS * 1e-6
    preference[global_index] = -1e-9
    return np.where(tied, preference[None, :], np.inf).argmin(axis=1)


def sign_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    positive = actual > ZERO_TOLERANCE
    harmful = actual < -ZERO_TOLERANCE
    positive_prevalence = float(positive.mean())
    harmful_prevalence = float(harmful.mean())
    positive_ap = average_precision(positive, predicted)
    harmful_ap = average_precision(harmful, -predicted)
    nonzero = positive | harmful
    sign_auc = roc_auc(positive[nonzero], predicted[nonzero]) if nonzero.any() else float("nan")
    return {
        "spearman_pred_u_actual_u": spearman_correlation(actual, predicted),
        "positive_ap": positive_ap,
        "positive_prevalence": positive_prevalence,
        "positive_ap_lift": positive_ap - positive_prevalence,
        "harmful_ap": harmful_ap,
        "harmful_prevalence": harmful_prevalence,
        "harmful_ap_lift": harmful_ap - harmful_prevalence,
        "positive_vs_harmful_auroc": sign_auc,
    }


def clustered_bootstrap(values, groups, samples: int, seed: int) -> tuple[float, float]:
    import pandas as pd

    frame = pd.DataFrame({"value": values, "group": groups})
    cluster = frame.groupby("group", sort=False).value.mean().to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 64):
        stop = min(start + 64, samples)
        indices = rng.integers(0, len(cluster), size=(stop - start, len(cluster)))
        boot[start:stop] = cluster[indices].mean(axis=1)
    return tuple(float(value) for value in np.percentile(boot, [2.5, 97.5]))


def policy_metrics(
    rr: np.ndarray,
    predicted: np.ndarray,
    global_indices: np.ndarray,
    groups: np.ndarray,
    available_headroom: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, np.ndarray]:
    if len(rr) != len(predicted) or len(rr) != len(global_indices):
        raise ValueError("Policy arrays are not aligned")
    chosen = np.empty(len(rr), dtype=np.int16)
    for global_index in np.unique(global_indices):
        mask = global_indices == global_index
        chosen[mask] = select_probe_actions(predicted[mask], int(global_index))
    row = np.arange(len(rr))
    selected_rr = rr[row, chosen]
    global_rr = rr[row, global_indices]
    gain = selected_rr - global_rr
    low, high = clustered_bootstrap(gain, groups, bootstrap_samples, bootstrap_seed)
    return {
        "count": int(len(rr)),
        "oof_mrr": float(selected_rr.mean()),
        "fold_specific_global_mrr": float(global_rr.mean()),
        "delta_mrr": float(gain.mean()),
        "clustered_ci95_low": low,
        "clustered_ci95_high": high,
        "headroom_recovery": float(gain.mean() / available_headroom),
        "negative_transfer_rate": float((gain < -ZERO_TOLERANCE).mean()),
        "positive_gain_rate": float((gain > ZERO_TOLERANCE).mean()),
        "changed_rate": float((chosen != global_indices).mean()),
    }, chosen


def fit_pca(train: np.ndarray, all_values: np.ndarray, seed: int, dimension: int = 16) -> np.ndarray:
    train = np.asarray(train, dtype=np.float32)
    all_values = np.asarray(all_values, dtype=np.float32)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = train - mean
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((centered.shape[1], dimension + 8), dtype=np.float32)
    sketch = centered @ omega
    for _ in range(2):
        stabilized, _ = np.linalg.qr(sketch, mode="reduced")
        sketch = centered @ (centered.T @ stabilized)
    basis, _ = np.linalg.qr(sketch, mode="reduced")
    _, _, vt = np.linalg.svd(basis.T @ centered, full_matrices=False)
    components = vt[:dimension].copy()
    for component in components:
        if component[int(np.argmax(np.abs(component)))] < 0:
            component *= -1
    return ((all_values - mean) @ components.T).astype(np.float32)


def project_latents(z_a: np.ndarray, z_b: np.ndarray, train_mask: np.ndarray, seed: int) -> np.ndarray:
    a = fit_pca(z_a[train_mask], z_a, seed)
    b = fit_pca(z_b[train_mask], z_b, seed + 1)
    return np.column_stack((a, b, a - b, np.abs(a - b))).astype(np.float32)
