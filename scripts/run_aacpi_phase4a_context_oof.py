from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aacpi_phase4a_common import (
    LATENT_DIM,
    LATENT_FEATURES,
    REPRESENTATIONS,
    STATIC_FEATURES,
    feature_contract,
    portable_path,
    reject_test_path,
    sha256_array,
    sha256_file,
)
from scripts.run_aacpi_phase3a_representation_oof import read_feature_table, sign_metrics
from scripts.train_aacpi_advantage_nested_cv import (
    calibration_rows,
    evaluate_predictions,
    grouped_fold_vector,
    inner_selection_key,
    load_yaml,
    model_configs,
    resolve_device,
    train_predict,
    validate_search_space,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen DEV-only Phase 4A contextual nested OOF.")
    parser.add_argument("--context-table", required=True)
    parser.add_argument("--latent-file", required=True)
    parser.add_argument("--phase3a-r3-oof", required=True)
    parser.add_argument("--output-dir", default="outputs/aacpi/phase4a")
    parser.add_argument("--search-space", default="docs/protocols/AACPI_V2_PHASE2_SEARCH_SPACE.yaml")
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def groups_hash(values) -> str:
    raw = "\n".join(sorted(set(map(str, values)))).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def standardization_audit(values: np.ndarray, frame, train_indices: np.ndarray) -> dict:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {
        "kind": "feature_standardization",
        "fit_row_count": int(len(train_indices)),
        "fit_original_triple_count": int(frame.iloc[train_indices].original_triple_id.nunique()),
        "fit_original_triples_sha256": groups_hash(frame.iloc[train_indices].original_triple_id),
        "mean_sha256": sha256_array(mean.astype(np.float64)),
        "scale_sha256": sha256_array(scale.astype(np.float64)),
        "heldout_rows_used_for_fit": 0,
    }


def load_latents(path: Path, frame):
    reject_test_path(path)
    with np.load(path, allow_pickle=False) as payload:
        try:
            query_ids = payload["query_id"].astype(str)
        except ValueError as exc:
            raise RuntimeError(
                "Latent NPZ contains object arrays; rebuild it with the schema-v2 fixed-width Unicode extractor"
            ) from exc
        z_a = payload["z_a"].astype(np.float64)
        z_b = payload["z_b"].astype(np.float64)
    if len(query_ids) != len(set(query_ids)) or z_a.shape[0] != len(query_ids) or z_b.shape[0] != len(query_ids):
        raise RuntimeError("Invalid latent query inventory")
    lookup = {value: index for index, value in enumerate(query_ids)}
    try:
        row_query_index = np.asarray([lookup[value] for value in frame.query_id.astype(str)], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"Missing latent for query {exc}") from exc
    if set(lookup) != set(frame.query_id.astype(str)):
        raise RuntimeError("Latent and context query sets differ")
    if not np.isfinite(z_a).all() or not np.isfinite(z_b).all():
        raise RuntimeError("Nonfinite raw latent")
    return z_a, z_b, row_query_index


def fit_pca(train: np.ndarray, all_values: np.ndarray, seed: int):
    if len(train) <= LATENT_DIM:
        raise RuntimeError("Too few training queries for frozen PCA dimension")
    train = np.asarray(train, dtype=np.float32)
    all_values = np.asarray(all_values, dtype=np.float32)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = train - mean
    # Frozen Halko randomized SVD: 8 oversamples and two power iterations.
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((centered.shape[1], LATENT_DIM + 8), dtype=np.float32)
    sketch = centered @ omega
    for _ in range(2):
        stabilized, _ = np.linalg.qr(sketch, mode="reduced")
        sketch = centered @ (centered.T @ stabilized)
    basis, _ = np.linalg.qr(sketch, mode="reduced")
    _, singular_values, vt = np.linalg.svd(basis.T @ centered, full_matrices=False)
    components = vt[:LATENT_DIM].copy()
    signs = np.ones(LATENT_DIM, dtype=np.float64)
    for index, component in enumerate(components):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            signs[index] = -1.0
    transformed = ((all_values - mean) @ components.T) * signs
    components *= signs[:, None]
    total_variance = float(centered.var(axis=0, ddof=1).sum())
    explained = (singular_values[:LATENT_DIM] ** 2) / max(len(train) - 1, 1)
    audit = {
        "input_dimension": int(train.shape[1]), "output_dimension": LATENT_DIM,
        "fit_query_count": int(train.shape[0]), "random_state": int(seed),
        "mean_sha256": sha256_array(mean),
        "components_sha256": sha256_array(components.astype(np.float64)),
        "explained_variance_ratio_sum": float(explained.sum() / total_variance) if total_variance > 0 else 0.0,
        "solver": "frozen_halko_randomized_svd", "oversamples": 8, "power_iterations": 2,
        "target_used": False,
    }
    return transformed.astype(np.float64), audit


def latent_projection(z_a, z_b, row_query_index, frame, train_mask, seed: int):
    fit_query_indices = np.unique(row_query_index[train_mask])
    a, audit_a = fit_pca(z_a[fit_query_indices], z_a, seed)
    b, audit_b = fit_pca(z_b[fit_query_indices], z_b, seed + 1)
    query_features = np.concatenate((a, b, a - b, np.abs(a - b)), axis=1)
    if query_features.shape[1] != len(LATENT_FEATURES) or not np.isfinite(query_features).all():
        raise AssertionError("Invalid projected latent feature matrix")
    audit = {
        "fit_original_triple_count": int(frame.loc[train_mask, "original_triple_id"].nunique()),
        "fit_original_triples_sha256": groups_hash(frame.loc[train_mask, "original_triple_id"]),
        "expert_a_pca": audit_a, "expert_b_pca": audit_b,
        "heldout_rows_used_for_fit": 0, "advantage_target_used_for_fit": False,
    }
    return query_features, audit


def metrics(actual, predicted, beta: float) -> dict:
    result = evaluate_predictions(actual, predicted, beta=beta)
    result.update(sign_metrics(actual, predicted))
    buckets = calibration_rows(actual, predicted)
    top = buckets[-1]
    result.update({
        "highest_10pct_actual_mean_advantage": float(top["actual_mean_u"]),
        "highest_10pct_positive_rate": float(top["positive_rate"]),
        "highest_10pct_harmful_rate": float(top["harmful_rate"]),
    })
    return result


def write_result(frame, oof, selected_ids, outer_vector, representation, root, audit, selection_rows, inner_rows, preprocessing_rows, beta):
    pair_id, dataset = str(frame.pair_id.iloc[0]), str(frame.dataset.iloc[0])
    small_dir = root / pair_id / representation.lower()
    raw_path = root / "oof_raw" / pair_id / representation.lower() / "dev_oof_predictions.csv.gz"
    small_dir.mkdir(parents=True, exist_ok=True); raw_path.parent.mkdir(parents=True, exist_ok=True)
    reference = np.isclose(frame.alpha, frame.alpha0, atol=1e-12, rtol=0.0)
    primary = metrics(frame.loc[~reference, "advantage"].to_numpy(float), oof[~reference], beta)
    calibration = calibration_rows(frame.loc[~reference, "advantage"].to_numpy(float), oof[~reference])
    for row in calibration:
        row.update({"dataset": dataset, "pair_id": pair_id, "representation": representation})
    output = frame.copy(); output["representation"] = representation
    output["outer_fold"] = outer_vector + 1; output["selected_config_id"] = selected_ids
    output["predicted_advantage_oof"] = oof
    output.to_csv(raw_path, index=False, compression="gzip")
    write_csv(small_dir / "outer_fold_selections.csv", selection_rows)
    if inner_rows:
        write_csv(small_dir / "inner_search_results.csv", inner_rows)
    write_csv(small_dir / "dev_oof_calibration.csv", calibration)
    metric_row = {"dataset": dataset, "pair_id": pair_id, "representation": representation, **primary}
    write_csv(small_dir / "dev_oof_metrics.csv", [metric_row])
    (small_dir / "preprocessing_manifest.json").write_text(json.dumps(preprocessing_rows, indent=2) + "\n", encoding="utf-8")
    payload = {
        **audit, "dry_run": False, "primary_metrics": primary,
        "h1_style_pass": all(primary[key] > 0 for key in ("spearman", "positive_auprc_lift", "harmful_auprc_lift", "highest_10pct_actual_mean_advantage")),
        "oof_predictions": {"path": portable_path(raw_path), "sha256": sha256_file(raw_path)},
        "outer_oof_complete": True, "policy_evaluation_performed": False,
    }
    (small_dir / "dev_oof_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (small_dir / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {pair_id} {representation} spearman={primary['spearman']:.6f} sign_auc={primary['positive_vs_harmful_auroc']:.6f}")
    return payload


def copy_c0(frame, r3_path, root, space, overwrite):
    import pandas as pd

    source = pd.read_csv(r3_path, compression="infer")
    keys = ["query_id", "alpha", "original_triple_id", "advantage"]
    if len(source) != len(frame) or not source[keys].reset_index(drop=True).equals(frame[keys].reset_index(drop=True)):
        raise RuntimeError("C0 source does not exactly match Phase 4A row inventory")
    if set(source.representation.astype(str).str.upper()) != {"R3"}:
        raise RuntimeError("C0 source is not Phase 3A R3")
    oof = source.predicted_advantage_oof.to_numpy(np.float64)
    outer_vector = source.outer_fold.to_numpy(np.int64) - 1
    selected = source.selected_config_id.astype(str).to_numpy()
    audit = {
        "schema_version": 1, "phase": "AACPI Phase 4A", "representation": "C0", "split": "dev",
        "dataset": str(frame.dataset.iloc[0]), "pair_id": str(frame.pair_id.iloc[0]),
        "control_source": {"path": portable_path(r3_path), "sha256": sha256_file(r3_path)},
        "prediction_vector_sha256": sha256_array(oof), "exact_phase3a_r3_copy": True,
        "n_rows": len(frame), "n_queries": int(frame.query_id.nunique()),
        "outer_original_triple_leakage": 0, "test_rows_accessed": 0, "test_evaluation_commands": 0,
    }
    selections = [{"representation": "C0", "outer_fold": fold, "selected_config_id": str(source.loc[source.outer_fold == fold, "selected_config_id"].iloc[0]), "source": "Phase3A_R3_exact_copy"} for fold in sorted(source.outer_fold.unique())]
    return write_result(frame, oof, selected, outer_vector, "C0", root, audit, selections, [], [{"rule": "exact_copy_no_preprocessing_refit"}], float(space["loss"]["beta"]))


def run_trained(representation, frame, static, target, z_a, z_b, row_query_index, root, space, configs, device, overwrite):
    pair_id, dataset = str(frame.pair_id.iloc[0]), str(frame.dataset.iloc[0])
    outer_vector, outer_audit = grouped_fold_vector(frame, folds=int(space["nested_cv"]["outer_folds"]), fold_seed=int(space["nested_cv"]["outer_fold_seed"]))
    oof = np.full(len(frame), np.nan); selected_ids = np.asarray([None] * len(frame), dtype=object)
    selection_rows, inner_rows, preprocessing_rows = [], [], []
    uses_latent = representation in {"C3", "C4"}
    outer_folds, inner_folds = int(space["nested_cv"]["outer_folds"]), int(space["nested_cv"]["inner_folds"])
    inner_seed, model_seed = int(space["nested_cv"]["inner_fold_seed"]), int(space["training"]["model_seed"])
    beta = float(space["loss"]["beta"])
    for outer_fold in range(outer_folds):
        outer_train, outer_hold = outer_vector != outer_fold, outer_vector == outer_fold
        outer_indices = np.flatnonzero(outer_train)
        inner_frame = frame.loc[outer_train]
        inner_vector, inner_audit = grouped_fold_vector(inner_frame, folds=inner_folds, fold_seed=inner_seed + outer_fold)
        latent_cache = {}
        if uses_latent:
            for inner_fold in range(inner_folds):
                fit_mask = np.zeros(len(frame), dtype=bool)
                fit_mask[outer_indices[inner_vector != inner_fold]] = True
                latent_cache[inner_fold], pca_audit = latent_projection(z_a, z_b, row_query_index, frame, fit_mask, 20260905 + outer_fold * 100 + inner_fold * 2)
                preprocessing_rows.append({"kind": "latent_pca", "scope": "inner", "outer_fold": outer_fold + 1, "inner_fold": inner_fold + 1, **pca_audit})
        for inner_fold in range(inner_folds):
            train_global = outer_indices[inner_vector != inner_fold]
            x_for_audit = static[train_global]
            if uses_latent:
                x_for_audit = np.column_stack((x_for_audit, latent_cache[inner_fold][row_query_index[train_global]]))
            preprocessing_rows.append({"scope": "inner", "outer_fold": outer_fold + 1, "inner_fold": inner_fold + 1, **standardization_audit(x_for_audit, frame, train_global)})
        candidates = []
        for config_index, config in enumerate(configs):
            inner_oof = np.full(len(outer_indices), np.nan)
            for inner_fold in range(inner_folds):
                train_local, hold_local = inner_vector != inner_fold, inner_vector == inner_fold
                train_global, hold_global = outer_indices[train_local], outer_indices[hold_local]
                x_train, x_hold = static[train_global], static[hold_global]
                if uses_latent:
                    latent = latent_cache[inner_fold]
                    x_train = np.column_stack((x_train, latent[row_query_index[train_global]]))
                    x_hold = np.column_stack((x_hold, latent[row_query_index[hold_global]]))
                seed = model_seed + outer_fold * 100_000 + config_index * 100 + inner_fold
                inner_oof[hold_local] = train_predict(x_train, target[train_global], x_hold, config=config, space=space, device=device, seed=seed)
            nonref = ~np.isclose(frame.iloc[outer_indices].alpha, frame.iloc[outer_indices].alpha0, atol=1e-12, rtol=0.0)
            result = evaluate_predictions(target[outer_indices][nonref], inner_oof[nonref], beta=beta)
            key = inner_selection_key(result, config); candidates.append((key, config, result))
            inner_rows.append({"representation": representation, "outer_fold": outer_fold + 1, "config_id": config.config_id, "hidden_width": config.hidden_width, "learning_rate": config.learning_rate, "negative_weight": config.negative_weight, **result})
            print(f"[INNER] {pair_id} {representation} outer={outer_fold+1}/{outer_folds} config={config.config_id}", flush=True)
        _, selected, selected_metrics = max(candidates, key=lambda item: item[0])
        x_train, x_hold = static[outer_train], static[outer_hold]
        if uses_latent:
            latent, pca_audit = latent_projection(z_a, z_b, row_query_index, frame, outer_train, 20260905 + outer_fold * 1000 + 999)
            preprocessing_rows.append({"kind": "latent_pca", "scope": "outer", "outer_fold": outer_fold + 1, **pca_audit})
            x_train = np.column_stack((x_train, latent[row_query_index[outer_train]]))
            x_hold = np.column_stack((x_hold, latent[row_query_index[outer_hold]]))
        preprocessing_rows.append({"scope": "outer", "outer_fold": outer_fold + 1, **standardization_audit(x_train, frame, np.flatnonzero(outer_train))})
        oof[outer_hold] = train_predict(x_train, target[outer_train], x_hold, config=selected, space=space, device=device, seed=model_seed + outer_fold * 100_000 + 99_999)
        selected_ids[outer_hold] = selected.config_id
        selection_rows.append({"representation": representation, "outer_fold": outer_fold + 1, "selected_config_id": selected.config_id, "hidden_width": selected.hidden_width, "learning_rate": selected.learning_rate, "negative_weight": selected.negative_weight, "outer_train_rows": int(outer_train.sum()), "outer_holdout_rows": int(outer_hold.sum()), "outer_train_groups": int(frame.loc[outer_train, 'original_triple_id'].nunique()), "outer_holdout_groups": int(frame.loc[outer_hold, 'original_triple_id'].nunique()), "inner_group_audit": json.dumps(inner_audit, sort_keys=True), **{f"selected_inner_{k}": v for k, v in selected_metrics.items()}})
        print(f"[OUTER] {pair_id} {representation} fold={outer_fold+1}/{outer_folds} selected={selected.config_id}", flush=True)
    if not np.isfinite(oof).all() or any(value is None for value in selected_ids):
        raise AssertionError("Incomplete Phase 4A OOF")
    audit = {
        "schema_version": 1, "phase": "AACPI Phase 4A", "representation": representation, "split": "dev",
        "dataset": dataset, "pair_id": pair_id, "feature_fields": [*STATIC_FEATURES[representation], *(LATENT_FEATURES if uses_latent else [])],
        "n_features": static.shape[1] + (len(LATENT_FEATURES) if uses_latent else 0),
        "n_rows": len(frame), "n_queries": int(frame.query_id.nunique()), "n_original_triples": int(frame.original_triple_id.nunique()),
        "outer_fold_audit": outer_audit, "frozen_phase2b_estimator": True,
        "fold_local_pca": uses_latent, "test_rows_accessed": 0, "test_evaluation_commands": 0, "policy_evaluations": 0,
    }
    return write_result(frame, oof, selected_ids, outer_vector, representation, root, audit, selection_rows, inner_rows, preprocessing_rows, beta)


def main() -> None:
    args = parse_args()
    paths = [Path(args.context_table), Path(args.latent_file), Path(args.phase3a_r3_oof), Path(args.search_space)]
    for path in paths:
        reject_test_path(path)
    root = Path(args.output_dir); space = load_yaml(paths[3]); validate_search_space(space)
    frame, all_static, target, _ = read_feature_table(paths[0], STATIC_FEATURES["C4"])
    static_lookup = {name: index for index, name in enumerate(STATIC_FEATURES["C4"])}
    z_a, z_b, row_query_index = load_latents(paths[1], frame)
    if args.dry_run:
        grouped_fold_vector(frame, folds=int(space["nested_cv"]["outer_folds"]), fold_seed=int(space["nested_cv"]["outer_fold_seed"]))
        print(f"[DRY-RUN OK] {frame.pair_id.iloc[0]} rows={len(frame)} queries={frame.query_id.nunique()}")
        return
    device, configs = resolve_device(args.device), model_configs(space)
    results = []
    for representation in args.representations:
        small_dir = root / str(frame.pair_id.iloc[0]) / representation.lower()
        if small_dir.exists() and any(small_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {small_dir}")
        if representation == "C0":
            results.append(copy_c0(frame, paths[2], root, space, args.overwrite)); continue
        indices = [static_lookup[name] for name in STATIC_FEATURES[representation]]
        results.append(run_trained(representation, frame, all_static[:, indices], target, z_a, z_b, row_query_index, root, space, configs, device, args.overwrite))
    manifest_path = root / str(frame.pair_id.iloc[0]) / "phase4a_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "phase": "AACPI Phase 4A", "split": "dev", "pair_id": str(frame.pair_id.iloc[0]),
        "representations": list(args.representations), "feature_contract": feature_contract(),
        "sources": [{"path": portable_path(path), "sha256": sha256_file(path)} for path in paths],
        "results": results, "test_rows_accessed": 0, "test_evaluation_commands": 0,
        "policy_evaluations": 0, "expert_retraining": 0, "checkpoint_reselection": 0,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
