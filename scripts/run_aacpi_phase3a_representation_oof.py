from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aacpi_phase3a_common import REPRESENTATION_FEATURES, portable_path, sha256_file
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


def read_feature_table(path: Path, feature_fields: list[str]):
    import pandas as pd

    identity = [
        "dataset", "pair_id", "split", "original_triple_id", "query_id", "seed",
        "direction", "head", "relation", "tail", "alpha0", "alpha", "delta_alpha",
        "abs_delta_alpha", "rr_anchor", "rr_action", "advantage",
    ]
    required = list(dict.fromkeys([*identity, *feature_fields]))
    frame = pd.read_csv(path, compression="infer", usecols=required)
    if frame.empty or set(frame["split"].astype(str)) != {"dev"}:
        raise RuntimeError("Phase 3A accepts DEV feature rows only")
    if frame[required].isna().any().any():
        raise ValueError("Phase 3A feature table contains missing values")
    features = frame[feature_fields].to_numpy(dtype=np.float64)
    target = frame["advantage"].to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Phase 3A feature table contains NaN/Inf")
    if not np.allclose(frame["abs_delta_alpha"], np.abs(frame["delta_alpha"]), atol=1e-12, rtol=0.0):
        raise ValueError("abs_delta_alpha is inconsistent")
    reference = np.isclose(frame["alpha"], frame["alpha0"], atol=1e-12, rtol=0.0)
    if not np.all(np.abs(target[reference]) <= 1e-12):
        raise ValueError("Reference action advantage is not zero")
    return frame, features, target, reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AACPI Phase 3A R0/R1/R2/R3 with frozen Phase 2B nested grouped CV."
    )
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--search-space", default="docs/protocols/AACPI_V2_PHASE2_SEARCH_SPACE.yaml")
    parser.add_argument("--representations", nargs="+", choices=tuple(REPRESENTATION_FEATURES), default=list(REPRESENTATION_FEATURES))
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
        writer.writeheader()
        writer.writerows(rows)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sign_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    nonzero = ~np.isclose(actual, 0.0, rtol=0.0, atol=1e-12)
    positive_vs_harmful = roc_auc(actual[nonzero] > 0.0, predicted[nonzero])
    active = ~np.isclose(actual, 0.0, rtol=0.0, atol=1e-12)
    activity_auprc = average_precision(active, np.abs(predicted))
    prevalence = float(active.mean())
    return {
        "positive_vs_harmful_auroc": positive_vs_harmful,
        "positive_vs_harmful_auroc_lift": positive_vs_harmful - 0.5,
        "nonzero_activity_auprc": activity_auprc,
        "nonzero_activity_prevalence": prevalence,
        "nonzero_activity_auprc_lift": activity_auprc - prevalence,
    }


def run_one(
    *,
    feature_path: Path,
    output_dir: Path,
    search_path: Path,
    representation: str,
    requested_device: str,
    dry_run: bool,
    overwrite: bool,
) -> dict:
    space = load_yaml(search_path)
    validate_search_space(space)
    configs = model_configs(space)
    feature_fields = REPRESENTATION_FEATURES[representation]
    frame, features, target, reference = read_feature_table(feature_path, feature_fields)
    pair_id = str(frame["pair_id"].iloc[0])
    dataset = str(frame["dataset"].iloc[0])
    if frame["pair_id"].astype(str).nunique() != 1 or frame["dataset"].astype(str).nunique() != 1:
        raise ValueError("Each Phase 3A run must contain one dataset/pair")
    outer_vector, outer_audit = grouped_fold_vector(
        frame,
        folds=int(space["nested_cv"]["outer_folds"]),
        fold_seed=int(space["nested_cv"]["outer_fold_seed"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "input_audit.json"
    if audit_path.exists() and not overwrite:
        raise FileExistsError(audit_path)
    audit = {
        "schema_version": 1,
        "phase": "AACPI Phase 3A",
        "representation": representation,
        "split": "dev",
        "dataset": dataset,
        "pair_id": pair_id,
        "feature_table": portable_path(feature_path),
        "feature_table_sha256": sha256_file(feature_path),
        "search_space": portable_path(search_path),
        "search_space_sha256": sha256_file(search_path),
        "feature_fields": feature_fields,
        "n_features": len(feature_fields),
        "n_rows": len(frame),
        "n_queries": int(frame["query_id"].nunique()),
        "n_original_triples": int(frame["original_triple_id"].nunique()),
        "outer_fold_audit": outer_audit,
        "frozen_phase2b_estimator": True,
        "test_rows_accessed": 0,
        "test_evaluation_commands": 0,
        "dry_run": dry_run,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if dry_run:
        print(f"[DRY-RUN OK] pair={pair_id} representation={representation} rows={len(frame)}")
        return audit

    expected = [
        output_dir / "dev_oof_predictions.csv.gz",
        output_dir / "outer_fold_selections.csv",
        output_dir / "inner_search_results.csv",
        output_dir / "dev_oof_metrics.json",
        output_dir / "dev_oof_metrics.csv",
        output_dir / "dev_oof_calibration.csv",
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")
    device = resolve_device(requested_device)
    outer_folds = int(space["nested_cv"]["outer_folds"])
    inner_folds = int(space["nested_cv"]["inner_folds"])
    inner_seed = int(space["nested_cv"]["inner_fold_seed"])
    model_seed = int(space["training"]["model_seed"])
    beta = float(space["loss"]["beta"])
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    selected_ids = np.asarray([None] * len(frame), dtype=object)
    selection_rows, inner_rows = [], []
    for outer_fold in range(outer_folds):
        outer_train = outer_vector != outer_fold
        outer_holdout = outer_vector == outer_fold
        outer_indices = np.flatnonzero(outer_train)
        inner_frame = frame.loc[outer_train]
        inner_vector, inner_audit = grouped_fold_vector(
            inner_frame, folds=inner_folds, fold_seed=inner_seed + outer_fold
        )
        candidates = []
        for config_index, config in enumerate(configs):
            inner_oof = np.full(len(outer_indices), np.nan, dtype=np.float64)
            for inner_fold in range(inner_folds):
                inner_train = inner_vector != inner_fold
                inner_hold = inner_vector == inner_fold
                seed = model_seed + outer_fold * 100_000 + config_index * 100 + inner_fold
                inner_oof[inner_hold] = train_predict(
                    features[outer_indices[inner_train]],
                    target[outer_indices[inner_train]],
                    features[outer_indices[inner_hold]],
                    config=config,
                    space=space,
                    device=device,
                    seed=seed,
                )
            if not np.isfinite(inner_oof).all():
                raise AssertionError("Incomplete inner OOF predictions")
            nonreference = ~reference[outer_indices]
            metrics = evaluate_predictions(target[outer_indices][nonreference], inner_oof[nonreference], beta=beta)
            key = inner_selection_key(metrics, config)
            candidates.append((key, config, metrics))
            inner_rows.append({
                "representation": representation,
                "outer_fold": outer_fold + 1,
                "config_id": config.config_id,
                "hidden_width": config.hidden_width,
                "learning_rate": config.learning_rate,
                "negative_weight": config.negative_weight,
                **metrics,
            })
            print(f"[INNER] {pair_id} {representation} outer={outer_fold+1}/{outer_folds} config={config.config_id}", flush=True)
        _, selected, selected_metrics = max(candidates, key=lambda item: item[0])
        oof[outer_holdout] = train_predict(
            features[outer_train], target[outer_train], features[outer_holdout],
            config=selected, space=space, device=device,
            seed=model_seed + outer_fold * 100_000 + 99_999,
        )
        selected_ids[outer_holdout] = selected.config_id
        selection_rows.append({
            "representation": representation,
            "outer_fold": outer_fold + 1,
            "selected_config_id": selected.config_id,
            "hidden_width": selected.hidden_width,
            "learning_rate": selected.learning_rate,
            "negative_weight": selected.negative_weight,
            "outer_train_rows": int(outer_train.sum()),
            "outer_holdout_rows": int(outer_holdout.sum()),
            "outer_train_groups": int(frame.loc[outer_train, "original_triple_id"].nunique()),
            "outer_holdout_groups": int(frame.loc[outer_holdout, "original_triple_id"].nunique()),
            "inner_group_audit": json.dumps(inner_audit, sort_keys=True),
            **{f"selected_inner_{key}": value for key, value in selected_metrics.items()},
        })
        print(f"[OUTER] {pair_id} {representation} fold={outer_fold+1}/{outer_folds} selected={selected.config_id}", flush=True)
    if not np.isfinite(oof).all() or any(value is None for value in selected_ids):
        raise AssertionError("Incomplete outer OOF predictions")
    nonreference = ~reference
    primary = evaluate_predictions(target[nonreference], oof[nonreference], beta=beta)
    primary.update(sign_metrics(target[nonreference], oof[nonreference]))
    calibration = calibration_rows(target[nonreference], oof[nonreference])
    for row in calibration:
        row.update({"dataset": dataset, "pair_id": pair_id, "representation": representation})
    top = calibration[-1]
    primary.update({
        "highest_10pct_actual_mean_advantage": float(top["actual_mean_u"]),
        "highest_10pct_positive_rate": float(top["positive_rate"]),
        "highest_10pct_harmful_rate": float(top["harmful_rate"]),
    })
    frame = frame.copy()
    frame["representation"] = representation
    frame["outer_fold"] = outer_vector + 1
    frame["selected_config_id"] = selected_ids
    frame["predicted_advantage_oof"] = oof
    frame.to_csv(output_dir / "dev_oof_predictions.csv.gz", index=False, compression="gzip")
    write_csv(output_dir / "outer_fold_selections.csv", selection_rows)
    write_csv(output_dir / "inner_search_results.csv", inner_rows)
    write_csv(output_dir / "dev_oof_calibration.csv", calibration)
    metric_row = {"dataset": dataset, "pair_id": pair_id, "representation": representation, **primary}
    write_csv(output_dir / "dev_oof_metrics.csv", [metric_row])
    payload = {
        **audit,
        "dry_run": False,
        "device": device,
        "primary_evaluation_rows": "nonreference_actions_only",
        "primary_metrics": primary,
        "h1_style_pass": all(
            primary[key] > 0.0
            for key in ("spearman", "positive_auprc_lift", "harmful_auprc_lift", "highest_10pct_actual_mean_advantage")
        ),
        "outer_oof_complete": True,
        "policy_evaluation_performed": False,
    }
    (output_dir / "dev_oof_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {pair_id} {representation} spearman={primary['spearman']:.6f} sign_auc={primary['positive_vs_harmful_auroc']:.6f}")
    return payload


def main() -> None:
    args = parse_args()
    feature_path, root, search_path = Path(args.feature_table), Path(args.output_dir), Path(args.search_space)
    if "test" in {part.lower() for part in feature_path.parts}:
        raise RuntimeError("Phase 3A refuses TEST input paths")
    results = []
    for representation in args.representations:
        results.append(run_one(
            feature_path=feature_path,
            output_dir=root / representation.lower(),
            search_path=search_path,
            representation=representation,
            requested_device=args.device,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        ))
    manifest = {
        "schema_version": 1,
        "phase": "AACPI Phase 3A",
        "split": "dev",
        "feature_table": portable_path(feature_path),
        "representations": args.representations,
        "runs": results,
        "test_rows_accessed": 0,
        "test_evaluation_commands": 0,
        "policy_evaluation_performed": False,
    }
    (root / "representation_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
