import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.experiment_modeling import (
    join_eval_fields,
    normalize_binary_train_rows,
    read_table,
    stratified_calibration_split,
)
from router.experiment_utils import materialize_policy_rows, summarize_policy_rows
from router.io_utils import write_csv
from router.router_models import train_logistic_router, train_xgb_router


OUTPUT_HEADER = [
    "model",
    "delta",
    "feature_set",
    "calibration",
    "best_tau",
    "best_mrr",
    "best_coverage",
    "best_gain_precision",
    "delta_to_uncalibrated_best",
    "n_train_base",
    "n_train_calibration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dev-calibrated clean probability routing.")
    parser.add_argument("--train-table", required=True)
    parser.add_argument("--test-table", required=True)
    parser.add_argument("--eval-targets", required=True)
    parser.add_argument("--model-type", required=True, choices=["logistic", "xgb"])
    parser.add_argument("--feature-set", default="C4")
    parser.add_argument("--delta", default="0.01")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--calibration-methods", nargs="+", default=["none", "platt", "isotonic"])
    parser.add_argument("--calibration-frac", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-scan-csv", default=None)
    return parser.parse_args()


def fit_base_router(rows: list[dict], model_type: str, feature_set: str, random_state: int):
    if model_type == "logistic":
        return train_logistic_router(rows, feature_set=feature_set, random_state=random_state, router_mode="clean")
    if model_type == "xgb":
        return train_xgb_router(rows, feature_set=feature_set, random_state=random_state, router_mode="clean")
    raise ValueError(f"Unsupported model_type: {model_type}")


def fit_calibrator(method: str, probs: list[float], labels: list[int], random_state: int):
    x = np.asarray(probs, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    if method == "none":
        return None
    if method == "platt":
        calibrator = LogisticRegression(random_state=random_state, solver="lbfgs")
        calibrator.fit(x, y)
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(np.asarray(probs, dtype=float), y)
        return calibrator
    raise ValueError(f"Unsupported calibration method: {method}")


def apply_calibration(method: str, calibrator, probs: list[float]) -> list[float]:
    if method == "none" or calibrator is None:
        return [float(value) for value in probs]
    if method == "platt":
        x = np.asarray(probs, dtype=float).reshape(-1, 1)
        return [float(value) for value in calibrator.predict_proba(x)[:, 1]]
    if method == "isotonic":
        return [float(value) for value in calibrator.predict(np.asarray(probs, dtype=float))]
    raise ValueError(f"Unsupported calibration method: {method}")


def main() -> None:
    args = parse_args()
    delta_str = f"{float(args.delta):.2f}"
    train_rows = normalize_binary_train_rows(read_table(args.train_table), delta_str)
    base_train_rows, calib_rows = stratified_calibration_split(
        train_rows,
        label_key="label_gain",
        test_size=float(args.calibration_frac),
        random_state=int(args.random_state),
    )
    test_rows = join_eval_fields(read_table(args.test_table), read_table(args.eval_targets), delta_str=delta_str)

    artifact = fit_base_router(base_train_rows, args.model_type, args.feature_set, int(args.random_state))
    calib_probs = artifact.predict_proba_from_rows(calib_rows)
    calib_labels = [int(row["label_gain"]) for row in calib_rows]
    raw_test_probs = artifact.predict_proba_from_rows(test_rows)

    scan_rows = []
    best_rows = []
    uncalibrated_best = None
    for method in args.calibration_methods:
        calibrator = fit_calibrator(method, calib_probs, calib_labels, int(args.random_state))
        calibrated_probs = apply_calibration(method, calibrator, raw_test_probs)
        scored_rows = []
        for row, prob in zip(test_rows, calibrated_probs):
            merged = dict(row)
            merged["router_prob"] = float(prob)
            scored_rows.append(merged)

        method_rows = []
        for tau in args.taus:
            routed_rows = materialize_policy_rows(
                rows=scored_rows,
                decision_fn=lambda row, tau=tau: (int(float(row["router_prob"]) >= float(tau)), tau, args.model_type),
                policy_name="clean_prob_calibration",
                config_id=f"{method}|tau={tau:.1f}",
            )
            summary = summarize_policy_rows(
                routed_rows,
                delta=float(delta_str),
                extra={
                    "model": args.model_type,
                    "delta": delta_str,
                    "feature_set": args.feature_set,
                    "calibration": method,
                    "tau": float(tau),
                    "n_train_base": len(base_train_rows),
                    "n_train_calibration": len(calib_rows),
                },
            )
            method_rows.append(summary)
            scan_rows.append(summary)

        method_rows.sort(key=lambda row: -float(row["overall_mrr"]))
        best = dict(method_rows[0])
        best_row = {
            "model": args.model_type,
            "delta": delta_str,
            "feature_set": args.feature_set,
            "calibration": method,
            "best_tau": float(best["tau"]),
            "best_mrr": float(best["overall_mrr"]),
            "best_coverage": float(best["fusion_coverage"]),
            "best_gain_precision": float(best["gain_precision"]),
            "delta_to_uncalibrated_best": 0.0,
            "n_train_base": len(base_train_rows),
            "n_train_calibration": len(calib_rows),
        }
        if method == "none":
            uncalibrated_best = best_row
        best_rows.append(best_row)

    if uncalibrated_best is not None:
        base_best = float(uncalibrated_best["best_mrr"])
        for row in best_rows:
            row["delta_to_uncalibrated_best"] = float(row["best_mrr"]) - base_best

    write_csv(args.out_csv, best_rows, OUTPUT_HEADER)
    if args.out_scan_csv:
        write_csv(args.out_scan_csv, scan_rows, sorted(scan_rows[0].keys()))
    print(f"[OK] wrote calibration summary -> {Path(args.out_csv).as_posix()}")
    if args.out_scan_csv:
        print(f"[OK] wrote calibration scan    -> {Path(args.out_scan_csv).as_posix()}")


if __name__ == "__main__":
    main()
