import argparse
import json
import pickle
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv, write_json
from router.metrics import compute_binary_metrics
from router.router_models import (
    FEATURE_SETS,
    RuleBasedRouter,
    compute_feature_importance_rows,
    train_logistic_router,
    train_xgb_router,
)


VALID_DELTAS = {"0.00": "gain_label_d0", "0.01": "gain_label_d001", "0.02": "gain_label_d002"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train router models in either legacy batch mode or contract mode.")

    parser.add_argument("--train-table", default=None, help="Contract mode: router_features_dev parquet/csv")
    parser.add_argument("--model-type", default=None, choices=["logistic", "xgb"], help="Contract mode only")
    parser.add_argument("--delta", default=None, help="Contract mode delta: 0.00 / 0.01 / 0.02")
    parser.add_argument("--out-dir", default=None, help="Contract mode output directory")

    parser.add_argument("--feature-set", default="FULL", help="FULL | F1 | F2 | F3 | F4")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5, help="classification threshold for train metrics")

    parser.add_argument("--deltas", nargs="+", default=["0.00", "0.01", "0.02"])
    parser.add_argument("--models", nargs="+", default=["rule", "logistic", "xgb"], help="Legacy mode: rule logistic xgb")
    parser.add_argument("--train-feature-dir", default="outputs/router/features")
    parser.add_argument("--out-model-dir", default="outputs/router/models")
    parser.add_argument("--out-eval-dir", default="outputs/router/eval")
    parser.add_argument("--rule-gamma", type=float, default=0.0)
    return parser.parse_args()


def read_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported file type: {path.as_posix()}")


def normalize_contract_rows(rows: list[dict], delta_str: str) -> list[dict]:
    label_name = VALID_DELTAS[delta_str]
    normalized = []
    for row in rows:
        out = dict(row)
        out["label_gain"] = int(row[label_name])
        normalized.append(out)
    return normalized


def train_one_model(model_name: str, rows: list[dict], feature_set: str, rule_gamma: float, random_state: int):
    if model_name == "rule":
        return RuleBasedRouter(gamma=rule_gamma)
    if model_name == "logistic":
        return train_logistic_router(rows, feature_set=feature_set, random_state=random_state)
    if model_name == "xgb":
        return train_xgb_router(rows, feature_set=feature_set, random_state=random_state)
    raise ValueError(f"Unsupported model: {model_name}")


def predict_for_rows(artifact, rows: list[dict], threshold: float) -> tuple[list[float], list[int]]:
    probs = artifact.predict_proba_from_rows(rows)
    preds = [int(p >= threshold) for p in probs]
    return probs, preds


def get_hyperparams(artifact) -> dict:
    if hasattr(artifact, "estimator") and hasattr(artifact.estimator, "get_params"):
        params = artifact.estimator.get_params()
        serializable = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                serializable[key] = value
            else:
                serializable[key] = str(value)
        return serializable
    return {}


def contract_mode(args: argparse.Namespace) -> None:
    if args.model_type is None or args.delta is None or args.out_dir is None:
        raise ValueError("Contract mode requires --train-table, --model-type, --delta, and --out-dir.")

    delta_str = f"{float(args.delta):.2f}"
    if delta_str not in VALID_DELTAS:
        raise ValueError(f"Unsupported delta: {args.delta}")
    if args.feature_set not in FEATURE_SETS:
        raise ValueError(f"Unsupported feature set: {args.feature_set}")

    train_path = Path(args.train_table)
    raw_rows = read_table(train_path)
    rows = normalize_contract_rows(raw_rows, delta_str)
    y_true = [int(row["label_gain"]) for row in rows]

    artifact = train_one_model(
        model_name=args.model_type,
        rows=rows,
        feature_set=args.feature_set,
        rule_gamma=args.rule_gamma,
        random_state=args.random_state,
    )
    probs, preds = predict_for_rows(artifact, rows, args.threshold)
    metrics = compute_binary_metrics(y_true, preds, probs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pkl"
    artifact.save(model_path)

    if getattr(artifact, "vectorizer", None) is not None:
        with (out_dir / "vectorizer.pkl").open("wb") as handle:
            pickle.dump(artifact.vectorizer, handle)

    feature_columns = list(FEATURE_SETS[args.feature_set])
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    positive_rate = float(sum(y_true) / len(y_true)) if y_true else 0.0
    summary = {
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "delta": float(delta_str),
        "delta_tag": delta_str,
        "n_train": len(rows),
        "positive_rate": positive_rate,
        "selected_features": feature_columns,
        "train_label_name": VALID_DELTAS[delta_str],
        "train_table": train_path.as_posix(),
        "threshold_for_metrics": float(args.threshold),
        "random_state": int(args.random_state),
        "hyperparams": get_hyperparams(artifact),
        "train_metrics": metrics,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] wrote model          -> {model_path.as_posix()}")
    print(f"[OK] wrote feature cols   -> {(out_dir / 'feature_columns.json').as_posix()}")
    print(f"[OK] wrote train summary  -> {(out_dir / 'train_summary.json').as_posix()}")
    if getattr(artifact, "vectorizer", None) is not None:
        print(f"[OK] wrote vectorizer     -> {(out_dir / 'vectorizer.pkl').as_posix()}")


def load_train_rows_legacy(base_dir: Path, delta_tag: str) -> list[dict]:
    path = base_dir / f"router_train_dev_delta_{delta_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing train feature file: {path}")
    return read_csv(path)


def legacy_mode(args: argparse.Namespace) -> None:
    train_feature_dir = Path(args.train_feature_dir)
    out_model_dir = Path(args.out_model_dir)
    out_eval_dir = Path(args.out_eval_dir)

    for delta_tag in args.deltas:
        rows = load_train_rows_legacy(train_feature_dir, delta_tag)
        y_true = [int(row["label_gain"]) for row in rows]

        metrics_payload = {
            "delta": delta_tag,
            "feature_set": args.feature_set,
            "threshold": float(args.threshold),
            "n_samples": len(rows),
            "models": {},
        }
        metrics_path = out_eval_dir / f"router_train_metrics_delta_{delta_tag}.json"
        if metrics_path.exists():
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_payload["delta"] = delta_tag
            metrics_payload["feature_set"] = args.feature_set
            metrics_payload["threshold"] = float(args.threshold)
            metrics_payload["n_samples"] = len(rows)
            metrics_payload.setdefault("models", {})
        feature_importance_rows: list[dict] = []

        for model_name in args.models:
            artifact = train_one_model(
                model_name=model_name,
                rows=rows,
                feature_set=args.feature_set,
                rule_gamma=args.rule_gamma,
                random_state=args.random_state,
            )
            probs, preds = predict_for_rows(artifact, rows, args.threshold)
            metrics_payload["models"][model_name] = compute_binary_metrics(y_true, preds, probs)

            if model_name in {"logistic", "xgb"}:
                model_path = out_model_dir / f"{model_name}_delta_{delta_tag}.pkl"
                artifact.save(model_path)
                print(f"[OK] wrote model          -> {model_path.as_posix()}")
                feature_importance_rows.extend(compute_feature_importance_rows(artifact))

        write_json(metrics_path, metrics_payload)
        print(f"[OK] wrote train metrics  -> {metrics_path.as_posix()}")

        if feature_importance_rows:
            for row in feature_importance_rows:
                row["delta"] = delta_tag
            importance_path = out_eval_dir / f"router_feature_importance_delta_{delta_tag}.csv"
            merged_rows = feature_importance_rows
            if importance_path.exists():
                existing_rows = read_csv(importance_path)
                keep_rows = [row for row in existing_rows if row.get("model") not in set(args.models)]
                merged_rows = keep_rows + feature_importance_rows
            write_csv(importance_path, merged_rows, ["delta", "model", "feature_set", "feature_name", "importance"])
            print(f"[OK] wrote importance     -> {importance_path.as_posix()}")


def main() -> None:
    args = parse_args()
    if args.train_table:
        contract_mode(args)
        return
    legacy_mode(args)


if __name__ == "__main__":
    main()
