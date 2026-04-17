import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv, write_json
from router.metrics import compute_binary_metrics
from router.router_models import (
    RuleBasedRouter,
    compute_feature_importance_rows,
    train_logistic_router,
    train_xgb_router,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", nargs="+", default=["0.00", "0.01", "0.02"])
    ap.add_argument("--models", nargs="+", default=["rule", "logistic", "xgb"], help="rule logistic xgb")
    ap.add_argument("--feature-set", default="FULL", help="FULL | F1 | F2 | F3 | F4")
    ap.add_argument("--train-feature-dir", default="outputs/router/features")
    ap.add_argument("--out-model-dir", default="outputs/router/models")
    ap.add_argument("--out-eval-dir", default="outputs/router/eval")
    ap.add_argument("--rule-gamma", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.5, help="classification threshold for train metrics")
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()


def load_train_rows(base_dir: Path, delta_tag: str) -> list[dict]:
    path = base_dir / f"router_train_dev_delta_{delta_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing train feature file: {path}")
    return read_csv(path)


def train_one_model(model_name: str, rows: list[dict], feature_set: str, rule_gamma: float, random_state: int):
    if model_name == "rule":
        return RuleBasedRouter(gamma=rule_gamma)
    if model_name == "logistic":
        return train_logistic_router(rows, feature_set=feature_set, random_state=random_state)
    if model_name == "xgb":
        return train_xgb_router(rows, feature_set=feature_set, random_state=random_state)
    raise ValueError(f"Unsupported model: {model_name}")


def predict_for_rows(model_name: str, artifact, rows: list[dict], threshold: float) -> tuple[list[float], list[int]]:
    probs = artifact.predict_proba_from_rows(rows)
    preds = [int(p >= threshold) for p in probs]
    return probs, preds


def main() -> None:
    args = parse_args()
    train_feature_dir = Path(args.train_feature_dir)
    out_model_dir = Path(args.out_model_dir)
    out_eval_dir = Path(args.out_eval_dir)

    for delta_tag in args.deltas:
        rows = load_train_rows(train_feature_dir, delta_tag)
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
            probs, preds = predict_for_rows(model_name, artifact, rows, args.threshold)
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
                keep_rows = [
                    row
                    for row in existing_rows
                    if row.get("model") not in set(args.models)
                ]
                merged_rows = keep_rows + feature_importance_rows
            write_csv(importance_path, merged_rows, ["delta", "model", "feature_set", "feature_name", "importance"])
            print(f"[OK] wrote importance     -> {importance_path.as_posix()}")


if __name__ == "__main__":
    main()
