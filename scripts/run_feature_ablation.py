import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv
from router.metrics import compute_binary_metrics
from router.router_models import FEATURE_SETS, train_logistic_router, train_xgb_router
from router.routing_utils import compute_eval_summary, compute_gain_precision, fusion_ratio_by_regime, hard_route, select_expert_row


OUTPUT_HEADER = [
    "model",
    "feature_set",
    "delta",
    "tau",
    "train_auc",
    "train_f1",
    "train_precision",
    "train_recall",
    "train_balanced_accuracy",
    "train_positive_rate",
    "overall_mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
    "gain_precision",
    "fusion_ratio_head_has_img",
    "fusion_ratio_head_no_img",
    "fusion_ratio_tail_no_img",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", default="0.01")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--models", nargs="+", default=["logistic", "xgb"])
    ap.add_argument("--feature-sets", nargs="+", default=["F1", "F2", "F3", "F4"])
    ap.add_argument("--train-feature-dir", default="outputs/router/features")
    ap.add_argument("--test-features", default="outputs/router/features/router_test_features.csv")
    ap.add_argument("--out-csv", default="outputs/router/eval/feature_ablation.csv")
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()


def delta_tag(delta: str) -> str:
    return f"{float(delta):.2f}"


def train_artifact(model_name: str, rows: list[dict], feature_set: str, random_state: int):
    if model_name == "logistic":
        return train_logistic_router(rows, feature_set=feature_set, random_state=random_state)
    if model_name == "xgb":
        return train_xgb_router(rows, feature_set=feature_set, random_state=random_state)
    raise ValueError(f"Unsupported ablation model: {model_name}")


def build_routed_rows(rows: list[dict], probs: list[float], tau: float, selected_by: str) -> list[dict]:
    routed_rows = []
    for row, prob in zip(rows, probs):
        merged = dict(row)
        merged["router_prob"] = float(prob)
        merged["threshold"] = float(tau)
        use_fusion = hard_route(prob, tau)
        routed_rows.append(select_expert_row(merged, use_fusion, selected_by=selected_by))
    return routed_rows


def main() -> None:
    args = parse_args()
    delta_str = delta_tag(args.delta)
    train_rows = read_csv(Path(args.train_feature_dir) / f"router_train_dev_delta_{delta_str}.csv")
    test_rows = read_csv(args.test_features)
    y_true = [int(row["label_gain"]) for row in train_rows]

    output_rows = []
    for model_name in args.models:
        for feature_set in args.feature_sets:
            if feature_set not in FEATURE_SETS:
                raise ValueError(f"Unknown feature set: {feature_set}")

            artifact = train_artifact(model_name, train_rows, feature_set, args.random_state)
            train_probs = artifact.predict_proba_from_rows(train_rows)
            train_preds = [int(p >= 0.5) for p in train_probs]
            train_metrics = compute_binary_metrics(y_true, train_preds, train_probs)

            test_probs = artifact.predict_proba_from_rows(test_rows)
            routed_rows = build_routed_rows(test_rows, test_probs, args.tau, selected_by=model_name)
            eval_payload = compute_eval_summary(routed_rows)
            regime_ratios = fusion_ratio_by_regime(routed_rows)

            output_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "delta": delta_str,
                    "tau": float(args.tau),
                    "train_auc": train_metrics["auc"],
                    "train_f1": train_metrics["f1"],
                    "train_precision": train_metrics["precision"],
                    "train_recall": train_metrics["recall"],
                    "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "train_positive_rate": train_metrics["positive_rate"],
                    "overall_mrr": eval_payload["overall"]["mrr"],
                    "hits1": eval_payload["overall"]["hits1"],
                    "hits3": eval_payload["overall"]["hits3"],
                    "hits10": eval_payload["overall"]["hits10"],
                    "fusion_coverage": eval_payload["overall"]["fusion_coverage"],
                    "gain_precision": compute_gain_precision(routed_rows, delta=float(delta_str)),
                    "fusion_ratio_head_has_img": regime_ratios.get("head_has_img", 0.0),
                    "fusion_ratio_head_no_img": regime_ratios.get("head_no_img", 0.0),
                    "fusion_ratio_tail_no_img": regime_ratios.get("tail_no_img", 0.0),
                }
            )

    write_csv(args.out_csv, output_rows, OUTPUT_HEADER)
    print(f"[OK] wrote feature ablation -> {Path(args.out_csv).as_posix()}")


if __name__ == "__main__":
    main()
