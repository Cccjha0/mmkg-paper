import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv
from router.metrics import compute_binary_metrics
from router.router_models import get_feature_sets, train_logistic_router, train_xgb_router
from router.routing_utils import compute_eval_summary, compute_gain_precision, fusion_ratio_by_regime, hard_route, select_expert_row


OUTPUT_HEADER = [
    "router_mode",
    "feature_set",
    "is_query_time_legal",
    "model",
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
    ap.add_argument("--router-mode", default="posthoc", choices=["clean", "posthoc"])
    ap.add_argument("--models", nargs="+", default=["logistic", "xgb"])
    ap.add_argument("--feature-sets", nargs="+", default=None)
    ap.add_argument("--train-feature-dir", default="outputs/router/features")
    ap.add_argument("--test-features", default=None)
    ap.add_argument("--eval-targets", default="outputs/router/features/router_eval_targets_shared_test.parquet")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()


def delta_tag(delta: str) -> str:
    return f"{float(delta):.2f}"


def default_train_path(train_feature_dir: str, router_mode: str, delta_str: str) -> Path:
    return Path(train_feature_dir) / f"router_train_dev_{router_mode}_delta_{delta_str}.csv"


def default_test_path(router_mode: str) -> str:
    if router_mode == "clean":
        return "outputs/router/features/router_test_clean_features.csv"
    return "outputs/router/features/router_test_posthoc_features.csv"


def train_artifact(model_name: str, rows: list[dict], feature_set: str, random_state: int, router_mode: str):
    if model_name == "logistic":
        return train_logistic_router(rows, feature_set=feature_set, random_state=random_state, router_mode=router_mode)
    if model_name == "xgb":
        return train_xgb_router(rows, feature_set=feature_set, random_state=random_state, router_mode=router_mode)
    raise ValueError(f"Unsupported ablation model: {model_name}")


def build_eval_meta_row(row: dict) -> dict:
    return {
        "target_regime": row["target_regime"],
        "rank_gate": row.get("rank_fusion", row.get("rank_gate", 0)),
        "rr_gate": row.get("rr_fusion", row.get("rr_gate", 0.0)),
        "rank_residual": row.get("rank_struct", row.get("rank_residual", 0)),
        "rr_residual": row.get("rr_struct", row.get("rr_residual", 0.0)),
    }


def load_eval_targets(path: str) -> dict[str, dict]:
    rows = read_csv(path) if str(path).lower().endswith(".csv") else __import__("pandas").read_parquet(path).to_dict(orient="records")
    return {str(row["query_id"]): row for row in rows}


def build_routed_rows(rows: list[dict], probs: list[float], tau: float, selected_by: str, eval_targets: dict[str, dict]) -> list[dict]:
    routed_rows = []
    for row, prob in zip(rows, probs):
        merged = dict(row)
        merged["router_prob"] = float(prob)
        merged["threshold"] = float(tau)
        use_fusion = hard_route(prob, tau)
        target = eval_targets[str(row["query_id"])]
        merged["target_regime"] = target["target_regime"]
        routed_rows.append(
            select_expert_row(
                merged,
                {
                    "target_regime": target["target_regime"],
                    "rank_gate": target.get("rank_gate", 0),
                    "rr_gate": target["rr_gate"],
                    "rank_residual": target.get("rank_residual", 0),
                    "rr_residual": target["rr_residual"],
                },
                use_fusion,
                selected_by=selected_by,
            )
        )
    return routed_rows


def main() -> None:
    args = parse_args()
    delta_str = delta_tag(args.delta)
    feature_sets = args.feature_sets or list(get_feature_sets(args.router_mode).keys())
    train_rows = read_csv(default_train_path(args.train_feature_dir, args.router_mode, delta_str))
    test_rows = read_csv(args.test_features or default_test_path(args.router_mode))
    eval_targets = load_eval_targets(args.eval_targets)
    y_true = [int(row["label_gain"]) for row in train_rows]

    output_rows = []
    for model_name in args.models:
        for feature_set in feature_sets:
            if feature_set not in get_feature_sets(args.router_mode):
                raise ValueError(f"Unknown feature set for {args.router_mode}: {feature_set}")

            artifact = train_artifact(model_name, train_rows, feature_set, args.random_state, args.router_mode)
            train_probs = artifact.predict_proba_from_rows(train_rows)
            train_preds = [int(p >= 0.5) for p in train_probs]
            train_metrics = compute_binary_metrics(y_true, train_preds, train_probs)

            test_probs = artifact.predict_proba_from_rows(test_rows)
            routed_rows = build_routed_rows(test_rows, test_probs, args.tau, selected_by=model_name, eval_targets=eval_targets)
            eval_payload = compute_eval_summary(routed_rows)
            regime_ratios = fusion_ratio_by_regime(routed_rows)

            output_rows.append(
                {
                    "router_mode": args.router_mode,
                    "feature_set": feature_set,
                    "is_query_time_legal": args.router_mode == "clean",
                    "model": model_name,
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

    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else Path("outputs/router/eval") / args.router_mode / f"feature_ablation_{args.router_mode}.csv"
    )
    write_csv(out_csv, output_rows, OUTPUT_HEADER)
    print(f"[OK] wrote feature ablation -> {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
