import argparse
import json
import pickle
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv, write_json
from router.router_models import CleanRuleBasedRouter, PosthocRuleBasedRouter
from router.routing_utils import compute_eval_summary_with_std, hard_route, select_expert_row, subgroup_eval_rows


PREDICTION_HEADER = [
    "query_id",
    "split",
    "seed",
    "direction",
    "target_regime",
    "relation_id",
    "selected_by",
    "router_prob",
    "threshold",
    "use_fusion",
    "selected_expert",
    "rank_final",
    "rr_final",
    "rank_gate",
    "rr_gate",
    "rank_residual",
    "rr_residual",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rule", "logistic", "xgb"])
    ap.add_argument("--delta", required=True, help="e.g. 0.01")
    ap.add_argument("--tau", type=float, required=True, help="hard threshold such as 0.5")
    ap.add_argument("--router-mode", default="posthoc", choices=["clean", "posthoc"])
    ap.add_argument("--feature-set", default=None)
    ap.add_argument("--rule-gamma", type=float, default=0.0)
    ap.add_argument("--test-features", default=None)
    ap.add_argument("--eval-targets", default="outputs/router/features/router_eval_targets_shared_test.parquet")
    ap.add_argument("--model-dir", default="outputs/router/models")
    ap.add_argument("--out-routing-dir", default="outputs/router/routing")
    ap.add_argument("--out-eval-dir", default="outputs/router/eval")
    return ap.parse_args()


def delta_tag(delta: str) -> str:
    return f"{float(delta):.2f}"


def tau_tag(tau: float) -> str:
    return f"{float(tau):.1f}"


def default_test_features(router_mode: str) -> str:
    if router_mode == "clean":
        return "outputs/router/features/router_test_clean_features.csv"
    return "outputs/router/features/router_test_posthoc_features.csv"


def infer_from_train_summary(model_dir: Path) -> dict:
    summary_path = model_dir / "train_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing train_summary.json under {model_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_router(model_name: str, model_dir: Path, router_mode: str, rule_gamma: float):
    if model_name == "rule":
        if router_mode == "clean":
            return CleanRuleBasedRouter(gamma=rule_gamma), {"feature_set": "rule", "is_query_time_legal": True}
        return PosthocRuleBasedRouter(gamma=rule_gamma), {"feature_set": "rule", "is_query_time_legal": False}

    summary = infer_from_train_summary(model_dir)
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing router model: {model_path}")
    with model_path.open("rb") as f:
        router = pickle.load(f)
    return router, {
        "feature_set": summary["feature_set"],
        "is_query_time_legal": bool(summary.get("is_query_time_legal", summary.get("router_mode") == "clean")),
    }


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


def main() -> None:
    args = parse_args()
    delta_str = delta_tag(args.delta)
    tau_str = tau_tag(args.tau)
    test_features = args.test_features or default_test_features(args.router_mode)
    rows = read_csv(test_features)
    eval_targets = load_eval_targets(args.eval_targets)

    model_dir = Path(args.model_dir)
    if args.model != "rule" and model_dir.name != args.router_mode:
        candidate = model_dir / args.router_mode
        if candidate.exists():
            model_dir = candidate

    router, meta = load_router(args.model, model_dir, args.router_mode, args.rule_gamma)
    probs = router.predict_proba_from_rows(rows)
    feature_set = args.feature_set or meta["feature_set"]

    routed_rows = []
    for row, prob in zip(rows, probs):
        route_row = dict(row)
        route_row["router_prob"] = float(prob)
        route_row["threshold"] = float(args.tau)
        use_fusion = hard_route(prob, args.tau)
        target = eval_targets[str(row["query_id"])]
        route_row["target_regime"] = target["target_regime"]
        routed_rows.append(
            select_expert_row(
                route_row,
                {
                    "target_regime": target["target_regime"],
                    "rank_gate": target.get("rank_gate", 0),
                    "rr_gate": target["rr_gate"],
                    "rank_residual": target.get("rank_residual", 0),
                    "rr_residual": target["rr_residual"],
                },
                use_fusion,
                selected_by=args.model,
            )
        )

    routing_path = (
        Path(args.out_routing_dir)
        / args.router_mode
        / f"test_router_predictions_{args.router_mode}_{args.model}_delta_{delta_str}_tau_{tau_str}_{feature_set}.csv"
    )
    write_csv(routing_path, routed_rows, PREDICTION_HEADER)
    print(f"[OK] wrote routed preds   -> {routing_path.as_posix()}")

    eval_payload = compute_eval_summary_with_std(routed_rows)
    eval_payload["router_mode"] = args.router_mode
    eval_payload["model"] = args.model
    eval_payload["delta"] = delta_str
    eval_payload["tau"] = float(args.tau)
    eval_payload["feature_set"] = feature_set
    eval_payload["is_query_time_legal"] = bool(meta["is_query_time_legal"])
    eval_path = (
        Path(args.out_eval_dir)
        / args.router_mode
        / f"router_eval_{args.router_mode}_{args.model}_delta_{delta_str}_tau_{tau_str}_{feature_set}.json"
    )
    write_json(eval_path, eval_payload)
    print(f"[OK] wrote eval json      -> {eval_path.as_posix()}")

    subgroup_path = (
        Path(args.out_eval_dir)
        / args.router_mode
        / f"router_eval_{args.router_mode}_{args.model}_delta_{delta_str}_tau_{tau_str}_{feature_set}_by_regime.csv"
    )
    write_csv(
        subgroup_path,
        subgroup_eval_rows(routed_rows),
        ["target_regime", "n_queries", "mrr", "hits1", "hits3", "hits10", "fusion_coverage"],
    )
    print(f"[OK] wrote subgroup eval  -> {subgroup_path.as_posix()}")


if __name__ == "__main__":
    main()
