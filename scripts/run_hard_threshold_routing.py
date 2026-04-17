import argparse
import pickle
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv, write_json
from router.router_models import RuleBasedRouter
from router.routing_utils import compute_eval_summary, hard_route, select_expert_row, subgroup_eval_rows


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
    ap.add_argument("--feature-set", default="FULL")
    ap.add_argument("--rule-gamma", type=float, default=0.0)
    ap.add_argument("--test-features", default="outputs/router/features/router_test_features.csv")
    ap.add_argument("--model-dir", default="outputs/router/models")
    ap.add_argument("--out-routing-dir", default="outputs/router/routing")
    ap.add_argument("--out-eval-dir", default="outputs/router/eval")
    return ap.parse_args()


def delta_tag(delta: str) -> str:
    return f"{float(delta):.2f}"


def tau_tag(tau: float) -> str:
    return f"{float(tau):.1f}"


def load_router(model_name: str, model_dir: Path, delta_str: str, rule_gamma: float):
    if model_name == "rule":
        return RuleBasedRouter(gamma=rule_gamma)
    model_path = model_dir / f"{model_name}_delta_{delta_str}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing router model: {model_path}")
    with model_path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    args = parse_args()
    delta_str = delta_tag(args.delta)
    tau_str = tau_tag(args.tau)
    rows = read_csv(args.test_features)
    router = load_router(args.model, Path(args.model_dir), delta_str, args.rule_gamma)

    probs = router.predict_proba_from_rows(rows)
    routed_rows = []
    for row, prob in zip(rows, probs):
        row = dict(row)
        row["router_prob"] = float(prob)
        row["threshold"] = float(args.tau)
        use_fusion = hard_route(prob, args.tau)
        routed_rows.append(select_expert_row(row, use_fusion, selected_by=args.model))

    routing_path = Path(args.out_routing_dir) / f"test_router_predictions_{args.model}_delta_{delta_str}_tau_{tau_str}.csv"
    write_csv(routing_path, routed_rows, PREDICTION_HEADER)
    print(f"[OK] wrote routed preds   -> {routing_path.as_posix()}")

    eval_payload = compute_eval_summary(routed_rows)
    eval_payload["model"] = args.model
    eval_payload["delta"] = delta_str
    eval_payload["tau"] = float(args.tau)
    eval_payload["feature_set"] = args.feature_set
    eval_path = Path(args.out_eval_dir) / f"router_eval_{args.model}_delta_{delta_str}_tau_{tau_str}.json"
    write_json(eval_path, eval_payload)
    print(f"[OK] wrote eval json      -> {eval_path.as_posix()}")

    subgroup_path = Path(args.out_eval_dir) / f"router_eval_{args.model}_delta_{delta_str}_tau_{tau_str}_by_regime.csv"
    write_csv(
        subgroup_path,
        subgroup_eval_rows(routed_rows),
        ["target_regime", "n_queries", "mrr", "hits1", "hits3", "hits10", "fusion_coverage"],
    )
    print(f"[OK] wrote subgroup eval  -> {subgroup_path.as_posix()}")


if __name__ == "__main__":
    main()
