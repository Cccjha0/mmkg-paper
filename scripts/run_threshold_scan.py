import argparse
import pickle
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv
from router.router_models import RuleBasedRouter
from router.routing_utils import compute_eval_summary, compute_gain_precision, fusion_ratio_by_regime, hard_route, select_expert_row


OUTPUT_HEADER = [
    "model",
    "delta",
    "tau",
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
    ap.add_argument("--model", required=True, choices=["rule", "logistic", "xgb"])
    ap.add_argument("--delta", required=True, help="e.g. 0.01")
    ap.add_argument("--taus", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--test-features", default="outputs/router/features/router_test_features.csv")
    ap.add_argument("--model-dir", default="outputs/router/models")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--rule-gamma", type=float, default=0.0)
    return ap.parse_args()


def delta_tag(delta: str) -> str:
    return f"{float(delta):.2f}"


def load_router(model_name: str, model_dir: Path, delta_str: str, rule_gamma: float):
    if model_name == "rule":
        return RuleBasedRouter(gamma=rule_gamma)
    model_path = model_dir / f"{model_name}_delta_{delta_str}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing router model: {model_path}")
    with model_path.open("rb") as f:
        return pickle.load(f)


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
    rows = read_csv(args.test_features)
    router = load_router(args.model, Path(args.model_dir), delta_str, args.rule_gamma)
    probs = router.predict_proba_from_rows(rows)

    output_rows = []
    for tau in args.taus:
        routed_rows = build_routed_rows(rows, probs, tau, selected_by=args.model)
        eval_payload = compute_eval_summary(routed_rows)
        regime_ratios = fusion_ratio_by_regime(routed_rows)
        output_rows.append(
            {
                "model": args.model,
                "delta": delta_str,
                "tau": float(tau),
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

    out_csv = Path(args.out_csv) if args.out_csv else Path("outputs/router/eval") / f"threshold_scan_{args.model}_delta_{delta_str}.csv"
    write_csv(out_csv, output_rows, OUTPUT_HEADER)
    print(f"[OK] wrote threshold scan -> {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
