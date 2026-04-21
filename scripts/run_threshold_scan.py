import argparse
import json
import pickle
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv
from router.router_models import CleanRuleBasedRouter, PosthocRuleBasedRouter
from router.routing_utils import compute_eval_summary, compute_gain_precision, fusion_ratio_by_regime, hard_route, select_expert_row


OUTPUT_HEADER = [
    "router_mode",
    "feature_set",
    "is_query_time_legal",
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
    ap.add_argument("--router-mode", default="posthoc", choices=["clean", "posthoc"])
    ap.add_argument("--feature-set", default=None)
    ap.add_argument("--taus", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--test-features", default=None)
    ap.add_argument("--eval-targets", default="outputs/router/features/router_eval_targets_shared_test.parquet")
    ap.add_argument("--model-dir", default="outputs/router/models")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--rule-gamma", type=float, default=0.0)
    return ap.parse_args()


def delta_tag(delta: str) -> str:
    return f"{float(delta):.2f}"


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

    output_rows = []
    for tau in args.taus:
        routed_rows = build_routed_rows(rows, probs, tau, selected_by=args.model, eval_targets=eval_targets)
        eval_payload = compute_eval_summary(routed_rows)
        regime_ratios = fusion_ratio_by_regime(routed_rows)
        output_rows.append(
            {
                "router_mode": args.router_mode,
                "feature_set": feature_set,
                "is_query_time_legal": bool(meta["is_query_time_legal"]),
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

    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else Path("outputs/router/eval") / args.router_mode / f"threshold_scan_{args.router_mode}_{args.model}_delta_{delta_str}_{feature_set}.csv"
    )
    write_csv(out_csv, output_rows, OUTPUT_HEADER)
    print(f"[OK] wrote threshold scan -> {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
