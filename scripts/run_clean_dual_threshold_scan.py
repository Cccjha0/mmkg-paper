import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.experiment_utils import load_prediction_rows, materialize_policy_rows, summarize_policy_rows, write_query_rows
from router.io_utils import write_csv


OUTPUT_HEADER = [
    "policy_name",
    "model",
    "delta",
    "feature_set",
    "tau_head",
    "tau_tail",
    "n_queries",
    "overall_mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
    "gain_precision",
    "head_has_img_mrr",
    "head_no_img_mrr",
    "tail_no_img_mrr",
    "head_has_img_coverage",
    "head_no_img_coverage",
    "tail_no_img_coverage",
    "head_has_img_n_queries",
    "head_no_img_n_queries",
    "tail_no_img_n_queries",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direction-specific dual threshold scan on clean router predictions.")
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--model", required=True, choices=["logistic", "xgb"])
    parser.add_argument("--delta", default="0.01")
    parser.add_argument("--feature-set", default="C4")
    parser.add_argument("--tau-heads", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--tau-tails", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-query-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_prediction_rows(args.prediction_csv)
    delta = float(args.delta)

    output_rows = []
    query_rows = []
    for tau_head in args.tau_heads:
        for tau_tail in args.tau_tails:
            config_id = f"tau_head={tau_head:.1f}|tau_tail={tau_tail:.1f}"

            def decision_fn(row: dict):
                tau = tau_head if str(row["direction"]) == "head" else tau_tail
                return int(float(row["router_prob"]) >= float(tau)), tau, args.model

            routed_rows = materialize_policy_rows(
                rows=rows,
                decision_fn=decision_fn,
                policy_name="clean_dual_tau_by_direction",
                config_id=config_id,
            )
            output_rows.append(
                summarize_policy_rows(
                    routed_rows,
                    delta=delta,
                    extra={
                        "policy_name": "clean_dual_tau_by_direction",
                        "model": args.model,
                        "delta": f"{delta:.2f}",
                        "feature_set": args.feature_set,
                        "tau_head": float(tau_head),
                        "tau_tail": float(tau_tail),
                    },
                )
            )
            query_rows.extend(routed_rows)

    output_rows.sort(key=lambda row: (-float(row["overall_mrr"]), float(row["tau_head"]), float(row["tau_tail"])))
    write_csv(args.out_csv, output_rows, OUTPUT_HEADER)
    if args.out_query_csv:
        write_query_rows(args.out_query_csv, query_rows)
    print(f"[OK] wrote dual-threshold scan -> {Path(args.out_csv).as_posix()}")
    if args.out_query_csv:
        print(f"[OK] wrote query-level rows  -> {Path(args.out_query_csv).as_posix()}")


if __name__ == "__main__":
    main()
