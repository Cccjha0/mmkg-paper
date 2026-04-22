import argparse
from itertools import product
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.experiment_utils import (
    load_prediction_rows,
    materialize_policy_rows,
    merge_clean_feature_columns,
    summarize_policy_rows,
    write_query_rows,
)
from router.io_utils import write_csv


GROUPS = [
    ("head", 0, "head_obs0"),
    ("head", 1, "head_obs1"),
    ("tail", 0, "tail_obs0"),
    ("tail", 1, "tail_obs1"),
]

OUTPUT_HEADER = [
    "policy_name",
    "model",
    "delta",
    "feature_set",
    "tau_head_obs0",
    "tau_head_obs1",
    "tau_tail_obs0",
    "tau_tail_obs1",
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
    parser = argparse.ArgumentParser(
        description="Run direction x observed_has_img conditional threshold scan on clean router predictions."
    )
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--model", required=True, choices=["logistic", "xgb"])
    parser.add_argument("--delta", default="0.01")
    parser.add_argument("--feature-set", default="C4")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-query-csv", default=None)
    return parser.parse_args()


def group_key(row: dict) -> tuple[str, int]:
    return str(row["direction"]), int(float(row["observed_has_img"]))


def main() -> None:
    args = parse_args()
    rows = merge_clean_feature_columns(
        load_prediction_rows(args.prediction_csv),
        args.feature_csv,
        ["observed_has_img"],
    )
    delta = float(args.delta)

    output_rows = []
    query_rows = []
    for tau_head_obs0, tau_head_obs1, tau_tail_obs0, tau_tail_obs1 in product(args.taus, repeat=4):
        tau_map = {
            ("head", 0): float(tau_head_obs0),
            ("head", 1): float(tau_head_obs1),
            ("tail", 0): float(tau_tail_obs0),
            ("tail", 1): float(tau_tail_obs1),
        }
        config_id = (
            f"head_obs0={tau_head_obs0:.1f}|head_obs1={tau_head_obs1:.1f}|"
            f"tail_obs0={tau_tail_obs0:.1f}|tail_obs1={tau_tail_obs1:.1f}"
        )

        def decision_fn(row: dict):
            tau = tau_map[group_key(row)]
            return int(float(row["router_prob"]) >= tau), tau, args.model

        routed_rows = materialize_policy_rows(
            rows=rows,
            decision_fn=decision_fn,
            policy_name="clean_tau_by_direction_and_observed_img",
            config_id=config_id,
        )
        output_rows.append(
            summarize_policy_rows(
                routed_rows,
                delta=delta,
                extra={
                    "policy_name": "clean_tau_by_direction_and_observed_img",
                    "model": args.model,
                    "delta": f"{delta:.2f}",
                    "feature_set": args.feature_set,
                    "tau_head_obs0": float(tau_head_obs0),
                    "tau_head_obs1": float(tau_head_obs1),
                    "tau_tail_obs0": float(tau_tail_obs0),
                    "tau_tail_obs1": float(tau_tail_obs1),
                },
            )
        )
        query_rows.extend(routed_rows)

    output_rows.sort(key=lambda row: -float(row["overall_mrr"]))
    write_csv(args.out_csv, output_rows, OUTPUT_HEADER)
    if args.out_query_csv:
        write_query_rows(args.out_query_csv, query_rows)
    print(f"[OK] wrote group-conditional scan -> {Path(args.out_csv).as_posix()}")
    if args.out_query_csv:
        print(f"[OK] wrote query-level rows     -> {Path(args.out_query_csv).as_posix()}")


if __name__ == "__main__":
    main()
