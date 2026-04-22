import argparse
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


OUTPUT_HEADER = [
    "policy_name",
    "model",
    "delta",
    "feature_set",
    "bucket_definition",
    "tau_neg",
    "tau_mid",
    "tau_pos",
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
    parser = argparse.ArgumentParser(description="Run relation-prior bucket policy scan on clean router predictions.")
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--model", required=True, choices=["rule", "logistic", "xgb"])
    parser.add_argument("--delta", default="0.01")
    parser.add_argument("--feature-set", default="C4")
    parser.add_argument("--tau-neg", nargs="+", default=["force_residual", "0.5", "0.7", "0.9"])
    parser.add_argument("--tau-mid", nargs="+", default=["0.3", "0.5", "0.7", "0.9"])
    parser.add_argument("--tau-pos", nargs="+", default=["0.3", "0.5", "0.7", "0.9", "force_fusion"])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-query-csv", default=None)
    return parser.parse_args()


def bucket_name(prior: float) -> str:
    if prior < 0.0:
        return "neg"
    if prior < 0.05:
        return "mid"
    return "pos"


def resolve_action(prob: float, tau_token: str) -> tuple[int, float]:
    token = str(tau_token).strip().lower()
    if token == "force_residual":
        return 0, 1.1
    if token == "force_fusion":
        return 1, -0.1
    tau = float(token)
    return int(float(prob) >= tau), tau


def main() -> None:
    args = parse_args()
    rows = merge_clean_feature_columns(
        load_prediction_rows(args.prediction_csv),
        args.feature_csv,
        ["relation_gain_prior", "relation_support", "relation_is_visual_prior"],
    )
    delta = float(args.delta)

    output_rows = []
    query_rows = []
    for tau_neg in args.tau_neg:
        for tau_mid in args.tau_mid:
            for tau_pos in args.tau_pos:
                config_id = f"neg={tau_neg}|mid={tau_mid}|pos={tau_pos}"

                def decision_fn(row: dict):
                    prior = float(row["relation_gain_prior"])
                    bucket = bucket_name(prior)
                    tau_token = {"neg": tau_neg, "mid": tau_mid, "pos": tau_pos}[bucket]
                    use_fusion, applied_tau = resolve_action(row["router_prob"], tau_token)
                    return use_fusion, applied_tau, args.model

                routed_rows = materialize_policy_rows(
                    rows=rows,
                    decision_fn=decision_fn,
                    policy_name="clean_tau_by_relation_prior_bucket",
                    config_id=config_id,
                )
                output_rows.append(
                    summarize_policy_rows(
                        routed_rows,
                        delta=delta,
                        extra={
                            "policy_name": "clean_tau_by_relation_prior_bucket",
                            "model": args.model,
                            "delta": f"{delta:.2f}",
                            "feature_set": args.feature_set,
                            "bucket_definition": "neg:<0 | mid:[0,0.05) | pos:>=0.05",
                            "tau_neg": tau_neg,
                            "tau_mid": tau_mid,
                            "tau_pos": tau_pos,
                        },
                    )
                )
                query_rows.extend(routed_rows)

    output_rows.sort(key=lambda row: -float(row["overall_mrr"]))
    write_csv(args.out_csv, output_rows, OUTPUT_HEADER)
    if args.out_query_csv:
        write_query_rows(args.out_query_csv, query_rows)
    print(f"[OK] wrote prior-bucket scan -> {Path(args.out_csv).as_posix()}")
    if args.out_query_csv:
        print(f"[OK] wrote query-level rows -> {Path(args.out_query_csv).as_posix()}")


if __name__ == "__main__":
    main()
