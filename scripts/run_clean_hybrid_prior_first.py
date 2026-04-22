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
    "prior_low_cutoff",
    "prior_high_cutoff",
    "uncertain_tau",
    "high_policy",
    "n_queries",
    "overall_mrr",
    "hits1",
    "hits3",
    "hits10",
    "fusion_coverage",
    "gain_precision",
    "fraction_direct_residual",
    "fraction_direct_fusion",
    "fraction_learned_zone",
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
    parser = argparse.ArgumentParser(description="Run hybrid prior-first policy scan on clean router predictions.")
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--model", required=True, choices=["logistic", "xgb"])
    parser.add_argument("--delta", default="0.01")
    parser.add_argument("--feature-set", default="C4")
    parser.add_argument("--a-values", nargs="+", type=float, default=[-0.02, 0.0])
    parser.add_argument("--b-values", nargs="+", type=float, default=[0.02, 0.05, 0.08])
    parser.add_argument("--uncertain-taus", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--high-policy", choices=["fusion", "learned"], default="fusion")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-query-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = load_prediction_rows(args.prediction_csv)
    rows = merge_clean_feature_columns(
        base_rows,
        args.feature_csv,
        ["observed_has_img", "relation_gain_prior", "relation_support", "relation_is_visual_prior"],
    )
    delta = float(args.delta)

    output_rows = []
    query_rows = []
    for a in args.a_values:
        for b in args.b_values:
            if b <= a:
                continue
            for uncertain_tau in args.uncertain_taus:
                counts = {"direct_residual": 0, "direct_fusion": 0, "learned_zone": 0}
                config_id = f"a={a:.2f}|b={b:.2f}|tau={uncertain_tau:.1f}|high={args.high_policy}"

                def decision_fn(row: dict):
                    prior = float(row["relation_gain_prior"])
                    observed_has_img = int(float(row["observed_has_img"]))
                    if prior <= a:
                        counts["direct_residual"] += 1
                        return 0, 1.1, "hybrid_prior_rule"
                    if prior >= b and observed_has_img == 1:
                        if args.high_policy == "fusion":
                            counts["direct_fusion"] += 1
                            return 1, -0.1, "hybrid_prior_rule"
                        counts["learned_zone"] += 1
                        return int(float(row["router_prob"]) >= float(uncertain_tau)), uncertain_tau, args.model
                    counts["learned_zone"] += 1
                    return int(float(row["router_prob"]) >= float(uncertain_tau)), uncertain_tau, args.model

                routed_rows = materialize_policy_rows(
                    rows=rows,
                    decision_fn=decision_fn,
                    policy_name="clean_hybrid_prior_first_policy",
                    config_id=config_id,
                )
                n_total = max(1, len(routed_rows))
                output_rows.append(
                    summarize_policy_rows(
                        routed_rows,
                        delta=delta,
                        extra={
                            "policy_name": "clean_hybrid_prior_first_policy",
                            "model": args.model,
                            "delta": f"{delta:.2f}",
                            "feature_set": args.feature_set,
                            "prior_low_cutoff": float(a),
                            "prior_high_cutoff": float(b),
                            "uncertain_tau": float(uncertain_tau),
                            "high_policy": args.high_policy,
                            "fraction_direct_residual": counts["direct_residual"] / n_total,
                            "fraction_direct_fusion": counts["direct_fusion"] / n_total,
                            "fraction_learned_zone": counts["learned_zone"] / n_total,
                        },
                    )
                )
                query_rows.extend(routed_rows)

    output_rows.sort(key=lambda row: -float(row["overall_mrr"]))
    write_csv(args.out_csv, output_rows, OUTPUT_HEADER)
    if args.out_query_csv:
        write_query_rows(args.out_query_csv, query_rows)
    print(f"[OK] wrote hybrid scan      -> {Path(args.out_csv).as_posix()}")
    if args.out_query_csv:
        print(f"[OK] wrote query-level rows -> {Path(args.out_query_csv).as_posix()}")


if __name__ == "__main__":
    main()
