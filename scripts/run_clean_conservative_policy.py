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
    "delta",
    "support_min",
    "cosine_min",
    "require_visual_prior",
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
    parser = argparse.ArgumentParser(description="Run conservative clean fallback policy without learned confidence.")
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--delta", default="0.01")
    parser.add_argument("--support-mins", nargs="+", type=int, default=[10, 30, 50, 100])
    parser.add_argument("--cosine-mins", nargs="+", type=float, default=[0.0, 0.05, 0.1])
    parser.add_argument("--require-visual-prior", action="store_true")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-query-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merge_clean_feature_columns(
        load_prediction_rows(args.prediction_csv),
        args.feature_csv,
        [
            "observed_has_img",
            "observed_text_img_cosine",
            "relation_support",
            "relation_is_visual_prior",
        ],
    )
    delta = float(args.delta)

    output_rows = []
    query_rows = []
    for support_min in args.support_mins:
        for cosine_min in args.cosine_mins:
            config_id = (
                f"support_min={support_min}|cosine_min={cosine_min:.2f}|"
                f"require_visual_prior={int(args.require_visual_prior)}"
            )

            def decision_fn(row: dict):
                visual_ok = int(float(row["relation_is_visual_prior"])) == 1 or not args.require_visual_prior
                use_fusion = int(
                    int(float(row["observed_has_img"])) == 1
                    and int(float(row["relation_support"])) >= int(support_min)
                    and float(row["observed_text_img_cosine"]) > float(cosine_min)
                    and visual_ok
                )
                threshold = float(cosine_min)
                return use_fusion, threshold, "conservative_rule"

            routed_rows = materialize_policy_rows(
                rows=rows,
                decision_fn=decision_fn,
                policy_name="clean_conservative_fallback_policy",
                config_id=config_id,
            )
            output_rows.append(
                summarize_policy_rows(
                    routed_rows,
                    delta=delta,
                    extra={
                        "policy_name": "clean_conservative_fallback_policy",
                        "delta": f"{delta:.2f}",
                        "support_min": int(support_min),
                        "cosine_min": float(cosine_min),
                        "require_visual_prior": int(args.require_visual_prior),
                    },
                )
            )
            query_rows.extend(routed_rows)

    output_rows.sort(key=lambda row: -float(row["overall_mrr"]))
    write_csv(args.out_csv, output_rows, OUTPUT_HEADER)
    if args.out_query_csv:
        write_query_rows(args.out_query_csv, query_rows)
    print(f"[OK] wrote conservative policy -> {Path(args.out_csv).as_posix()}")
    if args.out_query_csv:
        print(f"[OK] wrote query-level rows   -> {Path(args.out_query_csv).as_posix()}")


if __name__ == "__main__":
    main()
