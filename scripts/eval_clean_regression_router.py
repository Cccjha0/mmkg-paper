import argparse
import pickle
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.experiment_modeling import join_eval_fields, read_table
from router.experiment_utils import materialize_policy_rows, summarize_policy_rows, write_query_rows
from router.io_utils import write_csv


OUTPUT_HEADER = [
    "regressor_type",
    "feature_set",
    "theta",
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
    parser = argparse.ArgumentParser(description="Evaluate clean regression router over theta scan.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-table", required=True)
    parser.add_argument("--eval-targets", required=True)
    parser.add_argument("--thetas", nargs="+", type=float, default=[-0.02, -0.01, 0.0, 0.01, 0.02])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-query-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    with (model_dir / "model.pkl").open("rb") as handle:
        artifact = pickle.load(handle)

    test_rows = join_eval_fields(read_table(args.test_table), read_table(args.eval_targets))
    scores = artifact.predict_scores(test_rows)
    scored_rows = []
    for row, score in zip(test_rows, scores):
        merged = dict(row)
        merged["router_prob"] = float(score)
        scored_rows.append(merged)

    output_rows = []
    query_rows = []
    for theta in args.thetas:
        routed_rows = materialize_policy_rows(
            rows=scored_rows,
            decision_fn=lambda row, theta=theta: (int(float(row["router_prob"]) > float(theta)), theta, artifact.model_name),
            policy_name="clean_delta_rr_regression_router",
            config_id=f"theta={theta:.2f}",
        )
        output_rows.append(
            summarize_policy_rows(
                routed_rows,
                delta=0.01,
                extra={
                    "regressor_type": artifact.model_name,
                    "feature_set": artifact.feature_set,
                    "theta": float(theta),
                },
            )
        )
        query_rows.extend(routed_rows)

    output_rows.sort(key=lambda row: -float(row["overall_mrr"]))
    write_csv(args.out_csv, output_rows, OUTPUT_HEADER)
    if args.out_query_csv:
        write_query_rows(args.out_query_csv, query_rows)
    print(f"[OK] wrote regression scan -> {Path(args.out_csv).as_posix()}")
    if args.out_query_csv:
        print(f"[OK] wrote query rows     -> {Path(args.out_query_csv).as_posix()}")


if __name__ == "__main__":
    main()
