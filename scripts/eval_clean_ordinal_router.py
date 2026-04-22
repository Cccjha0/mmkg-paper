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
    "classifier_type",
    "feature_set",
    "decision_rule",
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
    parser = argparse.ArgumentParser(description="Evaluate ordinal clean router with decision rules.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-table", required=True)
    parser.add_argument("--eval-targets", required=True)
    parser.add_argument("--decision-rules", nargs="+", default=["strong_only", "weak_or_strong_positive"])
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-query-csv", default=None)
    return parser.parse_args()


def decide_use_fusion(predicted_class: int, rule: str) -> int:
    if rule == "strong_only":
        return int(predicted_class == 3)
    if rule == "weak_or_strong_positive":
        return int(predicted_class in {2, 3})
    raise ValueError(f"Unsupported decision rule: {rule}")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    with (model_dir / "model.pkl").open("rb") as handle:
        artifact = pickle.load(handle)

    test_rows = join_eval_fields(read_table(args.test_table), read_table(args.eval_targets))
    class_probs = artifact.predict_class_proba(test_rows)
    predicted_classes = [max(range(len(probs)), key=lambda idx: probs[idx]) for probs in class_probs]

    scored_rows = []
    for row, probs, predicted_class in zip(test_rows, class_probs, predicted_classes):
        merged = dict(row)
        merged["router_prob"] = float(max(probs))
        merged["predicted_class"] = int(predicted_class)
        scored_rows.append(merged)

    output_rows = []
    query_rows = []
    for rule in args.decision_rules:
        routed_rows = materialize_policy_rows(
            rows=scored_rows,
            decision_fn=lambda row, rule=rule: (
                decide_use_fusion(int(row["predicted_class"]), rule),
                int(row["predicted_class"]),
                artifact.model_name,
            ),
            policy_name="clean_ordinal_gain_router",
            config_id=rule,
        )
        output_rows.append(
            summarize_policy_rows(
                routed_rows,
                delta=0.01,
                extra={
                    "classifier_type": artifact.model_name,
                    "feature_set": artifact.feature_set,
                    "decision_rule": rule,
                },
            )
        )
        query_rows.extend(routed_rows)

    output_rows.sort(key=lambda row: -float(row["overall_mrr"]))
    write_csv(args.out_csv, output_rows, OUTPUT_HEADER)
    if args.out_query_csv:
        write_query_rows(args.out_query_csv, query_rows)
    print(f"[OK] wrote ordinal eval -> {Path(args.out_csv).as_posix()}")
    if args.out_query_csv:
        print(f"[OK] wrote query rows  -> {Path(args.out_query_csv).as_posix()}")


if __name__ == "__main__":
    main()
