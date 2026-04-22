import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.experiment_modeling import merge_delta_rr_targets, read_table, train_regression_artifact, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regression router to predict delta_rr on clean features.")
    parser.add_argument("--train-table", required=True)
    parser.add_argument("--gain-label-csvs", nargs="+", required=True)
    parser.add_argument("--model-type", required=True, choices=["linear", "xgb", "lgbm"])
    parser.add_argument("--feature-set", default="C4")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merge_delta_rr_targets(read_table(args.train_table), args.gain_label_csvs)
    artifact = train_regression_artifact(
        rows=rows,
        feature_set=args.feature_set,
        model_type=args.model_type,
        random_state=int(args.random_state),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact.save(out_dir / "model.pkl")
    summary = {
        "task": "regression",
        "router_mode": "clean",
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "train_table": Path(args.train_table).as_posix(),
        "gain_label_csvs": [Path(path).as_posix() for path in args.gain_label_csvs],
        "n_train": len(rows),
        "target_field": "delta_rr",
        "random_state": int(args.random_state),
        "delta_rr_mean": float(sum(float(row["delta_rr"]) for row in rows) / len(rows)) if rows else 0.0,
        "feature_names": artifact.feature_names,
        "metadata": artifact.metadata,
    }
    write_summary(out_dir / "train_summary.json", summary)
    (out_dir / "feature_columns.json").write_text(json.dumps(artifact.feature_names, indent=2), encoding="utf-8")

    print(f"[OK] wrote regression model   -> {model_path.as_posix()}")
    print(f"[OK] wrote regression summary -> {(out_dir / 'train_summary.json').as_posix()}")


if __name__ == "__main__":
    main()
