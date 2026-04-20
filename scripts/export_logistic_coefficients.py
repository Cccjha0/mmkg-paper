import argparse
import pickle
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv
from router.router_models import rows_to_feature_dicts


OUTPUT_HEADER = [
    "feature_name",
    "coefficient",
    "coefficient_abs",
    "direction",
    "rank_abs",
    "rank_within_direction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top positive and negative logistic coefficients.")
    parser.add_argument("--model-dir", default="outputs/router/models")
    parser.add_argument("--model-name", default="logistic_delta_0.01.pkl")
    parser.add_argument("--out", default="outputs/router/analysis/logistic_top_coefficients.csv")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--standardize-for-interpretation", action="store_true")
    parser.add_argument("--train-csv", default=None)
    return parser.parse_args()


def resolve_model_path(model_dir: Path, model_name: str) -> Path:
    direct_path = model_dir / model_name
    if direct_path.exists():
        return direct_path

    nested_path = model_dir / model_name / "model.pkl"
    if nested_path.exists():
        return nested_path

    alt_nested = model_dir / "logistic_delta_0.01_F4" / "model.pkl"
    if alt_nested.exists() and model_name == "logistic_delta_0.01.pkl":
        return alt_nested

    raise FileNotFoundError(f"Unable to locate logistic model from {direct_path.as_posix()}")


def load_artifact(model_path: Path):
    with model_path.open("rb") as f:
        artifact = pickle.load(f)
    if getattr(artifact, "model_name", None) != "logistic":
        raise SystemExit(f"Expected a logistic artifact, got {getattr(artifact, 'model_name', 'unknown')}.")
    if artifact.vectorizer is None:
        raise SystemExit("The logistic artifact is missing its vectorizer.")
    return artifact


def standardized_coefficients(artifact, train_csv: Path) -> np.ndarray:
    rows = read_csv(train_csv)
    payload = rows_to_feature_dicts(rows, artifact.feature_names)
    matrix = artifact.vectorizer.transform(payload).toarray()
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return artifact.estimator.coef_[0] * std


def build_rows(feature_names: list[str], coefficients: np.ndarray, topk: int) -> list[dict]:
    all_rows = []
    for name, coef in zip(feature_names, coefficients):
        direction = "positive_for_fusion" if float(coef) >= 0 else "negative_for_fusion"
        all_rows.append(
            {
                "feature_name": name,
                "coefficient": float(coef),
                "coefficient_abs": abs(float(coef)),
                "direction": direction,
            }
        )

    all_rows.sort(key=lambda row: row["coefficient_abs"], reverse=True)
    for idx, row in enumerate(all_rows, start=1):
        row["rank_abs"] = idx

    positive_rows = [row for row in all_rows if row["coefficient"] > 0][:topk]
    negative_rows = [row for row in all_rows if row["coefficient"] < 0][:topk]

    for idx, row in enumerate(positive_rows, start=1):
        row["rank_within_direction"] = idx
    for idx, row in enumerate(negative_rows, start=1):
        row["rank_within_direction"] = idx

    picked = positive_rows + negative_rows
    picked.sort(key=lambda row: (row["direction"], row["rank_within_direction"]))
    return picked


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(Path(args.model_dir), args.model_name)
    artifact = load_artifact(model_path)
    feature_names = list(artifact.vectorizer.get_feature_names_out())

    if args.standardize_for_interpretation:
        if not args.train_csv:
            raise SystemExit("--train-csv is required when --standardize-for-interpretation is enabled.")
        coefficients = standardized_coefficients(artifact, Path(args.train_csv))
    else:
        coefficients = artifact.estimator.coef_[0]

    rows = build_rows(feature_names, np.asarray(coefficients), args.topk)
    out_path = Path(args.out)
    write_csv(out_path, rows, OUTPUT_HEADER)
    print(f"[OK] wrote logistic coefficients -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
