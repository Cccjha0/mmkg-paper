import argparse
import pickle
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import write_csv


RAW_HEADER = [
    "feature_name",
    "feature_family",
    "importance_gain",
    "importance_weight",
    "importance_cover",
    "rank_gain",
]

GROUPED_HEADER = [
    "feature_family",
    "feature_count",
    "aggregated_gain",
    "aggregated_weight",
    "aggregated_cover",
    "rank_gain",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw and grouped XGBoost feature importance.")
    parser.add_argument("--model-dir", default="outputs/router/models")
    parser.add_argument("--model-name", default="xgb_delta_0.01.pkl")
    parser.add_argument("--out-dir", default="outputs/router/analysis")
    parser.add_argument("--topk-summary", type=int, default=0)
    return parser.parse_args()


def resolve_model_path(model_dir: Path, model_name: str) -> Path:
    direct_path = model_dir / model_name
    if direct_path.exists():
        return direct_path

    nested_path = model_dir / model_name / "model.pkl"
    if nested_path.exists():
        return nested_path

    alt_nested = model_dir / "xgb_delta_0.01_F4" / "model.pkl"
    if alt_nested.exists() and model_name == "xgb_delta_0.01.pkl":
        return alt_nested

    raise FileNotFoundError(f"Unable to locate XGBoost model from {direct_path.as_posix()}")


def load_artifact(model_path: Path):
    try:
        with model_path.open("rb") as f:
            artifact = pickle.load(f)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Loading the XGBoost router failed because `xgboost` is not installed in the current Python environment."
        ) from exc
    if getattr(artifact, "model_name", None) != "xgb":
        raise SystemExit(f"Expected an XGBoost artifact, got {getattr(artifact, 'model_name', 'unknown')}.")
    if artifact.vectorizer is None:
        raise SystemExit("The XGBoost artifact is missing its vectorizer.")
    return artifact


def infer_feature_family(feature_name: str) -> str:
    if feature_name.startswith("target_regime="):
        return "target_regime"
    if feature_name.startswith("direction="):
        return "direction"
    if feature_name.startswith("relation_id="):
        return "relation_id"
    if feature_name.startswith("relation_gain_prior"):
        return "relation_gain_prior"
    if feature_name.startswith("relation_fusion_win_rate"):
        return "relation_fusion_win_rate"
    if feature_name.startswith("relation_support"):
        return "relation_support"
    if feature_name.startswith("relation_is_visual_prior"):
        return "relation_is_visual_prior"
    if feature_name.startswith("text_img_cosine"):
        return "text_img_cosine"
    if feature_name.startswith("img_is_missing_replaced"):
        return "img_is_missing_replaced"
    if feature_name.startswith("fusion_margin"):
        return "fusion_margin"
    if feature_name.startswith("struct_margin"):
        return "struct_margin"
    if feature_name.startswith("delta_margin"):
        return "delta_margin"
    if feature_name.startswith("fusion_correct_score"):
        return "fusion_correct_score"
    if feature_name.startswith("struct_correct_score"):
        return "struct_correct_score"
    if feature_name.startswith("target_has_img"):
        return "target_has_img"
    return "other"


def booster_score_map(artifact, importance_type: str) -> dict[str, float]:
    booster = artifact.estimator.get_booster()
    raw_scores = booster.get_score(importance_type=importance_type)
    feature_names = artifact.vectorizer.get_feature_names_out()
    out = {name: 0.0 for name in feature_names}
    for raw_name, score in raw_scores.items():
        if raw_name.startswith("f"):
            idx = int(raw_name[1:])
            if 0 <= idx < len(feature_names):
                out[feature_names[idx]] = float(score)
    return out


def build_raw_rows(artifact) -> list[dict]:
    gain_map = booster_score_map(artifact, "gain")
    weight_map = booster_score_map(artifact, "weight")
    cover_map = booster_score_map(artifact, "cover")

    rows = []
    for feature_name in artifact.vectorizer.get_feature_names_out():
        rows.append(
            {
                "feature_name": feature_name,
                "feature_family": infer_feature_family(feature_name),
                "importance_gain": gain_map.get(feature_name, 0.0),
                "importance_weight": weight_map.get(feature_name, 0.0),
                "importance_cover": cover_map.get(feature_name, 0.0),
            }
        )
    rows.sort(key=lambda row: row["importance_gain"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank_gain"] = idx
    return rows


def build_grouped_rows(raw_rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in raw_rows:
        family = row["feature_family"]
        bucket = grouped.setdefault(
            family,
            {
                "feature_family": family,
                "feature_count": 0,
                "aggregated_gain": 0.0,
                "aggregated_weight": 0.0,
                "aggregated_cover": 0.0,
            },
        )
        bucket["feature_count"] += 1
        bucket["aggregated_gain"] += float(row["importance_gain"])
        bucket["aggregated_weight"] += float(row["importance_weight"])
        bucket["aggregated_cover"] += float(row["importance_cover"])

    rows = list(grouped.values())
    rows.sort(key=lambda row: row["aggregated_gain"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank_gain"] = idx
    return rows


def maybe_write_summary(grouped_rows: list[dict], out_dir: Path, topk: int) -> None:
    if topk <= 0:
        return
    picked = grouped_rows[:topk]
    lines = ["# XGBoost Grouped Feature Importance", ""]
    for row in picked:
        lines.append(
            f"- `{row['feature_family']}`: gain={float(row['aggregated_gain']):.6f}, "
            f"weight={float(row['aggregated_weight']):.6f}, cover={float(row['aggregated_cover']):.6f}"
        )
    summary_path = out_dir / "xgb_feature_importance_grouped_topk.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(Path(args.model_dir), args.model_name)
    artifact = load_artifact(model_path)

    raw_rows = build_raw_rows(artifact)
    grouped_rows = build_grouped_rows(raw_rows)

    out_dir = Path(args.out_dir)
    raw_path = out_dir / "xgb_feature_importance_raw.csv"
    grouped_path = out_dir / "xgb_feature_importance_grouped.csv"
    write_csv(raw_path, raw_rows, RAW_HEADER)
    write_csv(grouped_path, grouped_rows, GROUPED_HEADER)
    maybe_write_summary(grouped_rows, out_dir, args.topk_summary)

    print(f"[OK] wrote raw importance     -> {raw_path.as_posix()}")
    print(f"[OK] wrote grouped importance -> {grouped_path.as_posix()}")


if __name__ == "__main__":
    main()
