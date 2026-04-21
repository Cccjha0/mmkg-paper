import argparse
import csv
import json
import pickle
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.constants import ROUTER_MODE_CLEAN

PREDICTION_COLUMNS = [
    "query_id",
    "split",
    "seed",
    "direction",
    "relation_id",
    "router_mode",
    "feature_set",
    "is_query_time_legal",
    "prob_fusion",
    "target_regime",
    "rr_gate",
    "rr_residual",
]

SCAN_COLUMNS = [
    "router_mode",
    "feature_set",
    "is_query_time_legal",
    "model",
    "delta",
    "tau",
    "overall_mrr",
    "fusion_coverage",
    "gain_precision",
    "gain_recall",
    "n_selected_fusion",
    "n_total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained router on test features with threshold scan.")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.pkl and train_summary.json")
    parser.add_argument("--test-table", required=True, help="router_features_test parquet/csv")
    parser.add_argument("--eval-targets", required=True, help="router_eval_targets_shared_test parquet/csv")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--out-dir", default="outputs/router/eval")
    return parser.parse_args()


def read_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported file type: {path.as_posix()}")


def load_router(model_dir: Path):
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pkl under {model_dir}")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def load_train_summary(model_dir: Path) -> dict:
    summary_path = model_dir / "train_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing train_summary.json under {model_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def infer_model_stub(summary: dict) -> tuple[str, str, str, str, bool]:
    model_type = str(summary["model_type"])
    delta_value = float(summary["delta"])
    delta_tag = f"{delta_value:.2f}"
    router_mode = str(summary["router_mode"])
    feature_set = str(summary["feature_set"])
    legal = bool(summary.get("is_query_time_legal", router_mode == ROUTER_MODE_CLEAN))
    return model_type, delta_tag, router_mode, feature_set, legal


def join_eval_fields(test_rows: list[dict], eval_rows: list[dict]) -> list[dict]:
    eval_by_id = {str(row["query_id"]): row for row in eval_rows}
    joined = []
    for row in test_rows:
        query_id = str(row["query_id"])
        if query_id not in eval_by_id:
            raise RuntimeError(f"Missing eval target for query_id={query_id}")
        target = eval_by_id[query_id]
        merged = dict(row)
        merged["rr_gate"] = float(target["rr_gate"])
        merged["rr_residual"] = float(target["rr_residual"])
        merged["gain_label_d0"] = int(target["gain_label_d0"])
        merged["gain_label_d001"] = int(target["gain_label_d001"])
        merged["gain_label_d002"] = int(target["gain_label_d002"])
        merged["target_regime"] = str(target.get("target_regime", ""))
        joined.append(merged)
    return joined


def select_label_field(delta_tag: str) -> str:
    mapping = {"0.00": "gain_label_d0", "0.01": "gain_label_d001", "0.02": "gain_label_d002"}
    if delta_tag not in mapping:
        raise ValueError(f"Unsupported delta tag: {delta_tag}")
    return mapping[delta_tag]


def routed_rr(row: dict, tau: float) -> tuple[int, float]:
    use_fusion = int(float(row["prob_fusion"]) >= float(tau))
    rr = float(row["rr_gate"]) if use_fusion else float(row["rr_residual"])
    return use_fusion, rr


def safe_ratio(numer: int | float, denom: int | float) -> float:
    return float(numer / denom) if denom else 0.0


def resolve_prediction_path(out_dir: Path, router_mode: str, model_type: str, delta_tag: str, feature_set: str) -> Path:
    return out_dir / router_mode / f"router_predictions_{router_mode}_{model_type}_delta_{delta_tag}_{feature_set}.csv"


def resolve_scan_path(out_dir: Path, router_mode: str, model_type: str, delta_tag: str, feature_set: str) -> Path:
    return out_dir / router_mode / f"threshold_scan_{router_mode}_{model_type}_delta_{delta_tag}_{feature_set}.csv"


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)

    router = load_router(model_dir)
    summary = load_train_summary(model_dir)
    model_type, delta_tag, router_mode, feature_set, legal = infer_model_stub(summary)
    label_field = select_label_field(delta_tag)

    test_rows = read_table(Path(args.test_table))
    eval_rows = read_table(Path(args.eval_targets))
    joined_rows = join_eval_fields(test_rows, eval_rows)
    probs = router.predict_proba_from_rows(joined_rows)

    prediction_rows = []
    for row, prob in zip(joined_rows, probs):
        prediction_rows.append(
            {
                "query_id": row["query_id"],
                "split": row["split"],
                "seed": int(row["seed"]),
                "direction": row["direction"],
                "relation_id": int(row["relation_id"]),
                "router_mode": router_mode,
                "feature_set": feature_set,
                "is_query_time_legal": legal,
                "prob_fusion": float(prob),
                "target_regime": row["target_regime"],
                "rr_gate": float(row["rr_gate"]),
                "rr_residual": float(row["rr_residual"]),
            }
        )
        row["prob_fusion"] = float(prob)

    prediction_path = resolve_prediction_path(out_dir, router_mode, model_type, delta_tag, feature_set)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(prediction_rows)

    n_total = len(joined_rows)
    n_positive = sum(int(row[label_field]) for row in joined_rows)
    scan_rows = []
    for tau in args.taus:
        routed = [routed_rr(row, tau) for row in joined_rows]
        n_selected = sum(use_fusion for use_fusion, _ in routed)
        mrr = safe_ratio(sum(rr for _, rr in routed), n_total)
        selected_positive = sum(
            1 for row, (use_fusion, _) in zip(joined_rows, routed) if use_fusion and int(row[label_field]) == 1
        )
        scan_rows.append(
            {
                "router_mode": router_mode,
                "feature_set": feature_set,
                "is_query_time_legal": legal,
                "model": model_type,
                "delta": delta_tag,
                "tau": float(tau),
                "overall_mrr": mrr,
                "fusion_coverage": safe_ratio(n_selected, n_total),
                "gain_precision": safe_ratio(selected_positive, n_selected),
                "gain_recall": safe_ratio(selected_positive, n_positive),
                "n_selected_fusion": int(n_selected),
                "n_total": int(n_total),
            }
        )

    scan_path = resolve_scan_path(out_dir, router_mode, model_type, delta_tag, feature_set)
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    with scan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCAN_COLUMNS)
        writer.writeheader()
        writer.writerows(scan_rows)

    print(f"[OK] wrote predictions    -> {prediction_path.as_posix()}")
    print(f"[OK] wrote threshold scan -> {scan_path.as_posix()}")


if __name__ == "__main__":
    main()
