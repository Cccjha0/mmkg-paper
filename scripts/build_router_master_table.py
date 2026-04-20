import argparse
import re
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified test-side router master table.")
    parser.add_argument("--gate-test", required=True)
    parser.add_argument("--residual-test", required=True)
    parser.add_argument("--full-test", required=True)
    parser.add_argument("--router-preds", required=True)
    parser.add_argument("--delta", required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.as_posix()}")


def delta_code(delta: str) -> str:
    return f"d{int(round(float(delta) * 100)):03d}"


def tau_code(tau: float) -> str:
    return f"t{int(round(float(tau) * 10)):02d}"


def infer_model_name(router_preds: Path) -> str:
    match = re.search(r"router_predictions_([^_]+)_delta_", router_preds.name)
    if match:
        return match.group(1)
    return "router"


def prepare_outcome(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if "margin" not in df.columns and "score_margin" in df.columns:
        df = df.copy()
        df["margin"] = df["score_margin"]
    rename = {
        "rr": f"rr_{prefix}",
        "rank": f"rank_{prefix}",
        "correct_score": f"correct_score_{prefix}",
        "top1_score": f"top1_score_{prefix}",
        "top2_score": f"top2_score_{prefix}",
        "margin": f"margin_{prefix}",
    }
    keep = [
        "query_id",
        "direction",
        "relation_id",
        "head_id",
        "tail_id",
        "target_entity_id",
        "target_has_img",
        "target_regime",
        "rank",
        "rr",
        "correct_score",
        "top1_score",
        "top2_score",
        "margin",
    ]
    return df[keep].rename(columns=rename)


def main() -> None:
    args = parse_args()
    gate_df = prepare_outcome(read_table(Path(args.gate_test)), "gate")
    residual_df = prepare_outcome(read_table(Path(args.residual_test)), "residual")
    full_df = prepare_outcome(read_table(Path(args.full_test)), "full_official")
    pred_df = read_table(Path(args.router_preds))

    merged = gate_df.merge(
        residual_df[
            [
                "query_id",
                "rank_residual",
                "rr_residual",
                "correct_score_residual",
                "top1_score_residual",
                "top2_score_residual",
                "margin_residual",
            ]
        ],
        on="query_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        full_df[
            [
                "query_id",
                "rank_full_official",
                "rr_full_official",
                "correct_score_full_official",
                "top1_score_full_official",
                "top2_score_full_official",
                "margin_full_official",
            ]
        ],
        on="query_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        pred_df[["query_id", "prob_fusion"]],
        on="query_id",
        how="inner",
        validate="one_to_one",
    )

    model_name = infer_model_name(Path(args.router_preds))
    d_code = delta_code(args.delta)
    t_code = tau_code(args.tau)
    select_col = f"select_fusion_{t_code}"
    routed_col = f"rr_routed_{model_name}_{d_code}_{t_code}"

    merged["rr_oracle_gate_residual"] = merged[["rr_gate", "rr_residual"]].max(axis=1)
    merged["oracle_choose"] = merged.apply(
        lambda row: "gate" if float(row["rr_gate"]) >= float(row["rr_residual"]) else "residual",
        axis=1,
    )
    merged["rr_gain_gate_vs_residual"] = merged["rr_gate"] - merged["rr_residual"]
    merged["gain_label_d0"] = (merged["rr_gain_gate_vs_residual"] > 0.00).astype(int)
    merged["gain_label_d001"] = (merged["rr_gain_gate_vs_residual"] > 0.01).astype(int)
    merged["gain_label_d002"] = (merged["rr_gain_gate_vs_residual"] > 0.02).astype(int)

    merged[select_col] = (merged["prob_fusion"] >= float(args.tau)).astype(int)
    merged[routed_col] = merged.apply(
        lambda row: float(row["rr_gate"]) if int(row[select_col]) == 1 else float(row["rr_residual"]),
        axis=1,
    )

    merged["router_correctly_selected_gain_d001"] = (
        (merged[select_col] == 1) & (merged["gain_label_d001"] == 1)
    ).astype(int)
    merged["router_false_positive_d001"] = (
        (merged[select_col] == 1) & (merged["gain_label_d001"] == 0)
    ).astype(int)
    merged["router_false_negative_d001"] = (
        (merged[select_col] == 0) & (merged["gain_label_d001"] == 1)
    ).astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"[OK] wrote master table -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
