import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose oracle gap by regime and error type.")
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.as_posix()}")


def find_single_col(df: pd.DataFrame, prefix: str) -> str:
    matches = [col for col in df.columns if col.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one column with prefix {prefix}, found: {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    df = load_table(Path(args.master_table))
    routed_col = find_single_col(df, "rr_routed_")
    select_col = find_single_col(df, "select_fusion_")

    rows = []
    for regime, bucket in df.groupby("target_regime", sort=True):
        oracle_gate = bucket["oracle_choose"] == "gate"
        oracle_residual = bucket["oracle_choose"] == "residual"
        selected = bucket[select_col].astype(int) == 1

        false_negative_mask = oracle_gate & (~selected)
        false_positive_mask = oracle_residual & selected

        rows.append(
            {
                "regime": regime,
                "n_queries": int(len(bucket)),
                "mrr_oracle": float(bucket["rr_oracle_gate_residual"].astype(float).mean()),
                "mrr_xgb_routed": float(bucket[routed_col].astype(float).mean()),
                "oracle_gap": float(
                    bucket["rr_oracle_gate_residual"].astype(float).mean() - bucket[routed_col].astype(float).mean()
                ),
                "false_negative_rate": float(false_negative_mask.sum() / oracle_gate.sum()) if oracle_gate.sum() else 0.0,
                "false_positive_rate": float(false_positive_mask.sum() / oracle_residual.sum())
                if oracle_residual.sum()
                else 0.0,
                "lost_rr_mass_fn": float(
                    (bucket.loc[false_negative_mask, "rr_gate"].astype(float) - bucket.loc[false_negative_mask, "rr_residual"].astype(float)).sum()
                ),
                "lost_rr_mass_fp": float(
                    (bucket.loc[false_positive_mask, "rr_residual"].astype(float) - bucket.loc[false_positive_mask, "rr_gate"].astype(float)).sum()
                ),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] wrote analysis -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
