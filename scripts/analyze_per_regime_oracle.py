import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-regime oracle upper bound and learned routing gap.")
    parser.add_argument("--master-table", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.as_posix()}")


def find_routed_col(df: pd.DataFrame) -> str:
    matches = [col for col in df.columns if col.startswith("rr_routed_")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one routed RR column, found: {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    df = load_table(Path(args.master_table))
    routed_col = find_routed_col(df)

    rows = []
    for regime, bucket in df.groupby("target_regime", sort=True):
        mrr_residual = float(bucket["rr_residual"].astype(float).mean())
        mrr_gate = float(bucket["rr_gate"].astype(float).mean())
        mrr_xgb_routed = float(bucket[routed_col].astype(float).mean())
        mrr_oracle = float(bucket["rr_oracle_gate_residual"].astype(float).mean())
        rows.append(
            {
                "regime": regime,
                "n_queries": int(len(bucket)),
                "mrr_residual": mrr_residual,
                "mrr_gate": mrr_gate,
                "mrr_xgb_routed": mrr_xgb_routed,
                "mrr_oracle": mrr_oracle,
                "oracle_minus_xgb_gap": float(mrr_oracle - mrr_xgb_routed),
                "oracle_minus_residual_gap": float(mrr_oracle - mrr_residual),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] wrote analysis -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
