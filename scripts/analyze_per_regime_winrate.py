import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-regime win/lose/tie rates from router master table.")
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


def summarize_comparison(frame: pd.DataFrame, lhs_col: str, rhs_col: str, comparison: str) -> list[dict]:
    rows = []
    for regime, bucket in frame.groupby("target_regime", sort=True):
        lhs = bucket[lhs_col].astype(float)
        rhs = bucket[rhs_col].astype(float)
        diff = lhs - rhs
        n = len(bucket)
        rows.append(
            {
                "regime": regime,
                "comparison": comparison,
                "n_queries": int(n),
                "win_rate": float((diff > 0).mean()) if n else 0.0,
                "lose_rate": float((diff < 0).mean()) if n else 0.0,
                "tie_rate": float((diff == 0).mean()) if n else 0.0,
                "mean_rr_diff": float(diff.mean()) if n else 0.0,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    df = load_table(Path(args.master_table))
    routed_col = find_routed_col(df)

    output_rows = []
    output_rows.extend(summarize_comparison(df, "rr_gate", "rr_residual", "gate_vs_residual"))
    output_rows.extend(summarize_comparison(df, "rr_full_official", "rr_residual", "full_vs_residual"))
    output_rows.extend(summarize_comparison(df, routed_col, "rr_residual", "xgb_routed_vs_residual"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(out_path, index=False)
    print(f"[OK] wrote analysis -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
