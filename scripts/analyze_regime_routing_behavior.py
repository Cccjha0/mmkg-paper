import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze learned router behavior by target regime.")
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
        selected = bucket[select_col].astype(int)
        positives = bucket["gain_label_d001"].astype(int)
        selected_positive = ((selected == 1) & (positives == 1)).sum()
        n_selected = int((selected == 1).sum())
        n_positive = int((positives == 1).sum())

        rows.append(
            {
                "regime": regime,
                "n_queries": int(len(bucket)),
                "fusion_selection_rate": float(selected.mean()) if len(bucket) else 0.0,
                "gain_precision_d001": float(selected_positive / n_selected) if n_selected else 0.0,
                "gain_recall_d001": float(selected_positive / n_positive) if n_positive else 0.0,
                "mean_prob_fusion": float(bucket["prob_fusion"].astype(float).mean()),
                "mean_rr_routed": float(bucket[routed_col].astype(float).mean()),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] wrote analysis -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
