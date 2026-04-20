import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate seed-level CSV analyses into mean ± std summaries.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Seed-level CSV files to aggregate")
    parser.add_argument("--group-cols", nargs="+", required=True, help="Grouping columns, e.g. regime comparison")
    parser.add_argument("--count-col", default="n_queries", help="Fixed-count column to carry through")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_frames(paths: list[str]) -> pd.DataFrame:
    frames = []
    for idx, path_str in enumerate(paths, start=1):
        path = Path(path_str)
        frame = pd.read_csv(path)
        frame["seed_run"] = idx
        frame["source_file"] = path.as_posix()
        frames.append(frame)
    if not frames:
        raise RuntimeError("No input CSV files were provided.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    df = load_frames(args.inputs)

    missing_groups = [col for col in args.group_cols if col not in df.columns]
    if missing_groups:
        raise RuntimeError(f"Missing group columns: {missing_groups}")

    count_col = args.count_col if args.count_col in df.columns else None
    reserved = set(args.group_cols) | {"seed_run", "source_file"}
    if count_col:
        reserved.add(count_col)

    metric_cols = [
        col
        for col in df.columns
        if col not in reserved and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not metric_cols:
        raise RuntimeError("No numeric metric columns found to aggregate.")

    agg_rows = []
    for group_keys, bucket in df.groupby(args.group_cols, sort=True, dropna=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = {col: value for col, value in zip(args.group_cols, group_keys)}
        row["n_seeds"] = int(bucket["seed_run"].nunique())

        if count_col:
            unique_counts = sorted(bucket[count_col].dropna().astype(int).unique().tolist())
            if len(unique_counts) > 1:
                raise RuntimeError(
                    f"Count column {count_col} is inconsistent for group {group_keys}: {unique_counts}"
                )
            row[count_col] = int(unique_counts[0]) if unique_counts else None

        for metric in metric_cols:
            series = bucket[metric].astype(float)
            row[f"{metric}_mean"] = float(series.mean())
            row[f"{metric}_std"] = float(series.std(ddof=0))
        agg_rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agg_rows).to_csv(out_path, index=False)
    print(f"[OK] wrote seed summary -> {out_path.as_posix()}")


if __name__ == "__main__":
    main()
