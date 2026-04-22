import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv


OUTPUT_HEADER = [
    "model",
    "delta",
    "best_tau",
    "best_overall_mrr",
    "best_fusion_coverage",
    "best_gain_precision",
    "source_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize clean delta sensitivity from threshold scan CSV files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_rows = {}
    for path_str in args.inputs:
        for row in read_csv(path_str):
            key = (str(row.get("model", "")), str(row.get("delta", "")))
            metric = float(row["overall_mrr"])
            current = best_rows.get(key)
            if current is None or metric > float(current["best_overall_mrr"]):
                best_rows[key] = {
                    "model": key[0],
                    "delta": key[1],
                    "best_tau": row.get("tau", row.get("uncertain_tau", "")),
                    "best_overall_mrr": metric,
                    "best_fusion_coverage": float(row.get("fusion_coverage", 0.0)),
                    "best_gain_precision": float(row.get("gain_precision", 0.0)),
                    "source_file": Path(path_str).as_posix(),
                }
    output_rows = sorted(best_rows.values(), key=lambda row: (row["model"], row["delta"]))
    write_csv(args.out_csv, output_rows, OUTPUT_HEADER)
    print(f"[OK] wrote delta sensitivity -> {Path(args.out_csv).as_posix()}")


if __name__ == "__main__":
    main()
