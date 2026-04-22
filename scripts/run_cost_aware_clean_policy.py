import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv


OUTPUT_HEADER = [
    "policy_name",
    "model",
    "delta",
    "lambda",
    "utility",
    "overall_mrr",
    "fusion_coverage",
    "tau",
    "config_id",
    "source_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cost-aware utility over clean policy scan results.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.05])
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_rows = []
    for path_str in args.inputs:
        rows = read_csv(path_str)
        source_file = Path(path_str).as_posix()
        for row in rows:
            overall_mrr = float(row["overall_mrr"])
            fusion_coverage = float(row["fusion_coverage"])
            policy_name = str(row.get("policy_name", Path(path_str).stem))
            tau_value = row.get("tau", row.get("uncertain_tau", ""))
            config_id = str(
                row.get(
                    "config_id",
                    "|".join(
                        f"{key}={row[key]}"
                        for key in row.keys()
                        if key
                        in {
                            "tau",
                            "tau_head",
                            "tau_tail",
                            "tau_neg",
                            "tau_mid",
                            "tau_pos",
                            "uncertain_tau",
                            "prior_low_cutoff",
                            "prior_high_cutoff",
                            "high_policy",
                        }
                    ),
                )
            )
            for lambda_value in args.lambdas:
                output_rows.append(
                    {
                        "policy_name": policy_name,
                        "model": str(row.get("model", "")),
                        "delta": str(row.get("delta", "")),
                        "lambda": float(lambda_value),
                        "utility": overall_mrr - float(lambda_value) * fusion_coverage,
                        "overall_mrr": overall_mrr,
                        "fusion_coverage": fusion_coverage,
                        "tau": tau_value,
                        "config_id": config_id,
                        "source_file": source_file,
                    }
                )
    output_rows.sort(key=lambda row: (row["lambda"], row["policy_name"], -float(row["utility"])))
    write_csv(args.out_csv, output_rows, OUTPUT_HEADER)
    print(f"[OK] wrote cost-aware utility -> {Path(args.out_csv).as_posix()}")


if __name__ == "__main__":
    main()
