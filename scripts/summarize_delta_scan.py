import argparse
import json
import re
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.io_utils import read_csv, write_csv


OUTPUT_HEADER = [
    "router_mode",
    "feature_set",
    "model_type",
    "delta",
    "selected_tau",
    "best_mrr",
    "fusion_coverage",
    "gain_precision",
    "positive_label_rate_dev",
    "oracle_gap_at_best_tau",
    "source_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize delta scan results from threshold scan CSV files.")
    parser.add_argument("--scan-files", nargs="*", default=None)
    parser.add_argument("--label-stats", nargs="*", default=None)
    parser.add_argument("--out", default="outputs/router/analysis/delta_scan_summary.csv")
    parser.add_argument("--oracle-mrr", type=float, default=None)
    parser.add_argument("--summary-text-out", default=None)
    return parser.parse_args()


def default_scan_files() -> list[Path]:
    base = Path("outputs/router/eval")
    return sorted(base.glob("*\\threshold_scan_*_delta_*_*.csv"))


def default_label_stats() -> list[Path]:
    base = Path("outputs/router/eval")
    return sorted(base.glob("router_train_metrics_delta_*.json"))


def parse_scan_stub(path: Path) -> tuple[str, str, str, str]:
    match = re.search(
        r"threshold_scan_(?P<router_mode>clean|posthoc)_(?P<model>[a-z0-9]+)_delta_(?P<delta>[0-9.]+)_(?P<feature_set>[A-Za-z0-9_]+)\.csv$",
        path.name,
    )
    if not match:
        raise ValueError(f"Could not parse scan stub from {path.name}")
    return match.group("router_mode"), match.group("model"), match.group("delta"), match.group("feature_set")


def load_positive_rates(paths: list[Path]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        delta = str(payload.get("delta"))
        models = payload.get("models", {})
        if not models:
            continue
        first_model = next(iter(models.values()))
        positive_rate = first_model.get("positive_rate")
        if positive_rate is not None:
            mapping[delta] = float(positive_rate)
    return mapping


def best_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: float(row["overall_mrr"]))


def build_summary_rows(scan_files: list[Path], positive_rates: dict[str, float], oracle_mrr: float | None) -> list[dict]:
    out = []
    for path in scan_files:
        router_mode, model_type, delta, feature_set = parse_scan_stub(path)
        rows = read_csv(path)
        if not rows:
            continue
        picked = best_row(rows)
        best_mrr = float(picked["overall_mrr"])
        oracle_gap = ""
        if oracle_mrr is not None:
            oracle_gap = oracle_mrr - best_mrr
        elif "oracle_mrr" in picked:
            oracle_gap = float(picked["oracle_mrr"]) - best_mrr

        out.append(
            {
                "router_mode": router_mode,
                "feature_set": feature_set,
                "model_type": model_type,
                "delta": delta,
                "selected_tau": float(picked["tau"]),
                "best_mrr": best_mrr,
                "fusion_coverage": float(picked["fusion_coverage"]),
                "gain_precision": float(picked["gain_precision"]),
                "positive_label_rate_dev": positive_rates.get(delta, ""),
                "oracle_gap_at_best_tau": oracle_gap,
                "source_file": path.as_posix(),
            }
        )
    out.sort(key=lambda row: (row["router_mode"], row["model_type"], float(row["delta"])))
    return out


def build_summary_text(rows: list[dict]) -> str:
    lines = ["# Delta Scan Summary", ""]
    for row in rows:
        label_rate = row["positive_label_rate_dev"]
        label_rate_text = f"{float(label_rate):.4f}" if label_rate != "" else "N/A"
        oracle_gap = row["oracle_gap_at_best_tau"]
        oracle_gap_text = f"{float(oracle_gap):.4f}" if oracle_gap != "" else "N/A"
        lines.append(
            f"- `{row['router_mode']} / {row['model_type']} / {row['feature_set']}` @ `delta={row['delta']}`: best_tau={float(row['selected_tau']):.1f}, "
            f"best_mrr={float(row['best_mrr']):.4f}, coverage={float(row['fusion_coverage']):.4f}, "
            f"gain_precision={float(row['gain_precision']):.4f}, positive_label_rate_dev={label_rate_text}, "
            f"oracle_gap={oracle_gap_text}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    scan_files = [Path(p) for p in args.scan_files] if args.scan_files else default_scan_files()
    label_stats = [Path(p) for p in args.label_stats] if args.label_stats else default_label_stats()

    if not scan_files:
        raise SystemExit("No threshold scan CSV files were found.")

    positive_rates = load_positive_rates(label_stats)
    summary_rows = build_summary_rows(scan_files, positive_rates, args.oracle_mrr)
    out_path = Path(args.out)
    write_csv(out_path, summary_rows, OUTPUT_HEADER)
    print(f"[OK] wrote delta scan summary -> {out_path.as_posix()}")

    summary_text_out = Path(args.summary_text_out) if args.summary_text_out else out_path.with_suffix(".md")
    summary_text_out.write_text(build_summary_text(summary_rows), encoding="utf-8")
    print(f"[OK] wrote summary markdown  -> {summary_text_out.as_posix()}")


if __name__ == "__main__":
    main()
