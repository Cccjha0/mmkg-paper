import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.experiment_utils import (
    bootstrap_ci,
    load_score_map,
    paired_seed_deltas,
    write_significance_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seed-wise and paired-bootstrap significance for two router policies.")
    parser.add_argument("--left-file", required=True)
    parser.add_argument("--left-source", choices=["final", "residual", "gate"], default="final")
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-file", required=True)
    parser.add_argument("--right-source", choices=["final", "residual", "gate"], default="final")
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--comparison", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_scores = load_score_map(args.left_file, args.left_source)
    right_scores = load_score_map(args.right_file, args.right_source)
    seed_means, per_query_deltas, _ = paired_seed_deltas(left_scores, right_scores)
    ci = bootstrap_ci(per_query_deltas, n_bootstrap=args.bootstrap_samples, seed=args.bootstrap_seed)
    comparison = args.comparison or f"{args.left_label}_vs_{args.right_label}"
    write_significance_payload(
        out_json=args.out_json,
        out_csv=args.out_csv,
        comparison=comparison,
        left_label=args.left_label,
        right_label=args.right_label,
        seed_means=seed_means,
        per_query_deltas=per_query_deltas,
        bootstrap_bounds=ci,
    )
    print(f"[OK] wrote significance json -> {Path(args.out_json).as_posix()}")
    print(f"[OK] wrote significance csv  -> {Path(args.out_csv).as_posix()}")


if __name__ == "__main__":
    main()
