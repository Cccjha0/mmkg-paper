import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.experiment_utils import load_score_map
from router.io_utils import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute clean/posthoc oracle gap recovery on paired routed files.")
    parser.add_argument("--clean-file", required=True)
    parser.add_argument("--clean-label", default="clean_best")
    parser.add_argument("--posthoc-file", required=True)
    parser.add_argument("--posthoc-label", default="posthoc_best")
    parser.add_argument("--residual-file", required=True)
    parser.add_argument("--residual-source", choices=["final", "residual"], default="residual")
    parser.add_argument("--oracle-file", required=True)
    parser.add_argument("--oracle-source", choices=["final", "gate"], default="final")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def average_scores(score_map: dict[tuple[int, str], float]) -> float:
    return float(sum(score_map.values()) / len(score_map)) if score_map else 0.0


def recoverable_gap(model_score: float, residual_score: float, oracle_score: float) -> float:
    denom = oracle_score - residual_score
    if abs(denom) < 1e-12:
        return 0.0
    return float((model_score - residual_score) / denom)


def main() -> None:
    args = parse_args()
    clean_scores = load_score_map(args.clean_file, "final")
    posthoc_scores = load_score_map(args.posthoc_file, "final")
    residual_scores = load_score_map(args.residual_file, args.residual_source)
    oracle_scores = load_score_map(args.oracle_file, args.oracle_source)

    shared = set(clean_scores) & set(posthoc_scores) & set(residual_scores) & set(oracle_scores)
    if not shared:
        raise RuntimeError("No shared paired query rows found across clean/posthoc/residual/oracle inputs.")

    clean_shared = {key: clean_scores[key] for key in shared}
    posthoc_shared = {key: posthoc_scores[key] for key in shared}
    residual_shared = {key: residual_scores[key] for key in shared}
    oracle_shared = {key: oracle_scores[key] for key in shared}

    clean_mean = average_scores(clean_shared)
    posthoc_mean = average_scores(posthoc_shared)
    residual_mean = average_scores(residual_shared)
    oracle_mean = average_scores(oracle_shared)

    rows = [
        {
            "comparison": "oracle_gap_recovery",
            "n_paired_queries": int(len(shared)),
            "residual_only_mrr": residual_mean,
            "clean_mrr": clean_mean,
            "posthoc_mrr": posthoc_mean,
            "oracle_mrr": oracle_mean,
            "recoverable_gap_clean": recoverable_gap(clean_mean, residual_mean, oracle_mean),
            "recoverable_gap_posthoc": recoverable_gap(posthoc_mean, residual_mean, oracle_mean),
        }
    ]
    payload = dict(rows[0])
    payload["clean_label"] = args.clean_label
    payload["posthoc_label"] = args.posthoc_label

    write_csv(args.out_csv, rows, list(rows[0].keys()))
    write_json(args.out_json, payload)
    print(f"[OK] wrote oracle gap csv  -> {Path(args.out_csv).as_posix()}")
    print(f"[OK] wrote oracle gap json -> {Path(args.out_json).as_posix()}")


if __name__ == "__main__":
    main()
