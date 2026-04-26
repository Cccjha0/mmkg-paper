from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


COMPARISONS = {
    "residual": {
        "right_column": "rr_residual",
        "right_label": "Residual-only",
        "out_stem": "significance_ca_s3_vs_residual",
    },
    "e5": {
        "right_column": "rr_regression_clean_router",
        "right_label": "E5 regression clean router",
        "out_stem": "significance_ca_s3_vs_e5",
    },
    "clean_rule": {
        "right_column": "rr_clean_rule",
        "right_label": "Clean rule",
        "out_stem": "significance_ca_s3_vs_clean_rule",
    },
    "direction_specific": {
        "right_column": "rr_direction_specific_threshold",
        "right_label": "Direction-specific threshold",
        "out_stem": "significance_ca_s3_vs_direction_specific",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired bootstrap significance for CA-S3 full-ranking outputs.")
    parser.add_argument(
        "--candidate-query-rows",
        nargs="+",
        default=["outputs/candidate_router/eval/ca_s3_full_ranking_seed*_query_rows.csv"],
    )
    parser.add_argument(
        "--baseline-query-rows",
        default="outputs/router/eval/clean/baseline_locked_query_rows.csv",
    )
    parser.add_argument("--candidate-column", default="mixed_rr")
    parser.add_argument("--out-dir", default="outputs/candidate_router/eval")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        if matches:
            files.extend(matches)
        else:
            path = Path(pattern)
            if path.exists():
                files.append(path)
    files = sorted(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(f"No files matched: {patterns}")
    return files


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    rank = q * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def bootstrap_ci(values: list[float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(n_bootstrap):
        total = 0.0
        for _j in range(n):
            total += values[rng.randrange(n)]
        samples.append(total / n)
    return percentile(samples, 0.025), percentile(samples, 0.975)


def load_candidate_rows(paths: list[Path], candidate_column: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["seed", "query_id", candidate_column])
        frame = frame.rename(columns={candidate_column: "candidate_rr"})
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["seed"] = out["seed"].astype(int)
    out["query_id"] = out["query_id"].astype(str)
    if out.duplicated(["seed", "query_id"]).any():
        dupes = int(out.duplicated(["seed", "query_id"]).sum())
        raise RuntimeError(f"Candidate rows contain duplicate (seed, query_id) pairs: {dupes}")
    return out


def load_baseline_rows(path: Path) -> pd.DataFrame:
    columns = ["seed", "query_id", *sorted({spec["right_column"] for spec in COMPARISONS.values()})]
    frame = pd.read_csv(path, usecols=columns)
    frame["seed"] = frame["seed"].astype(int)
    frame["query_id"] = frame["query_id"].astype(str)
    if frame.duplicated(["seed", "query_id"]).any():
        dupes = int(frame.duplicated(["seed", "query_id"]).sum())
        raise RuntimeError(f"Baseline rows contain duplicate (seed, query_id) pairs: {dupes}")
    return frame


def write_outputs(out_dir: Path, stem: str, payload: dict, summary_row: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([summary_row]).to_csv(csv_path, index=False)
    print(f"[OK] wrote {json_path.as_posix()}")
    print(f"[OK] wrote {csv_path.as_posix()}")


def main() -> None:
    args = parse_args()
    candidate_paths = expand_inputs(args.candidate_query_rows)
    candidate = load_candidate_rows(candidate_paths, args.candidate_column)
    baseline = load_baseline_rows(Path(args.baseline_query_rows))
    out_dir = Path(args.out_dir)

    for key, spec in COMPARISONS.items():
        merged = candidate.merge(
            baseline[["seed", "query_id", spec["right_column"]]],
            on=["seed", "query_id"],
            how="inner",
        )
        if len(merged) != len(candidate):
            raise RuntimeError(
                f"Pairing mismatch for {key}: candidate_rows={len(candidate)}, paired_rows={len(merged)}"
            )
        merged = merged.rename(columns={spec["right_column"]: "baseline_rr"})
        merged["delta_rr"] = merged["candidate_rr"].astype(float) - merged["baseline_rr"].astype(float)
        per_query_deltas = merged["delta_rr"].tolist()
        seed_deltas = merged.groupby("seed")["delta_rr"].mean().sort_index()
        low, high = bootstrap_ci(per_query_deltas, args.bootstrap_samples, args.bootstrap_seed)

        payload = {
            "comparison": f"CA-S3 full-ranking vs {spec['right_label']}",
            "left_label": "CA-S3 candidate-aware full-ranking",
            "right_label": spec["right_label"],
            "candidate_column": args.candidate_column,
            "baseline_column": spec["right_column"],
            "candidate_query_rows": [path.as_posix() for path in candidate_paths],
            "baseline_query_rows": str(args.baseline_query_rows),
            "n_seeds": int(seed_deltas.shape[0]),
            "n_paired_queries": int(len(merged)),
            "candidate_mrr": float(merged["candidate_rr"].mean()),
            "baseline_mrr": float(merged["baseline_rr"].mean()),
            "mean_delta_mrr_querywise": float(merged["delta_rr"].mean()),
            "mean_delta_mrr_seedwise": float(seed_deltas.mean()),
            "std_delta_mrr_seedwise": float(seed_deltas.std(ddof=1)) if len(seed_deltas) > 1 else 0.0,
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_ci_95_querywise": {
                "low": low,
                "high": high,
            },
            "seed_level_deltas": [float(v) for v in seed_deltas.tolist()],
        }
        summary_row = {
            "comparison": payload["comparison"],
            "left_label": payload["left_label"],
            "right_label": payload["right_label"],
            "n_seeds": payload["n_seeds"],
            "n_paired_queries": payload["n_paired_queries"],
            "candidate_mrr": payload["candidate_mrr"],
            "baseline_mrr": payload["baseline_mrr"],
            "mean_delta_mrr_seedwise": payload["mean_delta_mrr_seedwise"],
            "std_delta_mrr_seedwise": payload["std_delta_mrr_seedwise"],
            "mean_delta_mrr_querywise": payload["mean_delta_mrr_querywise"],
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
        }
        write_outputs(out_dir, spec["out_stem"], payload, summary_row)


if __name__ == "__main__":
    main()
