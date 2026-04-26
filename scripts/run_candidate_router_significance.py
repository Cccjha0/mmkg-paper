from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


BASELINE_COMPARISONS = {
    "residual": {
        "right_column": "rr_residual",
        "right_label": "Residual-only",
        "out_suffix": "vs_residual",
    },
    "e5": {
        "right_column": "rr_regression_clean_router",
        "right_label": "E5 regression clean router",
        "out_suffix": "vs_e5",
    },
    "clean_rule": {
        "right_column": "rr_clean_rule",
        "right_label": "Clean rule",
        "out_suffix": "vs_clean_rule",
    },
    "direction_specific": {
        "right_column": "rr_direction_specific_threshold",
        "right_label": "Direction-specific threshold",
        "out_suffix": "vs_direction_specific",
    },
}

METHODS = {
    "ca_s1": {
        "label": "CA-S1 clean candidate full-ranking",
        "query_rows": ["outputs/candidate_router/eval/ca_s1_full_ranking_seed*_query_rows.csv"],
    },
    "ca_s2": {
        "label": "CA-S2 score-aware full-ranking",
        "query_rows": ["outputs/candidate_router/eval/ca_s2_full_ranking_seed*_query_rows.csv"],
    },
    "ca_s3": {
        "label": "CA-S3 clean + score-aware full-ranking",
        "query_rows": ["outputs/candidate_router/eval/ca_s3_full_ranking_seed*_query_rows.csv"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired bootstrap significance for candidate-router full-ranking outputs.")
    parser.add_argument(
        "--candidate-query-rows",
        nargs="+",
        default=None,
        help="Optional explicit query-row files. If omitted, all methods in --methods are processed.",
    )
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--candidate-key", default=None)
    parser.add_argument("--methods", nargs="+", default=["ca_s1", "ca_s2", "ca_s3"], choices=sorted(METHODS))
    parser.add_argument("--include-method-pairs", action="store_true", default=True)
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
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    samples = np.empty(n_bootstrap, dtype=np.float64)
    chunk = 200
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n), endpoint=False)
        samples[start:stop] = arr[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


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
    columns = ["seed", "query_id", *sorted({spec["right_column"] for spec in BASELINE_COMPARISONS.values()})]
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


def comparison_payload(
    *,
    merged: pd.DataFrame,
    left_label: str,
    right_label: str,
    comparison: str,
    left_column: str,
    right_column: str,
    candidate_paths: list[Path],
    baseline_source: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict, dict]:
    merged["delta_rr"] = merged["candidate_rr"].astype(float) - merged["baseline_rr"].astype(float)
    per_query_deltas = merged["delta_rr"].tolist()
    seed_deltas = merged.groupby("seed")["delta_rr"].mean().sort_index()
    low, high = bootstrap_ci(per_query_deltas, bootstrap_samples, bootstrap_seed)
    payload = {
        "comparison": comparison,
        "left_label": left_label,
        "right_label": right_label,
        "candidate_column": left_column,
        "baseline_column": right_column,
        "candidate_query_rows": [path.as_posix() for path in candidate_paths],
        "baseline_query_rows": baseline_source,
        "n_seeds": int(seed_deltas.shape[0]),
        "n_paired_queries": int(len(merged)),
        "candidate_mrr": float(merged["candidate_rr"].mean()),
        "baseline_mrr": float(merged["baseline_rr"].mean()),
        "mean_delta_mrr_querywise": float(merged["delta_rr"].mean()),
        "mean_delta_mrr_seedwise": float(seed_deltas.mean()),
        "std_delta_mrr_seedwise": float(seed_deltas.std(ddof=1)) if len(seed_deltas) > 1 else 0.0,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_ci_95_querywise": {"low": low, "high": high},
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
    return payload, summary_row


def main() -> None:
    args = parse_args()
    method_specs = {}
    if args.candidate_query_rows:
        key = args.candidate_key or "candidate"
        method_specs[key] = {
            "label": args.candidate_label or key,
            "query_rows": args.candidate_query_rows,
        }
    else:
        method_specs = {key: METHODS[key] for key in args.methods}

    baseline = load_baseline_rows(Path(args.baseline_query_rows))
    out_dir = Path(args.out_dir)
    loaded_methods = {}

    for method_key, method_spec in method_specs.items():
        candidate_paths = expand_inputs(method_spec["query_rows"])
        candidate = load_candidate_rows(candidate_paths, args.candidate_column)
        loaded_methods[method_key] = {
            "label": method_spec["label"],
            "paths": candidate_paths,
            "frame": candidate,
        }

        for key, spec in BASELINE_COMPARISONS.items():
            merged = candidate.merge(
                baseline[["seed", "query_id", spec["right_column"]]],
                on=["seed", "query_id"],
                how="inner",
            )
            if len(merged) != len(candidate):
                raise RuntimeError(
                    f"Pairing mismatch for {method_key}/{key}: candidate_rows={len(candidate)}, paired_rows={len(merged)}"
                )
            merged = merged.rename(columns={spec["right_column"]: "baseline_rr"})
            payload, summary_row = comparison_payload(
                merged=merged,
                left_label=method_spec["label"],
                right_label=spec["right_label"],
                comparison=f"{method_spec['label']} vs {spec['right_label']}",
                left_column=args.candidate_column,
                right_column=spec["right_column"],
                candidate_paths=candidate_paths,
                baseline_source=str(args.baseline_query_rows),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
            write_outputs(out_dir, f"significance_{method_key}_{spec['out_suffix']}", payload, summary_row)

    if args.include_method_pairs and {"ca_s2", "ca_s3"}.issubset(loaded_methods):
        left = loaded_methods["ca_s2"]
        right = loaded_methods["ca_s3"]
        merged = left["frame"].merge(
            right["frame"].rename(columns={"candidate_rr": "baseline_rr"}),
            on=["seed", "query_id"],
            how="inner",
        )
        if len(merged) != len(left["frame"]):
            raise RuntimeError(f"Pairing mismatch for ca_s2_vs_ca_s3: left={len(left['frame'])}, paired={len(merged)}")
        payload, summary_row = comparison_payload(
            merged=merged,
            left_label=left["label"],
            right_label=right["label"],
            comparison=f"{left['label']} vs {right['label']}",
            left_column=args.candidate_column,
            right_column=args.candidate_column,
            candidate_paths=left["paths"],
            baseline_source=";".join(path.as_posix() for path in right["paths"]),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_outputs(out_dir, "significance_ca_s2_vs_ca_s3", payload, summary_row)


if __name__ == "__main__":
    main()
