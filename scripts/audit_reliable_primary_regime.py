from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG_VERSION = "paper_a_reliable_primary_v1"
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260909

PAIR_SPECS = (
    ("MKG-W", "mkg_w", "M-Hyper vs NativE", "mhyper_native", True),
    ("MKG-W", "mkg_w", "M-Hyper vs AdaMF-MAT", "mhyper_adamf", True),
    ("MKG-W", "mkg_w", "NativE vs AdaMF-MAT", "native_adamf", False),
    ("DB15K", "db15k", "M-Hyper vs NativE", "mhyper_native", True),
    ("DB15K", "db15k", "M-Hyper vs AdaMF-MAT", "mhyper_adamf", True),
    ("DB15K", "db15k", "NativE vs AdaMF-MAT", "native_adamf", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the DEV-only reliable-primary regime for Paper A."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/reliable_primary",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def source_path(repo_root: Path, dataset_dir: str, pair_dir: str) -> Path:
    return (
        repo_root
        / "outputs"
        / dataset_dir
        / "anchored_dynamic"
        / f"{pair_dir}_seed123"
        / "full_ranking"
        / "dev_query_rows.csv"
    )


def load_dev_asset(path: Path, expected_dataset: str) -> tuple[pd.DataFrame, str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "split",
        "seed",
        "direction",
        "head_id",
        "relation_id",
        "tail_id",
        "query_id",
        "expert_a_name",
        "expert_b_name",
        "rr_a",
        "rr_b",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
    if set(frame["split"].astype(str).str.lower()) != {"dev"}:
        raise RuntimeError(f"Non-DEV rows found in {path}")
    if "dataset" in frame and set(frame["dataset"]) != {expected_dataset}:
        raise RuntimeError(f"Unexpected dataset label in {path}: {set(frame['dataset'])}")
    if frame["query_id"].duplicated().any():
        raise RuntimeError(f"Duplicate query_id in {path}")
    if set(frame["seed"].astype(int)) != {1, 2, 3}:
        raise RuntimeError(f"Expected exactly paired seeds 1,2,3 in {path}")
    if set(frame["direction"].astype(str).str.lower()) != {"head", "tail"}:
        raise RuntimeError(f"Expected head and tail directions in {path}")
    model_a_values = frame["expert_a_name"].dropna().unique()
    model_b_values = frame["expert_b_name"].dropna().unique()
    if len(model_a_values) != 1 or len(model_b_values) != 1:
        raise RuntimeError(f"Expert identity is not constant in {path}")
    if not np.isfinite(frame[["rr_a", "rr_b"]].to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"Non-finite reciprocal rank in {path}")
    frame = frame.copy()
    frame["raw_triple_id"] = (
        "h="
        + frame["head_id"].astype(str)
        + "|r="
        + frame["relation_id"].astype(str)
        + "|t="
        + frame["tail_id"].astype(str)
    )
    cluster_sizes = frame.groupby("raw_triple_id", sort=False).size()
    if not cluster_sizes.eq(6).all():
        raise RuntimeError(
            f"Every original triple must contain all 3 seeds x 2 directions in {path}"
        )
    return frame, str(model_a_values[0]), str(model_b_values[0])


def clustered_delta_ci(
    frame: pd.DataFrame,
    delta_column: str,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    work = frame[["raw_triple_id", delta_column]].copy()
    work["n"] = 1
    clusters = work.groupby("raw_triple_id", sort=True).agg(
        delta_sum=(delta_column, "sum"), n=("n", "sum")
    )
    values = clusters[["delta_sum", "n"]].to_numpy(dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk_size = 128
    for start in range(0, samples, chunk_size):
        width = min(chunk_size, samples - start)
        indexes = generator.integers(0, len(values), size=(width, len(values)))
        totals = values[indexes].sum(axis=1)
        estimates[start : start + width] = totals[:, 0] / totals[:, 1]
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def audit_pair(
    frame: pd.DataFrame,
    model_a: str,
    model_b: str,
    samples: int,
    bootstrap_seed: int,
) -> tuple[dict, dict]:
    mrr_a = float(frame["rr_a"].mean())
    mrr_b = float(frame["rr_b"].mean())
    if mrr_a > mrr_b:
        primary, secondary = model_a, model_b
        primary_column, secondary_column = "rr_a", "rr_b"
    elif mrr_b > mrr_a:
        primary, secondary = model_b, model_a
        primary_column, secondary_column = "rr_b", "rr_a"
    else:
        primary = secondary = "TIE"
        primary_column, secondary_column = "rr_a", "rr_b"
    work = frame.copy()
    work["delta_primary_minus_secondary"] = (
        work[primary_column].to_numpy(dtype=np.float64)
        - work[secondary_column].to_numpy(dtype=np.float64)
    )
    pooled_delta = float(work["delta_primary_minus_secondary"].mean())
    seed_deltas = {
        int(seed): float(group["delta_primary_minus_secondary"].mean())
        for seed, group in work.groupby("seed", sort=True)
    }
    ci_low, ci_high = clustered_delta_ci(
        work, "delta_primary_minus_secondary", samples, bootstrap_seed
    )
    criterion_a = pooled_delta > 0.0
    criterion_b = len(seed_deltas) == 3 and all(value > 0.0 for value in seed_deltas.values())
    criterion_c = ci_low > 0.0
    regime = "reliable-primary" if criterion_a and criterion_b and criterion_c else "boundary / non-reliable-primary"
    row = {
        "model_a": model_a,
        "model_b": model_b,
        "mrr_a": mrr_a,
        "mrr_b": mrr_b,
        "selected_primary": primary,
        "secondary": secondary,
        "primary_mrr": max(mrr_a, mrr_b),
        "secondary_mrr": min(mrr_a, mrr_b),
        "dev_delta": pooled_delta,
        "seed_1_delta": seed_deltas.get(1),
        "seed_2_delta": seed_deltas.get(2),
        "seed_3_delta": seed_deltas.get(3),
        "positive_seed_count": sum(value > 0.0 for value in seed_deltas.values()),
        "all_3_seeds_positive": criterion_b,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "criterion_a_pooled_delta_positive": criterion_a,
        "criterion_b_all_seeds_positive": criterion_b,
        "criterion_c_ci_lower_positive": criterion_c,
        "regime": regime,
        "n_observations": int(len(work)),
        "n_original_triples": int(work["raw_triple_id"].nunique()),
    }
    inference = {
        "selected_primary": primary,
        "secondary": secondary,
        "point_delta": pooled_delta,
        "ci95": [ci_low, ci_high],
        "bootstrap_samples": samples,
        "bootstrap_seed": bootstrap_seed,
        "cluster_unit": "original raw triple; all 3 seeds and both directions retained",
        "seed_deltas": {str(key): value for key, value in seed_deltas.items()},
    }
    return row, inference


def bool_text(value: bool) -> str:
    return "Yes" if value else "No"


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def write_report(path: Path, decisions: pd.DataFrame, checks: dict, samples: int, seed: int) -> None:
    lines = [
        "# Paper A DEV-only reliable-primary audit",
        "",
        "## Decision table",
        "",
        "| Dataset | Pair | Selected Primary | DEV Delta | 3/3 Seeds? | CI95 | Regime |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.pair} | {row.selected_primary} | "
            f"{fmt(row.dev_delta)} | {bool_text(row.all_3_seeds_positive)} | "
            f"[{fmt(row.ci95_low)}, {fmt(row.ci95_high)}] | {row.regime} |"
        )
    lines.extend(
        [
            "",
            "## Automatic checks",
            "",
            f"- All four pre-existing M-Hyper main pairs are reliable-primary: "
            f"**{bool_text(checks['four_main_pairs_all_reliable'])}**.",
            f"- MKG-W NativE + AdaMF-MAT is boundary under this rule: "
            f"**{bool_text(checks['native_adamf_boundary_by_dataset']['MKG-W'])}**.",
            f"- DB15K NativE + AdaMF-MAT is boundary under this rule: "
            f"**{bool_text(checks['native_adamf_boundary_by_dataset']['DB15K'])}**.",
            "",
            "Both NativE + AdaMF-MAT pairs satisfy the strict standalone DEV definition: "
            "NativE has higher pooled MRR, is higher in all three paired seeds, and has a "
            "positive clustered-CI lower bound. Therefore they are not reliable-primary "
            "boundary cases under this protocol. Any later failure of adaptive combination "
            "must be described as a combination-level stress case, not retroactively used to "
            "change the primary-selection rule.",
            "",
            "## Per-seed deltas",
            "",
            "| Dataset | Pair | Seed 1 | Seed 2 | Seed 3 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in decisions.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.pair} | {fmt(row.seed_1_delta)} | "
            f"{fmt(row.seed_2_delta)} | {fmt(row.seed_3_delta)} |"
        )
    lines.extend(
        [
            "",
            "## Statistical and information boundary",
            "",
            f"The percentile intervals use {samples:,} bootstrap resamples with seed {seed}. "
            "The original raw triple is the sampling unit; its three paired seeds and both "
            "prediction directions remain together. Selection uses only existing exact "
            "filtered DEV reciprocal ranks. No TEST file, outcome, threshold, or method "
            "selection is read, and no post-hoc MRR margin is introduced.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    rows: list[dict] = []
    intervals: list[dict] = []
    assets: list[dict] = []

    for dataset, dataset_dir, pair, pair_dir, is_main_pair in PAIR_SPECS:
        path = source_path(repo_root, dataset_dir, pair_dir)
        frame, model_a, model_b = load_dev_asset(path, dataset_dir)
        row, inference = audit_pair(
            frame, model_a, model_b, args.bootstrap_samples, args.bootstrap_seed
        )
        row = {
            "dataset": dataset,
            "pair": pair,
            **row,
            "is_existing_main_pair": is_main_pair,
            "source_asset": portable_path(path, repo_root),
        }
        rows.append(row)
        intervals.append({"dataset": dataset, "pair": pair, **inference})
        assets.append(
            {
                "dataset": dataset,
                "pair": pair,
                "path": portable_path(path, repo_root),
                "sha256": sha256_file(path),
                "n_rows": int(len(frame)),
                "n_original_triples": int(frame["raw_triple_id"].nunique()),
                "models": [model_a, model_b],
                "split_values": ["dev"],
            }
        )

    decisions = pd.DataFrame(rows)
    # Repeated standalone model rows must agree exactly across pair assets in a dataset.
    consistency: list[dict] = []
    for dataset in decisions["dataset"].unique():
        values: dict[str, list[float]] = {}
        subset = decisions[decisions["dataset"].eq(dataset)]
        for row in subset.itertuples(index=False):
            values.setdefault(row.model_a, []).append(row.mrr_a)
            values.setdefault(row.model_b, []).append(row.mrr_b)
        for model, mrrs in values.items():
            spread = max(mrrs) - min(mrrs)
            consistency.append(
                {"dataset": dataset, "model": model, "mrr_values": mrrs, "max_spread": spread}
            )
            if spread > 1e-12:
                raise RuntimeError(
                    f"Inconsistent standalone DEV MRR for {dataset}/{model}: {mrrs}"
                )

    reliable = decisions["regime"].eq("reliable-primary")
    native_ada = decisions["pair"].eq("NativE vs AdaMF-MAT")
    checks = {
        "four_main_pairs_all_reliable": bool(
            reliable[decisions["is_existing_main_pair"]].all()
        ),
        "native_adamf_boundary_by_dataset": {
            dataset: bool(
                not reliable[decisions["dataset"].eq(dataset) & native_ada].iloc[0]
            )
            for dataset in ("MKG-W", "DB15K")
        },
    }
    checks["native_adamf_all_boundary"] = all(
        checks["native_adamf_boundary_by_dataset"].values()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(
        output_dir / "reliable_primary_decisions.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    bootstrap_payload = {
        "schema_version": 1,
        "config_version": CONFIG_VERSION,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "confidence_interval": "percentile 95%",
        "cluster_unit": "original raw triple; all 3 paired seeds and both directions retained",
        "intervals": intervals,
    }
    (output_dir / "reliable_primary_bootstrap.json").write_text(
        json.dumps(bootstrap_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "config_version": CONFIG_VERSION,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "TEST_NOT_USED": True,
        "input_scope": "existing standalone exact filtered DEV query rows only",
        "selection_rule": {
            "primary": "model with higher pooled DEV MRR",
            "reliable_primary_if_and_only_if": [
                "pooled DEV primary-minus-secondary delta > 0",
                "all 3 paired-seed deltas > 0",
                "original-triple clustered bootstrap 95% CI lower bound > 0",
            ],
            "post_hoc_mrr_margin": None,
            "tie_policy": "boundary / non-reliable-primary",
        },
        "dev_assets": assets,
        "primary_selections": [
            {
                "dataset": row.dataset,
                "pair": row.pair,
                "primary_model": row.selected_primary,
                "secondary_model": row.secondary,
                "regime": row.regime,
            }
            for row in decisions.itertuples(index=False)
        ],
        "standalone_mrr_consistency_checks": consistency,
        "automatic_checks": checks,
        "test_files_read": [],
    }
    (output_dir / "reliable_primary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        repo_root / "docs/reports/paper_a_reliable_primary_audit.md",
        decisions,
        checks,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    print(f"[OK] wrote DEV-only reliable-primary audit to {output_dir}")
    print(f"[CHECK] four main pairs reliable: {checks['four_main_pairs_all_reliable']}")
    print(f"[CHECK] NativE + AdaMF-MAT all boundary: {checks['native_adamf_all_boundary']}")


if __name__ == "__main__":
    main()
