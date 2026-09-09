#!/usr/bin/env python3
"""Audit how far the final Anchored Dynamic policy moves from its static anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_paper_a_risk_coverage import PAIR_SPECS, portable_path


TOLERANCE = 1e-12
BUCKETS = (
    ("0", 0),
    ("(0,0.05]", 1),
    ("(0.05,0.10]", 2),
    ("(0.10,0.20]", 3),
    (">0.20", 4),
)

SUMMARY_COLUMNS = (
    "split",
    "dataset",
    "pair",
    "scope",
    "seed",
    "direction",
    "n_observations",
    "mrr_anchored",
    "global_mrr",
    "delta_mrr_vs_global",
    "n_changed_alpha",
    "changed_alpha_rate",
    "n_exact_anchor",
    "exact_anchor_rate",
    "mean_abs_alpha_delta",
    "median_abs_alpha_delta",
    "p90_abs_alpha_delta",
    "p95_abs_alpha_delta",
    "max_abs_alpha_delta",
    "positive_shift_rate",
    "negative_shift_rate",
    "boundary_alpha_rate",
    "saturation_rate",
    "fallback_rate",
)

BUCKET_COLUMNS = (
    "split",
    "dataset",
    "pair",
    "scope",
    "seed",
    "direction",
    "magnitude_bucket",
    "bucket_order",
    "n",
    "observation_rate",
    "delta_mrr_contribution",
    "mean_delta_rr_within_bucket",
    "harm_rate",
    "mean_harm",
    "mean_benefit",
    "n_harmful",
    "n_beneficial",
    "n_unchanged",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/correction_magnitude",
    )
    parser.add_argument(
        "--report-path",
        default="docs/reports/paper_a_correction_magnitude_report.md",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths(repo_root: Path, spec: dict[str, str], split: str) -> tuple[Path, Path]:
    base = (
        repo_root
        / "outputs"
        / spec["dataset_dir"]
        / "anchored_dynamic"
        / f"{spec['pair_dir']}_seed123"
    )
    rows = (
        base / "anchored_crossfit" / "dev_anchored_query_rows.csv"
        if split == "dev"
        else base / "test_anchored" / "test_locked_query_rows.csv"
    )
    return rows, base / "dev_lock" / "anchored_dev_lock.json"


def split_columns(split: str) -> dict[str, str]:
    if split == "dev":
        return {
            "alpha0": "alpha0_crossfit",
            "alpha_final": "alpha_anchored_crossfit",
            "rr_global": "rr_global_crossfit",
            "rr_anchored": "rr_anchored_crossfit",
            "stored_delta": "alpha_delta_from_anchor",
        }
    return {
        "alpha0": "alpha0_locked",
        "alpha_final": "alpha_anchored_locked",
        "rr_global": "rr_global",
        "rr_anchored": "rr_anchored_locked",
        "stored_delta": "",
    }


def magnitude_bucket(abs_delta: np.ndarray) -> pd.Categorical:
    labels = np.full(len(abs_delta), ">0.20", dtype=object)
    labels[abs_delta <= 0.20 + TOLERANCE] = "(0.10,0.20]"
    labels[abs_delta <= 0.10 + TOLERANCE] = "(0.05,0.10]"
    labels[abs_delta <= 0.05 + TOLERANCE] = "(0,0.05]"
    labels[abs_delta <= TOLERANCE] = "0"
    return pd.Categorical(
        labels,
        categories=[name for name, _ in BUCKETS],
        ordered=True,
    )


def load_pair_split(
    repo_root: Path,
    spec: dict[str, str],
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, lock_path = source_paths(repo_root, spec, split)
    if not path.exists() or not lock_path.exists():
        raise FileNotFoundError(f"Missing correction-magnitude source: {path} or {lock_path}")
    columns = split_columns(split)
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "query_id",
        "query_key",
        "seed",
        "direction",
        "anchored_fallback",
        "anchored_saturated",
        columns["alpha0"],
        columns["alpha_final"],
        columns["rr_global"],
        columns["rr_anchored"],
    }
    if columns["stored_delta"]:
        required.add(columns["stored_delta"])
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
    if frame.empty or set(frame["split"].astype(str)) != {split}:
        raise RuntimeError(f"Unexpected split content in {path}")
    if frame["query_id"].duplicated().any():
        raise RuntimeError(f"Duplicate query_id in {path}")

    alpha0 = frame[columns["alpha0"]].to_numpy(dtype=np.float64)
    alpha_final = frame[columns["alpha_final"]].to_numpy(dtype=np.float64)
    alpha_delta = alpha_final - alpha0
    if columns["stored_delta"]:
        stored_delta = frame[columns["stored_delta"]].to_numpy(dtype=np.float64)
        if not np.allclose(alpha_delta, stored_delta, atol=TOLERANCE, rtol=0.0):
            raise RuntimeError(f"Stored alpha delta mismatch in {path}")
    abs_delta = np.abs(alpha_delta)
    rr_global = frame[columns["rr_global"]].to_numpy(dtype=np.float64)
    rr_anchored = frame[columns["rr_anchored"]].to_numpy(dtype=np.float64)
    delta_rr = rr_anchored - rr_global
    fallback_raw = frame["anchored_fallback"].to_numpy(dtype=np.int64)
    saturated_raw = frame["anchored_saturated"].to_numpy(dtype=np.int64)
    if not set(np.unique(fallback_raw)).issubset({0, 1}):
        raise RuntimeError(f"Non-binary fallback flag in {path}")
    if not set(np.unique(saturated_raw)).issubset({0, 1}):
        raise RuntimeError(f"Non-binary saturation flag in {path}")
    fallback = fallback_raw.astype(bool)
    saturated = saturated_raw.astype(bool)

    if not np.isfinite(
        np.column_stack((alpha0, alpha_final, alpha_delta, rr_global, rr_anchored))
    ).all():
        raise RuntimeError(f"Non-finite required values in {path}")
    if np.any((alpha0 < -TOLERANCE) | (alpha0 > 1.0 + TOLERANCE)):
        raise RuntimeError(f"Anchor outside [0,1] in {path}")
    if np.any((alpha_final < -TOLERANCE) | (alpha_final > 1.0 + TOLERANCE)):
        raise RuntimeError(f"Final alpha outside [0,1] in {path}")
    if np.any(fallback & (abs_delta > TOLERANCE)):
        raise RuntimeError(f"Fallback query does not return to exact anchor in {path}")
    if np.any((abs_delta <= TOLERANCE) & (np.abs(delta_rr) > TOLERANCE)):
        raise RuntimeError(f"Exact-anchor query differs from Global RR in {path}")
    if not np.allclose(alpha_final * 20.0, np.rint(alpha_final * 20.0), atol=1e-10):
        raise RuntimeError(f"Final alpha is not on the exact 0.05 grid in {path}")

    state = pd.DataFrame(
        {
            "split": split,
            "dataset": spec["dataset"],
            "pair": spec["pair"],
            "query_id": frame["query_id"].astype(str),
            "query_key": frame["query_key"].astype(str),
            "seed": frame["seed"].astype(int),
            "direction": frame["direction"].astype(str),
            "alpha0": alpha0,
            "alpha_final": alpha_final,
            "alpha_delta": alpha_delta,
            "abs_alpha_delta": abs_delta,
            "fallback": fallback,
            "saturated": saturated,
            "boundary_alpha": np.isclose(alpha_final, 0.0, atol=TOLERANCE)
            | np.isclose(alpha_final, 1.0, atol=TOLERANCE),
            "rr_global": rr_global,
            "rr_anchored": rr_anchored,
            "delta_rr": delta_rr,
            "magnitude_bucket": magnitude_bucket(abs_delta),
        }
    )
    bucket_counts = state["magnitude_bucket"].value_counts(sort=False)
    if int(bucket_counts.sum()) != len(state):
        raise RuntimeError(f"Magnitude buckets are not exhaustive in {path}")
    audit = {
        "split": split,
        "dataset": spec["dataset"],
        "pair": spec["pair"],
        "source_rows": portable_path(path, repo_root),
        "source_rows_sha256": sha256_file(path),
        "dev_lock": portable_path(lock_path, repo_root),
        "dev_lock_sha256": sha256_file(lock_path),
        "n_observations": int(len(state)),
        "stored_alpha_delta_available": bool(columns["stored_delta"]),
        "stored_alpha_delta_reconstruction_passed": True,
        "fallback_returns_exact_anchor": True,
        "exact_anchor_returns_global_rr": True,
        "final_alpha_exact_grid_check": True,
        "bucket_partition_passed": True,
    }
    return state, audit


def summarize(
    state: pd.DataFrame,
    *,
    scope: str,
    seed: str | int = "pooled",
    direction: str = "pooled",
) -> dict[str, Any]:
    delta = state["alpha_delta"].to_numpy(dtype=np.float64)
    abs_delta = state["abs_alpha_delta"].to_numpy(dtype=np.float64)
    changed = abs_delta > TOLERANCE
    exact = ~changed
    return {
        "split": str(state.iloc[0]["split"]),
        "dataset": str(state.iloc[0]["dataset"]),
        "pair": str(state.iloc[0]["pair"]),
        "scope": scope,
        "seed": seed,
        "direction": direction,
        "n_observations": int(len(state)),
        "mrr_anchored": float(state["rr_anchored"].mean()),
        "global_mrr": float(state["rr_global"].mean()),
        "delta_mrr_vs_global": float(state["delta_rr"].mean()),
        "n_changed_alpha": int(changed.sum()),
        "changed_alpha_rate": float(changed.mean()),
        "n_exact_anchor": int(exact.sum()),
        "exact_anchor_rate": float(exact.mean()),
        "mean_abs_alpha_delta": float(np.mean(abs_delta)),
        "median_abs_alpha_delta": float(np.median(abs_delta)),
        "p90_abs_alpha_delta": float(np.quantile(abs_delta, 0.90)),
        "p95_abs_alpha_delta": float(np.quantile(abs_delta, 0.95)),
        "max_abs_alpha_delta": float(np.max(abs_delta)),
        "positive_shift_rate": float((delta > TOLERANCE).mean()),
        "negative_shift_rate": float((delta < -TOLERANCE).mean()),
        "boundary_alpha_rate": float(state["boundary_alpha"].mean()),
        "saturation_rate": float(state["saturated"].mean()),
        "fallback_rate": float(state["fallback"].mean()),
    }


def summarize_buckets(
    state: pd.DataFrame,
    *,
    scope: str,
    seed: str | int = "pooled",
    direction: str = "pooled",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_n = len(state)
    for bucket, order in BUCKETS:
        subset = state[state["magnitude_bucket"].eq(bucket)]
        delta = subset["delta_rr"].to_numpy(dtype=np.float64)
        harmful = delta < 0.0
        beneficial = delta > 0.0
        unchanged = ~(harmful | beneficial)
        rows.append(
            {
                "split": str(state.iloc[0]["split"]),
                "dataset": str(state.iloc[0]["dataset"]),
                "pair": str(state.iloc[0]["pair"]),
                "scope": scope,
                "seed": seed,
                "direction": direction,
                "magnitude_bucket": bucket,
                "bucket_order": order,
                "n": int(len(subset)),
                "observation_rate": float(len(subset) / total_n),
                "delta_mrr_contribution": float(delta.sum() / total_n)
                if len(subset)
                else 0.0,
                "mean_delta_rr_within_bucket": float(delta.mean())
                if len(subset)
                else None,
                "harm_rate": float(harmful.mean()) if len(subset) else None,
                "mean_harm": float((-delta[harmful]).mean())
                if harmful.any()
                else None,
                "mean_benefit": float(delta[beneficial].mean())
                if beneficial.any()
                else None,
                "n_harmful": int(harmful.sum()),
                "n_beneficial": int(beneficial.sum()),
                "n_unchanged": int(unchanged.sum()),
            }
        )
    return rows


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    summary: pd.DataFrame,
    buckets: pd.DataFrame,
    audits: list[dict[str, Any]],
) -> None:
    pair_rows = summary[summary["scope"].eq("pair")]
    test_rows = pair_rows[pair_rows["split"].eq("test")]
    test_buckets = buckets[
        buckets["scope"].eq("pair") & buckets["split"].eq("test")
    ]
    lines = [
        "# Paper A Anchored Dynamic Correction-Magnitude Audit",
        "",
        "## Main answer",
        "",
    ]
    for row in test_rows.itertuples(index=False):
        changed_buckets = test_buckets[
            test_buckets["dataset"].eq(row.dataset)
            & test_buckets["pair"].eq(row.pair)
            & ~test_buckets["magnitude_bucket"].eq("0")
        ]
        small = changed_buckets[
            changed_buckets["magnitude_bucket"].isin(
                ["(0,0.05]", "(0.05,0.10]"]
            )
        ]
        large = changed_buckets[
            changed_buckets["magnitude_bucket"].eq(">0.20")
        ]
        changed_n = int(changed_buckets["n"].sum())
        small_n = int(small["n"].sum())
        small_share = small_n / changed_n if changed_n else 0.0
        large_contribution = float(large["delta_mrr_contribution"].sum())
        contribution_share = (
            large_contribution / row.delta_mrr_vs_global
            if row.delta_mrr_vs_global > 0.0
            else float("nan")
        )
        lines.append(
            f"- {row.dataset} / {row.pair}: {100.0 * row.changed_alpha_rate:.2f}% "
            f"of observations change alpha; {100.0 * small_share:.2f}% of changed "
            f"observations stay within |Delta alpha| <= 0.10. Mean and p95 "
            f"|Delta alpha| are {row.mean_abs_alpha_delta:.4f} and "
            f"{row.p95_abs_alpha_delta:.4f}. Corrections above 0.20 cover "
            f"{100.0 * float(large['observation_rate'].sum()):.2f}% of all "
            f"observations and contribute "
            f"{large_contribution:+.6f} MRR"
            + (
                f" ({100.0 * contribution_share:.2f}% of the net Delta MRR)."
                if math.isfinite(contribution_share)
                else "."
            )
        )
    lines += [
        "",
        "The requested strong empirical claim is not supported across all four pairs. "
        "Anchor retention is pair-dependent: exact-anchor rates range from 5.74% to "
        "93.02% on locked TEST. Moreover, the >0.20 bucket supplies a majority of the "
        "net TEST Delta MRR in every pair. Conservative behavior is therefore supported "
        "in the structural sense of a static anchor, confidence fallback, and a "
        "DEV-locked bounded trust region, but not as a universal observation that gains "
        "come mainly from rare, <=0.10 shifts. This distinction should be preserved in "
        "the paper.",
        "",
        "## Definitions",
        "",
        "`alpha_delta = alpha_final - alpha0`. Exact anchor uses "
        f"`|alpha_delta| <= {TOLERANCE:g}`. Positive and negative shifts use strict "
        "signed comparisons outside that tolerance. Boundary alpha means final alpha is "
        "0 or 1. Saturation is the stored policy event in which the pre-clipped bounded "
        "proposal reaches or crosses 0 or 1; it is distinct from final boundary alpha "
        "after fallback and exact-grid mapping.",
        "",
        "For each magnitude bucket, Delta MRR contribution is "
        "`sum(rr_anchored - rr_global in bucket) / all observations in the group`. "
        "Bucket contributions therefore add exactly to the group's overall Delta MRR. "
        "Harm Rate is conditional on the bucket; Mean Harm and Mean Benefit are "
        "conditional on harmful and beneficial observations within that bucket.",
        "",
        "## Pair-level magnitude summary",
        "",
        "| Split | Dataset/Pair | Changed % | Exact anchor % | Mean abs Delta alpha | Median | p90 | p95 | Max | Positive % | Negative % | Boundary % | Saturation % | Delta MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows.itertuples(index=False):
        lines.append(
            f"| {row.split.upper()} | {row.dataset} / {row.pair} | "
            f"{100.0 * row.changed_alpha_rate:.2f} | "
            f"{100.0 * row.exact_anchor_rate:.2f} | "
            f"{row.mean_abs_alpha_delta:.4f} | "
            f"{row.median_abs_alpha_delta:.4f} | "
            f"{row.p90_abs_alpha_delta:.4f} | {row.p95_abs_alpha_delta:.4f} | "
            f"{row.max_abs_alpha_delta:.4f} | "
            f"{100.0 * row.positive_shift_rate:.2f} | "
            f"{100.0 * row.negative_shift_rate:.2f} | "
            f"{100.0 * row.boundary_alpha_rate:.2f} | "
            f"{100.0 * row.saturation_rate:.2f} | "
            f"{row.delta_mrr_vs_global:+.6f} |"
        )
    lines += [
        "",
        "## Locked TEST bucket decomposition",
        "",
        "| Dataset/Pair | Abs Delta alpha bucket | n (%) | Delta MRR contribution | Harm % | Mean Harm | Mean Benefit |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in test_buckets.itertuples(index=False):
        lines.append(
            f"| {row.dataset} / {row.pair} | {row.magnitude_bucket} | "
            f"{row.n} ({100.0 * row.observation_rate:.2f}) | "
            f"{row.delta_mrr_contribution:+.6f} | "
            f"{fmt(None if pd.isna(row.harm_rate) else 100.0 * row.harm_rate, 2)} | "
            f"{fmt(row.mean_harm)} | {fmt(row.mean_benefit)} |"
        )
    lines += [
        "",
        "## Reproducibility and information boundary",
        "",
        "This is a read-only post-hoc audit. It trains no model, changes no anchor, beta, "
        "confidence threshold, fallback decision, or TEST result, and performs no "
        "selection from TEST. DEV and TEST are reported separately.",
        "",
        "| Split | Dataset/Pair | Rows | Stored Delta checked | Fallback exact-anchor check | Source SHA-256 | DEV lock SHA-256 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for item in audits:
        lines.append(
            f"| {item['split'].upper()} | {item['dataset']} / {item['pair']} | "
            f"{item['n_observations']} | "
            f"{'yes' if item['stored_alpha_delta_available'] else 'reconstructed'} | "
            f"yes | `{item['source_rows_sha256']}` | `{item['dev_lock_sha256']}` |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if not report_path.is_absolute():
        report_path = repo_root / report_path
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_summary: list[dict[str, Any]] = []
    seed_summary: list[dict[str, Any]] = []
    direction_summary: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        for spec in PAIR_SPECS:
            state, source_audit = load_pair_split(repo_root, spec, split)
            audits.append(source_audit)
            pair_summary.append(summarize(state, scope="pair"))
            bucket_rows.extend(summarize_buckets(state, scope="pair"))
            for seed, subset in state.groupby("seed", sort=True):
                seed_summary.append(
                    summarize(subset, scope="seed", seed=int(seed))
                )
                bucket_rows.extend(
                    summarize_buckets(subset, scope="seed", seed=int(seed))
                )
            for direction, subset in state.groupby("direction", sort=True):
                direction_summary.append(
                    summarize(
                        subset,
                        scope="direction",
                        direction=str(direction),
                    )
                )
                bucket_rows.extend(
                    summarize_buckets(
                        subset,
                        scope="direction",
                        direction=str(direction),
                    )
                )

    summary = pd.DataFrame(pair_summary)[list(SUMMARY_COLUMNS)]
    by_seed = pd.DataFrame(seed_summary)[list(SUMMARY_COLUMNS)]
    by_direction = pd.DataFrame(direction_summary)[list(SUMMARY_COLUMNS)]
    buckets = pd.DataFrame(bucket_rows)[list(BUCKET_COLUMNS)]

    for keys, subset in buckets.groupby(
        ["split", "dataset", "pair", "scope", "seed", "direction"],
        dropna=False,
        sort=False,
    ):
        if int(subset["n"].sum()) <= 0:
            raise RuntimeError(f"Empty bucket group: {keys}")
        if not math.isclose(
            float(subset["observation_rate"].sum()), 1.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"Bucket rates do not sum to one: {keys}")
        expected = (
            summary
            if keys[3] == "pair"
            else by_seed
            if keys[3] == "seed"
            else by_direction
        )
        mask = (
            expected["split"].eq(keys[0])
            & expected["dataset"].eq(keys[1])
            & expected["pair"].eq(keys[2])
        )
        if keys[3] == "seed":
            mask &= expected["seed"].astype(str).eq(str(keys[4]))
        if keys[3] == "direction":
            mask &= expected["direction"].eq(keys[5])
        target = float(expected.loc[mask, "delta_mrr_vs_global"].iloc[0])
        if not math.isclose(
            float(subset["delta_mrr_contribution"].sum()),
            target,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Bucket contributions do not reconcile: {keys}")

    outputs = {
        "correction_magnitude_summary.csv": summary,
        "correction_magnitude_buckets.csv": buckets,
        "correction_magnitude_by_seed.csv": by_seed,
        "correction_magnitude_by_direction.csv": by_direction,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False, na_rep="")
    write_report(report_path, summary, buckets, audits)

    payload = {
        "schema_version": 1,
        "analysis": "Paper A Anchored Dynamic correction-magnitude audit",
        "alpha_delta_definition": "alpha_final - alpha0",
        "exact_anchor_tolerance": TOLERANCE,
        "magnitude_buckets": [name for name, _ in BUCKETS],
        "delta_mrr_contribution_definition": "sum(rr_anchored - rr_global within bucket) / all observations in the group",
        "pure_posthoc_analysis": True,
        "model_or_policy_parameters_modified": False,
        "test_used_for_selection": False,
        "test_is_diagnostic_only": True,
        "source_reconstruction_passed": True,
        "bucket_partition_and_reconciliation_passed": True,
        "source_audit": audits,
        "output_hashes": {
            name: sha256_file(output_dir / name) for name in outputs
        },
    }
    (output_dir / "correction_magnitude_audit.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote correction-magnitude audit to {output_dir}")
    print(f"[OK] wrote report to {report_path}")


if __name__ == "__main__":
    main()
