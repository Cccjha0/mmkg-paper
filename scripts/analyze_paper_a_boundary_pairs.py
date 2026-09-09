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


UNCHANGED_TOLERANCE = 1e-12
CONFIG_VERSION = "paper_a_boundary_pairs_v1"
QUERY_GEOMETRY_FIELDS = (
    "geometry_direction_tail",
    "geometry_a_top1",
    "geometry_a_top5_mean",
    "geometry_a_top1_top2_margin",
    "geometry_a_score_std",
    "geometry_b_top1",
    "geometry_b_top5_mean",
    "geometry_b_top1_top2_margin",
    "geometry_b_score_std",
    "geometry_top1_delta_a_minus_b",
    "geometry_top5_delta_a_minus_b",
    "geometry_margin_delta_a_minus_b",
    "geometry_std_delta_a_minus_b",
)
PAIR_SPECS = (
    ("MKG-W", "mkg_w", "NativE + AdaMF-MAT", "mkgw_native_adamf"),
    ("DB15K", "db15k", "NativE + AdaMF-MAT", "db15k_native_adamf"),
)
METHODS = (
    ("Strongest standalone", "rr_primary", "alpha_primary", None),
    ("Query-zscore 0.5", "rr_equal", "alpha_equal", None),
    ("Global alpha", "rr_global", "alpha_global_locked", None),
    ("Query-soft", "rr_query_soft", "alpha_query_soft_locked", "query_soft_fallback"),
    ("DynaSemble", "rr_dynasemble", "effective_alpha_expert_a", "dynasemble_fallback"),
    ("Anchored Dynamic", "rr_anchored", "alpha_anchored_locked", "anchored_fallback"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize immutable TEST results for the two Paper A stress pairs."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/boundary_pairs",
    )
    parser.add_argument(
        "--reliable-decisions",
        default=(
            "outputs/paper_a_safe_correction/reliable_primary/"
            "reliable_primary_decisions.csv"
        ),
    )
    parser.add_argument(
        "--report", default="docs/reports/paper_a_boundary_pair_report.md"
    )
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


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")


def load_pair(
    repo_root: Path,
    output_dir: Path,
    reliable: pd.DataFrame,
    dataset: str,
    dataset_dir: str,
    pair: str,
    pair_name: str,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    pair_dir = output_dir / dataset_dir / "native_adamf"
    anchored_path = pair_dir / "anchored" / "test_anchored" / "test_locked_query_rows.csv"
    anchored_lock_path = pair_dir / "anchored" / "dev_lock" / "anchored_dev_lock.json"
    dyna_path = pair_dir / "dynasemble" / "test_query_rows.csv"
    dyna_lock_path = pair_dir / "dynasemble" / "lock.json"
    for path in (anchored_path, anchored_lock_path, dyna_path, dyna_lock_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Boundary-pair pipeline is incomplete; missing {portable_path(path, repo_root)}"
            )

    anchored = pd.read_csv(anchored_path)
    dyna = pd.read_csv(dyna_path)
    required_anchored = {
        "query_id",
        "query_key",
        "split",
        "seed",
        "direction",
        "head_id",
        "relation_id",
        "tail_id",
        "expert_a_name",
        "expert_b_name",
        "rr_a",
        "rr_b",
        "rr_equal",
        "rr_global",
        "rr_query_soft_locked",
        "rr_anchored_locked",
        "alpha0_locked",
        "alpha_query_soft_locked",
        "alpha_anchored_locked",
        "anchored_fallback",
        *QUERY_GEOMETRY_FIELDS,
    }
    required_dyna = {
        "query_id",
        "seed",
        "direction",
        "rr_global",
        "rr_dynasemble",
        "effective_alpha_expert_a",
    }
    require_columns(anchored, required_anchored, anchored_path)
    require_columns(dyna, required_dyna, dyna_path)
    if set(anchored["split"]) != {"test"} or set(dyna["split"]) != {"test"}:
        raise RuntimeError(f"Only immutable TEST rows are accepted for {dataset}/{pair}")
    if anchored["query_id"].duplicated().any() or dyna["query_id"].duplicated().any():
        raise RuntimeError(f"Duplicate query_id for {dataset}/{pair}")
    if len(anchored) != len(dyna) or set(anchored["query_id"]) != set(dyna["query_id"]):
        raise RuntimeError(f"Anchored/DynaSemble identity mismatch for {dataset}/{pair}")

    decision = reliable[
        reliable["dataset"].eq(dataset)
        & reliable["pair"].eq("NativE vs AdaMF-MAT")
    ]
    if len(decision) != 1:
        raise RuntimeError(f"Missing unique DEV-only primary decision for {dataset}/{pair}")
    decision_row = decision.iloc[0]
    selected_primary = str(decision_row["selected_primary"])
    expert_a_values = set(anchored["expert_a_name"])
    if expert_a_values != {selected_primary}:
        raise RuntimeError(
            f"Pipeline orientation is not DEV-selected primary for {dataset}/{pair}: "
            f"selected={selected_primary}, expert_a={sorted(expert_a_values)}"
        )
    if set(anchored["expert_b_name"]) != {"AdaMF-MAT"}:
        raise RuntimeError(f"Unexpected secondary for {dataset}/{pair}")

    merged = anchored.merge(
        dyna[
            [
                "query_id",
                "seed",
                "direction",
                "rr_global",
                "rr_dynasemble",
                "effective_alpha_expert_a",
            ]
        ],
        on="query_id",
        suffixes=("", "_dyna"),
        validate="one_to_one",
    )
    for column in ("seed", "direction"):
        if not merged[column].equals(merged[f"{column}_dyna"]):
            raise RuntimeError(f"{column} mismatch for {dataset}/{pair}")
    global_error = float(
        np.max(
            np.abs(
                merged["rr_global"].to_numpy(dtype=np.float64)
                - merged["rr_global_dyna"].to_numpy(dtype=np.float64)
            )
        )
    )
    if global_error > 5e-7:
        raise RuntimeError(f"Global RR mismatch for {dataset}/{pair}: {global_error}")
    merged = merged.drop(columns=["seed_dyna", "direction_dyna", "rr_global_dyna"])
    merged.insert(0, "dataset_label", dataset)
    merged.insert(1, "pair_label", pair)
    merged["selected_primary"] = selected_primary
    merged["reliable_primary_regime"] = decision_row["regime"]
    merged["rr_primary"] = merged["rr_a"].astype(float)
    merged["rr_query_soft"] = merged["rr_query_soft_locked"].astype(float)
    merged["rr_anchored"] = merged["rr_anchored_locked"].astype(float)
    merged["alpha_primary"] = 1.0
    merged["alpha_equal"] = 0.5
    merged["alpha_global_locked"] = merged["alpha0_locked"].astype(float)
    feature_values = merged[list(QUERY_GEOMETRY_FIELDS)].to_numpy(dtype=np.float64)
    merged["query_soft_fallback"] = (~np.isfinite(feature_values).all(axis=1)).astype(int)
    merged["dynasemble_fallback"] = 0
    merged["raw_triple_id"] = (
        "h="
        + merged["head_id"].astype(str)
        + "|r="
        + merged["relation_id"].astype(str)
        + "|t="
        + merged["tail_id"].astype(str)
    )
    cluster_sizes = merged.groupby("raw_triple_id", sort=False).size()
    if not cluster_sizes.eq(6).all():
        raise RuntimeError(f"Incomplete raw-triple clusters for {dataset}/{pair}")

    lock = json.loads(anchored_lock_path.read_text(encoding="utf-8"))
    dyna_lock = json.loads(dyna_lock_path.read_text(encoding="utf-8"))
    pair_metadata = {
        "dataset": dataset,
        "pair": pair,
        "pair_name": pair_name,
        "selected_primary": selected_primary,
        "secondary": str(decision_row["secondary"]),
        "reliable_primary_regime": str(decision_row["regime"]),
        "dev_primary_delta": float(decision_row["dev_delta"]),
        "dev_primary_ci95": [
            float(decision_row["ci95_low"]),
            float(decision_row["ci95_high"]),
        ],
        "dev_all_3_seeds_positive": as_bool(
            decision_row["all_3_seeds_positive"]
        ),
        "alpha0": float(lock["alpha0"]),
        "anchored_beta": float(lock["beta"]),
        "anchored_confidence_threshold": float(lock["confidence_threshold"]),
        "dynasemble_learned_weight_expert": dyna_lock["method_config"][
            "learned_weight_expert"
        ],
    }
    sources = [
        {
            "dataset": dataset,
            "role": role,
            "path": portable_path(path, repo_root),
            "sha256": sha256_file(path),
        }
        for role, path in (
            ("anchored_test_rows", anchored_path),
            ("anchored_dev_lock", anchored_lock_path),
            ("dynasemble_test_rows", dyna_path),
            ("dynasemble_dev_lock", dyna_lock_path),
        )
    ]
    return merged, pair_metadata, sources


def summarize_method(
    frame: pd.DataFrame,
    method: str,
    rr_column: str,
    alpha_column: str,
    fallback_column: str | None,
) -> dict:
    rr = frame[rr_column].to_numpy(dtype=np.float64)
    global_rr = frame["rr_global"].to_numpy(dtype=np.float64)
    delta = rr - global_rr
    harmful = delta < -UNCHANGED_TOLERANCE
    beneficial = delta > UNCHANGED_TOLERANCE
    alpha = frame[alpha_column].to_numpy(dtype=np.float64)
    anchor = frame["alpha_global_locked"].to_numpy(dtype=np.float64)
    deviation = np.abs(alpha - anchor)
    return {
        "method": method,
        "n_observations": int(len(frame)),
        "mrr": float(rr.mean()),
        "delta_vs_global": float(delta.mean()),
        "harm_rate": float(harmful.mean()),
        "benefit_rate": float(beneficial.mean()),
        "mean_harm": float((-delta[harmful]).mean()) if harmful.any() else None,
        "fallback_rate": (
            float(frame[fallback_column].astype(float).mean())
            if fallback_column is not None
            else None
        ),
        "changed_alpha_rate": float((deviation > UNCHANGED_TOLERANCE).mean()),
        "mean_abs_alpha_deviation": float(deviation.mean()),
        "p95_abs_alpha_deviation": float(np.quantile(deviation, 0.95)),
    }


def summarize(frame: pd.DataFrame, scope: str, group: str | None = None) -> pd.DataFrame:
    rows = []
    groups = [("", frame)] if group is None else frame.groupby(group, sort=True)
    for group_value, subset in groups:
        for method, rr_column, alpha_column, fallback_column in METHODS:
            row = summarize_method(
                subset, method, rr_column, alpha_column, fallback_column
            )
            row.update(
                {
                    "dataset": str(subset["dataset_label"].iloc[0]),
                    "pair": str(subset["pair_label"].iloc[0]),
                    "split": "test",
                    "scope": scope,
                    "seed": int(group_value) if group == "seed" else "",
                    "direction": str(group_value) if group == "direction" else "",
                    "selected_primary": str(subset["selected_primary"].iloc[0]),
                    "reliable_primary_regime": str(
                        subset["reliable_primary_regime"].iloc[0]
                    ),
                    "n_original_triples": int(subset["raw_triple_id"].nunique()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def result_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Dataset | Method | MRR | Delta vs Global | Harm % | Mean Harm | Fallback % | Changed-alpha % | Mean abs(alpha-alpha0) | P95 abs(alpha-alpha0) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        fallback = "--" if pd.isna(row.fallback_rate) else f"{100 * row.fallback_rate:.2f}"
        lines.append(
            f"| {row.dataset} | {row.method} | {row.mrr:.6f} | "
            f"{row.delta_vs_global:+.6f} | {100 * row.harm_rate:.2f} | "
            f"{fmt(row.mean_harm)} | {fallback} | {100 * row.changed_alpha_rate:.2f} | "
            f"{row.mean_abs_alpha_deviation:.4f} | {row.p95_abs_alpha_deviation:.4f} |"
        )
    return lines


def write_report(
    path: Path,
    pooled: pd.DataFrame,
    by_seed: pd.DataFrame,
    by_direction: pd.DataFrame,
    pair_metadata: list[dict],
) -> None:
    anchored = pooled[pooled["method"].eq("Anchored Dynamic")]
    all_reliable = all(
        item["reliable_primary_regime"] == "reliable-primary"
        for item in pair_metadata
    )
    lines = [
        "# Paper A boundary / stress-pair report",
        "",
        "## Framing and prerequisite audit",
        "",
        "These two pairs were retained as pre-specified stress cases regardless of outcome. "
        "The frozen DEV-only reliable-primary rule selected NativE as primary for both pairs.",
        "",
        "| Dataset | Selected primary | DEV primary delta | 3/3 seeds? | DEV CI95 | Regime |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for item in pair_metadata:
        lines.append(
            f"| {item['dataset']} | {item['selected_primary']} | "
            f"{item['dev_primary_delta']:+.6f} | "
            f"{'Yes' if item['dev_all_3_seeds_positive'] else 'No'} | "
            f"[{item['dev_primary_ci95'][0]:.6f}, {item['dev_primary_ci95'][1]:.6f}] | "
            f"{item['reliable_primary_regime']} |"
        )
    lines.extend(["", "## Immutable TEST results", "", *result_table(pooled)])
    lines.extend(
        [
            "",
            "## Stability",
            "",
            "| Dataset | Method | Seed 1 delta | Seed 2 delta | Seed 3 delta | Head delta | Tail delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in pooled["dataset"].unique():
        for method in ("Query-soft", "DynaSemble", "Anchored Dynamic"):
            seed_rows = by_seed[
                by_seed["dataset"].eq(dataset) & by_seed["method"].eq(method)
            ].set_index("seed")
            direction_rows = by_direction[
                by_direction["dataset"].eq(dataset)
                & by_direction["method"].eq(method)
            ].set_index("direction")
            lines.append(
                f"| {dataset} | {method} | {seed_rows.loc[1, 'delta_vs_global']:+.6f} | "
                f"{seed_rows.loc[2, 'delta_vs_global']:+.6f} | "
                f"{seed_rows.loc[3, 'delta_vs_global']:+.6f} | "
                f"{direction_rows.loc['head', 'delta_vs_global']:+.6f} | "
                f"{direction_rows.loc['tail', 'delta_vs_global']:+.6f} |"
            )
    positive_pairs = int((anchored["delta_vs_global"] > 0.0).sum())
    anchored_seed = by_seed[by_seed["method"].eq("Anchored Dynamic")]
    anchored_direction = by_direction[by_direction["method"].eq("Anchored Dynamic")]
    lines.extend(
        [
            "",
            "## Boundary-case interpretation",
            "",
            f"Anchored Dynamic is positive against Global on {positive_pairs}/2 stress pairs, "
            f"{int((anchored_seed['delta_vs_global'] > 0).sum())}/6 pair-seeds, and "
            f"{int((anchored_direction['delta_vs_global'] > 0).sum())}/4 pair-directions. "
            "These observations characterize correction behavior; they do not license a claim "
            "that arbitrary model pairs universally benefit.",
            "",
        ]
    )
    if all_reliable:
        lines.append(
            "The proposed explanatory condition—absence of a reliable primary—does not occur "
            "in either evaluated pair. Consequently these results cannot establish that missing "
            "primary reliability is associated with poorer or less stable correction. If a method "
            "fails here, the evidence instead points to a combination-level limitation despite a "
            "reliable standalone primary; if it succeeds, that still does not support universal "
            "pairwise benefit. A genuine non-reliable-primary pair is required to test the stated "
            "association directly."
        )
    else:
        lines.append(
            "At least one pair lacks a reliable primary under the frozen DEV-only rule. The "
            "direction and stability table above should be used to judge whether that condition "
            "coincides with poorer correction, without treating two observational pairs as a "
            "universal causal guarantee."
        )
    lines.extend(
        [
            "",
            "## Information boundary",
            "",
            "Base MMKGC models were not retrained. Global alpha, Anchored beta/threshold, "
            "Query-soft, and DynaSemble selectors were selected or fitted on DEV only. TEST was "
            "applied immutably and used only for the displayed final analysis. No boundary-pair "
            "outcome changes the main method or its hyperparameters.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    reliable_path = (repo_root / args.reliable_decisions).resolve()
    report_path = (repo_root / args.report).resolve()
    if not reliable_path.exists():
        raise FileNotFoundError(reliable_path)
    reliable = pd.read_csv(reliable_path)
    frames = []
    metadata = []
    sources = [
        {
            "role": "dev_only_reliable_primary_decisions",
            "path": portable_path(reliable_path, repo_root),
            "sha256": sha256_file(reliable_path),
        }
    ]
    for spec in PAIR_SPECS:
        frame, pair_metadata, pair_sources = load_pair(
            repo_root, output_dir, reliable, *spec
        )
        frames.append(frame)
        metadata.append(pair_metadata)
        sources.extend(pair_sources)

    pooled_parts = []
    seed_parts = []
    direction_parts = []
    for frame in frames:
        pooled_parts.append(summarize(frame, "pooled"))
        seed_parts.append(summarize(frame, "seed", "seed"))
        direction_parts.append(summarize(frame, "direction", "direction"))
    pooled = pd.concat(pooled_parts, ignore_index=True)
    by_seed = pd.concat(seed_parts, ignore_index=True)
    by_direction = pd.concat(direction_parts, ignore_index=True)
    query_rows = pd.concat(frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output_dir / "boundary_pair_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    by_seed.to_csv(output_dir / "boundary_pair_results_by_seed.csv", index=False)
    by_direction.to_csv(output_dir / "boundary_pair_results_by_direction.csv", index=False)
    keep_columns = [
        "dataset_label",
        "pair_label",
        "selected_primary",
        "reliable_primary_regime",
        "query_id",
        "query_key",
        "raw_triple_id",
        "seed",
        "direction",
        "head_id",
        "relation_id",
        "tail_id",
        "rr_primary",
        "rr_equal",
        "rr_global",
        "rr_query_soft",
        "rr_dynasemble",
        "rr_anchored",
        "alpha_global_locked",
        "alpha_query_soft_locked",
        "effective_alpha_expert_a",
        "alpha_anchored_locked",
        "query_soft_fallback",
        "dynasemble_fallback",
        "anchored_fallback",
    ]
    query_rows[keep_columns].to_csv(
        output_dir / "boundary_pair_query_rows.csv", index=False
    )
    manifest = {
        "schema_version": 1,
        "config_version": CONFIG_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_models_retrained": False,
        "hyperparameter_selection_scope": "DEV only",
        "test_application": "immutable final apply and analysis only",
        "main_method_changed_from_boundary_results": False,
        "reliable_primary_protocol_applied_before_test_analysis": True,
        "pair_metadata": metadata,
        "source_assets": sources,
    }
    (output_dir / "boundary_pair_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(report_path, pooled, by_seed, by_direction, metadata)
    print(f"[OK] wrote boundary-pair analysis to {output_dir}")
    print(f"[OK] wrote report to {report_path}")


if __name__ == "__main__":
    main()
