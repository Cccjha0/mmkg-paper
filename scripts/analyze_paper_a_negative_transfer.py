from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


UNCHANGED_TOLERANCE = 1e-12
DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 20260909
METHOD_ORDER = ("Global", "Query-soft", "DynaSemble", "Anchored Dynamic")

PAIR_SPECS = (
    {
        "dataset": "MKG-W",
        "dataset_dir": "mkg_w",
        "pair": "M-Hyper + NativE",
        "pair_dir": "mhyper_native",
        "anchored_dir": "outputs/mkg_w/anchored_dynamic/mhyper_native_seed123",
    },
    {
        "dataset": "MKG-W",
        "dataset_dir": "mkg_w",
        "pair": "M-Hyper + AdaMF-MAT",
        "pair_dir": "mhyper_adamf",
        "anchored_dir": "outputs/mkg_w/anchored_dynamic/mhyper_adamf_seed123",
    },
    {
        "dataset": "DB15K",
        "dataset_dir": "db15k",
        "pair": "M-Hyper + NativE",
        "pair_dir": "mhyper_native",
        "anchored_dir": "outputs/db15k/anchored_dynamic/mhyper_native_seed123",
    },
    {
        "dataset": "DB15K",
        "dataset_dir": "db15k",
        "pair": "M-Hyper + AdaMF-MAT",
        "pair_dir": "mhyper_adamf",
        "anchored_dir": "outputs/db15k/anchored_dynamic/mhyper_adamf_seed123",
    },
)

IDENTITY_COLUMNS = (
    "query_id",
    "query_key",
    "seed",
    "direction",
    "head_id",
    "relation_id",
    "tail_id",
    "target_entity_id",
)

SUMMARY_COLUMNS = (
    "split",
    "dataset",
    "pair",
    "method",
    "scope",
    "seed",
    "direction",
    "n_observations",
    "mrr",
    "delta_mrr_vs_global",
    "delta_mrr_ci95_low",
    "delta_mrr_ci95_high",
    "harmful_query_rate",
    "harmful_query_rate_ci95_low",
    "harmful_query_rate_ci95_high",
    "beneficial_query_rate",
    "unchanged_query_rate",
    "mean_harm",
    "mean_harm_ci95_low",
    "mean_harm_ci95_high",
    "mean_benefit",
    "median_delta_rr",
    "n_harmful",
    "n_beneficial",
    "n_unchanged",
    "n_original_triple_clusters",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze query-level negative transfer for frozen Paper A methods. "
            "This script reads existing result assets and never trains or modifies a model."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--dynasemble-root",
        default="outputs/paper_a_safe_correction/dynasemble",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/negative_transfer",
    )
    parser.add_argument("--split", choices=("dev", "test", "both"), default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_source_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def source_paths(
    repo_root: Path,
    dynasemble_root: Path,
    spec: dict,
    split: str,
) -> tuple[Path, Path]:
    anchored_root = repo_root / spec["anchored_dir"]
    if split == "test":
        anchored = anchored_root / "test_anchored" / "test_locked_query_rows.csv"
    else:
        anchored = anchored_root / "dev_lock" / "dev_locked_query_rows.csv"
    dynasemble = dynasemble_root / spec["dataset_dir"] / spec["pair_dir"] / f"{split}_query_rows.csv"
    return anchored, dynasemble


def validate_identity(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    if left["query_id"].duplicated().any() or right["query_id"].duplicated().any():
        raise RuntimeError(f"Duplicate query_id in {label}")
    if len(left) != len(right) or set(left["query_id"]) != set(right["query_id"]):
        raise RuntimeError(f"Query identity mismatch in {label}")
    check = left[list(IDENTITY_COLUMNS)].merge(
        right[list(IDENTITY_COLUMNS)],
        on="query_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    for column in IDENTITY_COLUMNS[1:]:
        if not check[f"{column}_left"].equals(check[f"{column}_right"]):
            raise RuntimeError(f"Observation identity mismatch for {column} in {label}")


def load_pair_split(
    repo_root: Path,
    dynasemble_root: Path,
    spec: dict,
    split: str,
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    anchored_path, dynasemble_path = source_paths(repo_root, dynasemble_root, spec, split)
    if not anchored_path.exists():
        raise FileNotFoundError(anchored_path)
    anchored = pd.read_csv(anchored_path)
    required = {
        *IDENTITY_COLUMNS,
        "split",
        "rr_global",
        "rr_query_soft_locked",
        "rr_anchored_locked",
    }
    missing = required - set(anchored.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {anchored_path}: {sorted(missing)}")
    if set(anchored["split"]) != {split}:
        raise RuntimeError(f"Unexpected split values in {anchored_path}")
    source_audit = [
        {
            "dataset": spec["dataset"],
            "pair": spec["pair"],
            "split": split,
            "role": "anchored_locked_query_rows",
            "path": portable_source_path(anchored_path, repo_root),
            "sha256": sha256_file(anchored_path),
            "n_rows": len(anchored),
        }
    ]
    method_columns = {
        "Global": "rr_global",
        "Query-soft": "rr_query_soft_locked",
        "Anchored Dynamic": "rr_anchored_locked",
    }
    missing_methods = []
    if dynasemble_path.exists():
        dynasemble = pd.read_csv(dynasemble_path)
        required_dyna = {*IDENTITY_COLUMNS, "rr_global", "rr_dynasemble"}
        missing_dyna = required_dyna - set(dynasemble.columns)
        if missing_dyna:
            raise RuntimeError(
                f"Missing columns in {dynasemble_path}: {sorted(missing_dyna)}"
            )
        validate_identity(anchored, dynasemble, str(dynasemble_path))
        merged = anchored.merge(
            dynasemble[["query_id", "rr_global", "rr_dynasemble"]],
            on="query_id",
            suffixes=("", "_dynasemble"),
            validate="one_to_one",
        )
        error = np.max(
            np.abs(
                merged["rr_global"].to_numpy(dtype=np.float64)
                - merged["rr_global_dynasemble"].to_numpy(dtype=np.float64)
            )
        )
        if error > 5e-7:
            raise RuntimeError(
                f"Global reciprocal-rank mismatch in {dynasemble_path}: {error}"
            )
        anchored = merged.drop(columns=["rr_global_dynasemble"])
        method_columns["DynaSemble"] = "rr_dynasemble"
        source_audit.append(
            {
                "dataset": spec["dataset"],
                "pair": spec["pair"],
                "split": split,
                "role": "dynasemble_query_rows",
                "path": portable_source_path(dynasemble_path, repo_root),
                "sha256": sha256_file(dynasemble_path),
                "n_rows": len(dynasemble),
                "max_abs_global_rr_error": float(error),
            }
        )
    else:
        missing_methods.append("DynaSemble")

    base = anchored[list(IDENTITY_COLUMNS)].copy()
    base.insert(0, "split", split)
    base.insert(1, "dataset", spec["dataset"])
    base.insert(2, "pair", spec["pair"])
    base["raw_triple_id"] = (
        "h="
        + base["head_id"].astype(str)
        + "|r="
        + base["relation_id"].astype(str)
        + "|t="
        + base["tail_id"].astype(str)
    )
    frames = []
    for method in METHOD_ORDER:
        column = method_columns.get(method)
        if column is None:
            continue
        frame = base.copy()
        frame["method"] = method
        frame["rr_global"] = anchored["rr_global"].to_numpy(dtype=np.float64)
        frame["rr_method"] = anchored[column].to_numpy(dtype=np.float64)
        frame["delta_rr"] = frame["rr_method"] - frame["rr_global"]
        delta = frame["delta_rr"].to_numpy(dtype=np.float64)
        frame["outcome"] = np.where(
            np.abs(delta) <= UNCHANGED_TOLERANCE,
            "unchanged",
            np.where(delta > 0.0, "beneficial", "harmful"),
        )
        frame["observation_id"] = frame["query_id"] + "|method=" + method
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), source_audit, missing_methods


def point_summary(frame: pd.DataFrame) -> dict:
    delta = frame["delta_rr"].to_numpy(dtype=np.float64)
    harmful = delta < -UNCHANGED_TOLERANCE
    beneficial = delta > UNCHANGED_TOLERANCE
    unchanged = ~(harmful | beneficial)
    harm_magnitude = -delta[harmful]
    benefits = delta[beneficial]
    return {
        "n_observations": int(len(frame)),
        "mrr": float(frame["rr_method"].mean()),
        "delta_mrr_vs_global": float(delta.mean()),
        "harmful_query_rate": float(harmful.mean()),
        "beneficial_query_rate": float(beneficial.mean()),
        "unchanged_query_rate": float(unchanged.mean()),
        "mean_harm": float(harm_magnitude.mean()) if harm_magnitude.size else None,
        "mean_benefit": float(benefits.mean()) if benefits.size else None,
        "median_delta_rr": float(np.median(delta)),
        "n_harmful": int(harmful.sum()),
        "n_beneficial": int(beneficial.sum()),
        "n_unchanged": int(unchanged.sum()),
        "n_original_triple_clusters": int(frame["raw_triple_id"].nunique()),
    }


def clustered_bootstrap(
    frame: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict:
    if samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    work = frame[["raw_triple_id", "delta_rr"]].copy()
    work["n"] = 1
    work["harmful"] = (work["delta_rr"] < -UNCHANGED_TOLERANCE).astype(np.int64)
    work["harm_magnitude"] = np.where(
        work["harmful"].astype(bool), -work["delta_rr"], 0.0
    )
    clusters = work.groupby("raw_triple_id", sort=True).agg(
        delta_sum=("delta_rr", "sum"),
        n=("n", "sum"),
        harmful_count=("harmful", "sum"),
        harm_sum=("harm_magnitude", "sum"),
    )
    values = clusters[["delta_sum", "n", "harmful_count", "harm_sum"]].to_numpy(
        dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    delta_samples = np.empty(samples, dtype=np.float64)
    harm_rate_samples = np.empty(samples, dtype=np.float64)
    mean_harm_samples = np.full(samples, np.nan, dtype=np.float64)
    chunk_size = 128
    for start in range(0, samples, chunk_size):
        width = min(chunk_size, samples - start)
        indexes = generator.integers(0, values.shape[0], size=(width, values.shape[0]))
        totals = values[indexes].sum(axis=1)
        delta_samples[start : start + width] = totals[:, 0] / totals[:, 1]
        harm_rate_samples[start : start + width] = totals[:, 2] / totals[:, 1]
        has_harm = totals[:, 2] > 0
        chunk_mean_harm = np.full(width, np.nan, dtype=np.float64)
        chunk_mean_harm[has_harm] = totals[has_harm, 3] / totals[has_harm, 2]
        mean_harm_samples[start : start + width] = chunk_mean_harm

    def interval(values_: np.ndarray) -> list[float | None]:
        finite = values_[np.isfinite(values_)]
        if not finite.size:
            return [None, None]
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    delta_ci = interval(delta_samples)
    harm_rate_ci = interval(harm_rate_samples)
    mean_harm_ci = interval(mean_harm_samples)
    return {
        "n_original_triple_clusters": int(values.shape[0]),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "cluster_unit": "original raw triple; all available seeds and directions retained",
        "delta_mrr_ci95": delta_ci,
        "harmful_query_rate_ci95": harm_rate_ci,
        "mean_harm_ci95": mean_harm_ci,
    }


def summarize(
    query_rows: pd.DataFrame,
    scope: str,
    group_columns: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    result_rows = []
    bootstrap_rows = []
    for keys, frame in query_rows.groupby(
        group_columns,
        sort=False,
        dropna=False,
        observed=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = {
            column: value.item() if isinstance(value, np.generic) else value
            for column, value in zip(group_columns, keys, strict=True)
        }
        point = point_summary(frame)
        if identity["method"] == "Global":
            inference = {
                "n_original_triple_clusters": point["n_original_triple_clusters"],
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
                "cluster_unit": (
                    "original raw triple; all available seeds and directions retained"
                ),
                "delta_mrr_ci95": [0.0, 0.0],
                "harmful_query_rate_ci95": [0.0, 0.0],
                "mean_harm_ci95": [None, None],
            }
        else:
            inference = clustered_bootstrap(frame, bootstrap_samples, bootstrap_seed)
        row = {
            **identity,
            "scope": scope,
            "seed": identity.get("seed", ""),
            "direction": identity.get("direction", ""),
            **point,
            "delta_mrr_ci95_low": inference["delta_mrr_ci95"][0],
            "delta_mrr_ci95_high": inference["delta_mrr_ci95"][1],
            "harmful_query_rate_ci95_low": inference[
                "harmful_query_rate_ci95"
            ][0],
            "harmful_query_rate_ci95_high": inference[
                "harmful_query_rate_ci95"
            ][1],
            "mean_harm_ci95_low": inference["mean_harm_ci95"][0],
            "mean_harm_ci95_high": inference["mean_harm_ci95"][1],
        }
        result_rows.append(row)
        bootstrap_rows.append({**identity, "scope": scope, **inference})
    result = pd.DataFrame(result_rows)
    return result[list(SUMMARY_COLUMNS)], bootstrap_rows


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, na_rep="")


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Dataset/Pair | Method | Delta MRR | Harm % | Benefit % | Unchanged % | Mean Harm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.dataset} / {row.pair} | {row.method} | "
            f"{fmt(row.delta_mrr_vs_global)} | {100.0 * row.harmful_query_rate:.2f} | "
            f"{100.0 * row.beneficial_query_rate:.2f} | "
            f"{100.0 * row.unchanged_query_rate:.2f} | {fmt(row.mean_harm)} |"
        )
    return lines


def write_markdown(
    path: Path,
    pooled: pd.DataFrame,
    stress: pd.DataFrame,
    missing_methods: list[dict],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    report_split = "test" if "test" in set(pooled["split"]) else str(pooled.iloc[0]["split"])
    report = pooled[pooled["split"].eq(report_split)]
    adaptive = report[report["method"].ne("Global")]
    anchored = report[report["method"].eq("Anchored Dynamic")]
    query_soft = report[report["method"].eq("Query-soft")]
    dynasemble = report[report["method"].eq("DynaSemble")]
    anchored_positive = int((anchored["delta_mrr_vs_global"] > 0.0).sum())
    anchored_positive_ci = int((anchored["delta_mrr_ci95_low"] > 0.0).sum())
    query_soft_positive = int((query_soft["delta_mrr_vs_global"] > 0.0).sum())
    query_soft_more_beneficial = int(
        (query_soft["beneficial_query_rate"] > query_soft["harmful_query_rate"]).sum()
    )
    dynasemble_positive = int((dynasemble["delta_mrr_vs_global"] > 0.0).sum())
    dynasemble_line = (
        f"- DynaSemble improves mean MRR on {dynasemble_positive}/4 pairs. Its four "
        "Delta MRR estimates and materially different harm rates quantify "
        "pair-sensitive behavior under a strong primary."
        if len(dynasemble) == 4
        else "- DynaSemble query-level outputs are incomplete for this split and are "
        "reported as missing rather than reconstructed or tuned."
    )
    stress_by_method = stress.set_index("method")

    def stress_value(method: str, column: str) -> float:
        return float(stress_by_method.loc[method, column])

    lines = [
        "# Paper A query-level negative-transfer analysis",
        "",
        f"## {report_split.upper()} main table",
        "",
        *markdown_table(report),
        "",
        "Delta MRR is the average of `rr_method - rr_global`. Mean Harm is the "
        "positive magnitude `mean(-delta_rr | delta_rr < 0)`. Outcomes with "
        f"`abs(delta_rr) <= {UNCHANGED_TOLERANCE:g}` are unchanged.",
        "",
        "Average performance improvement and query-level safety are different estimands: "
        "a method can improve MRR while still harming many queries, or improve more queries "
        "than it harms while losing MRR because harmful corrections are larger.",
        "",
        "## Result interpretation",
        "",
        f"- Anchored Dynamic improves mean MRR on {anchored_positive}/4 pairs; all "
        f"{anchored_positive_ci}/4 clustered 95% CIs have a positive lower bound. Its "
        f"harmful-query rate is not zero, ranging from "
        f"{100.0 * anchored['harmful_query_rate'].min():.2f}% to "
        f"{100.0 * anchored['harmful_query_rate'].max():.2f}%. Thus the supported claim "
        "is lower observed negative-transfer risk, not query-wise safety.",
        f"- Query-soft improves mean MRR on only {query_soft_positive}/4 pairs even "
        f"though beneficial queries outnumber harmful queries on "
        f"{query_soft_more_beneficial}/4 pairs. The sign reversal comes from loss "
        "magnitude: harmful corrections can be much larger than beneficial ones.",
        dynasemble_line,
        f"- Across all {len(adaptive)} adaptive pair-method results, Delta MRR alone "
        "does not identify how often rankings are harmed or how large conditional "
        "losses are; both dimensions should be reported.",
        "",
        "## Stress case: DB15K / M-Hyper + AdaMF-MAT",
        "",
        *markdown_table(stress),
        "",
        "In this stress case, Query-soft has Delta MRR "
        f"{stress_value('Query-soft', 'delta_mrr_vs_global'):.6f} with "
        f"{100.0 * stress_value('Query-soft', 'harmful_query_rate'):.2f}% harmful "
        "queries; "
        + (
            "DynaSemble has Delta MRR "
            f"{stress_value('DynaSemble', 'delta_mrr_vs_global'):.6f} with "
            f"{100.0 * stress_value('DynaSemble', 'harmful_query_rate'):.2f}% "
            "harmful queries; "
            if "DynaSemble" in stress_by_method.index
            else "DynaSemble is missing; "
        )
        + "Anchored Dynamic has Delta MRR "
        f"{stress_value('Anchored Dynamic', 'delta_mrr_vs_global'):.6f} with "
        f"{100.0 * stress_value('Anchored Dynamic', 'harmful_query_rate'):.2f}% "
        "harmful queries. This is a joint comparison of average utility and observed "
        "harm frequency, not a formal worst-case guarantee.",
        "",
        "## Statistical inference",
        "",
        f"Percentile 95% intervals use {bootstrap_samples:,} clustered bootstrap samples "
        f"with seed {bootstrap_seed}. The bootstrap unit is the original raw triple; all "
        "available seeds and both directions remain inside that unit. The six "
        "seed-direction observations are never treated as independent samples. Intervals "
        "for Delta MRR, Harm Rate, and Mean Harm are stored in `bootstrap_ci.json` and the "
        "summary CSV files.",
        "",
        "## Information boundary",
        "",
        "This is final TEST analysis of immutable existing query-level outputs. No model or "
        "combiner is trained, no threshold or method is selected, and no TEST statistic is "
        "fed back into Global, Query-soft, DynaSemble, or Anchored Dynamic.",
    ]
    if missing_methods:
        lines.extend(
            [
                "",
                "## Missing methods",
                "",
                *[
                    f"- {row['split']} / {row['dataset']} / {row['pair']}: "
                    f"{', '.join(row['methods'])}"
                    for row in missing_methods
                ],
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    dynasemble_root = (repo_root / args.dynasemble_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    splits = ("dev", "test") if args.split == "both" else (args.split,)
    frames = []
    source_audit = []
    missing_methods = []
    for split in splits:
        for spec in PAIR_SPECS:
            frame, audit, missing = load_pair_split(
                repo_root, dynasemble_root, spec, split
            )
            frames.append(frame)
            source_audit.extend(audit)
            if missing:
                missing_methods.append(
                    {
                        "split": split,
                        "dataset": spec["dataset"],
                        "pair": spec["pair"],
                        "methods": missing,
                    }
                )
    query_rows = pd.concat(frames, ignore_index=True)
    query_rows["method"] = pd.Categorical(
        query_rows["method"], categories=METHOD_ORDER, ordered=True
    )
    query_rows = query_rows.sort_values(
        [
            "split",
            "dataset",
            "pair",
            "method",
            "seed",
            "direction",
            "relation_id",
            "head_id",
            "tail_id",
        ],
        kind="stable",
    )

    base_groups = ["split", "dataset", "pair", "method"]
    pooled, pooled_bootstrap = summarize(
        query_rows,
        "pooled",
        base_groups,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    by_seed, seed_bootstrap = summarize(
        query_rows,
        "seed",
        [*base_groups, "seed"],
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    by_direction, direction_bootstrap = summarize(
        query_rows,
        "direction",
        [*base_groups, "direction"],
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "negative_transfer_summary.csv", pooled)
    write_csv(output_dir / "negative_transfer_by_seed.csv", by_seed)
    write_csv(output_dir / "negative_transfer_by_direction.csv", by_direction)
    write_csv(output_dir / "negative_transfer_query_rows.csv", query_rows)

    report_split = "test" if "test" in splits else splits[0]
    stress = pooled[
        pooled["split"].eq(report_split)
        & pooled["dataset"].eq("DB15K")
        & pooled["pair"].eq("M-Hyper + AdaMF-MAT")
        & pooled["method"].isin(("Query-soft", "DynaSemble", "Anchored Dynamic"))
    ].copy()
    write_csv(output_dir / "stress_case_db15k_mhyper_adamf.csv", stress)
    (output_dir / "stress_case_db15k_mhyper_adamf.md").write_text(
        "\n".join(markdown_table(stress)) + "\n", encoding="utf-8"
    )

    bootstrap_payload = {
        "schema_version": 1,
        "analysis": "Paper A query-level negative transfer relative to Global alpha",
        "unchanged_tolerance": UNCHANGED_TOLERANCE,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "cluster_unit": (
            "original raw triple; all seeds and both directions retained in pooled analysis"
        ),
        "test_is_final_analysis_only": True,
        "no_model_or_combiner_training": True,
        "no_test_driven_threshold_or_method_changes": True,
        "source_assets": source_audit,
        "missing_methods": missing_methods,
        "intervals": [*pooled_bootstrap, *seed_bootstrap, *direction_bootstrap],
    }
    (output_dir / "bootstrap_ci.json").write_text(
        json.dumps(bootstrap_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        output_dir / "negative_transfer_summary.md",
        pooled,
        stress,
        missing_methods,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    print(f"[OK] wrote negative-transfer analysis to {output_dir}")


if __name__ == "__main__":
    main()
