from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ablate_anchored_dynamic import (
    FEATURE_GROUPS,
    feature_matrix,
    fit_geometry_model,
    model_outputs,
)
from scripts.analyze_paper_a_negative_transfer import clustered_bootstrap
from scripts.crossfit_anchored_dynamic import (
    apply_policy,
    apply_query_soft,
    read_csv,
    select_bounded_policy,
)
from scripts.crossfit_heterogeneous_dev_policies import (
    alpha_column,
    assign_grouped_folds,
    best_alpha,
    triple_key,
)


DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 20260910
DEFAULT_FOLD_SEED = 20260901
DEFAULT_RANDOM_STATE = 20260902
UNCHANGED_TOLERANCE = 1e-12

VARIANTS = (
    ("A_full_anchored", "Full Anchored"),
    ("B_no_anchor_query_soft", "No Anchor / Query-soft"),
    ("C_no_bound", "No Bound"),
    ("D_no_fallback", "No Fallback"),
)

PAIR_SPECS = (
    {
        "dataset": "MKG-W",
        "dataset_dir": "mkg_w",
        "pair": "M-Hyper + NativE",
        "pair_dir": "mhyper_native",
    },
    {
        "dataset": "MKG-W",
        "dataset_dir": "mkg_w",
        "pair": "M-Hyper + AdaMF-MAT",
        "pair_dir": "mhyper_adamf",
    },
    {
        "dataset": "DB15K",
        "dataset_dir": "db15k",
        "pair": "M-Hyper + NativE",
        "pair_dir": "mhyper_native",
    },
    {
        "dataset": "DB15K",
        "dataset_dir": "db15k",
        "pair": "M-Hyper + AdaMF-MAT",
        "pair_dir": "mhyper_adamf",
    },
)

SUMMARY_COLUMNS = (
    "split",
    "dataset",
    "pair",
    "variant_id",
    "variant",
    "n_observations",
    "mrr",
    "global_mrr",
    "delta_mrr_vs_global",
    "delta_mrr_ci95_low",
    "delta_mrr_ci95_high",
    "harmful_query_rate",
    "harmful_query_rate_ci95_low",
    "harmful_query_rate_ci95_high",
    "beneficial_query_rate",
    "mean_harm",
    "mean_harm_ci95_low",
    "mean_harm_ci95_high",
    "fallback_rate",
    "changed_from_anchor_rate",
    "mean_abs_alpha_deviation",
    "p95_abs_alpha_deviation",
    "saturation_rate",
    "n_harmful",
    "n_beneficial",
    "n_unchanged",
    "n_original_triple_clusters",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the strictly matched four-variant Paper A core safety ablation using "
            "existing exact-ranking assets. Base MMKGC models are never retrained."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_a_safe_correction/core_ablation",
    )
    parser.add_argument(
        "--report-path",
        default="docs/reports/paper_a_core_ablation_report.md",
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


def pair_root(repo_root: Path, spec: dict) -> Path:
    return (
        repo_root
        / "outputs"
        / spec["dataset_dir"]
        / "anchored_dynamic"
        / f"{spec['pair_dir']}_seed123"
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_for_variant(
    rows: list[dict],
    *,
    variant_id: str,
    decision: np.ndarray,
    probability: np.ndarray,
    nonfinite: np.ndarray,
    alpha0: float,
    beta: float,
    threshold: float,
    alphas: tuple[float, ...],
) -> dict[str, np.ndarray]:
    if variant_id == "B_no_anchor_query_soft":
        query_soft = apply_query_soft(
            rows,
            probability_a=probability,
            nonfinite=nonfinite,
            alpha0=alpha0,
            alphas=alphas,
        )
        return {
            **query_soft,
            "fallback": nonfinite.copy(),
            "saturated": np.zeros(len(rows), dtype=bool),
        }
    applied_beta = 1.0 if variant_id == "C_no_bound" else beta
    applied_threshold = 0.0 if variant_id == "D_no_fallback" else threshold
    return apply_policy(
        rows,
        decision=decision,
        probability_a=probability,
        nonfinite=nonfinite,
        alpha0=alpha0,
        beta=applied_beta,
        confidence_threshold=applied_threshold,
        alphas=alphas,
    )


def append_policy_rows(
    records: dict[str, list[dict]],
    rows: list[dict],
    *,
    split: str,
    spec: dict,
    fold: int | str,
    alpha0: float,
    beta: float,
    threshold: float,
    policies: dict[str, dict[str, np.ndarray]],
) -> None:
    labels = dict(VARIANTS)
    for variant_id, policy in policies.items():
        for index, source in enumerate(rows):
            applied = float(policy["applied"][index])
            global_column = "rr_global_crossfit" if split == "dev" else "rr_global"
            records[variant_id].append(
                {
                    "split": split,
                    "dataset": spec["dataset"],
                    "pair": spec["pair"],
                    "variant_id": variant_id,
                    "variant": labels[variant_id],
                    "query_id": source["query_id"],
                    "raw_triple_id": triple_key(source),
                    "head_id": int(source["head_id"]),
                    "relation_id": int(source["relation_id"]),
                    "tail_id": int(source["tail_id"]),
                    "seed": int(source["seed"]),
                    "direction": source["direction"],
                    "fold": fold,
                    "full_beta": beta,
                    "full_confidence_threshold": threshold,
                    "alpha0": alpha0,
                    "alpha_continuous": float(policy["continuous"][index]),
                    "alpha_applied": applied,
                    "abs_alpha_deviation": abs(applied - alpha0),
                    "rr_method": float(policy["rr"][index]),
                    "rr_global": float(source[global_column]),
                    "fallback": int(policy["fallback"][index]),
                    "saturated": int(policy["saturated"][index]),
                }
            )


def validate_group_contract(rows: list[dict], seeds: list[int]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[triple_key(row)] += 1
    expected = 2 * len(seeds)
    malformed = {key: count for key, count in counts.items() if count != expected}
    if malformed:
        raise RuntimeError(
            f"Expected {expected} seed-direction observations per raw triple; "
            f"found {len(malformed)} malformed clusters"
        )


def compare_dev_reference(
    records: dict[str, list[dict]],
    p3_rows_path: Path,
    p3_summary: dict,
) -> None:
    reference = {row["query_id"]: row for row in read_csv(p3_rows_path)}
    full = records["A_full_anchored"]
    if len(reference) != len(full):
        raise RuntimeError("Reconstructed Full DEV row count differs from P3 reference")
    for row in full:
        expected = reference.get(row["query_id"])
        if expected is None:
            raise RuntimeError(f"Missing P3 Full reference for {row['query_id']}")
        checks = (
            (row["alpha_applied"], float(expected["alpha_applied"])),
            (row["rr_method"], float(expected["rr_method"])),
            (row["fallback"], int(expected["fallback"])),
            (row["saturated"], int(expected["saturated"])),
        )
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in checks):
            raise RuntimeError(f"Reconstructed Full DEV policy differs for {row['query_id']}")

    expected_results = {row["config_id"]: row for row in p3_summary["results"]}
    for variant_id, config_id in (
        ("A_full_anchored", "expanded_selected"),
        ("B_no_anchor_query_soft", "query_soft_full"),
    ):
        observed_mrr = float(np.mean([row["rr_method"] for row in records[variant_id]]))
        expected_mrr = float(expected_results[config_id]["mrr"])
        if not math.isclose(observed_mrr, expected_mrr, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Reconstructed {variant_id} DEV MRR differs from P3")


def run_dev_pair(repo_root: Path, spec: dict) -> tuple[pd.DataFrame, dict]:
    root = pair_root(repo_root, spec)
    p3_path = root / "p3_ablation" / "dev_p3_summary.json"
    p3_rows_path = root / "p3_ablation" / "dev_p3_selected_query_rows.csv"
    p3 = read_json(p3_path)
    query_path = root / "baseline_crossfit" / "dev_crossfit_query_rows.csv"
    selection_path = root / "full_ranking" / "selection.json"
    rows = read_csv(query_path)
    selection = read_json(selection_path)
    alphas = tuple(float(value) for value in selection["alpha_grid"])
    betas = tuple(float(value) for value in p3["beta_grid"])
    thresholds = tuple(float(value) for value in p3["confidence_threshold_grid"])
    fold_seed = int(p3["fold_audit"]["fold_seed"])
    folds = int(p3["fold_audit"]["n_folds"])
    seeds = sorted({int(row["seed"]) for row in rows})
    validate_group_contract(rows, seeds)
    assignment, fold_audit = assign_grouped_folds(rows, folds, fold_seed)
    if fold_audit != p3["fold_audit"]:
        raise RuntimeError(f"Grouped fold reconstruction differs for {spec['pair']}")
    full_group_index = list(FEATURE_GROUPS).index("full_geometry")
    selected_reference = {int(row["fold"]): row for row in p3["selected_by_fold"]}
    records: dict[str, list[dict]] = defaultdict(list)
    selected_by_fold = []
    for fold in range(folds):
        train_rows = [row for row in rows if assignment[triple_key(row)] != fold]
        heldout_rows = [row for row in rows if assignment[triple_key(row)] == fold]
        alpha0, _ = best_alpha(train_rows, alphas)
        if not all(
            int(row["crossfit_fold"]) == fold + 1
            and math.isclose(
                float(row["alpha_global_crossfit"]),
                alpha0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in heldout_rows
        ):
            raise RuntimeError("DEV query rows do not match the grouped fold assignment")
        model, train_x, train_nonfinite = fit_geometry_model(
            train_rows,
            fields=tuple(FEATURE_GROUPS["full_geometry"]),
            random_state=DEFAULT_RANDOM_STATE + fold * 10 + full_group_index,
        )
        heldout_x, heldout_nonfinite = feature_matrix(
            heldout_rows,
            tuple(FEATURE_GROUPS["full_geometry"]),
        )
        train_decision, train_probability = model_outputs(model, train_x)
        heldout_decision, heldout_probability = model_outputs(model, heldout_x)
        beta, threshold, _ = select_bounded_policy(
            train_rows,
            decision=train_decision,
            probability_a=train_probability,
            nonfinite=train_nonfinite,
            alpha0=alpha0,
            betas=betas,
            confidence_thresholds=thresholds,
            alphas=alphas,
        )
        expected = selected_reference[fold + 1]
        if not (
            math.isclose(beta, float(expected["selected_beta"]), abs_tol=1e-12)
            and math.isclose(
                threshold,
                float(expected["selected_confidence_threshold"]),
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(f"Full DEV selection differs in fold {fold + 1}")
        policies = {
            variant_id: policy_for_variant(
                heldout_rows,
                variant_id=variant_id,
                decision=heldout_decision,
                probability=heldout_probability,
                nonfinite=heldout_nonfinite,
                alpha0=alpha0,
                beta=beta,
                threshold=threshold,
                alphas=alphas,
            )
            for variant_id, _ in VARIANTS
        }
        append_policy_rows(
            records,
            heldout_rows,
            split="dev",
            spec=spec,
            fold=fold + 1,
            alpha0=alpha0,
            beta=beta,
            threshold=threshold,
            policies=policies,
        )
        selected_by_fold.append(
            {
                "fold": fold + 1,
                "alpha0": alpha0,
                "beta": beta,
                "confidence_threshold": threshold,
                "train_observations": len(train_rows),
                "heldout_observations": len(heldout_rows),
            }
        )
    compare_dev_reference(records, p3_rows_path, p3)
    frame = pd.concat(
        [pd.DataFrame(records[variant_id]) for variant_id, _ in VARIANTS],
        ignore_index=True,
    )
    audit = {
        "split": "dev",
        "dataset": spec["dataset"],
        "pair": spec["pair"],
        "source_query_rows": portable_path(query_path, repo_root),
        "source_query_rows_sha256": sha256_file(query_path),
        "source_p3_summary": portable_path(p3_path, repo_root),
        "source_p3_summary_sha256": sha256_file(p3_path),
        "source_p3_full_rows": portable_path(p3_rows_path, repo_root),
        "source_p3_full_rows_sha256": sha256_file(p3_rows_path),
        "fold_seed": fold_seed,
        "random_state": DEFAULT_RANDOM_STATE,
        "model": "same P3 full_geometry logistic combiner per fold",
        "full_selection": "original training-fold beta/threshold selection protocol",
        "selected_by_fold": selected_by_fold,
        "reference_reconstruction_passed": True,
    }
    return frame, audit


def run_test_pair(repo_root: Path, spec: dict) -> tuple[pd.DataFrame, dict]:
    root = pair_root(repo_root, spec)
    lock_path = root / "dev_lock" / "anchored_dev_lock.json"
    query_path = root / "test_anchored" / "test_locked_query_rows.csv"
    lock = read_json(lock_path)
    rows = read_csv(query_path)
    alphas = tuple(float(value) for value in lock["alpha_grid"])
    alpha0 = float(lock["alpha0"])
    beta = float(lock["beta"])
    threshold = float(lock["confidence_threshold"])
    matrix, nonfinite = feature_matrix(
        rows,
        tuple(FEATURE_GROUPS["full_geometry"]),
    )
    del matrix
    decision = np.asarray([float(row["anchored_decision"]) for row in rows])
    probability = np.asarray([float(row["anchored_probability_a"]) for row in rows])
    policies = {
        variant_id: policy_for_variant(
            rows,
            variant_id=variant_id,
            decision=decision,
            probability=probability,
            nonfinite=nonfinite,
            alpha0=alpha0,
            beta=beta,
            threshold=threshold,
            alphas=alphas,
        )
        for variant_id, _ in VARIANTS
    }
    for variant_id, alpha_field, rr_field in (
        ("A_full_anchored", "alpha_anchored_locked", "rr_anchored_locked"),
        ("B_no_anchor_query_soft", "alpha_query_soft_locked", "rr_query_soft_locked"),
    ):
        policy = policies[variant_id]
        expected_alpha = np.asarray([float(row[alpha_field]) for row in rows])
        expected_rr = np.asarray([float(row[rr_field]) for row in rows])
        if not np.allclose(policy["applied"], expected_alpha, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Reconstructed TEST alpha differs for {variant_id}")
        if not np.allclose(policy["rr"], expected_rr, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Reconstructed TEST RR differs for {variant_id}")
    records: dict[str, list[dict]] = defaultdict(list)
    append_policy_rows(
        records,
        rows,
        split="test",
        spec=spec,
        fold="locked",
        alpha0=alpha0,
        beta=beta,
        threshold=threshold,
        policies=policies,
    )
    frame = pd.concat(
        [pd.DataFrame(records[variant_id]) for variant_id, _ in VARIANTS],
        ignore_index=True,
    )
    audit = {
        "split": "test",
        "dataset": spec["dataset"],
        "pair": spec["pair"],
        "source_query_rows": portable_path(query_path, repo_root),
        "source_query_rows_sha256": sha256_file(query_path),
        "source_lock": portable_path(lock_path, repo_root),
        "source_lock_sha256": sha256_file(lock_path),
        "alpha0": alpha0,
        "beta": beta,
        "confidence_threshold": threshold,
        "model": "existing final DEV-locked full_geometry logistic combiner outputs",
        "reference_reconstruction_passed": True,
        "test_selection_or_tuning": False,
    }
    return frame, audit


def point_metrics(frame: pd.DataFrame) -> dict:
    delta = frame["rr_method"].to_numpy(dtype=np.float64) - frame[
        "rr_global"
    ].to_numpy(dtype=np.float64)
    harmful = delta < -UNCHANGED_TOLERANCE
    beneficial = delta > UNCHANGED_TOLERANCE
    unchanged = ~(harmful | beneficial)
    harm = -delta[harmful]
    deviation = frame["abs_alpha_deviation"].to_numpy(dtype=np.float64)
    return {
        "n_observations": int(len(frame)),
        "mrr": float(frame["rr_method"].mean()),
        "global_mrr": float(frame["rr_global"].mean()),
        "delta_mrr_vs_global": float(delta.mean()),
        "harmful_query_rate": float(harmful.mean()),
        "beneficial_query_rate": float(beneficial.mean()),
        "mean_harm": float(harm.mean()) if harm.size else None,
        "fallback_rate": float(frame["fallback"].mean()),
        "changed_from_anchor_rate": float(
            (deviation > UNCHANGED_TOLERANCE).mean()
        ),
        "mean_abs_alpha_deviation": float(deviation.mean()),
        "p95_abs_alpha_deviation": float(np.quantile(deviation, 0.95)),
        "saturation_rate": float(frame["saturated"].mean()),
        "n_harmful": int(harmful.sum()),
        "n_beneficial": int(beneficial.sum()),
        "n_unchanged": int(unchanged.sum()),
        "n_original_triple_clusters": int(frame["raw_triple_id"].nunique()),
    }


def validate_matched_contract(all_rows: pd.DataFrame) -> None:
    for keys, pair_rows in all_rows.groupby(
        ["split", "dataset", "pair"], sort=False, observed=True
    ):
        variants = {
            variant_id: frame.sort_values("query_id").reset_index(drop=True)
            for variant_id, frame in pair_rows.groupby(
                "variant_id", sort=False, observed=True
            )
        }
        if set(variants) != {variant_id for variant_id, _ in VARIANTS}:
            raise RuntimeError(f"Missing matched variant for {keys}")
        full = variants["A_full_anchored"]
        no_anchor = variants["B_no_anchor_query_soft"]
        no_bound = variants["C_no_bound"]
        no_fallback = variants["D_no_fallback"]
        for comparison in (no_anchor, no_bound, no_fallback):
            if not full["query_id"].equals(comparison["query_id"]):
                raise RuntimeError(f"Observation-set mismatch for {keys}")
            for column in (
                "rr_global",
                "alpha0",
                "full_beta",
                "full_confidence_threshold",
            ):
                if not np.allclose(
                    full[column], comparison[column], rtol=0.0, atol=1e-12
                ):
                    raise RuntimeError(f"Matched {column} mismatch for {keys}")
        if not np.array_equal(full["fallback"], no_bound["fallback"]):
            raise RuntimeError(f"No Bound changed the Full fallback mask for {keys}")
        if np.any(no_fallback["fallback"].to_numpy() > full["fallback"].to_numpy()):
            raise RuntimeError(f"No Fallback introduced an extra fallback for {keys}")
        full_adapted = ~full["fallback"].to_numpy(dtype=bool)
        for column in ("alpha_applied", "rr_method"):
            if not np.allclose(
                full.loc[full_adapted, column],
                no_fallback.loc[full_adapted, column],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"No Fallback changed a non-fallback Full {column} for {keys}"
                )


def summarize(
    all_rows: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    summaries = []
    intervals = []
    groups = ["split", "dataset", "pair", "variant_id", "variant"]
    for keys, frame in all_rows.groupby(groups, sort=False, observed=True):
        identity = dict(zip(groups, keys, strict=True))
        metrics = point_metrics(frame)
        bootstrap_frame = frame[["raw_triple_id"]].copy()
        bootstrap_frame["delta_rr"] = (
            frame["rr_method"].to_numpy(dtype=np.float64)
            - frame["rr_global"].to_numpy(dtype=np.float64)
        )
        inference = clustered_bootstrap(
            bootstrap_frame,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        summaries.append(
            {
                **identity,
                **metrics,
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
        )
        intervals.append({**identity, **inference})
    return pd.DataFrame(summaries)[list(SUMMARY_COLUMNS)], intervals


def summarize_subgroups(all_rows: pd.DataFrame, column: str) -> pd.DataFrame:
    output = []
    groups = ["split", "dataset", "pair", "variant_id", "variant", column]
    for keys, frame in all_rows.groupby(groups, sort=False, observed=True):
        identity = dict(zip(groups, keys, strict=True))
        output.append({**identity, **point_metrics(frame)})
    return pd.DataFrame(output)


def aggregate_positivity(
    summary: pd.DataFrame,
    by_seed: pd.DataFrame,
    by_direction: pd.DataFrame,
) -> pd.DataFrame:
    output = []
    for (split, variant_id, variant), pair_rows in summary.groupby(
        ["split", "variant_id", "variant"], sort=False
    ):
        seed_rows = by_seed[
            by_seed["split"].eq(split) & by_seed["variant_id"].eq(variant_id)
        ]
        direction_rows = by_direction[
            by_direction["split"].eq(split)
            & by_direction["variant_id"].eq(variant_id)
        ]
        positive_pairs = int((pair_rows["delta_mrr_vs_global"] > 0.0).sum())
        positive_seeds = int((seed_rows["delta_mrr_vs_global"] > 0.0).sum())
        positive_directions = int(
            (direction_rows["delta_mrr_vs_global"] > 0.0).sum()
        )
        output.append(
            {
                "split": split,
                "variant_id": variant_id,
                "variant": variant,
                "positive_pairs": positive_pairs,
                "total_pairs": int(len(pair_rows)),
                "all_4_pairs_positive": positive_pairs == 4,
                "positive_pair_seeds": positive_seeds,
                "total_pair_seeds": int(len(seed_rows)),
                "all_12_pair_seeds_positive": positive_seeds == 12,
                "positive_pair_directions": positive_directions,
                "total_pair_directions": int(len(direction_rows)),
                "all_8_pair_directions_positive": positive_directions == 8,
            }
        )
    return pd.DataFrame(output)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="")


def main_table_lines(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Dataset/Pair | Variant | Delta MRR | Harm % | Mean Harm | Changed % |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        mean_harm = "--" if pd.isna(row.mean_harm) else f"{row.mean_harm:.6f}"
        lines.append(
            f"| {row.dataset} / {row.pair} | {row.variant} | "
            f"{row.delta_mrr_vs_global:+.6f} | "
            f"{100.0 * row.harmful_query_rate:.2f} | {mean_harm} | "
            f"{100.0 * row.changed_from_anchor_rate:.2f} |"
        )
    return lines


def write_report(
    path: Path,
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    audits: list[dict],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    test = summary[summary["split"].eq("test")]
    stress = test[
        test["dataset"].eq("DB15K")
        & test["pair"].eq("M-Hyper + AdaMF-MAT")
    ]
    aggregate_test = aggregate[aggregate["split"].eq("test")]
    lookup = test.set_index(["dataset", "pair", "variant_id"])
    stress_lookup = stress.set_index("variant_id")
    full_positive = aggregate_test[
        aggregate_test["variant_id"].eq("A_full_anchored")
    ].iloc[0]
    no_anchor_positive = aggregate_test[
        aggregate_test["variant_id"].eq("B_no_anchor_query_soft")
    ].iloc[0]
    no_anchor_higher_delta = 0
    no_bound_worse = 0
    no_bound_higher_harm = 0
    no_bound_higher_mean_harm = 0
    no_bound_larger_deviation = 0
    for spec in PAIR_SPECS:
        key = (spec["dataset"], spec["pair"])
        full = lookup.loc[(*key, "A_full_anchored")]
        no_anchor = lookup.loc[(*key, "B_no_anchor_query_soft")]
        no_bound = lookup.loc[(*key, "C_no_bound")]
        no_anchor_higher_delta += int(
            no_anchor["delta_mrr_vs_global"] > full["delta_mrr_vs_global"]
        )
        no_bound_worse += int(
            no_bound["delta_mrr_vs_global"] < full["delta_mrr_vs_global"]
        )
        no_bound_higher_harm += int(
            no_bound["harmful_query_rate"] > full["harmful_query_rate"]
        )
        no_bound_higher_mean_harm += int(
            no_bound["mean_harm"] > full["mean_harm"]
        )
        no_bound_larger_deviation += int(
            no_bound["mean_abs_alpha_deviation"]
            > full["mean_abs_alpha_deviation"]
        )
    lines = [
        "# Paper A Core Safety-Component Ablation",
        "",
        "## Scope and matched design",
        "",
        "This experiment changes only the requested safety component. It retrains no base "
        "MMKGC model and does not alter Anchored Dynamic. Grouped OOF DEV reuses the exact "
        "P3 folds and full-geometry logistic model. Full selects beta and confidence "
        "threshold on each outer training fold using the existing protocol; No Bound and "
        "No Fallback inherit that fold's Full winner. TEST uses the immutable final DEV "
        "lock and performs no tuning.",
        "",
        "No Anchor is the current Query-soft logistic baseline. No Bound fixes beta=1.0 "
        "while retaining Full's alpha0, model, confidence mechanism, and threshold. No "
        "Fallback retains Full's alpha0, beta, and model but fixes the confidence threshold "
        "to zero; only non-finite features may fall back.",
        "",
        "## TEST main table",
        "",
        *main_table_lines(test),
        "",
        "## Aggregate positivity",
        "",
        "| Variant | Pairs positive | Pair-seeds positive | Pair-directions positive |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in aggregate_test.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.positive_pairs}/{row.total_pairs} | "
            f"{row.positive_pair_seeds}/{row.total_pair_seeds} | "
            f"{row.positive_pair_directions}/{row.total_pair_directions} |"
        )
    lines.extend(
        [
            "",
            "## Answers to the core questions",
            "",
            f"- Anchor: Full is positive on {full_positive['positive_pairs']}/4 TEST "
            f"pairs, {full_positive['positive_pair_seeds']}/12 pair-seeds, and "
            f"{full_positive['positive_pair_directions']}/8 pair-directions. Query-soft "
            f"is positive on {no_anchor_positive['positive_pairs']}/4, "
            f"{no_anchor_positive['positive_pair_seeds']}/12, and "
            f"{no_anchor_positive['positive_pair_directions']}/8, respectively. This "
            "shows that anchoring reduces pair-dependent negative transfer and improves "
            "sign consistency. It is not uniform dominance: Query-soft has a larger Delta "
            f"MRR on {no_anchor_higher_delta}/4 TEST pairs.",
            f"- Bound: fixing beta=1.0 reduces Delta MRR relative to Full on "
            f"{no_bound_worse}/4 TEST pairs and raises harm rate on "
            f"{no_bound_higher_harm}/4; it also raises conditional Mean Harm on "
            f"{no_bound_higher_mean_harm}/4 and mean alpha deviation on "
            f"{no_bound_larger_deviation}/4. Thus the local bound is supported as a "
            "conservative risk-control component, even when beta=1.0 obtains greater "
            "average utility.",
            "- Fallback stress case: on DB15K / M-Hyper + AdaMF-MAT, Full has Delta MRR "
            f"{stress_lookup.loc['A_full_anchored', 'delta_mrr_vs_global']:+.6f} and harm "
            f"rate {100.0 * stress_lookup.loc['A_full_anchored', 'harmful_query_rate']:.2f}%; "
            "No Fallback has Delta MRR "
            f"{stress_lookup.loc['D_no_fallback', 'delta_mrr_vs_global']:+.6f} and harm "
            f"rate {100.0 * stress_lookup.loc['D_no_fallback', 'harmful_query_rate']:.2f}%. "
            "The difference isolates the protection supplied by the locked confidence "
            "fallback.",
            "",
            "## Statistical inference",
            "",
            f"Delta MRR, Harm Rate, and Mean Harm intervals use {bootstrap_samples:,} "
            f"percentile clustered-bootstrap samples with seed {bootstrap_seed}. The "
            "cluster is the original raw triple, so all seeds and both directions remain "
            "inside one resampling unit.",
            "",
            "## Audit boundary",
            "",
            "The reconstructed Full and Query-soft DEV results match the existing P3 "
            "assets. Their locked TEST alpha and reciprocal ranks match the existing TEST "
            "query rows. Source hashes and fold-wise Full selections are recorded in "
            "`outputs/paper_a_safe_correction/core_ablation/audit.json`.",
        ]
    )
    if any(audit.get("test_selection_or_tuning") for audit in audits):
        raise RuntimeError("Audit indicates forbidden TEST selection")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    report_path = (repo_root / args.report_path).resolve()
    frames = []
    audits = []
    for spec in PAIR_SPECS:
        dev, dev_audit = run_dev_pair(repo_root, spec)
        test, test_audit = run_test_pair(repo_root, spec)
        frames.extend((dev, test))
        audits.extend((dev_audit, test_audit))
    all_rows = pd.concat(frames, ignore_index=True)
    validate_matched_contract(all_rows)
    summary, intervals = summarize(
        all_rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    by_seed = summarize_subgroups(all_rows, "seed")
    by_direction = summarize_subgroups(all_rows, "direction")
    aggregate = aggregate_positivity(summary, by_seed, by_direction)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "core_ablation_summary.csv", summary)
    write_csv(output_dir / "core_ablation_by_seed.csv", by_seed)
    write_csv(output_dir / "core_ablation_by_direction.csv", by_direction)
    write_csv(output_dir / "core_ablation_aggregate.csv", aggregate)
    (output_dir / "core_ablation_main_table.md").write_text(
        "\n".join(main_table_lines(summary[summary["split"].eq("test")])) + "\n",
        encoding="utf-8",
    )
    (output_dir / "clustered_bootstrap_ci.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
                "cluster_unit": (
                    "original raw triple; all seeds and both directions retained"
                ),
                "intervals": intervals,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    audit_payload = {
        "schema_version": 1,
        "analysis": "strictly matched Paper A core safety-component ablation",
        "variants": {
            "A_full_anchored": (
                "Full selected beta and confidence threshold from existing DEV protocol"
            ),
            "B_no_anchor_query_soft": "current Query-soft logistic baseline",
            "C_no_bound": "same as Full with beta fixed to 1.0",
            "D_no_fallback": "same as Full with confidence threshold fixed to 0",
        },
        "no_base_model_training": True,
        "test_selection_or_tuning": False,
        "ablation_specific_hyperparameter_search": False,
        "matched_contract_validation_passed": True,
        "source_audit": audits,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        report_path,
        summary,
        aggregate,
        audits,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    print(f"[OK] wrote core ablation to {output_dir}")
    print(f"[OK] wrote report to {report_path}")


if __name__ == "__main__":
    main()
