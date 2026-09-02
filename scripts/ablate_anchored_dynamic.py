from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.query_geometry import QUERY_GEOMETRY_FIELDS
from scripts.crossfit_anchored_dynamic import (
    apply_policy,
    apply_query_soft,
    nearest_alpha,
    parse_grid,
    quantiles,
    read_csv,
    select_bounded_policy,
)
from scripts.crossfit_heterogeneous_dev_policies import (
    alpha_column,
    assign_grouped_folds,
    best_alpha,
    clustered_interval,
    metric,
    triple_key,
    write_csv,
)


FEATURE_GROUPS = {
    "single_confidence": (
        "geometry_direction_tail",
        "geometry_a_top1",
        "geometry_a_top5_mean",
        "geometry_a_top1_top2_margin",
        "geometry_a_score_std",
        "geometry_b_top1",
        "geometry_b_top5_mean",
        "geometry_b_top1_top2_margin",
        "geometry_b_score_std",
    ),
    "cross_expert_disagreement": (
        "geometry_direction_tail",
        "geometry_top1_delta_a_minus_b",
        "geometry_top5_delta_a_minus_b",
        "geometry_margin_delta_a_minus_b",
        "geometry_std_delta_a_minus_b",
    ),
    "full_geometry": QUERY_GEOMETRY_FIELDS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DEV-only grouped cross-fit ablations for anchored dynamic score combination."
        )
    )
    parser.add_argument("--query-rows", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260901)
    parser.add_argument("--random-state", type=int, default=20260902)
    parser.add_argument(
        "--betas",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
    )
    parser.add_argument("--confidence-thresholds", default="0.00,0.10,0.20,0.30")
    parser.add_argument("--anchor-strengths", default="0.00,0.25,0.50,0.75,1.00")
    parser.add_argument("--reference-beta", type=float, default=0.20)
    return parser.parse_args()


def feature_matrix(
    rows: list[dict], fields: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [[float(row[name]) for name in fields] for row in rows],
        dtype=np.float64,
    )
    nonfinite = ~np.isfinite(matrix).all(axis=1)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, nonfinite


def fit_geometry_model(
    train_rows: list[dict],
    *,
    fields: tuple[str, ...],
    random_state: int,
):
    train_x, train_nonfinite = feature_matrix(train_rows, fields)
    train_y = np.asarray(
        [int(float(row["rr_a"]) > float(row["rr_b"])) for row in train_rows],
        dtype=np.int64,
    )
    train_tie = np.asarray(
        [float(row["rr_a"]) == float(row["rr_b"]) for row in train_rows],
        dtype=bool,
    )
    fit_mask = ~train_tie
    if len(np.unique(train_y[fit_mask])) != 2:
        raise RuntimeError("Training fold does not contain both expert-winner classes")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
            solver="liblinear",
        ),
    )
    model.fit(train_x[fit_mask], train_y[fit_mask])
    return model, train_x, train_nonfinite


def model_outputs(model, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    decision = np.asarray(model.decision_function(matrix), dtype=np.float64)
    probability = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return decision, probability


def apply_probability_shrinkage(
    rows: list[dict],
    *,
    probability_a: np.ndarray,
    nonfinite: np.ndarray,
    alpha0: float,
    strength: float,
    alphas: tuple[float, ...],
) -> dict[str, np.ndarray]:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("anchor strength must lie in [0, 1]")
    continuous = float(alpha0) + float(strength) * (probability_a - float(alpha0))
    continuous = np.clip(continuous, 0.0, 1.0)
    continuous = np.where(nonfinite, float(alpha0), continuous)
    applied = np.asarray(
        [nearest_alpha(float(value), alphas, float(alpha0)) for value in continuous],
        dtype=np.float64,
    )
    rr = np.asarray(
        [float(row[alpha_column(float(alpha))]) for row, alpha in zip(rows, applied)],
        dtype=np.float64,
    )
    return {
        "confidence": np.abs(2.0 * probability_a - 1.0),
        "fallback": nonfinite,
        "unconstrained": continuous,
        "continuous": continuous,
        "applied": applied,
        "rr": rr,
        "saturated": np.zeros(len(rows), dtype=bool),
    }


def config_id(prefix: str, value: float) -> str:
    return f"{prefix}_{value:.2f}".replace(".", "p")


def observation_key(row: dict) -> tuple[int, str, int, int, int]:
    return (
        int(row["seed"]),
        str(row["direction"]),
        int(row["relation_id"]),
        int(row["head_id"]),
        int(row["tail_id"]),
    )


def append_records(
    records: dict[str, list[dict]],
    metadata: dict[str, dict],
    *,
    identifier: str,
    label: str,
    family: str,
    feature_group: str,
    rows: list[dict],
    fold: int,
    alpha0: float,
    policy: dict[str, np.ndarray],
    beta: float | str = "",
    threshold: float | str = "",
    strength: float | str = "",
) -> None:
    metadata.setdefault(
        identifier,
        {
            "config_id": identifier,
            "label": label,
            "family": family,
            "feature_group": feature_group,
            "beta": beta,
            "confidence_threshold": threshold,
            "anchor_strength": strength,
        },
    )
    for index, row in enumerate(rows):
        applied = float(policy["applied"][index])
        records[identifier].append(
            {
                "pair_name": row["pair_name"],
                "dataset": row["dataset"],
                "split": row["split"],
                "query_key": row["query_key"],
                "query_id": row["query_id"],
                "head_id": row["head_id"],
                "relation_id": row["relation_id"],
                "tail_id": row["tail_id"],
                "seed": row["seed"],
                "direction": row["direction"],
                "fold": fold,
                "alpha0": alpha0,
                "alpha_applied": applied,
                "alpha_delta": applied - float(alpha0),
                "rr_method": float(policy["rr"][index]),
                "rr_a": float(row["rr_a"]),
                "rr_global_crossfit": float(row["rr_global_crossfit"]),
                "rr_relation_crossfit": float(row["rr_relation_crossfit"]),
                "fallback": int(policy["fallback"][index]),
                "saturated": int(policy["saturated"][index]),
            }
        )


def global_policy(rows: list[dict], alpha0: float) -> dict[str, np.ndarray]:
    count = len(rows)
    rr = np.asarray([float(row["rr_global_crossfit"]) for row in rows], dtype=np.float64)
    return {
        "applied": np.full(count, alpha0, dtype=np.float64),
        "rr": rr,
        "fallback": np.zeros(count, dtype=bool),
        "saturated": np.zeros(count, dtype=bool),
    }


def summarize_config(config_rows: list[dict], meta: dict) -> dict:
    result = metric([float(row["rr_method"]) for row in config_rows])
    global_mrr = metric([float(row["rr_global_crossfit"]) for row in config_rows])["mrr"]
    interval = clustered_interval(
        config_rows,
        "rr_method",
        reference="rr_global_crossfit",
    )
    query_soft_interval = clustered_interval(
        config_rows,
        "rr_method",
        reference="rr_query_soft",
    )
    relation_interval = clustered_interval(
        config_rows,
        "rr_method",
        reference="rr_relation_crossfit",
    )
    seed_mrr = []
    for seed in sorted({int(row["seed"]) for row in config_rows}):
        values = [
            float(row["rr_method"])
            for row in config_rows
            if int(row["seed"]) == seed
        ]
        seed_mrr.append(float(np.mean(values)))
    deltas = np.asarray([float(row["alpha_delta"]) for row in config_rows])
    result.update(meta)
    result.update(
        {
            "delta_vs_global": result["mrr"] - global_mrr,
            "ci95_low_vs_global": interval["ci95_low"],
            "ci95_high_vs_global": interval["ci95_high"],
            "delta_vs_query_soft": query_soft_interval["mean_delta"],
            "ci95_low_vs_query_soft": query_soft_interval["ci95_low"],
            "ci95_high_vs_query_soft": query_soft_interval["ci95_high"],
            "delta_vs_relation": relation_interval["mean_delta"],
            "ci95_low_vs_relation": relation_interval["ci95_low"],
            "ci95_high_vs_relation": relation_interval["ci95_high"],
            "seed_mrr_std": statistics.stdev(seed_mrr) if len(seed_mrr) > 1 else 0.0,
            "fallback_rate": float(np.mean([int(row["fallback"]) for row in config_rows])),
            "saturation_rate": float(np.mean([int(row["saturated"]) for row in config_rows])),
            "boundary_rate": float(
                np.mean([float(row["alpha_applied"]) in {0.0, 1.0} for row in config_rows])
            ),
            "changed_from_anchor_rate": float(np.mean(np.abs(deltas) > 1e-12)),
        }
    )
    return result


def write_markdown(path: Path, results: list[dict]) -> None:
    lines = [
        "| Configuration | Family | DEV MRR | Delta vs. Global | 95% CI vs. Global | Delta vs. Query-soft | Seed std | Fallback | Saturation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['label']} | {row['family']} | {row['mrr']:.6f} | "
            f"{row['delta_vs_global']:+.6f} | "
            f"[{row['ci95_low_vs_global']:+.6f}, {row['ci95_high_vs_global']:+.6f}] | "
            f"{row['delta_vs_query_soft']:+.6f} | "
            f"{row['seed_mrr_std']:.6f} | {row['fallback_rate']:.2%} | "
            f"{row['saturation_rate']:.2%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    betas = parse_grid(args.betas, name="beta", lower=0.0, upper=0.5)
    thresholds = parse_grid(
        args.confidence_thresholds,
        name="confidence threshold",
        lower=0.0,
        upper=1.0,
    )
    strengths = parse_grid(
        args.anchor_strengths,
        name="anchor strength",
        lower=0.0,
        upper=1.0,
    )
    if not 0.0 <= args.reference_beta <= 0.5:
        raise ValueError("--reference-beta must lie in [0, 0.5]")

    query_path = Path(args.query_rows)
    selection_path = Path(args.selection_json)
    out_dir = Path(args.output_dir)
    rows = read_csv(query_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not rows or {row["split"] for row in rows} != {"dev"}:
        raise RuntimeError("P3 ablation requires non-empty DEV query rows")
    if {row["pair_name"] for row in rows} != {selection["pair_name"]}:
        raise RuntimeError("Query rows and selection pair_name differ")

    alphas = tuple(float(value) for value in selection["alpha_grid"])
    required = {
        "pair_name",
        "dataset",
        "split",
        "query_key",
        "query_id",
        "head_id",
        "relation_id",
        "tail_id",
        "seed",
        "direction",
        "rr_a",
        "rr_b",
        "crossfit_fold",
        "alpha_global_crossfit",
        "rr_global_crossfit",
        "rr_relation_crossfit",
        *QUERY_GEOMETRY_FIELDS,
        *(alpha_column(alpha) for alpha in alphas),
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Query rows are missing required fields: {sorted(missing)}")

    seeds = sorted({int(row["seed"]) for row in rows})
    assignment, fold_audit = assign_grouped_folds(rows, args.folds, args.fold_seed)
    expected_per_group = 2 * len(seeds)
    group_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        group_counts[triple_key(row)] += 1
    malformed = {key: count for key, count in group_counts.items() if count != expected_per_group}
    if malformed:
        raise RuntimeError(
            f"Expected {expected_per_group} observations per original triple; "
            f"found {len(malformed)} malformed groups"
        )

    records: dict[str, list[dict]] = defaultdict(list)
    metadata: dict[str, dict] = {}
    selected_by_fold = []
    for fold in range(args.folds):
        train_rows = [row for row in rows if assignment[triple_key(row)] != fold]
        heldout_rows = [row for row in rows if assignment[triple_key(row)] == fold]
        alpha0, global_train_mrr = best_alpha(train_rows, alphas)
        if not all(
            int(row["crossfit_fold"]) == fold + 1
            and math.isclose(
                float(row["alpha_global_crossfit"]), alpha0, rel_tol=0.0, abs_tol=1e-12
            )
            for row in heldout_rows
        ):
            raise RuntimeError(
                "Input baseline cross-fit rows do not match the requested fold assignment"
            )

        append_records(
            records,
            metadata,
            identifier="global",
            label="Global alpha",
            family="baseline",
            feature_group="none",
            rows=heldout_rows,
            fold=fold + 1,
            alpha0=alpha0,
            policy=global_policy(heldout_rows, alpha0),
        )

        group_outputs = {}
        for group_index, (group_name, fields) in enumerate(FEATURE_GROUPS.items()):
            model, train_x, train_nonfinite = fit_geometry_model(
                train_rows,
                fields=tuple(fields),
                random_state=args.random_state + fold * 10 + group_index,
            )
            heldout_x, heldout_nonfinite = feature_matrix(heldout_rows, tuple(fields))
            train_decision, train_probability = model_outputs(model, train_x)
            heldout_decision, heldout_probability = model_outputs(model, heldout_x)
            group_outputs[group_name] = {
                "train_decision": train_decision,
                "train_probability": train_probability,
                "train_nonfinite": train_nonfinite,
                "heldout_decision": heldout_decision,
                "heldout_probability": heldout_probability,
                "heldout_nonfinite": heldout_nonfinite,
            }

        full = group_outputs["full_geometry"]
        query_soft = apply_query_soft(
            heldout_rows,
            probability_a=full["heldout_probability"],
            nonfinite=full["heldout_nonfinite"],
            alpha0=alpha0,
            alphas=alphas,
        )
        query_soft_policy = {
            **query_soft,
            "fallback": full["heldout_nonfinite"],
            "saturated": np.zeros(len(heldout_rows), dtype=bool),
        }
        append_records(
            records,
            metadata,
            identifier="query_soft_full",
            label="Query-soft (no anchor)",
            family="anchor",
            feature_group="full_geometry",
            rows=heldout_rows,
            fold=fold + 1,
            alpha0=alpha0,
            policy=query_soft_policy,
            strength=1.0,
        )

        for beta in betas:
            policy = apply_policy(
                heldout_rows,
                decision=full["heldout_decision"],
                probability_a=full["heldout_probability"],
                nonfinite=full["heldout_nonfinite"],
                alpha0=alpha0,
                beta=beta,
                confidence_threshold=0.0,
                alphas=alphas,
            )
            append_records(
                records,
                metadata,
                identifier=config_id("beta", beta),
                label=f"Anchored beta={beta:.2f}",
                family="beta_curve",
                feature_group="full_geometry",
                rows=heldout_rows,
                fold=fold + 1,
                alpha0=alpha0,
                policy=policy,
                beta=beta,
                threshold=0.0,
            )

        for strength in strengths:
            if math.isclose(strength, 0.0, abs_tol=1e-12) or math.isclose(
                strength, 1.0, abs_tol=1e-12
            ):
                continue
            policy = apply_probability_shrinkage(
                heldout_rows,
                probability_a=full["heldout_probability"],
                nonfinite=full["heldout_nonfinite"],
                alpha0=alpha0,
                strength=strength,
                alphas=alphas,
            )
            append_records(
                records,
                metadata,
                identifier=config_id("anchor_strength", strength),
                label=f"Probability shrinkage={strength:.2f}",
                family="anchor",
                feature_group="full_geometry",
                rows=heldout_rows,
                fold=fold + 1,
                alpha0=alpha0,
                policy=policy,
                strength=strength,
            )

        for group_name in ("single_confidence", "cross_expert_disagreement"):
            group = group_outputs[group_name]
            policy = apply_policy(
                heldout_rows,
                decision=group["heldout_decision"],
                probability_a=group["heldout_probability"],
                nonfinite=group["heldout_nonfinite"],
                alpha0=alpha0,
                beta=args.reference_beta,
                confidence_threshold=0.0,
                alphas=alphas,
            )
            append_records(
                records,
                metadata,
                identifier=f"features_{group_name}",
                label=f"Features: {group_name}",
                family="feature_ablation",
                feature_group=group_name,
                rows=heldout_rows,
                fold=fold + 1,
                alpha0=alpha0,
                policy=policy,
                beta=args.reference_beta,
                threshold=0.0,
            )

        for threshold in thresholds:
            if math.isclose(threshold, 0.0, abs_tol=1e-12):
                continue
            policy = apply_policy(
                heldout_rows,
                decision=full["heldout_decision"],
                probability_a=full["heldout_probability"],
                nonfinite=full["heldout_nonfinite"],
                alpha0=alpha0,
                beta=args.reference_beta,
                confidence_threshold=threshold,
                alphas=alphas,
            )
            append_records(
                records,
                metadata,
                identifier=config_id("fallback", threshold),
                label=f"Fallback threshold={threshold:.2f}",
                family="fallback_curve",
                feature_group="full_geometry",
                rows=heldout_rows,
                fold=fold + 1,
                alpha0=alpha0,
                policy=policy,
                beta=args.reference_beta,
                threshold=threshold,
            )

        selected_beta, selected_threshold, selected_train_mrr = select_bounded_policy(
            train_rows,
            decision=full["train_decision"],
            probability_a=full["train_probability"],
            nonfinite=full["train_nonfinite"],
            alpha0=alpha0,
            betas=betas,
            confidence_thresholds=thresholds,
            alphas=alphas,
        )
        selected = apply_policy(
            heldout_rows,
            decision=full["heldout_decision"],
            probability_a=full["heldout_probability"],
            nonfinite=full["heldout_nonfinite"],
            alpha0=alpha0,
            beta=selected_beta,
            confidence_threshold=selected_threshold,
            alphas=alphas,
        )
        append_records(
            records,
            metadata,
            identifier="expanded_selected",
            label="Expanded anchored (nested selection)",
            family="primary_candidate",
            feature_group="full_geometry",
            rows=heldout_rows,
            fold=fold + 1,
            alpha0=alpha0,
            policy=selected,
            beta="nested",
            threshold="nested",
        )
        selected_by_fold.append(
            {
                "fold": fold + 1,
                "train_triple_groups": len({triple_key(row) for row in train_rows}),
                "heldout_triple_groups": len({triple_key(row) for row in heldout_rows}),
                "alpha0": alpha0,
                "selected_beta": selected_beta,
                "selected_confidence_threshold": selected_threshold,
                "global_train_mrr": global_train_mrr,
                "selected_train_mrr": selected_train_mrr,
                "global_heldout_mrr": float(
                    np.mean([float(row["rr_global_crossfit"]) for row in heldout_rows])
                ),
                "selected_heldout_mrr": float(selected["rr"].mean()),
                "fallback_rate": float(selected["fallback"].mean()),
                "saturation_rate": float(selected["saturated"].mean()),
            }
        )

    query_soft_lookup = {
        observation_key(row): float(row["rr_method"])
        for row in records["query_soft_full"]
    }
    for config_rows in records.values():
        for row in config_rows:
            key = observation_key(row)
            if key not in query_soft_lookup:
                raise RuntimeError(f"Missing Query-soft reference row for {key}")
            row["rr_query_soft"] = query_soft_lookup[key]

    results = [summarize_config(records[key], metadata[key]) for key in metadata]
    order = {
        "baseline": 0,
        "primary_candidate": 1,
        "beta_curve": 2,
        "anchor": 3,
        "feature_ablation": 4,
        "fallback_curve": 5,
    }
    results.sort(
        key=lambda row: (
            order[row["family"]],
            str(row["beta"]),
            str(row["anchor_strength"]),
            row["config_id"],
        )
    )

    by_seed = []
    by_direction = []
    by_fold = []
    for result in results:
        config_rows = records[result["config_id"]]
        for seed in seeds:
            subset = [row for row in config_rows if int(row["seed"]) == seed]
            by_seed.append(
                {
                    "config_id": result["config_id"],
                    "label": result["label"],
                    "seed": seed,
                    **metric([float(row["rr_method"]) for row in subset]),
                    "delta_vs_global": float(
                        np.mean(
                            [
                                float(row["rr_method"]) - float(row["rr_global_crossfit"])
                                for row in subset
                            ]
                        )
                    ),
                }
            )
        for direction in ("head", "tail"):
            subset = [row for row in config_rows if row["direction"] == direction]
            by_direction.append(
                {
                    "config_id": result["config_id"],
                    "label": result["label"],
                    "direction": direction,
                    **metric([float(row["rr_method"]) for row in subset]),
                    "delta_vs_global": float(
                        np.mean(
                            [
                                float(row["rr_method"]) - float(row["rr_global_crossfit"])
                                for row in subset
                            ]
                        )
                    ),
                }
            )
        for fold in range(1, args.folds + 1):
            subset = [row for row in config_rows if int(row["fold"]) == fold]
            by_fold.append(
                {
                    "config_id": result["config_id"],
                    "label": result["label"],
                    "fold": fold,
                    **metric([float(row["rr_method"]) for row in subset]),
                    "delta_vs_global": float(
                        np.mean(
                            [
                                float(row["rr_method"]) - float(row["rr_global_crossfit"])
                                for row in subset
                            ]
                        )
                    ),
                }
            )

    selected_rows = sorted(
        records["expanded_selected"],
        key=lambda row: (
            int(row["fold"]),
            int(row["seed"]),
            row["direction"],
            int(row["relation_id"]),
            int(row["head_id"]),
            int(row["tail_id"]),
        ),
    )
    selected_deltas = np.asarray([float(row["alpha_delta"]) for row in selected_rows])
    selected_diagnostics = {
        "alpha_delta_from_anchor": quantiles(selected_deltas),
        "fallback_rate": float(np.mean([int(row["fallback"]) for row in selected_rows])),
        "saturation_rate": float(np.mean([int(row["saturated"]) for row in selected_rows])),
        "boundary_rate": float(
            np.mean([float(row["alpha_applied"]) in {0.0, 1.0} for row in selected_rows])
        ),
        "changed_from_anchor_rate": float(np.mean(np.abs(selected_deltas) > 1e-12)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "dev_p3_results.csv", results)
    write_csv(out_dir / "dev_p3_results_by_seed.csv", by_seed)
    write_csv(out_dir / "dev_p3_results_by_direction.csv", by_direction)
    write_csv(out_dir / "dev_p3_results_by_fold.csv", by_fold)
    write_csv(out_dir / "dev_p3_selected_by_fold.csv", selected_by_fold)
    write_csv(out_dir / "dev_p3_selected_query_rows.csv", selected_rows)
    write_markdown(out_dir / "dev_p3_results.md", results)
    summary = {
        "schema_version": 1,
        "pair_name": selection["pair_name"],
        "dataset": selection["dataset"],
        "source_query_rows": str(query_path),
        "source_selection": str(selection_path),
        "seeds": seeds,
        "beta_grid": list(betas),
        "confidence_threshold_grid": list(thresholds),
        "anchor_strength_grid": list(strengths),
        "reference_beta": args.reference_beta,
        "feature_groups": {name: list(fields) for name, fields in FEATURE_GROUPS.items()},
        "fold_audit": fold_audit,
        "leakage_guard": "all seeds and both directions of one original triple share one fold",
        "selection_rule": (
            "expanded anchored beta and fallback threshold are selected only on each outer "
            "training fold; fixed curves are diagnostics evaluated on held-out folds"
        ),
        "results": results,
        "selected_by_fold": selected_by_fold,
        "selected_diagnostics": selected_diagnostics,
        "interpretation_boundary": (
            "All fitting, nested policy selection, and ablations use DEV grouped cross-fitting. "
            "No TEST outcomes, target ids, target/reference scores, ranks, reciprocal ranks, "
            "or raw relation ids are policy inputs."
        ),
    }
    (out_dir / "dev_p3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {out_dir / 'dev_p3_results.md'}")


if __name__ == "__main__":
    main()
