from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.query_geometry import QUERY_GEOMETRY_FIELDS
from scripts.crossfit_heterogeneous_dev_policies import (
    alpha_column,
    assign_grouped_folds,
    best_alpha,
    clustered_interval,
    metric,
    triple_key,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a bounded dynamic score combiner with grouped DEV cross-fitting. "
            "The policy is alpha(q)=clip(alpha0+beta*tanh(g(phi(q))),0,1)."
        )
    )
    parser.add_argument("--query-rows", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260901)
    parser.add_argument("--random-state", type=int, default=20260902)
    parser.add_argument("--betas", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--confidence-thresholds", default="0.00,0.10,0.20")
    return parser.parse_args()


def parse_grid(raw: str, *, name: str, lower: float, upper: float) -> tuple[float, ...]:
    values = tuple(sorted({float(part.strip()) for part in raw.split(",") if part.strip()}))
    if not values:
        raise ValueError(f"{name} grid is empty")
    if any(not lower <= value <= upper for value in values):
        raise ValueError(f"{name} values must lie in [{lower}, {upper}]")
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [[float(row[name]) for name in QUERY_GEOMETRY_FIELDS] for row in rows],
        dtype=np.float64,
    )
    nonfinite = ~np.isfinite(matrix).all(axis=1)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, nonfinite


def nearest_alpha(value: float, alphas: tuple[float, ...], anchor: float) -> float:
    return min(alphas, key=lambda alpha: (abs(alpha - value), abs(alpha - anchor), alpha))


def apply_policy(
    rows: list[dict],
    *,
    decision: np.ndarray,
    probability_a: np.ndarray,
    nonfinite: np.ndarray,
    alpha0: float,
    beta: float,
    confidence_threshold: float,
    alphas: tuple[float, ...],
) -> dict[str, np.ndarray]:
    confidence = np.abs(2.0 * probability_a - 1.0)
    fallback = nonfinite | (confidence < float(confidence_threshold))
    unconstrained = float(alpha0) + float(beta) * np.tanh(decision)
    continuous = np.clip(unconstrained, 0.0, 1.0)
    continuous = np.where(fallback, float(alpha0), continuous)
    applied = np.asarray(
        [nearest_alpha(float(value), alphas, float(alpha0)) for value in continuous],
        dtype=np.float64,
    )
    rr = np.asarray(
        [float(row[alpha_column(float(alpha))]) for row, alpha in zip(rows, applied)],
        dtype=np.float64,
    )
    return {
        "confidence": confidence,
        "fallback": fallback,
        "unconstrained": unconstrained,
        "continuous": continuous,
        "applied": applied,
        "rr": rr,
        "saturated": (unconstrained <= 0.0) | (unconstrained >= 1.0),
    }


def apply_query_soft(
    rows: list[dict],
    *,
    probability_a: np.ndarray,
    nonfinite: np.ndarray,
    alpha0: float,
    alphas: tuple[float, ...],
) -> dict[str, np.ndarray]:
    continuous = np.where(nonfinite, float(alpha0), np.clip(probability_a, 0.0, 1.0))
    applied = np.asarray(
        [nearest_alpha(float(value), alphas, float(alpha0)) for value in continuous],
        dtype=np.float64,
    )
    rr = np.asarray(
        [float(row[alpha_column(float(alpha))]) for row, alpha in zip(rows, applied)],
        dtype=np.float64,
    )
    return {"continuous": continuous, "applied": applied, "rr": rr}


def select_bounded_policy(
    rows: list[dict],
    *,
    decision: np.ndarray,
    probability_a: np.ndarray,
    nonfinite: np.ndarray,
    alpha0: float,
    betas: tuple[float, ...],
    confidence_thresholds: tuple[float, ...],
    alphas: tuple[float, ...],
) -> tuple[float, float, float]:
    candidates = []
    for beta in betas:
        for threshold in confidence_thresholds:
            policy = apply_policy(
                rows,
                decision=decision,
                probability_a=probability_a,
                nonfinite=nonfinite,
                alpha0=alpha0,
                beta=beta,
                confidence_threshold=threshold,
                alphas=alphas,
            )
            mean_mrr = float(policy["rr"].mean())
            # Conservative deterministic tie-breaking: prefer a smaller
            # correction range, then a larger fallback region.
            candidates.append((mean_mrr, -beta, threshold, beta, threshold))
    winner = max(candidates)
    return float(winner[3]), float(winner[4]), float(winner[0])


def summarize_methods(rows: list[dict], expert_a: str, expert_b: str) -> list[dict]:
    methods = (
        (expert_a, "rr_a", "fixed expert"),
        (expert_b, "rr_b", "fixed expert"),
        ("Query-zscore 0.5", "rr_equal", "fixed"),
        ("Global alpha (5-fold cross-fit)", "rr_global_crossfit", "held-out triples"),
        ("Query-soft logistic (5-fold cross-fit)", "rr_query_soft_crossfit", "free dynamic alpha"),
        ("Anchored dynamic (5-fold cross-fit)", "rr_anchored_crossfit", "bounded correction"),
        ("Relation alpha (5-fold cross-fit)", "rr_relation_crossfit", "secondary diagnostic"),
        ("Oracle", "rr_oracle", "answer-aware upper bound"),
    )
    anchor = metric([float(row["rr_a"]) for row in rows])["mrr"]
    global_mrr = metric([float(row["rr_global_crossfit"]) for row in rows])["mrr"]
    output = []
    for method, column, notes in methods:
        result = metric([float(row[column]) for row in rows])
        result.update(
            {
                "method": method,
                "delta_vs_a": result["mrr"] - anchor,
                "delta_vs_global": result["mrr"] - global_mrr,
                "notes": notes,
            }
        )
        output.append(result)
    return output


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| Method | DEV MRR | Hits@1 | Hits@3 | Hits@10 | Delta vs. A | Delta vs. Global |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['mrr']:.6f} | {row['hits@1']:.6f} | "
            f"{row['hits@3']:.6f} | {row['hits@10']:.6f} | "
            f"{row['delta_vs_a']:+.6f} | {row['delta_vs_global']:+.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


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
    query_path = Path(args.query_rows)
    selection_path = Path(args.selection_json)
    out_dir = Path(args.output_dir)
    rows = read_csv(query_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not rows or {row["split"] for row in rows} != {"dev"}:
        raise RuntimeError("Anchored cross-fitting requires non-empty DEV query rows")
    if {row["pair_name"] for row in rows} != {selection["pair_name"]}:
        raise RuntimeError("Query rows and selection pair_name differ")

    alphas = tuple(float(value) for value in selection["alpha_grid"])
    required = {
        "query_key",
        "head_id",
        "relation_id",
        "tail_id",
        "seed",
        "direction",
        "rr_a",
        "rr_b",
        "rr_equal",
        "crossfit_fold",
        "alpha_global_crossfit",
        "rr_global_crossfit",
        "rr_relation_crossfit",
        "rr_oracle",
        *QUERY_GEOMETRY_FIELDS,
        *(alpha_column(alpha) for alpha in alphas),
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Query rows are missing required fields: {sorted(missing)}")

    seeds = sorted({int(row["seed"]) for row in rows})
    assignment, fold_audit = assign_grouped_folds(rows, args.folds, args.fold_seed)
    expected_per_group = 2 * len(seeds)
    counts: dict[str, int] = {}
    for row in rows:
        key = triple_key(row)
        counts[key] = counts.get(key, 0) + 1
    malformed = {key: count for key, count in counts.items() if count != expected_per_group}
    if malformed:
        raise RuntimeError(
            f"Expected {expected_per_group} observations per original triple; "
            f"found {len(malformed)} malformed groups"
        )

    output_rows = []
    fold_results = []
    for fold in range(args.folds):
        train_rows = [row for row in rows if assignment[triple_key(row)] != fold]
        heldout_rows = [row for row in rows if assignment[triple_key(row)] == fold]
        alpha0, global_train_mrr = best_alpha(train_rows, alphas)
        if not all(
            int(row["crossfit_fold"]) == fold + 1
            and math.isclose(
                float(row["alpha_global_crossfit"]),
                alpha0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(row["rr_global_crossfit"]),
                float(row[alpha_column(alpha0)]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in heldout_rows
        ):
            raise RuntimeError(
                "Input baseline cross-fit rows do not match the requested fold assignment"
            )

        train_x, train_nonfinite = feature_matrix(train_rows)
        heldout_x, heldout_nonfinite = feature_matrix(heldout_rows)
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
            raise RuntimeError(f"Fold {fold + 1} does not contain both expert-winner classes")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=args.random_state + fold,
                solver="liblinear",
            ),
        )
        model.fit(train_x[fit_mask], train_y[fit_mask])
        train_decision = np.asarray(model.decision_function(train_x), dtype=np.float64)
        train_probability = np.asarray(model.predict_proba(train_x)[:, 1], dtype=np.float64)
        heldout_decision = np.asarray(model.decision_function(heldout_x), dtype=np.float64)
        heldout_probability = np.asarray(model.predict_proba(heldout_x)[:, 1], dtype=np.float64)

        beta, threshold, anchored_train_mrr = select_bounded_policy(
            train_rows,
            decision=train_decision,
            probability_a=train_probability,
            nonfinite=train_nonfinite,
            alpha0=alpha0,
            betas=betas,
            confidence_thresholds=thresholds,
            alphas=alphas,
        )
        anchored = apply_policy(
            heldout_rows,
            decision=heldout_decision,
            probability_a=heldout_probability,
            nonfinite=heldout_nonfinite,
            alpha0=alpha0,
            beta=beta,
            confidence_threshold=threshold,
            alphas=alphas,
        )
        query_soft = apply_query_soft(
            heldout_rows,
            probability_a=heldout_probability,
            nonfinite=heldout_nonfinite,
            alpha0=alpha0,
            alphas=alphas,
        )

        for index, source in enumerate(heldout_rows):
            row = dict(source)
            row.update(
                {
                    "anchored_fold": fold + 1,
                    "alpha0_crossfit": alpha0,
                    "anchored_beta": beta,
                    "anchored_confidence_threshold": threshold,
                    "anchored_decision": float(heldout_decision[index]),
                    "anchored_probability_a": float(heldout_probability[index]),
                    "anchored_confidence": float(anchored["confidence"][index]),
                    "anchored_fallback": int(anchored["fallback"][index]),
                    "anchored_saturated": int(anchored["saturated"][index]),
                    "alpha_anchored_continuous": float(anchored["continuous"][index]),
                    "alpha_anchored_crossfit": float(anchored["applied"][index]),
                    "alpha_delta_from_anchor": float(anchored["applied"][index] - alpha0),
                    "rr_anchored_crossfit": float(anchored["rr"][index]),
                    "alpha_query_soft_continuous": float(query_soft["continuous"][index]),
                    "alpha_query_soft_crossfit": float(query_soft["applied"][index]),
                    "rr_query_soft_crossfit": float(query_soft["rr"][index]),
                }
            )
            output_rows.append(row)

        fold_results.append(
            {
                "fold": fold + 1,
                "train_triple_groups": len({triple_key(row) for row in train_rows}),
                "heldout_triple_groups": len({triple_key(row) for row in heldout_rows}),
                "alpha0": alpha0,
                "beta": beta,
                "confidence_threshold": threshold,
                "global_train_mrr": global_train_mrr,
                "anchored_train_mrr": anchored_train_mrr,
                "expert_a_mrr": metric([float(row["rr_a"]) for row in heldout_rows])["mrr"],
                "global_mrr": metric([float(row["rr_global_crossfit"]) for row in heldout_rows])["mrr"],
                "query_soft_mrr": float(query_soft["rr"].mean()),
                "anchored_mrr": float(anchored["rr"].mean()),
                "relation_mrr": metric(
                    [float(row["rr_relation_crossfit"]) for row in heldout_rows]
                )["mrr"],
                "fallback_rate": float(anchored["fallback"].mean()),
                "saturation_rate": float(anchored["saturated"].mean()),
            }
        )

    output_rows.sort(
        key=lambda row: (
            int(row["anchored_fold"]),
            int(row["seed"]),
            str(row["direction"]),
            int(row["relation_id"]),
            int(row["head_id"]),
            int(row["tail_id"]),
        )
    )
    expert_a = str(selection["expert_a_name"])
    expert_b = str(selection["expert_b_name"])
    results = summarize_methods(output_rows, expert_a, expert_b)
    by_seed = []
    for seed in seeds:
        seed_rows = [row for row in output_rows if int(row["seed"]) == seed]
        for method, column in (
            (expert_a, "rr_a"),
            ("Global alpha (5-fold cross-fit)", "rr_global_crossfit"),
            ("Query-soft logistic (5-fold cross-fit)", "rr_query_soft_crossfit"),
            ("Anchored dynamic (5-fold cross-fit)", "rr_anchored_crossfit"),
            ("Relation alpha (5-fold cross-fit)", "rr_relation_crossfit"),
            ("Oracle", "rr_oracle"),
        ):
            by_seed.append(
                {"seed": seed, "method": method, **metric([float(row[column]) for row in seed_rows])}
            )

    delta = np.asarray([float(row["alpha_delta_from_anchor"]) for row in output_rows])
    diagnostics = {
        "alpha_delta_from_anchor": quantiles(delta),
        "fallback_rate": sum(int(row["anchored_fallback"]) for row in output_rows) / len(output_rows),
        "saturation_rate": sum(int(row["anchored_saturated"]) for row in output_rows) / len(output_rows),
        "applied_boundary_rate": sum(
            float(row["alpha_anchored_crossfit"]) in {0.0, 1.0} for row in output_rows
        )
        / len(output_rows),
        "changed_from_anchor_rate": sum(
            not math.isclose(
                float(row["alpha_anchored_crossfit"]),
                float(row["alpha0_crossfit"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in output_rows
        )
        / len(output_rows),
    }
    intervals = {
        "query_soft_vs_global": clustered_interval(
            output_rows, "rr_query_soft_crossfit", reference="rr_global_crossfit"
        ),
        "anchored_vs_a": clustered_interval(output_rows, "rr_anchored_crossfit"),
        "anchored_vs_global": clustered_interval(
            output_rows, "rr_anchored_crossfit", reference="rr_global_crossfit"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "dev_anchored_query_rows.csv", output_rows)
    write_csv(out_dir / "dev_anchored_results.csv", results)
    write_csv(out_dir / "dev_anchored_results_by_seed.csv", by_seed)
    write_csv(out_dir / "dev_anchored_results_by_fold.csv", fold_results)
    write_markdown(out_dir / "dev_anchored_results.md", results)
    summary = {
        "schema_version": 1,
        "pair_name": selection["pair_name"],
        "dataset": selection["dataset"],
        "expert_a_name": expert_a,
        "expert_b_name": expert_b,
        "source_query_rows": str(query_path),
        "source_selection": str(selection_path),
        "seeds": seeds,
        "formula": "clip(alpha0 + beta * tanh(g(phi(q))), 0, 1)",
        "model": "median imputation + standard scaling + balanced logistic regression",
        "label": "expert A has strictly larger reciprocal rank; ties excluded from fitting",
        "alpha_application": "continuous policy output rounded to nearest exact-ranking alpha grid value",
        "query_geometry_fields": list(QUERY_GEOMETRY_FIELDS),
        "forbidden_inputs": [
            "raw relation id",
            "target entity id",
            "target modality availability",
            "target/reference score",
            "rank",
            "reciprocal rank",
        ],
        "beta_grid": list(betas),
        "confidence_threshold_grid": list(thresholds),
        "fold_audit": fold_audit,
        "leakage_guard": "all seeds and both directions of one original triple share one fold",
        "results": results,
        "results_by_fold": fold_results,
        "clustered_intervals": intervals,
        "diagnostics": diagnostics,
        "interpretation_boundary": (
            "All model fitting and policy selection are cross-fitted on DEV. "
            "No TEST evaluation outcomes or answer-aware query features are used; "
            "candidate masking retains the protocol's standard filtered-fact index."
        ),
    }
    (out_dir / "dev_anchored_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {out_dir / 'dev_anchored_results.md'}")


if __name__ == "__main__":
    main()
