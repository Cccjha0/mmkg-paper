from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.query_geometry import QUERY_GEOMETRY_FIELDS
from scripts.ablate_anchored_dynamic import fit_geometry_model, model_outputs
from scripts.crossfit_anchored_dynamic import (
    apply_policy,
    apply_query_soft,
    parse_grid,
    read_csv,
    select_bounded_policy,
)
from scripts.crossfit_heterogeneous_dev_policies import alpha_column, metric, write_csv


DEFAULT_BETAS = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50"
DEFAULT_THRESHOLDS = "0.00,0.10,0.20,0.30"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the final anchored dynamic combiner on all DEV observations, lock it, "
            "or apply that immutable lock to exact full-ranking TEST rows."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock = subparsers.add_parser("lock", help="Fit and select the final policy on DEV only")
    lock.add_argument("--dev-query-rows", required=True)
    lock.add_argument("--selection-json", required=True)
    lock.add_argument(
        "--crossfit-summary",
        required=True,
        help="Completed P3 DEV cross-fit summary that fixed the policy family and grids",
    )
    lock.add_argument("--output-dir", required=True)
    lock.add_argument("--betas", default=DEFAULT_BETAS)
    lock.add_argument("--confidence-thresholds", default=DEFAULT_THRESHOLDS)
    lock.add_argument("--random-state", type=int, default=20260902)

    apply = subparsers.add_parser("apply", help="Apply a locked DEV policy to TEST rows")
    apply.add_argument("--test-query-rows", required=True)
    apply.add_argument("--lock-json", required=True)
    apply.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def required_columns(alphas: tuple[float, ...]) -> set[str]:
    return {
        "pair_name",
        "dataset",
        "protocol_version",
        "expert_a_name",
        "expert_b_name",
        "split",
        "seed",
        "rr_a",
        "rr_b",
        "rr_global",
        "rr_oracle",
        *QUERY_GEOMETRY_FIELDS,
        *(alpha_column(alpha) for alpha in alphas),
    }


def validate_rows(
    rows: list[dict[str, str]],
    *,
    split: str,
    metadata: dict,
    alphas: tuple[float, ...],
) -> None:
    if not rows:
        raise RuntimeError(f"{split.upper()} query rows are empty")
    missing = required_columns(alphas) - set(rows[0])
    if missing:
        raise RuntimeError(f"Query rows are missing required fields: {sorted(missing)}")
    expected = {
        "pair_name": str(metadata["pair_name"]),
        "dataset": str(metadata["dataset"]),
        "protocol_version": str(metadata["protocol_version"]),
        "expert_a_name": str(metadata["expert_a_name"]),
        "expert_b_name": str(metadata["expert_b_name"]),
        "split": split,
    }
    for field, value in expected.items():
        observed = {row[field] for row in rows}
        if observed != {value}:
            raise RuntimeError(f"{field} mismatch: expected {value!r}, observed {sorted(observed)!r}")
    observed_seeds = sorted({int(row["seed"]) for row in rows})
    expected_seeds = sorted(int(value) for value in metadata["seeds"])
    if observed_seeds != expected_seeds:
        raise RuntimeError(
            f"Seed mismatch: expected {expected_seeds}, observed {observed_seeds}"
        )
    for row in rows:
        for alpha in alphas:
            value = float(row[alpha_column(alpha)])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError("Alpha-grid reciprocal ranks must be finite and positive")


def add_policy_columns(
    rows: list[dict[str, str]],
    *,
    decision: np.ndarray,
    probability: np.ndarray,
    anchored: dict[str, np.ndarray],
    query_soft: dict[str, np.ndarray],
    alpha0: float,
    beta: float,
    threshold: float,
) -> list[dict]:
    output = []
    for index, source in enumerate(rows):
        row = dict(source)
        row.update(
            {
                "alpha0_locked": alpha0,
                "anchored_beta_locked": beta,
                "anchored_confidence_threshold_locked": threshold,
                "anchored_decision": float(decision[index]),
                "anchored_probability_a": float(probability[index]),
                "anchored_confidence": float(anchored["confidence"][index]),
                "anchored_fallback": int(anchored["fallback"][index]),
                "anchored_saturated": int(anchored["saturated"][index]),
                "alpha_anchored_continuous": float(anchored["continuous"][index]),
                "alpha_anchored_locked": float(anchored["applied"][index]),
                "rr_anchored_locked": float(anchored["rr"][index]),
                "alpha_query_soft_continuous": float(query_soft["continuous"][index]),
                "alpha_query_soft_locked": float(query_soft["applied"][index]),
                "rr_query_soft_locked": float(query_soft["rr"][index]),
            }
        )
        output.append(row)
    return output


def summarize(rows: list[dict], expert_a: str, expert_b: str) -> list[dict]:
    methods = [
        (expert_a, "rr_a", "fixed expert"),
        (expert_b, "rr_b", "fixed expert"),
        ("Global alpha", "rr_global", "DEV-locked static anchor"),
    ]
    if "rr_relation" in rows[0]:
        methods.append(("Relation alpha", "rr_relation", "DEV-locked diagnostic"))
    methods.extend(
        [
            ("Query-soft logistic", "rr_query_soft_locked", "DEV-locked no-anchor ablation"),
            ("Anchored dynamic", "rr_anchored_locked", "DEV-locked primary method"),
            ("Oracle", "rr_oracle", "answer-aware upper bound"),
        ]
    )
    global_mrr = metric([float(row["rr_global"]) for row in rows])["mrr"]
    output = []
    for method, column, notes in methods:
        result = metric([float(row[column]) for row in rows])
        result.update(
            {
                "method": method,
                "delta_vs_global": result["mrr"] - global_mrr,
                "notes": notes,
            }
        )
        output.append(result)
    return output


def summarize_by_seed(rows: list[dict], expert_a: str, expert_b: str) -> list[dict]:
    methods = [
        (expert_a, "rr_a"),
        (expert_b, "rr_b"),
        ("Global alpha", "rr_global"),
        ("Query-soft logistic", "rr_query_soft_locked"),
        ("Anchored dynamic", "rr_anchored_locked"),
        ("Oracle", "rr_oracle"),
    ]
    output = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        global_mrr = metric([float(row["rr_global"]) for row in seed_rows])["mrr"]
        for method, column in methods:
            result = metric([float(row[column]) for row in seed_rows])
            output.append(
                {
                    "seed": seed,
                    "method": method,
                    **result,
                    "delta_vs_global": result["mrr"] - global_mrr,
                }
            )
    return output


def write_markdown(path: Path, rows: list[dict], split: str) -> None:
    lines = [
        f"| Method | {split.upper()} MRR | Hits@1 | Hits@3 | Hits@10 | Delta vs. Global |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['mrr']:.6f} | {row['hits@1']:.6f} | "
            f"{row['hits@3']:.6f} | {row['hits@10']:.6f} | "
            f"{row['delta_vs_global']:+.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnostics(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "n_rows": count,
        "fallback_rate": sum(int(row["anchored_fallback"]) for row in rows) / count,
        "saturation_rate": sum(int(row["anchored_saturated"]) for row in rows) / count,
        "changed_from_anchor_rate": sum(
            not math.isclose(
                float(row["alpha_anchored_locked"]),
                float(row["alpha0_locked"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in rows
        )
        / count,
    }


def write_outputs(
    out_dir: Path,
    *,
    split: str,
    rows: list[dict],
    metadata: dict,
    lock: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = summarize(rows, metadata["expert_a_name"], metadata["expert_b_name"])
    by_seed = summarize_by_seed(rows, metadata["expert_a_name"], metadata["expert_b_name"])
    write_csv(out_dir / f"{split}_locked_query_rows.csv", rows)
    write_csv(out_dir / f"{split}_locked_results.csv", results)
    write_csv(out_dir / f"{split}_locked_results_by_seed.csv", by_seed)
    write_markdown(out_dir / f"{split}_locked_results.md", results, split)
    summary = {
        "schema_version": 1,
        "pair_name": metadata["pair_name"],
        "dataset": metadata["dataset"],
        "protocol_version": metadata["protocol_version"],
        "split": split,
        "expert_a_name": metadata["expert_a_name"],
        "expert_b_name": metadata["expert_b_name"],
        "seeds": metadata["seeds"],
        "policy": lock,
        "results": results,
        "results_by_seed": by_seed,
        "diagnostics": diagnostics(rows),
        "interpretation_boundary": (
            "The combiner model, alpha anchor, beta, and confidence threshold were "
            "selected using DEV only. TEST outcomes are used only for final reporting."
        ),
    }
    (out_dir / f"{split}_locked_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def lock_policy(args: argparse.Namespace) -> None:
    query_path = Path(args.dev_query_rows).resolve()
    selection_path = Path(args.selection_json).resolve()
    crossfit_path = Path(args.crossfit_summary).resolve()
    out_dir = Path(args.output_dir)
    selection = read_json(selection_path)
    crossfit = read_json(crossfit_path)
    rows = read_csv(query_path)
    alphas = tuple(float(value) for value in selection["alpha_grid"])
    validate_rows(rows, split="dev", metadata=selection, alphas=alphas)

    betas = parse_grid(args.betas, name="beta", lower=0.0, upper=0.5)
    thresholds = parse_grid(
        args.confidence_thresholds,
        name="confidence threshold",
        lower=0.0,
        upper=1.0,
    )
    for field in (
        "pair_name",
        "dataset",
        "seeds",
    ):
        if crossfit.get(field) != selection.get(field):
            raise RuntimeError(f"P3 cross-fit summary and DEV selection differ on {field}")
    if tuple(float(value) for value in crossfit["beta_grid"]) != betas:
        raise RuntimeError("--betas must exactly match the completed P3 cross-fit grid")
    if tuple(float(value) for value in crossfit["confidence_threshold_grid"]) != thresholds:
        raise RuntimeError(
            "--confidence-thresholds must exactly match the completed P3 cross-fit grid"
        )
    result_ids = {str(row["config_id"]) for row in crossfit.get("results", [])}
    if "expanded_selected" not in result_ids:
        raise RuntimeError("P3 summary does not contain the expanded nested-selection result")
    if tuple(crossfit.get("feature_groups", {}).get("full_geometry", ())) != tuple(
        QUERY_GEOMETRY_FIELDS
    ):
        raise RuntimeError("P3 full-geometry feature schema does not match this implementation")
    alpha0 = float(selection["global_alpha"])
    if alpha0 not in alphas:
        raise RuntimeError("Locked global alpha is absent from the exact-ranking alpha grid")

    model, matrix, nonfinite = fit_geometry_model(
        rows, fields=QUERY_GEOMETRY_FIELDS, random_state=args.random_state
    )
    decision, probability = model_outputs(model, matrix)
    beta, threshold, selected_dev_mrr = select_bounded_policy(
        rows,
        decision=decision,
        probability_a=probability,
        nonfinite=nonfinite,
        alpha0=alpha0,
        betas=betas,
        confidence_thresholds=thresholds,
        alphas=alphas,
    )
    anchored = apply_policy(
        rows,
        decision=decision,
        probability_a=probability,
        nonfinite=nonfinite,
        alpha0=alpha0,
        beta=beta,
        confidence_threshold=threshold,
        alphas=alphas,
    )
    query_soft = apply_query_soft(
        rows,
        probability_a=probability,
        nonfinite=nonfinite,
        alpha0=alpha0,
        alphas=alphas,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "anchored_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    lock = {
        "schema_version": 1,
        "pair_name": selection["pair_name"],
        "dataset": selection["dataset"],
        "protocol_version": selection["protocol_version"],
        "expert_a_name": selection["expert_a_name"],
        "expert_b_name": selection["expert_b_name"],
        "seeds": [int(value) for value in selection["seeds"]],
        "formula": "clip(alpha0 + beta * tanh(g(phi(q))), 0, 1)",
        "model": "median imputation + standard scaling + balanced logistic regression",
        "label": "expert A has strictly larger reciprocal rank; ties excluded from fitting",
        "alpha_application": "rounded to nearest exact-ranking alpha grid value",
        "alpha_grid": list(alphas),
        "alpha0": alpha0,
        "beta": beta,
        "confidence_threshold": threshold,
        "beta_grid": list(betas),
        "confidence_threshold_grid": list(thresholds),
        "selected_full_dev_mrr": selected_dev_mrr,
        "random_state": args.random_state,
        "query_geometry_fields": list(QUERY_GEOMETRY_FIELDS),
        "source_dev_query_rows": str(query_path),
        "source_dev_query_rows_sha256": sha256(query_path),
        "source_selection_json": str(selection_path),
        "source_selection_json_sha256": sha256(selection_path),
        "source_crossfit_summary": str(crossfit_path),
        "source_crossfit_summary_sha256": sha256(crossfit_path),
        "model_file": model_path.name,
        "model_sha256": sha256(model_path),
        "selection_boundary": "all fitting and hyperparameter selection use DEV only",
    }
    lock_path = out_dir / "anchored_dev_lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    locked_rows = add_policy_columns(
        rows,
        decision=decision,
        probability=probability,
        anchored=anchored,
        query_soft=query_soft,
        alpha0=alpha0,
        beta=beta,
        threshold=threshold,
    )
    write_outputs(out_dir, split="dev", rows=locked_rows, metadata=selection, lock=lock)
    print(
        f"[LOCKED] alpha0={alpha0:.2f} beta={beta:.2f} threshold={threshold:.2f} "
        f"model={model_path} lock={lock_path}",
        flush=True,
    )


def apply_locked_policy(args: argparse.Namespace) -> None:
    query_path = Path(args.test_query_rows).resolve()
    lock_path = Path(args.lock_json).resolve()
    out_dir = Path(args.output_dir)
    lock = read_json(lock_path)
    rows = read_csv(query_path)
    alphas = tuple(float(value) for value in lock["alpha_grid"])
    validate_rows(rows, split="test", metadata=lock, alphas=alphas)
    if tuple(lock["query_geometry_fields"]) != tuple(QUERY_GEOMETRY_FIELDS):
        raise RuntimeError("Locked geometry schema does not match the current implementation")

    model_path = lock_path.parent / lock["model_file"]
    if not model_path.exists() or sha256(model_path) != lock["model_sha256"]:
        raise RuntimeError("Locked model is missing or its SHA256 does not match")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    matrix = np.asarray(
        [[float(row[field]) for field in QUERY_GEOMETRY_FIELDS] for row in rows],
        dtype=np.float64,
    )
    nonfinite = ~np.isfinite(matrix).all(axis=1)
    matrix[~np.isfinite(matrix)] = np.nan
    decision, probability = model_outputs(model, matrix)
    alpha0 = float(lock["alpha0"])
    beta = float(lock["beta"])
    threshold = float(lock["confidence_threshold"])
    anchored = apply_policy(
        rows,
        decision=decision,
        probability_a=probability,
        nonfinite=nonfinite,
        alpha0=alpha0,
        beta=beta,
        confidence_threshold=threshold,
        alphas=alphas,
    )
    query_soft = apply_query_soft(
        rows,
        probability_a=probability,
        nonfinite=nonfinite,
        alpha0=alpha0,
        alphas=alphas,
    )
    locked_rows = add_policy_columns(
        rows,
        decision=decision,
        probability=probability,
        anchored=anchored,
        query_soft=query_soft,
        alpha0=alpha0,
        beta=beta,
        threshold=threshold,
    )
    write_outputs(out_dir, split="test", rows=locked_rows, metadata=lock, lock=lock)
    print(f"[OK] wrote {out_dir / 'test_locked_results.md'}", flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "lock":
        lock_policy(args)
    else:
        apply_locked_policy(args)


if __name__ == "__main__":
    main()
