from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TextIO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.constants import QUERY_GEOMETRY_FIELDS
from scripts.crossfit_heterogeneous_dev_policies import alpha_column, best_alpha, triple_key


SCHEMA_VERSION = 1
LOCAL_OFFSETS = (Decimal("0.05"), Decimal("0.10"), Decimal("0.20"), Decimal("0.30"))
ZERO_TOLERANCE = 1e-15
MRR_TOLERANCE = 1e-12
RR_RECIPROCAL_TOLERANCE = 1e-12
OUTPUT_FIELDS = (
    "dataset",
    "protocol_version",
    "expert_a",
    "expert_b",
    "pair_id",
    "split",
    "original_triple_id",
    "query_key",
    "query_id",
    "seed",
    "direction",
    "head",
    "relation",
    "tail",
    "target_entity_id",
    "alpha0",
    "alpha",
    "delta_alpha",
    "abs_delta_alpha",
    "rr_anchor",
    "rr_action",
    "advantage",
    "rank_anchor",
    "rank_action",
    *QUERY_GEOMETRY_FIELDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an AACPI V2 DEV query-by-local-action advantage table from the "
            "existing exact full-ranking alpha-grid export. This command rejects TEST rows."
        )
    )
    parser.add_argument("--query-rows", required=True, help="Existing DEV full-ranking query rows CSV.")
    parser.add_argument("--selection-json", required=True, help="DEV Global-alpha selection JSON.")
    parser.add_argument("--output-dir", default="outputs/aacpi/utility_tables")
    parser.add_argument(
        "--output-format",
        choices=("csv", "csv.gz"),
        default="csv.gz",
        help="Machine-readable table format. csv.gz is a gzip-compressed CSV.",
    )
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--validation-seed", type=int, default=20260904)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def clean_prefix(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    if not prefix:
        raise ValueError("pair_id cannot be converted to a safe output prefix")
    return prefix


def local_action_grid(alpha0: float) -> tuple[float, ...]:
    anchor = Decimal(str(alpha0))
    candidates = {anchor}
    for offset in LOCAL_OFFSETS:
        candidates.add(max(Decimal("0"), min(Decimal("1"), anchor - offset)))
        candidates.add(max(Decimal("0"), min(Decimal("1"), anchor + offset)))
    return tuple(float(value) for value in sorted(candidates))


def rank_from_rr(rr: float, *, context: str) -> int:
    if not math.isfinite(rr) or rr <= 0.0 or rr > 1.0:
        raise ValueError(f"Invalid reciprocal rank at {context}: {rr!r}")
    rank = int(round(1.0 / rr))
    if rank < 1 or not math.isclose(rr, 1.0 / rank, rel_tol=0.0, abs_tol=RR_RECIPROCAL_TOLERANCE):
        raise ValueError(f"RR is not the reciprocal of an integer rank at {context}: {rr!r}")
    return rank


def classify_advantage(value: float) -> str:
    if value > ZERO_TOLERANCE:
        return "positive"
    if value < -ZERO_TOLERANCE:
        return "negative"
    return "zero"


def format_alpha(value: float, *, signed: bool = False) -> str:
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    ordered = sorted(values)

    def percentile(probability: float) -> float:
        index = probability * (len(ordered) - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return float(ordered[lower])
        weight = index - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "min": float(ordered[0]),
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": float(ordered[-1]),
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
    }


def open_output_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def validate_source_contract(rows: list[dict[str, str]], selection: dict) -> dict:
    if not rows:
        raise RuntimeError("Refusing to build an AACPI table from empty query rows")
    required_selection = {
        "pair_name",
        "dataset",
        "protocol_version",
        "expert_a_name",
        "expert_b_name",
        "seeds",
        "score_normalization",
        "alpha_grid",
        "global_alpha",
        "global_dev_mrr",
    }
    missing_selection = sorted(required_selection - set(selection))
    if missing_selection:
        raise ValueError(f"Selection JSON is missing fields: {missing_selection}")
    if selection["score_normalization"] != "query_zscore":
        raise ValueError(
            "AACPI V2 phase 1 requires the frozen query_zscore normalization; "
            f"found {selection['score_normalization']!r}"
        )

    alphas = tuple(float(value) for value in selection["alpha_grid"])
    required_row_fields = {
        "pair_name",
        "dataset",
        "protocol_version",
        "expert_a_name",
        "expert_b_name",
        "query_key",
        "query_id",
        "split",
        "seed",
        "direction",
        "relation_id",
        "head_id",
        "tail_id",
        "target_entity_id",
        "rr_global",
        "rr_oracle",
        *QUERY_GEOMETRY_FIELDS,
        *(alpha_column(alpha) for alpha in alphas),
    }
    missing_rows = sorted(required_row_fields - set(rows[0]))
    if missing_rows:
        raise ValueError(f"Query rows are missing fields: {missing_rows}")

    expected_metadata = {
        "pair_name": str(selection["pair_name"]),
        "dataset": str(selection["dataset"]),
        "protocol_version": str(selection["protocol_version"]),
        "expert_a_name": str(selection["expert_a_name"]),
        "expert_b_name": str(selection["expert_b_name"]),
    }
    query_ids: set[str] = set()
    group_coverage: dict[str, set[tuple[int, str]]] = defaultdict(set)
    expected_seeds = {int(value) for value in selection["seeds"]}
    expected_observations = {(seed, direction) for seed in expected_seeds for direction in ("head", "tail")}

    for index, row in enumerate(rows, start=2):
        if row.get("split") != "dev":
            raise RuntimeError(
                "AACPI utility construction is DEV-only; "
                f"row {index} has split={row.get('split')!r}"
            )
        for field, expected in expected_metadata.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"Source/selection metadata mismatch at row {index}: "
                    f"{field}={row.get(field)!r}, expected {expected!r}"
                )
        query_id = str(row["query_id"]).strip()
        if not query_id:
            raise ValueError(f"Missing query_id at source row {index}")
        if query_id in query_ids:
            raise ValueError(f"Duplicate query_id in source rows: {query_id}")
        query_ids.add(query_id)
        group = triple_key(row)
        if not group:
            raise ValueError(f"Missing original-triple key at source row {index}")
        direction = str(row["direction"])
        seed = int(row["seed"])
        if direction not in {"head", "tail"}:
            raise ValueError(f"Invalid direction at source row {index}: {direction!r}")
        if seed not in expected_seeds:
            raise ValueError(f"Unexpected seed at source row {index}: {seed}")
        group_coverage[group].add((seed, direction))
        for feature in QUERY_GEOMETRY_FIELDS:
            if not math.isfinite(float(row[feature])):
                raise ValueError(f"Non-finite geometry feature {feature} at source row {index}")

    incomplete = {
        group: sorted(expected_observations - observations)
        for group, observations in group_coverage.items()
        if observations != expected_observations
    }
    if incomplete:
        sample_group = next(iter(incomplete))
        raise ValueError(
            "Original-triple seed/direction coverage is incomplete; "
            f"example {sample_group}: missing={incomplete[sample_group]}"
        )

    recomputed_alpha, recomputed_mrr = best_alpha(rows, alphas)
    alpha0 = float(selection["global_alpha"])
    locked_mrr = float(selection["global_dev_mrr"])
    if not math.isclose(recomputed_alpha, alpha0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            "Locked Global alpha does not reproduce under the existing exact-ranking "
            f"selection rule: locked={alpha0}, recomputed={recomputed_alpha}"
        )
    if not math.isclose(recomputed_mrr, locked_mrr, rel_tol=0.0, abs_tol=MRR_TOLERANCE):
        raise RuntimeError(
            "Locked Global DEV MRR does not reproduce: "
            f"locked={locked_mrr:.15g}, recomputed={recomputed_mrr:.15g}"
        )

    anchor_column = alpha_column(alpha0)
    source_anchor_mrr = sum(float(row[anchor_column]) for row in rows) / len(rows)
    source_global_mrr = sum(float(row["rr_global"]) for row in rows) / len(rows)
    if not math.isclose(source_anchor_mrr, source_global_mrr, rel_tol=0.0, abs_tol=MRR_TOLERANCE):
        raise RuntimeError("rr_global does not match the exact alpha0 column in source rows")
    if not math.isclose(source_anchor_mrr, locked_mrr, rel_tol=0.0, abs_tol=MRR_TOLERANCE):
        raise RuntimeError("Source alpha0 MRR does not match the locked Global DEV MRR")

    return {
        "alphas": alphas,
        "alpha0": alpha0,
        "n_query_instances": len(rows),
        "n_original_triples": len(group_coverage),
        "expected_group_observations": len(expected_observations),
        "recomputed_global_alpha": recomputed_alpha,
        "recomputed_global_dev_mrr": recomputed_mrr,
        "source_anchor_mrr": source_anchor_mrr,
    }


def build_utility_table(
    query_path: Path,
    selection_path: Path,
    output_dir: Path,
    *,
    output_format: str = "csv.gz",
    validation_samples: int = 64,
    validation_seed: int = 20260904,
    overwrite: bool = False,
) -> dict[str, Path | dict]:
    if validation_samples < 0:
        raise ValueError("validation_samples must be non-negative")
    query_path = query_path.resolve()
    selection_path = selection_path.resolve()
    rows = read_csv(query_path)
    selection = read_json(selection_path)
    source_audit = validate_source_contract(rows, selection)
    alpha0 = float(source_audit["alpha0"])
    actions = local_action_grid(alpha0)
    global_grid = tuple(float(value) for value in source_audit["alphas"])
    missing_actions = [alpha for alpha in actions if alpha not in global_grid]
    if missing_actions:
        raise RuntimeError(
            "The frozen local action grid is not present in the existing exact-ranking export: "
            f"{missing_actions}. Regenerating or interpolating ranking outcomes is outside phase 1."
        )
    if alpha0 not in actions:
        raise AssertionError("alpha0 is missing from the local action grid")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in actions):
        raise AssertionError("Local action grid contains alpha outside [0, 1]")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = clean_prefix(str(selection["pair_name"]))
    extension = ".csv.gz" if output_format == "csv.gz" else ".csv"
    table_path = output_dir / f"{prefix}_dev_utility_table{extension}"
    summary_path = output_dir / f"{prefix}_dev_utility_summary.json"
    manifest_path = output_dir / f"{prefix}_dev_source_manifest.json"
    for path in (table_path, summary_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output without --overwrite: {path}")

    sign_counts: Counter[str] = Counter()
    nonreference_sign_counts: Counter[str] = Counter()
    best_delta_counts: Counter[str] = Counter()
    best_alpha_counts: Counter[str] = Counter()
    per_delta: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "positive": 0, "zero": 0, "negative": 0, "sum_advantage": 0.0}
    )
    all_advantages: list[float] = []
    nonreference_advantages: list[float] = []
    distinct_rr_ratios: list[float] = []
    best_advantages: list[float] = []
    best_rr_values: list[float] = []
    anchor_rr_values: list[float] = []
    endpoint_oracle_values: list[float] = []
    positive_opportunity_queries = 0
    anchor_consistency_max_error = 0.0
    reference_rows_checked = 0

    rng = random.Random(validation_seed)
    sample_count = min(validation_samples, len(rows))
    sampled_indices = set(rng.sample(range(len(rows)), sample_count)) if sample_count else set()
    sampled_rank_checks = 0

    with open_output_csv(table_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row_index, source in enumerate(rows):
            query_id = str(source["query_id"])
            original_triple_id = triple_key(source)
            if not original_triple_id:
                raise AssertionError(f"original_triple_id is missing for {query_id}")
            rr_anchor = float(source[alpha_column(alpha0)])
            rank_anchor = rank_from_rr(rr_anchor, context=f"{query_id} alpha0")
            if not math.isclose(rr_anchor, float(source["rr_global"]), rel_tol=0.0, abs_tol=MRR_TOLERANCE):
                raise AssertionError(f"alpha0 RR differs from rr_global for {query_id}")
            anchor_rr_values.append(rr_anchor)
            endpoint_oracle_values.append(float(source["rr_oracle"]))

            query_actions: list[tuple[float, float, float, int]] = []
            query_anchor_values: list[float] = []
            for alpha in actions:
                rr_action = float(source[alpha_column(alpha)])
                rank_action = rank_from_rr(rr_action, context=f"{query_id} alpha={alpha:.2f}")
                advantage = rr_action - rr_anchor
                if not math.isfinite(advantage):
                    raise AssertionError(f"Non-finite advantage for {query_id} alpha={alpha:.2f}")
                delta = round(alpha - alpha0, 10)
                sign = classify_advantage(advantage)
                sign_counts[sign] += 1
                all_advantages.append(advantage)
                delta_key = format_alpha(delta, signed=True)
                delta_stats = per_delta[delta_key]
                delta_stats["count"] = int(delta_stats["count"]) + 1
                delta_stats[sign] = int(delta_stats[sign]) + 1
                delta_stats["sum_advantage"] = float(delta_stats["sum_advantage"]) + advantage
                if not math.isclose(alpha, alpha0, rel_tol=0.0, abs_tol=1e-12):
                    nonreference_sign_counts[sign] += 1
                    nonreference_advantages.append(advantage)
                else:
                    reference_rows_checked += 1
                    if sign != "zero" or advantage != 0.0 or rr_action != rr_anchor or rank_action != rank_anchor:
                        raise AssertionError(f"Invalid alpha0 reference row for {query_id}")
                query_anchor_values.append(rr_anchor)
                query_actions.append((alpha, delta, rr_action, rank_action))
                output = {
                    "dataset": selection["dataset"],
                    "protocol_version": selection["protocol_version"],
                    "expert_a": selection["expert_a_name"],
                    "expert_b": selection["expert_b_name"],
                    "pair_id": selection["pair_name"],
                    "split": "dev",
                    "original_triple_id": original_triple_id,
                    "query_key": source["query_key"],
                    "query_id": query_id,
                    "seed": int(source["seed"]),
                    "direction": source["direction"],
                    "head": int(source["head_id"]),
                    "relation": int(source["relation_id"]),
                    "tail": int(source["tail_id"]),
                    "target_entity_id": int(source["target_entity_id"]),
                    "alpha0": alpha0,
                    "alpha": alpha,
                    "delta_alpha": delta,
                    "abs_delta_alpha": abs(delta),
                    "rr_anchor": rr_anchor,
                    "rr_action": rr_action,
                    "advantage": advantage,
                    "rank_anchor": rank_anchor,
                    "rank_action": rank_action,
                }
                output.update({field: float(source[field]) for field in QUERY_GEOMETRY_FIELDS})
                writer.writerow(output)

                if row_index in sampled_indices:
                    source_rr = float(source[alpha_column(alpha)])
                    if rr_action != source_rr or rank_action != rank_from_rr(
                        source_rr, context=f"sample {query_id} alpha={alpha:.2f}"
                    ):
                        raise AssertionError("Utility rank differs from the source exact-ranking export")
                    if alpha == 0.0 and "rank_b" in source and rank_action != int(source["rank_b"]):
                        raise AssertionError("Alpha=0 rank differs from source expert-B endpoint")
                    if alpha == 1.0 and "rank_a" in source and rank_action != int(source["rank_a"]):
                        raise AssertionError("Alpha=1 rank differs from source expert-A endpoint")
                    sampled_rank_checks += 1

            anchor_consistency_max_error = max(
                anchor_consistency_max_error,
                max(abs(value - rr_anchor) for value in query_anchor_values),
            )
            distinct_rr_ratios.append(len({item[2] for item in query_actions}) / len(actions))
            max_rr = max(item[2] for item in query_actions)
            best_candidates = [item for item in query_actions if item[2] == max_rr]
            best = min(best_candidates, key=lambda item: (abs(item[1]), item[0]))
            best_advantage = best[2] - rr_anchor
            best_advantages.append(best_advantage)
            best_rr_values.append(best[2])
            best_delta_counts[format_alpha(best[1], signed=True)] += 1
            best_alpha_counts[format_alpha(best[0])] += 1
            if best_advantage > ZERO_TOLERANCE:
                positive_opportunity_queries += 1

    expected_rows = len(rows) * len(actions)
    observed_rows = sum(sign_counts.values())
    if observed_rows != expected_rows:
        raise AssertionError(f"Unexpected utility-row count: {observed_rows} != {expected_rows}")
    if reference_rows_checked != len(rows):
        raise AssertionError("Each query must have exactly one checked alpha0 reference row")
    if anchor_consistency_max_error > 0.0:
        raise AssertionError("rr_anchor changed across action rows for a query")

    anchor_mrr = sum(anchor_rr_values) / len(anchor_rr_values)
    local_oracle_mrr = sum(best_rr_values) / len(best_rr_values)
    endpoint_oracle_mrr = sum(endpoint_oracle_values) / len(endpoint_oracle_values)
    local_headroom = local_oracle_mrr - anchor_mrr
    endpoint_gap = endpoint_oracle_mrr - anchor_mrr
    per_delta_summary = {}
    for delta, values in sorted(per_delta.items(), key=lambda item: float(item[0])):
        count = int(values["count"])
        per_delta_summary[delta] = {
            "count": count,
            "positive_rate": int(values["positive"]) / count,
            "zero_rate": int(values["zero"]) / count,
            "negative_rate": int(values["negative"]) / count,
            "mean_advantage": float(values["sum_advantage"]) / count,
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": "AACPI V2 phase-1 utility surface",
        "evidence_role": "DEV-only descriptive supervision; not policy selection",
        "dataset": selection["dataset"],
        "protocol_version": selection["protocol_version"],
        "pair_id": selection["pair_name"],
        "expert_a": selection["expert_a_name"],
        "expert_b": selection["expert_b_name"],
        "split": "dev",
        "score_normalization": selection["score_normalization"],
        "alpha0": alpha0,
        "action_grid": list(actions),
        "n_original_triples": int(source_audit["n_original_triples"]),
        "n_query_instances": len(rows),
        "n_query_action_rows": observed_rows,
        "n_actions_per_query": len(actions),
        "positive_action_rate": sign_counts["positive"] / observed_rows,
        "zero_action_rate": sign_counts["zero"] / observed_rows,
        "negative_action_rate": sign_counts["negative"] / observed_rows,
        "nonreference_positive_action_rate": (
            nonreference_sign_counts["positive"] / sum(nonreference_sign_counts.values())
        ),
        "nonreference_zero_action_rate": (
            nonreference_sign_counts["zero"] / sum(nonreference_sign_counts.values())
        ),
        "nonreference_negative_action_rate": (
            nonreference_sign_counts["negative"] / sum(nonreference_sign_counts.values())
        ),
        "positive_opportunity_query_rate": positive_opportunity_queries / len(rows),
        "best_delta_alpha_distribution": {
            key: {"count": count, "rate": count / len(rows)}
            for key, count in sorted(best_delta_counts.items(), key=lambda item: float(item[0]))
        },
        "best_alpha_distribution": {
            key: {"count": count, "rate": count / len(rows)}
            for key, count in sorted(best_alpha_counts.items(), key=lambda item: float(item[0]))
        },
        "advantage_statistics": quantiles(all_advantages),
        "nonreference_advantage_statistics": quantiles(nonreference_advantages),
        "best_advantage_statistics": quantiles(best_advantages),
        "per_delta_alpha_statistics": per_delta_summary,
        "utility_sparsity": {
            "zero_action_rate": sign_counts["zero"] / observed_rows,
            "nonreference_zero_action_rate": (
                nonreference_sign_counts["zero"] / sum(nonreference_sign_counts.values())
            ),
            "no_positive_action_query_rate": 1.0 - positive_opportunity_queries / len(rows),
            "mean_distinct_rr_fraction_of_action_grid": sum(distinct_rr_ratios) / len(distinct_rr_ratios),
        },
        "local_action_potential": {
            "global_anchor_mrr": anchor_mrr,
            "oracle_local_mrr": local_oracle_mrr,
            "oracle_local_headroom": local_headroom,
            "endpoint_oracle_mrr": endpoint_oracle_mrr,
            "local_headroom_as_fraction_of_endpoint_oracle_gap": (
                local_headroom / endpoint_gap if endpoint_gap > 0.0 else None
            ),
            "note": "Descriptive answer-aware upper surface over the frozen local actions; not a deployable policy.",
        },
        "best_action_tie_break": "maximum RR, then minimum abs(delta_alpha), then smaller alpha",
        "validation": {
            "dev_only": True,
            "global_alpha_rule_reproduced": True,
            "global_mrr_sanity_check_passed": True,
            "alpha_bounds_passed": True,
            "reference_advantage_zero_passed": True,
            "anchor_consistency_max_abs_error": anchor_consistency_max_error,
            "finite_advantage_passed": True,
            "original_triple_ids_complete": True,
            "query_action_count_passed": True,
            "group_coverage_passed": True,
            "expected_seed_direction_observations_per_group": int(
                source_audit["expected_group_observations"]
            ),
            "sampled_source_exact_rank_queries": sample_count,
            "sampled_source_exact_rank_action_checks": sampled_rank_checks,
            "sample_check_note": (
                "Ranks were checked against the existing evaluator's stored exact RR columns; "
                "the evaluator and expert checkpoints were not rerun."
            ),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    source_summary_path = query_path.with_name("dev_summary.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": selection["dataset"],
        "pair_id": selection["pair_name"],
        "split": "dev",
        "evidence_role": "AACPI DEV-only utility supervision",
        "test_policy": "TEST inputs rejected; MKG-W/DB15K TEST is retrospective/secondary only",
        "test_exposure_boundary": {
            "status": "retrospective",
            "evidence_role": "secondary",
            "eligible_for_aacpi_method_selection": False,
        },
        "source_query_rows": {
            "path": portable_path(query_path),
            "sha256": sha256_file(query_path),
        },
        "source_selection": {
            "path": portable_path(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "source_full_ranking_summary": (
            {
                "path": portable_path(source_summary_path),
                "sha256": sha256_file(source_summary_path),
            }
            if source_summary_path.exists()
            else None
        ),
        "output_table": {
            "path": portable_path(table_path),
            "format": output_format,
            "sha256": sha256_file(table_path),
        },
        "source_exact_ranking_implementation": "scripts/eval_heterogeneous_complementarity.py",
        "filtered_ranking_protocol": {
            "protocol_version": selection["protocol_version"],
            "filtering": "all-split true-fact filtering; target retained",
            "rank_ties": "strictly greater candidate scores plus one",
            "directions": ["head", "tail"],
        },
        "reused_global_selection_and_group_key": "scripts/crossfit_heterogeneous_dev_policies.py",
        "feature_contract": "router/constants.py::QUERY_GEOMETRY_FIELDS",
        "score_normalization": selection["score_normalization"],
        "global_alpha_grid": list(global_grid),
        "global_alpha": alpha0,
        "local_action_rule": "clip(alpha0 +/- {0.05,0.10,0.20,0.30}, [0,1]); include alpha0; deduplicate; sort",
        "local_action_grid": list(actions),
        "group_key": "h=<head_id>|r=<relation_id>|t=<tail_id>",
        "fold_assignment_performed": False,
        "output_fields": list(OUTPUT_FIELDS),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "table_path": table_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    result = build_utility_table(
        Path(args.query_rows),
        Path(args.selection_json),
        Path(args.output_dir),
        output_format=args.output_format,
        validation_samples=args.validation_samples,
        validation_seed=args.validation_seed,
        overwrite=args.overwrite,
    )
    summary = result["summary"]
    print(f"[OK] wrote {result['table_path']}")
    print(f"[OK] wrote {result['summary_path']}")
    print(f"[OK] wrote {result['manifest_path']}")
    print(
        "[DEV] "
        f"dataset={summary['dataset']} pair={summary['pair_id']} alpha0={summary['alpha0']:.2f} "
        f"queries={summary['n_query_instances']} rows={summary['n_query_action_rows']} "
        f"positive_opportunity={summary['positive_opportunity_query_rate']:.6f} "
        f"oracle_local_headroom={summary['local_action_potential']['oracle_local_headroom']:.6f}"
    )


if __name__ == "__main__":
    main()
