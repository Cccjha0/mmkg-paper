from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

from router.io_utils import read_csv, write_csv, write_json
from router.routing_utils import compute_eval_summary, compute_gain_precision, select_expert_row


QUERY_ROUTING_HEADER = [
    "policy_name",
    "config_id",
    "query_id",
    "split",
    "seed",
    "direction",
    "target_regime",
    "relation_id",
    "selected_by",
    "router_prob",
    "threshold",
    "use_fusion",
    "selected_expert",
    "rank_final",
    "rr_final",
    "rank_gate",
    "rr_gate",
    "rank_residual",
    "rr_residual",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def _safe_int(value: object, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def load_prediction_rows(path: str | Path) -> list[dict]:
    rows = read_csv(path)
    payload = []
    for row in rows:
        payload.append(
            {
                "query_id": str(row["query_id"]),
                "split": str(row.get("split", "")),
                "seed": _safe_int(row.get("seed"), default=0),
                "direction": str(row.get("direction", "")),
                "relation_id": _safe_int(row.get("relation_id"), default=0),
                "router_prob": _safe_float(row.get("prob_fusion", row.get("router_prob")), default=0.0),
                "target_regime": str(row.get("target_regime", "")),
                "rr_gate": _safe_float(row.get("rr_gate"), default=0.0),
                "rr_residual": _safe_float(row.get("rr_residual"), default=0.0),
            }
        )
    return payload


def merge_clean_feature_columns(rows: list[dict], feature_csv: str | Path, columns: list[str]) -> list[dict]:
    if not columns:
        return [dict(row) for row in rows]
    feature_rows = {str(row["query_id"]): row for row in read_csv(feature_csv)}
    merged_rows = []
    for row in rows:
        query_id = str(row["query_id"])
        if query_id not in feature_rows:
            raise RuntimeError(f"Missing clean feature row for query_id={query_id}")
        merged = dict(row)
        feature_row = feature_rows[query_id]
        for column in columns:
            if column not in feature_row:
                raise RuntimeError(f"Missing feature column {column} in {feature_csv}")
            merged[column] = feature_row[column]
        merged_rows.append(merged)
    return merged_rows


def merge_relation_prior_columns(rows: list[dict], prior_csv: str | Path) -> list[dict]:
    prior_rows = {}
    for row in read_csv(prior_csv):
        relation_id = str(_safe_int(str(row["relation_id"]).replace("rel_", "").replace("relation_", ""), default=-1))
        prior_rows[relation_id] = row

    merged_rows = []
    for row in rows:
        relation_key = str(_safe_int(row["relation_id"], default=-1))
        if relation_key not in prior_rows:
            raise RuntimeError(f"Missing relation prior row for relation_id={relation_key}")
        prior_row = prior_rows[relation_key]
        merged = dict(row)
        merged["relation_gain_prior"] = _safe_float(
            prior_row.get("mean_delta_rr_shrunk", prior_row.get("mean_delta_rr", 0.0)),
            default=0.0,
        )
        merged["relation_support"] = _safe_int(prior_row.get("n_queries"), default=0)
        merged["relation_is_visual_prior"] = _safe_int(prior_row.get("is_visual_prior"), default=0)
        merged_rows.append(merged)
    return merged_rows


def _build_routed_row(
    row: dict,
    use_fusion: int,
    policy_name: str,
    config_id: str,
    threshold: float,
    selected_by: str,
) -> dict:
    routed = select_expert_row(
        {
            "query_id": row["query_id"],
            "split": row["split"],
            "seed": row["seed"],
            "direction": row["direction"],
            "relation_id": row["relation_id"],
            "router_prob": row["router_prob"],
            "threshold": threshold,
            "target_regime": row["target_regime"],
        },
        {
            "target_regime": row["target_regime"],
            "rank_gate": row.get("rank_gate", 0),
            "rr_gate": row["rr_gate"],
            "rank_residual": row.get("rank_residual", 0),
            "rr_residual": row["rr_residual"],
        },
        use_fusion=use_fusion,
        selected_by=selected_by,
    )
    routed["policy_name"] = policy_name
    routed["config_id"] = config_id
    return routed


def summarize_policy_rows(
    routed_rows: list[dict],
    delta: float,
    extra: dict | None = None,
    include_regime_metrics: bool = True,
) -> dict:
    summary = compute_eval_summary(routed_rows)
    overall = summary["overall"]
    row = {
        "n_queries": overall["n_queries"],
        "overall_mrr": overall["mrr"],
        "hits1": overall["hits1"],
        "hits3": overall["hits3"],
        "hits10": overall["hits10"],
        "fusion_coverage": overall["fusion_coverage"],
        "gain_precision": compute_gain_precision(routed_rows, delta=delta),
    }
    if include_regime_metrics:
        for regime in ["head_has_img", "head_no_img", "tail_no_img"]:
            stats = summary["by_regime"].get(regime, {})
            row[f"{regime}_mrr"] = float(stats.get("mrr", 0.0))
            row[f"{regime}_coverage"] = float(stats.get("fusion_coverage", 0.0))
            row[f"{regime}_n_queries"] = int(stats.get("n_queries", 0))
    if extra:
        row.update(extra)
    return row


def write_query_rows(path: str | Path, rows: list[dict]) -> None:
    write_csv(path, rows, QUERY_ROUTING_HEADER)


def materialize_policy_rows(
    rows: list[dict],
    decision_fn,
    policy_name: str,
    config_id: str,
    default_threshold: float = 0.0,
) -> list[dict]:
    routed_rows = []
    for row in rows:
        decision = decision_fn(row)
        if isinstance(decision, tuple):
            use_fusion, threshold, selected_by = decision
        else:
            use_fusion, threshold, selected_by = decision, default_threshold, policy_name
        routed_rows.append(
            _build_routed_row(
                row=row,
                use_fusion=int(use_fusion),
                policy_name=policy_name,
                config_id=config_id,
                threshold=float(threshold),
                selected_by=str(selected_by),
            )
        )
    return routed_rows


def load_score_map(path: str | Path, source: str) -> dict[tuple[int, str], float]:
    rows = read_csv(path)
    payload = {}
    for row in rows:
        seed = _safe_int(row.get("seed"), default=0)
        query_id = str(row["query_id"])
        if source == "final":
            value = _safe_float(row.get("rr_final"), default=0.0)
        elif source == "residual":
            value = _safe_float(row.get("rr_residual"), default=0.0)
        elif source == "gate":
            value = _safe_float(row.get("rr_gate"), default=0.0)
        else:
            raise ValueError(f"Unsupported score source: {source}")
        payload[(seed, query_id)] = value
    return payload


def paired_seed_deltas(
    left_scores: dict[tuple[int, str], float],
    right_scores: dict[tuple[int, str], float],
) -> tuple[list[float], list[float], list[dict]]:
    shared = sorted(set(left_scores) & set(right_scores))
    if not shared:
        raise RuntimeError("No shared (seed, query_id) pairs found between the two inputs.")

    per_query = []
    by_seed = defaultdict(list)
    for seed, query_id in shared:
        delta = float(left_scores[(seed, query_id)] - right_scores[(seed, query_id)])
        by_seed[int(seed)].append(delta)
        per_query.append({"seed": int(seed), "query_id": query_id, "delta_rr": delta})

    seed_means = [sum(bucket) / len(bucket) for _, bucket in sorted(by_seed.items())]
    all_deltas = [row["delta_rr"] for row in per_query]
    return seed_means, all_deltas, per_query


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    variance = float(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, math.sqrt(variance)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def bootstrap_ci(values: list[float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(n_bootstrap):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(float(sum(draw) / len(draw)))
    return percentile(samples, 0.025), percentile(samples, 0.975)


def write_significance_payload(
    out_json: str | Path,
    out_csv: str | Path,
    comparison: str,
    left_label: str,
    right_label: str,
    seed_means: list[float],
    per_query_deltas: list[float],
    bootstrap_bounds: tuple[float, float],
) -> None:
    seed_mean, seed_std = mean_std(seed_means)
    query_mean, _ = mean_std(per_query_deltas)
    payload = {
        "comparison": comparison,
        "left_label": left_label,
        "right_label": right_label,
        "n_seeds": len(seed_means),
        "n_paired_queries": len(per_query_deltas),
        "mean_delta_mrr_seedwise": seed_mean,
        "std_delta_mrr_seedwise": seed_std,
        "mean_delta_mrr_querywise": query_mean,
        "bootstrap_ci_95_querywise": {
            "low": bootstrap_bounds[0],
            "high": bootstrap_bounds[1],
        },
        "seed_level_deltas": seed_means,
    }
    write_json(out_json, payload)
    write_csv(
        out_csv,
        [
            {
                "comparison": comparison,
                "left_label": left_label,
                "right_label": right_label,
                "n_seeds": len(seed_means),
                "n_paired_queries": len(per_query_deltas),
                "mean_delta_mrr_seedwise": seed_mean,
                "std_delta_mrr_seedwise": seed_std,
                "mean_delta_mrr_querywise": query_mean,
                "bootstrap_ci_low": bootstrap_bounds[0],
                "bootstrap_ci_high": bootstrap_bounds[1],
            }
        ],
        [
            "comparison",
            "left_label",
            "right_label",
            "n_seeds",
            "n_paired_queries",
            "mean_delta_mrr_seedwise",
            "std_delta_mrr_seedwise",
            "mean_delta_mrr_querywise",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ],
    )
