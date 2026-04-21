from __future__ import annotations

from collections import Counter, defaultdict


def hard_route(prob: float, tau: float) -> int:
    return int(float(prob) >= float(tau))


def _coerce_eval_meta_row(route_row: dict, eval_meta_row: dict | None) -> dict:
    if eval_meta_row is not None:
        return eval_meta_row
    return {
        "target_regime": route_row.get("target_regime", ""),
        "rank_gate": route_row.get("rank_fusion", 0),
        "rr_gate": route_row.get("rr_fusion", 0.0),
        "rank_residual": route_row.get("rank_struct", 0),
        "rr_residual": route_row.get("rr_struct", 0.0),
    }


def select_expert_row(
    route_row: dict,
    eval_meta_row: dict | None,
    use_fusion: int,
    selected_by: str | None = None,
) -> dict:
    use_fusion = int(use_fusion)
    eval_meta_row = _coerce_eval_meta_row(route_row, eval_meta_row)
    if use_fusion:
        selected_expert = "gate_only"
        rank_final = int(eval_meta_row["rank_gate"])
        rr_final = float(eval_meta_row["rr_gate"])
    else:
        selected_expert = "residual_only"
        rank_final = int(eval_meta_row["rank_residual"])
        rr_final = float(eval_meta_row["rr_residual"])

    return {
        "query_id": route_row["query_id"],
        "split": route_row["split"],
        "seed": int(route_row["seed"]),
        "direction": route_row["direction"],
        "target_regime": eval_meta_row["target_regime"],
        "relation_id": int(route_row["relation_id"]),
        "selected_by": selected_by or "",
        "router_prob": float(route_row["router_prob"]),
        "threshold": float(route_row["threshold"]),
        "use_fusion": use_fusion,
        "selected_expert": selected_expert,
        "rank_final": rank_final,
        "rr_final": rr_final,
        "rank_gate": int(eval_meta_row["rank_gate"]),
        "rr_gate": float(eval_meta_row["rr_gate"]),
        "rank_residual": int(eval_meta_row["rank_residual"]),
        "rr_residual": float(eval_meta_row["rr_residual"]),
    }


def compute_coverage(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return float(sum(int(row["use_fusion"]) for row in rows) / len(rows))


def compute_gain_precision(rows: list[dict], delta: float = 0.0) -> float:
    """Evaluation-only diagnostic metric for selected fusion routes."""
    selected = [row for row in rows if int(row["use_fusion"]) == 1]
    if not selected:
        return 0.0
    positive = sum(float(row["rr_gate"]) - float(row["rr_residual"]) > float(delta) for row in selected)
    return float(positive / len(selected))


def _metric_bundle(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "n_queries": 0,
            "mrr": 0.0,
            "hits1": 0.0,
            "hits3": 0.0,
            "hits10": 0.0,
            "fusion_coverage": 0.0,
        }
    hits1 = sum(int(row["rank_final"]) <= 1 for row in rows) / n
    hits3 = sum(int(row["rank_final"]) <= 3 for row in rows) / n
    hits10 = sum(int(row["rank_final"]) <= 10 for row in rows) / n
    mrr = sum(float(row["rr_final"]) for row in rows) / n
    return {
        "n_queries": n,
        "mrr": float(mrr),
        "hits1": float(hits1),
        "hits3": float(hits3),
        "hits10": float(hits10),
        "fusion_coverage": compute_coverage(rows),
    }


def compute_eval_summary(rows: list[dict]) -> dict:
    overall = _metric_bundle(rows)
    by_regime_buckets: dict[str, list[dict]] = defaultdict(list)
    by_seed_buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_regime_buckets[str(row["target_regime"])].append(row)
        by_seed_buckets[str(row["seed"])].append(row)

    return {
        "overall": overall,
        "by_regime": {regime: _metric_bundle(bucket) for regime, bucket in sorted(by_regime_buckets.items())},
        "by_seed": {seed: _metric_bundle(bucket) for seed, bucket in sorted(by_seed_buckets.items(), key=lambda x: int(x[0]))},
    }


def compute_eval_summary_with_std(rows: list[dict]) -> dict:
    summary = compute_eval_summary(rows)
    seed_mrrs = [stats["mrr"] for stats in summary["by_seed"].values()]
    if not seed_mrrs:
        summary["seed_mean_mrr"] = 0.0
        summary["seed_std_mrr"] = 0.0
        return summary

    mean_mrr = float(sum(seed_mrrs) / len(seed_mrrs))
    variance = float(sum((value - mean_mrr) ** 2 for value in seed_mrrs) / len(seed_mrrs))
    summary["seed_mean_mrr"] = mean_mrr
    summary["seed_std_mrr"] = variance**0.5
    return summary


def subgroup_eval_rows(rows: list[dict]) -> list[dict]:
    payload = []
    for regime, stats in compute_eval_summary(rows)["by_regime"].items():
        payload.append(
            {
                "target_regime": regime,
                "n_queries": stats["n_queries"],
                "mrr": stats["mrr"],
                "hits1": stats["hits1"],
                "hits3": stats["hits3"],
                "hits10": stats["hits10"],
                "fusion_coverage": stats["fusion_coverage"],
            }
        )
    return payload


def fusion_ratio_by_regime(rows: list[dict]) -> dict[str, float]:
    totals: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        regime = str(row["target_regime"])
        totals[regime]["total"] += 1
        totals[regime]["fusion"] += int(row["use_fusion"])
    return {
        regime: float(counter["fusion"] / counter["total"]) if counter["total"] else 0.0
        for regime, counter in sorted(totals.items())
    }
