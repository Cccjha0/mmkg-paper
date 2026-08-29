from __future__ import annotations

from collections import Counter, defaultdict


def mark_visual_prior(mean_delta_rr: float, gamma: float) -> int:
    return int(float(mean_delta_rr) > float(gamma))


def shrink_mean_delta_rr(mean_delta_rr: float, n_queries: int, k: float = 20.0) -> float:
    return float(mean_delta_rr) * float(n_queries / (n_queries + k)) if (n_queries + k) else 0.0


def compute_relation_gain_stats(
    gate_rows: list[dict],
    residual_rows: list[dict],
    gamma: float,
    use_shrinkage: bool = False,
    shrink_k: float = 20.0,
) -> list[dict]:
    gate_by_id = {row["query_id"]: row for row in gate_rows}
    residual_by_id = {row["query_id"]: row for row in residual_rows}
    if len(gate_by_id) != len(gate_rows) or len(residual_by_id) != len(residual_rows):
        raise RuntimeError("Duplicate query_id detected in gate or residual query rows.")

    gate_ids = set(gate_by_id)
    residual_ids = set(residual_by_id)
    if gate_ids != residual_ids:
        missing_in_gate = len(residual_ids - gate_ids)
        missing_in_residual = len(gate_ids - residual_ids)
        raise RuntimeError(
            f"query_id mismatch between experts: missing_in_gate={missing_in_gate}, missing_in_residual={missing_in_residual}"
        )

    buckets: dict[int, dict] = defaultdict(
        lambda: {
            "n_queries": 0,
            "sum_rr_gate": 0.0,
            "sum_rr_residual": 0.0,
            "sum_delta_rr": 0.0,
            "fusion_win": 0,
            "struct_win": 0,
            "regime_counter": Counter(),
            "relation_name": "",
        }
    )

    for query_id in sorted(gate_ids):
        gate = gate_by_id[query_id]
        residual = residual_by_id[query_id]
        for field in ("relation_id", "target_regime"):
            if str(gate[field]) != str(residual[field]):
                raise RuntimeError(
                    f"Expert mismatch for query_id={query_id}: "
                    f"{field} gate={gate[field]!r}, residual={residual[field]!r}."
                )
        relation_id = int(gate["relation_id"])
        rr_gate = float(gate["rr"])
        rr_residual = float(residual["rr"])
        delta_rr = rr_gate - rr_residual
        regime = str(gate["target_regime"])

        bucket = buckets[relation_id]
        bucket["n_queries"] += 1
        bucket["sum_rr_gate"] += rr_gate
        bucket["sum_rr_residual"] += rr_residual
        bucket["sum_delta_rr"] += delta_rr
        bucket["relation_name"] = gate.get("relation_name", f"rel_{relation_id:04d}")
        bucket["regime_counter"][regime] += 1
        if rr_gate > rr_residual:
            bucket["fusion_win"] += 1
        elif rr_residual > rr_gate:
            bucket["struct_win"] += 1

    rows: list[dict] = []
    for relation_id in sorted(buckets):
        bucket = buckets[relation_id]
        n = int(bucket["n_queries"])
        mean_rr_gate = bucket["sum_rr_gate"] / n if n else 0.0
        mean_rr_residual = bucket["sum_rr_residual"] / n if n else 0.0
        raw_mean_delta_rr = bucket["sum_delta_rr"] / n if n else 0.0
        mean_delta_rr = (
            shrink_mean_delta_rr(raw_mean_delta_rr, n, shrink_k) if use_shrinkage else raw_mean_delta_rr
        )
        fusion_win_rate = bucket["fusion_win"] / n if n else 0.0
        struct_win_rate = bucket["struct_win"] / n if n else 0.0
        head_has_img_ratio = bucket["regime_counter"]["head_has_img"] / n if n else 0.0
        tail_no_img_ratio = bucket["regime_counter"]["tail_no_img"] / n if n else 0.0
        general_regime_ratios = {
            f"{direction}_{tag}_ratio": bucket["regime_counter"][f"{direction}_{tag}"] / n if n else 0.0
            for direction in ("head", "tail")
            for tag in ("T0V0", "T0V1", "T1V0", "T1V1")
        }
        rows.append(
            {
                "relation_id": relation_id,
                "relation_name": bucket["relation_name"],
                "n_queries": n,
                "mean_rr_gate": mean_rr_gate,
                "mean_rr_residual": mean_rr_residual,
                "mean_delta_rr": mean_delta_rr,
                "mean_delta_rr_raw": raw_mean_delta_rr,
                "mean_delta_rr_shrunk": shrink_mean_delta_rr(raw_mean_delta_rr, n, shrink_k),
                "fusion_win_rate": fusion_win_rate,
                "struct_win_rate": struct_win_rate,
                "head_has_img_ratio": head_has_img_ratio,
                "tail_no_img_ratio": tail_no_img_ratio,
                "is_visual_prior": mark_visual_prior(mean_delta_rr, gamma),
                "is_fusion_prior": mark_visual_prior(mean_delta_rr, gamma),
                **general_regime_ratios,
            }
        )
    return rows


def summarize_relation_gain_stats(rows: list[dict], gamma: float) -> dict:
    n_relations = len(rows)
    visual_prior_count = sum(int(row["is_visual_prior"]) for row in rows)
    if not rows:
        return {
            "gamma": float(gamma),
            "n_relations": 0,
            "n_visual_prior": 0,
            "visual_prior_rate": 0.0,
            "top_positive_relations": [],
            "top_negative_relations": [],
        }

    sorted_rows = sorted(rows, key=lambda item: item["mean_delta_rr"], reverse=True)
    return {
        "gamma": float(gamma),
        "n_relations": n_relations,
        "n_visual_prior": visual_prior_count,
        "visual_prior_rate": float(visual_prior_count / n_relations) if n_relations else 0.0,
        "top_positive_relations": [
            {
                "relation_id": row["relation_id"],
                "relation_name": row["relation_name"],
                "mean_delta_rr": row["mean_delta_rr"],
                "fusion_win_rate": row["fusion_win_rate"],
                "n_queries": row["n_queries"],
            }
            for row in sorted_rows[:10]
        ],
        "top_negative_relations": [
            {
                "relation_id": row["relation_id"],
                "relation_name": row["relation_name"],
                "mean_delta_rr": row["mean_delta_rr"],
                "fusion_win_rate": row["fusion_win_rate"],
                "n_queries": row["n_queries"],
            }
            for row in sorted_rows[-10:]
        ],
    }
