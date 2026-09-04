from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from router.constants import QUERY_GEOMETRY_FIELDS


TOP_KS = (5, 10, 20)
R0_FEATURES = [*QUERY_GEOMETRY_FIELDS, "delta_alpha", "abs_delta_alpha"]
R1_ADDITIONS = [
    "r1_expert_top5_jaccard",
    "r1_expert_top10_jaccard",
    "r1_expert_top20_jaccard",
    "r1_union20_rank_spearman",
    "r1_union20_rank_displacement_mean",
    "r1_union20_rank_displacement_median",
    "r1_union20_rank_displacement_max",
    "r1_expert_top1_same",
    "r1_a_top1_rank_under_b",
    "r1_b_top1_rank_under_a",
]
R2_ADDITIONS = [
    "r2_response_top5_jaccard",
    "r2_response_top10_jaccard",
    "r2_response_top20_jaccard",
    "r2_response_union20_rank_spearman",
    "r2_response_union20_rank_displacement_mean",
    "r2_response_union20_rank_displacement_median",
    "r2_response_union20_rank_displacement_max",
    "r2_response_top1_changed",
    "r2_action_top1_top2_margin",
    "r2_response_top1_top2_margin_change",
    "r2_response_top5_mean_score_change",
    "r2_response_score_std_change",
]
R3_ADDITIONS = [
    "r3_train_relation_frequency_log1p",
    "r3_train_observed_entity_frequency_log1p",
    "r3_train_observed_entity_direction_frequency_log1p",
    "r3_train_observed_entity_unique_relation_count_log1p",
    "r3_observed_entity_has_text",
    "r3_observed_entity_has_image",
    "r3_train_relation_target_text_support",
    "r3_train_relation_target_image_support",
]
REPRESENTATION_FEATURES = {
    "R0": R0_FEATURES,
    "R1": [*R0_FEATURES, *R1_ADDITIONS],
    "R2": [*R0_FEATURES, *R2_ADDITIONS],
    "R3": [*R0_FEATURES, *R2_ADDITIONS, *R3_ADDITIONS],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def stable_order(scores: np.ndarray) -> np.ndarray:
    """Full descending score order with ascending candidate-id tie break (tests/small inputs)."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Candidate scores must be one finite vector")
    candidate_ids = np.arange(scores.size, dtype=np.int64)
    return np.lexsort((candidate_ids, -scores))


def stable_top_order(scores: np.ndarray, k: int = 20) -> np.ndarray:
    """Deterministic top-k without sorting the full candidate vector."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 < k <= scores.size:
        raise ValueError("Invalid candidate score vector/top-k")
    threshold = float(np.partition(scores, scores.size - k)[scores.size - k])
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    need = k - len(above)
    selected = np.concatenate([above, tied[:need]])
    return selected[np.lexsort((selected, -scores[selected]))]


def ordinal_ranks(order: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
    return ranks


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 1.0
    rx, ry = average_ranks(x), average_ranks(y)
    sx, sy = float(rx.std()), float(ry.std())
    if sx == 0.0 or sy == 0.0:
        return 1.0 if np.array_equal(rx, ry) else 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def ranking_comparison(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    if len(left_scores) != len(right_scores):
        raise ValueError("Ranking lengths differ")
    n = len(left_scores)
    if n < max(TOP_KS):
        raise ValueError(f"At least {max(TOP_KS)} candidates are required")
    left_order, right_order = stable_top_order(left_scores, 20), stable_top_order(right_scores, 20)
    result = {
        f"{prefix}_top{k}_jaccard": jaccard(left_order[:k], right_order[:k])
        for k in TOP_KS
    }
    union = np.asarray(sorted(set(left_order[:20]) | set(right_order[:20])), dtype=np.int64)
    left_union_order = union[np.lexsort((union, -np.asarray(left_scores)[union]))]
    right_union_order = union[np.lexsort((union, -np.asarray(right_scores)[union]))]
    left_union_rank = {int(candidate): rank for rank, candidate in enumerate(left_union_order, start=1)}
    right_union_rank = {int(candidate): rank for rank, candidate in enumerate(right_union_order, start=1)}
    lrank = np.asarray([left_union_rank[int(candidate)] for candidate in union], dtype=np.int64)
    rrank = np.asarray([right_union_rank[int(candidate)] for candidate in union], dtype=np.int64)
    scale = float(max(len(union) - 1, 1))
    displacement = np.abs(lrank - rrank).astype(np.float64) / scale
    result.update(
        {
            f"{prefix}_union20_rank_spearman": spearman(lrank, rrank),
            f"{prefix}_union20_rank_displacement_mean": float(displacement.mean()),
            f"{prefix}_union20_rank_displacement_median": float(np.median(displacement)),
            f"{prefix}_union20_rank_displacement_max": float(displacement.max()),
        }
    )
    return result


def cross_expert_features(scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, float]:
    order_a, order_b = stable_top_order(scores_a, 20), stable_top_order(scores_b, 20)
    result = ranking_comparison(scores_a, scores_b, prefix="r1_expert")
    # Contract names retain union20 instead of expert_union20.
    for suffix in (
        "rank_spearman",
        "rank_displacement_mean",
        "rank_displacement_median",
        "rank_displacement_max",
    ):
        result[f"r1_union20_{suffix}"] = result.pop(f"r1_expert_union20_{suffix}")
    n = len(scores_a)
    scale = float(max(n - 1, 1))
    def full_rank(scores: np.ndarray, candidate: int) -> int:
        value = scores[candidate]
        ids = np.arange(len(scores))
        return int(np.count_nonzero(scores > value) + np.count_nonzero((scores == value) & (ids < candidate)) + 1)
    result.update(
        {
            "r1_expert_top1_same": float(order_a[0] == order_b[0]),
            "r1_a_top1_rank_under_b": float((full_rank(scores_b, int(order_a[0])) - 1) / scale),
            "r1_b_top1_rank_under_a": float((full_rank(scores_a, int(order_b[0])) - 1) / scale),
        }
    )
    return result


def action_response_features(anchor: np.ndarray, action: np.ndarray) -> dict[str, float]:
    anchor, action = np.asarray(anchor, dtype=np.float64), np.asarray(action, dtype=np.float64)
    order_anchor, order_action = stable_top_order(anchor, 20), stable_top_order(action, 20)
    result = ranking_comparison(anchor, action, prefix="r2_response")
    anchor_margin = float(anchor[order_anchor[0]] - anchor[order_anchor[1]])
    action_margin = float(action[order_action[0]] - action[order_action[1]])
    anchor_top5_mean = float(anchor[order_anchor[:5]].mean())
    action_top5_mean = float(action[order_action[:5]].mean())
    result.update(
        {
            "r2_response_top1_changed": float(order_anchor[0] != order_action[0]),
            "r2_action_top1_top2_margin": action_margin,
            "r2_response_top1_top2_margin_change": action_margin - anchor_margin,
            "r2_response_top5_mean_score_change": action_top5_mean - anchor_top5_mean,
            "r2_response_score_std_change": float(action.std() - anchor.std()),
        }
    )
    return result


def validate_reference_response(row: dict[str, float], atol: float = 1e-10) -> None:
    expected_one = [
        "r2_response_top5_jaccard",
        "r2_response_top10_jaccard",
        "r2_response_top20_jaccard",
        "r2_response_union20_rank_spearman",
    ]
    expected_zero = [
        "r2_response_union20_rank_displacement_mean",
        "r2_response_union20_rank_displacement_median",
        "r2_response_union20_rank_displacement_max",
        "r2_response_top1_changed",
        "r2_response_top1_top2_margin_change",
        "r2_response_top5_mean_score_change",
        "r2_response_score_std_change",
    ]
    if any(abs(float(row[key]) - 1.0) > atol for key in expected_one):
        raise AssertionError("Reference response identity statistic is not one")
    if any(abs(float(row[key])) > atol for key in expected_zero):
        raise AssertionError("Reference response change statistic is not zero")


def feature_manifest() -> dict:
    return {
        "schema_version": 1,
        "status": "frozen_before_systematic_comparison",
        "top_k": list(TOP_KS),
        "rank_definition": "full unfiltered scores; descending score; ascending candidate id tie-break",
        "union20_rank_definition": "ordinal ranks recomputed within the union of the two top-20 sets",
        "rank_displacement_scale": "union displacement divided by max(|union|-1,1); cross-top1 rank divided by max(num_entities-1,1)",
        "candidate_domain": "full unfiltered candidate set",
        "score_normalization": "query_zscore via router.score_combination",
        "representations": REPRESENTATION_FEATURES,
        "removed_redundancies": {
            "membership_churn": "entering=leaving and both are a one-to-one transform of Jaccard for equal-size top-k sets",
            "concentration_change": "duplicates top5 mean score change because query-zscore mixture global mean is zero",
            "direction_context": "already present as geometry_direction_tail in Base13",
        },
        "answer_agnostic": True,
        "target_fields_used": [],
    }
