from __future__ import annotations

import torch


SCORE_NORMALIZATION_MODES = ("none", "query_zscore", "rank_based")


def canonical_score_normalization(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "rank":
        normalized = "rank_based"
    if normalized not in SCORE_NORMALIZATION_MODES:
        raise ValueError(
            f"Unsupported score normalization {mode!r}; expected one of "
            f"{', '.join(SCORE_NORMALIZATION_MODES)}."
        )
    return normalized


def _query_zscore(scores: torch.Tensor, eps: float) -> torch.Tensor:
    finite = torch.isfinite(scores)
    finite_values = torch.where(finite, scores, torch.zeros_like(scores))
    count = finite.sum(dim=1, keepdim=True).clamp_min(1)
    mean = finite_values.sum(dim=1, keepdim=True) / count
    centered = torch.where(finite, scores - mean, torch.zeros_like(scores))
    variance = centered.square().sum(dim=1, keepdim=True) / count
    normalized = centered / (variance.sqrt() + float(eps))
    return torch.where(finite, normalized, torch.full_like(normalized, float("-inf")))


def _rank_based(scores: torch.Tensor, tie_policy: str = "ordinal") -> torch.Tensor:
    finite = torch.isfinite(scores)
    safe = torch.where(finite, scores, torch.full_like(scores, float("-inf")))
    order = torch.argsort(safe, dim=1, descending=True, stable=True)
    rank_values = torch.arange(1, scores.size(1) + 1, device=scores.device, dtype=order.dtype)
    rank_values = rank_values.unsqueeze(0).expand_as(order)
    if tie_policy == "ordinal":
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, rank_values)
        reciprocal_rank_score = ranks.to(dtype=scores.dtype).reciprocal()
        return torch.where(
            finite,
            reciprocal_rank_score,
            torch.full_like(reciprocal_rank_score, float("-inf")),
        )
    if tie_policy != "competition":
        raise ValueError("rank_tie_policy must be 'ordinal' or 'competition'.")

    sorted_scores = safe.gather(1, order)
    starts_tie_group = torch.ones_like(sorted_scores, dtype=torch.bool)
    starts_tie_group[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    sorted_competition_ranks = torch.where(
        starts_tie_group,
        rank_values,
        torch.zeros_like(rank_values),
    ).cummax(dim=1).values
    competition_ranks = torch.empty_like(order)
    competition_ranks.scatter_(1, order, sorted_competition_ranks)
    reciprocal_rank_score = competition_ranks.to(dtype=scores.dtype).reciprocal()
    return torch.where(
        finite,
        reciprocal_rank_score,
        torch.full_like(reciprocal_rank_score, float("-inf")),
    )


def normalize_candidate_scores(
    scores: torch.Tensor,
    mode: str = "none",
    eps: float = 1e-8,
    rank_tie_policy: str = "ordinal",
) -> torch.Tensor:
    """Normalize each expert's candidate distribution without answer access."""
    if scores.ndim != 2:
        raise ValueError("Candidate scores must be a [queries, entities] matrix.")
    mode = canonical_score_normalization(mode)
    if mode == "none":
        return scores.clone()
    if mode == "query_zscore":
        return _query_zscore(scores, eps=eps)
    return _rank_based(scores, tie_policy=rank_tie_policy)


def _finite_fill(scores: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(scores)
    count = finite.sum(dim=1, keepdim=True)
    finite_values = torch.where(finite, scores, torch.zeros_like(scores))
    mean = finite_values.sum(dim=1, keepdim=True) / count.clamp_min(1)
    centered = torch.where(finite, scores - mean, torch.zeros_like(scores))
    spread = centered.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    low = mean - spread - 1.0
    return torch.where(finite, scores, low)


def combine_expert_scores(
    fusion_scores: torch.Tensor,
    structural_scores: torch.Tensor,
    alpha: float | torch.Tensor,
    normalization: str = "none",
    eps: float = 1e-8,
    rank_tie_policy: str = "ordinal",
) -> torch.Tensor:
    if fusion_scores.shape != structural_scores.shape:
        raise ValueError("Expert candidate-score matrices must have identical shapes.")
    fusion = normalize_candidate_scores(
        fusion_scores,
        normalization,
        eps=eps,
        rank_tie_policy=rank_tie_policy,
    )
    structural = normalize_candidate_scores(
        structural_scores,
        normalization,
        eps=eps,
        rank_tie_policy=rank_tie_policy,
    )
    fusion_safe = _finite_fill(fusion)
    structural_safe = _finite_fill(structural)
    alpha_tensor = torch.as_tensor(alpha, dtype=fusion.dtype, device=fusion.device)
    if alpha_tensor.ndim == 0:
        alpha_tensor = alpha_tensor.expand(fusion.size(0)).unsqueeze(1)
    elif alpha_tensor.ndim == 1 and alpha_tensor.numel() == fusion.size(0):
        alpha_tensor = alpha_tensor.unsqueeze(1)
    else:
        raise ValueError("alpha must be scalar or contain one value per query.")
    if bool(((alpha_tensor < 0.0) | (alpha_tensor > 1.0)).any()):
        raise ValueError("alpha values must lie in [0, 1].")
    mixed = alpha_tensor * fusion_safe + (1.0 - alpha_tensor) * structural_safe
    both_filtered = (~torch.isfinite(fusion_scores)) & (~torch.isfinite(structural_scores))
    return mixed.masked_fill(both_filtered, float("-inf"))


def shrink_relation_alpha(
    relation_alpha: float,
    support: int,
    global_alpha: float,
    shrinkage_lambda: float,
) -> float:
    """Shrink a validation-selected relation alpha toward the global alpha."""
    support = max(0, int(support))
    shrinkage_lambda = float(shrinkage_lambda)
    if shrinkage_lambda < 0.0:
        raise ValueError("shrinkage_lambda must be non-negative.")
    denominator = support + shrinkage_lambda
    if denominator == 0.0:
        return float(global_alpha)
    return float(
        support / denominator * float(relation_alpha)
        + shrinkage_lambda / denominator * float(global_alpha)
    )
