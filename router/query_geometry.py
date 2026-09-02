from __future__ import annotations

import torch


QUERY_GEOMETRY_FIELDS = (
    "geometry_direction_tail",
    "geometry_a_top1",
    "geometry_a_top5_mean",
    "geometry_a_top1_top2_margin",
    "geometry_a_score_std",
    "geometry_b_top1",
    "geometry_b_top5_mean",
    "geometry_b_top1_top2_margin",
    "geometry_b_score_std",
    "geometry_top1_delta_a_minus_b",
    "geometry_top5_delta_a_minus_b",
    "geometry_margin_delta_a_minus_b",
    "geometry_std_delta_a_minus_b",
)


def _safe_scores(scores: torch.Tensor) -> torch.Tensor:
    finite = scores[torch.isfinite(scores)]
    low = float(finite.min().item()) - 1.0 if finite.numel() else -100.0
    high = float(finite.max().item()) + 1.0 if finite.numel() else 100.0
    return torch.nan_to_num(scores, nan=low, posinf=high, neginf=low)


def _score_stats(scores: torch.Tensor) -> tuple[torch.Tensor, ...]:
    safe = _safe_scores(scores)
    top = torch.topk(safe, k=min(5, safe.size(1)), dim=1).values
    top1 = top[:, 0]
    top2 = top[:, 1] if top.size(1) > 1 else top1
    return top1, top.mean(dim=1), top1 - top2, safe.std(dim=1)


def query_geometry_tensor(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    """Return answer-agnostic score geometry for two expert candidate matrices.

    The API deliberately accepts neither target ids nor reference/target scores.
    Filtered candidate masks are permitted because they are determined from the
    observed query and the evaluation fact index, not the hidden answer score.
    """
    if scores_a.ndim != 2 or scores_b.ndim != 2:
        raise ValueError("Expert score matrices must be two-dimensional.")
    if scores_a.shape != scores_b.shape:
        raise ValueError("Expert score matrices must have identical shapes.")
    if scores_a.size(1) < 2:
        raise ValueError("Expert score matrices must contain at least two candidates.")
    if direction not in {"head", "tail"}:
        raise ValueError("direction must be 'head' or 'tail'.")

    a1, a5, am, astd = _score_stats(scores_a)
    b1, b5, bm, bstd = _score_stats(scores_b)
    direction_tail = torch.full_like(a1, 1.0 if direction == "tail" else 0.0)
    features = torch.stack(
        (
            direction_tail,
            a1,
            a5,
            am,
            astd,
            b1,
            b5,
            bm,
            bstd,
            a1 - b1,
            a5 - b5,
            am - bm,
            astd - bstd,
        ),
        dim=1,
    )
    if features.size(1) != len(QUERY_GEOMETRY_FIELDS):
        raise RuntimeError("Query-geometry field contract is inconsistent.")
    return features


def query_geometry_rows(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    direction: str,
) -> list[dict[str, float]]:
    matrix = query_geometry_tensor(scores_a, scores_b, direction).detach().cpu()
    return [
        {name: float(value) for name, value in zip(QUERY_GEOMETRY_FIELDS, row.tolist())}
        for row in matrix
    ]
