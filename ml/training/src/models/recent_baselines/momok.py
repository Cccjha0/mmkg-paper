"""Reciprocal-direction scoring primitive required by a future MoMoK model."""

from __future__ import annotations

import torch


def reciprocal_head_triples(
    triples: torch.LongTensor,
    inverse_relation_ids: torch.LongTensor,
) -> torch.LongTensor:
    """Map ``(h, r, t)`` head queries to reciprocal tail queries ``(t, r⁻¹, h)``."""
    if triples.ndim != 2 or triples.shape[-1] != 3:
        raise ValueError(f"Expected triples with shape [batch, 3], got {tuple(triples.shape)}.")
    if inverse_relation_ids.ndim != 1:
        raise ValueError("inverse_relation_ids must be a one-dimensional relation-id lookup table.")
    relation_ids = triples[:, 1]
    if relation_ids.numel() and (relation_ids.min() < 0 or relation_ids.max() >= inverse_relation_ids.numel()):
        raise ValueError("A triple contains a relation id not present in inverse_relation_ids.")

    inverse_ids = inverse_relation_ids.to(device=triples.device)[relation_ids]
    return torch.stack((triples[:, 2], inverse_ids, triples[:, 0]), dim=1)


class ReciprocalHeadScoringMixin:
    """Implement head scoring by converting it to reciprocal tail scoring.

    A concrete MoMoK model must define ``inverse_relation_ids`` and
    ``score_tail``.  It may retain a distinct ``score`` implementation for
    ordinary triple scoring, but evaluator head ranking always follows this
    reciprocal path.
    """

    inverse_relation_ids: torch.LongTensor

    def score_head(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score_tail(reciprocal_head_triples(triples, self.inverse_relation_ids))
