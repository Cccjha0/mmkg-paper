"""Shared reciprocal-relation helpers for one-vs-all recent baselines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


def build_inverse_relation_ids(num_relations: int) -> torch.LongTensor:
    """Return the base/inverse lookup for ``[0, R)`` and ``[R, 2R)``."""
    if num_relations <= 0:
        raise ValueError("num_relations must be positive.")
    base = torch.arange(num_relations, dtype=torch.long)
    return torch.cat((base + num_relations, base), dim=0)


def reciprocal_head_triples(
    triples: torch.LongTensor,
    inverse_relation_ids: torch.LongTensor,
) -> torch.LongTensor:
    """Map ``(h, r, t)`` head queries to reciprocal tail queries ``(t, r^-1, h)``."""
    if triples.ndim != 2 or triples.shape[-1] != 3:
        raise ValueError(f"Expected triples with shape [batch, 3], got {tuple(triples.shape)}.")
    if inverse_relation_ids.ndim != 1:
        raise ValueError("inverse_relation_ids must be a one-dimensional relation-id lookup table.")
    relation_ids = triples[:, 1]
    if relation_ids.numel() and (
        relation_ids.min() < 0 or relation_ids.max() >= inverse_relation_ids.numel()
    ):
        raise ValueError("A triple contains a relation id not present in inverse_relation_ids.")

    inverse_ids = inverse_relation_ids.to(device=triples.device)[relation_ids]
    return torch.stack((triples[:, 2], inverse_ids, triples[:, 0]), dim=1)


def augment_with_reciprocals(
    triples: Sequence[Sequence[int]] | torch.LongTensor,
    num_relations: int,
) -> list[tuple[int, int, int]]:
    """Append ``(t, r + R, h)`` for every base-relation training triple."""
    rows: Iterable[Sequence[int]] = triples.detach().cpu().tolist() if isinstance(triples, torch.Tensor) else triples
    original = [tuple(map(int, row)) for row in rows]
    for h, r, t in original:
        if not 0 <= r < num_relations:
            raise ValueError(
                f"Training relation id {r} is outside the base range [0, {num_relations})."
            )
    reciprocal = [(t, r + num_relations, h) for h, r, t in original]
    return original + reciprocal


class ReciprocalHeadScoringMixin:
    """Implement head scoring by converting it to reciprocal tail scoring."""

    inverse_relation_ids: torch.LongTensor

    def score_head(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score_tail(reciprocal_head_triples(triples, self.inverse_relation_ids))


__all__ = [
    "ReciprocalHeadScoringMixin",
    "augment_with_reciprocals",
    "build_inverse_relation_ids",
    "reciprocal_head_triples",
]
