"""Backward-compatible exports for MoMoK's reciprocal scoring contract."""

from ml.training.src.models.recent_baselines.reciprocal import (
    ReciprocalHeadScoringMixin,
    augment_with_reciprocals,
    build_inverse_relation_ids,
    reciprocal_head_triples,
)

__all__ = [
    "ReciprocalHeadScoringMixin",
    "augment_with_reciprocals",
    "build_inverse_relation_ids",
    "reciprocal_head_triples",
]
