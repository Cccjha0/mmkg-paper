"""Shared RotatE-family scoring utilities.

All functions return scores where larger values indicate better triples, which
matches the repository's filtered-ranking evaluator.
"""

from __future__ import annotations

import math

import torch


def _validate_rotate_shapes(h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> None:
    if h.shape != t.shape:
        raise ValueError(f"RotatE head/tail shapes must match; got {h.shape} and {t.shape}.")
    if h.ndim == 0 or h.shape[-1] % 2 != 0:
        raise ValueError("RotatE entity embeddings must have an even final dimension [real | imaginary].")
    if r.shape[-1] != h.shape[-1] // 2:
        raise ValueError(
            "RotatE relation embeddings must have half the entity final dimension; "
            f"got relation={r.shape[-1]}, entity={h.shape[-1]}."
        )


def rotate_distance(
    h: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    embedding_range: float | torch.Tensor,
) -> torch.Tensor:
    """Return the standard L2-per-complex-dimension RotatE distance.

    ``h`` and ``t`` store concatenated real and imaginary parts with final size
    ``2d``.  ``r`` stores ``d`` phase parameters and is converted using the
    original RotatE scale ``embedding_range / pi``.
    """
    _validate_rotate_shapes(h, r, t)
    if isinstance(embedding_range, torch.Tensor):
        if embedding_range.numel() != 1:
            raise ValueError("embedding_range must be scalar.")
        embedding_range = embedding_range.to(device=r.device, dtype=r.dtype)
        if float(embedding_range.detach().cpu().item()) <= 0:
            raise ValueError("embedding_range must be positive.")
    elif embedding_range <= 0:
        raise ValueError("embedding_range must be positive.")

    phase = r / (embedding_range / math.pi)
    re_r = torch.cos(phase)
    im_r = torch.sin(phase)
    re_h, im_h = torch.chunk(h, 2, dim=-1)
    re_t, im_t = torch.chunk(t, 2, dim=-1)

    re_delta = re_h * re_r - im_h * im_r - re_t
    im_delta = re_h * im_r + im_h * re_r - im_t
    return torch.linalg.vector_norm(torch.stack((re_delta, im_delta), dim=0), dim=0).sum(dim=-1)


def rotate_score(
    h: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    margin: float | torch.Tensor,
    embedding_range: float | torch.Tensor,
) -> torch.Tensor:
    """Return a higher-is-better RotatE score: ``margin - distance``."""
    return margin - rotate_distance(h, r, t, embedding_range)
