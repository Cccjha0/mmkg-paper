"""Reusable adversarial components for recent multimodal KGC baselines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import torch
import torch.nn as nn


class AdversarialLossModel(Protocol):
    def adversarial_loss(self, pos: torch.LongTensor, neg: torch.LongTensor) -> torch.Tensor: ...


class AdversarialGPLossModel(Protocol):
    def adversarial_gp_loss(self, pos: torch.LongTensor, neg: torch.LongTensor) -> torch.Tensor: ...


class CombinedGenerator(nn.Module):
    """Generate paired visual/text embeddings from all real entity modalities.

    This is the official NativE ``CombinedGenerator`` architecture without its
    hard-coded CUDA allocation.  Generated tensors already live in the shared
    ``2d`` modality-projection space used by the KGC model.
    """

    def __init__(
        self,
        *,
        noise_dim: int,
        structure_dim: int,
        modality_dim: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if min(noise_dim, structure_dim, modality_dim, hidden_dim) <= 0:
            raise ValueError("Generator dimensions must be positive.")
        self.noise_dim = noise_dim
        input_dim = noise_dim + structure_dim + 2 * modality_dim
        self.generator_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2 * modality_dim),
        )

    def forward(
        self,
        structural: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if structural.shape[:-1] != visual.shape[:-1] or structural.shape[:-1] != text.shape[:-1]:
            raise ValueError("Generator modality batches must share leading dimensions.")
        noise = torch.randn(
            (*structural.shape[:-1], self.noise_dim),
            device=structural.device,
            dtype=structural.dtype,
        )
        generated = self.generator_model(torch.cat((noise, structural, visual, text), dim=-1))
        return torch.chunk(generated, chunks=2, dim=-1)


def gradient_penalty(
    score_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    real_embeddings: Sequence[torch.Tensor],
    fake_embeddings: Sequence[torch.Tensor],
    *,
    coefficient: float = 0.1,
) -> torch.Tensor:
    """Compute NativE's WGAN-GP penalty on interpolated triple embeddings.

    The official trainer differentiates the RotatE score with respect to the
    interpolated head embedding (the first gradient returned by autograd).  We
    preserve that behavior for comparability.
    """
    if len(real_embeddings) != 3 or len(fake_embeddings) != 3:
        raise ValueError("gradient_penalty expects [head, relation, tail] embeddings.")
    batch_size = real_embeddings[0].shape[0]
    if batch_size == 0:
        raise ValueError("gradient_penalty requires a non-empty batch.")
    device = real_embeddings[0].device
    dtype = real_embeddings[0].dtype
    alpha = torch.rand((batch_size, 1), device=device, dtype=dtype)
    interpolated = []
    for real, fake in zip(real_embeddings, fake_embeddings):
        if real.shape != fake.shape or real.shape[0] != batch_size:
            raise ValueError("Real and fake embedding shapes must match.")
        value = alpha * real.detach() + (1.0 - alpha) * fake.detach()
        interpolated.append(value.requires_grad_(True))

    scores = score_fn(interpolated[0], interpolated[1], interpolated[2])
    head_gradient = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((head_gradient.norm(2, dim=1) - 1.0) ** 2).mean() * coefficient


def generator_gradient_norm(generator: nn.Module) -> float:
    """Return the global L2 gradient norm for generator diagnostics."""
    norm_sq = 0.0
    for parameter in generator.parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach()
            norm_sq += float(torch.sum(gradient * gradient).item())
    return norm_sq ** 0.5
