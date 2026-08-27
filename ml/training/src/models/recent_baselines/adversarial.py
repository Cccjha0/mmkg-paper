"""Contracts shared by adversarial recent-baseline training engines.

Concrete models must expose the engine-specific loss method documented in
``trainer_recent.py``.  Loss implementations are intentionally deferred until
the corresponding baseline is integrated.
"""

from __future__ import annotations

from typing import Protocol

import torch


class AdversarialLossModel(Protocol):
    def adversarial_loss(self, pos: torch.LongTensor, neg: torch.LongTensor) -> torch.Tensor: ...


class AdversarialGPLossModel(Protocol):
    def adversarial_gp_loss(self, pos: torch.LongTensor, neg: torch.LongTensor) -> torch.Tensor: ...
