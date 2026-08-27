"""Shared primitives for recent multimodal-KGC baselines.

Concrete baseline implementations deliberately live in sibling modules.  This
package only establishes shared contracts so existing models remain untouched.
"""

from __future__ import annotations

import torch


class DirectionalScoringMixin:
    """Default directional-scoring contract for recent baseline models.

    Models whose score is direction-independent inherit these two methods.
    Reciprocal-relation models may override ``score_head``.
    """

    def score_tail(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score(triples)

    def score_head(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score(triples)


__all__ = ["DirectionalScoringMixin"]
