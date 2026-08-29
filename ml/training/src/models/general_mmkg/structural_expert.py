from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.src.models.decoders.complex import ComplEx


class MMKGStructuralExpertLP(nn.Module):
    """Pure structure-oriented expert for ``mmkg_general_v1``.

    The constructor deliberately has no modality features or availability
    masks.  Its state dict therefore cannot depend on text/image preprocessing.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d: int = 256,
        neg_ratio: int = 10,
        adv_temperature: float = 1.0,
        entity_l2_weight: float = 1e-6,
        structural_scale_init: float = -2.0,
        structural_scale_l2_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.d = int(d)
        self.neg_ratio = int(neg_ratio)
        self.adv_temperature = float(adv_temperature)
        self.entity_l2_weight = float(entity_l2_weight)
        self.structural_scale_l2_weight = float(structural_scale_l2_weight)

        self.entity_structural = nn.Embedding(self.num_entities, self.d)
        self.structural_scale = nn.Parameter(torch.tensor(float(structural_scale_init)))
        self.decoder = ComplEx(num_relations=self.num_relations, d=self.d)
        nn.init.xavier_uniform_(self.entity_structural.weight)

    def entity_repr(self, eids: torch.LongTensor) -> torch.Tensor:
        return F.softplus(self.structural_scale) * self.entity_structural(eids)

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        head = self.entity_repr(triples[:, 0])
        relation = triples[:, 1]
        tail = self.entity_repr(triples[:, 2])
        return self.decoder.score(head, relation, tail)

    @torch.no_grad()
    def score_eval(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score(triples)

    def _self_adversarial_loss(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        if negative_scores.numel() != positive_scores.numel() * self.neg_ratio:
            raise ValueError("Negative score count must equal batch_size * neg_ratio.")
        negative_scores = negative_scores.view(positive_scores.size(0), self.neg_ratio)
        positive_loss = F.softplus(-positive_scores)
        with torch.no_grad():
            weights = F.softmax(self.adv_temperature * negative_scores, dim=1)
        negative_loss = (weights * F.softplus(negative_scores)).sum(dim=1)
        return (positive_loss + negative_loss).mean()

    def forward(self, positive: torch.LongTensor, negative: torch.LongTensor) -> torch.Tensor:
        loss = self._self_adversarial_loss(self.score(positive), self.score(negative))
        scale = F.softplus(self.structural_scale)
        regularization = (
            self.entity_l2_weight * self.entity_structural.weight.pow(2).mean()
            + self.structural_scale_l2_weight * scale.pow(2)
        )
        return loss + regularization
