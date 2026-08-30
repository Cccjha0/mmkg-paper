import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.src.models.decoders.complex import ComplEx


class StructureComplExLP(nn.Module):
    """
    Pure structural baseline:
    - learn one embedding per entity
    - use the shared ComplEx decoder for relation scoring
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d: int = 256,
        neg_ratio: int = 10,
        adv_temperature: float = 1.0,
        entity_l2_weight: float = 1e-6,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.d = d
        self.neg_ratio = neg_ratio
        self.adv_temperature = adv_temperature
        self.entity_l2_weight = entity_l2_weight

        self.entity = nn.Embedding(num_entities, d)
        nn.init.xavier_uniform_(self.entity.weight)

        self.decoder = ComplEx(num_relations=num_relations, d=d)

    def entity_repr(self, eids: torch.LongTensor) -> torch.Tensor:
        return self.entity(eids)

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        h = self.entity_repr(triples[:, 0])
        t = self.entity_repr(triples[:, 2])
        r = triples[:, 1]
        return self.decoder.score(h, r, t)

    def forward(self, pos: torch.LongTensor, neg: torch.LongTensor) -> torch.Tensor:
        pos_score = self.score(pos)
        neg_score = self.score(neg)

        if neg_score.numel() != pos_score.numel() * self.neg_ratio:
            raise ValueError("Negative score count must equal batch_size * neg_ratio.")
        neg_score = neg_score.view(pos_score.size(0), self.neg_ratio)
        pos_loss = F.softplus(-pos_score)
        with torch.no_grad():
            neg_weight = F.softmax(neg_score * self.adv_temperature, dim=1)
        neg_loss = (neg_weight * F.softplus(neg_score)).sum(dim=1)

        reg = self.entity_l2_weight * self.entity.weight.pow(2).mean()
        return (pos_loss + neg_loss).mean() + reg


class StructureTuckERLP(nn.Module):
    """
    Pure structural TuckER baseline under the current unified trainer:
    - learn one embedding per entity
    - learn one embedding per relation
    - learn a shared Tucker core tensor
    - use the same negative-sampling training/eval protocol as other models
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d: int = 256,
        neg_ratio: int = 10,
        adv_temperature: float = 1.0,
        entity_l2_weight: float = 1e-6,
        relation_l2_weight: float = 1e-6,
        core_l2_weight: float = 1e-6,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.d = d
        self.neg_ratio = neg_ratio
        self.adv_temperature = adv_temperature
        self.entity_l2_weight = entity_l2_weight
        self.relation_l2_weight = relation_l2_weight
        self.core_l2_weight = core_l2_weight

        self.entity = nn.Embedding(num_entities, d)
        self.relation = nn.Embedding(num_relations, d)
        self.core = nn.Parameter(torch.empty(d, d, d))

        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.xavier_uniform_(self.core)

    def entity_repr(self, eids: torch.LongTensor) -> torch.Tensor:
        return self.entity(eids)

    def relation_repr(self, rids: torch.LongTensor) -> torch.Tensor:
        return self.relation(rids)

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        h = self.entity_repr(triples[:, 0])   # [B, d]
        r = self.relation_repr(triples[:, 1]) # [B, d]
        t = self.entity_repr(triples[:, 2])   # [B, d]

        core_flat = self.core.view(self.d, -1)             # [d, d*d]
        wr = torch.matmul(r, core_flat).view(-1, self.d, self.d)
        hr = torch.bmm(h.unsqueeze(1), wr).squeeze(1)      # [B, d]
        return (hr * t).sum(dim=-1)

    def forward(self, pos: torch.LongTensor, neg: torch.LongTensor) -> torch.Tensor:
        pos_score = self.score(pos)
        neg_score = self.score(neg)

        pos_loss = F.softplus(-pos_score).mean()
        neg_weight = F.softmax(neg_score * self.adv_temperature, dim=0).detach()
        neg_loss = (neg_weight * F.softplus(neg_score)).sum()

        reg = (
            self.entity_l2_weight * self.entity.weight.pow(2).mean()
            + self.relation_l2_weight * self.relation.weight.pow(2).mean()
            + self.core_l2_weight * self.core.pow(2).mean()
        )
        return pos_loss + neg_loss + reg
