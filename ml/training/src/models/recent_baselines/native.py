"""OpenBG-IMG adapter for NativE's relation-guided multimodal RotatE model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.src.models.recent_baselines import DirectionalScoringMixin
from ml.training.src.models.recent_baselines.rotate_utils import rotate_score


class OpenBGNativE(DirectionalScoringMixin, nn.Module):
    """NativE using the repository's fixed raw text/image feature caches.

    The implementation ports the official ``AdvRelRotatE`` three-modality
    path: two-layer modality projections, entity attention, relation gate, and
    RotatE scoring.  Missing image rows remain the protocol-defined zero raw
    embeddings; NativE does not introduce a learned missing-image vector.
    """

    def __init__(
        self,
        *,
        text_feat: torch.Tensor,
        img_feat: torch.Tensor,
        has_img: torch.Tensor,
        num_entities: int,
        num_relations: int,
        d: int = 128,
        margin: float = 6.0,
        epsilon: float = 2.0,
    ) -> None:
        super().__init__()
        if text_feat.ndim != 2 or img_feat.ndim != 2:
            raise ValueError("NativE feature caches must be rank-2 tensors.")
        if text_feat.shape[0] != num_entities or img_feat.shape[0] != num_entities:
            raise ValueError("Feature rows must equal num_entities.")
        if has_img.numel() != num_entities:
            raise ValueError("has_img length must equal num_entities.")
        if d <= 0 or num_relations <= 0:
            raise ValueError("d and num_relations must be positive.")

        self.d = d
        self.dim_e = 2 * d
        self.dim_r = d
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.register_buffer("text_feat", text_feat.detach().clone().float())
        self.register_buffer("img_feat", img_feat.detach().clone().float())
        self.register_buffer("has_img", has_img.detach().clone().to(dtype=torch.bool))

        self.ent_embeddings = nn.Embedding(num_entities, self.dim_e)
        self.rel_embeddings = nn.Embedding(num_relations, self.dim_r)
        ent_range = (margin + epsilon) / self.dim_e
        rel_range = (margin + epsilon) / self.dim_r
        self.register_buffer("ent_embedding_range", torch.tensor(ent_range, dtype=torch.float32))
        self.register_buffer("rel_embedding_range", torch.tensor(rel_range, dtype=torch.float32))
        self.register_buffer("margin", torch.tensor(margin, dtype=torch.float32))
        nn.init.uniform_(self.ent_embeddings.weight, -ent_range, ent_range)
        nn.init.uniform_(self.rel_embeddings.weight, -rel_range, rel_range)

        self.img_proj = nn.Sequential(
            nn.Linear(img_feat.shape[1], self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat.shape[1], self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e),
        )
        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.rel_gate = nn.Embedding(num_relations, 1)
        nn.init.uniform_(self.rel_gate.weight, -ent_range, ent_range)

    def get_joint_embeddings(
        self,
        structural: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
        relation_gate: torch.Tensor,
    ) -> torch.Tensor:
        modalities = torch.stack((structural, visual, text), dim=1)
        attention_logits = self.ent_attn(torch.tanh(modalities)).squeeze(-1)
        # Official NativE V8 relation-guided adaptive attention.
        temperature = torch.sigmoid(relation_gate)
        attention_weights = F.softmax(attention_logits / temperature, dim=-1)
        return torch.sum(attention_weights.unsqueeze(-1) * modalities, dim=1)

    def get_batch_ent_multimodal_embs(
        self,
        entity_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.ent_embeddings(entity_ids),
            self.img_proj(self.img_feat[entity_ids]),
            self.text_proj(self.text_feat[entity_ids]),
        )

    def _joint_for_triples(
        self,
        triples: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if triples.ndim != 2 or triples.shape[-1] != 3:
            raise ValueError(f"Expected triples with shape [batch, 3], got {tuple(triples.shape)}.")
        h_ids, r_ids, t_ids = triples.unbind(dim=1)
        h_structural, h_visual, h_text = self.get_batch_ent_multimodal_embs(h_ids)
        t_structural, t_visual, t_text = self.get_batch_ent_multimodal_embs(t_ids)
        relation = self.rel_embeddings(r_ids)
        relation_gate = self.rel_gate(r_ids)
        head = self.get_joint_embeddings(h_structural, h_visual, h_text, relation_gate)
        tail = self.get_joint_embeddings(t_structural, t_visual, t_text, relation_gate)
        return head, relation, tail

    def score_from_embeddings(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        return rotate_score(head, relation, tail, self.margin, self.rel_embedding_range)

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score_from_embeddings(*self._joint_for_triples(triples))

    def score_and_embeddings(
        self,
        triples: torch.LongTensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        embeddings = self._joint_for_triples(triples)
        return self.score_from_embeddings(*embeddings), embeddings

    def fake_scores_and_embeddings(
        self,
        triples: torch.LongTensor,
        *,
        fake_head_visual: torch.Tensor,
        fake_tail_visual: torch.Tensor,
        fake_head_text: torch.Tensor,
        fake_tail_text: torch.Tensor,
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h_ids, r_ids, t_ids = triples.unbind(dim=1)
        h_structural, h_visual, h_text = self.get_batch_ent_multimodal_embs(h_ids)
        t_structural, t_visual, t_text = self.get_batch_ent_multimodal_embs(t_ids)
        relation = self.rel_embeddings(r_ids)
        relation_gate = self.rel_gate(r_ids)

        real_head = self.get_joint_embeddings(h_structural, h_visual, h_text, relation_gate)
        real_tail = self.get_joint_embeddings(t_structural, t_visual, t_text, relation_gate)
        fake_head = self.get_joint_embeddings(
            h_structural,
            fake_head_visual,
            fake_head_text,
            relation_gate,
        )
        fake_tail = self.get_joint_embeddings(
            t_structural,
            fake_tail_visual,
            fake_tail_text,
            relation_gate,
        )
        scores = [
            self.score_from_embeddings(fake_head, relation, real_tail),
            self.score_from_embeddings(real_head, relation, fake_tail),
            self.score_from_embeddings(fake_head, relation, fake_tail),
        ]
        return scores, (fake_head, relation, fake_tail)

    def regularization(self, triples: torch.LongTensor) -> torch.Tensor:
        h_ids, r_ids, t_ids = triples.unbind(dim=1)
        return (
            self.ent_embeddings(h_ids).pow(2).mean()
            + self.ent_embeddings(t_ids).pow(2).mean()
            + self.rel_embeddings(r_ids).pow(2).mean()
        ) / 3.0
