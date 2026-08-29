"""OpenBG-IMG adapter for AdaMF-MAT's adaptive multimodal RotatE model."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.training.src.models.recent_baselines import DirectionalScoringMixin
from ml.training.src.models.recent_baselines.rotate_utils import rotate_score


class OpenBGAdaMFMAT(DirectionalScoringMixin, nn.Module):
    """AdaMF-MAT using the protocol-fixed OpenBG-IMG raw feature caches.

    The upstream ``AdvMixRotatE`` path fine-tunes its pretrained feature
    matrices.  The OpenBG protocol instead keeps the shared raw caches fixed,
    so this adapter registers them as buffers and trains only the official
    one-layer visual/text projections on top.  Missing images therefore remain
    zero raw embeddings and ``has_img`` is retained for evaluation diagnostics.
    """

    def __init__(
        self,
        *,
        text_feat: torch.Tensor,
        img_feat: torch.Tensor,
        has_img: torch.Tensor,
        has_text: torch.Tensor | None = None,
        num_entities: int,
        num_relations: int,
        d: int = 128,
        margin: float = 6.0,
        epsilon: float = 2.0,
    ) -> None:
        super().__init__()
        if text_feat.ndim != 2 or img_feat.ndim != 2:
            raise ValueError("AdaMF-MAT feature caches must be rank-2 tensors.")
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
        if has_text is not None:
            if has_text.numel() != num_entities:
                raise ValueError("has_text length must equal num_entities.")
            self.register_buffer("has_text", has_text.detach().clone().bool())
        else:
            self.has_text = None

        self.ent_embeddings = nn.Embedding(num_entities, self.dim_e)
        self.rel_embeddings = nn.Embedding(num_relations, self.dim_r)
        ent_range = (margin + epsilon) / self.dim_e
        rel_range = (margin + epsilon) / self.dim_r
        self.register_buffer("ent_embedding_range", torch.tensor(ent_range, dtype=torch.float32))
        self.register_buffer("rel_embedding_range", torch.tensor(rel_range, dtype=torch.float32))
        self.register_buffer("margin", torch.tensor(margin, dtype=torch.float32))
        nn.init.uniform_(self.ent_embeddings.weight, -ent_range, ent_range)
        nn.init.uniform_(self.rel_embeddings.weight, -rel_range, rel_range)

        self.img_proj = nn.Linear(img_feat.shape[1], self.dim_e)
        self.text_proj = nn.Linear(text_feat.shape[1], self.dim_e)
        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)

    def get_joint_embeddings(
        self,
        structural: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        modalities = torch.stack((structural, visual, text), dim=1)
        attention_scores = self.ent_attn(torch.tanh(modalities)).squeeze(-1)
        attention = torch.softmax(attention_scores, dim=-1)
        return torch.sum(attention.unsqueeze(-1) * modalities, dim=1)

    def get_attention(
        self,
        structural: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        modalities = torch.stack((structural, visual, text), dim=1)
        return torch.softmax(self.ent_attn(torch.tanh(modalities)).squeeze(-1), dim=-1)

    def get_batch_ent_embs(self, entity_ids: torch.LongTensor) -> torch.Tensor:
        return self.ent_embeddings(entity_ids)

    def get_batch_ent_multimodal_embs(
        self,
        entity_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual = self.img_proj(self.img_feat[entity_ids])
        text = self.text_proj(self.text_feat[entity_ids])
        if self.has_text is not None:
            visual = visual * self.has_img[entity_ids].unsqueeze(-1)
            text = text * self.has_text[entity_ids].unsqueeze(-1)
        return self.ent_embeddings(entity_ids), visual, text

    def score_from_embeddings(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        return rotate_score(head, relation, tail, self.margin, self.rel_embedding_range)

    def _joint_for_triples(
        self,
        triples: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if triples.ndim != 2 or triples.shape[-1] != 3:
            raise ValueError(f"Expected triples with shape [batch, 3], got {tuple(triples.shape)}.")
        h_ids, r_ids, t_ids = triples.unbind(dim=1)
        head = self.get_joint_embeddings(*self.get_batch_ent_multimodal_embs(h_ids))
        tail = self.get_joint_embeddings(*self.get_batch_ent_multimodal_embs(t_ids))
        return head, self.rel_embeddings(r_ids), tail

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score_from_embeddings(*self._joint_for_triples(triples))

    def fake_scores(
        self,
        triples: torch.LongTensor,
        *,
        fake_head_visual: torch.Tensor,
        fake_tail_visual: torch.Tensor,
        fake_head_text: torch.Tensor,
        fake_tail_text: torch.Tensor,
    ) -> list[torch.Tensor]:
        h_ids, r_ids, t_ids = triples.unbind(dim=1)
        h_structural, h_visual, h_text = self.get_batch_ent_multimodal_embs(h_ids)
        t_structural, t_visual, t_text = self.get_batch_ent_multimodal_embs(t_ids)
        relation = self.rel_embeddings(r_ids)

        real_head = self.get_joint_embeddings(h_structural, h_visual, h_text)
        real_tail = self.get_joint_embeddings(t_structural, t_visual, t_text)
        fake_head = self.get_joint_embeddings(h_structural, fake_head_visual, fake_head_text)
        fake_tail = self.get_joint_embeddings(t_structural, fake_tail_visual, fake_tail_text)
        return [
            self.score_from_embeddings(fake_head, relation, real_tail),
            self.score_from_embeddings(real_head, relation, fake_tail),
            self.score_from_embeddings(fake_head, relation, fake_tail),
        ]

    def regularization(self, triples: torch.LongTensor) -> torch.Tensor:
        h_ids, r_ids, t_ids = triples.unbind(dim=1)
        return (
            self.ent_embeddings(h_ids).pow(2).mean()
            + self.ent_embeddings(t_ids).pow(2).mean()
            + self.rel_embeddings(r_ids).pow(2).mean()
        ) / 3.0
