from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.src.models.decoders.complex import ComplEx


class AvailabilityAwareGatedFusion(nn.Module):
    """Relation-aware vector fusion with explicit modality availability masks."""

    def __init__(self, d: int, num_relations: int, use_layernorm: bool = True) -> None:
        super().__init__()
        self.d = int(d)
        self.use_layernorm = bool(use_layernorm)
        if self.use_layernorm:
            self.text_norm = nn.LayerNorm(self.d)
            self.image_norm = nn.LayerNorm(self.d)
        # Keep the expressive per-dimension gate used by Gate v1 while adding
        # strict availability masking.  A scalar two-logit gate proved too
        # restrictive on DB15K because all embedding dimensions had to share
        # the same modality mixture.
        self.gate = nn.Linear(2 * self.d + 2, self.d)
        self.relation_bias = nn.Embedding(num_relations, self.d)
        self.fallback = nn.Parameter(torch.zeros(self.d))
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.relation_bias.weight)
        nn.init.normal_(self.fallback, mean=0.0, std=0.02)

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        relations: torch.LongTensor,
        has_text: torch.Tensor,
        has_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text.shape != image.shape:
            raise ValueError("Projected text and image tensors must have identical shapes.")
        has_text = has_text.to(device=text.device, dtype=torch.bool)
        has_img = has_img.to(device=text.device, dtype=torch.bool)
        if has_text.ndim != 1 or has_img.ndim != 1 or has_text.shape != has_img.shape:
            raise ValueError("Availability masks must be matching rank-1 tensors.")

        if self.use_layernorm:
            text = self.text_norm(text)
            image = self.image_norm(image)

        text_mask = has_text.unsqueeze(-1)
        image_mask = has_img.unsqueeze(-1)
        # Unavailable raw/projected values cannot affect either gate logits or
        # the fused representation.
        text_observed = torch.where(text_mask, text, torch.zeros_like(text))
        image_observed = torch.where(image_mask, image, torch.zeros_like(image))
        availability = torch.stack([has_text, has_img], dim=-1)
        logits = self.gate(
            torch.cat([text_observed, image_observed, availability.to(text.dtype)], dim=-1)
        ) + self.relation_bias(relations)

        any_available = availability.any(dim=-1)
        both_available = has_text & has_img
        text_only = has_text & ~has_img
        image_only = ~has_text & has_img

        learned_text_weight = torch.sigmoid(logits)
        text_weight = torch.zeros_like(learned_text_weight)
        text_weight = torch.where(both_available.unsqueeze(-1), learned_text_weight, text_weight)
        text_weight = torch.where(text_only.unsqueeze(-1), torch.ones_like(text_weight), text_weight)

        image_weight = torch.zeros_like(learned_text_weight)
        image_weight = torch.where(both_available.unsqueeze(-1), 1.0 - learned_text_weight, image_weight)
        image_weight = torch.where(image_only.unsqueeze(-1), torch.ones_like(image_weight), image_weight)
        weights = torch.stack([text_weight, image_weight], dim=1)

        fused = text_weight * text_observed + image_weight * image_observed
        fallback = self.fallback.unsqueeze(0).expand_as(fused)
        fused = torch.where(any_available.unsqueeze(-1), fused, fallback)
        return fused, weights


class MMKGAvailabilityAwareFusionLP(nn.Module):
    """Availability-aware multimodal expert for ``mmkg_general_v1``."""

    def __init__(
        self,
        text_feat: torch.Tensor,
        img_feat: torch.Tensor,
        has_text: torch.Tensor,
        has_img: torch.Tensor,
        num_relations: int,
        d: int = 256,
        use_layernorm: bool = True,
        neg_ratio: int = 10,
        adv_temperature: float = 1.0,
        text_dropout: float = 0.0,
        img_dropout: float = 0.0,
        gate_reg_weight: float = 1e-3,
        gate_reg_target: float = 0.5,
    ) -> None:
        super().__init__()
        if text_feat.ndim != 2 or img_feat.ndim != 2:
            raise ValueError("text_feat and img_feat must be rank-2 tensors.")
        if text_feat.size(0) != img_feat.size(0):
            raise ValueError("Text and image features must share entity alignment.")
        if has_text.numel() != text_feat.size(0) or has_img.numel() != text_feat.size(0):
            raise ValueError("Availability masks must contain one value per entity.")
        for name, value in (("text_dropout", text_dropout), ("img_dropout", img_dropout)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

        self.num_entities = int(text_feat.size(0))
        self.num_relations = int(num_relations)
        self.d = int(d)
        self.neg_ratio = int(neg_ratio)
        self.adv_temperature = float(adv_temperature)
        self.text_dropout = float(text_dropout)
        self.img_dropout = float(img_dropout)
        self.gate_reg_weight = float(gate_reg_weight)
        self.gate_reg_target = float(gate_reg_target)

        self.register_buffer("text_feat", text_feat.detach().float().clone())
        self.register_buffer("img_feat", img_feat.detach().float().clone())
        self.register_buffer("has_text", has_text.detach().bool().clone())
        self.register_buffer("has_img", has_img.detach().bool().clone())

        text_dim = int(text_feat.size(1))
        image_dim = int(img_feat.size(1))
        self.text_proj = nn.Identity() if text_dim == self.d else nn.Linear(text_dim, self.d)
        self.img_proj = nn.Identity() if image_dim == self.d else nn.Linear(image_dim, self.d)
        self.text_adapter = nn.Sequential(nn.Linear(self.d, self.d), nn.GELU(), nn.LayerNorm(self.d))
        self.img_adapter = nn.Sequential(nn.Linear(self.d, self.d), nn.GELU(), nn.LayerNorm(self.d))
        self.fusion = AvailabilityAwareGatedFusion(
            d=self.d,
            num_relations=self.num_relations,
            use_layernorm=use_layernorm,
        )
        self.decoder = ComplEx(num_relations=self.num_relations, d=self.d)

    def _effective_availability(self, eids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        has_text = self.has_text[eids]
        has_img = self.has_img[eids]
        if self.training and self.text_dropout > 0.0:
            text_dropped = torch.rand(eids.size(0), device=eids.device) < self.text_dropout
            has_text = has_text & ~text_dropped
        if self.training and self.img_dropout > 0.0:
            image_dropped = torch.rand(eids.size(0), device=eids.device) < self.img_dropout
            has_img = has_img & ~image_dropped
        return has_text, has_img

    def entity_with_relation(
        self,
        eids: torch.LongTensor,
        relations: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.text_adapter(self.text_proj(self.text_feat[eids]))
        image = self.img_adapter(self.img_proj(self.img_feat[eids]))
        has_text, has_img = self._effective_availability(eids)
        return self.fusion(text, image, relations, has_text, has_img)

    @torch.no_grad()
    def gate_for_entities(self, eids: torch.LongTensor) -> torch.Tensor:
        """Return deterministic per-entity mean text weights for diagnostics."""
        relations = eids.remainder(self.num_relations)
        _, weights = self.entity_with_relation(eids, relations)
        return weights[:, 0, :].mean(dim=-1)

    def _score_with_aux(
        self,
        triples: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relation = triples[:, 1]
        head, head_weights = self.entity_with_relation(triples[:, 0], relation)
        tail, tail_weights = self.entity_with_relation(triples[:, 2], relation)
        return self.decoder.score(head, relation, tail), head_weights, tail_weights

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        scores, _, _ = self._score_with_aux(triples)
        return scores

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

    def _gate_regularization(self, *weight_tensors: torch.Tensor) -> torch.Tensor:
        if self.gate_reg_weight <= 0.0:
            return self.fusion.fallback.new_zeros(())
        weights = torch.cat(weight_tensors, dim=0)
        both_available = (weights[:, 0, :].sum(dim=-1) > 0) & (weights[:, 1, :].sum(dim=-1) > 0)
        if not bool(both_available.any()):
            return weights.new_zeros(())
        text_weight = weights[both_available, 0, :].mean()
        return self.gate_reg_weight * (text_weight - self.gate_reg_target).pow(2)

    def forward(self, positive: torch.LongTensor, negative: torch.LongTensor) -> torch.Tensor:
        positive_scores, pos_head_weights, pos_tail_weights = self._score_with_aux(positive)
        negative_scores, neg_head_weights, neg_tail_weights = self._score_with_aux(negative)
        return self._self_adversarial_loss(positive_scores, negative_scores) + self._gate_regularization(
            pos_head_weights,
            pos_tail_weights,
            neg_head_weights,
            neg_tail_weights,
        )
