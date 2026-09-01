"""OpenBG-IMG adapter for APKGC's RotatE-based multimodal model.

This module ports the APKGC main path (structural embeddings, frozen modality
features, Gaussian missing-image replacement, noise augmentation, Mformer
fusion, attention penalty, and RotatE scoring) without importing the external
project's OpenKE or Transformers runtime.
"""

from __future__ import annotations

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.training.src.models.recent_baselines import DirectionalScoringMixin
from ml.training.src.models.recent_baselines.rotate_utils import rotate_score


class APKGCAttentionLayer(nn.Module):
    """APKGC's three-token Mformer layer with its attention-penalty rule."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        *,
        dropout: float = 0.1,
        use_intermediate: bool = False,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.use_intermediate = use_intermediate
        if use_intermediate:
            intermediate_size = intermediate_size or hidden_size
            self.intermediate = nn.Linear(hidden_size, intermediate_size)
            self.intermediate_output = nn.Linear(intermediate_size, hidden_size)
            self.intermediate_dropout = nn.Dropout(dropout)
            self.intermediate_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # Values match the upstream APKGC attention penalty implementation.
        self.penalty_threshold = 10.0
        self.penalty_max_threshold = 15.0
        self.window_size = 1
        self.penalty_weights = 0.001
        self.origin_threshold = 0.01

    def _transpose_for_scores(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        value = value.view(batch, tokens, self.num_attention_heads, self.attention_head_size)
        return value.permute(0, 2, 1, 3)

    def _apply_attention_penalty(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Port the upstream reassignment rule before softmax.

        APKGC uses three modality tokens, so the small explicit loop keeps the
        original windowed formulation readable and avoids an external dependency.
        """
        token_count = attention_scores.shape[-1]
        penalty_scores = torch.zeros_like(attention_scores)
        for index in range(token_count):
            penalty_scores[..., index] = attention_scores[..., index:].prod(dim=-1)
        exceeds = penalty_scores > self.penalty_threshold
        reassigned = attention_scores.clone()

        for offset in range(-self.window_size, self.window_size + 1):
            if offset == 0:
                continue
            source_start = max(0, offset)
            source_end = min(token_count, token_count + offset)
            target_start = max(0, -offset)
            target_end = min(token_count, token_count - offset)
            mask = exceeds[..., source_start:source_end]
            if not bool(mask.any()):
                continue
            source_scores = attention_scores[..., source_start:source_end]
            penalty_value = penalty_scores[..., source_start:source_end]
            clamped = penalty_value.clamp(min=-self.origin_threshold, max=self.origin_threshold)
            clamped_attention = clamped * source_scores
            token_value = F.softmax(-penalty_value, dim=0)
            token_value = torch.linalg.vector_norm(token_value, dim=-1, keepdim=True)
            token_value = F.normalize(token_value, p=2, dim=-1)
            token_value = token_value.clamp(max=self.penalty_max_threshold)
            penalty_attention = source_scores * token_value * self.penalty_weights
            reassigned[..., target_start:target_end] += torch.where(
                mask,
                -torch.abs(clamped_attention),
                torch.abs(penalty_attention),
            )
        return reassigned

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._transpose_for_scores(self.query(hidden_states))
        key = self._transpose_for_scores(self.key(hidden_states))
        value = self._transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        probabilities = F.softmax(self._apply_attention_penalty(scores), dim=-1)
        probabilities = self.attention_dropout(probabilities)
        context = torch.matmul(probabilities, value)
        context = context.permute(0, 2, 1, 3).contiguous().view_as(hidden_states)

        output = self.output_norm(self.output_dropout(self.output_dense(context)) + hidden_states)
        if self.use_intermediate:
            intermediate = F.gelu(self.intermediate(output))
            output = self.intermediate_norm(
                self.intermediate_dropout(self.intermediate_output(intermediate)) + output
            )
        return output, probabilities


class OpenBGAPKGC(DirectionalScoringMixin, nn.Module):
    """APKGC on the fixed OpenBG-IMG raw feature caches.

    The model follows the upstream Mformer mean/graph fusion paths.  Missing
    images are sampled once from the valid-image Gaussian, as in APKGC, while
    ``img_feat`` itself remains the unchanged, shared raw cache.
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
        num_hidden_layers: int = 1,
        num_attention_heads: int = 1,
        joint_way: str = "Mformer_hd_mean",
        num_proj: int = 1,
        add_noise: bool = False,
        noise_update: str = "epoch",
        noise_ratio: float = 0.2,
        mask_ratio: float = 0.7,
        adv_temperature: float = 2.0,
        attention_dropout: float = 0.1,
        use_intermediate: bool = False,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        if text_feat.shape[0] != num_entities or img_feat.shape[0] != num_entities:
            raise ValueError("Feature rows must equal num_entities.")
        if has_img.numel() != num_entities:
            raise ValueError("has_img length must equal num_entities.")
        if d <= 0 or num_relations <= 0:
            raise ValueError("d and num_relations must be positive.")
        if num_proj not in {1, 2}:
            raise ValueError("num_proj must be 1 or 2.")
        if noise_update not in {"epoch", "step"}:
            raise ValueError("noise_update must be 'epoch' or 'step'.")
        if "Mformer" not in joint_way or not any(mode in joint_way for mode in ("mean", "graph")):
            raise ValueError("APKGC supports the official Mformer mean and graph fusion paths.")

        self.d = d
        self.dim_e = 2 * d
        self.joint_way = joint_way
        self.num_proj = num_proj
        self.add_noise = add_noise
        self.noise_update = noise_update

        self.register_buffer("text_feat", text_feat.detach().clone().float())
        self.register_buffer("img_feat", img_feat.detach().clone().float())
        self.register_buffer("has_img", has_img.detach().clone().to(dtype=torch.bool))
        if has_text is not None:
            if has_text.numel() != num_entities:
                raise ValueError("has_text length must equal num_entities.")
            self.register_buffer("has_text", has_text.detach().clone().bool())
        else:
            self.has_text = None
        valid_img = self.img_feat[self.has_img]
        if valid_img.numel() == 0:
            raise ValueError("APKGC requires at least one entity with an image feature.")
        self.register_buffer("img_mean", valid_img.mean(dim=0))
        self.register_buffer("img_std", valid_img.std(dim=0))
        text_stats_source = self.text_feat[self.has_text] if self.has_text is not None else self.text_feat
        if text_stats_source.numel() == 0:
            raise ValueError("APKGC requires at least one entity with a text feature.")
        self.register_buffer("text_mean", text_stats_source.mean(dim=0))
        self.register_buffer("text_std", text_stats_source.std(dim=0))

        img_filled = self.img_feat.clone()
        missing = ~self.has_img
        if bool(missing.any()):
            img_filled[missing] = self.img_mean + self.img_std * torch.randn_like(img_filled[missing])
        self.register_buffer("img_filled", img_filled)
        if self.has_text is not None:
            text_filled = self.text_feat.clone()
            text_missing = ~self.has_text
            if bool(text_missing.any()):
                text_filled[text_missing] = self.text_mean + self.text_std * torch.randn_like(text_filled[text_missing])
            self.register_buffer("text_filled", text_filled)
        else:
            self.text_filled = self.text_feat

        self.ent_embeddings = nn.Embedding(num_entities, self.dim_e)
        self.rel_embeddings = nn.Embedding(num_relations, d)
        ent_range = (margin + epsilon) / self.dim_e
        rel_range = (margin + epsilon) / d
        self.register_buffer("ent_embedding_range", torch.tensor(ent_range, dtype=torch.float32))
        self.register_buffer("rel_embedding_range", torch.tensor(rel_range, dtype=torch.float32))
        self.register_buffer("margin", torch.tensor(margin, dtype=torch.float32))
        nn.init.uniform_(self.ent_embeddings.weight, -ent_range, ent_range)
        nn.init.uniform_(self.rel_embeddings.weight, -rel_range, rel_range)

        self.text_proj = nn.Linear(self.text_feat.shape[1], self.dim_e)
        self.img_proj = nn.Linear(self.img_feat.shape[1], self.dim_e)
        if num_proj == 2:
            self.text_proj_2 = nn.Linear(self.dim_e, self.dim_e)
            self.img_proj_2 = nn.Linear(self.dim_e, self.dim_e)

        self.fusion_layer = nn.ModuleList(
            APKGCAttentionLayer(
                self.dim_e,
                num_attention_heads,
                dropout=attention_dropout,
                use_intermediate=use_intermediate,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        )
        if not self.fusion_layer:
            raise ValueError("Mformer fusion requires at least one hidden layer.")

        # The upstream implementation exposes these as trainable Parameters.
        self.mask_ratio = nn.Parameter(torch.tensor(mask_ratio, dtype=torch.float32))
        self.noise_ratio = nn.Parameter(torch.tensor(noise_ratio, dtype=torch.float32), requires_grad=False)
        self.register_buffer("adv_temperature", torch.tensor(adv_temperature, dtype=torch.float32))
        self._epoch_img_noise: torch.Tensor | None = None
        self._epoch_text_noise: torch.Tensor | None = None
        self._epoch_entity_noise: torch.Tensor | None = None
        self._epoch_entity_noise_mask: torch.Tensor | None = None

    def _add_noise_to_embeddings(self, embeddings: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        noisy = embeddings.clone()
        noise_mask = torch.rand(noisy.shape[0], device=noisy.device) < self.noise_ratio
        if bool(noise_mask.any()):
            selected = noisy[noise_mask]
            sampled = mean + std * torch.randn_like(selected)
            noisy[noise_mask] = (1.0 - self.mask_ratio) * selected + self.mask_ratio * sampled
        return noisy

    @torch.no_grad()
    def update_noise(self) -> None:
        """Create graph-free epoch noise snapshots while the model is training.

        The upstream APKGC implementation constructs these tensors from
        ``.data``.  They are stochastic inputs fixed for the epoch, not model
        parameters; retaining their graph would make the second batch attempt
        to backpropagate through the first batch's freed graph.
        """
        self._epoch_img_noise = self._add_noise_to_embeddings(self.img_filled, self.img_mean, self.img_std)
        self._epoch_text_noise = self._add_noise_to_embeddings(self.text_filled, self.text_mean, self.text_std)
        entity_weights = self.ent_embeddings.weight
        entity_mean = entity_weights.detach().mean(dim=0)
        entity_std = entity_weights.detach().std(dim=0)
        self._epoch_entity_noise = entity_mean + entity_std * torch.randn_like(entity_weights)
        self._epoch_entity_noise_mask = torch.rand(entity_weights.shape[0], device=entity_weights.device) < self.noise_ratio

    def on_epoch_start(self, epoch: int) -> None:
        if self.training and self.add_noise and self.noise_update == "epoch":
            self.update_noise()

    def _entity_modalities(self, entity_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        structural = self.ent_embeddings(entity_ids)
        image = self.img_filled[entity_ids]
        text = self.text_filled[entity_ids]
        if not (self.training and self.add_noise):
            return structural, image, text

        if self.noise_update == "epoch":
            if self._epoch_img_noise is None:
                self.update_noise()
            assert self._epoch_text_noise is not None
            assert self._epoch_entity_noise is not None
            assert self._epoch_entity_noise_mask is not None
            image = self._epoch_img_noise[entity_ids]
            text = self._epoch_text_noise[entity_ids]
            entity_mask = self._epoch_entity_noise_mask[entity_ids]
            structural = structural.clone()
            structural[entity_mask] = (
                (1.0 - self.mask_ratio) * structural[entity_mask]
                + self.mask_ratio * self._epoch_entity_noise[entity_ids][entity_mask]
            )
            return structural, image, text

        entity_mean = self.ent_embeddings.weight.detach().mean(dim=0)
        entity_std = self.ent_embeddings.weight.detach().std(dim=0)
        return (
            self._add_noise_to_embeddings(structural, entity_mean, entity_std),
            self._add_noise_to_embeddings(image, self.img_mean, self.img_std),
            self._add_noise_to_embeddings(text, self.text_mean, self.text_std),
        )

    def _project_modalities(self, image: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.img_proj(image)
        text = self.text_proj(text)
        if self.num_proj == 2:
            image = self.img_proj_2(image)
            text = self.text_proj_2(text)
        return image, text

    def get_joint_embeddings(self, structural: torch.Tensor, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.tanh(torch.stack((structural, image, text), dim=1))
        for layer in self.fusion_layer:
            hidden_states, _ = layer(hidden_states)
        if "graph" in self.joint_way:
            return hidden_states[:, 0, :]
        return hidden_states.mean(dim=1)

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        if triples.ndim != 2 or triples.shape[-1] != 3:
            raise ValueError(f"Expected triples with shape [batch, 3], got {tuple(triples.shape)}.")
        h_ids, r_ids, t_ids = triples.unbind(dim=1)
        h_structural, h_image, h_text = self._entity_modalities(h_ids)
        t_structural, t_image, t_text = self._entity_modalities(t_ids)
        h_image, h_text = self._project_modalities(h_image, h_text)
        t_image, t_text = self._project_modalities(t_image, t_text)
        h_joint = self.get_joint_embeddings(h_structural, h_image, h_text)
        t_joint = self.get_joint_embeddings(t_structural, t_image, t_text)
        relation = self.rel_embeddings(r_ids)
        return rotate_score(h_joint, relation, t_joint, self.margin, self.rel_embedding_range)

    def forward(self, pos: torch.LongTensor, neg: torch.LongTensor) -> torch.Tensor:
        pos_score = self.score(pos)
        neg_score = self.score(neg)
        if neg_score.numel() % pos_score.numel() != 0:
            raise ValueError("Negative score count must be an integer multiple of positive score count.")
        neg_score = neg_score.view(pos_score.numel(), -1)
        weights = F.softmax(neg_score * self.adv_temperature, dim=-1).detach()
        return -(F.logsigmoid(pos_score).mean() + (weights * F.logsigmoid(-neg_score)).sum(dim=-1).mean()) / 2
