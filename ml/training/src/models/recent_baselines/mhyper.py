"""OpenBG adapter for the released M-Hyper ``M_Hyper_B`` model.

The computational path follows ``external/M-Hyper/src/models.py`` while the
surrounding data, reciprocal evaluation, and Dev-only selection contracts are
provided by this repository.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ml.training.src.models.recent_baselines.reciprocal import (
    ReciprocalHeadScoringMixin,
    augment_with_reciprocals,
    build_inverse_relation_ids,
)


def complex_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Released M_Hyper_B complex product (higher-level quaternion path is unused)."""
    if left.shape[-1] != right.shape[-1] or left.shape[-1] % 2:
        raise ValueError("complex_mul requires equal, even final dimensions.")
    left_real, left_imag = left.chunk(2, dim=-1)
    right_real, right_imag = right.chunk(2, dim=-1)
    return torch.cat(
        (
            left_real * right_real - left_imag * right_imag,
            left_real * right_imag + left_imag * right_real,
        ),
        dim=-1,
    )


def split_component_norm(value: torch.Tensor, num_components: int) -> torch.Tensor:
    """Match upstream ``get_norm`` over equal contiguous components."""
    if value.shape[-1] % num_components:
        raise ValueError("The final dimension must be divisible by num_components.")
    squared = torch.stack([chunk.square() for chunk in value.chunk(num_components, dim=-1)], dim=0)
    return squared.sum(dim=0).sqrt()


class FineGrainedEntityRepresentationFactorization(nn.Module):
    """Faithful FERF module from the released M_Hyper_B path."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.fc_s = nn.Linear(rank * 6, rank * 2)
        self.fc_v = nn.Linear(rank * 6, rank * 2)
        self.fc_t = nn.Linear(rank * 6, rank * 2)

    def forward(
        self,
        structure_independent: torch.Tensor,
        structure_projected: torch.Tensor,
        image_independent: torch.Tensor,
        image_projected: torch.Tensor,
        text_independent: torch.Tensor,
        text_projected: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        recon_structure = self.fc_s(
            torch.cat((structure_independent, image_projected, text_projected), dim=-1)
        )
        recon_image = self.fc_v(
            torch.cat((structure_projected, image_independent, text_projected), dim=-1)
        )
        recon_text = self.fc_t(
            torch.cat((structure_projected, image_projected, text_independent), dim=-1)
        )
        return (
            structure_projected + structure_independent,
            image_projected + image_independent,
            text_projected + text_independent,
            recon_structure,
            recon_image,
            recon_text,
        )


class RelationAwareModalityFusion(nn.Module):
    """Released relation-specific temperature fusion (R2MF)."""

    def __init__(self, rank: int, num_relations: int) -> None:
        super().__init__()
        self.fc_s = nn.Linear(rank * 18, 1)
        self.fc_v = nn.Linear(rank * 18, 1)
        self.fc_t = nn.Linear(rank * 18, 1)
        self.ids_r = nn.Parameter(torch.ones(num_relations))

    def forward(
        self,
        structure: torch.Tensor,
        image: torch.Tensor,
        text: torch.Tensor,
        relation: torch.Tensor,
        relation_ids: torch.LongTensor,
    ) -> torch.Tensor:
        scores = torch.cat(
            (
                self.fc_s(torch.cat((structure, relation), dim=-1)),
                self.fc_v(torch.cat((image, relation), dim=-1)),
                self.fc_t(torch.cat((text, relation), dim=-1)),
            ),
            dim=-1,
        )
        temperature = self.ids_r[relation_ids].view(-1, 1)
        weights = F.softmax(scores / temperature, dim=-1)
        return weights[:, 0:1] * structure + weights[:, 1:2] * image + weights[:, 2:3] * text


class SparseGaussianNoise(nn.Module):
    """Generate Gaussian noise on the released fixed fraction of entity rows."""

    def __init__(self, preserve_ratio: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= preserve_ratio <= 1.0:
            raise ValueError("preserve_ratio must be in [0, 1].")
        self.preserve_ratio = float(preserve_ratio)

    def forward(self, num_entities: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        num_preserve = int(num_entities * self.preserve_ratio)
        output = torch.zeros((num_entities, mean.shape[-1]), dtype=mean.dtype, device=mean.device)
        if num_preserve == 0:
            return output
        selected = torch.randperm(num_entities, device=mean.device)[:num_preserve]
        sampled = torch.normal(
            mean=mean.expand(num_preserve, -1),
            std=std.expand(num_preserve, -1),
        )
        output[selected] = sampled
        return output


class OpenBGMHyper(ReciprocalHeadScoringMixin, nn.Module):
    """M-Hyper adapted to fixed OpenBG raw features and unified filtered eval."""

    def __init__(
        self,
        *,
        text_feat: torch.Tensor,
        img_feat: torch.Tensor,
        has_img: torch.Tensor,
        has_text: torch.Tensor | None = None,
        num_entities: int,
        num_relations: int,
        rank: int = 128,
        init_size: float = 1e-3,
        noise_preserve_ratio: float = 0.2,
        wn3_weight: float = 5e-3,
        pca_init: bool = True,
        pca_fit_scope: str = "train_entities",
        pca_random_state: int | None = None,
        faithful_upstream_reconstruction: bool = True,
    ) -> None:
        super().__init__()
        if text_feat.ndim != 2 or img_feat.ndim != 2:
            raise ValueError("text_feat and img_feat must be two-dimensional.")
        if text_feat.shape[0] != num_entities or img_feat.shape[0] != num_entities:
            raise ValueError("Feature row counts must equal num_entities.")
        if has_img.numel() != num_entities:
            raise ValueError("has_img must contain one indicator per entity.")
        if pca_fit_scope != "train_entities":
            raise ValueError("M-Hyper protocol requires pca_fit_scope='train_entities'.")
        if not faithful_upstream_reconstruction:
            raise ValueError("The formal adapter requires faithful_upstream_reconstruction=true.")

        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.rank = int(rank)
        self.init_size = float(init_size)
        self.wn3_weight = float(wn3_weight)
        self.pca_init = bool(pca_init)
        self.pca_fit_scope = pca_fit_scope
        self.pca_random_state = pca_random_state
        self.faithful_upstream_reconstruction = bool(faithful_upstream_reconstruction)

        # Keep the shared raw cache unchanged. Upstream applies init_size inside
        # the model, so that scaling is performed lazily by _scaled_features().
        self.register_buffer("text_feat", text_feat.detach().float().clone())
        self.register_buffer("img_feat", img_feat.detach().float().clone())
        self.register_buffer("has_img", has_img.detach().bool().clone())
        if has_text is not None:
            if has_text.numel() != num_entities:
                raise ValueError("has_text must contain one indicator per entity.")
            self.register_buffer("has_text", has_text.detach().bool().clone())
        else:
            self.has_text = None
        self.register_buffer("inverse_relation_ids", build_inverse_relation_ids(num_relations))
        # Audit-only runtime metadata. It is deliberately excluded from the
        # checkpoint because its dynamic shape would prevent strict reloads.
        self.register_buffer("pca_fit_entity_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("pca_fit_image_entity_ids", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("pca_fit_text_entity_ids", torch.empty(0, dtype=torch.long), persistent=False)

        self.all = nn.Embedding(num_entities, 2 * rank, sparse=True)
        self.structure = nn.Embedding(num_entities, 2 * rank, sparse=True)
        self.stru = nn.Embedding(num_entities, 2 * rank, sparse=True)
        self.img = nn.Embedding(num_entities, 2 * rank, sparse=True)
        self.text = nn.Embedding(num_entities, 2 * rank, sparse=True)
        self.rel_embedding = nn.Embedding(2 * num_relations, 16 * rank, sparse=True)
        for embedding in (self.all, self.structure, self.stru, self.img, self.text, self.rel_embedding):
            embedding.weight.data.mul_(init_size)

        self.img_proj = nn.Linear(img_feat.shape[1], 2 * rank)
        self.text_proj = nn.Linear(text_feat.shape[1], 2 * rank)
        self.ferf = FineGrainedEntityRepresentationFactorization(rank)
        self.rel_fusion = RelationAwareModalityFusion(rank, 2 * num_relations)
        self.sparse_noise = SparseGaussianNoise(noise_preserve_ratio)

        # Preserve upstream statistics exactly, including the additional
        # init_size scaling applied to the structural noise statistics.
        self.register_buffer("stru_mean", self.stru.weight.detach().mean(dim=0, keepdim=True) * init_size)
        self.register_buffer("stru_std", self.stru.weight.detach().std(dim=0, keepdim=True) * init_size)
        scaled_img, scaled_text = self._scaled_features()
        if self.has_text is not None:
            valid_img = scaled_img[self.has_img]
            valid_text = scaled_text[self.has_text]
            if valid_img.shape[0] < 2 or valid_text.shape[0] < 2:
                raise ValueError("General M-Hyper requires at least two observed entities per modality.")
        else:
            valid_img, valid_text = scaled_img, scaled_text
        self.register_buffer("img_mean", valid_img.mean(dim=0, keepdim=True))
        self.register_buffer("img_std", valid_img.std(dim=0, keepdim=True))
        self.register_buffer("text_mean", valid_text.mean(dim=0, keepdim=True))
        self.register_buffer("text_std", valid_text.std(dim=0, keepdim=True))

        self._training_prepared = not self.pca_init
        self._eval_cache: dict[str, torch.Tensor] | None = None

    def _scaled_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.img_feat * self.init_size, self.text_feat * self.init_size

    def _apply_general_masks(
        self, projected_image: torch.Tensor, projected_text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.has_text is None:
            return projected_image, projected_text
        return (
            projected_image * self.has_img.unsqueeze(-1),
            projected_text * self.has_text.unsqueeze(-1),
        )

    def _independent_modalities(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.has_text is None:
            return self.img.weight, self.text.weight
        return (
            self.img.weight * self.has_img.unsqueeze(-1),
            self.text.weight * self.has_text.unsqueeze(-1),
        )

    @torch.no_grad()
    def prepare_training(self, train_triples: Sequence[Sequence[int]] | torch.LongTensor) -> None:
        """Fit upstream PCA initializers using train-visible entities only."""
        rows = (
            train_triples.detach().cpu().to(dtype=torch.long)
            if isinstance(train_triples, torch.Tensor)
            else torch.tensor(train_triples, dtype=torch.long)
        )
        if rows.ndim != 2 or rows.shape[1] != 3:
            raise ValueError("train_triples must have shape [N, 3].")
        fit_ids = torch.unique(torch.cat((rows[:, 0], rows[:, 2]), dim=0), sorted=True)
        self.pca_fit_entity_ids = fit_ids.to(device=self.pca_fit_entity_ids.device)
        if self.has_text is None:
            image_fit_ids = fit_ids
            text_fit_ids = fit_ids
        else:
            has_img_cpu = self.has_img.detach().cpu()
            has_text_cpu = self.has_text.detach().cpu()
            image_fit_ids = fit_ids[has_img_cpu[fit_ids]]
            text_fit_ids = fit_ids[has_text_cpu[fit_ids]]
        self.pca_fit_image_entity_ids = image_fit_ids.to(device=self.pca_fit_image_entity_ids.device)
        self.pca_fit_text_entity_ids = text_fit_ids.to(device=self.pca_fit_text_entity_ids.device)
        if not self.pca_init:
            self._training_prepared = True
            return

        from sklearn.decomposition import PCA

        scaled_img, scaled_text = self._scaled_features()

        def fit_and_transform(features: torch.Tensor, modality_fit_ids: torch.LongTensor, modality: str) -> torch.Tensor:
            values = features.detach().cpu().numpy()
            fit_ids_cpu = modality_fit_ids.cpu()
            components = 2 * self.rank
            if fit_ids_cpu.numel() < components or values.shape[1] < components:
                raise ValueError(
                    f"Train-visible {modality} PCA needs at least {components} observed entities "
                    f"and feature dimensions; got entities={fit_ids_cpu.numel()}, dim={values.shape[1]}."
                )
            pca = PCA(n_components=2 * self.rank, random_state=self.pca_random_state)
            pca.fit(values[fit_ids_cpu.numpy()])
            transformed = pca.transform(values)
            return torch.from_numpy(transformed).to(dtype=torch.float32)

        # Upstream multiplies the PCA output by init_size once more.
        img_reduced = fit_and_transform(scaled_img, image_fit_ids, "image") * self.init_size
        text_reduced = fit_and_transform(scaled_text, text_fit_ids, "text") * self.init_size
        self.img.weight.copy_(img_reduced.to(device=self.img.weight.device))
        self.text.weight.copy_(text_reduced.to(device=self.text.weight.device))
        self._training_prepared = True
        self._eval_cache = None

    def augment_train_triples(
        self,
        train_triples: Sequence[Sequence[int]] | torch.LongTensor,
    ) -> list[tuple[int, int, int]]:
        return augment_with_reciprocals(train_triples, self.num_relations)

    @staticmethod
    def _independent_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        positive = F.cosine_similarity(left, right, dim=1)
        shuffled = right[torch.randperm(right.shape[0], device=right.device)]
        negative = F.cosine_similarity(left, shuffled, dim=1)
        return negative.mean() - positive.mean()

    def _clean_modalities(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        scaled_img, scaled_text = self._scaled_features()
        projected_structure = self.structure.weight
        projected_image = self.img_proj(scaled_img)
        projected_text = self.text_proj(scaled_text)
        projected_image, projected_text = self._apply_general_masks(projected_image, projected_text)
        independent_image, independent_text = self._independent_modalities()
        output = self.ferf(
            self.stru.weight,
            projected_structure,
            independent_image,
            projected_image,
            independent_text,
            projected_text,
        )
        return output[0], output[1], output[2], (
            projected_structure,
            projected_image,
            projected_text,
            output[3],
            output[4],
            output[5],
        )

    def _noisy_modalities(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        scaled_img, scaled_text = self._scaled_features()
        noisy_structure = self.structure.weight + self.sparse_noise(
            self.num_entities, self.stru_mean, self.stru_std
        )
        noisy_img_raw = scaled_img + self.sparse_noise(self.num_entities, self.img_mean, self.img_std)
        noisy_text_raw = scaled_text + self.sparse_noise(self.num_entities, self.text_mean, self.text_std)
        projected_image = self.img_proj(noisy_img_raw)
        projected_text = self.text_proj(noisy_text_raw)
        projected_image, projected_text = self._apply_general_masks(projected_image, projected_text)
        independent_image, independent_text = self._independent_modalities()
        output = self.ferf(
            self.stru.weight,
            noisy_structure,
            independent_image,
            projected_image,
            independent_text,
            projected_text,
        )
        return output[0], output[1], output[2], (
            noisy_structure,
            projected_image,
            projected_text,
            output[3],
            output[4],
            output[5],
        )

    def _entity_embedding(
        self,
        structure: torch.Tensor,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((self.all.weight, structure, image, text), dim=-1)

    def _queries(
        self,
        triples: torch.LongTensor,
        structure: torch.Tensor,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_ids, relation_ids, _ = triples.unbind(dim=1)
        relation = self.rel_embedding(relation_ids)
        fused = self.rel_fusion(
            structure[head_ids], image[head_ids], text[head_ids], relation, relation_ids
        ) + self.all.weight[head_ids]
        lhs = torch.cat((fused, structure[head_ids], image[head_ids], text[head_ids]), dim=-1)
        lhs = lhs + relation[:, self.rank * 8 :]
        query = complex_mul(lhs, relation[:, : self.rank * 8])
        return query, lhs

    def _forward_training(
        self, triples: torch.LongTensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if not self._training_prepared:
            raise RuntimeError("M-Hyper PCA is not initialized; use OneVsAllTrainer/prepare_training first.")
        noisy_structure, noisy_image, noisy_text, noisy_parts = self._noisy_modalities()
        clean_structure, clean_image, clean_text, clean_parts = self._clean_modalities()
        head_ids, relation_ids, tail_ids = triples.unbind(dim=1)
        relation = self.rel_embedding(relation_ids)

        clean_fused = self.rel_fusion(
            clean_structure[head_ids], clean_image[head_ids], clean_text[head_ids], relation, relation_ids
        ) + self.all.weight[head_ids]
        noisy_fused = self.rel_fusion(
            noisy_structure[head_ids], noisy_image[head_ids], noisy_text[head_ids], relation, relation_ids
        ) + self.all.weight[head_ids]
        consistency_loss = F.mse_loss(clean_fused, noisy_fused, reduction="mean")

        # Faithfully preserve the released M_Hyper_B reconstruction target,
        # including its final proj_no_stru item (rather than proj_no_text).
        recon_predictions = torch.cat(
            (noisy_parts[3], clean_parts[3], noisy_parts[4], clean_parts[4], noisy_parts[5], clean_parts[5]),
            dim=-1,
        )
        recon_targets = torch.cat(
            (noisy_parts[0], clean_parts[0], noisy_parts[1], clean_parts[1], noisy_parts[2], clean_parts[0]),
            dim=-1,
        )
        reconstruction_loss = F.mse_loss(recon_predictions, recon_targets, reduction="mean")

        # The released forward computes these terms but leaves them commented
        # out of the returned auxiliary loss. Keep both behaviours faithful.
        _ = self._independent_loss(noisy_parts[0], self.stru.weight) + self._independent_loss(
            clean_parts[0], self.stru.weight
        )
        _ = self._independent_loss(noisy_parts[1], self.img.weight) + self._independent_loss(
            clean_parts[1], self.img.weight
        )
        _ = self._independent_loss(noisy_parts[2], self.text.weight) + self._independent_loss(
            clean_parts[2], self.text.weight
        )

        entity_embedding = self._entity_embedding(noisy_structure, noisy_image, noisy_text)
        lhs = torch.cat(
            (noisy_fused, noisy_structure[head_ids], noisy_image[head_ids], noisy_text[head_ids]), dim=-1
        )
        lhs = lhs + relation[:, self.rank * 8 :]
        query = complex_mul(lhs, relation[:, : self.rank * 8])
        logits = query @ entity_embedding.transpose(0, 1)
        rhs = entity_embedding[tail_ids]
        factors = (
            split_component_norm(lhs, 8),
            split_component_norm(relation[:, : self.rank * 8], 8),
            split_component_norm(rhs, 8),
        )
        return logits, factors, consistency_loss + reconstruction_loss

    def one_vs_all_loss(self, triples: torch.LongTensor) -> torch.Tensor:
        logits, (head_factor, relation_factor, tail_factor), auxiliary = self._forward_training(triples)
        fit = F.cross_entropy(logits, triples[:, 2])
        wn3 = (
            2.0 * head_factor.pow(3).sum()
            + 2.0 * tail_factor.pow(3).sum()
            + 0.5 * relation_factor.pow(3).sum()
        )
        return fit + self.wn3_weight * wn3 / triples.shape[0] + auxiliary

    def _build_eval_cache(self) -> dict[str, torch.Tensor]:
        structure, image, text, _ = self._clean_modalities()
        return {
            "structure": structure,
            "image": image,
            "text": text,
            "entities": self._entity_embedding(structure, image, text),
        }

    @torch.inference_mode()
    def prepare_eval_cache(self) -> None:
        """Rebuild clean entity representations once after model.eval()/checkpoint load."""
        self._eval_cache = self._build_eval_cache()

    def _get_eval_cache(self) -> dict[str, torch.Tensor]:
        if self._eval_cache is None:
            self._eval_cache = self._build_eval_cache()
        return self._eval_cache

    def train(self, mode: bool = True):
        if mode:
            self._eval_cache = None
        return super().train(mode)

    def inference_all(self, triples: torch.LongTensor) -> torch.Tensor:
        cache = self._get_eval_cache()
        query, _ = self._queries(triples, cache["structure"], cache["image"], cache["text"])
        return query @ cache["entities"].transpose(0, 1)

    def score_tail(self, triples: torch.LongTensor) -> torch.Tensor:
        """Score exact candidates with the same clean query used by full inference."""
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("triples must have shape [N, 3].")
        cache = self._get_eval_cache()
        pairs = triples[:, :2]
        unique_pairs, inverse = torch.unique(pairs, dim=0, sorted=False, return_inverse=True)
        representative = torch.column_stack(
            (unique_pairs, torch.zeros(unique_pairs.shape[0], dtype=torch.long, device=triples.device))
        )
        unique_query, _ = self._queries(
            representative, cache["structure"], cache["image"], cache["text"]
        )
        query = unique_query[inverse]
        candidates = cache["entities"][triples[:, 2]]
        return (query * candidates).sum(dim=-1)

    def score(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.score_tail(triples)


__all__ = [
    "FineGrainedEntityRepresentationFactorization",
    "OpenBGMHyper",
    "RelationAwareModalityFusion",
    "SparseGaussianNoise",
    "complex_mul",
    "split_component_norm",
]
