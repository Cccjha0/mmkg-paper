"""Training-engine adapters for recent multimodal-KGC baselines.

These adapters reuse the established artifact, dev-checkpoint, and evaluation
path from :class:`TrainerYAML`, but make each non-standard loss contract
explicit.
"""

from __future__ import annotations

from __future__ import annotations

import torch
import torch.nn.functional as F

from ml.training.src.data.sampler_recent import (
    bernoulli_filtered_negative_sample,
    build_relation_statistics,
)
from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.models.recent_baselines.adversarial import (
    CombinedGenerator,
    MultiGenerator,
    generator_gradient_norm,
    gradient_penalty,
)
from ml.training.src.train.trainer_yaml import TrainerYAML


class RecentTrainerBase(TrainerYAML):
    """Base class for recent engines that train from filtered negatives."""

    loss_method_name: str = ""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sampler_name = self.cfg["training"].get("sampler", "bernoulli_filtered").lower()
        if self.sampler_name != "bernoulli_filtered":
            raise ValueError(
                f"{self.__class__.__name__} requires training.sampler='bernoulli_filtered'; "
                f"got {self.sampler_name!r}."
            )
        self.relation_stats = build_relation_statistics(self.train_triples)
        # Training-time filtering must not consume dev/test labels.  The
        # all-split truth maps retained by TrainerYAML remain evaluation-only.
        self.train_true_tails, self.train_true_heads = build_true_facts(self.train_triples)

    def _sample_negatives(self, pos: torch.LongTensor) -> torch.LongTensor:
        return bernoulli_filtered_negative_sample(
            pos=pos,
            num_entities=self.num_entities,
            true_heads=self.train_true_heads,
            true_tails=self.train_true_tails,
            relation_stats=self.relation_stats,
            neg_ratio=self.neg_ratio,
        )

    def _compute_loss(self, pos: torch.LongTensor, neg: torch.LongTensor | None) -> torch.Tensor:
        loss_method = getattr(self.model, self.loss_method_name, None)
        if loss_method is None:
            raise TypeError(
                f"{self.__class__.__name__} requires model.{self.loss_method_name}(pos, neg). "
                "The selected model does not implement this engine contract."
            )
        if neg is None:
            raise RuntimeError(f"{self.__class__.__name__} requires sampled negative triples.")
        return loss_method(pos, neg)


class AdversarialTrainer(RecentTrainerBase):
    """AdaMF-MAT's two-optimizer modality-adversarial training engine."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self.model, "dim_e"):
            raise TypeError(f"{self.__class__.__name__} requires a model exposing dim_e.")
        tr = self.cfg["training"]
        self.generator = self._build_generator(tr).to(self.device)
        self.generator_lr = float(tr.get("generator_lr", 1e-4))
        self.generator_optim = torch.optim.Adam(self.generator.parameters(), lr=self.generator_lr)
        self.mu = float(tr.get("mu", 0.0))
        self.adv_temperature = float(tr.get("adv_temperature", 2.0))
        self.regularization_weight = float(tr.get("regularization_weight", 0.0))

    def _build_generator(self, training_cfg: dict) -> torch.nn.Module:
        return MultiGenerator(
            noise_dim=int(training_cfg.get("generator_noise_dim", 64)),
            structure_dim=int(self.model.dim_e),
            modality_dim=int(self.model.dim_e),
            hidden_dim=int(training_cfg.get("generator_hidden_dim", 512)),
        )

    def _self_adversarial_sigmoid_loss(
        self,
        positive_score: torch.Tensor,
        negative_score: torch.Tensor,
    ) -> torch.Tensor:
        if negative_score.numel() % positive_score.numel() != 0:
            raise ValueError("Negative score count must be an integer multiple of positive score count.")
        negative_score = negative_score.view(positive_score.numel(), -1)
        weights = F.softmax(negative_score * self.adv_temperature, dim=-1).detach()
        return -(
            F.logsigmoid(positive_score).mean()
            + (weights * F.logsigmoid(-negative_score)).sum(dim=-1).mean()
        ) / 2.0

    def _official_adv_mix_sigmoid_loss(
        self,
        positive_score: torch.Tensor,
        negative_score: torch.Tensor,
    ) -> torch.Tensor:
        """Port SigmoidLoss as used on real/fake scores by AdvMixTrainer.

        The official code applies its adversarial softmax across the complete
        positive batch for each fake-score vector, rather than treating each
        real/fake pair as an independent one-negative group.
        """
        weights = F.softmax(negative_score * self.adv_temperature, dim=-1).detach()
        return -(
            F.logsigmoid(positive_score).mean()
            + (weights * F.logsigmoid(-negative_score)).sum(dim=-1).mean()
        ) / 2.0

    @staticmethod
    def _set_model_requires_grad(model: torch.nn.Module, enabled: bool) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(enabled)

    @torch.no_grad()
    def _detached_generated_modalities(
        self,
        pos: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_ids, _, t_ids = pos.unbind(dim=1)
        h_structural = self.model.get_batch_ent_embs(h_ids)
        t_structural = self.model.get_batch_ent_embs(t_ids)
        fake_h_visual, fake_h_text = self.generator(h_structural)
        fake_t_visual, fake_t_text = self.generator(t_structural)
        return fake_h_visual, fake_t_visual, fake_h_text, fake_t_text

    def _generator_structures(
        self,
        pos: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_ids, _, t_ids = pos.unbind(dim=1)
        with torch.no_grad():
            head = self.model.get_batch_ent_embs(h_ids).detach()
            tail = self.model.get_batch_ent_embs(t_ids).detach()
        return head, tail

    def _train_step(self, pos: torch.LongTensor, neg: torch.LongTensor | None) -> dict[str, float]:
        if neg is None:
            raise RuntimeError("AdversarialTrainer requires negative triples.")

        # Model/discriminator step: ordinary KGC loss plus AdaMF-MAT's three
        # real-vs-fake SigmoidLoss terms.
        self.optim.zero_grad(set_to_none=True)
        positive_score = self.model.score(pos)
        negative_score = self.model.score(neg)
        kgc_loss = self._self_adversarial_sigmoid_loss(positive_score, negative_score)
        if self.regularization_weight:
            kgc_loss = kgc_loss + self.regularization_weight * self.model.regularization(pos)

        fake_modalities = self._detached_generated_modalities(pos)
        fake_scores = self.model.fake_scores(
            pos,
            fake_head_visual=fake_modalities[0],
            fake_tail_visual=fake_modalities[1],
            fake_head_text=fake_modalities[2],
            fake_tail_text=fake_modalities[3],
        )
        adversarial_loss = sum(
            self._official_adv_mix_sigmoid_loss(positive_score, fake_score)
            for fake_score in fake_scores
        )
        model_loss = kgc_loss + self.mu * adversarial_loss
        model_loss.backward()
        grad_stats = self._compute_grad_group_stats()
        self.optim.step()

        # Generator step: generated scores are treated as positives and the
        # real positive-triple score as the negative target, matching upstream.
        self.generator.train()
        self.generator_optim.zero_grad(set_to_none=True)
        h_structural, t_structural = self._generator_structures(pos)
        fake_h_visual, fake_h_text = self.generator(h_structural)
        fake_t_visual, fake_t_text = self.generator(t_structural)
        self._set_model_requires_grad(self.model, False)
        try:
            generator_scores = self.model.fake_scores(
                pos,
                fake_head_visual=fake_h_visual,
                fake_tail_visual=fake_t_visual,
                fake_head_text=fake_h_text,
                fake_tail_text=fake_t_text,
            )
            generator_loss = sum(
                self._official_adv_mix_sigmoid_loss(fake_score, positive_score.detach())
                for fake_score in generator_scores
            )
            generator_loss.backward()
        finally:
            self._set_model_requires_grad(self.model, True)
        generator_grad = generator_gradient_norm(self.generator)
        self.generator_optim.step()

        return {
            "loss": float(model_loss.detach().item()),
            "kgc_loss": float(kgc_loss.detach().item()),
            "adversarial_loss": float(adversarial_loss.detach().item()),
            "generator_loss": float(generator_loss.detach().item()),
            "generator_grad_norm": generator_grad,
            **grad_stats,
        }


class AdversarialGPTrainer(AdversarialTrainer):
    """NativE-style model/discriminator plus generator training engine."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        tr = self.cfg["training"]
        self.gradient_penalty_coefficient = float(tr.get("gradient_penalty_coefficient", 0.1))

    def _build_generator(self, training_cfg: dict) -> torch.nn.Module:
        return CombinedGenerator(
            noise_dim=int(training_cfg.get("generator_noise_dim", 64)),
            structure_dim=int(self.model.dim_e),
            modality_dim=int(self.model.dim_e),
            hidden_dim=int(training_cfg.get("generator_hidden_dim", 512)),
        )

    @torch.no_grad()
    def _detached_generated_modalities(
        self,
        pos: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_ids, _, t_ids = pos.unbind(dim=1)
        h_structural, h_visual, h_text = self.model.get_batch_ent_multimodal_embs(h_ids)
        t_structural, t_visual, t_text = self.model.get_batch_ent_multimodal_embs(t_ids)
        fake_h_visual, fake_h_text = self.generator(h_structural, h_visual, h_text)
        fake_t_visual, fake_t_text = self.generator(t_structural, t_visual, t_text)
        return fake_h_visual, fake_t_visual, fake_h_text, fake_t_text

    def _generator_inputs(
        self,
        pos: torch.LongTensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h_ids, _, t_ids = pos.unbind(dim=1)
        with torch.no_grad():
            head = tuple(value.detach() for value in self.model.get_batch_ent_multimodal_embs(h_ids))
            tail = tuple(value.detach() for value in self.model.get_batch_ent_multimodal_embs(t_ids))
        return head, tail

    def _train_step(self, pos: torch.LongTensor, neg: torch.LongTensor | None) -> dict[str, float]:
        if neg is None:
            raise RuntimeError("AdversarialGPTrainer requires negative triples.")

        # Model/discriminator step: KGC + real/fake Wasserstein term + GP.
        self.optim.zero_grad(set_to_none=True)
        positive_score, real_embeddings = self.model.score_and_embeddings(pos)
        negative_score = self.model.score(neg)
        kgc_loss = self._self_adversarial_sigmoid_loss(positive_score, negative_score)
        if self.regularization_weight:
            kgc_loss = kgc_loss + self.regularization_weight * self.model.regularization(pos)

        fake_modalities = self._detached_generated_modalities(pos)
        fake_scores, fake_embeddings = self.model.fake_scores_and_embeddings(
            pos,
            fake_head_visual=fake_modalities[0],
            fake_tail_visual=fake_modalities[1],
            fake_head_text=fake_modalities[2],
            fake_tail_text=fake_modalities[3],
        )
        adversarial_loss = sum(
            -positive_score.mean() + fake_score.mean()
            for fake_score in fake_scores
        )
        gp = gradient_penalty(
            self.model.score_from_embeddings,
            real_embeddings,
            fake_embeddings,
            coefficient=self.gradient_penalty_coefficient,
        )
        if not bool(torch.isfinite(gp)):
            raise FloatingPointError("NativE gradient penalty became non-finite.")
        model_loss = kgc_loss + self.mu * (adversarial_loss + gp)
        model_loss.backward()
        grad_stats = self._compute_grad_group_stats()
        self.optim.step()

        # Generator step: freeze KGC parameters but retain gradients through
        # the KGC operations into generated modality embeddings.
        self.generator.train()
        self.generator_optim.zero_grad(set_to_none=True)
        head_inputs, tail_inputs = self._generator_inputs(pos)
        fake_h_visual, fake_h_text = self.generator(*head_inputs)
        fake_t_visual, fake_t_text = self.generator(*tail_inputs)
        self._set_model_requires_grad(self.model, False)
        try:
            generator_scores, _ = self.model.fake_scores_and_embeddings(
                pos,
                fake_head_visual=fake_h_visual,
                fake_tail_visual=fake_t_visual,
                fake_head_text=fake_h_text,
                fake_tail_text=fake_t_text,
            )
            generator_loss = sum(
                (self.model.margin - score).mean()
                for score in generator_scores
            ) / len(generator_scores)
            generator_loss.backward()
        finally:
            self._set_model_requires_grad(self.model, True)
        generator_grad = generator_gradient_norm(self.generator)
        self.generator_optim.step()

        return {
            "loss": float(model_loss.detach().item()),
            "kgc_loss": float(kgc_loss.detach().item()),
            "adversarial_loss": float(adversarial_loss.detach().item()),
            "generator_loss": float(generator_loss.detach().item()),
            "gradient_penalty": float(gp.detach().item()),
            "generator_grad_norm": generator_grad,
            **grad_stats,
        }


class OneVsAllTrainer(TrainerYAML):
    """Engine for models implementing ``one_vs_all_loss(pos)``.

    One-vs-all objectives construct their complete target vector internally;
    therefore this engine intentionally never calls either negative sampler.
    """

    def __init__(
        self,
        model,
        train_triples,
        dev_triples,
        test_triples,
        num_entities,
        true_tails,
        true_heads,
        cfg: dict,
    ) -> None:
        prepare_training = getattr(model, "prepare_training", None)
        if prepare_training is not None:
            # PCA/other data-dependent initialization may inspect training
            # entities only; it runs before reciprocal triples are appended.
            prepare_training(train_triples)
        augment = getattr(model, "augment_train_triples", None)
        if augment is not None:
            train_triples = augment(train_triples)
        super().__init__(
            model=model,
            train_triples=train_triples,
            dev_triples=dev_triples,
            test_triples=test_triples,
            num_entities=num_entities,
            true_tails=true_tails,
            true_heads=true_heads,
            cfg=cfg,
        )

    def _sample_negatives(self, pos: torch.LongTensor) -> None:
        return None

    def _compute_loss(self, pos: torch.LongTensor, neg: torch.LongTensor | None) -> torch.Tensor:
        loss_method = getattr(self.model, "one_vs_all_loss", None)
        if loss_method is None:
            raise TypeError(
                "OneVsAllTrainer requires model.one_vs_all_loss(pos). "
                "The selected model does not implement this engine contract."
            )
        return loss_method(pos)
