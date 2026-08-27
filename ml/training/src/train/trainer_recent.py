"""Training-engine adapters for recent multimodal-KGC baselines.

These adapters reuse the established artifact, dev-checkpoint, and evaluation
path from :class:`TrainerYAML`, but make each non-standard loss contract
explicit.  Concrete baselines are added in later milestones.
"""

from __future__ import annotations

import torch

from ml.training.src.data.sampler_recent import (
    bernoulli_filtered_negative_sample,
    build_relation_statistics,
)
from ml.training.src.data.build_true_facts import build_true_facts
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
                "No concrete recent baseline is implemented in M1.1."
            )
        if neg is None:
            raise RuntimeError(f"{self.__class__.__name__} requires sampled negative triples.")
        return loss_method(pos, neg)


class AdversarialTrainer(RecentTrainerBase):
    """Engine for models implementing ``adversarial_loss(pos, neg)``."""

    loss_method_name = "adversarial_loss"


class AdversarialGPTrainer(RecentTrainerBase):
    """Engine for models implementing ``adversarial_gp_loss(pos, neg)``."""

    loss_method_name = "adversarial_gp_loss"


class OneVsAllTrainer(TrainerYAML):
    """Engine for models implementing ``one_vs_all_loss(pos)``.

    One-vs-all objectives construct their complete target vector internally;
    therefore this engine intentionally never calls either negative sampler.
    """

    def _sample_negatives(self, pos: torch.LongTensor) -> None:
        return None

    def _compute_loss(self, pos: torch.LongTensor, neg: torch.LongTensor | None) -> torch.Tensor:
        loss_method = getattr(self.model, "one_vs_all_loss", None)
        if loss_method is None:
            raise TypeError(
                "OneVsAllTrainer requires model.one_vs_all_loss(pos). "
                "No concrete recent baseline is implemented in M1.1."
            )
        return loss_method(pos)
