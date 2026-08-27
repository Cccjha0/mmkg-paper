import torch

from ml.training.src.eval.filtered_ranking import filtered_ranking_eval


class DirectionalOnlyModel(torch.nn.Module):
    """Fails if the evaluator falls back to the wrong scoring interface."""

    def __init__(self):
        super().__init__()

    def score(self, triples):
        raise AssertionError("Directional models must not call score() during ranking.")

    def score_tail(self, triples):
        return triples[:, 2].float()

    def score_head(self, triples):
        return -triples[:, 0].float()


class LegacyScoreOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def score(self, triples):
        return triples[:, 2].float()


def test_filtered_ranking_uses_directional_score_methods_for_targets_and_candidates():
    triples = torch.tensor([[0, 0, 2]], dtype=torch.long)
    model = DirectionalOnlyModel()

    tail = filtered_ranking_eval(
        model=model,
        triples=triples,
        true_tails={},
        true_heads={},
        num_entities=3,
        device="cpu",
        direction="tail",
    )
    head = filtered_ranking_eval(
        model=model,
        triples=triples,
        true_tails={},
        true_heads={},
        num_entities=3,
        device="cpu",
        direction="head",
    )

    assert tail["mrr"] == 1.0
    assert head["mrr"] == 1.0


def test_legacy_score_only_models_remain_compatible():
    metrics = filtered_ranking_eval(
        model=LegacyScoreOnlyModel(),
        triples=torch.tensor([[0, 0, 2]], dtype=torch.long),
        true_tails={},
        true_heads={},
        num_entities=3,
        device="cpu",
        direction="tail",
    )
    assert metrics["mrr"] == 1.0
