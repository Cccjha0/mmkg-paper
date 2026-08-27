import torch

from ml.training.src.data.sampler_recent import (
    bernoulli_filtered_negative_sample,
    build_relation_statistics,
)
from ml.training.src.eval.filtered_ranking import filtered_ranking_eval


class RecordingDirectionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tail_calls = []
        self.head_calls = []

    @staticmethod
    def _score(triples):
        return (100 * triples[:, 0] + 10 * triples[:, 1] + triples[:, 2]).float()

    def score_tail(self, triples):
        self.tail_calls.append((triples.detach().clone(), self._score(triples).detach().clone()))
        return self._score(triples)

    def score_head(self, triples):
        self.head_calls.append((triples.detach().clone(), self._score(triples).detach().clone()))
        return self._score(triples)


def _score_for_exact_triple(calls, triple):
    for triples, scores in calls[1:]:
        matches = (triples == triple).all(dim=1)
        if matches.any():
            return scores[matches][0]
    raise AssertionError("Candidate batch did not contain the target triple.")


def test_directional_target_and_candidate_scoring_are_consistent():
    triple = torch.tensor([1, 0, 2], dtype=torch.long)
    model = RecordingDirectionalModel()
    filtered_ranking_eval(
        model=model,
        triples=triple.unsqueeze(0),
        true_tails={},
        true_heads={},
        num_entities=4,
        device="cpu",
        direction="both",
    )

    assert torch.equal(model.tail_calls[0][0][0], triple)
    assert torch.equal(model.head_calls[0][0][0], triple)
    assert model.tail_calls[0][1][0] == _score_for_exact_triple(model.tail_calls, triple)
    assert model.head_calls[0][1][0] == _score_for_exact_triple(model.head_calls, triple)


def test_recent_filtered_sampler_never_returns_a_known_true_fact():
    positives = torch.tensor([[0, 0, 1]], dtype=torch.long)
    stats = build_relation_statistics([(0, 0, 1), (2, 0, 1)])
    negatives = bernoulli_filtered_negative_sample(
        pos=positives,
        num_entities=4,
        true_heads={(0, 1): {0, 2}},
        true_tails={(0, 0): {1, 3}},
        relation_stats=stats,
        neg_ratio=64,
    )

    for h, r, t in negatives.tolist():
        assert h not in {0, 2} or (r, t) != (0, 1)
        assert t not in {1, 3} or (h, r) != (0, 0)
