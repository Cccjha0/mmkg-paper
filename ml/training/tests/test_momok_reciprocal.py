import torch

from ml.training.src.models.recent_baselines.momok import ReciprocalHeadScoringMixin, reciprocal_head_triples


class ToyReciprocalModel(ReciprocalHeadScoringMixin):
    def __init__(self):
        self.inverse_relation_ids = torch.tensor([3, 2, 1, 0], dtype=torch.long)
        self.tail_input = None

    def score_tail(self, triples):
        self.tail_input = triples.detach().clone()
        return triples[:, 1].float()


def test_reciprocal_head_triple_conversion():
    triples = torch.tensor([[4, 0, 5], [6, 2, 7]], dtype=torch.long)
    inverse_relation_ids = torch.tensor([3, 2, 1, 0], dtype=torch.long)
    actual = reciprocal_head_triples(triples, inverse_relation_ids)
    expected = torch.tensor([[5, 3, 4], [7, 1, 6]], dtype=torch.long)
    assert torch.equal(actual, expected)


def test_momok_head_scoring_delegates_to_reciprocal_tail_scoring():
    model = ToyReciprocalModel()
    triples = torch.tensor([[4, 0, 5]], dtype=torch.long)
    score = model.score_head(triples)
    assert torch.equal(model.tail_input, torch.tensor([[5, 3, 4]], dtype=torch.long))
    assert score.item() == 3.0
