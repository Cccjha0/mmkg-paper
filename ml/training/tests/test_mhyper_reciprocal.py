import torch

from ml.training.src.models.recent_baselines.reciprocal import (
    augment_with_reciprocals,
    build_inverse_relation_ids,
    reciprocal_head_triples,
)


def test_mhyper_reciprocal_training_augmentation_and_head_conversion():
    triples = [(1, 0, 2), (3, 1, 4)]
    assert augment_with_reciprocals(triples, num_relations=2) == [
        (1, 0, 2),
        (3, 1, 4),
        (2, 2, 1),
        (4, 3, 3),
    ]
    inverse = build_inverse_relation_ids(2)
    query = torch.tensor([[1, 0, 2], [3, 1, 4]], dtype=torch.long)
    expected = torch.tensor([[2, 2, 1], [4, 3, 3]], dtype=torch.long)
    assert torch.equal(reciprocal_head_triples(query, inverse), expected)
