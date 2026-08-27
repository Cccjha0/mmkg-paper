import torch

from ml.training.src.models.recent_baselines.rotate_utils import rotate_score


def test_rotate_score_is_larger_for_a_closer_rotated_triple():
    h = torch.tensor([[1.0, 0.0]])
    r = torch.tensor([[0.0]])
    close_tail = torch.tensor([[1.0, 0.0]])
    far_tail = torch.tensor([[0.0, 1.0]])

    close_score = rotate_score(h, r, close_tail, margin=6.0, embedding_range=10.0)
    far_score = rotate_score(h, r, far_tail, margin=6.0, embedding_range=10.0)
    assert close_score.item() > far_score.item()
