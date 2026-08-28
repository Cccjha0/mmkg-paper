import torch

from ml.training.src.models.recent_baselines.mhyper import OpenBGMHyper


def _model() -> OpenBGMHyper:
    torch.manual_seed(3)
    return OpenBGMHyper(
        text_feat=torch.randn(6, 5),
        img_feat=torch.randn(6, 4),
        has_img=torch.tensor([1, 1, 0, 1, 0, 1], dtype=torch.bool),
        num_entities=6,
        num_relations=2,
        rank=2,
        pca_init=False,
    ).eval()


def test_mhyper_exact_candidate_score_matches_full_one_vs_all_column():
    model = _model()
    triples = torch.tensor([[0, 0, 1], [0, 0, 4], [2, 1, 3]], dtype=torch.long)
    full = model.inference_all(triples)
    exact = model.score_tail(triples)
    expected = full[torch.arange(triples.shape[0]), triples[:, 2]]
    assert torch.allclose(exact, expected, atol=1e-6, rtol=1e-6)


def test_mhyper_head_score_is_reciprocal_tail_score():
    model = _model()
    triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
    reciprocal = torch.tensor([[1, 2, 0], [3, 3, 2]], dtype=torch.long)
    assert torch.allclose(model.score_head(triples), model.score_tail(reciprocal))
