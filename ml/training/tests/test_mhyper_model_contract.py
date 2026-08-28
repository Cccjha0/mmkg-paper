import torch

from ml.training.src.models.recent_baselines.mhyper import OpenBGMHyper


def _model() -> OpenBGMHyper:
    torch.manual_seed(11)
    image = torch.randn(7, 5)
    image[2].zero_()
    return OpenBGMHyper(
        text_feat=torch.randn(7, 6),
        img_feat=image,
        has_img=torch.tensor([1, 1, 0, 1, 1, 0, 1], dtype=torch.bool),
        num_entities=7,
        num_relations=3,
        rank=2,
        pca_init=False,
    )


def test_mhyper_released_embedding_shapes():
    model = _model()
    assert model.all.weight.shape == (7, 4)
    assert model.stru.weight.shape == (7, 4)
    assert model.img.weight.shape == (7, 4)
    assert model.text.weight.shape == (7, 4)
    assert model.rel_embedding.weight.shape == (6, 32)
    model.eval()
    logits = model.inference_all(torch.tensor([[0, 0, 1]], dtype=torch.long))
    assert logits.shape == (1, 7)
    assert model._get_eval_cache()["entities"].shape == (7, 16)


def test_mhyper_one_vs_all_loss_is_finite_and_has_gradients():
    model = _model().train()
    triples = torch.tensor([[0, 0, 1], [2, 4, 3]], dtype=torch.long)
    loss = model.one_vs_all_loss(triples)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in gradients)
    values = [
        gradient.coalesce().values() if gradient.is_sparse else gradient
        for gradient in gradients
        if gradient is not None
    ]
    assert all(bool(torch.isfinite(value).all()) for value in values)
    assert any(float(value.abs().sum()) > 0.0 for value in values)


def test_mhyper_eval_is_deterministic_and_missing_raw_image_stays_zero():
    model = _model().eval()
    triples = torch.tensor([[0, 0, 2], [1, 1, 5]], dtype=torch.long)
    first = model.score_tail(triples)
    second = model.score_tail(triples)
    assert torch.equal(first, second)
    assert torch.count_nonzero(model.img_feat[2]).item() == 0
    assert not bool(model.has_img[2])
    assert torch.isfinite(first).all()
