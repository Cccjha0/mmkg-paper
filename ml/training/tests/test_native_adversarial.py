import torch

from ml.training.src.models.recent_baselines.adversarial import (
    CombinedGenerator,
    generator_gradient_norm,
    gradient_penalty,
)
from ml.training.src.models.recent_baselines.native import OpenBGNativE


def _build_native():
    return OpenBGNativE(
        text_feat=torch.randn(5, 3),
        img_feat=torch.randn(5, 4),
        has_img=torch.tensor([True, True, False, True, False]),
        num_entities=5,
        num_relations=2,
        d=2,
        margin=6.0,
    )


def _generate(model, generator, triples):
    h_ids, _, t_ids = triples.unbind(dim=1)
    head = model.get_batch_ent_multimodal_embs(h_ids)
    tail = model.get_batch_ent_multimodal_embs(t_ids)
    fake_h_visual, fake_h_text = generator(*head)
    fake_t_visual, fake_t_text = generator(*tail)
    scores, embeddings = model.fake_scores_and_embeddings(
        triples,
        fake_head_visual=fake_h_visual,
        fake_tail_visual=fake_t_visual,
        fake_head_text=fake_h_text,
        fake_tail_text=fake_t_text,
    )
    return scores, embeddings


def test_native_score_is_finite_and_has_one_value_per_triple():
    model = _build_native().eval()
    triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
    score = model.score(triples)
    assert score.shape == (2,)
    assert bool(torch.isfinite(score).all())


def test_native_generator_receives_nonzero_gradients():
    model = _build_native().train()
    generator = CombinedGenerator(
        noise_dim=3,
        structure_dim=model.dim_e,
        modality_dim=model.dim_e,
        hidden_dim=8,
    )
    triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
    scores, _ = _generate(model, generator, triples)
    generator_loss = sum((model.margin - score).mean() for score in scores) / len(scores)
    generator_loss.backward()
    assert generator_gradient_norm(generator) > 0.0


def test_native_gradient_penalty_is_finite():
    model = _build_native().train()
    generator = CombinedGenerator(
        noise_dim=3,
        structure_dim=model.dim_e,
        modality_dim=model.dim_e,
        hidden_dim=8,
    )
    triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
    _, real_embeddings = model.score_and_embeddings(triples)
    _, fake_embeddings = _generate(model, generator, triples)
    penalty = gradient_penalty(
        model.score_from_embeddings,
        real_embeddings,
        fake_embeddings,
        coefficient=0.1,
    )
    assert penalty.ndim == 0
    assert bool(torch.isfinite(penalty))
