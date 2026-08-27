import torch

from ml.training.src.models.recent_baselines.adamf_mat import OpenBGAdaMFMAT
from ml.training.src.models.recent_baselines.adversarial import MultiGenerator, generator_gradient_norm


def _build_adamf_mat():
    return OpenBGAdaMFMAT(
        text_feat=torch.randn(5, 3),
        img_feat=torch.randn(5, 4),
        has_img=torch.tensor([True, True, False, True, False]),
        num_entities=5,
        num_relations=2,
        d=2,
        margin=6.0,
    )


def test_adamf_mat_attention_is_normalized_and_score_is_finite():
    model = _build_adamf_mat().eval()
    entity_ids = torch.tensor([0, 2], dtype=torch.long)
    attention = model.get_attention(*model.get_batch_ent_multimodal_embs(entity_ids))
    assert attention.shape == (2, 3)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2))

    triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
    score = model.score(triples)
    assert score.shape == (2,)
    assert bool(torch.isfinite(score).all())


def test_adamf_mat_generator_receives_nonzero_gradients():
    model = _build_adamf_mat().train()
    generator = MultiGenerator(
        noise_dim=3,
        structure_dim=model.dim_e,
        modality_dim=model.dim_e,
        hidden_dim=8,
    )
    triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
    h_ids, _, t_ids = triples.unbind(dim=1)
    fake_h_visual, fake_h_text = generator(model.get_batch_ent_embs(h_ids).detach())
    fake_t_visual, fake_t_text = generator(model.get_batch_ent_embs(t_ids).detach())
    fake_scores = model.fake_scores(
        triples,
        fake_head_visual=fake_h_visual,
        fake_tail_visual=fake_t_visual,
        fake_head_text=fake_h_text,
        fake_tail_text=fake_t_text,
    )
    sum(-score.mean() for score in fake_scores).backward()
    assert generator_gradient_norm(generator) > 0.0
