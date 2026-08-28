import torch

from ml.training.src.models.recent_baselines.mhyper import OpenBGMHyper


def test_mhyper_pca_fit_ids_are_train_visible_only():
    model = OpenBGMHyper(
        text_feat=torch.randn(6, 4),
        img_feat=torch.randn(6, 4),
        has_img=torch.ones(6, dtype=torch.bool),
        num_entities=6,
        num_relations=2,
        rank=1,
        pca_init=True,
        pca_random_state=1,
    )
    model.prepare_training([(0, 0, 1), (1, 1, 2)])
    assert torch.equal(model.pca_fit_entity_ids.cpu(), torch.tensor([0, 1, 2]))
    assert 3 not in model.pca_fit_entity_ids.tolist()
