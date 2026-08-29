import torch

from ml.training.src.models.recent_baselines.apkgc import OpenBGAPKGC


def test_epoch_noise_snapshot_can_train_across_multiple_batches():
    """Regression: an epoch snapshot must not retain the first batch graph."""
    model = OpenBGAPKGC(
        text_feat=torch.randn(4, 3),
        img_feat=torch.randn(4, 2),
        has_img=torch.tensor([True, True, False, False]),
        num_entities=4,
        num_relations=1,
        d=2,
        add_noise=True,
        noise_update="epoch",
        noise_ratio=0.2,
        mask_ratio=0.7,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    positive = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.long)
    negative = torch.tensor([[0, 0, 2], [1, 0, 3]], dtype=torch.long)

    model.train()
    model.on_epoch_start(epoch=1)
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(positive, negative)
        loss.backward()
        optimizer.step()


def test_db15k_official_graph_fusion_path_scores_triples():
    model = OpenBGAPKGC(
        text_feat=torch.randn(4, 3),
        img_feat=torch.randn(4, 2),
        has_img=torch.tensor([True, True, False, False]),
        has_text=torch.tensor([True, False, True, False]),
        num_entities=4,
        num_relations=1,
        d=2,
        num_proj=2,
        joint_way="Mformer_hd_graph",
        num_attention_heads=2,
    )
    triples = torch.tensor([[0, 0, 1], [2, 0, 3]], dtype=torch.long)
    scores = model.score(triples)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
