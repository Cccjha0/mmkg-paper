from __future__ import annotations

import torch

from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.data.dataset_spec import DatasetBundle, FeatureBundle, MMKG_GENERAL_V1
from ml.training.src.eval.filtered_ranking import filtered_ranking_eval
from ml.training.src.models.build_model import build_model
from ml.training.src.models.general_mmkg.availability_fusion import (
    AvailabilityAwareGatedFusion,
    MMKGAvailabilityAwareFusionLP,
)
from ml.training.src.models.general_mmkg.structural_expert import MMKGStructuralExpertLP


def _bundle(text: torch.Tensor, image: torch.Tensor) -> DatasetBundle:
    features = FeatureBundle(
        text_features=text,
        image_features=image,
        has_text=torch.tensor([True, True, False, False]),
        has_img=torch.tensor([True, False, True, False]),
    )
    return DatasetBundle(
        name="toy",
        protocol_version=MMKG_GENERAL_V1,
        train_triples=[(0, 0, 1)],
        valid_triples=[(1, 0, 2)],
        test_triples=[(2, 0, 3)],
        num_entities=4,
        num_relations=1,
        entity2id={f"e{idx}": idx for idx in range(4)},
        relation2id={"r0": 0},
        features=features,
    )


def test_structural_v2_is_independent_of_modality_tensors_and_masks() -> None:
    cfg = {
        "protocol": {"version": MMKG_GENERAL_V1},
        "embedding": {"d": 4},
        "model": {"name": "mmkg_structural_v2", "num_relations": 1},
        "training": {},
    }
    original = _bundle(torch.randn(4, 3), torch.randn(4, 5))
    changed = _bundle(torch.full((4, 8), 999.0), torch.full((4, 2), -999.0))
    changed.features.has_text.logical_not_()
    changed.features.has_img.logical_not_()
    torch.manual_seed(7)
    first, _ = build_model(cfg, dataset_bundle=original)
    torch.manual_seed(7)
    second, _ = build_model(cfg, dataset_bundle=changed)

    triples = torch.tensor([(0, 0, 1), (2, 0, 3)], dtype=torch.long)
    assert torch.equal(first.score(triples), second.score(triples))
    forbidden = {"text_feat", "img_feat", "has_text", "has_img"}
    assert forbidden.isdisjoint(first.state_dict())


def test_availability_aware_fusion_covers_all_four_states_without_nan() -> None:
    fusion = AvailabilityAwareGatedFusion(d=4, num_relations=1, use_layernorm=False)
    text = torch.randn(4, 4)
    image = torch.randn(4, 4)
    has_text = torch.tensor([True, True, False, False])
    has_img = torch.tensor([True, False, True, False])
    fused, weights = fusion(text, image, torch.zeros(4, dtype=torch.long), has_text, has_img)

    assert torch.isfinite(fused).all()
    assert torch.isfinite(weights).all()
    assert weights[1].tolist() == [1.0, 0.0]
    assert weights[2].tolist() == [0.0, 1.0]
    assert weights[3].tolist() == [0.0, 0.0]
    assert torch.equal(fused[1], text[1])
    assert torch.equal(fused[2], image[2])
    assert torch.equal(fused[3], fusion.fallback)


def test_independent_dropout_uses_the_same_missing_availability_path() -> None:
    model = MMKGAvailabilityAwareFusionLP(
        text_feat=torch.randn(4, 3),
        img_feat=torch.randn(4, 5),
        has_text=torch.tensor([True, True, False, False]),
        has_img=torch.tensor([True, False, True, False]),
        num_relations=1,
        d=4,
        text_dropout=1.0,
        img_dropout=0.0,
    )
    model.train()
    _, weights = model.entity_with_relation(
        torch.arange(4),
        torch.zeros(4, dtype=torch.long),
    )
    assert weights[0].tolist() == [0.0, 1.0]
    assert weights[1].tolist() == [0.0, 0.0]
    assert weights[2].tolist() == [0.0, 1.0]
    assert weights[3].tolist() == [0.0, 0.0]


def test_general_v2_models_use_unified_filtered_bidirectional_ranking() -> None:
    triples = torch.tensor([(0, 0, 1), (1, 0, 2), (2, 0, 3), (3, 0, 0)], dtype=torch.long)
    true_tails, true_heads = build_true_facts([tuple(row) for row in triples.tolist()])
    models = [
        MMKGStructuralExpertLP(num_entities=4, num_relations=1, d=4),
        MMKGAvailabilityAwareFusionLP(
            text_feat=torch.randn(4, 3),
            img_feat=torch.randn(4, 5),
            has_text=torch.tensor([True, True, False, False]),
            has_img=torch.tensor([True, False, True, False]),
            num_relations=1,
            d=4,
        ),
    ]
    for model in models:
        metrics = filtered_ranking_eval(
            model=model,
            triples=triples,
            true_tails=true_tails,
            true_heads=true_heads,
            num_entities=4,
            chunk_size=2,
            query_batch_size=2,
            device="cpu",
            direction="both",
        )
        assert metrics["count"] == 4
        assert 0.0 < metrics["mrr"] <= 1.0
