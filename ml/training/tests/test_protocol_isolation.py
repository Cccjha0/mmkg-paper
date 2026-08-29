from __future__ import annotations

import torch

from ml.training.src.data.dataset_spec import DatasetBundle, FeatureBundle, MMKG_GENERAL_V1, OPENBG_LEGACY_V1
from ml.training.src.models.build_model import build_model
from ml.training.src.models.openbg_img_gated_lp import OpenBGImgGateOnlyLP


def _model(protocol: str, has_text: torch.Tensor | None) -> OpenBGImgGateOnlyLP:
    return OpenBGImgGateOnlyLP(
        text_feat=torch.zeros(3, 4),
        img_feat=torch.zeros(3, 5),
        has_img=torch.tensor([True, False, True]),
        has_text=has_text,
        protocol_version=protocol,
        num_relations=2,
        d=4,
    )


def test_legacy_state_dict_has_no_general_text_mask_or_missing_parameter() -> None:
    keys = set(_model(OPENBG_LEGACY_V1, None).state_dict())
    assert "has_text" not in keys
    assert "t_missing" not in keys


def test_general_state_dict_records_text_mask_and_missing_parameter() -> None:
    keys = set(_model(MMKG_GENERAL_V1, torch.tensor([True, False, True])).state_dict())
    assert "has_text" in keys
    assert "t_missing" in keys


def test_legacy_bundle_does_not_enable_general_recent_baseline_masks() -> None:
    features = FeatureBundle(
        text_features=torch.randn(4, 4),
        image_features=torch.randn(4, 5),
        has_text=torch.ones(4, dtype=torch.bool),
        has_img=torch.tensor([True, False, True, False]),
    )
    bundle = DatasetBundle(
        name="openbg_img",
        protocol_version=OPENBG_LEGACY_V1,
        train_triples=[(0, 0, 1)],
        valid_triples=[(1, 0, 2)],
        test_triples=[(2, 0, 3)],
        num_entities=4,
        num_relations=1,
        entity2id={f"ent_{idx:06d}": idx for idx in range(4)},
        relation2id={"rel_0000": 0},
        features=features,
    )
    cfg = {
        "dataset": {"name": "openbg_img", "cache_format": "raw"},
        "model": {"name": "openbg_img_native", "num_relations": 1, "dim": 2},
        "training": {},
    }
    model, _ = build_model(cfg, dataset_bundle=bundle)
    assert model.has_text is None
    assert "has_text" not in model.state_dict()
