from __future__ import annotations

import pytest
import torch

from ml.training.src.data.dataset_spec import DatasetBundle, FeatureBundle, MMKG_GENERAL_V1


def make_bundle() -> DatasetBundle:
    return DatasetBundle(
        name="toy",
        protocol_version=MMKG_GENERAL_V1,
        train_triples=[(0, 0, 1)],
        valid_triples=[(1, 0, 2)],
        test_triples=[(2, 0, 0)],
        num_entities=3,
        num_relations=1,
        entity2id={"e0": 0, "e1": 1, "e2": 2},
        relation2id={"r0": 0},
        features=FeatureBundle(
            text_features=torch.zeros(3, 4),
            image_features=torch.zeros(3, 5),
            has_text=torch.tensor([True, False, True]),
            has_img=torch.tensor([False, True, True]),
        ),
    )


def test_valid_bundle_accepts_independent_masks() -> None:
    make_bundle().validate()


def test_mapping_ids_must_be_contiguous() -> None:
    bundle = make_bundle()
    broken = DatasetBundle(**{**bundle.__dict__, "entity2id": {"e0": 0, "e1": 1, "e2": 3}})
    with pytest.raises(ValueError, match="contiguous"):
        broken.validate()


def test_nonfinite_canonical_features_are_rejected() -> None:
    bundle = make_bundle()
    text = bundle.features.text_features.clone()
    text[0, 0] = float("nan")
    broken_features = FeatureBundle(
        text_features=text,
        image_features=bundle.features.image_features,
        has_text=bundle.features.has_text,
        has_img=bundle.features.has_img,
    )
    broken = DatasetBundle(**{**bundle.__dict__, "features": broken_features})
    with pytest.raises(ValueError, match="NaN or infinite"):
        broken.validate()
