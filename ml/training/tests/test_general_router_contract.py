from __future__ import annotations

import json
from pathlib import Path

import torch

from router.feature_utils import load_general_cache_bundle
from router.router_models import CleanRuleBasedRouter
from router.router_models import CLEAN_FEATURE_SETS


def _write_general_cache(path: Path, *, shared: bool) -> None:
    path.mkdir(parents=True)
    torch.save(torch.ones(2, 3), path / "text_feat.pt")
    torch.save(torch.ones(2, 3), path / "img_feat.pt")
    torch.save(torch.tensor([True, True]), path / "has_text.pt")
    torch.save(torch.tensor([True, False]), path / "has_img.pt")
    manifest = {
        "dataset": "toy",
        "protocol": "mmkg_general_v1",
        "cross_modal_similarity": {
            "type": "cosine" if shared else "none",
            "shared_embedding_space": shared,
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_general_cosine_requires_explicit_shared_space(tmp_path: Path) -> None:
    independent = tmp_path / "independent"
    shared = tmp_path / "shared"
    _write_general_cache(independent, shared=False)
    _write_general_cache(shared, shared=True)
    assert load_general_cache_bundle(independent)["cross_modal_cosine_enabled"] is False
    assert load_general_cache_bundle(shared)["cross_modal_cosine_enabled"] is True


def test_general_rule_treats_text_and_visual_availability_symmetrically() -> None:
    router = CleanRuleBasedRouter(gamma=0.0)
    general_text_only = {
        "observed_has_text": 1,
        "observed_has_img": 0,
        "observed_modality_count": 1,
        "relation_gain_prior": 0.1,
    }
    general_visual_only = {
        "observed_has_text": 0,
        "observed_has_img": 1,
        "observed_modality_count": 1,
        "relation_gain_prior": 0.1,
    }
    assert router.predict_from_rows([general_text_only, general_visual_only]) == [1, 1]


def test_legacy_rule_keeps_frozen_image_gate() -> None:
    router = CleanRuleBasedRouter(gamma=0.0)
    assert router.predict_from_rows([{"observed_has_img": 0, "relation_gain_prior": 0.1}]) == [0]
    assert router.predict_from_rows([{"observed_has_img": 1, "relation_gain_prior": 0.1}]) == [1]


def test_general_clean_profiles_exclude_answer_and_target_leakage() -> None:
    forbidden = {
        "target_has_text",
        "target_has_img",
        "target_regime",
        "target_entity_id",
        "correct_score",
        "rr",
        "reciprocal_rank",
        "rank_fusion",
        "rank_struct",
    }
    for profile in ("G1", "G2", "G3", "G4"):
        features = set(CLEAN_FEATURE_SETS[profile])
        assert features.isdisjoint(forbidden)
        assert "relation_id" not in features
