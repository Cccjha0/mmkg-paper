from __future__ import annotations

import numpy as np
import pytest

from scripts.aacpi_phase3a_common import (
    R1_ADDITIONS,
    R2_ADDITIONS,
    REPRESENTATION_FEATURES,
    action_response_features,
    cross_expert_features,
    validate_reference_response,
)
from scripts.build_aacpi_action_response_features import reject_test_path


def test_reference_action_response_is_exact_identity() -> None:
    scores = np.linspace(-2.0, 2.0, 30)
    row = action_response_features(scores, scores.copy())
    assert list(row) == R2_ADDITIONS
    validate_reference_response(row)
    assert row["r2_action_top1_top2_margin"] > 0.0


def test_response_features_change_without_target_identity() -> None:
    anchor = np.arange(30, dtype=np.float64)
    action = anchor.copy()
    action[0], action[-1] = action[-1] + 2.0, action[0]
    row = action_response_features(anchor, action)
    assert row["r2_response_top1_changed"] == 1.0
    assert row["r2_response_top20_jaccard"] < 1.0
    assert all(np.isfinite(value) for value in row.values())


def test_cross_expert_contract_and_normalized_cross_ranks() -> None:
    scores_a = np.arange(30, dtype=np.float64)
    scores_b = scores_a[::-1].copy()
    row = cross_expert_features(scores_a, scores_b)
    assert list(row) == R1_ADDITIONS
    assert row["r1_expert_top1_same"] == 0.0
    assert row["r1_a_top1_rank_under_b"] == pytest.approx(1.0)
    assert row["r1_b_top1_rank_under_a"] == pytest.approx(1.0)


def test_frozen_family_nesting() -> None:
    assert REPRESENTATION_FEATURES["R1"][:15] == REPRESENTATION_FEATURES["R0"]
    assert REPRESENTATION_FEATURES["R2"][:15] == REPRESENTATION_FEATURES["R0"]
    assert REPRESENTATION_FEATURES["R3"][: len(REPRESENTATION_FEATURES["R2"])] == REPRESENTATION_FEATURES["R2"]


def test_test_paths_are_rejected_before_io() -> None:
    with pytest.raises(RuntimeError, match="TEST"):
        reject_test_path(__import__("pathlib").Path("outputs/aacpi/test/features.csv"))
