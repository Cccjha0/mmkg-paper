from __future__ import annotations

import pytest
import torch

from router.score_combination import (
    combine_expert_scores,
    normalize_candidate_scores,
    shrink_relation_alpha,
)
from router.prior_utils import compute_relation_gain_stats
from scripts.eval_score_ensemble_baselines import (
    combine_with_reference_targets,
    eval_mixed_rr,
    reference_ranks_and_rr,
)


@pytest.mark.parametrize("mode", ["none", "query_zscore", "rank_based", "rank"])
def test_score_normalization_uses_only_candidate_distributions(mode: str) -> None:
    scores = torch.tensor([[3.0, 1.0, float("-inf"), 2.0], [4.0, 4.0, 0.0, -1.0]])
    before_target_choice = normalize_candidate_scores(scores, mode)
    # Choosing a different hidden target cannot affect normalization because
    # target ids are not an input to the normalization API.
    hypothetical_target_a = torch.tensor([0, 2])
    hypothetical_target_b = torch.tensor([3, 1])
    assert not torch.equal(hypothetical_target_a, hypothetical_target_b)
    assert torch.equal(before_target_choice, normalize_candidate_scores(scores, mode))
    assert torch.isneginf(before_target_choice[0, 2])


def test_query_zscore_is_query_local_and_finite_on_constant_rows() -> None:
    scores = torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    normalized = normalize_candidate_scores(scores, "query_zscore")
    assert normalized[0].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert torch.equal(normalized[1], torch.zeros(3))
    assert torch.isfinite(normalized).all()


def test_combination_preserves_filtered_candidates_for_every_mode() -> None:
    fusion = torch.tensor([[2.0, float("-inf"), 0.0]])
    structural = torch.tensor([[0.0, float("-inf"), 2.0]])
    for mode in ("none", "query_zscore", "rank_based"):
        mixed = combine_expert_scores(fusion, structural, 0.5, normalization=mode)
        assert torch.isneginf(mixed[0, 1])
        assert torch.isfinite(mixed[0, [0, 2]]).all()


def test_rank_normalization_uses_competition_ranks_for_ties() -> None:
    scores = torch.tensor([[4.0, 4.0, 1.0, float("-inf")]])
    normalized = normalize_candidate_scores(scores, "rank_based")
    assert normalized[0, :3].tolist() == pytest.approx([1.0, 1.0, 1.0 / 3.0])
    assert torch.isneginf(normalized[0, 3])


@pytest.mark.parametrize("mode", ["query_zscore", "rank_based"])
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_general_normalized_endpoints_keep_exact_raw_scores(mode: str, alpha: float) -> None:
    fusion_candidates = torch.tensor([[4.0, 4.0, float("-inf"), 1.0]])
    structural_candidates = torch.tensor([[8.0, 7.0, 7.0, float("-inf")]])
    fusion_reference = torch.tensor([4.0])
    structural_reference = torch.tensor([7.0])

    mixed, reference = combine_with_reference_targets(
        fusion_candidates,
        structural_candidates,
        fusion_reference,
        structural_reference,
        alpha,
        mode,
    )
    expected_candidates = fusion_candidates if alpha == 1.0 else structural_candidates
    expected_reference = fusion_reference if alpha == 1.0 else structural_reference
    assert torch.equal(mixed, expected_candidates)
    assert torch.equal(reference, expected_reference)


def test_relation_alpha_shrinkage_matches_documented_formula() -> None:
    assert shrink_relation_alpha(0.8, support=10, global_alpha=0.2, shrinkage_lambda=10) == pytest.approx(0.5)
    assert shrink_relation_alpha(0.8, support=0, global_alpha=0.2, shrinkage_lambda=10) == pytest.approx(0.2)


def test_general_relation_gain_shrinks_toward_global_dev_mean() -> None:
    fusion = []
    structural = []
    deltas = [1.0, -1.0, -1.0, -1.0]
    for idx, delta in enumerate(deltas):
        relation_id = 0 if idx == 0 else 1
        base = {
            "query_id": f"q{idx}",
            "relation_id": relation_id,
            "relation_name": f"r{relation_id}",
            "target_regime": "head_T1V1",
        }
        fusion.append({**base, "rr": 1.0 if delta > 0 else 0.0})
        structural.append({**base, "rr": 0.0 if delta > 0 else 1.0})
    rows = compute_relation_gain_stats(
        fusion,
        structural,
        gamma=0.0,
        use_shrinkage=True,
        shrink_k=1.0,
        shrink_toward_global=True,
    )
    # Global delta is -0.5; relation 0 has raw delta +1 with support 1.
    assert rows[0]["mean_delta_rr_shrunk"] == pytest.approx(0.25)
    assert rows[0]["shrinkage_target"] == pytest.approx(-0.5)


@pytest.mark.parametrize("mode", ["none", "query_zscore", "rank_based"])
def test_general_score_interpolation_uses_canonical_reference_endpoints(mode: str) -> None:
    target_ids = torch.tensor([0])
    fusion_candidates = torch.tensor([[0.0, 10.0, 9.0]])
    structural_candidates = torch.tensor([[8.0, 7.0, 6.0]])
    fusion_reference = torch.tensor([11.0])
    structural_reference = torch.tensor([5.0])

    _, fusion_rr = reference_ranks_and_rr(fusion_candidates, fusion_reference)
    _, structural_rr = reference_ranks_and_rr(structural_candidates, structural_reference)
    assert eval_mixed_rr(
        fusion_candidates,
        structural_candidates,
        target_ids,
        1.0,
        score_normalization=mode,
        gate_target_scores=fusion_reference,
        residual_target_scores=structural_reference,
    ) == fusion_rr
    assert eval_mixed_rr(
        fusion_candidates,
        structural_candidates,
        target_ids,
        0.0,
        score_normalization=mode,
        gate_target_scores=fusion_reference,
        residual_target_scores=structural_reference,
    ) == structural_rr


def test_legacy_score_interpolation_keeps_full_matrix_target_semantics() -> None:
    target_ids = torch.tensor([0])
    fusion_candidates = torch.tensor([[0.0, 10.0, 9.0]])
    structural_candidates = torch.tensor([[8.0, 7.0, 6.0]])
    # Without general-protocol reference scores, alpha=1 retains the frozen
    # legacy behavior and ranks the target value gathered from the matrix.
    assert eval_mixed_rr(
        fusion_candidates,
        structural_candidates,
        target_ids,
        1.0,
    ) == pytest.approx([1.0 / 3.0])
