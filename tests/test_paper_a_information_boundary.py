"""B07/C04: change gold AND filter facts through real export code paths."""
from types import SimpleNamespace
import inspect
from pathlib import Path
import uuid

import numpy as np
import pytest
import torch

from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.data.tsv_reader import read_integer_triples
from ml.training.src.eval.filtered_ranking import (
    _build_dense_filter_mask, _apply_filter_mask,
    prepare_true_heads_index, prepare_true_tails_index,
)
from router.information_boundary import (
    SCORE_INFORMATION_CONTRACT, require_score_information_contract, require_unfiltered_rows,
)
from router.query_geometry import QUERY_GEOMETRY_FIELDS, query_geometry_tensor
from scripts.crossfit_anchored_dynamic import apply_policy, feature_matrix
from scripts.eval_heterogeneous_complementarity import (
    evaluate_unit, score_expert_block, filtered_copy, query_zscore_with_reference,
    ranks_against_reference,
)
from scripts.eval_openbg_dynasemble import (
    evaluate_direction, normalize_and_features, true_indexes_for_expert,
)


class ToyScorer:
    def __init__(self, values):
        self.values = torch.tensor(values, dtype=torch.float32)

    def score_tail(self, triples):
        return self.values[triples[:, 2]]

    def score_head(self, triples):
        return self.values[triples[:, 0]]


def expert(values, name):
    return SimpleNamespace(
        model=ToyScorer(values), num_entities=len(values), query_batch_size=2,
        chunk_size=3, seed=1, name=name,
        bundle=SimpleNamespace(name="toy", protocol_version="mmkg_general_v1"),
    )


def setup(direction):
    # Same observable query, two different valid answers with different scores.
    triples = [(0, 0, 1), (0, 0, 2)] if direction == "tail" else [(1, 0, 0), (2, 0, 0)]
    tails, heads = build_true_facts(triples + triples)
    index = prepare_true_tails_index(tails) if direction == "tail" else prepare_true_heads_index(heads)
    a = expert([1, 12, 5, 4, 2, 0], "A")
    b = expert([3, 1, 9, 2, 4, 0], "B")
    return triples, index, a, b


@pytest.mark.parametrize("direction", ["head", "tail"])
def test_gold_and_truth_cannot_change_exported_features_or_policy_weights(direction):
    triples, index, a, b = setup(direction)
    outputs = []
    for truth in (index, {}):
        rows = evaluate_unit(
            expert_a=a, expert_b=b, triples=triples, split="dev", direction=direction,
            true_index=truth, alphas=(0., .5, 1.), rrf_k=60., selection=None,
            export_alpha_grid=True, export_x4_features=False, x4_context=None,
            device="cpu", progress_every=0, pair_name="toy", filter_fact_scope="train_dev_test",
        )
        require_unfiltered_rows(rows)
        matrix, nonfinite = feature_matrix(rows)
        np.testing.assert_array_equal(matrix[0], matrix[1])
        # Fixed nonconstant selector: any gold dependence in geometry reaches weights.
        decision = matrix[:, 3] / 10 - matrix[:, 7] / 7
        probability = 1 / (1 + np.exp(-decision))
        policy = apply_policy(rows, decision=decision, probability_a=probability,
                              nonfinite=nonfinite, alpha0=.5, beta=.5,
                              confidence_threshold=0, alphas=(0., .5, 1.))
        assert policy["continuous"][0] == policy["continuous"][1]
        assert policy["applied"][0] == policy["applied"][1]
        outputs.append((matrix, policy["continuous"]))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("direction", ["head", "tail"])
def test_old_post_mask_path_is_a_detectable_counterexample(direction):
    triples, index, a, b = setup(direction)
    q = torch.tensor(triples)
    masked_a, _, raw_a = score_expert_block(a, q, direction, index, "cpu", retain_unfiltered=True)
    masked_b, _, raw_b = score_expert_block(b, q, direction, index, "cpu", retain_unfiltered=True)
    assert not torch.equal(query_geometry_tensor(masked_a, masked_b, direction)[0],
                           query_geometry_tensor(masked_a, masked_b, direction)[1])
    assert torch.equal(raw_a[0], raw_a[1]) and torch.equal(raw_b[0], raw_b[1])
    original = raw_a.clone()
    filtered_copy(raw_a, q, direction, index)
    assert torch.equal(raw_a, original)
    assert raw_a.data_ptr() != masked_a.data_ptr()


@pytest.mark.parametrize("direction", ["head", "tail"])
def test_gold_kept_other_answers_removed_and_dense_sparse_masks_agree(direction):
    triples, index, a, _ = setup(direction)
    q = torch.tensor(triples)
    masked, target, raw = score_expert_block(a, q, direction, index, "cpu", retain_unfiltered=True)
    target_ids = q[:, 2] if direction == "tail" else q[:, 0]
    rows = torch.arange(2)
    torch.testing.assert_close(masked[rows, target_ids], target)
    assert torch.isneginf(masked[0, 2]) and torch.isneginf(masked[1, 1])
    exclusions = [torch.tensor([1, 2]), torch.tensor([1, 2])]
    dense = _build_dense_filter_mask(exclusions, target_ids, 6, torch.device("cpu"))
    sparse_exclusions = [ids[ids != gold] for ids, gold in zip(exclusions, target_ids)]
    for dense_mask in (dense, None):
        chunks = []
        for start, end in ((0, 3), (3, 6)):
            chunk = raw[:, start:end].clone()
            _apply_filter_mask(chunk, dense_filter_mask=dense_mask, filt_excl_list=sparse_exclusions,
                               start=start, end=end, device=torch.device("cpu"))
            chunks.append(chunk)
        torch.testing.assert_close(torch.cat(chunks, 1), masked)
    # Independent rank count, with the designated gold explicitly excluded from filter.
    expected = [1 + sum(float(v) > float(target[i]) for j, v in enumerate(raw[i])
                        if j not in ({1, 2} - {int(target_ids[i])})) for i in range(2)]
    assert ranks_against_reference(masked, target).tolist() == expected


def test_normalization_parameters_and_dynasemble_features_ignore_reference_scores():
    raw = torch.tensor([[1., 12., 5., 4., 2., 0.]]).repeat(2, 1)
    reference = torch.tensor([12., 5.])
    normalized, ref = query_zscore_with_reference(raw, reference)
    expected = (raw - raw.mean(1, keepdim=True)) / (raw.std(1, keepdim=True, correction=0) + 1e-8)
    torch.testing.assert_close(normalized, expected)
    assert torch.equal(normalized[0], normalized[1]) and ref[0] != ref[1]
    _, _, features = normalize_and_features(raw, reference)
    assert torch.equal(features[0], features[1])
    assert set(inspect.signature(query_geometry_tensor).parameters) == {"scores_a", "scores_b", "direction"}


class NonconstantSelector(torch.nn.Module):
    def forward(self, features):
        return 1 + features[:, :1] + 2 * features[:, 2:3]


@pytest.mark.parametrize("direction", ["head", "tail"])
def test_dynasemble_actual_evaluation_weights_ignore_gold_and_filter(direction):
    triples, index, a, b = setup(direction)
    weights = []
    for truth in (index, {}):
        rows = evaluate_direction(
            expert_a=a, expert_b=b, selector=NonconstantSelector(), triples=triples,
            direction=direction, true_index=truth, global_alpha=.5, split="dev",
            device="cpu", progress_every=0, selector_hash="toy", selection_hash="toy", pair_name="toy",
        )
        weights.extend(row["weight_expert_a"] for row in rows)
    assert len(set(weights)) == 1


def test_truth_union_deduplicates_filters_but_reader_preserves_observations():
    train, dev, test = [(0, 0, 1)], [(0, 0, 2)], [(0, 0, 3)]
    obj = SimpleNamespace(bundle=SimpleNamespace(train_triples=train * 2,
                          valid_triples=dev, test_triples=test))
    assert true_indexes_for_expert(obj)["tail"][(0, 0)].tolist() == [1, 2, 3]
    assert true_indexes_for_expert(obj, include_test=False)["tail"][(0, 0)].tolist() == [1, 2]
    assert build_true_facts(train)[0][(0, 0)] == {1}
    path = Path.cwd() / "tmp" / f"b07_triples_{uuid.uuid4().hex}.tsv"
    path.parent.mkdir(exist_ok=True)
    path.write_text("0\t0\t1\n0\t0\t1\n", encoding="utf-8")
    assert read_integer_triples(path) == train * 2
    path.unlink()


def test_legacy_artifacts_are_rejected():
    for payload in ({}, {"score_information_contract": "filtered_v1"}):
        with pytest.raises(ValueError, match="re-export"):
            require_score_information_contract(payload)
    require_score_information_contract({"score_information_contract": SCORE_INFORMATION_CONTRACT})


def test_selector_training_truth_does_not_even_read_test_triples():
    class TrainingBundle:
        train_triples = [(0, 0, 1)]
        valid_triples = [(0, 0, 2)]

        @property
        def test_triples(self):
            raise AssertionError("TEST facts reached supervised negative sampling")

    index = true_indexes_for_expert(SimpleNamespace(bundle=TrainingBundle()), include_test=False)
    assert index["tail"][(0, 0)].tolist() == [1, 2]


def test_canonical_preprocessing_rejects_duplicate_and_cross_split_facts():
    from ml.training.scripts.preprocess_external_mmkg import _assert_clean_canonical_splits
    with pytest.raises(ValueError, match="duplicates"):
        _assert_clean_canonical_splits({"train": [(0, 0, 1)] * 2, "valid": [], "test": []})
    with pytest.raises(ValueError, match="exact_overlaps"):
        _assert_clean_canonical_splits({"train": [(0, 0, 1)], "valid": [(0, 0, 1)], "test": []})


def test_exported_mixture_rr_uses_full_distribution_normalization_before_filter():
    triples, index, a, b = setup("tail")
    rows = evaluate_unit(expert_a=a, expert_b=b, triples=triples, split="dev", direction="tail",
                         true_index=index, alphas=(0., .5, 1.), rrf_k=60., selection=None,
                         export_alpha_grid=True, export_x4_features=False, x4_context=None,
                         device="cpu", progress_every=0, pair_name="toy", filter_fact_scope="train_dev_test")
    av, bv = a.model.values.numpy(), b.model.values.numpy()
    mixed = .5 * (av-av.mean()) / (av.std()+1e-8) + .5 * (bv-bv.mean()) / (bv.std()+1e-8)
    for row, gold in zip(rows, (1, 2)):
        competitors = [e for e in range(6) if e not in ({1, 2} - {gold})]
        rank = 1 + sum(mixed[e] > mixed[gold] for e in competitors)
        assert row["rr_equal"] == 1 / rank
        assert row["rr_alpha_1_00"] == row["rr_a"]
        assert row["rr_alpha_0_00"] == row["rr_b"]
