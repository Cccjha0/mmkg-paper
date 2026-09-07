from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_local_identifiability import (
    K_VALUES,
    action_indices,
    classify,
    encode_positive_sets,
    jaccard_against_multiset,
    jaccard_per_center,
    popcount32,
)
from scripts.exp2_information_common import ALPHAS


def contract() -> dict:
    return json.loads(Path("docs/protocols/EXP5_LOCAL_IDENTIFIABILITY_CONTRACT.json").read_text(encoding="utf-8"))


def test_action_tie_break_is_nearest_then_smaller() -> None:
    values = np.zeros((2, len(ALPHAS)))
    values[0, [8, 12]] = 1
    values[1, [10, 11]] = 1
    assert action_indices(values, 10).tolist() == [8, 10]


def test_positive_mask_and_popcount() -> None:
    utility = np.zeros((2, len(ALPHAS)))
    utility[0, [0, 2, 20]] = 1
    masks = encode_positive_sets(utility)
    assert popcount32(masks).tolist() == [3, 0]


def test_both_empty_jaccard_is_missing_not_one() -> None:
    center = np.asarray([0, 3], dtype=np.uint32)
    neighbor = np.asarray([[0, 1], [1, 3]], dtype=np.uint32)
    jaccard, empty = jaccard_per_center(center, neighbor)
    assert jaccard[0] == 0.0
    assert empty[0] == 0.5
    assert np.isclose(jaccard[1], 0.75)
    all_empty, rate = jaccard_per_center(np.asarray([0], dtype=np.uint32), np.asarray([[0, 0]], dtype=np.uint32))
    assert np.isnan(all_empty[0]) and rate[0] == 1.0


def test_multiset_jaccard_matches_expanded_neighbors() -> None:
    center = np.asarray([0, 1, 3, 3], dtype=np.uint32)
    sampled = np.asarray([0, 1, 1, 2, 3, 3], dtype=np.uint32)
    expanded = np.broadcast_to(sampled, (len(center), len(sampled)))
    expected, expected_empty = jaccard_per_center(center, expanded)
    actual, actual_empty = jaccard_against_multiset(center, sampled)
    assert np.allclose(actual, expected, equal_nan=True)
    assert np.allclose(actual_empty, expected_empty)


def synthetic_summary(local_pairs: int, zero_pairs: int) -> pd.DataFrame:
    rows = []
    for pair_index, pair_id in enumerate(contract()["pair_ids"]):
        dataset = "mkg_w" if pair_id.startswith("mkgw") else "db15k"
        for k in K_VALUES:
            signal = pair_index < local_pairs and k in (5, 10)
            zero = pair_index < zero_pairs and k == 10
            rows.append({
                "dataset": dataset, "pair_id": pair_id, "representation": "X4", "k": k,
                "direction_agreement_lift_ci95_low": 0.01 if signal else -0.01,
                "neighbor_consensus_utility_ci95_low": -0.01 if zero else (0.001 if signal else -0.001),
                "neighbor_consensus_utility_ci95_high": 0.01 if (zero or signal) else -0.0001,
                "consensus_utility_lift_ci95_low": 0.001 if signal else -0.001,
            })
    return pd.DataFrame(rows)


def test_local_identifiable_gate() -> None:
    recorded, final, local, evidence = classify(synthetic_summary(5, 0), contract())
    assert "E5_LOCAL_IDENTIFIABLE" in recorded
    assert final == "E5_LOCAL_IDENTIFIABLE"
    assert evidence["local_signal_pairs"] == 5


def test_local_ambiguity_gate() -> None:
    recorded, final, _, evidence = classify(synthetic_summary(1, 5), contract())
    assert recorded == ["E5_LOCAL_AMBIGUITY"]
    assert final == "E5_LOCAL_AMBIGUITY"
    assert evidence["primary_k_consensus_ci_includes_zero_pairs"] == 5
