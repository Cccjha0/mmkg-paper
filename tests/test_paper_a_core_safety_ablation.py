from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.ablate_paper_a_core_safety_components import (
    point_metrics,
    policy_for_variant,
    validate_group_contract,
)


def exact_rows() -> list[dict]:
    return [
        {
            "rr_alpha_0_00": "0.2",
            "rr_alpha_0_25": "0.35",
            "rr_alpha_0_50": "0.5",
            "rr_alpha_1_00": "0.8",
        },
        {
            "rr_alpha_0_00": "0.3",
            "rr_alpha_0_25": "0.45",
            "rr_alpha_0_50": "0.6",
            "rr_alpha_1_00": "0.9",
        },
    ]


def apply(variant_id: str) -> dict[str, np.ndarray]:
    return policy_for_variant(
        exact_rows(),
        variant_id=variant_id,
        decision=np.asarray([0.3, -2.0]),
        probability=np.asarray([0.9, 0.55]),
        nonfinite=np.asarray([False, False]),
        alpha0=0.5,
        beta=0.2,
        threshold=0.3,
        alphas=(0.0, 0.25, 0.5, 1.0),
    )


def test_matched_variants_change_only_the_requested_component() -> None:
    full = apply("A_full_anchored")
    no_bound = apply("C_no_bound")
    no_fallback = apply("D_no_fallback")
    query_soft = apply("B_no_anchor_query_soft")

    assert full["applied"].tolist() == [0.5, 0.5]
    assert full["fallback"].tolist() == [False, True]
    assert no_bound["applied"].tolist() == [1.0, 0.5]
    assert no_bound["fallback"].tolist() == [False, True]
    assert no_fallback["applied"].tolist() == [0.5, 0.25]
    assert no_fallback["fallback"].tolist() == [False, False]
    assert query_soft["applied"].tolist() == [1.0, 0.5]


def test_point_metrics_separates_harm_change_and_deviation() -> None:
    frame = pd.DataFrame(
        {
            "rr_method": [0.7, 0.4, 0.5, 0.5],
            "rr_global": [0.5, 0.5, 0.5, 0.5],
            "abs_alpha_deviation": [0.2, 0.2, 0.0, 0.0],
            "fallback": [0, 0, 1, 1],
            "saturated": [0, 1, 0, 0],
            "raw_triple_id": ["t1", "t1", "t2", "t2"],
        }
    )
    result = point_metrics(frame)

    assert math.isclose(result["delta_mrr_vs_global"], 0.025)
    assert math.isclose(result["harmful_query_rate"], 0.25)
    assert math.isclose(result["beneficial_query_rate"], 0.25)
    assert math.isclose(result["mean_harm"], 0.1)
    assert math.isclose(result["fallback_rate"], 0.5)
    assert math.isclose(result["changed_from_anchor_rate"], 0.5)
    assert math.isclose(result["mean_abs_alpha_deviation"], 0.1)
    assert math.isclose(result["p95_abs_alpha_deviation"], 0.2)
    assert math.isclose(result["saturation_rate"], 0.25)


def test_group_contract_requires_all_seeds_and_directions() -> None:
    rows = []
    for seed in (1, 2, 3):
        for direction in ("head", "tail"):
            rows.append(
                {
                    "head_id": "1",
                    "relation_id": "2",
                    "tail_id": "3",
                    "seed": str(seed),
                    "direction": direction,
                }
            )
    validate_group_contract(rows, [1, 2, 3])
