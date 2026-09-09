from __future__ import annotations

import math

import pandas as pd

from scripts.analyze_paper_a_risk_coverage import (
    apply_coverage,
    nearest_alpha,
    summarize_state,
)


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["test"] * 4,
            "dataset": ["Toy"] * 4,
            "pair": ["A + B"] * 4,
            "query_id": ["q1", "q2", "q3", "q4"],
            "direction": ["head", "tail", "head", "tail"],
            "confidence": [0.8, 0.9, 0.95, 0.1],
            "nonfinite_fallback": [False, False, True, False],
            "alpha0": [0.5] * 4,
            "alpha_raw_grid": [0.6, 0.4, 0.9, 0.5],
            "rr_global": [0.5, 0.5, 0.5, 0.5],
            "rr_raw": [0.7, 0.4, 0.9, 0.5],
        }
    )


def test_nearest_alpha_prefers_anchor_on_distance_tie() -> None:
    grid = (0.0, 1.0)
    assert nearest_alpha(0.5, grid, anchor=0.0) == 0.0
    assert nearest_alpha(0.5, grid, anchor=1.0) == 1.0
    assert nearest_alpha(0.5, grid, anchor=0.5) == 0.0


def test_coverage_excludes_nonfinite_and_uses_deterministic_order() -> None:
    result = apply_coverage(synthetic_frame(), 0.5)

    assert result["selected_for_adaptation"].tolist() == [True, True, False, False]
    assert result["changed_alpha"].tolist() == [True, True, False, False]
    assert result["rr_selective"].tolist() == [0.7, 0.4, 0.5, 0.5]


def test_summary_separates_coverage_change_and_harm() -> None:
    state = apply_coverage(synthetic_frame(), 0.5)
    result = summarize_state(
        state,
        scope="pair",
        dataset="Toy",
        pair="A + B",
        direction="pooled",
        coverage=0.5,
    )

    assert math.isclose(result["adaptation_coverage"], 0.5)
    assert math.isclose(result["changed_alpha_rate"], 0.5)
    assert math.isclose(result["delta_mrr_vs_global"], 0.025)
    assert math.isclose(result["harmful_query_rate"], 0.25)
    assert math.isclose(result["beneficial_query_rate"], 0.25)
    assert math.isclose(result["mean_harm"], 0.1)
    assert result["n_nonfinite_fallback"] == 1


def test_full_coverage_still_falls_back_for_nonfinite_rows() -> None:
    result = apply_coverage(synthetic_frame(), 1.0)

    assert result["selected_for_adaptation"].tolist() == [True, True, False, True]
    assert math.isclose(result["selected_for_adaptation"].mean(), 0.75)
