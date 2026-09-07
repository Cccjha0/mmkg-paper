from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_high_value_selectivity import (
    METRICS,
    clustered_bootstrap,
    load_contract,
    matched_random_baselines,
    point_metrics,
    top_selection,
)


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for triple in range(8):
        for seed in (1, 2, 3):
            for direction in ("head", "tail"):
                good = triple < 2
                rows.append({
                    "original_triple_id": f"t{triple}",
                    "seed": seed,
                    "direction": direction,
                    "relation_id": triple % 2,
                    "realized_utility": 0.1 if good else -0.01,
                    "changed": 1,
                    "stable2_nonempty": int(good),
                    "stable2_gain": 0.08 if good else 0.0,
                    "stable3_nonempty": int(good),
                    "stable3_gain": 0.05 if good else 0.0,
                    "consensus_gain": 0.09 if good else 0.001,
                    "high_gain_top10": int(triple == 0),
                    "raw_oracle_gain": 0.12 if good else 0.01,
                })
    return pd.DataFrame(rows)


def test_contract_freezes_x4_score_and_no_route_change() -> None:
    contract = load_contract(Path("docs/protocols/FROZEN_HIGH_VALUE_SELECTIVITY_AUDIT.json"))
    assert contract["ranking_rule"]["primary_scalar"] == "predicted_u[row, chosen_action_index[row]]"
    assert contract["decision_rule"]["new_route_or_go_gate"] is False
    assert tuple(contract["metrics"]) == METRICS


def test_top_selection_uses_ceil_and_lexicographic_tie_break() -> None:
    score = np.asarray([1.0, 1.0, 0.5, 0.0])
    query = np.asarray(["b", "a", "c", "d"])
    selected = top_selection(score, query, 0.25)
    assert selected.tolist() == [False, True, False, False]
    assert top_selection(score, query, 0.26).sum() == 2


def test_metric_definitions_capture_stable_and_high_gain_rows() -> None:
    frame = synthetic_frame()
    selected = frame.original_triple_id.isin(["t0", "t1"]).to_numpy()
    metrics = point_metrics(frame, selected, 1e-12)
    assert np.isclose(metrics["selected_mean_realized_utility"], 0.1)
    assert np.isclose(metrics["selective_population_gain"], 0.025)
    assert np.isclose(metrics["stable2_opportunity_rate"], 1.0)
    assert np.isclose(metrics["stable2_opportunity_enrichment"], 4.0)
    assert np.isclose(metrics["stable2_gain_capture"], 1.0)
    assert np.isclose(metrics["high_gain_recall"], 1.0)


def test_clustered_bootstrap_preserves_triple_rows() -> None:
    frame = synthetic_frame()
    selected = frame.original_triple_id.isin(["t0", "t1"]).to_numpy()
    result = clustered_bootstrap(frame, {0.25: selected}, samples=200, seed=5, tolerance=1e-12)
    low, high = result[0.25]["stable2_opportunity_rate"]
    assert np.isclose(low, 1.0) and np.isclose(high, 1.0)


def test_matched_random_keeps_selection_count_and_returns_all_metrics() -> None:
    frame = synthetic_frame()
    selected = np.zeros(len(frame), dtype=bool)
    selected[::4] = True
    result = matched_random_baselines(frame, {0.25: selected}, samples=25, seed=11, tolerance=1e-12)
    assert set(result[0.25]) == set(METRICS)
    assert all(len(value) == 3 for value in result[0.25].values())


def test_generated_part2_svgs_are_parseable_when_present() -> None:
    root = Path("outputs/complementarity_identifiability/part2_selectivity_audit")
    for path in root.glob("figure*.svg"):
        ET.parse(path)
