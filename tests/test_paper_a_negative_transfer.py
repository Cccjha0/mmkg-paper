from __future__ import annotations

import math

import pandas as pd

from scripts.analyze_paper_a_negative_transfer import (
    METHOD_ORDER,
    clustered_bootstrap,
    point_summary,
    summarize,
)


def synthetic_rows() -> pd.DataFrame:
    rows = []
    deltas = {
        "h=1|r=0|t=2": (0.2, -0.1, 0.0, 0.0),
        "h=3|r=0|t=4": (-0.4, 0.1, 0.0, 0.0),
    }
    for triple, values in deltas.items():
        for index, delta in enumerate(values):
            rows.append(
                {
                    "split": "test",
                    "dataset": "Toy",
                    "pair": "A + B",
                    "method": "Query-soft",
                    "seed": index // 2 + 1,
                    "direction": "head" if index % 2 == 0 else "tail",
                    "raw_triple_id": triple,
                    "rr_global": 0.5,
                    "rr_method": 0.5 + delta,
                    "delta_rr": delta,
                }
            )
    return pd.DataFrame(rows)


def test_point_summary_uses_query_level_outcomes() -> None:
    result = point_summary(synthetic_rows())

    assert result["n_observations"] == 8
    assert result["n_original_triple_clusters"] == 2
    assert result["n_harmful"] == 2
    assert result["n_beneficial"] == 2
    assert result["n_unchanged"] == 4
    assert math.isclose(result["delta_mrr_vs_global"], -0.025)
    assert math.isclose(result["harmful_query_rate"], 0.25)
    assert math.isclose(result["mean_harm"], 0.25)
    assert math.isclose(result["mean_benefit"], 0.15)


def test_clustered_bootstrap_resamples_raw_triples() -> None:
    result = clustered_bootstrap(synthetic_rows(), samples=500, seed=7)

    assert result["n_original_triple_clusters"] == 2
    assert result["bootstrap_samples"] == 500
    assert result["cluster_unit"].startswith("original raw triple")
    assert len(result["delta_mrr_ci95"]) == 2
    assert len(result["harmful_query_rate_ci95"]) == 2
    assert len(result["mean_harm_ci95"]) == 2


def test_summarize_ignores_unobserved_categorical_methods() -> None:
    frame = synthetic_rows()
    frame["method"] = pd.Categorical(
        frame["method"], categories=METHOD_ORDER, ordered=True
    )

    result, inference = summarize(
        frame,
        "pooled",
        ["split", "dataset", "pair", "method"],
        bootstrap_samples=100,
        bootstrap_seed=11,
    )

    assert result["method"].tolist() == ["Query-soft"]
    assert len(inference) == 1

