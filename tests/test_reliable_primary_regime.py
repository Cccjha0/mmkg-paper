from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_reliable_primary_regime.py"
SPEC = importlib.util.spec_from_file_location("reliable_primary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_frame(seed_deltas: dict[int, float], triples: int = 20) -> pd.DataFrame:
    rows = []
    for triple in range(triples):
        for seed, delta in seed_deltas.items():
            for direction in ("head", "tail"):
                rows.append(
                    {
                        "raw_triple_id": f"h={triple}|r=0|t={triple + 1}",
                        "seed": seed,
                        "direction": direction,
                        "rr_a": 0.5 + delta,
                        "rr_b": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def test_reliable_requires_all_three_conditions() -> None:
    frame = synthetic_frame({1: 0.03, 2: 0.02, 3: 0.01})
    row, inference = MODULE.audit_pair(frame, "A", "B", 500, 7)
    assert row["selected_primary"] == "A"
    assert row["all_3_seeds_positive"] is True
    assert row["ci95_low"] > 0.0
    assert row["regime"] == "reliable-primary"
    assert inference["cluster_unit"].startswith("original raw triple")


def test_one_nonpositive_seed_forces_boundary() -> None:
    frame = synthetic_frame({1: 0.04, 2: 0.04, 3: -0.01})
    row, _ = MODULE.audit_pair(frame, "A", "B", 500, 7)
    assert row["dev_delta"] > 0.0
    assert row["positive_seed_count"] == 2
    assert row["all_3_seeds_positive"] is False
    assert row["regime"] == "boundary / non-reliable-primary"


def test_clustered_bootstrap_keeps_cluster_means() -> None:
    frame = synthetic_frame({1: 0.02, 2: 0.02, 3: 0.02}, triples=8)
    frame["delta"] = frame["rr_a"] - frame["rr_b"]
    low, high = MODULE.clustered_delta_ci(frame, "delta", 200, 17)
    assert np.isclose(low, 0.02)
    assert np.isclose(high, 0.02)
