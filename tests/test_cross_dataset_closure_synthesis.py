from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "synthesize_cross_dataset_closure", ROOT / "scripts" / "synthesize_cross_dataset_closure.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_pair_evidence_invariants() -> None:
    contract, inventory = MODULE.load_contract(
        ROOT / "docs" / "protocols" / "CROSS_DATASET_COMPLEMENTARITY_CLOSURE_SYNTHESIS_CONTRACT.json"
    )
    MODULE.validate_frozen_status(contract)
    pairs = MODULE.build_pair_evidence(contract)
    assert len(inventory) == len(contract["sources"])
    assert len(pairs) == 9
    assert set(pairs.dataset) == {"mkg_w", "db15k", "mkg_y"}
    assert pairs.groupby("dataset").size().eq(3).all()
    assert (pairs.available_headroom > 0).all()
    assert (pairs.stable2_ci95_low > 0).all()
    assert (pairs.stable3_ci95_low > 0).all()
    assert (pairs.loso_ci95_low > 0).all()


def test_summary_reconciles_to_pair_table() -> None:
    contract, _ = MODULE.load_contract(
        ROOT / "docs" / "protocols" / "CROSS_DATASET_COMPLEMENTARITY_CLOSURE_SYNTHESIS_CONTRACT.json"
    )
    pairs = MODULE.build_pair_evidence(contract)
    summary = MODULE.build_dataset_summary(pairs)
    assert len(summary) == 3
    for row in summary.itertuples(index=False):
        group = pairs.loc[pairs.dataset == row.dataset]
        assert np.isclose(row.median_x4_recovery, group.x4_recovery.median())
        assert row.local_signal_pairs == int(group.local_signal_pair.sum())
