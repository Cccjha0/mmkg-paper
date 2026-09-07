from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_cross_seed_transfer import (
    CLASSIFICATIONS,
    action_indices,
    bootstrap_joint,
    classify,
)
from scripts.exp2_information_common import ALPHAS


def test_oracle_tie_break_prefers_nearest_alpha0_then_smaller() -> None:
    rr = np.zeros((3, len(ALPHAS)), dtype=np.float64)
    global_index = 10
    rr[0, [8, 12]] = 1.0
    rr[1, [9, 11]] = 2.0
    rr[2, [10, 11]] = 3.0
    chosen = action_indices(rr, global_index)
    assert chosen.tolist() == [8, 9, 10]


def test_joint_bootstrap_ratio_preserves_paired_clusters() -> None:
    arrays = {
        "raw": np.asarray([1.0, 2.0, 3.0]),
        "transfer": np.asarray([0.25, 0.50, 0.75]),
        "loso": np.asarray([0.5, 1.0, 1.5]),
    }
    intervals, ratios = bootstrap_joint(arrays, samples=200, seed=17)
    assert intervals["raw"][0] > 0
    assert np.allclose(ratios[("transfer", "raw")], (0.25, 0.25))
    assert np.allclose(ratios[("loso", "raw")], (0.5, 0.5))


def _summary(transfer_low: list[float], recovery: list[float], loso_low: list[float], loso_recovery: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": ["mkg_w"] * 3 + ["db15k"] * 3,
        "transfer_ci95_low": transfer_low,
        "transfer_recovery": recovery,
        "loso_ci95_low": loso_low,
        "loso_recovery": loso_recovery,
    })


def _contract() -> dict:
    return json.loads(Path("docs/protocols/EXP4_CROSS_SEED_TRANSFER_CONTRACT.json").read_text(encoding="utf-8"))


def test_substantially_transferable_gate() -> None:
    frame = _summary([.01, .01, -.01, .01, .01, -.01], [.3] * 6, [.01] * 4 + [-.01] * 2, [.3] * 6)
    decision, evidence = classify(frame, _contract())
    assert decision == "E4_SUBSTANTIALLY_TRANSFERABLE"
    assert evidence["substantially_transferable_gate"] is True


def test_residual_dominated_gate() -> None:
    frame = _summary([-.01] * 5 + [.01], [.02] * 6, [-.01] * 6, [.03] * 6)
    decision, evidence = classify(frame, _contract())
    assert decision == "E4_RESIDUAL_DOMINATED"
    assert evidence["residual_dominated_gate"] is True


def test_intermediate_fallback() -> None:
    frame = _summary([.01, .01, -.01, -.01, -.01, -.01], [.15] * 6, [.01, .01, -.01, -.01, -.01, -.01], [.15] * 6)
    decision, _ = classify(frame, _contract())
    assert decision == "E4_INTERMEDIATE"
    assert decision in CLASSIFICATIONS


def test_generated_exp4_svgs_are_parseable_when_present() -> None:
    root = Path("outputs/complementarity_identifiability/exp4_cross_seed_transfer")
    for path in root.glob("figure*.svg"):
        ET.parse(path)
