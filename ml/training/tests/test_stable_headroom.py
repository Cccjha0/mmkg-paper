from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_stable_headroom import (
    FINAL_INTERPRETATIONS,
    GAIN_CLASSIFICATIONS,
    STABLE_CLASSIFICATIONS,
    bootstrap_part_a,
    choose_allowed_action,
    classify_concentration,
    classify_joint,
    classify_stability,
    concentration_metrics,
)
from scripts.exp2_information_common import ALPHAS


def contract() -> dict:
    return json.loads(Path("docs/protocols/EXP6_STABLE_HEADROOM_CONTRACT.json").read_text(encoding="utf-8"))


def test_allowed_action_tie_break_and_empty_fallback() -> None:
    mean_rr = np.zeros((3, len(ALPHAS)), dtype=np.float64)
    allowed = np.zeros_like(mean_rr, dtype=bool)
    allowed[0, [8, 12]] = True
    mean_rr[0, [8, 12]] = 1.0
    allowed[1, [9, 11]] = True
    mean_rr[1, [9, 11]] = 1.0
    chosen = choose_allowed_action(mean_rr, allowed, global_index=10)
    assert chosen.tolist() == [8, 9, 10]


def test_concentration_metrics_use_ceil_and_exact_support() -> None:
    gains = np.asarray([6.0, 2.0, 1.0, 1.0])
    result, order, curve = concentration_metrics(gains, (0.01, 0.10, 0.20))
    assert result["top1_triple_count"] == 1
    assert np.isclose(result["top10_gain_share"], 0.6)
    assert result["q50_triple_count"] == 1
    assert np.isclose(result["q50"], 0.25)
    assert order[0] == 0 and np.isclose(curve[-1], 1.0)
    expected_neff = (10.0**2 / (36.0 + 4.0 + 1.0 + 1.0)) / 4.0
    assert np.isclose(result["effective_support"], expected_neff)


def test_paired_bootstrap_ratios_are_unclipped() -> None:
    arrays = {
        "raw": np.asarray([1.0, 2.0, 3.0]),
        "consensus": np.asarray([0.5, 1.0, 1.5]),
        "stable2": np.asarray([0.25, 0.5, 0.75]),
        "stable3": np.asarray([0.1, 0.2, 0.3]),
        "consensus_changed_rate": np.asarray([0.0, 0.5, 1.0]),
        "stable2_opportunity_rate": np.asarray([1.0, 1.0, 1.0]),
        "stable3_opportunity_rate": np.asarray([0.0, 0.0, 0.0]),
    }
    _, ratios = bootstrap_part_a(arrays, samples=200, seed=7)
    assert np.allclose(ratios["consensus"], (0.5, 0.5))
    assert np.allclose(ratios["stable2"], (0.25, 0.25))
    assert np.allclose(ratios["stable3"], (0.1, 0.1))


def synthetic_summary() -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": ["mkg_w"] * 3 + ["db15k"] * 3,
        "stable2_ci95_low": [.01] * 4 + [-.01] * 2,
        "stable2_recovery": [.30] * 6,
        "stable3_ci95_low": [.01, -.01, -.01, .01, .01, -.01],
        "stable3_recovery": [.12] * 6,
        "top10_gain_share": [.6] * 4 + [.4] * 2,
        "q50": [.08] * 4 + [.2] * 2,
    })


def test_stable_high_and_high_concentration_gates() -> None:
    frame = synthetic_summary()
    stable, stable_evidence = classify_stability(frame, contract())
    gain, gain_evidence = classify_concentration(frame, contract())
    assert stable == "E6_STABLE_HIGH" and stable in STABLE_CLASSIFICATIONS
    assert stable_evidence["stable_high_gate"] is True
    assert gain == "E6_GAIN_HIGHLY_CONCENTRATED" and gain in GAIN_CLASSIFICATIONS
    assert gain_evidence["highly_concentrated_gate"] is True


def test_joint_selective_and_mixed_routes() -> None:
    e4_payload = {"classification": "E4_INTERMEDIATE"}
    e4_summary = pd.DataFrame({
        "dataset": ["mkg_w"] * 3 + ["db15k"] * 3,
        "loso_ci95_low": [.01] * 6,
        "loso_recovery": [.2] * 6,
    })
    e5_payload = {"final_classification": "E5_INTERMEDIATE", "evidence": {"x6_recovery": False}}
    decision, _ = classify_joint(e4_payload, e4_summary, e5_payload, "E6_STABLE_INTERMEDIATE", "E6_GAIN_HIGHLY_CONCENTRATED", contract())
    assert decision == "FINAL_SELECTIVE_RARE_OPPORTUNITY"
    decision, _ = classify_joint(e4_payload, e4_summary, e5_payload, "E6_STABLE_COLLAPSE", "E6_GAIN_HIGHLY_CONCENTRATED", contract())
    assert decision == "FINAL_MIXED_LIMITS"
    assert decision in FINAL_INTERPRETATIONS


def test_generated_exp6_svgs_are_parseable_when_present() -> None:
    root = Path("outputs/complementarity_identifiability/exp6_stable_headroom")
    for path in root.glob("figure*.svg"):
        ET.parse(path)
