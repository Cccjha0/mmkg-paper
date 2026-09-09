from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_paper_a_boundary_pairs.py"
SPEC = importlib.util.spec_from_file_location("boundary_pairs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_safety_and_alpha_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "rr_global": [0.5, 0.5, 0.5, 0.5],
            "rr_method": [0.6, 0.4, 0.5, 0.3],
            "alpha_method": [0.7, 0.5, 0.6, 0.4],
            "alpha_global_locked": [0.6, 0.6, 0.6, 0.6],
            "fallback": [0, 1, 0, 1],
        }
    )
    result = MODULE.summarize_method(
        frame, "Method", "rr_method", "alpha_method", "fallback"
    )
    assert np.isclose(result["mrr"], 0.45)
    assert np.isclose(result["delta_vs_global"], -0.05)
    assert np.isclose(result["harm_rate"], 0.5)
    assert np.isclose(result["benefit_rate"], 0.25)
    assert np.isclose(result["mean_harm"], 0.15)
    assert np.isclose(result["fallback_rate"], 0.5)
    assert np.isclose(result["changed_alpha_rate"], 0.75)
    assert np.isclose(result["mean_abs_alpha_deviation"], 0.1)


def test_static_method_has_no_fallback_rate() -> None:
    frame = pd.DataFrame(
        {
            "rr_global": [0.5, 0.5],
            "rr_method": [0.5, 0.5],
            "alpha_method": [0.5, 0.5],
            "alpha_global_locked": [0.5, 0.5],
        }
    )
    result = MODULE.summarize_method(
        frame, "Static", "rr_method", "alpha_method", None
    )
    assert result["fallback_rate"] is None
    assert result["harm_rate"] == 0.0
    assert result["mean_harm"] is None
