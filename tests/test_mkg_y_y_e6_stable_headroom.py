from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.audit_mkg_y_y_e6_stable_headroom as audit
import scripts.audit_stable_headroom as core


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/protocols/MKG_Y_Y_E6_STABLE_HEADROOM_CONTRACT.json"
RUNNER = ROOT / "scripts/run_mkg_y_y_e6_stable_headroom.ps1"


def test_y_e6_contract_is_frozen_descriptive_dev_only() -> None:
    contract = audit.load_contract(CONTRACT)
    assert contract["split"] == "dev"
    assert tuple(contract["pair_ids"]) == audit.PAIR_IDS
    assert contract["assessment"]["classification"] is False
    assert contract["assessment"]["route_selection"] is False
    assert contract["assessment"]["outcome"] == audit.OUTCOME
    assert contract["test_access"] == 0


def test_stable_action_tie_break_and_empty_fallback() -> None:
    mean_rr = np.asarray([[0.2, 0.3, 0.3], [0.2, 0.3, 0.4]])
    allowed = np.asarray([[True, True, True], [False, False, False]])
    old_alphas = core.ALPHAS
    core.ALPHAS = np.asarray([0.0, 0.5, 1.0])
    try:
        chosen = core.choose_allowed_action(mean_rr, allowed, global_index=1)
    finally:
        core.ALPHAS = old_alphas
    assert chosen.tolist() == [1, 1]


def test_y_e6_runner_has_no_gpu_or_test_execution() -> None:
    script = RUNNER.read_text(encoding="utf-8").lower()
    assert "audit_mkg_y_y_e6_stable_headroom.py" in script
    assert "cuda" not in script
    assert '"--split", "test"' not in script
    assert "checkpoint" not in script
