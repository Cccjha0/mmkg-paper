from __future__ import annotations

import json
from pathlib import Path

import scripts.audit_local_identifiability as audit


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/protocols/MKG_Y_Y_E5_LOCAL_IDENTIFIABILITY_CONTRACT.json"
RUNNER = ROOT / "scripts/run_mkg_y_y_e5_local_identifiability.ps1"


def test_y_e5_contract_is_frozen_dev_only() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen_before_systematic_run"
    assert contract["split"] == "dev"
    assert contract["dataset"] == "mkg_y"
    assert contract["pair_ids"] == list(audit.Y_PAIR_IDS)
    assert contract["k_values"] == [5, 10, 20, 50]
    assert contract["representations"]["X4"]["dimension"] == 40
    assert contract["representations"]["X6"]["status"] == "X6_FIXED_VECTOR_UNAVAILABLE"
    assert contract["assessment"]["classification"] is False
    assert contract["test_access"] == 0
    assert contract["new_selector"] == 0
    assert contract["checkpoint_execution"] == 0


def test_y_profile_restores_core_profile() -> None:
    audit.configure_profile("mkg_y")
    try:
        contract = audit.load_contract(CONTRACT)
        assert tuple(contract["pair_ids"]) == audit.Y_PAIR_IDS
    finally:
        audit.configure_profile("core")
    assert audit.PAIR_IDS == audit.CORE_PAIR_IDS


def test_y_e5_runner_keeps_boundaries() -> None:
    script = RUNNER.read_text(encoding="utf-8").lower()
    assert "audit_local_identifiability.py" in script
    assert '"--profile", "mkg_y"' in script
    assert "[int]$centerbatchsize = 4096" in script
    assert '"--split", "test"' not in script
    assert "full_ranking" not in script
