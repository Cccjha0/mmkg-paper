from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/protocols/MKG_Y_Y_E4_CROSS_SEED_TRANSFER_CONTRACT.json"
OUTPUT_ROOT = ROOT / "outputs/complementarity_identifiability/mkg_y_y_e4_cross_seed_transfer"


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_freezes_direct_transfer_and_loso_only() -> None:
    contract = load_contract()
    assert contract["status"] == "frozen_before_systematic_run"
    assert contract["split"] == "dev"
    assert contract["seeds"] == [1, 2, 3]
    assert contract["directions"] == ["head", "tail"]
    assert contract["transfer"]["target_reselection"] is False
    assert contract["transfer"]["negative_utility_clipped"] is False
    assert contract["loso"]["training_seeds"] == 2
    assert contract["assessment"]["is_progression_gate"] is False
    assert contract["test_access"] == 0
    assert contract["new_selector"] == 0


def test_completed_y_e4_reproduces_y_e1_and_reports_stability() -> None:
    if not (OUTPUT_ROOT / "audit_manifest.json").exists():
        return
    summary = pd.read_csv(OUTPUT_ROOT / "pair_summary.csv")
    exp1 = pd.read_csv(ROOT / "outputs/complementarity_identifiability/mkg_y_y_e1_landscape/pair_statistics.csv")
    merged = summary.merge(exp1[["pair_id", "available_headroom"]], on="pair_id", validate="one_to_one")
    assert len(summary) == 3
    assert np.allclose(merged.raw_oracle_headroom, merged.available_headroom, rtol=0, atol=1e-12)
    assert (summary.transfer_ci95_low > 0).all()
    assert (summary.loso_ci95_low > 0).all()
    assert ((summary.beneficial_transfer_rate + summary.zero_transfer_rate + summary.harmful_transfer_rate) - 1.0).abs().max() < 1e-12
    audit = json.loads((OUTPUT_ROOT / "audit_manifest.json").read_text(encoding="utf-8"))
    assert audit["assessment"]["outcome"] == "Y_E4_CROSS_SEED_REPLICATION_REPORTED"
    assert audit["operational_audit"]["test_access"] == 0
    assert audit["operational_audit"]["target_reselection"] == 0
    assert audit["next_step_started"] == 0


def test_runner_has_no_gpu_or_checkpoint_execution() -> None:
    script = (ROOT / "scripts/run_mkg_y_y_e4_cross_seed_transfer.ps1").read_text(encoding="utf-8").lower()
    assert "audit_mkg_y_y_e4_cross_seed_transfer.py" in script
    assert "cuda" not in script
    assert "full_ranking" not in script
    assert '"--split", "test"' not in script
