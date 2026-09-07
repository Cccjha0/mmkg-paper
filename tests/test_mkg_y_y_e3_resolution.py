from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/protocols/MKG_Y_Y_E3_ADAPTIVE_RESOLUTION_CONTRACT.json"
OUTPUT_ROOT = ROOT / "outputs/complementarity_identifiability/mkg_y_y_e3_resolution"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def iter_declared_sources(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for nested in value.values():
            yield from iter_declared_sources(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_declared_sources(nested)


def test_contract_freezes_y_e3_without_gate_or_new_selector() -> None:
    contract = load_contract()
    assert contract["status"] == "frozen_before_systematic_run"
    assert contract["split"] == "dev"
    assert contract["pair_ids"] == [
        "mkg_y_mhyper_native",
        "mkg_y_mhyper_adamf",
        "mkg_y_native_adamf",
    ]
    assert contract["outer_cv"]["folds"] == 5
    assert contract["outer_cv"]["grouping_key"] == "original_triple_id"
    assert contract["inner_cv"]["folds"] == 3
    assert contract["levels"]["L4"]["minimum_support_candidates"] == [10, 25, 50, 100]
    assert contract["levels"]["L5"]["new_training"] is False
    assert contract["assessment"]["is_progression_gate"] is False
    assert contract["assessment"]["route_selection"] is False
    assert contract["test_access"] == 0
    assert contract["new_query_selector"] == 0


def test_direct_source_boundaries_match_or_normalize_to_disk() -> None:
    contract = load_contract()
    for source in iter_declared_sources(contract["source_boundaries"]):
        path = ROOT / source["path"]
        actual = sha256(path)
        if actual != source["sha256"]:
            normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            assert hashlib.sha256(normalized).hexdigest() == source["sha256"]


def test_runner_has_only_preflight_and_cpu_resolution_run() -> None:
    script = (ROOT / "scripts/run_mkg_y_y_e3_resolution.ps1").read_text(encoding="utf-8")
    assert '[ValidateSet("Preflight", "Run")]' in script
    assert "audit_mkg_y_y_e3_resolution.py" in script
    assert "cuda" not in script.lower()
    assert '"--split", "test"' not in script.lower()


def test_completed_audit_preserves_protocol_boundaries() -> None:
    if not (OUTPUT_ROOT / "audit_manifest.json").exists():
        return
    audit = json.loads((OUTPUT_ROOT / "audit_manifest.json").read_text(encoding="utf-8"))
    assessment = json.loads((OUTPUT_ROOT / "replication_assessment.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUTPUT_ROOT / "pair_level_metrics.csv")
    assert audit["split"] == "dev"
    assert audit["operational_audit"]["test_access"] == 0
    assert audit["operational_audit"]["outer_triple_leakage"] == 0
    assert audit["operational_audit"]["new_l5_training"] == 0
    assert audit["next_step_started"] == 0
    assert assessment["outcome"] == "Y_E3_RESOLUTION_REPLICATION_REPORTED"
    assert assessment["is_progression_gate"] is False
    assert len(metrics) == 18
    assert set(metrics["level"]) == {"L0", "L1", "L2", "L3", "L4", "L5"}
    assert np.allclose(metrics.loc[metrics.level == "L0", "delta_mrr"], 0.0)
