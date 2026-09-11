"""Execution binding tests: changing a TEST policy cannot preserve audit success."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("history_audit", ROOT / "scripts/audit_paper_a_test_history.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    base = ROOT / "outputs/paper_a_safe_correction/information_boundary_v2/mkgw_mhyper_native"
    lock = json.loads((base / "dev_lock/anchored_dev_lock.json").read_text())
    summary = json.loads((base / "test_anchored/test_locked_summary.json").read_text())
    model = (base / "dev_lock/anchored_model.pkl").read_bytes()
    return lock, summary, model


def test_recorded_binding_is_valid_without_claiming_chronology():
    lock, summary, model = fixture()
    assert audit.verify_policy_binding(lock, summary, model) is None
    # These actual receipts have no trustworthy first-human-view timestamp.
    assert "first_human_test_inspection_time" not in lock


@pytest.mark.parametrize("field,value", [("beta", 0.5), ("confidence_threshold", 0.3),
                                         ("query_geometry_fields", ["target_rank"])])
def test_changed_test_policy_is_rejected(field, value):
    lock, summary, model = fixture()
    altered = copy.deepcopy(summary)
    altered["policy"][field] = value
    with pytest.raises(ValueError, match="complete DEV policy"):
        audit.verify_policy_binding(lock, altered, model)


def test_changed_model_is_rejected():
    lock, summary, model = fixture()
    with pytest.raises(ValueError, match="preference model"):
        audit.verify_policy_binding(lock, summary, model + b"tampered")


def test_dev_receipt_cannot_stand_in_for_test():
    lock, summary, model = fixture()
    summary["split"] = "dev"
    with pytest.raises(ValueError, match="TEST application"):
        audit.verify_policy_binding(lock, summary, model)


def test_frozen_exposure_snapshot_records_both_datasets():
    evidence = ROOT / "outputs/paper_a_safe_correction/test_history_review_v1"
    timeline = json.loads((evidence / "timeline.json").read_text())
    row = next(r for r in timeline if r["id"] == "E07")
    entry = next(e for e in row["evidence"] if e["repository_path"].endswith("DEV_PROTOCOL_FREEZE.md"))
    data = (ROOT / entry["snapshot"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert b"MKG-W and DB15K TEST results have already been inspected" in data
