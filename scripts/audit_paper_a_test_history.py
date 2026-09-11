"""C07/C08: reconstruct recorded history without inferring blind TEST access.

This is a retrospective evidence audit, not a preregistration or an experiment.
Git dates describe recorded commits; they are not trusted execution/access times.
"""
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_a_safe_correction/test_history_review_v1"
SPEC = ROOT / "docs/protocols/paper_a_test_history_review.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def verify_policy_binding(lock, summary, model_bytes):
    """Validate execution identity, deliberately making no chronology inference."""
    if summary.get("split") != "test":
        raise ValueError("Expected a TEST application receipt")
    if summary.get("policy") != lock:
        raise ValueError("TEST receipt differs from the complete DEV policy")
    if digest(model_bytes) != lock["model_sha256"]:
        raise ValueError("Serialized preference model differs from DEV lock")


def main():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots = OUT / "evidence"
    snapshots.mkdir(exist_ok=True)
    sources = {}

    def bind(path):
        rel = path.relative_to(ROOT).as_posix()
        sources[rel] = digest(path.read_bytes())
        return rel

    def save(name, value):
        path = OUT / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
        return path

    timeline = []
    blobs = {}
    for event in spec["events"]:
        revision = git("rev-parse", event["revision"]).decode().strip()
        metadata = git("show", "-s", "--format=%aI%n%cI%n%s", revision).decode().splitlines()
        record = dict(event, revision=revision, author_date=metadata[0],
                      committer_date=metadata[1], commit_subject=metadata[2], evidence=[])
        for index, rel in enumerate(event["paths"], 1):
            data = git("show", f"{revision}:{rel}")
            path = snapshots / f"{event['id']}_{index:02d}_{Path(rel).name}"
            path.write_bytes(data)
            copy = bind(path)
            oid = git("rev-parse", f"{revision}:{rel}").decode().strip()
            record["evidence"].append({"repository_path": rel, "git_blob": oid,
                                       "sha256": digest(data), "snapshot": copy})
            blobs[event["id"], rel] = data
        timeline.append(record)

    # Evidence statements are checked in the pinned versions, not in rewritten prose.
    initial = blobs["E03", "scripts/crossfit_anchored_dynamic.py"].decode()
    p3 = blobs["E04", "scripts/ablate_anchored_dynamic.py"].decode()
    assert 'default="0.05,0.10,0.15,0.20"' in initial
    assert "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50" in p3
    exposure = list(csv.DictReader(io.StringIO(blobs[
        "E07", "docs/protocols/aacpi_test_exposure_manifest.csv"].decode())))
    assert len(exposure) == 6
    assert all(r["test_previously_accessed"].startswith("yes") for r in exposure)
    assert {r["dataset"] for r in exposure} == {"MKG-W", "DB15K"}
    freeze = blobs["E07", "docs/protocols/AACPI_V2_DEV_PROTOCOL_FREEZE.md"].decode()
    assert "TEST results have already been inspected" in freeze
    historical = json.loads(blobs[
        "E06", "outputs/anchored_dynamic/four_pair_summary/four_pair_summary.json"])
    assert len(historical["results"]) == 4
    assert {r["evidence_tier"] for r in historical["results"]} == {
        "confirmatory", "secondary replication"}
    old_audit = json.loads(blobs[
        "E10", "outputs/paper_a_safe_correction/protocol_audit/audit.json"])
    assert old_audit["summary"]["checks"] == old_audit["summary"]["passed"] == 77

    locks = []
    # The old source manifest was committed alongside results. Check its receipts,
    # but never turn that shared commit date into an earlier lock timestamp.
    for item in historical["source_manifest"]:
        if item["role"] != "dev_lock":
            continue
        lock_path = ROOT / item["path"]
        lock_bytes = lock_path.read_bytes()
        assert digest(lock_bytes) == item["sha256"], item["path"]
        lock = json.loads(lock_bytes)
        summary_item = next(r for r in historical["source_manifest"]
                            if r["dataset"] == item["dataset"] and r["pair"] == item["pair"]
                            and r["role"] == "test_locked_summary")
        summary_path = ROOT / summary_item["path"]
        summary_bytes = summary_path.read_bytes()
        assert digest(summary_bytes) == summary_item["sha256"]
        model_path = lock_path.parent / "anchored_model.pkl"
        verify_policy_binding(lock, json.loads(summary_bytes), model_path.read_bytes())
        bind(model_path)
        copies = {}
        for label, data in (("lock", lock_bytes), ("test_receipt", summary_bytes)):
            path = snapshots / f"legacy_{len(locks)+1}_{label}.json"
            path.write_bytes(data)
            copies[label] = bind(path)
        locks.append({"version": "legacy", "dataset": item["dataset"], "pair": item["pair"],
                      "lock_path": item["path"], "lock_sha256": digest(lock_bytes),
                      "test_receipt_sha256": digest(summary_bytes), "model_sha256": lock["model_sha256"],
                      "alpha0": lock["alpha0"], "beta": lock["beta"],
                      "tau": lock["confidence_threshold"], "snapshots": copies,
                      "policy_model_identity_verified": True,
                      "first_tracked_receipt_event": "E06",
                      "independently_timestamped_lock_before_first_test_inspection": False})

    rerun_manifest_path = ROOT / "paper_a_draft/rerun_source_manifest.json"
    rerun_manifest = json.loads(rerun_manifest_path.read_text())
    bind(rerun_manifest_path)
    for pair in spec["corrected_pairs"]:
        folder = ROOT / "outputs/paper_a_safe_correction/information_boundary_v2" / pair
        lock_path = folder / "dev_lock/anchored_dev_lock.json"
        summary_path = folder / "test_anchored/test_locked_summary.json"
        model_path = lock_path.parent / "anchored_model.pkl"
        lock = json.loads(lock_path.read_text())
        for path in (lock_path, summary_path, model_path):
            rel = bind(path)
            assert sources[rel] == rerun_manifest["sources"][rel], rel
        verify_policy_binding(lock, json.loads(summary_path.read_text()), model_path.read_bytes())
        assert lock["score_information_contract"] == "unfiltered_features_and_normalization_v2"
        assert len(lock["query_geometry_fields"]) == 13 and max(lock["beta_grid"]) == 0.5
        locks.append({"version": "information_boundary_v2", "pair": pair,
                      "lock_path": lock_path.relative_to(ROOT).as_posix(),
                      "lock_sha256": digest(lock_path.read_bytes()),
                      "test_receipt_sha256": digest(summary_path.read_bytes()),
                      "model_sha256": lock["model_sha256"], "alpha0": lock["alpha0"],
                      "beta": lock["beta"], "tau": lock["confidence_threshold"],
                      "feature_fields": lock["query_geometry_fields"],
                      "source_selection_sha256": lock["source_selection_json_sha256"],
                      "source_crossfit_sha256": lock["source_crossfit_summary_sha256"],
                      "policy_model_identity_verified": True,
                      "first_tracked_audit_event": "E12",
                      "independently_timestamped_lock_before_first_test_inspection": False})
    assert len(locks) == 10
    for name, data in (("timeline.json", timeline), ("frozen_objects.json", spec["frozen_objects"]),
                       ("policy_bindings.json", locks), ("claim_disposition.json", spec["claims"])):
        bind(save(name, data))
    bind(SPEC)
    bind(Path(__file__))
    audit = {
        "version": "test_history_review_v1", "status": "history_evidence_checks_passed",
        "failures": [], "reviewed_source_tip": git("rev-parse", spec["source_tip"]).decode().strip(),
        "scope": "retrospective documentation; no training, selection or TEST evaluation",
        "timeline_events": len(timeline), "historical_policy_bindings_checked": 4,
        "corrected_policy_bindings_checked": 6, "recorded_prior_exposure_pairs": len(exposure),
        "legacy_integrity_checks": 77,
        "legacy_integrity_check_scope": "execution consistency; not design independence",
        "first_human_test_inspection_time": {"MKG-W": None, "DB15K": None},
        "current_confirmatory_status": {"MKG-W": False, "DB15K": False},
        "minimum_reporting_conditions_met_by": "traceable timeline, frozen-object inventory and claim downgrade",
        "historical_test_influence_on_design_excluded": False,
        "test_used_for_new_selection": False,
        "sources": sources,
    }
    save("audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
