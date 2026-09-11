import copy
import json

import pytest

from scripts.audit_paper_a_feature_provenance import KINDS, RETURN, REVIEW, ROOT, check_report


def inputs():
    payload = json.loads(RETURN.read_text(encoding="utf-8-sig"))
    manifests = {ds: json.loads((REVIEW / f"{ds}_split_manifest.json").read_text(encoding="utf-8"))["training_manifest"] for ds in KINDS}
    lock = json.loads((ROOT / "docs/EXTERNAL_SOURCES_LOCK.json").read_text())
    return copy.deepcopy(payload), manifests, lock


def test_matching_return_confirms_identity_but_not_encoder():
    result = check_report(*inputs())
    assert result["source_files_checked"] == 6
    assert result["sampled_objects_checked"] == 20
    assert result["nonempty_root_attribute_maps"] == 0
    assert result["nonempty_sampled_object_attribute_maps"] == 0
    assert result["encoder_provenance_established"] is False
    assert result["raw_files_rehashed_locally"] is False


def test_official_download_source_does_not_identify_encoder():
    evidence = json.loads((REVIEW / "feature_download_source.json").read_text(encoding="utf-8"))
    result = check_report(*inputs(), download_evidence=evidence)
    assert result["released_feature_download_source_documented"] is True
    assert result["encoder_provenance_established"] is False


@pytest.mark.parametrize("field", ["folder_url", "server_sha256"])
def test_mismatched_download_source_is_rejected(field):
    evidence = json.loads((REVIEW / "feature_download_source.json").read_text(encoding="utf-8"))
    if field == "folder_url":
        evidence["maintainer_statement"][field] = "https://example.com/unrelated"
    else:
        evidence["files"][0][field] = "0" * 64
    with pytest.raises(AssertionError):
        check_report(*inputs(), download_evidence=evidence)


@pytest.mark.parametrize("corruption", ["self_attested_hash", "bytes", "missing", "duplicate", "dimension", "key_count", "test_selection"])
def test_inconsistent_return_is_rejected(corruption):
    payload, manifests, lock = inputs()
    record = next(r for r in payload["records"] if r["kind"] == "text_h5")
    if corruption == "self_attested_hash":
        record["sha256"] = record["expected_sha256"] = "0" * 64
    elif corruption == "bytes":
        record["bytes"] += 1
    elif corruption == "missing":
        payload["records"].pop()
    elif corruption == "duplicate":
        payload["records"].append(copy.deepcopy(payload["records"][0]))
    elif corruption == "dimension":
        record["first_five_objects"][0]["shape"][1] = 768
    elif corruption == "key_count":
        record["root_key_count"] += 1
    else:
        payload["test_used_for_selection"] = True
    with pytest.raises(AssertionError):
        check_report(payload, manifests, lock)
