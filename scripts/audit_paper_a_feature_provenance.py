"""Check returned source fingerprints against the pre-existing C01--C03 locks.

This validates the server report, not locally absent HDF5 bytes. It never infers
an encoder from filenames, shapes, an empty attribute map, or a success flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "outputs/paper_a_safe_correction/data_checkpoint_review_v1"
RETURN = ROOT / "outputs/paper_a_safe_correction/data_checkpoint_feature_provenance.json"
KINDS = {
    "mkg_w": {"image_h5", "text_h5", "feature_key_map"},
    "db15k": {"image_h5", "text_h5", "same_as"},
}


def check_report(payload, manifests, source_lock, download_evidence=None):
    assert payload["status"] == "source_hashes_verified"
    assert payload["scorers_executed"] is False
    assert payload["test_used_for_selection"] is False
    expected_keys = {(ds, kind) for ds, kinds in KINDS.items() for kind in kinds}
    records = payload["records"]
    keys = [(r["dataset"], r["kind"]) for r in records]
    assert len(keys) == len(set(keys)), "Duplicate dataset/kind record"
    assert set(keys) == expected_keys, "Missing or unexpected source record"
    checked, sampled = [], 0
    root_nonempty, object_nonempty = 0, 0
    for record in records:
        ds, kind = record["dataset"], record["kind"]
        manifest = manifests[ds]
        expected = manifest["source_lock"]["verified_files"][kind]
        locked = source_lock["datasets"][ds]["files"][kind]
        assert expected["sha256"] == locked["sha256"]
        assert expected["bytes"] == locked["bytes"]
        assert record["exists"] is True and record["hash_matches"] is True
        assert record["sha256"] == record["expected_sha256"] == locked["sha256"], "Source fingerprint mismatch"
        assert record["bytes"] == locked["bytes"], "Source size mismatch"
        assert PureWindowsPath(record["path"]).name == PureWindowsPath(expected["path"]).name
        row = {"dataset": ds, "kind": kind, "sha256": record["sha256"],
               "bytes": record["bytes"], "prior_lock_matches": True}
        if kind.endswith("_h5"):
            modality = "image" if kind == "image_h5" else "text"
            feature = manifest["features"][modality]
            assert record["root_key_count"] == feature["hdf5_keys"], "HDF5 key count mismatch"
            attrs = record["root_attributes"]
            assert isinstance(attrs, dict)
            objects = record["first_five_objects"]
            assert len(objects) == min(5, record["root_key_count"])
            assert len({item["key"] for item in objects}) == len(objects)
            for item in objects:
                assert isinstance(item["key"], str)
                shape = item["shape"]
                assert isinstance(shape, list) and len(shape) == 2
                assert all(type(v) is int and v > 0 for v in shape)
                assert shape[1] == feature["dimension"], "Sampled feature dimension mismatch"
                assert isinstance(item["attributes"], dict)
                object_nonempty += bool(item["attributes"])
            sampled += len(objects)
            root_nonempty += bool(attrs)
            row.update(root_key_count=record["root_key_count"], sampled_objects=len(objects),
                       feature_dimension=feature["dimension"], root_attributes_empty=not bool(attrs),
                       sampled_attributes_empty=all(not item["attributes"] for item in objects))
        checked.append(row)
    source_documented = False
    if download_evidence is not None:
        evidence = download_evidence
        assert evidence["version"] == "feature_download_source_evidence_v1"
        assert evidence["maintainer_statement"]["direct_h5_download"] is True
        folder = evidence["folder"]["url"]
        assert evidence["maintainer_statement"]["folder_url"] == folder
        assert evidence["public_repository"]["observed_download_url"].split("?")[0] == folder
        assert evidence["public_repository"]["url"] == "https://github.com/quqxui/MMRNS"
        assert len(evidence["public_repository"]["commit"]) == 40
        files = evidence["files"]
        file_keys = [(r["dataset"], r["kind"]) for r in files]
        assert len(file_keys) == 4 and set(file_keys) == {(ds, k) for ds, k in expected_keys if k.endswith("_h5")}
        for file in files:
            expected = manifests[file["dataset"]]["source_lock"]["verified_files"][file["kind"]]
            assert file["filename"] == PureWindowsPath(expected["path"]).name
            assert file["server_bytes"] == expected["bytes"]
            assert file["server_sha256"] == expected["sha256"]
            assert file["url"] == "https://drive.google.com/file/d/" + file["drive_file_id"] + "/view"
            assert abs(float(file["displayed_size"].removesuffix(" MB")) - expected["bytes"] / 1024**2) <= 0.05
            assert file["remote_content_sha256_verified"] is False
        assert evidence["encoder_version_established"] is False
        source_documented = True
    return {"version": "feature_provenance_return_review_v1",
            "status": "source_identity_evidence_checks_passed",
            "source_files_checked": len(records), "hdf5_files_checked": 4,
            "sampled_objects_checked": sampled,
            "nonempty_root_attribute_maps": root_nonempty,
            "nonempty_sampled_object_attribute_maps": object_nonempty,
            "encoder_provenance_established": False,
            "released_feature_download_source_documented": source_documented,
            "raw_files_rehashed_locally": False,
            "scorers_executed": False, "test_used_for_selection": False,
            "checks": checked, "failures": [],
            "evidence_scope": "Server-reported file hashes/sizes are independently compared with the prior source lock and training manifests. Source bytes remain on the server.",
            "metadata_scope": "Root attributes plus five objects per HDF5; no assertion about uninspected objects or original encoding/download history.",
            "remaining_requirement": "Original encoder/checkpoint/version and extraction settings are unresolved; this limits extraction-level reproducibility, not the documented use of released frozen feature files."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RETURN)
    parser.add_argument("--output", type=Path, default=REVIEW / "feature_provenance_audit.json")
    args = parser.parse_args()
    paths = [args.input, ROOT / "docs/EXTERNAL_SOURCES_LOCK.json"]
    payload, lock = [json.loads(p.read_text(encoding="utf-8-sig")) for p in paths]
    manifests = {}
    for ds in KINDS:
        path = REVIEW / f"{ds}_split_manifest.json"
        paths.append(path)
        manifests[ds] = json.loads(path.read_text(encoding="utf-8"))["training_manifest"]
    download_path = REVIEW / "feature_download_source.json"
    download = json.loads(download_path.read_text(encoding="utf-8"))
    paths.append(download_path)
    result = check_report(payload, manifests, lock, download)
    paths.extend([Path(__file__), ROOT / "scripts/collect_paper_a_feature_provenance.py"])
    result["sources"] = {p.resolve().relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "source_files_checked", "hdf5_files_checked",
                                                  "sampled_objects_checked", "encoder_provenance_established")}))


if __name__ == "__main__":
    main()
