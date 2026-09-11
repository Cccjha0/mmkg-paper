"""Optional server-side, read-only source-feature metadata collection for C02.

Reads checksums and HDF5 attributes only; never trains a model or reads TEST scores.
An absent encoder attribute remains unknown. No provenance is inferred from dimensions.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_a_safe_correction/data_checkpoint_feature_provenance.json")
    args = parser.parse_args()
    import h5py
    records = []
    for dataset in ("mkg_w", "db15k"):
        manifest_path = ROOT / "outputs/paper_a_safe_correction/data_checkpoint_review_v1" / f"{dataset}_split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["training_manifest"]
        for kind, source in manifest["source_lock"]["verified_files"].items():
            if kind not in ("image_h5", "text_h5", "same_as", "feature_key_map"):
                continue
            path = ROOT / source["path"].replace("\\", "/")
            record = {"dataset": dataset, "kind": kind, "path": str(path), "exists": path.is_file(),
                      "expected_sha256": source["sha256"]}
            if path.is_file():
                record.update(bytes=path.stat().st_size, sha256=sha(path))
                record["hash_matches"] = record["sha256"] == record["expected_sha256"]
                if kind.endswith("_h5"):
                    with h5py.File(path, "r") as handle:
                        record["root_attributes"] = {str(k): str(v)[:2000] for k, v in handle.attrs.items()}
                        record["root_key_count"] = len(handle)
                        record["first_five_objects"] = []
                        for name in list(handle.keys())[:5]:
                            item = handle[name]
                            record["first_five_objects"].append({"key": name,
                                "shape": list(item.shape) if isinstance(item, h5py.Dataset) else None,
                                "attributes": {str(k): str(v)[:2000] for k, v in item.attrs.items()}})
            records.append(record)
    passed = all(r.get("hash_matches", False) for r in records)
    payload = {"status": "source_hashes_verified" if passed else "missing_or_mismatched_source",
               "records": records, "scorers_executed": False, "test_used_for_selection": False,
               "interpretation": "Attributes are evidence only. Empty metadata does not identify an encoder or its download origin."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
