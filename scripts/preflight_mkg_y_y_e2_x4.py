from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp2_information_common import load_contract, portable_path, reject_test_path, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify every frozen MKG-Y Y-E2 X4 input boundary without model inference.")
    parser.add_argument("--contract", default="docs/protocols/MKG_Y_Y_E2_X4_OOF_CONTRACT.json")
    parser.add_argument("--output", default="outputs/complementarity_identifiability/mkg_y_y_e2_information/preflight.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def records(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value["path"], value["sha256"]
        for nested in value.values():
            yield from records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from records(nested)


def verify(path: Path, expected: str) -> None:
    reject_test_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected:
        raise RuntimeError(f"Frozen source hash mismatch: {path}")


def main() -> None:
    args = parse_args()
    contract_path, output_path = Path(args.contract), Path(args.output)
    reject_test_path(contract_path)
    reject_test_path(output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    contract = load_contract(contract_path)
    prerequisite = contract["prerequisite"]
    verify(Path(prerequisite["manifest"]), prerequisite["sha256"])
    assessment = json.loads(Path(prerequisite["manifest"]).read_text(encoding="utf-8"))
    if assessment.get("assessment", {}).get("outcome") != prerequisite["required_assessment"]:
        raise RuntimeError("Y-E1 prerequisite assessment mismatch")
    verified = {}
    for path_string, expected in records(contract["source_boundaries"]):
        path = Path(path_string)
        verify(path, expected)
        verified[portable_path(path)] = expected
    for pair_id in contract["pair_ids"]:
        boundary = contract["source_boundaries"]["full_ranking_manifests"][pair_id]
        manifest = json.loads(Path(boundary["path"]).read_text(encoding="utf-8"))
        pair = manifest.get("pair", {})
        if (
            pair.get("pair_id") != pair_id
            or pair.get("status") != "PASS"
            or pair.get("filter_fact_scope") != "train_dev"
            or int(pair.get("test_rows_opened", -1)) != 0
        ):
            raise RuntimeError(f"Invalid frozen Y2 boundary for {pair_id}")
        for path_string, expected in manifest.get("files", {}).items():
            verify(Path(path_string), expected)
            verified[portable_path(Path(path_string))] = expected
        for runs in manifest.get("accepted_runs", {}).values():
            for record in runs.values():
                checkpoint = Path(record["run_dir"]) / "best.ckpt"
                verify(checkpoint, record["checkpoint_sha256"])
                verified[portable_path(checkpoint)] = record["checkpoint_sha256"]
    payload = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "status": "PASS",
        "contract": {"path": portable_path(contract_path), "sha256": sha256_file(contract_path)},
        "verified_source_count": len(verified),
        "split": "dev",
        "dataset": "mkg_y",
        "pairs": contract["pair_ids"],
        "representation": "X4",
        "x6_run": 0,
        "test_access": 0,
        "checkpoint_inference": 0,
        "learner_training": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
