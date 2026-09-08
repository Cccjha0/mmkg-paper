from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the frozen three-dataset closure TEST boundary without opening TEST.")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def verify(record: dict) -> Path:
    path = Path(record["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_lf(path) if "sha256_lf" in record else sha256_file(path)
    expected = record.get("sha256_lf", record.get("sha256"))
    if actual != expected:
        raise RuntimeError(f"Frozen source hash mismatch: {path}; expected={expected}, actual={actual}")
    return path


def selected_configuration(path: Path, pair: dict, contract: dict, hyperparameters: dict) -> tuple[str, dict, int]:
    frame = pd.read_csv(path)
    required = {"outer_fold", "selected_learner", "selected_config", "inner_probe_delta_mrr"}
    if set(frame.outer_fold.astype(int)) != {1, 2, 3, 4, 5} or not required.issubset(frame.columns):
        raise RuntimeError(f"Invalid frozen nested selection inventory: {path}")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        config = json.dumps(json.loads(row.selected_config), sort_keys=True, separators=(",", ":"))
        grouped[(str(row.selected_learner), config)].append(float(row.inner_probe_delta_mrr))

    def key(item: tuple[str, str]) -> tuple:
        learner, config_text = item
        config = json.loads(config_text)
        learner_index = contract["learner_order"].index(learner)
        config_index = hyperparameters[learner].index(config)
        values = grouped[item]
        return len(values), sum(values) / len(values), -learner_index, -config_index

    learner, config_text = max(grouped, key=key)
    votes = len(grouped[(learner, config_text)])
    frozen = pair["final_x4"]
    if learner != frozen["learner"] or json.loads(config_text) != frozen["config"] or votes != int(frozen["outer_fold_votes"]):
        raise RuntimeError(f"Frozen final X4 configuration does not reproduce its DEV-only selection rule: {pair['pair_id']}")
    return learner, json.loads(config_text), votes


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    contract = load_json(contract_path)
    if (
        contract.get("status") != "frozen_before_one_time_test"
        or contract.get("split") != "test"
        or contract.get("pair_count") != 9
        or contract.get("datasets") != ["mkg_w", "db15k", "mkg_y"]
        or contract.get("representation") != "X4"
    ):
        raise RuntimeError("Invalid closure TEST contract header")
    expected_pairs = {
        f"{prefix}_{suffix}"
        for prefix in ("mkgw", "db15k", "mkg_y")
        for suffix in ("mhyper_native", "mhyper_adamf", "native_adamf")
    }
    if {row["pair_id"] for row in contract["pairs"]} != expected_pairs:
        raise RuntimeError("Closure TEST contract must contain exactly the frozen nine pairs")
    if contract["alpha_grid"] != [round(index * 0.05, 2) for index in range(21)]:
        raise RuntimeError("Closure TEST alpha grid changed")

    global_paths = [verify(record) for record in contract["global_dev_sources"]]
    synthesis = load_json(Path("outputs/complementarity_identifiability/cross_dataset_closure_synthesis/synthesis_assessment.json"))
    if synthesis.get("outcome") != "CROSS_DATASET_CLOSURE_SYNTHESIS_COMPLETE":
        raise RuntimeError("Three-dataset DEV closure synthesis is not complete")
    core_contract = load_json(Path("docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json"))
    y_contract = load_json(Path("docs/protocols/MKG_Y_Y_E2_X4_OOF_CONTRACT.json"))

    verified_sources = [contract_path, *global_paths]
    final_configs = []
    for pair in contract["pairs"]:
        utility_path = verify(pair["dev_utility_manifest"])
        selection_path = verify(pair["dev_selection"])
        asset_manifest_path = verify(pair["dev_x4_asset_manifest"])
        nested_path = verify(pair["dev_nested_selection"])
        run_manifest_path = verify(pair["run_manifest"])
        verified_sources.extend((utility_path, selection_path, asset_manifest_path, nested_path, run_manifest_path))
        utility = load_json(utility_path)
        selection = load_json(selection_path)
        asset = load_json(asset_manifest_path)
        if utility.get("split") != "dev" or utility.get("pair_id") != pair["pair_id"]:
            raise RuntimeError(f"Invalid DEV utility manifest: {pair['pair_id']}")
        if Path(utility["source_selection"]["path"]) != selection_path:
            raise RuntimeError(f"Utility/selection path mismatch: {pair['pair_id']}")
        if selection.get("score_normalization") != "query_zscore" or float(selection["global_alpha"]) != float(pair["alpha0"]):
            raise RuntimeError(f"Frozen selection mismatch: {pair['pair_id']}")
        if asset.get("split") != "dev" or asset.get("pair_id") != pair["pair_id"]:
            raise RuntimeError(f"Invalid DEV X4 asset manifest: {pair['pair_id']}")
        if asset.get("feature_fields_x4") != contract["x4_feature_fields"]:
            raise RuntimeError(f"X4 field contract mismatch: {pair['pair_id']}")
        asset_path = Path(asset["output"]["path"])
        if sha256_file(asset_path) != asset["output"]["sha256"]:
            raise RuntimeError(f"DEV X4 asset hash mismatch: {asset_path}")
        verified_sources.append(asset_path)
        hp = y_contract["hyperparameters"] if pair["dataset"] == "mkg_y" else core_contract["hyperparameters"]
        learner, config, votes = selected_configuration(nested_path, pair, contract, hp)
        final_configs.append({"pair_id": pair["pair_id"], "alpha0": pair["alpha0"], "learner": learner, "config": config, "outer_fold_votes": votes})

        for expert_key in (pair["expert_a_key"], pair["expert_b_key"]):
            runs = contract["experts"][expert_key]
            if [int(row["seed"]) for row in runs] != [1, 2, 3]:
                raise RuntimeError(f"Invalid checkpoint seed inventory: {expert_key}")
            for row in runs:
                run_dir = Path(row["run_dir"])
                checkpoint = run_dir / "best.ckpt"
                config_path = run_dir / "config_merged.json"
                if sha256_file(checkpoint) != row["checkpoint_sha256"]:
                    raise RuntimeError(f"Frozen checkpoint hash mismatch: {checkpoint}")
                config_payload = load_json(config_path)
                if int(config_payload.get("system", {}).get("seed", -1)) != int(row["seed"]):
                    raise RuntimeError(f"Frozen checkpoint/config seed mismatch: {run_dir}")
                verified_sources.extend((checkpoint, config_path))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "status": "PASS",
        "contract": {"path": contract_path.as_posix(), "sha256": sha256_file(contract_path)},
        "pair_count": 9,
        "dataset_count": 3,
        "final_x4_configurations": final_configs,
        "verified_source_count": len({path.resolve() for path in verified_sources}),
        "test_access": 0,
        "checkpoint_inference": 0,
        "checkpoint_retraining": 0,
        "checkpoint_reselection": 0,
        "policy_tuning": 0,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
