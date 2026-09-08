from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp2_information_common import (
    ALPHAS,
    RR_COLUMNS,
    action_descriptors,
    portable_path,
    select_probe_actions,
    sha256_file,
)
from scripts.run_exp2_information_nested_oof import expand_tabular, train_predict_tabular


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the DEV-frozen final X4 probe once and apply it to closure TEST.")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def verify(record: dict) -> Path:
    path = Path(record["path"])
    if "sha256_lf" in record:
        normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        actual = hashlib.sha256(normalized).hexdigest()
        expected = record["sha256_lf"]
    else:
        actual = sha256_file(path)
        expected = record["sha256"]
    if actual != expected:
        raise RuntimeError(f"Frozen source hash mismatch: {path}")
    return path


def align(source: np.ndarray, target: np.ndarray, label: str) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(source)}
    if len(lookup) != len(source) or set(lookup) != set(map(str, target)):
        raise RuntimeError(f"{label} query inventory mismatch")
    return np.asarray([lookup[str(value)] for value in target], dtype=np.int64)


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    contract = load_json(contract_path)
    pair = next((row for row in contract["pairs"] if row["pair_id"] == args.pair_id), None)
    if pair is None:
        raise ValueError(f"Pair is absent from frozen TEST contract: {args.pair_id}")
    if contract.get("status") != "frozen_before_one_time_test" or contract.get("representation") != "X4":
        raise RuntimeError("Closure TEST contract is not frozen for X4")

    output_dir = Path(args.output_root) / args.pair_id
    output_path = output_dir / "test_x4_policy_rows.csv.gz"
    summary_path = output_dir / "test_x4_summary.json"
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(f"Repeated frozen X4 TEST application is prohibited: {output_dir}")

    utility_manifest_path = verify(pair["dev_utility_manifest"])
    feature_manifest_path = verify(pair["dev_x4_asset_manifest"])
    nested_selection_path = verify(pair["dev_nested_selection"])
    selection_path = verify(pair["dev_selection"])
    utility_manifest = load_json(utility_manifest_path)
    feature_manifest = load_json(feature_manifest_path)
    selection = load_json(selection_path)
    if utility_manifest.get("split") != "dev" or feature_manifest.get("split") != "dev":
        raise RuntimeError("Final X4 fit sources must be DEV-only")
    if utility_manifest.get("pair_id") != args.pair_id or feature_manifest.get("pair_id") != args.pair_id:
        raise RuntimeError("Final X4 source pair identity mismatch")
    if float(selection["global_alpha"]) != float(pair["alpha0"]):
        raise RuntimeError("Frozen full-DEV alpha0 mismatch")

    feature_path = Path(feature_manifest["output"]["path"])
    if sha256_file(feature_path) != feature_manifest["output"]["sha256"]:
        raise RuntimeError(f"DEV X4 feature asset hash mismatch: {feature_path}")
    query_path = Path(utility_manifest["source_query_rows"]["path"])
    if sha256_file(query_path) != utility_manifest["source_query_rows"]["sha256"]:
        raise RuntimeError(f"DEV exact RR query hash mismatch: {query_path}")

    with np.load(feature_path, allow_pickle=False) as payload:
        dev_ids = payload["query_id"].astype(str)
        dev_x4 = payload["features_x4"].astype(np.float32)
    fields = list(feature_manifest["feature_fields_x4"])
    if fields != contract["x4_feature_fields"] or dev_x4.shape != (len(dev_ids), len(fields)):
        raise RuntimeError("Frozen X4 feature schema mismatch")
    exact = pd.read_csv(query_path, usecols=["query_id", *RR_COLUMNS])
    order = align(exact.query_id.to_numpy(str), dev_ids, "DEV exact RR")
    dev_rr = exact.iloc[order][RR_COLUMNS].to_numpy(np.float64)

    alpha0 = float(pair["alpha0"])
    global_candidates = np.flatnonzero(np.isclose(ALPHAS, alpha0, atol=0, rtol=0))
    if len(global_candidates) != 1:
        raise RuntimeError("Frozen alpha0 is absent from the action grid")
    global_index = int(global_candidates[0])
    targets = dev_rr - dev_rr[:, [global_index]]
    descriptors = action_descriptors(global_index)
    query_indices = np.arange(len(dev_ids), dtype=np.int64)
    train_x = expand_tabular(dev_x4, query_indices, descriptors)
    train_y = targets.reshape(-1).astype(np.float64)

    test_path = Path(args.raw_root) / args.pair_id / "test_query_rows.csv"
    test = pd.read_csv(test_path)
    required = {"query_id", "split", "seed", "direction", "head_id", "relation_id", "tail_id", *fields, *RR_COLUMNS}
    missing = sorted(required - set(test.columns))
    if missing:
        raise RuntimeError(f"TEST X4 export is incomplete for {args.pair_id}: {missing}")
    if set(test.split.astype(str)) != {"test"} or test.query_id.duplicated().any():
        raise RuntimeError("Invalid TEST query inventory")
    test_x4 = test[fields].to_numpy(np.float32)
    if not np.isfinite(test_x4).all():
        raise RuntimeError("Non-finite TEST X4 features")
    eval_x = expand_tabular(test_x4, np.arange(len(test), dtype=np.int64), descriptors)

    learner = pair["final_x4"]["learner"]
    config = pair["final_x4"]["config"]
    predicted = train_predict_tabular(
        learner,
        config,
        train_x,
        train_y,
        eval_x,
        contract,
        args.device,
        int(contract["training"]["model_seed"]),
    ).reshape(len(test), len(ALPHAS))
    chosen = select_probe_actions(predicted, global_index)
    rr = test[RR_COLUMNS].to_numpy(np.float64)
    row_index = np.arange(len(test))
    selected_rr = rr[row_index, chosen]
    global_rr = rr[:, global_index]
    gain = selected_rr - global_rr

    output = test[["query_id", "seed", "direction", "head_id", "relation_id", "tail_id"]].copy()
    output.insert(0, "pair_id", args.pair_id)
    output.insert(0, "dataset", pair["dataset"])
    output["alpha0"] = alpha0
    output["selected_alpha"] = ALPHAS[chosen]
    output["predicted_advantage"] = predicted[row_index, chosen]
    output["global_rr"] = global_rr
    output["selected_rr"] = selected_rr
    output["gain"] = gain
    output_dir.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    output.to_csv(output_path, index=False, compression=compression, lineterminator="\n")

    summary = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "split": "test",
        "dataset": pair["dataset"],
        "pair_id": args.pair_id,
        "representation": "X4",
        "alpha0": alpha0,
        "learner": learner,
        "config": config,
        "configuration_freeze_rule": contract["final_x4_configuration_rule"],
        "n_dev_queries": int(len(dev_ids)),
        "n_test_queries": int(len(test)),
        "test_mrr": float(selected_rr.mean()),
        "test_global_mrr": float(global_rr.mean()),
        "test_gain": float(gain.mean()),
        "changed_rate": float((chosen != global_index).mean()),
        "negative_transfer_rate": float((gain < 0).mean()),
        "positive_transfer_rate": float((gain > 0).mean()),
        "sources": [
            {"path": portable_path(path), "sha256": sha256_file(path)}
            for path in (
                contract_path,
                utility_manifest_path,
                feature_manifest_path,
                nested_selection_path,
                selection_path,
                feature_path,
                query_path,
                test_path,
            )
        ],
        "output": {"path": portable_path(output_path), "sha256": sha256_file(output_path)},
        "test_labels_used_for_training_or_action_selection": 0,
        "test_hyperparameter_selection": 0,
        "checkpoint_retraining": 0,
        "expert_checkpoint_reselection": 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {args.pair_id}: frozen X4 TEST gain={summary['test_gain']:+.6f}")


if __name__ == "__main__":
    main()
