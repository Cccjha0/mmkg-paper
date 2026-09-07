from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp2_information_common import (
    load_contract,
    portable_path,
    reject_test_path,
    representation_features,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audited MKG-Y Y-E2 frozen X4 query assets.")
    parser.add_argument("--contract", default="docs/protocols/MKG_Y_Y_E2_X4_OOF_CONTRACT.json")
    parser.add_argument("--phase4a-root", default="outputs/complementarity_identifiability/mkg_y_y_e2_information/context")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/mkg_y_y_e2_information/assets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def verify(path: Path, expected: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Hash mismatch: {path}; expected={expected}, actual={actual}")


def query_source(manifest: dict) -> tuple[Path, str]:
    matches = [(Path(path), digest) for path, digest in manifest.get("files", {}).items() if Path(path).name == "dev_query_rows.csv"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact DEV query-row source, found {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    phase4a_root = Path(args.phase4a_root)
    output_dir = Path(args.output_dir)
    for path in (contract_path, phase4a_root, output_dir):
        reject_test_path(path)
    contract = load_contract(contract_path)
    if contract.get("protocol_profile") != "mkg_y_frozen_x4_replication":
        raise ValueError("This builder accepts only the frozen MKG-Y X4 replication contract")
    prerequisite = contract["prerequisite"]
    prerequisite_path = Path(prerequisite["manifest"])
    verify(prerequisite_path, prerequisite["sha256"])
    prerequisite_payload = json.loads(prerequisite_path.read_text(encoding="utf-8"))
    if prerequisite_payload.get("assessment", {}).get("outcome") != prerequisite["required_assessment"]:
        raise RuntimeError("MKG-Y Y-E1 prerequisite assessment mismatch")

    all_fields = representation_features(contract)["X4"]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pair_id in contract["pair_ids"]:
        boundary = contract["source_boundaries"]["full_ranking_manifests"][pair_id]
        ranking_manifest_path = Path(boundary["path"])
        verify(ranking_manifest_path, boundary["sha256"])
        ranking_manifest = json.loads(ranking_manifest_path.read_text(encoding="utf-8"))
        pair = ranking_manifest.get("pair", {})
        if pair.get("pair_id") != pair_id or pair.get("status") != "PASS" or int(pair.get("test_rows_opened", -1)) != 0:
            raise RuntimeError(f"Invalid frozen Y2 source manifest for {pair_id}")
        query_path, query_hash = query_source(ranking_manifest)
        verify(query_path, query_hash)

        context_manifest_path = phase4a_root / pair_id / "context_feature_manifest.json"
        context_manifest = json.loads(context_manifest_path.read_text(encoding="utf-8"))
        if (
            context_manifest.get("split") != "dev"
            or context_manifest.get("dataset") != "mkg_y"
            or context_manifest.get("pair_id") != pair_id
            or int(context_manifest.get("test_rows_accessed", -1)) != 0
            or context_manifest.get("structural_statistics_source") != "train_only"
        ):
            raise RuntimeError(f"Invalid TRAIN-only Phase 4A context manifest for {pair_id}")
        context_path = Path(context_manifest["output"]["path"])
        verify(context_path, context_manifest["output"]["sha256"])

        output_path = output_dir / f"{pair_id}_query_information.npz"
        manifest_path = output_dir / f"{pair_id}_query_information_manifest.json"
        if not args.overwrite and (output_path.exists() or manifest_path.exists()):
            raise FileExistsError(f"Refusing to overwrite MKG-Y Y-E2 asset for {pair_id}")
        context = pd.read_csv(
            context_path,
            usecols=["query_id", "original_triple_id", "seed", "direction", "head", "relation", "tail", *all_fields],
        )
        if set(context.direction.astype(str)) != {"head", "tail"} or sorted(context.seed.astype(int).unique()) != [1, 2, 3]:
            raise RuntimeError(f"Invalid seed/direction inventory for {pair_id}")
        varying = [field for field in all_fields if context.groupby("query_id", sort=False)[field].nunique(dropna=False).max() != 1]
        if varying:
            raise RuntimeError(f"Query-static feature contract violated for {pair_id}: {varying}")
        query_context = context.drop_duplicates("query_id", keep="first")[["query_id", *all_fields]]
        source = pd.read_csv(query_path, usecols=["query_id", "seed", "direction", "head_id", "relation_id", "tail_id"])
        if source.query_id.duplicated().any() or set(source.query_id.astype(str)) != set(query_context.query_id.astype(str)):
            raise RuntimeError(f"Y2 exact rows / Phase 4A query mismatch for {pair_id}")
        query = source.merge(query_context, on="query_id", how="left", validate="one_to_one")
        matrix = query[all_fields].to_numpy(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite X4 features for {pair_id}")
        payload = {
            "query_id": query.query_id.to_numpy(str),
            "seed": query.seed.to_numpy(np.int16),
            "direction": query.direction.to_numpy(str),
            "head_id": query.head_id.to_numpy(np.int64),
            "relation_id": query.relation_id.to_numpy(np.int64),
            "tail_id": query.tail_id.to_numpy(np.int64),
            "features_x4": matrix,
        }
        if not args.dry_run:
            np.savez_compressed(output_path, **payload)
        manifest = {
            "schema_version": 1,
            "experiment": contract["experiment"],
            "split": "dev",
            "dataset": "mkg_y",
            "pair_id": pair_id,
            "dry_run": bool(args.dry_run),
            "n_queries": int(len(query)),
            "n_original_triples": int(source[["head_id", "relation_id", "tail_id"]].drop_duplicates().shape[0]),
            "feature_fields_x4": all_fields,
            "representation_dimensions": {"X4": len(all_fields)},
            "sources": [
                {"role": "y_e1_audit", "path": portable_path(prerequisite_path), "sha256": sha256_file(prerequisite_path)},
                {"role": "y2_full_ranking_manifest", "path": portable_path(ranking_manifest_path), "sha256": sha256_file(ranking_manifest_path)},
                {"role": "exact_dev_query_rows", "path": portable_path(query_path), "sha256": sha256_file(query_path)},
                {"role": "phase4a_context", "path": portable_path(context_path), "sha256": sha256_file(context_path)},
                {"role": "phase4a_context_manifest", "path": portable_path(context_manifest_path), "sha256": sha256_file(context_manifest_path)},
                {"role": "feature_contract", "path": portable_path(contract_path), "sha256": sha256_file(contract_path)},
            ],
            "output": None if args.dry_run else {"path": portable_path(output_path), "sha256": sha256_file(output_path)},
            "test_rows_accessed": 0,
            "checkpoint_training": 0,
            "checkpoint_reselection": 0,
            "policy_development": 0,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        status = "dry_run_ok" if args.dry_run else "built"
        records.append({"pair_id": pair_id, "n_queries": int(len(query)), "status": status})
        print(f"[{status.upper()}] {pair_id}: queries={len(query)}")
    (output_dir / "asset_build_summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
