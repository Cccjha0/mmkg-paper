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
    PAIR_IDS,
    load_contract,
    portable_path,
    reject_test_path,
    representation_features,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audited query-static X1-X4 assets for Experiment 2.")
    parser.add_argument("--exp1-manifest", default="outputs/complementarity_identifiability/exp1_landscape/audit_manifest.json")
    parser.add_argument("--phase4a-root", default="outputs/aacpi/phase4a")
    parser.add_argument("--contract", default="docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/exp2_information/assets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def verify(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Hash mismatch: {path}; expected={expected}, actual={actual}")


def main() -> None:
    args = parse_args()
    exp1_path, phase4_root, contract_path, output_dir = map(
        Path, (args.exp1_manifest, args.phase4a_root, args.contract, args.output_dir)
    )
    for path in (exp1_path, phase4_root, contract_path, output_dir):
        reject_test_path(path)
    contract = load_contract(contract_path)
    features = representation_features(contract)
    all_fields = features["X4"]
    exp1 = json.loads(exp1_path.read_text(encoding="utf-8"))
    if exp1.get("gate", {}).get("decision") != "GO":
        raise RuntimeError("Experiment 1 Available Complementarity Gate did not pass")
    source_by_pair = {}
    for source in exp1["sources"]:
        if source["role"] == "source_query_rows":
            source_by_pair[source["pair_id"]] = source
    if set(source_by_pair) != set(PAIR_IDS):
        raise RuntimeError("Experiment 1 source inventory does not contain exactly six pairs")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pair_id in PAIR_IDS:
        context_manifest_path = phase4_root / pair_id / "context_feature_manifest.json"
        latent_manifest_path = phase4_root / pair_id / "latent_extraction_manifest.json"
        for path in (context_manifest_path, latent_manifest_path):
            reject_test_path(path)
        context_manifest = json.loads(context_manifest_path.read_text(encoding="utf-8"))
        latent_manifest = json.loads(latent_manifest_path.read_text(encoding="utf-8"))
        if context_manifest.get("split") != "dev" or latent_manifest.get("split") != "dev":
            raise RuntimeError(f"Non-DEV Phase 4A asset for {pair_id}")
        if int(context_manifest.get("test_rows_accessed", -1)) != 0 or int(latent_manifest.get("test_rows_accessed", -1)) != 0:
            raise RuntimeError(f"Phase 4A TEST exposure for {pair_id}")
        context_path = Path(context_manifest["output"]["path"])
        latent_path = Path(latent_manifest["output"]["path"])
        query_path = Path(source_by_pair[pair_id]["path"])
        verify(query_path, source_by_pair[pair_id]["sha256"])
        verify(context_path, context_manifest["output"]["sha256"])
        verify(latent_path, latent_manifest["output"]["sha256"])
        output_path = output_dir / f"{pair_id}_query_information.npz"
        manifest_path = output_dir / f"{pair_id}_query_information_manifest.json"
        if not args.overwrite and (output_path.exists() or manifest_path.exists()):
            raise FileExistsError(f"Refusing to overwrite Experiment 2 asset for {pair_id}")
        context = pd.read_csv(
            context_path,
            usecols=["query_id", "original_triple_id", "seed", "direction", "head", "relation", "tail", *all_fields],
        )
        if set(context.direction.astype(str)) != {"head", "tail"} or sorted(context.seed.astype(int).unique()) != [1, 2, 3]:
            raise RuntimeError(f"Invalid seed/direction inventory for {pair_id}")
        grouped = context.groupby("query_id", sort=False)
        varying = [field for field in all_fields if grouped[field].nunique(dropna=False).max() != 1]
        if varying:
            raise RuntimeError(f"Query-static feature contract violated for {pair_id}: {varying}")
        query = context.drop_duplicates("query_id", keep="first").reset_index(drop=True)
        source = pd.read_csv(query_path, usecols=["query_id", "seed", "direction", "head_id", "relation_id", "tail_id"])
        if len(source) != len(query) or set(source.query_id.astype(str)) != set(query.query_id.astype(str)):
            raise RuntimeError(f"Experiment 1 / Phase 4A query mismatch for {pair_id}")
        query = source.merge(query[["query_id", *all_fields]], on="query_id", how="left", validate="one_to_one")
        matrix = query[all_fields].to_numpy(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite information features for {pair_id}")
        with np.load(latent_path, allow_pickle=False) as latent:
            latent_ids = latent["query_id"].astype(str)
            z_a_shape, z_b_shape = latent["z_a"].shape, latent["z_b"].shape
        if set(latent_ids) != set(query.query_id.astype(str)):
            raise RuntimeError(f"Latent query mismatch for {pair_id}")
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
            "experiment": "Experiment 2 — Information–Identifiability Audit",
            "split": "dev",
            "pair_id": pair_id,
            "dry_run": bool(args.dry_run),
            "n_queries": int(len(query)),
            "feature_fields_x4": all_fields,
            "representation_dimensions": {key: len(value) for key, value in features.items()},
            "raw_latent_shapes": {"z_a": z_a_shape, "z_b": z_b_shape},
            "sources": [
                {"role": "exp1_audit", "path": portable_path(exp1_path), "sha256": sha256_file(exp1_path)},
                {"role": "exact_dev_query_rows", "path": portable_path(query_path), "sha256": sha256_file(query_path)},
                {"role": "phase4a_context", "path": portable_path(context_path), "sha256": sha256_file(context_path)},
                {"role": "phase4a_context_manifest", "path": portable_path(context_manifest_path), "sha256": sha256_file(context_manifest_path)},
                {"role": "frozen_latents", "path": portable_path(latent_path), "sha256": sha256_file(latent_path)},
                {"role": "latent_manifest", "path": portable_path(latent_manifest_path), "sha256": sha256_file(latent_manifest_path)},
                {"role": "feature_contract", "path": portable_path(contract_path), "sha256": sha256_file(contract_path)},
            ],
            "output": None if args.dry_run else {"path": portable_path(output_path), "sha256": sha256_file(output_path)},
            "test_rows_accessed": 0,
            "checkpoint_training": 0,
            "checkpoint_reselection": 0,
            "policy_development": 0,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        records.append({"pair_id": pair_id, "n_queries": len(query), "status": "dry_run_ok" if args.dry_run else "built"})
        print(f"[{records[-1]['status'].upper()}] {pair_id}: queries={len(query)}")
    (output_dir / "asset_build_summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
