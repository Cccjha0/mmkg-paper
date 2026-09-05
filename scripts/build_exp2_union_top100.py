from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp2_information_common import load_contract, portable_path, reject_test_path, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct frozen DEV union-top-100 candidate evidence for Experiment 2 X6.")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--utility-manifest-dir", default="outputs/aacpi/utility_tables")
    parser.add_argument("--phase3a-asset-root", default="outputs/aacpi/action_response_assets")
    parser.add_argument("--exp1-manifest", default="outputs/complementarity_identifiability/exp1_landscape/audit_manifest.json")
    parser.add_argument("--contract", default="docs/protocols/EXP2_INFORMATION_FEATURE_CONTRACT.json")
    parser.add_argument("--output-dir", default="outputs/complementarity_identifiability/exp2_information/candidate_assets")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_runs(summary: dict) -> dict[int, tuple[Path, Path]]:
    runs: dict[int, dict[str, Path]] = defaultdict(dict)
    for row in summary.get("endpoint_reproduction", []):
        runs[int(row["seed"])][str(row["expert"])] = Path(row["run_dir"])
    if set(runs) != {1, 2, 3} or any(set(value) != {"A", "B"} for value in runs.values()):
        raise RuntimeError("Incomplete frozen expert provenance")
    return {seed: (value["A"], value["B"]) for seed, value in runs.items()}


def candidate_block(scores_a: np.ndarray, scores_b: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    from scripts.aacpi_phase3a_common import ordinal_ranks, stable_order, stable_top_order

    n = len(scores_a)
    order_a = stable_order(scores_a)
    order_b = stable_order(scores_b)
    rank_a = ordinal_ranks(order_a)
    rank_b = ordinal_ranks(order_b)
    top_a = stable_top_order(scores_a, top_k)
    top_b = stable_top_order(scores_b, top_k)
    members_a, members_b = set(map(int, top_a)), set(map(int, top_b))
    union = np.asarray(sorted(members_a | members_b), dtype=np.int64)
    scale = float(max(n - 1, 1))
    normalized_rank_a = (rank_a[union] - 1) / scale
    normalized_rank_b = (rank_b[union] - 1) / scale
    score_delta = scores_a[union] - scores_b[union]
    rank_delta = normalized_rank_a - normalized_rank_b
    fields = np.column_stack(
        (
            scores_a[union], scores_b[union], normalized_rank_a, normalized_rank_b,
            np.fromiter((candidate in members_a for candidate in union), dtype=np.float32),
            np.fromiter((candidate in members_b for candidate in union), dtype=np.float32),
            score_delta, np.abs(score_delta), rank_delta, np.abs(rank_delta),
        )
    ).astype(np.float32)
    if len(union) > 2 * top_k or not np.isfinite(fields).all():
        raise AssertionError("Invalid union-top-k candidate block")
    return fields, union


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    contract = load_contract(contract_path)
    spec = contract["representations"]["X6_candidate"]
    top_k, max_union = int(spec["top_k_per_expert"]), int(spec["maximum_union_size"])
    if top_k != 100 or max_union != 200 or spec.get("candidate_embeddings") != "excluded_before_first_systematic_run":
        raise RuntimeError("X6 candidate contract is not frozen at top-100 without embeddings")
    utility_manifest_path = Path(args.utility_manifest_dir) / f"{args.pair_id}_dev_source_manifest.json"
    phase3_manifest_path = Path(args.phase3a_asset_root) / args.pair_id / "candidate_score_source_manifest.json"
    exp1_manifest_path = Path(args.exp1_manifest)
    output_dir = Path(args.output_dir)
    for path in (utility_manifest_path, phase3_manifest_path, exp1_manifest_path, output_dir):
        reject_test_path(path)
    exp1 = json.loads(exp1_manifest_path.read_text(encoding="utf-8"))
    if exp1.get("gate", {}).get("decision") != "GO":
        raise RuntimeError("Experiment 1 Available Complementarity Gate did not pass")
    utility_manifest = json.loads(utility_manifest_path.read_text(encoding="utf-8"))
    phase3_manifest = json.loads(phase3_manifest_path.read_text(encoding="utf-8"))
    if utility_manifest.get("split") != "dev" or phase3_manifest.get("split") != "dev":
        raise RuntimeError("X6 builder accepts DEV sources only")
    if int(phase3_manifest.get("test_rows_accessed", -1)) != 0:
        raise RuntimeError("Phase 3A candidate source exposed TEST")
    query_path = Path(utility_manifest["source_query_rows"]["path"])
    summary_path = Path(utility_manifest["source_full_ranking_summary"]["path"])
    if sha256_file(query_path) != utility_manifest["source_query_rows"]["sha256"]:
        raise RuntimeError("Exact DEV query-row hash mismatch")
    if sha256_file(summary_path) != utility_manifest["source_full_ranking_summary"]["sha256"]:
        raise RuntimeError("Full-ranking summary hash mismatch")
    for path_text, expected in phase3_manifest.get("source_files", {}).items():
        path = Path(path_text)
        if sha256_file(path) != expected:
            raise RuntimeError(f"Frozen candidate-reconstruction source hash mismatch: {path}")
    query = pd.read_csv(
        query_path,
        usecols=["query_id", "pair_name", "dataset", "split", "seed", "direction", "head_id", "relation_id", "tail_id"],
    )
    if set(query.split.astype(str)) != {"dev"} or set(query.pair_name.astype(str)) != {args.pair_id}:
        raise RuntimeError("Invalid X6 query inventory")
    if query.query_id.duplicated().any() or sorted(query.seed.astype(int).unique()) != [1, 2, 3]:
        raise RuntimeError("X6 requires unique query IDs and seeds 1/2/3")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = resolve_runs(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.pair_id}_union_top100.npz"
    manifest_path = output_dir / f"{args.pair_id}_union_top100_manifest.json"
    if not args.overwrite and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"Refusing to overwrite X6 asset for {args.pair_id}")
    base_manifest = {
        "schema_version": 1,
        "experiment": "Experiment 2 — Information–Identifiability Audit",
        "representation": "X6",
        "split": "dev",
        "pair_id": args.pair_id,
        "top_k_per_expert": top_k,
        "maximum_union_size": max_union,
        "candidate_fields": spec["candidate_fields"],
        "candidate_embeddings_used": False,
        "candidate_identity_feature_used": False,
        "n_queries": int(len(query)),
        "sources": [
            {"path": portable_path(exp1_manifest_path), "sha256": sha256_file(exp1_manifest_path)},
            {"path": portable_path(utility_manifest_path), "sha256": sha256_file(utility_manifest_path)},
            {"path": portable_path(query_path), "sha256": sha256_file(query_path)},
            {"path": portable_path(summary_path), "sha256": sha256_file(summary_path)},
            {"path": portable_path(phase3_manifest_path), "sha256": sha256_file(phase3_manifest_path)},
            {"path": portable_path(contract_path), "sha256": sha256_file(contract_path)},
        ],
        "test_rows_accessed": 0,
        "checkpoint_training": 0,
        "checkpoint_reselection": 0,
        "full_ranking_evaluation": 0,
    }
    if args.dry_run:
        base_manifest.update({"dry_run": True, "output": None})
        manifest_path.write_text(json.dumps(base_manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[DRY-RUN OK] {args.pair_id}: queries={len(query)} top_k={top_k}")
        return

    import torch
    from router.score_combination import normalize_candidate_scores
    from scripts.build_aacpi_action_response_features import load_frozen_model, resolve_device, score_unfiltered

    device = resolve_device(args.device)
    candidate_features = np.zeros((len(query), max_union, len(spec["candidate_fields"])), dtype=np.float32)
    candidate_mask = np.zeros((len(query), max_union), dtype=bool)
    candidate_ids = np.full((len(query), max_union), -1, dtype=np.int32)
    query_lookup = {value: index for index, value in enumerate(query.query_id.astype(str))}
    for seed in (1, 2, 3):
        run_a, run_b = runs[seed]
        cfg_a, bundle_a, model_a, n_a, _ = load_frozen_model(run_a, device)
        cfg_b, bundle_b, model_b, n_b, _ = load_frozen_model(run_b, device)
        if n_a != n_b or bundle_a.valid_triples != bundle_b.valid_triples:
            raise RuntimeError("Frozen experts are dataset-incompatible")
        for direction in ("head", "tail"):
            current = query[(query.seed.astype(int) == seed) & (query.direction.astype(str) == direction)]
            qbatch = int(args.query_batch_size or min(cfg_a["evaluation"].get("query_batch_size", 8), cfg_b["evaluation"].get("query_batch_size", 8)))
            chunk = int(args.chunk_size or min(cfg_a["evaluation"].get("chunk_size", 4096), cfg_b["evaluation"].get("chunk_size", 4096)))
            for start in range(0, len(current), qbatch):
                block = current.iloc[start : start + qbatch]
                triples_array = block[["head_id", "relation_id", "tail_id"]].to_numpy(np.int64)
                triples_array[:, 2 if direction == "tail" else 0] = 0
                triples = torch.tensor(triples_array, dtype=torch.long)
                raw_a = score_unfiltered(model_a, triples, direction=direction, num_entities=n_a, chunk_size=chunk, device=device)
                raw_b = score_unfiltered(model_b, triples, direction=direction, num_entities=n_b, chunk_size=chunk, device=device)
                z_a = normalize_candidate_scores(raw_a, "query_zscore").numpy().astype(np.float64)
                z_b = normalize_candidate_scores(raw_b, "query_zscore").numpy().astype(np.float64)
                for local, row in enumerate(block.itertuples(index=False)):
                    features, ids = candidate_block(z_a[local], z_b[local], top_k)
                    output_index = query_lookup[str(row.query_id)]
                    candidate_features[output_index, : len(ids)] = features
                    candidate_mask[output_index, : len(ids)] = True
                    candidate_ids[output_index, : len(ids)] = ids
                print(f"[X6] pair={args.pair_id} seed={seed} direction={direction} queries={min(start+qbatch,len(current))}/{len(current)}", flush=True)
        del model_a, model_b
        if device == "cuda":
            torch.cuda.empty_cache()
    if not candidate_mask.any(axis=1).all() or not np.isfinite(candidate_features).all():
        raise RuntimeError("Incomplete X6 reconstruction")
    np.savez_compressed(
        output_path,
        query_id=query.query_id.to_numpy(str),
        candidate_features=candidate_features,
        candidate_mask=candidate_mask,
        candidate_ids_audit_only=candidate_ids,
    )
    base_manifest.update(
        {
            "dry_run": False,
            "output": {"path": portable_path(output_path), "sha256": sha256_file(output_path)},
            "union_size": {
                "min": int(candidate_mask.sum(axis=1).min()),
                "mean": float(candidate_mask.sum(axis=1).mean()),
                "max": int(candidate_mask.sum(axis=1).max()),
            },
            "target_entity_id_used_for_features": False,
            "candidate_ids_stored_for_audit_not_model_input": True,
        }
    )
    manifest_path.write_text(json.dumps(base_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
