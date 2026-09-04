from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aacpi_phase4a_common import portable_path, reject_test_path, sha256_file


MODEL_LABELS = {
    "mmkg_mhyper": "M-Hyper",
    "mmkg_native": "NativE",
    "mmkg_adamf_mat": "AdaMF-MAT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract answer-agnostic query latents from frozen experts.")
    parser.add_argument("--phase3a-feature-table", required=True)
    parser.add_argument("--phase3a-source-manifest", required=True)
    parser.add_argument("--output-latents", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> str:
    import torch
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def resolve_runs(source_manifest: dict) -> dict[tuple[str, int], Path]:
    runs: dict[tuple[str, int], Path] = {}
    source_hashes = source_manifest["source_files"]
    for raw_path, expected_hash in source_hashes.items():
        path = Path(raw_path)
        if path.name != "config_merged.json":
            continue
        reject_test_path(path)
        if not path.exists() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Missing or changed frozen config: {path}")
        cfg = json.loads(path.read_text(encoding="utf-8"))
        label = MODEL_LABELS.get(str(cfg["model"]["name"]))
        if label is None:
            continue
        seed = int(cfg["system"]["seed"])
        checkpoint = path.parent / "best.ckpt"
        checkpoint_key = checkpoint.as_posix()
        if checkpoint_key not in source_hashes or sha256_file(checkpoint) != source_hashes[checkpoint_key]:
            raise RuntimeError(f"Frozen checkpoint hash mismatch: {checkpoint}")
        key = (label, seed)
        if key in runs:
            raise RuntimeError(f"Duplicate frozen run: {key}")
        runs[key] = path.parent
    return runs


def extract_batch(model, rows, device: str) -> torch.Tensor:
    import torch
    direction = str(rows.direction.iloc[0])
    if set(rows.direction.astype(str)) != {direction}:
        raise ValueError("Latent batch must contain one direction")
    relation = torch.as_tensor(rows.relation.to_numpy(np.int64), device=device)
    known = torch.as_tensor(
        rows.head.to_numpy(np.int64) if direction == "tail" else rows.tail.to_numpy(np.int64),
        device=device,
    )
    if model.__class__.__name__ == "OpenBGMHyper":
        if direction == "head":
            relation = model.inverse_relation_ids[relation]
        query_triples = torch.stack((known, relation, torch.zeros_like(known)), dim=1)
        cache = model._get_eval_cache()
        query, _ = model._queries(query_triples, cache["structure"], cache["image"], cache["text"])
        return query.detach().cpu().float()
    structural, visual, text = model.get_batch_ent_multimodal_embs(known)
    if model.__class__.__name__ == "OpenBGNativE":
        fused = model.get_joint_embeddings(structural, visual, text, model.rel_gate(relation))
    elif model.__class__.__name__ == "OpenBGAdaMFMAT":
        fused = model.get_joint_embeddings(structural, visual, text)
    else:
        raise TypeError(f"Unapproved Phase 4A expert class: {model.__class__.__name__}")
    return torch.cat((fused, model.rel_embeddings(relation)), dim=1).detach().cpu().float()


def main() -> None:
    args = parse_args()
    import pandas as pd
    import torch
    from scripts.build_aacpi_action_response_features import load_frozen_model

    table_path, source_path = Path(args.phase3a_feature_table), Path(args.phase3a_source_manifest)
    output_path, manifest_path = Path(args.output_latents), Path(args.output_manifest)
    for path in (table_path, source_path, output_path, manifest_path):
        reject_test_path(path)
    if any(path.exists() for path in (output_path, manifest_path)) and not args.overwrite:
        raise FileExistsError("Refusing to overwrite Phase 4A latent outputs")
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    if source_manifest.get("split") != "dev" or int(source_manifest.get("test_rows_accessed", -1)) != 0:
        raise RuntimeError("Latent source is not DEV-only")
    frame = pd.read_csv(table_path, compression="infer", usecols=[
        "dataset", "pair_id", "expert_a", "expert_b", "split", "query_id", "original_triple_id",
        "seed", "direction", "head", "relation", "tail",
    ]).drop_duplicates("query_id").reset_index(drop=True)
    if frame.empty or set(frame.split.astype(str)) != {"dev"} or frame.query_id.duplicated().any():
        raise RuntimeError("Invalid DEV query inventory")
    experts = (str(frame.expert_a.iloc[0]), str(frame.expert_b.iloc[0]))
    runs, device = resolve_runs(source_manifest), resolve_device(args.device)
    expected = {(expert, int(seed)) for expert in experts for seed in sorted(frame.seed.unique())}
    if not expected <= set(runs):
        raise RuntimeError(f"Missing frozen runs: {sorted(expected-set(runs))}")
    outputs: dict[str, np.ndarray] = {
        "query_id": frame.query_id.astype(str).to_numpy(),
        "original_triple_id": frame.original_triple_id.astype(str).to_numpy(),
        "seed": frame.seed.to_numpy(np.int64),
        "direction": frame.direction.astype(str).to_numpy(),
    }
    dimensions, checkpoint_sources = {}, []
    for position, expert in zip(("a", "b"), experts):
        latent = None
        for seed in sorted(frame.seed.unique()):
            run_dir = runs[(expert, int(seed))]
            cfg, bundle, model, _, sources = load_frozen_model(run_dir, device)
            indices = np.flatnonzero(frame.seed.to_numpy() == seed)
            parts = []
            for direction in ("head", "tail"):
                direction_indices = indices[frame.iloc[indices].direction.to_numpy() == direction]
                for start in range(0, len(direction_indices), args.batch_size):
                    batch_indices = direction_indices[start : start + args.batch_size]
                    with torch.inference_mode():
                        part = extract_batch(model, frame.iloc[batch_indices], device).numpy()
                    parts.append((batch_indices, part))
            seed_dim = {part.shape[1] for _, part in parts}
            if len(seed_dim) != 1:
                raise RuntimeError("Inconsistent latent dimensions within seed")
            dim = seed_dim.pop()
            if latent is None:
                latent = np.empty((len(frame), dim), dtype=np.float32)
            if latent.shape[1] != dim:
                raise RuntimeError("Latent dimension changes across seeds")
            for batch_indices, part in parts:
                latent[batch_indices] = part
            checkpoint = run_dir / "best.ckpt"; config = run_dir / "config_merged.json"
            checkpoint_sources.append({
                "expert": expert, "seed": int(seed),
                "checkpoint": {"path": portable_path(checkpoint), "sha256": sha256_file(checkpoint)},
                "config": {"path": portable_path(config), "sha256": sha256_file(config)},
                "model_class": model.__class__.__name__,
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if latent is None or not np.isfinite(latent).all():
            raise AssertionError(f"Invalid latent for {expert}")
        outputs[f"z_{position}"] = latent
        dimensions[expert] = int(latent.shape[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **outputs)
    payload = {
        "schema_version": 1, "phase": "AACPI Phase 4A", "split": "dev",
        "dataset": str(frame.dataset.iloc[0]), "pair_id": str(frame.pair_id.iloc[0]),
        "expert_a": experts[0], "expert_b": experts[1], "n_queries": len(frame),
        "raw_dimensions": dimensions,
        "semantics": {
            "M-Hyper": "pre-candidate dot-product query; reciprocal transform for head prediction",
            "NativE": "known-side relation-gated fused entity concatenated with relation embedding",
            "AdaMF-MAT": "known-side fused entity concatenated with relation embedding",
        },
        "candidate_independent": True, "current_target_identity_used": False,
        "source_table": {"path": portable_path(table_path), "sha256": sha256_file(table_path)},
        "source_manifest": {"path": portable_path(source_path), "sha256": sha256_file(source_path)},
        "frozen_runs": checkpoint_sources,
        "output": {"path": portable_path(output_path), "sha256": sha256_file(output_path)},
        "expert_training_performed": False, "checkpoint_selection_performed": False,
        "test_rows_accessed": 0, "test_evaluation_commands": 0, "policy_evaluations": 0,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {payload['pair_id']} frozen query latents dims={dimensions} -> {output_path}")


if __name__ == "__main__":
    main()
