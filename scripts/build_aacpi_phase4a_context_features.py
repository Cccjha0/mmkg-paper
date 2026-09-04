from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training.src.data.tsv_reader import read_integer_triples
from scripts.aacpi_phase4a_common import (
    C1_ADDITIONS,
    C2_ADDITIONS,
    feature_contract,
    portable_path,
    reject_test_path,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TRAIN-only Phase 4A structural/modality DEV context.")
    parser.add_argument("--phase3a-feature-table", required=True)
    parser.add_argument("--phase3a-source-manifest", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_path(manifest: dict, filename: str) -> Path:
    matches = [Path(path) for path in manifest["source_files"] if Path(path).name == filename]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename} in source manifest, found {matches}")
    path = matches[0]
    reject_test_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    expected = manifest["source_files"][path.as_posix()]
    if sha256_file(path) != expected:
        raise RuntimeError(f"Source hash mismatch: {path}")
    return path


def train_statistics(triples: list[tuple[int, int, int]], num_entities: int, has_img, has_text) -> dict:
    entity_degree, entity_head, entity_tail = Counter(), Counter(), Counter()
    neighbors: dict[int, set[int]] = defaultdict(set)
    relation_frequency = Counter()
    relation_heads: dict[int, set[int]] = defaultdict(set)
    relation_tails: dict[int, set[int]] = defaultdict(set)
    relation_known_image: dict[tuple[int, str], list[float]] = defaultdict(list)
    relation_known_text: dict[tuple[int, str], list[float]] = defaultdict(list)
    for head, relation, tail in triples:
        entity_degree[head] += 1; entity_degree[tail] += 1
        entity_head[head] += 1; entity_tail[tail] += 1
        neighbors[head].add(tail); neighbors[tail].add(head)
        relation_frequency[relation] += 1
        relation_heads[relation].add(head); relation_tails[relation].add(tail)
        relation_known_image[(relation, "tail")].append(float(has_img[head]))
        relation_known_text[(relation, "tail")].append(float(has_text[head]))
        relation_known_image[(relation, "head")].append(float(has_img[tail]))
        relation_known_text[(relation, "head")].append(float(has_text[tail]))
    return {
        "entity_degree": entity_degree,
        "entity_head": entity_head,
        "entity_tail": entity_tail,
        "neighbors": neighbors,
        "relation_frequency": relation_frequency,
        "relation_heads": relation_heads,
        "relation_tails": relation_tails,
        "relation_known_image": {k: float(np.mean(v)) for k, v in relation_known_image.items()},
        "relation_known_text": {k: float(np.mean(v)) for k, v in relation_known_text.items()},
        "num_entities": num_entities,
    }


def context_row(row, stats: dict) -> dict[str, float]:
    direction, relation = str(row.direction), int(row.relation)
    known = int(row.head) if direction == "tail" else int(row.tail)
    relation_count = stats["relation_frequency"][relation]
    unique_heads = len(stats["relation_heads"][relation])
    unique_tails = len(stats["relation_tails"][relation])
    role_diversity = unique_heads if direction == "tail" else unique_tails
    relation_image = stats["relation_known_image"].get((relation, direction), 0.0)
    relation_text = stats["relation_known_text"].get((relation, direction), 0.0)
    return {
        "c1_known_entity_neighborhood_fraction": len(stats["neighbors"][known]) / max(stats["num_entities"] - 1, 1),
        "c1_relation_unique_heads_log1p": math.log1p(unique_heads),
        "c1_relation_unique_tails_log1p": math.log1p(unique_tails),
        "c1_relation_tails_per_head_log1p": math.log1p(relation_count / max(unique_heads, 1)),
        "c1_relation_heads_per_tail_log1p": math.log1p(relation_count / max(unique_tails, 1)),
        "c1_relation_head_tail_log_ratio": math.log((unique_heads + 1.0) / (unique_tails + 1.0)),
        "c1_relation_known_role_diversity_log1p": math.log1p(role_diversity),
        "c2_relation_known_image_support": relation_image,
        "c2_relation_known_text_support": relation_text,
    }


def main() -> None:
    args = parse_args()
    import pandas as pd
    import torch

    feature_path, source_manifest_path = Path(args.phase3a_feature_table), Path(args.phase3a_source_manifest)
    output_path, manifest_path = Path(args.output_table), Path(args.output_manifest)
    for path in (feature_path, source_manifest_path, output_path, manifest_path):
        reject_test_path(path)
    if any(path.exists() for path in (output_path, manifest_path)) and not args.overwrite:
        raise FileExistsError("Refusing to overwrite Phase 4A context outputs")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("split") != "dev" or int(source_manifest.get("test_rows_accessed", -1)) != 0:
        raise RuntimeError("Phase 3A source manifest is not DEV-only")
    train_path = source_path(source_manifest, "train.tsv")
    has_img_path = source_path(source_manifest, "has_img.pt")
    has_text_path = source_path(source_manifest, "has_text.pt")
    entity_map_path = source_path(source_manifest, "entity2id.json")
    entity_map = json.loads(entity_map_path.read_text(encoding="utf-8"))
    has_img = torch.load(has_img_path, map_location="cpu").bool().numpy()
    has_text = torch.load(has_text_path, map_location="cpu").bool().numpy()
    if len(has_img) != len(entity_map) or len(has_text) != len(entity_map):
        raise RuntimeError("Modality masks do not match entity mapping")
    stats = train_statistics(read_integer_triples(train_path), len(entity_map), has_img, has_text)
    frame = pd.read_csv(feature_path, compression="infer")
    if frame.empty or set(frame["split"].astype(str)) != {"dev"}:
        raise RuntimeError("Context builder accepts only nonempty DEV assets")
    before = frame[["query_id", "alpha", "advantage", "alpha0", "original_triple_id"]].copy()
    unique = frame.drop_duplicates("query_id")
    if unique["query_id"].duplicated().any():
        raise AssertionError("query_id uniqueness check failed")
    values = pd.DataFrame([context_row(row, stats) for row in unique.itertuples(index=False)])
    values.insert(0, "query_id", unique["query_id"].to_numpy())
    frame = frame.merge(values, on="query_id", how="left", validate="many_to_one")
    if frame[C1_ADDITIONS + C2_ADDITIONS].isna().any().any() or not np.isfinite(frame[C1_ADDITIONS + C2_ADDITIONS].to_numpy(float)).all():
        raise AssertionError("Missing/nonfinite Phase 4A context")
    after = frame[["query_id", "alpha", "advantage", "alpha0", "original_triple_id"]]
    if not before.equals(after):
        raise AssertionError("Utility identity/order changed while adding context")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, compression="gzip" if output_path.suffix == ".gz" else None)
    payload = {
        "schema_version": 1, "phase": "AACPI Phase 4A", "split": "dev",
        "dataset": str(frame.dataset.iloc[0]), "pair_id": str(frame.pair_id.iloc[0]),
        "source_phase3a_table": {"path": portable_path(feature_path), "sha256": sha256_file(feature_path)},
        "source_phase3a_manifest": {"path": portable_path(source_manifest_path), "sha256": sha256_file(source_manifest_path)},
        "train_source": {"path": portable_path(train_path), "sha256": sha256_file(train_path)},
        "modality_sources": [{"path": portable_path(p), "sha256": sha256_file(p)} for p in (has_img_path, has_text_path)],
        "feature_contract": feature_contract(), "n_rows": len(frame),
        "n_queries": int(frame.query_id.nunique()), "n_original_triples": int(frame.original_triple_id.nunique()),
        "output": {"path": portable_path(output_path), "sha256": sha256_file(output_path)},
        "structural_statistics_source": "train_only", "current_target_properties_used": False,
        "test_rows_accessed": 0, "test_evaluation_commands": 0, "policy_evaluations": 0,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {payload['pair_id']} Phase 4A context rows={len(frame)} -> {output_path}")


if __name__ == "__main__":
    main()
