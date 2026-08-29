from __future__ import annotations

import json
import hashlib
from pathlib import Path

from ml.training.src.data.dataset_spec import DatasetBundle, MMKG_GENERAL_V1
from ml.training.src.data.feature_bundle import load_processed_feature_bundle
from ml.training.src.data.tsv_reader import read_integer_triples


def _load_mapping(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return {str(key): int(value) for key, value in payload.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_openke_mmkg(cfg: dict) -> DatasetBundle:
    dataset_cfg = cfg["dataset"]
    directory = Path(dataset_cfg["processed_dir"])
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run the external-dataset preprocessing command first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = cfg.get("protocol", {}).get("version", MMKG_GENERAL_V1)
    if protocol != MMKG_GENERAL_V1:
        raise ValueError("openke_mmkg datasets require protocol.version=mmkg_general_v1")
    if manifest.get("protocol") != protocol:
        raise ValueError(
            f"Manifest protocol {manifest.get('protocol')!r} does not match config protocol {protocol!r}."
        )
    if manifest.get("dataset") != dataset_cfg["name"]:
        raise ValueError(
            f"Manifest dataset {manifest.get('dataset')!r} does not match config dataset {dataset_cfg['name']!r}."
        )
    split_paths = {name: directory / f"{name}.tsv" for name in ("train", "valid", "test")}
    expected_hashes = manifest.get("hashes", {})
    expected_split_hashes = expected_hashes.get("splits", {})
    if expected_split_hashes:
        actual_split_hashes = {name: _sha256_file(path) for name, path in split_paths.items()}
        mismatched = sorted(
            name for name, actual in actual_split_hashes.items() if actual != expected_split_hashes.get(name)
        )
        if mismatched:
            raise ValueError(f"Canonical split hashes do not match manifest: {mismatched}")
    train_triples = read_integer_triples(split_paths["train"])
    valid_triples = read_integer_triples(split_paths["valid"])
    test_triples = read_integer_triples(split_paths["test"])
    expected_counts = manifest.get("counts", {})
    actual_counts = {
        "train": len(train_triples),
        "valid": len(valid_triples),
        "test": len(test_triples),
    }
    mismatches = {
        split: (expected_counts.get(split), actual)
        for split, actual in actual_counts.items()
        if expected_counts.get(split) != actual
    }
    if mismatches:
        raise ValueError(f"Canonical split counts do not match manifest: {mismatches}")
    entity2id = _load_mapping(directory / "entity2id.json")
    relation2id = _load_mapping(directory / "relation2id.json")
    expected_entity_hash = expected_hashes.get("entity_mapping")
    expected_relation_hash = expected_hashes.get("relation_mapping")
    if expected_entity_hash and _sha256_json(entity2id) != expected_entity_hash:
        raise ValueError("Canonical entity mapping hash does not match manifest")
    if expected_relation_hash and _sha256_json(relation2id) != expected_relation_hash:
        raise ValueError("Canonical relation mapping hash does not match manifest")
    bundle = DatasetBundle(
        name=dataset_cfg["name"],
        protocol_version=protocol,
        train_triples=train_triples,
        valid_triples=valid_triples,
        test_triples=test_triples,
        num_entities=int(manifest["counts"]["entities"]),
        num_relations=int(manifest["counts"]["relations"]),
        entity2id=entity2id,
        relation2id=relation2id,
        features=load_processed_feature_bundle(directory),
        manifest=manifest,
    )
    bundle.validate()
    return bundle
