from __future__ import annotations

import json
from pathlib import Path

from ml.training.src.data.dataset_spec import DatasetBundle, MMKG_GENERAL_V1
from ml.training.src.data.feature_bundle import load_processed_feature_bundle
from ml.training.src.data.tsv_reader import read_integer_triples


def _load_mapping(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return {str(key): int(value) for key, value in payload.items()}


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
    train_triples = read_integer_triples(directory / "train.tsv")
    valid_triples = read_integer_triples(directory / "valid.tsv")
    test_triples = read_integer_triples(directory / "test.tsv")
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
    bundle = DatasetBundle(
        name=dataset_cfg["name"],
        protocol_version=protocol,
        train_triples=train_triples,
        valid_triples=valid_triples,
        test_triples=test_triples,
        num_entities=int(manifest["counts"]["entities"]),
        num_relations=int(manifest["counts"]["relations"]),
        entity2id=_load_mapping(directory / "entity2id.json"),
        relation2id=_load_mapping(directory / "relation2id.json"),
        features=load_processed_feature_bundle(directory),
        manifest=manifest,
    )
    bundle.validate()
    return bundle
