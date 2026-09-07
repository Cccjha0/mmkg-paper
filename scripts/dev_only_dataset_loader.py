from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml.training.src.data.dataset_spec import DatasetBundle, MMKG_GENERAL_V1
from ml.training.src.data.feature_bundle import load_processed_feature_bundle
from ml.training.src.data.tsv_reader import read_integer_triples


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _read_mapping(path: Path) -> dict[str, int]:
    return {str(key): int(value) for key, value in _read_json(path).items()}


def _reject_test_path(path: Path) -> None:
    tokens = {part.lower() for part in path.parts}
    if "test" in tokens or path.name.lower().startswith("test"):
        raise RuntimeError(f"DEV-only loader refuses TEST paths: {path}")


def load_dev_only_dataset_bundle(cfg: dict) -> tuple[DatasetBundle, list[Path]]:
    """Load TRAIN/DEV model assets without opening the canonical TEST row file."""
    if cfg.get("dataset", {}).get("loader") != "openke_mmkg":
        raise ValueError("DEV-only loader supports frozen openke_mmkg runs only")
    if cfg.get("protocol", {}).get("version") != MMKG_GENERAL_V1:
        raise ValueError("DEV-only loader requires mmkg_general_v1")

    directory = Path(cfg["dataset"]["processed_dir"])
    paths = {
        "manifest": directory / "manifest.json",
        "train": directory / "train.tsv",
        "valid": directory / "valid.tsv",
        "entity2id": directory / "entity2id.json",
        "relation2id": directory / "relation2id.json",
        "text_feat": directory / "text_feat.pt",
        "img_feat": directory / "img_feat.pt",
        "has_text": directory / "has_text.pt",
        "has_img": directory / "has_img.pt",
    }
    for path in paths.values():
        _reject_test_path(path)
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = _read_json(paths["manifest"])
    train = read_integer_triples(paths["train"])
    valid = read_integer_triples(paths["valid"])
    expected = manifest.get("counts", {})
    if len(train) != int(expected.get("train", -1)):
        raise RuntimeError("TRAIN count does not match the canonical manifest")
    if len(valid) != int(expected.get("valid", -1)):
        raise RuntimeError("DEV count does not match the canonical manifest")
    expected_hashes = manifest.get("hashes", {}).get("splits", {})
    if expected_hashes.get("train") and _sha256_file(paths["train"]) != expected_hashes["train"]:
        raise RuntimeError("TRAIN hash does not match the canonical manifest")
    if expected_hashes.get("valid") and _sha256_file(paths["valid"]) != expected_hashes["valid"]:
        raise RuntimeError("DEV hash does not match the canonical manifest")

    bundle = DatasetBundle(
        name=str(manifest["dataset"]),
        protocol_version=MMKG_GENERAL_V1,
        train_triples=train,
        valid_triples=valid,
        test_triples=[],
        num_entities=int(manifest["counts"]["entities"]),
        num_relations=int(manifest["counts"]["relations"]),
        entity2id=_read_mapping(paths["entity2id"]),
        relation2id=_read_mapping(paths["relation2id"]),
        features=load_processed_feature_bundle(directory),
        manifest={
            "dev_only_no_test_access": True,
            "canonical_manifest_path": str(paths["manifest"]),
        },
    )
    bundle.validate()
    return bundle, list(paths.values())
