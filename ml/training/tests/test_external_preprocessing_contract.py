from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.training.scripts.preprocess_external_mmkg import (
    build_alignment,
    construct_canonical_splits,
    inspect_hdf5,
    mkg_y_feature_key_candidates,
    read_openke_mapping,
    read_openke_triples,
)


def test_entity_and_relation_mapping_and_split_integrity(tmp_path: Path) -> None:
    entity_path = tmp_path / "entity2id.txt"
    relation_path = tmp_path / "relation2id.txt"
    split_path = tmp_path / "train2id.txt"
    entity_path.write_text("3\ne0 0\ne1 1\ne2 2\n", encoding="utf-8")
    relation_path.write_text("2\nr0 0\nr1 1\n", encoding="utf-8")
    # OpenKE rows are head, tail, relation; canonical rows are head, relation, tail.
    split_path.write_text("2\n0 1 0\n2 1 1\n", encoding="utf-8")
    entities, n_entities = read_openke_mapping(entity_path, "entity")
    relations, n_relations = read_openke_mapping(relation_path, "relation")
    assert entities == {"e0": 0, "e1": 1, "e2": 2}
    assert relations == {"r0": 0, "r1": 1}
    assert read_openke_triples(split_path, n_entities, n_relations) == [(0, 0, 1), (2, 1, 1)]


def test_feature_alignment_uses_keys_not_hdf5_order() -> None:
    entity2id = {"e0": 0, "e1": 1, "e2": 2}
    key_map = {"e0": "z-last", "e1": "a-first"}
    aligned, missing = build_alignment(entity2id, {"a-first", "z-last"}, key_map.get, "text")
    assert aligned == [["z-last"], ["a-first"], None]
    assert missing == ["e2"]


def test_explicit_feature_alignment_can_share_a_key() -> None:
    aligned, missing = build_alignment(
        {"e0": 0, "e1": 1}, {"shared"}, lambda _entity: "shared", "image"
    )
    assert aligned == [["shared"], ["shared"]]
    assert missing == []


def test_mkg_y_feature_key_candidates_preserve_modality_specific_upstream_conventions() -> None:
    assert mkg_y_feature_key_candidates("AC/DC", "text") == ["AC/DC", "DC"]
    assert mkg_y_feature_key_candidates("AC/DC", "image") == ["AC/DC", "ACDC"]
    assert mkg_y_feature_key_candidates("Example_F.C.", "image") == [
        "Example_F.C.",
        "Example_F.C",
    ]


def test_hdf5_numeric_audit_reports_nonfinite_values(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "features.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("healthy", data=np.asarray([[1.0, 2.0]], dtype=np.float32))
        handle.create_dataset("broken", data=np.asarray([[np.nan, np.inf]], dtype=np.float32))
    _keys, dimension, _dimensions, health = inspect_hdf5(path)
    assert dimension == 2
    assert health["all_finite"] is False
    assert health["nonfinite_values"] == 2
    assert "broken" in health["nonfinite_key_examples"]


def test_db15k_repaired_split_is_deterministic_disjoint_and_train_covered() -> None:
    source = {
        "train": [
            (0, 0, 1),
            (1, 0, 2),
            (2, 0, 3),
            (3, 0, 0),
            (0, 1, 2),
            (2, 1, 0),
            (1, 1, 3),
            (3, 1, 1),
        ],
        # Deliberately contaminated; it must be audited but never used as DEV.
        "valid": [(0, 0, 1), (9, 1, 9)],
        "test": [(4, 0, 5)],
    }
    kwargs = {
        "split_policy": "db15k_train_holdout_v1",
        "db15k_valid_fraction": 0.25,
        "split_seed": 2025,
    }
    first, metadata = construct_canonical_splits("db15k", source, **kwargs)
    second, _ = construct_canonical_splits("db15k", source, **kwargs)
    assert first == second
    assert len(first["valid"]) == 2
    assert first["test"] == source["test"]
    assert not (set(first["train"]) & set(first["valid"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["valid"]) & set(first["test"]))
    train_entities = {entity for h, _, t in first["train"] for entity in (h, t)}
    valid_entities = {entity for h, _, t in first["valid"] for entity in (h, t)}
    assert valid_entities <= train_entities
    assert metadata["name"] == "db15k_train_holdout_v1"


def test_official_policy_rejects_contaminated_splits() -> None:
    source = {
        "train": [(0, 0, 1)],
        "valid": [(0, 0, 1)],
        "test": [(1, 0, 2)],
    }
    with pytest.raises(ValueError, match="Canonical split integrity failed"):
        construct_canonical_splits(
            "db15k",
            source,
            split_policy="official",
            db15k_valid_fraction=0.1,
            split_seed=2025,
        )
