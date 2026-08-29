from __future__ import annotations

from pathlib import Path

from ml.training.scripts.preprocess_external_mmkg import (
    build_alignment,
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
