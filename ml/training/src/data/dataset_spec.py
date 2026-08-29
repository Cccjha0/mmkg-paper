from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


OPENBG_LEGACY_V1 = "openbg_legacy_v1"
MMKG_GENERAL_V1 = "mmkg_general_v1"
SUPPORTED_PROTOCOLS = {OPENBG_LEGACY_V1, MMKG_GENERAL_V1}


@dataclass(frozen=True)
class FeatureBundle:
    """Entity-aligned canonical modality tensors.

    Missingness is represented only by ``has_text``/``has_img``.  Feature
    values (including all-zero rows) are never interpreted as availability.
    """

    text_features: torch.Tensor
    image_features: torch.Tensor
    has_text: torch.Tensor
    has_img: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_entities(self) -> int:
        return int(self.text_features.shape[0])

    def validate(self) -> None:
        if self.text_features.ndim != 2 or self.image_features.ndim != 2:
            raise ValueError("text_features and image_features must both be rank-2 tensors.")
        n = self.num_entities
        if int(self.image_features.shape[0]) != n:
            raise ValueError("Text and image feature row counts differ.")
        if self.has_text.ndim != 1 or self.has_img.ndim != 1:
            raise ValueError("has_text and has_img must both be rank-1 tensors.")
        if int(self.has_text.numel()) != n or int(self.has_img.numel()) != n:
            raise ValueError("Availability masks must contain one entry per entity.")
        if self.has_text.dtype != torch.bool or self.has_img.dtype != torch.bool:
            raise ValueError("Availability masks must use torch.bool dtype.")
        if not bool(torch.isfinite(self.text_features).all()):
            raise ValueError("text_features contains NaN or infinite values.")
        if not bool(torch.isfinite(self.image_features).all()):
            raise ValueError("image_features contains NaN or infinite values.")


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    protocol_version: str
    train_triples: list[tuple[int, int, int]]
    valid_triples: list[tuple[int, int, int]]
    test_triples: list[tuple[int, int, int]]
    num_entities: int
    num_relations: int
    entity2id: dict[str, int]
    relation2id: dict[str, int]
    features: FeatureBundle
    manifest: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.protocol_version not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unsupported protocol version: {self.protocol_version!r}")
        self.features.validate()
        if self.features.num_entities != self.num_entities:
            raise ValueError("Feature rows do not match num_entities.")
        _validate_mapping(self.entity2id, self.num_entities, "entity")
        _validate_mapping(self.relation2id, self.num_relations, "relation")
        for split_name, triples in (
            ("train", self.train_triples),
            ("valid", self.valid_triples),
            ("test", self.test_triples),
        ):
            for row, (head, relation, tail) in enumerate(triples, start=1):
                if not (0 <= head < self.num_entities and 0 <= tail < self.num_entities):
                    raise ValueError(f"{split_name} row {row} contains an invalid entity id.")
                if not 0 <= relation < self.num_relations:
                    raise ValueError(f"{split_name} row {row} contains an invalid relation id.")


def _validate_mapping(mapping: dict[str, int], expected_count: int, kind: str) -> None:
    if len(mapping) != expected_count:
        raise ValueError(
            f"{kind} mapping count {len(mapping)} does not match expected count {expected_count}."
        )
    ids = list(mapping.values())
    if len(ids) != len(set(ids)):
        raise ValueError(f"{kind} mapping contains duplicate integer ids.")
    if set(ids) != set(range(expected_count)):
        raise ValueError(f"{kind} mapping ids must be contiguous in [0, {expected_count}).")
