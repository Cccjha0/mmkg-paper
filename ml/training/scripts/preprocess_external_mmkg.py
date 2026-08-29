from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Callable
from urllib.parse import unquote

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


OPENKE_FILES = {
    "train": "train2id.txt",
    "valid": "valid2id.txt",
    "test": "test2id.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an audited canonical MKG-W or DB15K dataset from official OpenKE splits and keyed HDF5 features."
    )
    parser.add_argument("--dataset", required=True, choices=["mkg_w", "db15k"])
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--text-h5", type=Path, required=True)
    parser.add_argument("--image-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-key-map",
        type=Path,
        help="TSV with entity token and HDF5 key. Required for MKG-W; optional override for DB15K.",
    )
    parser.add_argument(
        "--db15k-same-as",
        type=Path,
        help="Official MMKB DB15K_SameAsLink.txt used to align DBpedia entities to image Freebase MIDs.",
    )
    parser.add_argument("--text-pooling", default="mean", choices=["mean"])
    parser.add_argument("--image-pooling", default="mean", choices=["mean"])
    parser.add_argument("--source-version", default="official OpenKE benchmark files (repository external mirror)")
    parser.add_argument("--text-encoder", default="unknown (upstream HDF5 metadata not provided)")
    parser.add_argument("--image-encoder", default="BEIT_16-224 (from supplied filename)")
    parser.add_argument("--audit-only", action="store_true", help="Validate and print a report without writing tensors.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def read_openke_mapping(path: Path, kind: str) -> tuple[dict[str, int], int]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty {kind} mapping: {path}")
    declared = int(lines[0])
    mapping: dict[str, int] = {}
    ids: set[int] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        try:
            token, raw_id = line.rsplit(maxsplit=1)
            integer_id = int(raw_id)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid OpenKE mapping row") from exc
        if token in mapping:
            raise ValueError(f"{path}:{line_number}: duplicate {kind} token {token!r}")
        if integer_id in ids:
            raise ValueError(f"{path}:{line_number}: duplicate {kind} id {integer_id}")
        mapping[token] = integer_id
        ids.add(integer_id)
    if len(mapping) != declared:
        raise ValueError(f"{path}: declared {declared} {kind}s but found {len(mapping)} rows")
    if ids != set(range(declared)):
        raise ValueError(f"{path}: {kind} ids are not contiguous in [0, {declared})")
    return mapping, declared


def read_openke_triples(path: Path, num_entities: int, num_relations: int) -> list[tuple[int, int, int]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty split: {path}")
    declared = int(lines[0])
    triples: list[tuple[int, int, int]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number}: expected head tail relation")
        head, tail, relation = (int(value) for value in parts)
        if not (0 <= head < num_entities and 0 <= tail < num_entities):
            raise ValueError(f"{path}:{line_number}: entity id outside [0, {num_entities})")
        if not 0 <= relation < num_relations:
            raise ValueError(f"{path}:{line_number}: relation id outside [0, {num_relations})")
        triples.append((head, relation, tail))
    if len(triples) != declared:
        raise ValueError(f"{path}: declared {declared} triples but found {len(triples)} rows")
    return triples


def read_feature_key_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\r\n")
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: expected entity<TAB>feature_key")
            entity, key = parts
            if not entity or not key:
                raise ValueError(f"{path}:{line_number}: entity and feature_key must both be non-empty")
            if entity in mapping and mapping[entity] != key:
                raise ValueError(f"{path}:{line_number}: conflicting key for {entity!r}")
            mapping[entity] = key
    return mapping


def read_db15k_same_as(path: Path) -> dict[str, list[str]]:
    pattern = re.compile(r"^(?P<mid>/m/\S+)\s+<SameAs>\s+(?P<entity><http://dbpedia\.org/resource/[^>]+>)\s+\.$")
    mapping: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = pattern.match(line.strip())
            if match is None:
                raise ValueError(f"{path}:{line_number}: invalid MMKB sameAs row")
            entity = match.group("entity")
            mid = match.group("mid").replace("/m/", "m.", 1)
            if mid not in mapping.setdefault(entity, []):
                mapping[entity].append(mid)
    return mapping


def dbpedia_basename(entity: str) -> str:
    prefix = "<http://dbpedia.org/resource/"
    if not entity.startswith(prefix) or not entity.endswith(">"):
        raise ValueError(f"Not a DBpedia resource entity: {entity!r}")
    return unquote(entity[len(prefix) : -1])


def build_key_resolvers(
    args: argparse.Namespace,
    image_h5_keys: set[str] | None = None,
) -> tuple[Callable[[str], str | list[str] | None], Callable[[str], str | list[str] | None], dict]:
    explicit = read_feature_key_map(args.feature_key_map)
    metadata: dict[str, object] = {}
    if args.dataset == "mkg_w":
        if not explicit:
            raise ValueError(
                "MKG-W HDF5 keys are page titles while entity2id uses Wikidata QIDs. "
                "Provide --feature-key-map; ordering-based alignment is forbidden."
            )
        metadata["feature_key_map"] = {"path": str(args.feature_key_map), "sha256": sha256_file(args.feature_key_map)}
        return explicit.get, explicit.get, metadata

    same_as: dict[str, list[str]] = {}
    if args.db15k_same_as is not None:
        same_as = read_db15k_same_as(args.db15k_same_as)
        metadata["db15k_same_as"] = {
            "path": str(args.db15k_same_as),
            "sha256": sha256_file(args.db15k_same_as),
            "entities_with_multiple_mids": sum(len(mids) > 1 for mids in same_as.values()),
            "disambiguation": "pool all official SameAs MIDs present in image HDF5",
        }
    if not same_as and not explicit:
        raise ValueError(
            "DB15K image HDF5 keys are Freebase MIDs. Provide the official --db15k-same-as file "
            "or an explicit --feature-key-map."
        )

    def text_key(entity: str) -> str | None:
        return explicit.get(entity, dbpedia_basename(entity))

    def image_key(entity: str) -> str | None:
        if entity in explicit:
            return explicit[entity]
        candidates = same_as.get(entity, [])
        present = [mid for mid in candidates if image_h5_keys is not None and mid in image_h5_keys]
        # A DBpedia entity may have more than one official Freebase SameAs MID,
        # and more than one can carry image patches. Pool all explicitly linked
        # keys instead of silently selecting one by file/order.
        return present or None

    return text_key, image_key, metadata


def inspect_hdf5(path: Path) -> tuple[set[str], int, dict[str, int]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 preprocessing requires h5py; install requirements.txt first.") from exc
    with h5py.File(path, "r") as handle:
        keys = set(handle.keys())
        if not keys:
            raise ValueError(f"HDF5 file contains no root datasets: {path}")
        dimensions: dict[str, int] = {}
        for key in keys:
            dataset = handle[key]
            if dataset.ndim != 2 or dataset.shape[0] < 1:
                raise ValueError(f"{path}:{key}: expected a non-empty [K, D] dataset")
            dimensions[key] = int(dataset.shape[1])
        unique_dims = set(dimensions.values())
        if len(unique_dims) != 1:
            raise ValueError(f"{path}: inconsistent feature dimensions: {sorted(unique_dims)}")
        return keys, unique_dims.pop(), dimensions


def build_alignment(
    entity2id: dict[str, int],
    h5_keys: set[str],
    resolver: Callable[[str], str | list[str] | None],
    modality: str,
) -> tuple[list[list[str] | None], list[str]]:
    aligned: list[list[str] | None] = [None] * len(entity2id)
    missing: list[str] = []
    for entity, entity_id in entity2id.items():
        resolved = resolver(entity)
        keys = [resolved] if isinstance(resolved, str) else list(resolved or [])
        keys = [key for key in keys if key in h5_keys]
        if not keys:
            missing.append(entity)
            continue
        # Explicit crosswalks may legitimately map two KG entities to the same
        # Wikipedia page/Freebase media node. Record this in the audit rather
        # than guessing a winner by entity order.
        aligned[entity_id] = keys
    return aligned, missing


def pool_hdf5(path: Path, aligned_keys: list[list[str] | None], feature_dim: int) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    features = np.zeros((len(aligned_keys), feature_dim), dtype=np.float32)
    present = np.zeros(len(aligned_keys), dtype=np.bool_)
    with h5py.File(path, "r") as handle:
        for entity_id, keys in enumerate(aligned_keys):
            if keys is None:
                continue
            values = np.concatenate([np.asarray(handle[key], dtype=np.float32) for key in keys], axis=0)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite {entity_id=} feature values for keys={keys!r}")
            features[entity_id] = values.mean(axis=0, dtype=np.float32)
            present[entity_id] = True
    return features, present


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_triples(path: Path, triples: list[tuple[int, int, int]]) -> None:
    path.write_text("".join(f"{h}\t{r}\t{t}\n" for h, r, t in triples), encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark = args.benchmark_dir
    entity2id, num_entities = read_openke_mapping(benchmark / "entity2id.txt", "entity")
    relation2id, num_relations = read_openke_mapping(benchmark / "relation2id.txt", "relation")
    explicit_keys = read_feature_key_map(args.feature_key_map)
    unknown_crosswalk_entities = sorted(set(explicit_keys) - set(entity2id))
    if unknown_crosswalk_entities:
        raise ValueError(
            "Feature crosswalk contains entities outside entity2id.txt: "
            f"{unknown_crosswalk_entities[:10]}"
        )
    splits = {
        name: read_openke_triples(benchmark / filename, num_entities, num_relations)
        for name, filename in OPENKE_FILES.items()
    }
    text_keys, text_dim, _ = inspect_hdf5(args.text_h5)
    image_keys, image_dim, _ = inspect_hdf5(args.image_h5)
    text_resolver, image_resolver, resolver_metadata = build_key_resolvers(args, image_h5_keys=image_keys)
    aligned_text, missing_text = build_alignment(entity2id, text_keys, text_resolver, "text")
    aligned_image, missing_image = build_alignment(entity2id, image_keys, image_resolver, "image")
    aligned_text_count = num_entities - len(missing_text)
    aligned_image_count = num_entities - len(missing_image)
    if aligned_text_count == 0 or aligned_image_count == 0:
        raise ValueError(
            "Feature alignment produced zero observed entities for a modality: "
            f"text={aligned_text_count}, image={aligned_image_count}. "
            "Check the HDF5 key namespace and explicit crosswalk."
        )

    audit = {
        "dataset": args.dataset,
        "status": "pass",
        "counts": {
            "entities": num_entities,
            "relations": num_relations,
            "train": len(splits["train"]),
            "valid": len(splits["valid"]),
            "test": len(splits["test"]),
        },
        "features": {
            "text": {
                "hdf5_keys": len(text_keys),
                "aligned_entities": aligned_text_count,
                "missing_entities": len(missing_text),
                "coverage": aligned_text_count / num_entities,
                "dimension": text_dim,
                "pooling": args.text_pooling,
                "entities_with_multiple_keys": sum(keys is not None and len(keys) > 1 for keys in aligned_text),
                "shared_keys": sum(len(keys or []) for keys in aligned_text)
                - len({key for keys in aligned_text if keys for key in keys}),
            },
            "image": {
                "hdf5_keys": len(image_keys),
                "aligned_entities": aligned_image_count,
                "missing_entities": len(missing_image),
                "coverage": aligned_image_count / num_entities,
                "dimension": image_dim,
                "pooling": args.image_pooling,
                "entities_with_multiple_keys": sum(keys is not None and len(keys) > 1 for keys in aligned_image),
                "shared_keys": sum(len(keys or []) for keys in aligned_image)
                - len({key for keys in aligned_image if keys for key in keys}),
            },
        },
        "missing_examples": {"text": missing_text[:20], "image": missing_image[:20]},
        "alignment": resolver_metadata,
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.audit_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "audit_report.json", audit)
        print(f"[OK] audit report written to {args.output_dir / 'audit_report.json'}")
        return

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Writing canonical .pt caches requires PyTorch.") from exc
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    text_features, has_text = pool_hdf5(args.text_h5, aligned_text, text_dim)
    image_features, has_img = pool_hdf5(args.image_h5, aligned_image, image_dim)
    write_triples(output / "train.tsv", splits["train"])
    write_triples(output / "valid.tsv", splits["valid"])
    write_triples(output / "test.tsv", splits["test"])
    write_json(output / "entity2id.json", entity2id)
    write_json(output / "relation2id.json", relation2id)
    torch.save(torch.from_numpy(text_features), output / "text_feat.pt")
    torch.save(torch.from_numpy(image_features), output / "img_feat.pt")
    torch.save(torch.from_numpy(has_text), output / "has_text.pt")
    torch.save(torch.from_numpy(has_img), output / "has_img.pt")

    manifest = {
        **audit,
        "protocol": "mmkg_general_v1",
        "split": "official",
        "source": args.source_version,
        "feature_sources": {
            "text": {
                "path": str(args.text_h5),
                "sha256": sha256_file(args.text_h5),
                "encoder": args.text_encoder,
            },
            "image": {
                "path": str(args.image_h5),
                "sha256": sha256_file(args.image_h5),
                "encoder": args.image_encoder,
            },
        },
        "hashes": {
            "entity_mapping": sha256_json(entity2id),
            "relation_mapping": sha256_json(relation2id),
            "splits": {name: sha256_file(output / f"{name}.tsv") for name in splits},
            "canonical_features": {
                "text_feat": sha256_array(text_features),
                "img_feat": sha256_array(image_features),
                "has_text": sha256_array(has_text),
                "has_img": sha256_array(has_img),
            },
        },
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "audit_report.json", audit)
    print(f"[OK] canonical dataset written to {output}")


if __name__ == "__main__":
    main()
