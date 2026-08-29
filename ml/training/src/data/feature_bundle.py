from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ml.training.src.data.dataset_spec import FeatureBundle


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_tensor(directory: Path, candidates: list[str]) -> tuple[torch.Tensor, str]:
    for name in candidates:
        path = directory / name
        if path.exists():
            return torch.load(path, map_location="cpu"), name
    raise FileNotFoundError(f"No cache file found in {directory} for candidates: {candidates}")


def load_openbg_feature_bundle(cache_dir: str | Path, cache_format: str) -> FeatureBundle:
    """Load the frozen OpenBG cache contract without changing its semantics."""

    directory = Path(cache_dir)
    cache_format = (cache_format or "legacy").lower()
    if cache_format == "legacy":
        text, text_name = _load_tensor(directory, ["text_emb.pt"])
        image, image_name = _load_tensor(directory, ["img_emb.pt"])
    elif cache_format == "raw":
        text, text_name = _load_tensor(directory, ["text_feat_raw.pt"])
        image, image_name = _load_tensor(directory, ["img_feat_raw.pt", "img_emb_raw.pt"])
    elif cache_format == "auto":
        try:
            text, text_name = _load_tensor(directory, ["text_feat_raw.pt"])
            image, image_name = _load_tensor(directory, ["img_feat_raw.pt", "img_emb_raw.pt"])
            print(f"[Dataset] using raw OpenBG caches: text={text_name}, image={image_name}")
        except FileNotFoundError:
            text, text_name = _load_tensor(directory, ["text_emb.pt"])
            image, image_name = _load_tensor(directory, ["img_emb.pt"])
            print("[Dataset] raw OpenBG caches unavailable, using legacy projected caches")
    else:
        raise ValueError(f"Unsupported dataset.cache_format: {cache_format}")

    has_img, mask_name = _load_tensor(directory, ["has_img.pt"])
    # Text coverage was historically implicit and complete.  This explicit
    # all-true mask is metadata only; legacy models do not register/use it.
    has_text = torch.ones(int(text.shape[0]), dtype=torch.bool)
    bundle = FeatureBundle(
        text_features=text.float(),
        image_features=image.float(),
        has_text=has_text,
        has_img=has_img.bool(),
        metadata={
            "loader": "openbg_legacy",
            "text_file": text_name,
            "image_file": image_name,
            "image_mask_file": mask_name,
        },
    )
    bundle.validate()
    return bundle


def load_processed_feature_bundle(processed_dir: str | Path) -> FeatureBundle:
    directory = Path(processed_dir)
    required = {
        "text_features": directory / "text_feat.pt",
        "image_features": directory / "img_feat.pt",
        "has_text": directory / "has_text.pt",
        "has_img": directory / "has_img.pt",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Canonical processed feature files are missing: " + ", ".join(missing))
    bundle = FeatureBundle(
        text_features=torch.load(required["text_features"], map_location="cpu").float(),
        image_features=torch.load(required["image_features"], map_location="cpu").float(),
        has_text=torch.load(required["has_text"], map_location="cpu").bool(),
        has_img=torch.load(required["has_img"], map_location="cpu").bool(),
        metadata={"loader": "canonical_processed", "processed_dir": str(directory)},
    )
    bundle.validate()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("hashes", {}).get("canonical_features")
        if expected is not None:
            actual = {
                "text_feat": _sha256_tensor(bundle.text_features),
                "img_feat": _sha256_tensor(bundle.image_features),
                "has_text": _sha256_tensor(bundle.has_text),
                "has_img": _sha256_tensor(bundle.has_img),
            }
            if actual != expected:
                mismatched = sorted(key for key in actual if actual[key] != expected.get(key))
                raise ValueError(f"Canonical feature hashes do not match manifest: {mismatched}")
    return bundle
