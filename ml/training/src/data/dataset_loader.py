from __future__ import annotations

from ml.training.src.data.dataset_spec import DatasetBundle, OPENBG_LEGACY_V1
from ml.training.src.data.readers.openbg_legacy import load_openbg_legacy
from ml.training.src.data.readers.openke_mmkg import load_openke_mmkg


def infer_loader_name(cfg: dict) -> str:
    dataset_cfg = cfg.get("dataset", {})
    explicit = dataset_cfg.get("loader")
    if explicit:
        return str(explicit)
    protocol = cfg.get("protocol", {}).get("version", OPENBG_LEGACY_V1)
    if protocol == OPENBG_LEGACY_V1:
        return "openbg_legacy"
    raise ValueError("General datasets must set dataset.loader explicitly.")


def load_dataset_bundle(cfg: dict) -> DatasetBundle:
    loader = infer_loader_name(cfg)
    if loader == "openbg_legacy":
        return load_openbg_legacy(cfg)
    if loader == "openke_mmkg":
        return load_openke_mmkg(cfg)
    raise ValueError(f"Unsupported dataset.loader={loader!r}")
