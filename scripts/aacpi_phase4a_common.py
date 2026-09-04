from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.aacpi_phase3a_common import REPRESENTATION_FEATURES


REPRESENTATIONS = ("C0", "C1", "C2", "C3", "C4")
C0_FEATURES = list(REPRESENTATION_FEATURES["R3"])
C1_ADDITIONS = [
    "c1_known_entity_neighborhood_fraction",
    "c1_relation_unique_heads_log1p",
    "c1_relation_unique_tails_log1p",
    "c1_relation_tails_per_head_log1p",
    "c1_relation_heads_per_tail_log1p",
    "c1_relation_head_tail_log_ratio",
    "c1_relation_known_role_diversity_log1p",
]
C2_ADDITIONS = [
    "c2_relation_known_image_support",
    "c2_relation_known_text_support",
]
LATENT_DIM = 16
LATENT_FEATURES = [
    *[f"c3_z_a_{i:02d}" for i in range(LATENT_DIM)],
    *[f"c3_z_b_{i:02d}" for i in range(LATENT_DIM)],
    *[f"c3_z_diff_{i:02d}" for i in range(LATENT_DIM)],
    *[f"c3_z_abs_diff_{i:02d}" for i in range(LATENT_DIM)],
]
STATIC_FEATURES = {
    "C0": C0_FEATURES,
    "C1": [*C0_FEATURES, *C1_ADDITIONS],
    "C2": [*C0_FEATURES, *C2_ADDITIONS],
    "C3": C0_FEATURES,
    "C4": [*C0_FEATURES, *C1_ADDITIONS, *C2_ADDITIONS],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def reject_test_path(path: Path) -> None:
    tokens = {part.lower() for part in path.parts}
    if "test" in tokens or path.name.lower().startswith("test"):
        raise RuntimeError(f"Phase 4A refuses TEST-like path: {path}")


def feature_contract() -> dict:
    return {
        "schema_version": 1,
        "status": "frozen_before_systematic_phase4a_run",
        "representations": {
            "C0": {"static": STATIC_FEATURES["C0"], "latent": []},
            "C1": {"static": STATIC_FEATURES["C1"], "latent": []},
            "C2": {"static": STATIC_FEATURES["C2"], "latent": []},
            "C3": {"static": STATIC_FEATURES["C3"], "latent": LATENT_FEATURES},
            "C4": {"static": STATIC_FEATURES["C4"], "latent": LATENT_FEATURES},
        },
        "latent_preprocessing": {
            "method": "separate_expert_pca",
            "solver": "frozen_halko_randomized_svd",
            "oversamples": 8,
            "power_iterations": 2,
            "dimension_per_expert": LATENT_DIM,
            "fit_scope": "current_training_original_triple_groups_only",
            "inner_cv_rule": "fit on inner-training groups for inner validation",
            "outer_oof_rule": "fit on outer-training groups for outer holdout",
            "target_used": False,
            "component_sign_rule": "largest-absolute-loading is positive",
            "interactions": ["projected_z_a", "projected_z_b", "z_a_minus_z_b", "absolute_difference"],
        },
        "answer_agnostic": True,
        "target_fields_used": [],
    }
