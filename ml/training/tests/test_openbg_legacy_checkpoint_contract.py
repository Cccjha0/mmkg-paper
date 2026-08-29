from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pytest
import torch

from ml.training.src.data.dataset_loader import load_dataset_bundle
from ml.training.src.data.dataset_spec import OPENBG_LEGACY_V1
from ml.training.src.models.build_model import build_model


ROOT = Path(__file__).resolve().parents[3]
RUN_DIRS = [
    "ml/artifacts/outputs/openbg_img_gate_only/20260329_202546_seed1",
    "ml/artifacts/outputs/openbg_img_residual_only/20260329_202016_seed1",
    "ml/artifacts/outputs/openbg_img_gated_vec_res_rel/20260329_205816_seed1",
]
FIXTURE = json.loads((ROOT / "ml/training/tests/fixtures/openbg_legacy_regression_v1.json").read_text(encoding="utf-8"))
EXPECTED_BY_RUN_DIR = {entry["run_dir"]: entry for entry in FIXTURE["models"].values()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_checkpoint_artifacts_have_immutable_pins() -> None:
    assert set(RUN_DIRS) == set(EXPECTED_BY_RUN_DIR)
    for expected in EXPECTED_BY_RUN_DIR.values():
        assert int(expected["checkpoint_bytes"]) > 0
        assert len(str(expected["checkpoint_sha256"])) == 64


@pytest.mark.parametrize("relative_run_dir", RUN_DIRS)
def test_frozen_checkpoint_loads_strictly(relative_run_dir: str) -> None:
    run_dir = ROOT / relative_run_dir
    checkpoint = run_dir / "best.ckpt"
    if not checkpoint.exists():
        if os.environ.get("REQUIRE_OPENBG_LEGACY_CHECKPOINTS") == "1":
            pytest.fail(f"Required frozen OpenBG checkpoint is missing: {checkpoint}")
        pytest.skip(
            "Frozen OpenBG checkpoint is external. Set REQUIRE_OPENBG_LEGACY_CHECKPOINTS=1 "
            "in the release regression job to make absence a failure."
        )
    expected = EXPECTED_BY_RUN_DIR[relative_run_dir]
    assert checkpoint.stat().st_size == int(expected["checkpoint_bytes"])
    assert _sha256_file(checkpoint) == expected["checkpoint_sha256"]
    cfg = json.loads((run_dir / "config_merged.json").read_text(encoding="utf-8"))
    bundle = load_dataset_bundle(cfg)
    assert bundle.protocol_version == OPENBG_LEGACY_V1
    model, _ = build_model(cfg, dataset_bundle=bundle)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
