from __future__ import annotations

import json
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


@pytest.mark.parametrize("relative_run_dir", RUN_DIRS)
def test_frozen_checkpoint_loads_strictly(relative_run_dir: str) -> None:
    run_dir = ROOT / relative_run_dir
    if not (run_dir / "best.ckpt").exists():
        pytest.skip("Frozen OpenBG checkpoints are external artifacts and are not present in this checkout.")
    cfg = json.loads((run_dir / "config_merged.json").read_text(encoding="utf-8"))
    bundle = load_dataset_bundle(cfg)
    assert bundle.protocol_version == OPENBG_LEGACY_V1
    model, _ = build_model(cfg, dataset_bundle=bundle)
    state = torch.load(run_dir / "best.ckpt", map_location="cpu")
    model.load_state_dict(state, strict=True)
