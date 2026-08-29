from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def _config(name: str) -> dict:
    return yaml.safe_load((ROOT / "ml" / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


def test_native_released_script_anchors_are_dataset_specific() -> None:
    mkg_w = _config("mkg_w_native")
    db15k = _config("db15k_native")
    assert (mkg_w["model"]["dim"], mkg_w["model"]["margin"]) == (250, 4.0)
    assert (db15k["model"]["dim"], db15k["model"]["margin"]) == (250, 12.0)


def test_adamf_mat_released_script_anchors_are_dataset_specific() -> None:
    mkg_w = _config("mkg_w_adamf_mat")
    db15k = _config("db15k_adamf_mat")
    assert (mkg_w["model"]["dim"], mkg_w["model"]["margin"]) == (200, 12.0)
    assert (db15k["model"]["dim"], db15k["model"]["margin"]) == (250, 12.0)


def test_apkgc_released_script_fusion_anchors_are_dataset_specific() -> None:
    mkg_w = _config("mkg_w_apkgc")
    db15k = _config("db15k_apkgc")
    assert (mkg_w["model"]["num_proj"], mkg_w["model"]["joint_way"]) == (
        1,
        "Mformer_hd_mean",
    )
    assert (db15k["model"]["num_proj"], db15k["model"]["joint_way"]) == (
        2,
        "Mformer_hd_graph",
    )


def test_all_external_starting_configs_are_dev_only() -> None:
    for dataset in ("mkg_w", "db15k"):
        for model in (
            "adamf_mat",
            "apkgc",
            "complex",
            "gate_only",
            "gate_residual",
            "mhyper",
            "native",
            "residual_only",
        ):
            assert _config(f"{dataset}_{model}")["evaluation"]["run_test"] is False
