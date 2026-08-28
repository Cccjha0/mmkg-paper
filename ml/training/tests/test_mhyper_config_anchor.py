from pathlib import Path

import yaml


def test_mhyper_formal_config_matches_official_command_anchor():
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load((root / "ml/configs/openbg_img_mhyper.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["rank"] == 128
    assert cfg["model"]["init_size"] == 0.001
    assert cfg["model"]["noise_preserve_ratio"] == 0.2
    assert cfg["training"]["engine"] == "one_vs_all"
    assert cfg["training"]["optimizer"] == "adagrad"
    assert cfg["training"]["lr"] == 0.1
    assert cfg["training"]["batch_size"] == 1000
    assert cfg["training"]["wn3_weight"] == 0.005
    assert cfg["training"]["epochs"] == 200
    assert cfg["training"]["eval_every"] == 5
    assert cfg["training"]["early_stop_patience"] is None
