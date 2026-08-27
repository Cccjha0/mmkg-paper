from unittest.mock import patch

import torch

from ml.training.scripts.run_train import build_trainer
from ml.training.src.train.trainer_recent import AdversarialGPTrainer, AdversarialTrainer, OneVsAllTrainer
from ml.training.src.train.trainer_yaml import TrainerYAML


def _cfg(root, engine):
    return {
        "training": {"engine": engine, "lr": 0.001, "sampler": "bernoulli_filtered"},
        "evaluation": {},
        "output": {"root_dir": root, "exp_name": "dispatch_test"},
        "system": {"seed": 1, "device": "cpu"},
        "dataset": {},
    }


def _build(engine):
    model = torch.nn.Linear(1, 1)
    model.dim_e = 2
    with (
        patch("ml.training.src.train.trainer_yaml.make_run_dir", return_value="dispatch-test-run"),
        patch("ml.training.src.train.trainer_yaml.save_json"),
        patch("ml.training.src.train.trainer_yaml.copy_file"),
    ):
        return build_trainer(
            model=model,
            train_triples=[(0, 0, 1)],
            dev_triples=[(0, 0, 1)],
            test_triples=[(0, 0, 1)],
            num_entities=2,
            true_tails={(0, 0): {1}},
            true_heads={(0, 1): {0}},
            cfg=_cfg("unused", engine),
        )


def test_training_engine_dispatches_without_affecting_standard_default():
    assert isinstance(_build("standard"), TrainerYAML)
    assert isinstance(_build("adversarial"), AdversarialTrainer)
    assert isinstance(_build("adversarial_gp"), AdversarialGPTrainer)
    assert isinstance(_build("one_vs_all"), OneVsAllTrainer)


def test_training_engine_rejects_unknown_value():
    try:
        _build("unknown")
    except ValueError as exc:
        assert "Unsupported training.engine" in str(exc)
    else:
        raise AssertionError("Unknown training engine must fail fast.")
