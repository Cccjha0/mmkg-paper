import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.training.src.utils.config import load_config
from ml.training.src.utils.seed import set_seed
from ml.training.src.data.tsv_reader import read_allow_2or3
from ml.training.src.data.build_true_facts import build_true_facts
from ml.training.src.models.build_model import build_model
from ml.training.src.train.trainer_yaml import TrainerYAML
from ml.training.src.train.trainer_recent import (
    AdversarialGPTrainer,
    AdversarialTrainer,
    OneVsAllTrainer,
)


def resolve_device(requested: str) -> str:
    requested = (requested or "cuda").lower()

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            print("[WARN] cuda not available, switching to mps")
            return "mps"
        print("[WARN] cuda not available, switching to cpu")
        return "cpu"

    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            print("[WARN] mps not available, switching to cuda")
            return "cuda"
        print("[WARN] mps not available, switching to cpu")
        return "cpu"

    if requested == "cpu":
        return "cpu"

    print(f"[WARN] unknown device '{requested}', auto-selecting")
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_trainer(*, model, train_triples, dev_triples, test_triples, num_entities, true_tails, true_heads, cfg):
    """Dispatch to a training engine without changing legacy defaults."""
    engine = cfg["training"].get("engine", "standard").lower()
    trainer_kwargs = {
        "model": model,
        "train_triples": train_triples,
        "dev_triples": dev_triples,
        "test_triples": test_triples,
        "num_entities": num_entities,
        "true_tails": true_tails,
        "true_heads": true_heads,
        "cfg": cfg,
    }
    if engine == "standard":
        return TrainerYAML(**trainer_kwargs)
    if engine == "adversarial":
        return AdversarialTrainer(**trainer_kwargs)
    if engine == "adversarial_gp":
        return AdversarialGPTrainer(**trainer_kwargs)
    if engine == "one_vs_all":
        return OneVsAllTrainer(**trainer_kwargs)
    raise ValueError(
        f"Unsupported training.engine={engine!r}. "
        "Expected one of: standard, adversarial, adversarial_gp, one_vs_all."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="path to experiment yaml, e.g., ml/configs/openbg_img_gate_only.yaml")
    ap.add_argument("--common", default="ml/configs/common.yaml", help="path to common yaml")
    ap.add_argument(
        "--profile-steps",
        type=int,
        default=None,
        help="profile this many training batches, print a [Perf] summary, then stop before evaluation",
    )
    ap.add_argument(
        "--profile-warmup-steps",
        type=int,
        default=2,
        help="unmeasured warmup batches used with --profile-steps (default: 2)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, args.common)
    if args.profile_steps is not None:
        if args.profile_steps <= 0 or args.profile_warmup_steps < 0:
            ap.error("--profile-steps must be positive and --profile-warmup-steps non-negative")
        cfg["training"].update(
            {
                "profile_timing": True,
                "profile_steps": args.profile_steps,
                "profile_warmup_steps": args.profile_warmup_steps,
                "profile_stop_after": True,
            }
        )

    # seed
    seed = cfg["system"].get("seed", 1)
    deterministic = cfg["system"].get("deterministic", False)
    set_seed(seed, deterministic=deterministic)

    # device
    device = resolve_device(cfg["system"].get("device", "cuda"))
    cfg["system"]["device"] = device

    # load triples
    train_path = cfg["dataset"]["train"]
    dev_path = cfg["dataset"]["dev"]
    test_path = cfg["dataset"]["test"]

    train3, _, bad_train = read_allow_2or3(train_path)
    dev3, _, bad_dev = read_allow_2or3(dev_path)
    test3, test2, bad_test = read_allow_2or3(test_path)

    if bad_train or bad_dev or bad_test:
        print(f"[WARN] malformed lines skipped: train={bad_train}, dev={bad_dev}, test={bad_test}")

    if len(train3) == 0 or len(dev3) == 0:
        raise RuntimeError("Train/Dev must contain 3-column triples for training/evaluation.")

    has_labeled_test = len(test3) > 0
    has_query_only_test = len(test2) > 0 and len(test3) == 0
    if not has_labeled_test and not has_query_only_test:
        raise RuntimeError("Test file must contain either 3-column triples or 2-column query pairs.")

    if has_labeled_test:
        print(f"train triples: {len(train3)} | dev triples: {len(dev3)} | test triples: {len(test3)}")
    else:
        print(f"train triples: {len(train3)} | dev triples: {len(dev3)} | test queries: {len(test2)}")
        print("[WARN] test file is query-only (2 columns); final test ranking metrics will be skipped.")

    # build filtered facts. Only labeled triples can contribute to filtered ranking facts.
    true_tails, true_heads = build_true_facts(train3 + dev3 + test3)

    # build model from config
    model, num_entities = build_model(cfg)
    model = model.to(device)

    # run trainer
    trainer = build_trainer(
        model=model,
        train_triples=train3,
        dev_triples=dev3,
        test_triples=test3,
        num_entities=num_entities,
        true_tails=true_tails,
        true_heads=true_heads,
        cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
