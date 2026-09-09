from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.eval_openbg_dynasemble import (
    ReleasedDynaSembleSelector,
    clustered_bootstrap_interval,
    frozen_method_config,
    normalize_and_features,
    validate_baseline_selection,
)


def test_released_selector_topology_is_unchanged() -> None:
    selector = ReleasedDynaSembleSelector()
    layers = list(selector.layers)
    assert [type(layer) for layer in layers] == [
        nn.Linear,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
    ]
    assert (layers[0].in_features, layers[0].out_features) == (4, 16)
    assert (layers[1].in_features, layers[1].out_features) == (16, 16)
    assert (layers[3].in_features, layers[3].out_features) == (16, 1)


def test_pair_generalization_changes_only_expert_identity() -> None:
    native = frozen_method_config("M-Hyper", "NativE")
    adamf = frozen_method_config("M-Hyper", "AdaMF-MAT")
    differing = {key for key in native if native[key] != adamf[key]}
    assert differing == {"fixed_weight_expert"}
    assert native["mlp_topology"] == [4, 16, 16, 1]
    assert native["num_epochs"] == 1
    assert native["strict_negative_count"] == 9999
    assert native["train_batch_size"] == 16


def test_filtered_minmax_features_and_reference() -> None:
    scores = torch.tensor([[1.0, 3.0, float("-inf"), 2.0]])
    normalized, reference, features = normalize_and_features(
        scores, torch.tensor([2.0])
    )
    assert normalized[0, :2].tolist() == pytest.approx([0.0, 1.0])
    assert torch.isneginf(normalized[0, 2])
    assert normalized[0, 3].item() == pytest.approx(0.5)
    assert reference.tolist() == pytest.approx([0.5])
    assert features.shape == (1, 2)
    assert features[0, 0].item() == pytest.approx(0.5)
    assert features[0, 1].item() == pytest.approx(0.25)


def test_clustered_bootstrap_is_reproducible_and_clusters_seeds_directions() -> None:
    rows = []
    for triple, delta in ((10, 0.1), (20, -0.05), (30, 0.2)):
        for seed in (1, 2, 3):
            for direction in ("head", "tail"):
                rows.append(
                    {
                        "head_id": triple,
                        "relation_id": 0,
                        "tail_id": triple + 1,
                        "rr_dynasemble": delta,
                        "rr_global": 0.0,
                        "seed": seed,
                        "direction": direction,
                    }
                )
    first = clustered_bootstrap_interval(rows, "rr_dynasemble", "rr_global")
    second = clustered_bootstrap_interval(rows, "rr_dynasemble", "rr_global")
    assert first == second
    assert first["n_original_triple_clusters"] == 3
    assert first["mean_delta"] == pytest.approx((0.1 - 0.05 + 0.2) / 3)


def test_openbg_frozen_config_matches_existing_lock_if_present() -> None:
    lock_path = Path(
        "outputs/openbg_img/openbg_dynasemble_test/mhyper_native/dev_lock.json"
    )
    if not lock_path.exists():
        pytest.skip("OpenBG lock is not present in this checkout")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["method_config"] == frozen_method_config("M-Hyper", "NativE")


@pytest.mark.parametrize(
    ("selection_path", "pair_name", "expert_b"),
    [
        (
            "outputs/mkg_w/anchored_dynamic/mhyper_native_seed123/full_ranking/selection.json",
            "mkgw_mhyper_native_dynasemble",
            "NativE",
        ),
        (
            "outputs/mkg_w/anchored_dynamic/mhyper_adamf_seed123/full_ranking/selection.json",
            "mkgw_mhyper_adamf_dynasemble",
            "AdaMF-MAT",
        ),
        (
            "outputs/db15k/anchored_dynamic/mhyper_native_seed123/full_ranking/selection.json",
            "db15k_mhyper_native_dynasemble",
            "NativE",
        ),
        (
            "outputs/db15k/anchored_dynamic/mhyper_adamf_seed123/full_ranking/selection.json",
            "db15k_mhyper_adamf_dynasemble",
            "AdaMF-MAT",
        ),
    ],
)
def test_four_pair_baseline_locks_match_cli_identity(
    selection_path: str, pair_name: str, expert_b: str
) -> None:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    validate_baseline_selection(selection, pair_name, "M-Hyper", expert_b)
