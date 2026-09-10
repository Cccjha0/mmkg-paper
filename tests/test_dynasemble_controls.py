"""Small synthetic CPU tests only; no experiment assets or GPU training."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from router.dynasemble_controls import (ControlSelector, coefficients, complete_order_equal,
    fact_sets, grouped_folds, training_candidates, choose_configuration, VARIANTS)
from scripts.eval_openbg_dynasemble import normalize_and_features
from scripts import paper_a_dynasemble_controls as pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace():
    path = (ROOT / "tmp" / ("dynasemble_test_" + uuid.uuid4().hex)).resolve()
    assert path.is_relative_to((ROOT / "tmp").resolve())
    path.mkdir(parents=True)
    return path


def upstream_class():
    path = ROOT / "external/KGC-Ensemble/NBFNet/script/selector.py"
    if not path.exists():
        pytest.skip("Optional pinned upstream checkout absent; source audit report records the local comparison")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Selector")
    namespace = {"torch": torch, "nn": nn, "np": np, "json": json}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Selector"]


def test_pinned_source_forward_loss_and_gradient_equivalence():
    upstream = upstream_class()
    cfg = SimpleNamespace(model=SimpleNamespace(init=2.))
    torch.manual_seed(23)
    original = upstream(cfg, nn.Identity(), None, None, None, 2, [16, 16], 4)
    torch.manual_seed(23)
    current = ControlSelector("R1_source")
    for i in (0, 1, 3):
        torch.testing.assert_close(original.mlp[i].weight, current.layers[i].weight, rtol=0, atol=0)
        torch.testing.assert_close(original.mlp[i].bias, current.layers[i].bias, rtol=0, atol=0)
    x = torch.rand(8, 4)
    scores_a, scores_b = torch.rand(8, 12), torch.rand(8, 12)
    predicted_old = original.mlp(x)*scores_a+scores_b
    predicted_new = current(x)[:, None]*scores_a+scores_b
    torch.testing.assert_close(predicted_old, predicted_new, rtol=0, atol=0)
    loss = nn.MultiMarginLoss(margin=2.)
    old_loss = loss(predicted_old, torch.zeros(8, dtype=torch.long))
    new_loss = loss(predicted_new, torch.zeros(8, dtype=torch.long))
    torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
    old_loss.backward()
    new_loss.backward()
    for i in (0, 1, 3):
        torch.testing.assert_close(original.mlp[i].weight.grad, current.layers[i].weight.grad, rtol=0, atol=0)


def test_source_features_before_gather_and_gold_independence():
    original = upstream_class()
    raw = torch.tensor([[3., 8., 2., 1., 5., 4., 9., 11., 10., 0., 7., 6.]])
    normalized, _, features = normalize_and_features(raw)
    outputs = []
    for candidates in (torch.tensor([[1, 2, 4]]), torch.tensor([[2, 1, 7]])):
        scores, feat = original.get_features_and_normalize(SimpleNamespace(training=True), raw.clone(), candidates)
        torch.testing.assert_close(scores, normalized.gather(1, candidates))
        torch.testing.assert_close(torch.cat(feat, 1), features)
        outputs.append(torch.cat(feat, 1))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    # The historical sampled-support path differs, so this test detects that regression.
    _, _, sampled_features = normalize_and_features(raw[:, [1, 2, 4]])
    assert not torch.allclose(sampled_features, features)


def test_dead_relu_and_softplus_gradient_diagnostic():
    for variant, expect_zero in (("R2_init", True), ("R3_softplus", False)):
        model = ControlSelector(variant)
        with torch.no_grad():
            model.layers[3].weight.zero_()
            model.layers[3].bias.fill_(-2.)
        model(torch.ones(3, 4)).sum().backward()
        assert bool(model.layers[3].bias.grad == 0) == expect_zero


@pytest.mark.parametrize("variant", VARIANTS)
def test_gold_or_filter_metadata_cannot_enter_control_selector(variant):
    features = torch.tensor([[.8, .01, .7, .02], [.8, .01, .7, .02]])
    model = ControlSelector(variant)
    values = model(features)
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    assert model.forward.__code__.co_argcount == 2


def test_loss_sampling_uses_supplied_training_facts_only():
    a = np.tile(np.arange(12, dtype=np.float32), (2, 1))
    queries = np.array([[0, 0, 1, 1], [1, 0, 0, 0]])
    facts = fact_sets([[0, 0, 1], [0, 0, 2], [1, 0, 0], [2, 0, 0]])
    left, right = training_candidates(a, a, np.array([99, 98]), np.array([97, 96]), queries, facts, 100, np.random.default_rng(5))
    assert left[:, 0].tolist() == [99, 98]
    assert not np.isin(left[:, 1:], [1, 2]).any()
    assert (left[:, 1:] == 3).any()  # A fact not in the training set is not silently removed.
    np.testing.assert_array_equal(left[:, 1:], right[:, 1:])


def test_role_swap_and_finite_weight_ranking_endpoint():
    w = torch.tensor([1., 3.])
    a, b = coefficients(w, "R3_softplus")
    sa, sb = coefficients(w, "R4_swap")
    torch.testing.assert_close(a, sb)
    torch.testing.assert_close(b, sa)
    primary, secondary = np.array([0., 1., 2.]), np.array([0., 3., 1.])
    assert 3/4 != 1
    assert complete_order_equal(primary, 3*primary+secondary)
    assert not complete_order_equal(np.array([3., 2., 1.]), np.array([3., 1., 2.]))
    assert not complete_order_equal(np.array([1., 1., 2.]), np.array([1., 1.1, 2.]))


def test_folds_and_dev_selection_do_not_drop_seed_failures():
    triples = np.array([[i, 0, i+1] for i in range(9)])
    f = grouped_folds(triples)
    assert set(f) == {0, 1, 2}
    with pytest.raises(ValueError, match="Duplicate"):
        grouped_folds(np.concatenate((triples, triples[:1])))
    rows = [{"variant": "R3_softplus", "learning_rate": lr, "epoch": 1,
             "base_seed": 1, "selector_seed": s, "fold": 0, "mrr": .2 if s == 11 else 0., "n": 4}
            for lr in (1e-5, 5e-5) for s in (11, 23)]
    assert choose_configuration(rows)["heldout_dev_mrr"] == .1
    with pytest.raises(ValueError, match="coverage"):
        choose_configuration(rows[:-1])
    rows[0]["mrr"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        choose_configuration(rows)


def test_synthetic_cache_fit_lock_test_summary_and_tamper(workspace, monkeypatch):
    # A few twelve-entity toy queries; exercise the real orchestration without base assets.
    torch.set_num_threads(1)
    monkeypatch.setattr(pipeline, "NEGATIVES", 4)
    monkeypatch.setattr(pipeline, "SELECTOR_SEEDS", (11,))
    monkeypatch.setattr(pipeline, "LEARNING_RATES", (5e-5,))
    monkeypatch.setattr(pipeline, "CHECKPOINT_EPOCHS", (1,))
    monkeypatch.setattr(pipeline, "pair_assets", lambda pair: ["synthetic"])
    monkeypatch.setattr(pipeline, "cache_assets", lambda *args: {"synthetic": True})
    class Toy(nn.Module):
        def score_tail(self, q):
            return torch.sin(q[:, 0].float()*.7 + q[:, 2].float()*.4) + q[:, 2]*.1
        def score_head(self, q):
            return torch.cos(q[:, 2].float()*.3 + q[:, 0].float()*.5) + q[:, 0]*.1
    def load(name, path, device, dev_only_no_test_access=False):
        bs = int(path.name[-1])
        bundle = SimpleNamespace(name="toy", protocol_version="v1", entity2id={str(i): i for i in range(12)}, relation2id={"r": 0},
            train_triples=[(0, 0, 2)], valid_triples=[(i, 0, i+1) for i in range(6)],
            test_triples=[] if dev_only_no_test_access else [(i, 0, i+2) for i in range(6)])
        return SimpleNamespace(name=name, seed=bs, bundle=bundle, model=Toy(), num_entities=12, query_batch_size=4, chunk_size=5)
    monkeypatch.setattr(pipeline, "load_expert", load)
    pair = {"pair": "toy_pair", "dataset": "toy", "expert_a_name": "A", "expert_b_name": "B",
            "runs": [{"seed": s, "expert_a_run": f"toy/a{s}", "expert_b_run": f"toy/b{s}"} for s in (1, 2, 3)]}
    source = {"synthetic": "fixed"}
    pipeline.run_dev(pair, workspace, "cpu", source)
    folder = workspace / "toy_pair"
    lock = pipeline.read_json(folder / "dev_lock.json")
    assert len(lock["selectors"]) == 15
    pipeline.run_dev(pair, workspace, "cpu", source)  # immutable resume
    pipeline.run_test(pair, workspace, "cpu", source)
    pipeline.run_test(pair, workspace, "cpu", source)  # no re-evaluation on resume
    refs = []
    for s in (1, 2, 3):
        f = pd.read_csv(folder / "test" / f"R1_source_b{s}_s11.csv")
        for row in f.to_dict("records"):
            row.update(seed=s, rr_global=row["rr_a"], rr_anchored_locked=row["rr_a"], rr_query_soft_locked=row["rr_b"],
                       score_information_contract=pipeline.SCORE_INFORMATION_CONTRACT, rr_dynasemble=row["rr_method"])
            refs.append(row)
    input_root = workspace / "references"
    for directory, filename in (("test_anchored", "test_locked_query_rows.csv"), ("dynasemble", "test_query_rows.csv")):
        p = input_root / "toy_pair" / directory / filename
        p.parent.mkdir(parents=True)
        frame = pd.DataFrame(refs)
        if directory == "dynasemble":
            frame = frame.drop(columns=["base_seed", "rr_method", "rr_anchored_locked", "rr_query_soft_locked"])
        else:
            frame = frame.drop(columns=["base_seed"])
        frame.to_csv(p, index=False)
    monkeypatch.setattr(pipeline, "INPUT_ROOT", input_root)
    pipeline.summarize({"pairs": [pair]}, workspace)
    summary = pd.read_csv(workspace / "summary.csv")
    assert len(summary) == len(VARIANTS)+4
    assert set(summary.method) == set(VARIANTS) | {"R0_historical", "Global", "ADC", "Query-soft"}
    package = pipeline.bundle(workspace)
    with zipfile.ZipFile(package) as archive:
        assert "summary.csv" in archive.namelist()
        assert not any("cache" in Path(name).parts for name in archive.namelist())
        assert any(name.endswith("model.pt") for name in archive.namelist())
    model_path = folder / lock["selectors"][0]["relative_dir"] / "model.pt"
    model_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Locked file changed"):
        pipeline.verify_lock(folder, lock, pair)
    with pytest.raises(RuntimeError, match="Protocol/config/source changed"):
        pipeline.check_provenance({"revision": 1}, {"revision": 2})
