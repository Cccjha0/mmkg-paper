"""B09 mathematical, information-boundary and staged scoring integration checks."""
import json
import uuid
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit

from router.grid_sensitivity import METHODS, RESOLUTIONS, actions, project
from scripts.crossfit_anchored_dynamic import nearest_alpha
from scripts import paper_a_grid_sensitivity as study
from scripts import eval_heterogeneous_complementarity as evaluator


@pytest.mark.parametrize('step', [.05, .01])
def test_projection_matches_released_lexicographic_rule(step):
    grid = np.arange(round(1/step)+1)/round(1/step)
    mid = (grid[:-1]+grid[1:])/2
    values = np.r_[grid, mid, np.nextafter(mid, 0), np.nextafter(mid, 1), np.linspace(0, 1, 777)]
    for anchor in np.arange(21)/20:
        expected = [nearest_alpha(float(x), tuple(grid), float(anchor)) for x in values]
        actual = project(values, anchor, step)
        np.testing.assert_array_equal(actual, expected)
        assert np.max(np.abs(actual-values)) <= step/2+1e-15


def test_grid_deadzone_is_an_action_change_not_a_rank_cache_detail():
    g = np.array([np.arctanh(.02), -np.arctanh(.02)])
    weights, _ = actions(g, expit(g), False, .5, .5, 0)
    np.testing.assert_array_equal(weights['ADC_grid_005'], [.5, .5])
    np.testing.assert_allclose(weights['ADC_grid_001'], [.51, .49], atol=1e-15)
    np.testing.assert_allclose(weights['ADC_continuous'], [.51, .49], atol=1e-15)


def test_fallback_preserves_original_anchors_on_every_resolution():
    anchors = np.array([0, .55, 1, .55])
    weights, fallback = actions([0, 0, np.nan, 2], [.5, .5, np.nan, .9], [False, False, False, True], anchors, .5, .3)
    assert fallback.all()
    for resolution in RESOLUTIONS:
        np.testing.assert_array_equal(weights['ADC_' + resolution], anchors)
        np.testing.assert_array_equal(weights['Query-soft_' + resolution][2:], anchors[2:])
    np.testing.assert_array_equal(weights['Global'], anchors)


@pytest.mark.parametrize('step', [.05, .01, None])
def test_clipping_and_endpoint_behavior_are_retained(step):
    weights, fallback = actions([5., -5.], expit([5., -5.]), False, [1., 0.], .5, .1)
    name = next(name for name, value in RESOLUTIONS.items() if value == step)
    assert not fallback.any()
    np.testing.assert_array_equal(weights['ADC_' + name], [1., 0.])


@pytest.mark.parametrize('values,step', [([np.nan], .05), ([-.1], None), ([1.1], .01), ([.5], .02)])
def test_invalid_or_unplanned_projection_is_rejected(values, step):
    with pytest.raises(ValueError):
        project(values, .5, step)


def test_rank_cannot_be_obtained_by_interpolating_coarse_reciprocal_ranks():
    # Candidate-gold margin crosses zero at 0.512, between old grid points.
    a = torch.tensor([[0., .488]], dtype=torch.float32)
    b = torch.tensor([[0., -.512]], dtype=torch.float32)
    gold = torch.zeros(1)
    rr = [1/evaluator.mixed_ranks(a, b, gold, gold, x).item() for x in [.50, .51, .55]]
    assert rr == [1., 1., .5]
    assert rr[1] != pytest.approx(rr[0] + .2*(rr[2]-rr[0]))


@pytest.mark.parametrize('direction', ['head', 'tail'])
def test_new_rank_path_retains_gold_and_cannot_change_precomputed_weights(direction):
    raw_a = torch.tensor([[1., 12., 5., 4., 2., 0.]]).repeat(2, 1)
    raw_b = torch.tensor([[3., 1., 9., 2., 4., 0.]]).repeat(2, 1)
    q = torch.tensor([[0, 0, 1], [0, 0, 2]] if direction == 'tail' else [[1, 0, 0], [2, 0, 0]])
    weights, _ = actions([.4, .4], expit([.4, .4]), False, .55, .45, .1)
    before = {name: value.copy() for name, value in weights.items()}
    copies = [raw_a.clone(), raw_b.clone()]
    for facts in ({}, {(0, 0): torch.tensor([1, 2])}):
        actual = study.rank_actions(raw_a, raw_b, raw_a[:, [1, 2]].diagonal(), raw_b[:, [1, 2]].diagonal(), q, direction, facts, weights)
        for row, target in enumerate([1, 2]):
            for method, values in weights.items():
                a = raw_a[row]; b = raw_b[row]
                za = (a-a.mean())/(torch.std(a, correction=0)+1e-8)
                zb = (b-b.mean())/(torch.std(b, correction=0)+1e-8)
                alpha = torch.tensor(values[row], dtype=torch.float32)
                mix = alpha*za+(1-alpha)*zb
                reference = mix[target].clone()
                if facts:
                    mix[[x for x in [1, 2] if x != target]] = -torch.inf
                assert actual[method][row] == int((mix > reference).sum())+1
        if facts:
            np.testing.assert_array_equal(actual['Primary'], [1, 1])
            np.testing.assert_array_equal(actual['Secondary'], [4, 1])
    assert torch.equal(raw_a, copies[0]) and torch.equal(raw_b, copies[1])
    for method in METHODS:
        np.testing.assert_array_equal(weights[method], before[method])
        assert weights[method][0] == weights[method][1]


def test_dev_preparation_rejects_test_outcome_paths():
    study.check_phase_path('outputs/pair/dev_crossfit/rows.csv', 'dev')
    with pytest.raises(ValueError, match='cannot read TEST'):
        study.check_phase_path('outputs/pair/test_anchored/rows.csv', 'dev')


def test_staged_toy_scoring_resume_and_report(monkeypatch):
    """Real loop/filter/rank/receipt/report code; only base scoring and CUDA are mocked."""
    # pytest's mode-0700 temp fixture is incompatible with this Windows sandbox ACL.
    tmp_path = study.ROOT/'tmp'/('grid_sensitivity_' + uuid.uuid4().hex)
    tmp_path.mkdir(parents=True)
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'OUT', tmp_path/'review')
    monkeypatch.setattr(study, 'PAIRS', {'toy': 'Toy'})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'synthetic CPU fixture')
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    triples = [(0, 0, 1), (0, 0, 2)]
    bundle = SimpleNamespace(name='toy', protocol_version='fixture', entity2id={}, relation2id={},
                             train_triples=[], valid_triples=triples, test_triples=triples, manifest={'hashes': {}})
    records = []
    for seed in (1, 2, 3):
        for direction in ('head', 'tail'):
            for h, r, t in triples:
                row = dict(seed=seed, direction=direction, head_id=h, relation_id=r, tail_id=t,
                           g=.4, p=float(expit(.4)), nonfinite=False, anchor=.55, beta=.45, tau=.1, fallback=False, fold=1)
                weights, _ = actions([row['g']], [row['p']], False, .55, .45, .1)
                row.update({'alpha_'+m: float(w[0]) for m, w in weights.items()})
                # Deliberately wrong replay ranks must be reported, never silently dropped.
                row.update({'historical_rank_'+m: 3 for m in study.REPLAY})
                records.append(row)
    frame = pd.DataFrame(records)
    dev_path = study.OUT/'prepared/toy_dev.csv.gz'
    study.frame_write(dev_path, frame)
    info = dict(path=dev_path.relative_to(tmp_path).as_posix(), sha256=study.sha(dev_path), rows=len(frame))
    plan = dict(dev_plans={'toy': info}, full_locks={'toy': {'expert_a_name':'A','expert_b_name':'B','dataset':'toy'}}, checkpoints={})
    for seed in (1, 2, 3):
        for name in ('A', 'B'):
            run = f'runs/{name}{seed}'
            checkpoint = tmp_path/run/'best.ckpt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'toy')
            plan['checkpoints'][run] = dict(dataset='toy', model=name, seed=seed, checkpoint_sha256=study.sha(checkpoint))
    study.json_write(study.OUT/'plan.json', plan)
    monkeypatch.setattr(study, 'verify_plan', lambda: plan)
    opened = []
    monkeypatch.setattr(study, 'test_plan', lambda pair, value: (opened.append(pair) or frame.copy(), info))
    loads = []
    def load(name, run, device):
        loads.append(name)
        return SimpleNamespace(name=name, seed=int(run.name[-1]), num_entities=4, bundle=bundle,
                               query_batch_size=2, cfg={'_dataset_manifest': {'hashes': {}}})
    monkeypatch.setattr(evaluator, 'load_expert', load)
    def scores(expert, queries, direction, facts, device, retain_unfiltered):
        assert facts == {} and retain_unfiltered
        raw = torch.tensor([1., 4., 2., 3.] if expert.name=='A' else [2., 1., 4., 3.]).repeat(len(queries), 1)
        target = queries[:, 0 if direction=='head' else 2]
        gold = raw.gather(1, target[:, None]).reshape(-1)
        return raw.clone(), gold, raw
    monkeypatch.setattr(evaluator, 'score_expert_block', scores)
    with pytest.raises(FileNotFoundError):
        study.score('test', 'cuda')
    assert not opened and not loads
    study.score('dev', 'cuda')
    assert len(loads) == 6 and not opened
    study.score('dev', 'cuda')
    assert len(loads) == 6  # Verified resume does not reload/scorer-run a completed cell.
    study.score('test', 'cuda')
    study.report(pack=True)
    review = json.loads((study.OUT/'review.json').read_text())
    assert review['status'] == 'grid_scoring_complete_replay_review_required'
    assert not review['new_grid_selection'] and not review['current_paper_results_replaced']
    summary = pd.read_csv(study.OUT/'summary.csv')
    assert len(summary) == 14 and set(summary['method']) == set(METHODS)
    assert (summary[summary['method']=='Global'].mean_loss == 0).all()
    assert len(pd.read_csv(study.OUT/'paired_effects.csv')) == 26
    assert (tmp_path/'grid_sensitivity_review_v1_return.zip').exists()
    changed = study.OUT/'cells/dev_toy_s1_head.csv.gz'
    changed.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        study.score('dev', 'cuda')
