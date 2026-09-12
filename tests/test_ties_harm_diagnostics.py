import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from router.ties_harm_diagnostics import RankedBinary, action_populations, cluster_intervals, tie_summary


@pytest.mark.parametrize('seed', [7, 19, 61])
def test_weighted_tied_scores_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    y, scores = rng.integers(2, size=60), rng.integers(5, size=60)/5
    clusters = np.repeat(np.arange(10), 6)
    counts = rng.integers(4, size=10)
    prev, auc, ap, lift = RankedBinary(y, scores, clusters).evaluate(counts)
    weight = counts[clusters]
    assert auc == pytest.approx(roc_auc_score(y, scores, sample_weight=weight))
    assert ap == pytest.approx(average_precision_score(y, scores, sample_weight=weight))
    assert prev == pytest.approx(np.average(y, weights=weight))
    assert lift == pytest.approx(ap-prev)


def test_cluster_bootstrap_equals_full_expansion_before_mask():
    clusters = np.repeat(np.arange(5), 6)
    y = (np.arange(30) % 3 == 0).astype(int)
    scores = np.arange(30) % 4
    mask = np.arange(30) % 7 == 0
    evaluator = RankedBinary(y[mask], scores[mask], clusters[mask])
    got = cluster_intervals({'a': evaluator}, 5, replicates=60, seed=17)['a']
    rng = np.random.default_rng(17)
    results = []
    for _ in range(60):
        draws = rng.integers(5, size=5)
        expanded = np.concatenate([np.flatnonzero((clusters == k) & mask) for k in draws])
        if len(expanded) and len(np.unique(y[expanded])) == 2:
            results.append(roc_auc_score(y[expanded], scores[expanded]))
    assert got['auroc_valid_replicates'] == len(results)
    assert [got['auroc_lo'], got['auroc_hi']] == pytest.approx(np.quantile(results, [.025, .975]))


def test_constant_score_has_prevalence_ap_and_half_auc():
    prev, auc, ap, lift = RankedBinary([1, 0, 0, 0], [0.5]*4, np.arange(4)).evaluate(np.ones(4))
    assert (prev, auc, ap, lift) == (.25, .5, .25, 0.)


@pytest.mark.parametrize('y', [[], [0, 0], [1, 1]])
def test_empty_and_one_class_metrics(y):
    r = RankedBinary(y, np.ones(len(y)), np.arange(len(y))).evaluate(np.ones(max(1, len(y))))
    assert np.isnan(r[1])
    assert np.isnan(r[2]) if not any(y) else r[2] == 1


def test_projected_and_gated_population_denominators():
    p = action_populations([True]*4, [0.5, 0.6, 0.7, 0.5], [0.5, 0.5, 0.7, 0.5], [0.5]*4,
                           [.5, 1., .25, .5], [.5, .5, .25, .5])
    assert [int(p[k].sum()) for k in ('all', 'proposed', 'executed')] == [4, 2, 1]
    with pytest.raises(ValueError, match='not a valid'):
        action_populations([True], [.5], [.6], [.5], [.5], [.5])


def test_endpoint_ties_need_not_preserve_interior_rank():
    # Gold first. At each endpoint exactly one competitor beats it; at the
    # midpoint both do, despite equal standalone gold RR (1/2).
    a, b = np.array([0., 2., -1.]), np.array([0., -1., 2.])
    rr = lambda s: 1/(1+np.sum(s[1:] >= s[0]))
    assert rr(a) == rr(b) == .5 and rr((a+b)/2) == pytest.approx(1/3)
    r = tie_summary([True, False], [.5, 0.], [0., 0.], [1/3, .5], [.5, .5], True)
    assert r['n'] == r['active_n'] == r['harm_n'] == 1
    assert r['utility'] == pytest.approx(-1/6)
    assert r['contribution'] == pytest.approx(-1/12)


def test_tie_partition_includes_unchanged_and_conserves_utility():
    args = ([True, True, False], [.5, .6, .6], [.5]*3, [.5, .4, .7], [.5]*3)
    t, u, all_rows = [tie_summary(*args, population=p) for p in (True, False, None)]
    assert t['n'] == 2 and t['action_rate'] == .5 and t['harm_rate'] == .5
    assert t['contribution']+u['contribution'] == pytest.approx(all_rows['utility'])
    with pytest.raises(ValueError, match='Unchanged weights'):
        tie_summary([True], [.5], [.5], [.4], [.5])
