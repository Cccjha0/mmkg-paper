import itertools
import numpy as np
import pandas as pd
import pytest

from router.rejection_diagnostics import MatchedGate, observable_units, paired_intervals, stratum_ids
from router.matched_actions import weights, Action


def test_whole_query_and_row_order_invariance():
    keys = dict(seed=[1]*4, fold=[0]*4, direction=['tail']*4, relation=[3]*4, known_entity=[8, 8, 9, 10])
    units, inverse = observable_units(**keys)
    assert list(units.columns) == ['seed','fold','direction','relation','known_entity','repetitions']
    assert list(units.repetitions) == [2, 1, 1]
    gate = MatchedGate([0, 0, 0], [True, False, False])
    kept = gate.ordered([3, 1, 2], [0, 1, 2])
    assert kept[inverse][0] == kept[inverse][1]
    permutation = np.array([2, 1, 3, 0])
    again, remap = observable_units(**{k: np.asarray(v)[permutation] for k, v in keys.items()})
    pd.testing.assert_frame_equal(units, again)
    assert np.array_equal(kept[remap], kept[inverse][permutation])


def test_exact_repetition_weighted_intervention_in_every_draw():
    repetitions = np.array([1, 1, 1, 2, 2, 3, 3])
    full = np.array([1, 0, 0, 1, 0, 1, 0], dtype=bool)
    gate = MatchedGate(repetitions, full)
    for i in range(20):
        draw = gate.draw(np.random.default_rng(i))
        assert repetitions @ draw == repetitions @ full
    for priority in (np.arange(7), -np.arange(7)):
        assert repetitions @ gate.ordered(priority, np.arange(7)) == repetitions @ full


def test_shape_direction_histogram_and_mean_absolute_movement_match():
    direction = np.array(['head', 'head', 'tail', 'tail']*2)
    move = np.array([-2, -2, 2, 2, -4, -4, 4, 4])
    full = np.array([1, 0]*4, dtype=bool)
    strata = stratum_ids(dict(direction=direction, move=move))
    gate = MatchedGate(strata, full)
    for i in range(10):
        keep = gate.draw(np.random.default_rng(i))
        assert np.array_equal(np.bincount(strata, weights=keep), np.bincount(strata, weights=full))
        assert np.abs(move) @ keep == np.abs(move) @ full


def test_random_expectation_equals_enumerated_rejection_gates():
    gate = MatchedGate([0]*4, [True, True, False, False])
    outcomes = np.array([[.5, 0], [-.2, .2], [.1, 0], [-.3, .3]])
    possible = [outcomes[list(ids)].sum(axis=0)/4 for ids in itertools.combinations(range(4), 2)]
    expected = gate.probabilities() @ outcomes/4
    assert expected == pytest.approx(np.mean(possible, axis=0))
    assert gate.random_spread(outcomes, 4, 7, 50)['expected'] == pytest.approx(expected)


@pytest.mark.parametrize('keep', [[], [False]*4, [True]*4])
def test_zero_and_full_quotas_are_deterministic(keep):
    gate = MatchedGate(np.zeros(len(keep), dtype=int), keep)
    assert np.array_equal(gate.draw(np.random.default_rng(7)), keep)
    assert np.array_equal(gate.ordered(np.zeros(len(keep)), np.arange(len(keep))), keep)
    assert np.array_equal(gate.probabilities(), keep)


def test_tau_zero_is_exact_no_fallback_including_clipped_and_projected_zeros():
    g = np.array([-2., -.001, 0., 2.])
    p = 1/(1+np.exp(-g))
    a, f = weights(g, p, [False]*4, 1., Action('ADC+Global', .5, 0.))
    assert np.array_equal(a, [.5, 1., 1., 1.]) and not f.any()
    assert np.count_nonzero(a != 1) == 1


def test_paired_bootstrap_preserves_shared_six_row_effects():
    clusters = np.repeat(np.arange(4), 6)
    d = np.column_stack([np.repeat([1., 2., 3., 4.], 6), np.zeros(24)])
    mean, lo, hi = paired_intervals(d, clusters, 71, 100)
    rng = np.random.default_rng(71)
    reference = np.array([1., 2., 3., 4.])[rng.integers(4, size=(100, 4))].mean(axis=1)
    assert mean == pytest.approx([2.5, 0])
    assert lo == pytest.approx([np.quantile(reference, .025), 0])
    assert hi == pytest.approx([np.quantile(reference, .975), 0])
    with pytest.raises(ValueError, match='six observations'):
        paired_intervals(d[:-1], clusters[:-1], 71, 10)
