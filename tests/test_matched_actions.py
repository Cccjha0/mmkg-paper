import inspect

import numpy as np
import pytest

from router.matched_actions import Action, ALPHAS, FAMILIES, configurations, grid_indices, weights, select_candidate
from scripts.ablate_anchored_dynamic import apply_probability_shrinkage
from scripts.crossfit_anchored_dynamic import nearest_alpha
from scripts.analyze_paper_a_conservative_radius import policy


def signals():
    g = np.array([-100., -2., -1., -.3, 0., .3, 1., 2., 100., np.nan])
    return g, 1 / (1 + np.exp(-g)), ~np.isfinite(g)


def test_all_families_have_41_configs_and_one_explicit_global():
    for family in FAMILIES:
        configs = configurations(family)
        assert len(configs) == len(set(configs)) == 41
        assert sum(c.strength == 0 for c in configs) == 1
        for anchor in (0., .55, 1.):
            a, fb = weights(*signals(), anchor, configs[0])
            np.testing.assert_array_equal(a, np.full(len(a), anchor))
            assert fb.all()


def test_zero_and_one_shrinkage_match_global_and_existing_query_soft_mapping():
    g, p, invalid = signals()
    rows = [{f'rr_alpha_{a:.2f}'.replace('.', '_'): 1 / (i + 1) for i, a in enumerate(ALPHAS)} for _ in g]
    for strength in (0., .025, .5, 1.):
        new, _ = weights(g, p, invalid, .55, Action('Shrink', strength, 0.))
        old = apply_probability_shrinkage(rows, probability_a=p, nonfinite=invalid, alpha0=.55,
                                         strength=strength, alphas=tuple(ALPHAS))
        np.testing.assert_array_equal(new, old['applied'])


def test_projection_and_ties_match_production():
    mid = (ALPHAS[:-1] + ALPHAS[1:]) / 2
    x = np.concatenate([mid, np.nextafter(mid, 0), np.nextafter(mid, 1), ALPHAS])
    for anchor in ALPHAS:
        np.testing.assert_array_equal(ALPHAS[grid_indices(x, anchor)],
                                      [nearest_alpha(float(v), tuple(ALPHAS), float(anchor)) for v in x])


def test_adc_map_reproduces_original_and_bounded_maps_respect_radius():
    g, p, invalid = signals()
    grid = np.tile(1 / np.arange(1, 22), (len(g), 1))
    for anchor in (0., .55, 1.):
        for action in configurations('ADC+Global')[1:]:
            _, expected, fb = policy(grid, g, p, invalid, anchor, action.strength, action.tau)
            actual, actual_fb = weights(g, p, invalid, anchor, action)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual_fb, fb)
            for family in ('ADC+Global', 'Linear-p', 'Clip-g'):
                a, _ = weights(g, p, invalid, anchor, Action(family, action.strength, action.tau))
                assert np.max(np.abs(a - anchor)) <= action.strength + 1e-12


def test_dev_can_choose_global_when_all_interventions_harm_or_tie():
    candidates = [dict(strength=0., tau=0., mrr=.4), dict(strength=.1, tau=.3, mrr=.39)]
    assert select_candidate(candidates)['strength'] == 0
    candidates[1]['mrr'] = .4 + 5e-13
    assert select_candidate(candidates)['strength'] == 0
    candidates[1]['mrr'] = .41
    assert select_candidate(candidates)['strength'] == .1


def test_gold_and_ranking_outcomes_cannot_change_inference_weights():
    assert set(inspect.signature(weights).parameters) == {'decision', 'probability', 'nonfinite', 'anchor', 'action'}
    for family in FAMILIES:
        action = configurations(family)[-1]
        a, _ = weights(*signals(), .55, action)
        # Changing which answer wins changes evaluation, not the observable input.
        rank_grid = np.tile(np.arange(1, 22), (len(a), 1))
        rr1 = 1 / rank_grid[np.arange(len(a)), grid_indices(a, .55)]
        rr2 = 1 / rank_grid[:, ::-1][np.arange(len(a)), grid_indices(a, .55)]
        assert np.any(rr1 != rr2)
        np.testing.assert_array_equal(weights(*signals(), .55, action)[0], a)


def test_linear_logit_has_no_claimed_local_radius():
    g, p, invalid = signals()
    alpha, _ = weights(g, p, invalid, .55, Action('Linear-g', .05, 0.))
    assert np.max(np.abs(alpha - .55)) > .05


def test_invalid_shape_and_configuration_are_rejected():
    with pytest.raises(ValueError):
        weights(np.zeros(2), np.zeros(1), np.zeros(2), .55, Action('Shrink', .5, 0.))
    with pytest.raises(ValueError):
        configurations('unknown')


def test_expert_oracle_is_not_a_score_fusion_upper_bound():
    primary = np.array([.8, 1., 0.])
    secondary = np.array([.8, 0., 1.])
    primary = (primary - primary.mean()) / primary.std()
    secondary = (secondary - secondary.mean()) / secondary.std()
    rr = lambda scores: 1 / (1 + np.sum(scores > scores[0]))
    assert max(rr(primary), rr(secondary)) == .5
    assert rr((primary + secondary) / 2) == 1.


def test_dev_reader_rejects_test_path_before_opening():
    from pathlib import Path
    from scripts.analyze_paper_a_matched_alternatives import Inputs
    reader = Inputs('DEV')
    with pytest.raises(ValueError, match='DEV phase cannot open TEST'):
        reader.frame(Path(__file__).parent / 'test_query_rows.csv')
