import numpy as np
import pytest

from router.winner_signal_diagnostics import direction_rows, probability_metrics, utility_summary, weighted_optimum


def example():
    grid = np.full((4, 21), .25)
    grid[:, 0], grid[:, -1] = .2, .5  # A wins every standalone comparison.
    grid[:, 10] = .5
    grid[0, 9], grid[0, 11] = 1., 1/3  # A wins; only down helps; up harms.
    grid[1, 9], grid[1, 11] = 1., 1.  # Both directions help.
    grid[2, 9], grid[2, 11] = 1/3, .5  # Neither helps; a tie is not a gain.
    grid[3, 9], grid[3, 11] = .5, 1.  # Only up helps.
    return grid


def test_endpoint_winner_does_not_imply_anchor_direction_or_action_gain():
    grid = example()
    frame = direction_rows(grid, .5, .05, grid[:, -1], grid[:, 0], .55, grid[:, 11])
    assert frame.opportunity.tolist() == ['down_only', 'both', 'neither', 'up_only']
    assert frame.suggested_available.all() and frame.alignment.eq('winner_aligned').all()
    assert frame.suggested_gain.tolist() == [False, True, False, True]
    assert frame.actual_delta.iloc[0] < 0
    stats = utility_summary(frame, len(frame))
    assert stats['harm_rate'] == .25
    assert stats['conditional_loss'] == pytest.approx(1/6)
    assert stats['utility'] == pytest.approx(stats['mean_gain'] - stats['mean_loss'])


def test_blocked_endpoint_suggestion_is_distinct_from_feasible_discordance():
    grid = np.full((1, 21), .25)
    grid[0, 0], grid[0, -1], grid[0, -2] = .2, .5, 1.
    frame = direction_rows(grid, 1., .05, [.5], [.2], 1., [.5])
    assert frame.any_gain.iloc[0] and frame.opportunity.iloc[0] == 'down_only'
    assert not frame.suggested_available.iloc[0] and not frame.available_up.iloc[0]
    assert np.isnan(frame.best_up.iloc[0]) and frame.alignment.iloc[0] == 'unchanged'


def test_radius_and_step_do_not_use_distant_best_point():
    grid = example()[:1]
    grid[0, 12] = 1.
    narrow = direction_rows(grid, .5, .05, [.5], [.2], .55, grid[:, 11])
    wide = direction_rows(grid, .5, .1, [.5], [.2], .55, grid[:, 11])
    assert not narrow.suggested_gain.iloc[0] and wide.suggested_gain.iloc[0]
    assert wide.suggested_step.iloc[0] < 0  # A profitable magnitude does not save the chosen one.


def test_tied_endpoints_are_not_forced_into_primary_class():
    grid = example()[:1]
    grid[:, 0] = grid[:, -1]
    frame = direction_rows(grid, .5, .05, [.5], [.5], .55, grid[:, 11])
    assert frame.winner.iloc[0] == 'tie' and frame.alignment.iloc[0] == 'tied_endpoints'
    assert not frame.suggested_available.iloc[0]


@pytest.mark.parametrize('bad', ['endpoint', 'off_grid', 'rr', 'outside_radius'])
def test_inconsistent_cache_or_policy_is_rejected(bad):
    grid = example()[:1]
    a, action, actual = [.5], .55, grid[:, 11]
    if bad == 'endpoint': a = [.6]
    if bad == 'off_grid': action = .551
    if bad == 'rr': actual = [.1]
    if bad == 'outside_radius': action, actual = .6, grid[:, 12]
    with pytest.raises(ValueError):
        direction_rows(grid, .5, .05, a, [.2], action, actual)


def test_balanced_weighted_loss_optimum_is_not_natural_prevalence():
    p = .8
    assert weighted_optimum(p, 1 / (2 * (1 - p)), 1 / (2 * p)) == pytest.approx(.5)
    assert weighted_optimum(p, 1., 1.) == p
    scores = np.linspace(.01, .99, 99)
    risk = -(1 / (2 * p)) * p * np.log(scores) - (1 / (2 * (1 - p))) * (1 - p) * np.log1p(-scores)
    assert scores[np.argmin(risk)] == .5


def test_probability_metrics_use_natural_counts_and_keep_empty_bins():
    stats, bins = probability_metrics([0, 0, 1, 1], [.1, .1, .9, 1.])
    assert stats['brier'] == pytest.approx(.0075)
    assert stats['ece10'] == pytest.approx(.075)
    assert stats['prevalence'] == .5 and stats['auc'] == 1.
    assert len(bins) == 10 and sum(b['n'] for b in bins) == 4
    assert bins[1]['n'] == bins[9]['n'] == 2 and bins[0]['mean_score'] is None


def test_empty_alignment_group_has_undefined_conditional_loss_and_zero_contribution():
    frame = direction_rows(example(), .5, .05, [.5]*4, [.2]*4, .5, [.5]*4)
    stats = utility_summary(frame.iloc[:0], len(frame))
    assert stats['n'] == 0 and stats['conditional_loss'] is None and stats['utility_contribution'] == 0
