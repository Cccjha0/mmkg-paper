import numpy as np
import pytest
from scipy.special import expit

from scripts.analyze_paper_a_conservative_radius import ALPHAS, policy
from scripts.build_paper_a_action_semantics import proposal_floor


def apply(g, anchor=1., beta=.5, tau=.3, nonfinite=None):
    g = np.asarray(g, dtype=float)
    invalid = np.zeros(len(g), dtype=bool) if nonfinite is None else np.asarray(nonfinite)
    return policy(np.zeros((len(g), len(ALPHAS))), g, expit(g), invalid, anchor, beta, tau)[1:]


def test_confidence_and_proposal_are_algebraically_coupled():
    g = np.r_[np.linspace(-40, 40, 2001), -1e-12, 0, 1e-12]
    c = np.abs(2 * expit(g) - 1)
    np.testing.assert_allclose(c, np.abs(np.tanh(g / 2)), rtol=0, atol=5e-16)
    np.testing.assert_allclose(np.abs(np.tanh(g)), 2 * c / (1 + c * c), rtol=0, atol=5e-16)


def test_threshold_selects_larger_proposals_without_rescaling_survivors():
    g = np.linspace(-3, 3, 601)
    a0, f0 = apply(g, anchor=.55, beta=.45, tau=.1)
    a1, f1 = apply(g, anchor=.55, beta=.45, tau=.3)
    assert np.all(f1[f0])
    np.testing.assert_array_equal(a0[~f1], a1[~f1])
    assert np.min(np.abs(.45 * np.tanh(g[~f1]))) >= proposal_floor(.45, .3)


def test_da_inward_grid_floor_and_outward_zero():
    c = np.linspace(.3000000001, .9999, 1001)
    magnitude = 2 * np.arctanh(c)
    inward, rejected = apply(-magnitude)
    assert not rejected.any()
    assert np.min(1 - inward) == pytest.approx(.30)
    assert proposal_floor(.5, .3) == pytest.approx(.2752293577981651)
    outward, rejected = apply(magnitude)
    assert not rejected.any()
    np.testing.assert_array_equal(outward, 1.)


def test_clipping_prevents_a_general_realized_displacement_floor():
    g = [2 * np.arctanh(.31)]
    alpha, rejected = apply(g, anchor=.95)
    assert not rejected[0]
    assert alpha[0] - .95 == pytest.approx(.05)
    assert alpha[0] - .95 < proposal_floor(.5, .3)


def test_expanded_radius_retains_anchor_map_gate_and_nonfinite_fallback():
    g = [0., .3, 2., -2., 2.]
    alpha, rejected = apply(g, anchor=.55, beta=1., tau=.2, nonfinite=[False, False, False, False, True])
    np.testing.assert_array_equal(rejected, [True, True, False, False, True])
    np.testing.assert_allclose(alpha, [.55, .55, 1., 0., .55], rtol=0, atol=1e-12)
    assert alpha[1] != expit(g[1])
