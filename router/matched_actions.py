"""Gold-free action maps for the matched ADC alternatives experiment.

The same fitted preference model supplies every map. Ranking and DEV selection
are intentionally outside this module; inference never accepts labels or RR.
"""
from dataclasses import dataclass

import numpy as np

ALPHAS = np.arange(21, dtype=float) / 20
FAMILIES = ('ADC+Global', 'Shrink', 'Shrink-gate', 'Linear-p', 'Clip-g', 'Linear-g')


@dataclass(frozen=True)
class Action:
    family: str
    strength: float
    tau: float


def configurations(family: str) -> list[Action]:
    if family not in FAMILIES:
        raise ValueError(f'Unknown family: {family}')
    if family == 'Shrink':
        active = [Action(family, i / 40, 0.) for i in range(1, 41)]
    else:
        denominator = 10 if family == 'Shrink-gate' else 20
        active = [Action(family, i / denominator, tau)
                  for i in range(1, 11) for tau in (0., .1, .2, .3)]
    return [Action(family, 0., 0.), *active]


def grid_indices(values, anchor):
    values = np.asarray(values, dtype=float)
    hi = np.searchsorted(ALPHAS, values).clip(0, 20)
    lo = (hi - 1).clip(0)
    dl, dh = np.abs(ALPHAS[lo] - values), np.abs(ALPHAS[hi] - values)
    take_hi = (dh < dl) | ((dh == dl) & (np.abs(ALPHAS[hi] - anchor) < np.abs(ALPHAS[lo] - anchor)))
    return np.where(take_hi, hi, lo)


def weights(decision, probability, nonfinite, anchor: float, action: Action):
    """Return applied weights and fallback flags, without any answer information."""
    if action.family not in FAMILIES or not 0 <= action.strength <= 1 or not 0 <= action.tau <= 1:
        raise ValueError('Invalid action')
    if not np.any(ALPHAS == anchor):
        raise ValueError('Anchor must lie on the ranking grid')
    g, p, invalid = np.asarray(decision), np.asarray(probability), np.asarray(nonfinite, dtype=bool)
    if g.ndim != 1 or g.shape != p.shape or p.shape != invalid.shape:
        raise ValueError('Preference arrays must have identical one-dimensional shapes')
    if action.strength == 0:
        return np.full(len(g), anchor), np.ones(len(g), dtype=bool)
    invalid = invalid | ~np.isfinite(g) | ~np.isfinite(p)
    g, p = np.where(invalid, 0., g), np.where(invalid, .5, p)
    if np.any((p < 0) | (p > 1)):
        raise ValueError('Probability outside [0, 1]')
    if action.family in ('Shrink', 'Shrink-gate'):
        proposal = anchor + action.strength * (p - anchor)
    else:
        signal = {'ADC+Global': lambda: np.tanh(g), 'Linear-p': lambda: 2 * p - 1,
                  'Clip-g': lambda: np.clip(g, -1, 1), 'Linear-g': lambda: g}[action.family]()
        proposal = anchor + action.strength * signal
    fallback = invalid | (np.abs(2 * p - 1) < action.tau)
    continuous = np.where(fallback, anchor, np.clip(proposal, 0, 1))
    return ALPHAS[grid_indices(continuous, anchor)], fallback


def select_candidate(candidates):
    """Select by DEV MRR, preferring explicit Global in a numerical tie."""
    if not candidates or not all(np.isfinite(r['mrr']) for r in candidates):
        raise ValueError('Selection needs finite DEV scores')
    best = max(r['mrr'] for r in candidates)
    tied = [r for r in candidates if best - r['mrr'] <= 1e-12]
    return min(tied, key=lambda r: (r['strength'] != 0, r['strength'], -r['tau']))
