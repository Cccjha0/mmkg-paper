"""Fixed-grid actions for B09 diagnostics; gold and evaluator facts are not inputs."""
import numpy as np

RESOLUTIONS = {'grid_005': .05, 'grid_001': .01, 'continuous': None}
METHODS = ('Global',) + tuple(f'{p}_{r}' for p in ('ADC', 'Query-soft') for r in RESOLUTIONS)


def project(values, anchors, step):
    values, anchors = np.broadcast_arrays(np.asarray(values, dtype=float), np.asarray(anchors, dtype=float))
    if not np.isfinite(values).all() or not np.isfinite(anchors).all():
        raise ValueError('Finite continuous weights and anchors required')
    if np.any((values < 0) | (values > 1) | (anchors < 0) | (anchors > 1)):
        raise ValueError('Weights outside [0,1]')
    if step is None:
        return values.copy()
    if step not in (.05, .01):
        raise ValueError('Resolution was not fixed by the protocol')
    grid = np.arange(round(1 / step) + 1, dtype=float) / round(1 / step)
    hi = np.searchsorted(grid, values).clip(0, len(grid) - 1)
    lo = (hi - 1).clip(0)
    dl, dh = np.abs(grid[lo] - values), np.abs(grid[hi] - values)
    take_hi = (dh < dl) | ((dh == dl) & (np.abs(grid[hi] - anchors) < np.abs(grid[lo] - anchors)))
    return grid[np.where(take_hi, hi, lo)]


def actions(g, p, nonfinite, anchors, beta, tau):
    g, p, anchors, beta, tau = np.broadcast_arrays(g, p, anchors, beta, tau)
    nonfinite = np.asarray(nonfinite, dtype=bool) | ~np.isfinite(g) | ~np.isfinite(p)
    fallback = nonfinite | (np.abs(2 * p - 1) < tau)
    continuous = {
        'ADC': np.where(fallback, anchors, np.clip(anchors + beta * np.tanh(g), 0, 1)),
        'Query-soft': np.where(nonfinite, anchors, np.clip(p, 0, 1)),
    }
    result = {'Global': np.asarray(anchors, dtype=float).copy()}
    for policy, values in continuous.items():
        for resolution, step in RESOLUTIONS.items():
            result[f'{policy}_{resolution}'] = project(values, anchors, step)
    return result, fallback
