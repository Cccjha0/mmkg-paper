"""Answer-aware diagnostics only; never called by an inference selector."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ALPHAS = np.arange(21, dtype=float) / 20
TOL = 1e-12


def weighted_optimum(prevalence, weight0, weight1):
    """Population weighted binary log-loss optimum, not a calibration procedure."""
    p = np.asarray(prevalence, dtype=float)
    if np.any((p < 0) | (p > 1)) or weight0 <= 0 or weight1 <= 0:
        raise ValueError('Invalid prevalence or class weights')
    return weight1 * p / (weight1 * p + weight0 * (1 - p))


def probability_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError('Nonempty aligned labels and scores required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid binary labels or scores')
    bins = np.minimum(np.floor(10 * p).astype(int), 9)
    rows = []
    for b in range(10):
        mask = bins == b
        rows.append(dict(bin=b, lower=b / 10, upper=(b + 1) / 10, n=int(mask.sum()),
                         mean_score=float(p[mask].mean()) if mask.any() else None,
                         win_frequency=float(y[mask].mean()) if mask.any() else None))
    ece = sum(r['n'] * abs(r['mean_score'] - r['win_frequency']) for r in rows if r['n']) / len(y)
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return dict(n=len(y), prevalence=float(y.mean()), mean_score=float(p.mean()),
                brier=float(np.mean((p - y) ** 2)),
                log_loss=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
                ece10=float(ece), auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None), rows


def direction_rows(grid, anchor, radius, rr_a, rr_b, applied, actual_rr):
    grid = np.asarray(grid, dtype=float)
    n = len(grid)
    anchor, radius, rr_a, rr_b, applied, actual_rr = [np.broadcast_to(np.asarray(x, dtype=float), (n,))
                                                     for x in (anchor, radius, rr_a, rr_b, applied, actual_rr)]
    if grid.shape != (n, 21) or not np.isfinite(grid).all() or np.any((grid <= 0) | (grid > 1)):
        raise ValueError('Expected finite positive RR grid with 21 columns')
    for values in (anchor, applied):
        if not np.isfinite(values).all() or np.any(np.min(abs(values[:, None] - ALPHAS), axis=1) > TOL):
            raise ValueError('Anchor and action must be on the reported grid')
    if not np.isfinite(radius).all() or np.any((radius < 0) | (radius > 1)):
        raise ValueError('Invalid radius')
    aidx = np.rint(anchor * 20).astype(int)
    idx = np.rint(applied * 20).astype(int)
    ref = grid[np.arange(n), aidx]
    if not np.allclose(grid[:, -1], rr_a, rtol=0, atol=TOL) or not np.allclose(grid[:, 0], rr_b, rtol=0, atol=TOL):
        raise ValueError('Endpoint RR mismatch')
    if np.any(abs(applied - anchor) > radius + TOL) or not np.allclose(grid[np.arange(n), idx], actual_rr, rtol=0, atol=TOL):
        raise ValueError('Actual action/RR differs from fixed-radius cache')
    delta = grid - ref[:, None]
    allowed = abs(ALPHAS[None, :] - anchor[:, None]) <= radius[:, None] + TOL
    lower = allowed & (ALPHAS[None, :] < anchor[:, None] - TOL)
    upper = allowed & (ALPHAS[None, :] > anchor[:, None] + TOL)
    available_down, available_up = lower.any(1), upper.any(1)
    best_down = np.max(np.where(lower, delta, -np.inf), axis=1)
    best_up = np.max(np.where(upper, delta, -np.inf), axis=1)
    gain_down, gain_up = best_down > TOL, best_up > TOL
    winner = np.sign(rr_a - rr_b).astype(int)
    suggested_available = np.where(winner == 1, available_up, np.where(winner == -1, available_down, False))
    suggested_gain = np.where(winner == 1, gain_up, np.where(winner == -1, gain_down, False))
    actual_dir = np.sign(applied - anchor).astype(int)
    alignment = np.select([actual_dir == 0, winner == 0, actual_dir == winner],
                          ['unchanged', 'tied_endpoints', 'winner_aligned'], default='winner_opposed')
    down_step = delta[np.arange(n), np.maximum(aidx - 1, 0)]
    up_step = delta[np.arange(n), np.minimum(aidx + 1, 20)]
    return pd.DataFrame(dict(winner=np.select([winner == 1, winner == -1], ['A', 'B'], default='tie'),
        anchor=anchor, radius=radius, reference_rr=ref, actual_rr=actual_rr, actual_delta=actual_rr - ref,
        applied=applied, alignment=alignment, available_down=available_down, available_up=available_up,
        best_down=np.where(available_down, best_down, np.nan), best_up=np.where(available_up, best_up, np.nan),
        down_step=np.where(available_down, down_step, np.nan), up_step=np.where(available_up, up_step, np.nan),
        opportunity=np.array(['neither', 'down_only', 'up_only', 'both'])[gain_down.astype(int) + 2 * gain_up.astype(int)],
        any_gain=gain_down | gain_up, suggested_available=suggested_available, suggested_gain=suggested_gain,
        suggested_step=np.where(winner == 1, np.where(available_up, up_step, np.nan),
                                np.where(winner == -1, np.where(available_down, down_step, np.nan), np.nan))))


def utility_summary(frame, population_n):
    d = frame.actual_delta.to_numpy()
    bad, good = d < -TOL, d > TOL
    return dict(n=len(d), population_n=population_n, mass=len(d) / population_n,
                utility=float(d.mean()) if len(d) else None, utility_contribution=float(d.sum() / population_n),
                harm_rate=float(bad.mean()) if len(d) else None, benefit_rate=float(good.mean()) if len(d) else None,
                conditional_loss=float(-d[bad].mean()) if bad.any() else None,
                loss_p95=float(np.quantile(-d[bad], .95)) if bad.any() else None,
                mean_loss=float(np.maximum(-d, 0).mean()) if len(d) else None,
                mean_gain=float(np.maximum(d, 0).mean()) if len(d) else None)
