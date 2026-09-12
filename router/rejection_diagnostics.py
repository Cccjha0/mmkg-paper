"""Gold-free batch rejection allocations; offline outcome arithmetic is separate."""
import numpy as np
import pandas as pd

TOL = 1e-12


def observable_units(seed, fold, direction, relation, known_entity):
    keys = pd.DataFrame(dict(seed=seed, fold=fold, direction=direction, relation=relation, known_entity=known_entity))
    if keys.isna().any().any() or not set(keys.direction).issubset({'head', 'tail'}):
        raise ValueError('Incomplete observable query keys')
    inverse, unique = pd.factorize(pd.MultiIndex.from_frame(keys), sort=True)
    units = unique.to_frame(index=False)
    # pandas.factorize may discard MultiIndex level names.
    units.columns = keys.columns
    units['repetitions'] = np.bincount(inverse, minlength=len(units))
    return units, inverse


def stratum_ids(columns):
    return pd.factorize(pd.MultiIndex.from_frame(pd.DataFrame(columns)), sort=True)[0]


class MatchedGate:
    """Quotas depend only on observable keys and original gold-free keep flags."""
    def __init__(self, strata, full_keep):
        strata, keep = np.asarray(strata), np.asarray(full_keep, dtype=bool)
        if strata.ndim != 1 or strata.shape != keep.shape:
            raise ValueError('Unaligned gate inputs')
        self.n = len(keep)
        order = np.argsort(strata, kind='stable')
        starts = np.r_[0, np.flatnonzero(np.diff(strata[order]))+1] if self.n else np.array([], dtype=int)
        self.groups = [(idx, int(keep[idx].sum())) for idx in np.split(order, starts[1:])] if self.n else []

    def probabilities(self):
        p = np.zeros(self.n)
        for idx, k in self.groups:
            p[idx] = k/len(idx)
        return p

    def ordered(self, priority, tie_key):
        priority, tie_key = np.asarray(priority), np.asarray(tie_key)
        if priority.shape != (self.n,) or tie_key.shape != priority.shape or not np.isfinite(priority).all() or not np.isfinite(tie_key).all():
            raise ValueError('Finite aligned priorities required')
        keep = np.zeros(self.n, dtype=bool)
        for idx, k in self.groups:
            chosen = idx[np.lexsort((tie_key[idx], -priority[idx]))[:k]]
            keep[chosen] = True
        return keep

    def draw(self, rng):
        keep = np.zeros(self.n, dtype=bool)
        for idx, k in self.groups:
            if k == len(idx):
                keep[idx] = True
            elif k:
                keep[rng.choice(idx, k, replace=False)] = True
        return keep

    def random_spread(self, unit_outcomes, observation_count, seed, replicates=2000):
        """Outcome sums per query unit; expectation is exact, draws show spread."""
        values = np.asarray(unit_outcomes, dtype=float)
        if values.ndim != 2 or len(values) != self.n or not np.isfinite(values).all() or observation_count <= 0:
            raise ValueError('Invalid outcome sums')
        constant = np.zeros(values.shape[1])
        partial = []
        for idx, k in self.groups:
            if k == len(idx):
                constant += values[idx].sum(axis=0)
            elif k:
                partial.append((idx, k, values[idx].sum(axis=0)))
        rng = np.random.default_rng(seed)
        draws = np.empty((replicates, values.shape[1]))
        for j in range(replicates):
            total = constant.copy()
            for idx, k, all_sum in partial:
                if k <= len(idx)//2:
                    total += values[rng.choice(idx, k, replace=False)].sum(axis=0)
                else:
                    total += all_sum-values[rng.choice(idx, len(idx)-k, replace=False)].sum(axis=0)
            draws[j] = total/observation_count
        return dict(expected=self.probabilities() @ values/observation_count,
            low=np.quantile(draws, .025, axis=0), high=np.quantile(draws, .975, axis=0),
            partial_units=sum(len(idx) for idx, _, _ in partial),
            partial_strata=len(partial), total_strata=len(self.groups), replicates=replicates)


def paired_intervals(differences, clusters, seed, replicates=2000):
    d, clusters = np.asarray(differences, dtype=float), np.asarray(clusters, dtype=int)
    if d.ndim != 2 or len(d) != len(clusters) or not np.isfinite(d).all():
        raise ValueError('Invalid paired differences')
    counts = np.bincount(clusters)
    if not len(counts) or not np.all(counts == 6):
        raise ValueError('Each original triple must retain all six observations')
    means = np.column_stack([np.bincount(clusters, weights=d[:, j])/counts for j in range(d.shape[1])])
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, d.shape[1]))
    for start in range(0, replicates, 32):
        draws = rng.integers(len(counts), size=(min(32, replicates-start), len(counts)))
        result[start:start+len(draws)] = means[draws].mean(axis=1)
    return d.mean(axis=0), np.quantile(result, .025, axis=0), np.quantile(result, .975, axis=0)
