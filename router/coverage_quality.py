"""Gold-free nested allocations, with separate offline outcome accounting."""
import numpy as np

COUNTS = ('accepted_n', 'active_n', 'harm_n', 'delta_sum', 'loss_sum', 'movement_sum')
METRICS = ('accepted', 'intervention', 'harm_all', 'harm_accepted', 'harm_active', 'utility', 'mean_loss', 'mean_movement')
DECILES = np.arange(11, dtype=int)


class NestedOrder:
    """Integer per-stratum decile quotas; scores/keys contain no outcomes."""
    def __init__(self, strata, confidence, tie_key):
        strata, confidence, tie_key = map(np.asarray, (strata, confidence, tie_key))
        if not (strata.ndim == 1 and strata.shape == confidence.shape == tie_key.shape):
            raise ValueError('Unaligned order inputs')
        if not np.isfinite(confidence).all() or not np.isfinite(tie_key).all():
            raise ValueError('Nonfinite ordering inputs')
        order = np.argsort(strata, kind='stable')
        starts = np.r_[0, np.flatnonzero(np.diff(strata[order]))+1]
        self.groups = np.split(order, starts[1:]) if len(order) else []
        self.n = len(order)
        self.orders = [idx[np.lexsort((tie_key[idx], -confidence[idx]))] for idx in self.groups]

    def mask(self, decile):
        if decile not in DECILES:
            raise ValueError('Only predeclared deciles are supported')
        keep = np.zeros(self.n, dtype=bool)
        for order in self.orders:
            keep[order[:len(order)*decile//10]] = True
        return keep

    def totals(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or len(values) != self.n or not np.isfinite(values).all():
            raise ValueError('Invalid unit outcomes')
        result = np.zeros((11, values.shape[1]))
        expected = np.zeros_like(result)
        for order in self.orders:
            quotas = len(order)*DECILES//10
            prefix = np.vstack([np.zeros(values.shape[1]), np.cumsum(values[order], axis=0)])
            result += prefix[quotas]
            expected += (quotas/len(order))[:, None]*values[order].sum(axis=0)
        return result, expected

    def random_totals(self, values, seed, replicates=512):
        """One uniform permutation per stratum/draw reused at all deciles."""
        values = np.asarray(values, dtype=float)
        self.totals(values)  # Validate before allocating arrays.
        rng = np.random.default_rng(seed)
        result = np.zeros((replicates, 11, values.shape[1]))
        for idx in self.groups:
            quotas = len(idx)*DECILES//10
            for start in range(0, replicates, 32):
                size = min(32, replicates-start)
                perm = np.argsort(rng.random((size, len(idx))), axis=1, kind='stable')
                prefix = np.concatenate([np.zeros((size, 1, values.shape[1])),
                                         np.cumsum(values[idx][perm], axis=1)], axis=1)
                result[start:start+size] += prefix[:, quotas]
        return result


def rates(totals, n):
    """Ratios recomputed on each realized allocation; zero subsets undefined."""
    x = np.asarray(totals, dtype=float)
    if x.shape[-1] != len(COUNTS) or n <= 0:
        raise ValueError('Invalid totals or full denominator')
    a, i, h, u, loss, movement = np.moveaxis(x, -1, 0)
    div = lambda num, den: np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return np.stack([a/n, i/n, h/n, div(h,a), div(h,i), u/n, loss/n, movement/n], axis=-1)


def aggregate(totals, sizes, mode):
    """Pair axis first. Micro pools counts; macro averages pair-specific ratios."""
    x, sizes = np.asarray(totals), np.asarray(sizes)
    if len(x) != len(sizes) or np.any(sizes <= 0):
        raise ValueError('Invalid pair sizes')
    if mode == 'micro':
        return rates(x.sum(axis=0), sizes.sum())
    if mode == 'macro':
        # Propagate an undefined pair metric rather than silently dropping a pair.
        return np.mean(np.stack([rates(t,n) for t,n in zip(x,sizes)]), axis=0)
    raise ValueError('Unknown aggregation')


def distribution(values):
    """Finite-workload randomization spread, not a population confidence interval."""
    x = np.asarray(values)
    out = {}
    for j, name in enumerate(METRICS):
        valid = x[:,j][np.isfinite(x[:,j])]
        lo, hi = np.quantile(valid, [.025,.975]) if len(valid) else (np.nan,np.nan)
        out.update({name:float(valid.mean()) if len(valid) else np.nan,
                    name+'_random_lo':float(lo), name+'_random_hi':float(hi),
                    name+'_valid_draws':len(valid)})
    return out
