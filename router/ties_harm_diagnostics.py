"""Offline gold-aware diagnostics; never used as selector inputs or serving gates."""
import numpy as np

TOL = 1e-12
METRICS = ('prevalence', 'auroc', 'ap', 'ap_lift')


def tie_summary(tied, alpha, anchor, rr, reference, population=None):
    tied, alpha, anchor, rr, reference = [np.asarray(x) for x in (tied, alpha, anchor, rr, reference)]
    if not all(x.ndim == 1 and x.shape == tied.shape for x in (alpha, anchor, rr, reference)):
        raise ValueError('Expected aligned observation vectors')
    if not all(np.isfinite(x).all() for x in (alpha, anchor, rr, reference)):
        raise ValueError('Nonfinite outcomes or actions')
    mask = np.ones(len(tied), dtype=bool) if population is None else tied == population
    delta = rr[mask] - reference[mask]
    active = np.abs(alpha[mask] - anchor[mask]) > TOL
    harm, benefit = delta < -TOL, delta > TOL
    if np.any((harm | benefit) & ~active):
        raise ValueError('Unchanged weights cannot alter recorded RR')
    n = len(delta)
    return dict(total_n=len(tied), n=n, fraction=n/len(tied) if len(tied) else np.nan,
        active_n=int(active.sum()), action_rate=float(active.mean()) if n else np.nan,
        mean_abs_movement=float(np.abs(alpha[mask]-anchor[mask]).mean()) if n else np.nan,
        harm_n=int(harm.sum()), benefit_n=int(benefit.sum()),
        harm_rate=float(harm.mean()) if n else np.nan, benefit_rate=float(benefit.mean()) if n else np.nan,
        utility=float(delta.mean()) if n else np.nan,
        contribution=float(delta.sum()/len(tied)) if len(tied) else np.nan,
        mean_loss=float(np.maximum(-delta, 0).mean()) if n else np.nan,
        conditional_loss=float(-delta[harm].mean()) if harm.any() else np.nan)


def action_populations(valid, raw_alpha, applied, anchor, raw_rr, actual_rr):
    valid = np.asarray(valid, dtype=bool)
    raw_alpha, applied, anchor, raw_rr, actual_rr = [np.asarray(x) for x in (raw_alpha, applied, anchor, raw_rr, actual_rr)]
    if not all(x.shape == valid.shape and x.ndim == 1 for x in (raw_alpha, applied, anchor, raw_rr, actual_rr)):
        raise ValueError('Unaligned populations')
    proposed = valid & (np.abs(raw_alpha-anchor) > TOL)
    executed = np.abs(applied-anchor) > TOL
    if np.any(executed & ~proposed):
        raise ValueError('Executed action is not a valid nonzero proposal')
    if not np.allclose(raw_alpha[executed], applied[executed], atol=TOL, rtol=0) or not np.allclose(raw_rr[executed], actual_rr[executed], atol=TOL, rtol=0):
        raise ValueError('Ungated and executed actions differ on executed subset')
    return dict(all=valid, proposed=proposed, executed=executed)


class RankedBinary:
    """Pre-sort once; cluster multiplicities exactly reproduce expanded samples."""
    def __init__(self, labels, scores, clusters):
        labels, scores, clusters = np.asarray(labels), np.asarray(scores), np.asarray(clusters)
        if not (labels.ndim == 1 and labels.shape == scores.shape == clusters.shape):
            raise ValueError('Unaligned ranking inputs')
        if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
            raise ValueError('Binary labels and finite scores required')
        if not np.issubdtype(clusters.dtype, np.integer) or np.any(clusters < 0):
            raise ValueError('Nonnegative integer cluster indices required')
        order = np.argsort(-scores, kind='stable')
        self.labels, self.clusters = labels[order].astype(float), clusters[order]
        ordered = scores[order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1])+1] if len(labels) else np.array([], dtype=int)

    def evaluate(self, multiplicities):
        if len(self.labels) == 0:
            return np.full(4, np.nan)
        weights = np.asarray(multiplicities)[self.clusters]
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Invalid cluster multiplicity')
        total = np.add.reduceat(weights, self.starts).astype(float)
        pos = np.add.reduceat(weights*self.labels, self.starts)
        neg = total-pos
        p, n = pos.sum(), neg.sum()
        if p+n == 0:
            return np.full(4, np.nan)
        prev = p/(p+n)
        auc = float(np.sum(pos*(n-np.cumsum(neg)+neg/2))/(p*n)) if p and n else np.nan
        cumulative = np.cumsum(total)
        precision = np.divide(np.cumsum(pos), cumulative, out=np.zeros_like(pos), where=cumulative > 0)
        ap = float(np.sum(pos*precision)/p) if p else np.nan
        return np.array([prev, auc, ap, ap-prev])


def cluster_intervals(evaluators, cluster_count, replicates=2000, seed=20260915):
    if cluster_count < 1 or replicates < 2:
        raise ValueError('Positive cluster universe and at least two replicates required')
    rng = np.random.default_rng(seed)
    samples = {key: np.empty((replicates, 4)) for key in evaluators}
    for i in range(replicates):
        # Sample from the FULL triple universe before any subset restriction.
        counts = np.bincount(rng.integers(cluster_count, size=cluster_count), minlength=cluster_count)
        for key, evaluator in evaluators.items():
            samples[key][i] = evaluator.evaluate(counts)
    result = {}
    for key, evaluator in evaluators.items():
        row = {}
        point = evaluator.evaluate(np.ones(cluster_count))
        for j, metric in enumerate(METRICS):
            valid = samples[key][:, j][np.isfinite(samples[key][:, j])]
            low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
            row.update({metric: float(point[j]), metric+'_lo': float(low), metric+'_hi': float(high), metric+'_valid_replicates': len(valid)})
        result[key] = row
    return result
