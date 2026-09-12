"""Guarded application of existing ADC locks; no gold or filter inputs."""
import numpy as np


def guarded_outputs(model, matrix):
    """A valid fitted median/scaler/logistic pipeline sees only finite rows.

    Rejected rows have neutral diagnostic placeholders g=0, p=.5; these are not
    predictions. Malformed input shapes or damaged fitted state remain errors.
    """
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != model.n_features_in_:
        raise ValueError('Feature matrix does not match the fitted selector')
    if len(model.steps) != 3:
        raise ValueError('Expected the locked imputer/scaler/logistic pipeline')
    for value in (model[0].statistics_, model[1].mean_, model[1].scale_, model[2].coef_, model[2].intercept_):
        if not np.isfinite(value).all():
            raise ValueError('Invalid fitted selector state')
    if not (model[1].scale_ > 0).all() or not np.array_equal(model[2].classes_, [0, 1]):
        raise ValueError('Invalid fitted selector scale or classes')
    n = len(x)
    decision, probability = np.zeros(n), np.full(n, .5)
    reason = np.full(n, '', dtype=object)
    reason[~np.isfinite(x).all(axis=1)] = 'nonfinite_feature'
    rows = np.flatnonzero(reason == '')
    if len(rows):
        # A finite raw value can still overflow during centering/scaling.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            transformed = model[:-1].transform(x[rows])
        finite = np.isfinite(transformed).all(axis=1)
        reason[rows[~finite]] = 'nonfinite_preprocessing'
        rows, transformed = rows[finite], transformed[finite]
        if len(rows):
            with np.errstate(over='ignore', invalid='ignore'):
                g = np.asarray(model[-1].decision_function(transformed), dtype=float)
            finite = np.isfinite(g)
            reason[rows[~finite]] = 'nonfinite_logit'
            rows, transformed, g = rows[finite], transformed[finite], g[finite]
            if len(rows):
                with np.errstate(over='ignore', invalid='ignore'):
                    p = np.asarray(model[-1].predict_proba(transformed)[:, 1], dtype=float)
                finite = np.isfinite(p) & (p >= 0) & (p <= 1)
                reason[rows[~finite]] = 'invalid_probability'
                decision[rows[finite]], probability[rows[finite]] = g[finite], p[finite]
    invalid = reason != ''
    return dict(decision=decision, probability_a=probability, invalid=invalid,
                invalid_reason=reason, predicted=~invalid)


def apply_locked_features(model, matrix, *, alpha0, beta, threshold, alphas):
    """Apply the frozen map after guarded prediction, preserving original ties."""
    grid = tuple(float(v) for v in alphas)
    if not grid or not all(np.isfinite(grid)) or min(grid)<0 or max(grid)>1:
        raise ValueError('Invalid action grid')
    if alpha0 not in grid or not 0 <= beta <= 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid locked anchor/radius/threshold')
    out = guarded_outputs(model, matrix)
    g, p, invalid = out['decision'], out['probability_a'], out['invalid']
    confidence = np.abs(2*p-1)
    fallback = invalid | (confidence < threshold)
    continuous = np.full(len(g), float(alpha0))
    active = ~fallback
    continuous[active] = np.clip(alpha0 + beta*np.tanh(g[active]), 0, 1)
    soft = np.where(invalid, alpha0, p)
    def project(values):
        return np.asarray([min(grid, key=lambda a: (abs(a-v), abs(a-alpha0), a)) for v in values])
    out.update(confidence=confidence, fallback=fallback, continuous=continuous, applied=project(continuous),
               query_soft_continuous=soft, query_soft_applied=project(soft))
    return out
