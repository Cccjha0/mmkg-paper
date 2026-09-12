"""Algebraic diagnostics only; never a new production selector or feature input."""
import numpy as np

from router.constants import QUERY_GEOMETRY_FIELDS

FIELDS13 = tuple(QUERY_GEOMETRY_FIELDS)
FIELDS9 = FIELDS13[:9]


def expansion_matrix():
    """x13 = D @ x9 in exact arithmetic, before independent imputation."""
    d = np.zeros((13, 9))
    d[:9] = np.eye(9)
    for j in range(4):
        d[9+j, 1+j] = 1
        d[9+j, 5+j] = -1
    return d


def collapsed_raw_coefficients(model):
    """Return affine coefficients on COMPLETE raw 9D inputs for a fitted 13D model."""
    scaler, classifier = model[-2], model[-1]
    w = classifier.coef_[0] / scaler.scale_
    return expansion_matrix().T @ w, float(classifier.intercept_[0] - w @ scaler.mean_)


def augmented_standardized_map(scaler9, scaler13):
    """[z13,1] = M @ [z9,1]; includes liblinear's penalized unit intercept."""
    d = expansion_matrix()
    m = np.zeros((14, 10))
    m[:13, :9] = d * scaler9.scale_[None, :] / scaler13.scale_[:, None]
    m[:13, 9] = (d @ scaler9.mean_ - scaler13.mean_) / scaler13.scale_
    m[13, 9] = 1
    return m


def dependency_diagnostics(matrix13, model13):
    x = np.asarray(matrix13, dtype=float)
    complete = np.isfinite(x).all(axis=1)
    finite = x[complete]
    d = expansion_matrix()
    residual = finite - finite[:, :9] @ d.T
    w, b = collapsed_raw_coefficients(model13)
    errors = model13.decision_function(finite) - (finite[:, :9] @ w + b) if len(finite) else np.array([])
    imputed = model13[0].transform(np.where(np.isfinite(x), x, np.nan))
    imputed_error = imputed - imputed[:, :9] @ d.T
    return dict(rows=len(x), incomplete_rows=int((~complete).sum()),
                nonfinite_values=int((~np.isfinite(x)).sum()),
                max_difference_residual=float(np.max(np.abs(residual))) if len(finite) else None,
                max_imputed_difference_residual=float(np.max(np.abs(imputed_error))),
                max_collapsed_decision_error=float(np.max(np.abs(errors))) if len(errors) else None)
