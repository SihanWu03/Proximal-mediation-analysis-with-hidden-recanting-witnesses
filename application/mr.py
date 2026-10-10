"""Miles et al. (2020), Section 4.2: MR benchmark with factual M1 observed.

Mediator distributions have fitted linear normal means and pooled variances.
The control-arm outcome working model is linear. All nuisance parameters are
estimated from training data, including the analytic working-model integrals.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def fit_normal(features, outcome):
    model = LinearRegression().fit(features, outcome)
    residual = outcome - model.predict(features)
    variance = float(np.mean(residual ** 2))
    if not np.isfinite(variance) or variance <= 0:
        raise FloatingPointError("Invalid fitted conditional normal variance")
    return model, variance


def normal_log_ratio(value, numerator_mean, denominator_mean, variance):
    return ((value - denominator_mean) ** 2 - (value - numerator_mean) ** 2) / (2 * variance)


def evaluate_fold(train, evaluation, propensity=0.5):
    """Return held-out nested-mean scores; callers subtract the common baseline."""
    x, a, c, m, y = (np.asarray(train[k]) for k in ("x", "a", "m1", "m2", "y"))
    xe, ae, ce, me, ye = (np.asarray(evaluation[k]) for k in ("x", "a", "m1", "m2", "y"))
    zero, one = np.zeros_like(ae), np.ones_like(ae)
    c_model, c_variance = fit_normal(np.column_stack((x, a)), c)
    m_model, m_variance = fit_normal(np.column_stack((x, a, c)), m)
    mask0 = a == 0
    b_model = LinearRegression().fit(np.column_stack((x, c, m))[mask0], y[mask0])
    c0 = c_model.predict(np.column_stack((xe, zero)))
    c1 = c_model.predict(np.column_stack((xe, one)))
    m0 = m_model.predict(np.column_stack((xe, zero, ce)))
    m1 = m_model.predict(np.column_stack((xe, one, ce)))
    b = b_model.predict(np.column_stack((xe, ce, me)))
    bprime = b_model.predict(np.column_stack((xe, ce, m1)))
    m1_at_c0 = m_model.predict(np.column_stack((xe, one, c0)))
    bdoubleprime = b_model.predict(np.column_stack((xe, c0, m1_at_c0)))
    rm = np.exp(normal_log_ratio(me, m1, m0, m_variance))
    inv_rc = np.exp(normal_log_ratio(ce, c0, c1, c_variance))
    p = float(propensity)
    score = ((1 - ae) / (1 - p) * rm * (ye - b)
             + ae / p * inv_rc * (b - bprime)
             + (1 - ae) / (1 - p) * (bprime - bdoubleprime)
             + bdoubleprime)
    if not np.isfinite(score).all():
        raise FloatingPointError("Nonfinite observed-M1 MR score")
    return score
