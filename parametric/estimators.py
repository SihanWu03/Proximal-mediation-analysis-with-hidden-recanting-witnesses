"""Five proximal point estimates and the PMR influence-function interval."""
import numpy as np
from scipy.stats import norm

from .models import fit

ESTIMATORS = ("POR1", "POR2", "PH", "PIPW", "PMR")
OBSERVED = ("x", "a", "z", "w", "m2", "y")


def proximal_scores(data, nuisance):
    a = np.asarray(data["a"]).ravel()
    y = np.asarray(data["y"]).ravel()
    p, eta, h0, h1, q0, q1 = (
        np.asarray(nuisance[k]).ravel() for k in ("p", "eta", "h0", "h1", "q0", "q1")
    )
    if not np.all((p > 0) & (p < 1)):
        raise FloatingPointError("Non-interior fitted propensity; no clipping is applied")
    return {
        "POR1": eta,
        "POR2": (1-a)/(1-p)*h1,
        "PH": a/p*q1*h0,
        "PIPW": (1-a)/p*q0*y,
        "PMR": eta + a/p*q1*(h0-h1) + (1-a)/p*q0*(y-h0)
               + (1-a)/(1-p)*(h1-eta),
    }


def estimate_dataset(data, truth, scenario, confidence=.95):
    # Restrict fitting to observed variables.
    observed = {key: np.asarray(data[key]) for key in OBSERVED}
    fitted = fit(observed, scenario)
    nuisance = fitted.predict(observed)
    scores = proximal_scores(observed, nuisance)
    result = {}
    for name, score in scores.items():
        score = np.asarray(score, dtype=float).ravel()
        if not np.isfinite(score).all():
            raise FloatingPointError(f"Nonfinite {name} score")
        value = float(score.mean())
        result.update({f"{name}_estimate": value, f"{name}_residual": value-truth,
                       f"{name}_squared_error": (value-truth)**2})
    score = scores["PMR"]
    estimate = result["PMR_estimate"]
    variance = float(np.var(score-estimate, ddof=1))
    se = float(np.sqrt(variance/len(score)))
    half = float(norm.ppf((1+confidence)/2)*se)
    result.update(PMR_se=se, PMR_if_variance=variance,
        PMR_ci_lower=estimate-half, PMR_ci_upper=estimate+half,
        PMR_ci_width=2*half, PMR_ci_cover=int(estimate-half <= truth <= estimate+half),
        PMR_ci_method="original_EIF_only")
    return result, None

