"""Cross-fitted point estimates and inference from the hidden-witness EIF."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from .bridges import HiddenBridgeSet
from .kernels import KernelRidge
from .naive import NaiveConfig, NaiveEstimator

ESTIMATORS = ("POR1", "POR2", "PH", "PIPW", "PMR", "naive")
OBSERVED = ("x", "z", "w", "m2", "a", "y")


@dataclass
class EstimationConfig:
    folds: int = 5
    trees: int = 200
    forest_jobs: int = 1
    propensity_clip: float = 0.025
    eta_ridge: float = 0.001
    confidence: float = 0.95


def proximal_scores(data, nuisance, p1, eta):
    """Uncentered scores; the PMR score minus its mean is the estimated EIF.

    In particular the q0 term is divided by p1, NOT p0 (Theorem 4).
    POR2 implements Remark 3. No weight, bridge, outcome or score clipping.
    """
    a = np.asarray(data["a"]).ravel()
    y = np.asarray(data["y"]).ravel()
    p1, eta = np.asarray(p1).ravel(), np.asarray(eta).ravel()
    h0, h1, q0, q1 = (np.asarray(nuisance[k]).ravel() for k in ("h0", "h1", "q0", "q1"))
    i0, i1 = (a == 0).astype(float), (a == 1).astype(float)
    return {
        "POR1": eta,
        "POR2": i0 / (1 - p1) * h1,
        "PH": i1 / p1 * q1 * h0,
        "PIPW": i0 / p1 * q0 * y,
        "PMR": eta + i1 / p1 * q1 * (h0 - h1)
        + i0 / p1 * q0 * (y - h0) + i0 / (1 - p1) * (h1 - eta),
    }


def influence_interval(scores, truth, confidence=0.95):
    values = np.asarray(scores, dtype=float).ravel()
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("EIF inference requires at least two finite out-of-fold scores")
    estimate = float(values.mean())
    eif = values - estimate  # one global center, including between-fold variation
    variance = float(np.dot(eif, eif) / (len(values) - 1))
    se = math.sqrt(variance / len(values))
    half_width = float(norm.ppf((1 + confidence) / 2) * se)
    return dict(se=se, if_variance=variance, ci_lower=estimate - half_width,
                ci_upper=estimate + half_width, ci_width=2 * half_width,
                ci_cover=int(estimate - half_width <= truth <= estimate + half_width))


def estimate_dataset(data, truth, bridge_config, config, seed=0):
    """Every observation is evaluated once, using nuisances fit without that fold."""
    # Only observed variables are supplied to the nuisance learners.
    data = {k: np.asarray(data[k]) for k in OBSERVED}
    n = len(data["a"])
    a = data["a"].ravel()
    if config.folds < 2 or min(np.bincount(a.astype(int), minlength=2)) < config.folds:
        raise ValueError("Need at least folds observations in each treatment arm")
    all_scores = {k: np.full(n, np.nan) for k in ESTIMATORS}
    seen = np.zeros(n, dtype=int)
    splitter = StratifiedKFold(config.folds, shuffle=True, random_state=seed)
    from dataclasses import replace
    for fold, (train_index, eval_index) in enumerate(splitter.split(np.zeros(n), a)):
        train = {k: v[train_index] for k, v in data.items()}
        evaluation = {k: v[eval_index] for k, v in data.items()}
        fold_seed = int((seed + 7919 * (fold + 1)) % (2**32 - 1))
        leaf = max(10, math.ceil(math.sqrt(len(train_index))))
        propensity = RandomForestClassifier(
            n_estimators=config.trees, min_samples_leaf=leaf, max_features=1.0,
            n_jobs=config.forest_jobs, random_state=fold_seed,
        ).fit(train["x"], train["a"].ravel())
        train_raw = propensity.predict_proba(train["x"])[:, 1]
        eval_raw = propensity.predict_proba(evaluation["x"])[:, 1]
        eps = config.propensity_clip
        p_train = np.clip(train_raw, eps, 1 - eps)
        p_eval = np.clip(eval_raw, eps, 1 - eps)
        cfg = replace(bridge_config, seed=fold_seed)
        bridges = HiddenBridgeSet(cfg).fit(train, p_train)
        nuisance = bridges.predict(evaluation)
        loc0 = train["a"].ravel() == 0
        eta_targets = bridges.predict({k: v[loc0] for k, v in train.items()})["h1"]
        eta_model = KernelRidge(
            n_components=cfg.n_components, ridge=config.eta_ridge,
            device=cfg.device, dtype=cfg.dtype, seed=fold_seed + 37,
        ).fit(train["x"][loc0], eta_targets)
        eta = eta_model.predict(evaluation["x"])
        fold_scores = proximal_scores(evaluation, nuisance, p_eval, eta)
        naive_cfg = NaiveConfig(
            n_estimators=config.trees, min_samples_leaf=leaf, max_features=1.0,
            n_jobs=config.forest_jobs, propensity_clip=eps, random_state=fold_seed,
        )
        naive = NaiveEstimator(naive_cfg).fit(train)
        fold_scores["naive"] = naive.predict_scores(evaluation, p1_eval=p_eval)
        for name in ESTIMATORS:
            values = np.asarray(fold_scores[name]).ravel()
            if len(values) != len(eval_index) or not np.isfinite(values).all():
                raise FloatingPointError(f"Nonfinite or malformed {name} scores, fold {fold}")
            all_scores[name][eval_index] = values
        seen[eval_index] += 1
    if not np.all(seen == 1):
        raise AssertionError("Each observation must have exactly one out-of-fold score")
    result = {}
    for name, scores in all_scores.items():
        estimate = float(scores.mean())
        result.update({f"{name}_estimate": estimate, f"{name}_residual": estimate - truth,
                       f"{name}_squared_error": (estimate - truth)**2})
        if name in ("PMR", "naive"):
            result.update({f"{name}_{k}": v for k, v in
                           influence_interval(scores, truth, config.confidence).items()})
    return result, None
