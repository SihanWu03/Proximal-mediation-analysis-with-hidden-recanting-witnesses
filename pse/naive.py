"""Cross-fitted comparator using L=(Z,W) in the observed-witness EIF.

The target is E_X E_{L|A=0,X} E_{M2|A=1,L,X} E[Y|A=0,M2,L,X].
Its nuisance regressions use only the supplied outer training fold. Only
binary treatment probabilities are clipped; EIF scores are never truncated.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass(frozen=True)
class NaiveConfig:
    n_estimators: int = 200
    min_samples_leaf: int | None = None
    max_features: float = 1.0
    n_jobs: int = 1
    propensity_clip: float = 0.025
    random_state: int = 42


def _vector(value, name, n=None):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim == 2 and out.shape[1] == 1:
        out = out[:, 0]
    if out.ndim != 1 or (n is not None and len(out) != n):
        raise ValueError(f"{name} must have shape (n,) or (n, 1)")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains nonfinite values")
    return out


def _matrix(value, name, n=None):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or out.shape[1] < 1 or (n is not None and len(out) != n):
        raise ValueError(f"{name} must be a nonempty n-by-d matrix")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains nonfinite values")
    return out


def _observed(data):
    x = _matrix(data["x"], "x")
    n = len(x)
    result = {"x": x, "a": _vector(data["a"], "a", n),
              "y": _vector(data["y"], "y", n)}
    if not np.isin(result["a"], [0, 1]).all():
        raise ValueError("a must contain only zero and one")
    for key in ("z", "w", "m2"):
        result[key] = _matrix(data[key], key, n)
    return result


def miles_score(a, y, b, c, d, p1, p1_l, p1_ml):
    """Uncentered EIF with B, C, D and treatment density ratios.

    B=E[Y|A=0,M2,L,X], C=E[B|A=1,L,X], D=E[C|A=0,X].
    """
    a = _vector(a, "a")
    n = len(a)
    y, b, c, d, p1, p1_l, p1_ml = [
        _vector(value, name, n) for value, name in
        ((y, "y"), (b, "b"), (c, "c"), (d, "d"),
         (p1, "p1"), (p1_l, "p1_l"), (p1_ml, "p1_ml"))
    ]
    if not np.isin(a, [0, 1]).all():
        raise ValueError("a must contain only zero and one")
    for p in (p1, p1_l, p1_ml):
        if np.any((p <= 0) | (p >= 1)):
            raise ValueError("probabilities must lie strictly between zero and one")
    r_l = (1 - p1_l) / p1_l * p1 / (1 - p1)
    r_m = p1_ml / (1 - p1_ml) * (1 - p1_l) / p1_l
    return (d + (1 - a) / (1 - p1) * r_m * (y - b)
            + a / p1 * r_l * (b - c)
            + (1 - a) / (1 - p1) * (c - d))


class NaiveEstimator:
    """Fit one training fold; score using shared held-out P(A=1|X)."""

    def __init__(self, config=None):
        self.config = config or NaiveConfig()
        if not 0 < self.config.propensity_clip < 0.5:
            raise ValueError("propensity_clip must be between zero and 0.5")
        self._fitted = False

    def fit(self, train):
        data = _observed(train)
        x, a, y = data["x"], data["a"], data["y"]
        mask0, mask1 = a == 0, a == 1
        if min(mask0.sum(), mask1.sum()) < 2:
            raise ValueError("training folds need at least two observations in each arm")
        leaf = self.config.min_samples_leaf
        if leaf is None:
            leaf = max(10, int(np.ceil(np.sqrt(len(a)))))
        common = dict(n_estimators=self.config.n_estimators,
                      min_samples_leaf=leaf, max_features=self.config.max_features,
                      max_depth=None, n_jobs=self.config.n_jobs)
        seed = int(self.config.random_state) % (2**32 - 1)

        def child_seed(offset):
            return (seed + offset) % (2**32 - 1)

        lx = np.column_stack((data["z"], data["w"], x))
        mlx = np.column_stack((data["m2"], lx))
        self._dimensions = {key: data[key].shape[1] for key in ("x", "z", "w", "m2")}
        self.l_model = RandomForestClassifier(**common, random_state=child_seed(1))
        self.ml_model = RandomForestClassifier(**common, random_state=child_seed(2))
        self.l_model.fit(lx, a)
        self.ml_model.fit(mlx, a)
        self.b_model = RandomForestRegressor(**common, random_state=child_seed(3))
        self.b_model.fit(mlx[mask0], y[mask0])
        self.c_model = RandomForestRegressor(**common, random_state=child_seed(4))
        self.c_model.fit(lx[mask1], self.b_model.predict(mlx[mask1]))
        self.d_model = RandomForestRegressor(**common, random_state=child_seed(5))
        self.d_model.fit(x[mask0], self.c_model.predict(lx[mask0]))
        self._fitted = True
        return self

    def _probability(self, model, features, name):
        classes = np.asarray(model.classes_)
        index = np.flatnonzero(classes == 1)
        if len(index) != 1 or not np.any(classes == 0):
            raise ValueError(f"{name} classifier must contain both treatment classes")
        return _vector(model.predict_proba(features)[:, index[0]], name, len(features))

    def _stabilize(self, raw, name):
        if np.any((raw < 0) | (raw > 1)):
            raise ValueError(f"{name} is not a probability")
        floor = self.config.propensity_clip
        return np.clip(raw, floor, 1 - floor)

    def predict_scores(self, evaluation, p1_eval):
        if not self._fitted:
            raise RuntimeError("call fit before predict_scores")
        data = _observed(evaluation)
        for key, dimension in self._dimensions.items():
            if data[key].shape[1] != dimension:
                raise ValueError(f"{key} dimension differs from the training fold")
        lx = np.column_stack((data["z"], data["w"], data["x"]))
        mlx = np.column_stack((data["m2"], lx))
        nuisance = {
            "b": self.b_model.predict(mlx),
            "c": self.c_model.predict(lx),
            "d": self.d_model.predict(data["x"]),
            "p1": self._stabilize(_vector(p1_eval, "p1_eval", len(lx)), "p1"),
            "p1_l": self._stabilize(self._probability(self.l_model, lx, "p1_l"), "p1_l"),
            "p1_ml": self._stabilize(self._probability(self.ml_model, mlx, "p1_ml"), "p1_ml"),
        }
        scores = miles_score(evaluation["a"], evaluation["y"], **nuisance)
        if not np.isfinite(scores).all():
            raise FloatingPointError("nonfinite naive EIF scores")
        return scores
