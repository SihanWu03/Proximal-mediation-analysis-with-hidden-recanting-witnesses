"""Fixed linear data-generating process; only observed variables are returned."""
from __future__ import annotations

import numpy as np
from scipy.special import expit


class FixedLinearDGP:
    dimensions = (1, 2, 2, 1, 1)
    population_truth = 0.48

    def sample(self, n: int, seed: int) -> dict[str, np.ndarray]:
        if isinstance(n, bool) or int(n) != n or n < 1:
            raise ValueError("n must be a positive integer")
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 1))
        p = expit(0.6 * x)
        a = rng.binomial(1, p).astype(float)
        eu = rng.normal(size=(n, 1))
        em = rng.normal(size=(n, 1))
        ez = rng.normal(scale=np.sqrt(0.5), size=(n, 2))
        ew = rng.normal(scale=np.sqrt(0.5), size=(n, 2))
        ey = rng.normal(size=(n, 1))
        u = 0.5 * x + 0.5 * a + eu
        m = 0.3 * x + 0.6 * a + 0.5 * u + em
        z = 0.2 * x + 0.2 * a + u + ez
        w = 0.3 * x + u + ew
        y = 0.5 * x + 0.3 * a + u + 0.8 * m + 0.2 * w.mean(axis=1, keepdims=True) + ey
        return dict(x=x, a=a, z=z, w=w, m2=m, y=y)
