"""Final fixed Gaussian simulation for the hidden recanting witness study.

Case dimensions are ordered (X, Z, W, M1, M2). M1 is latent and is never
returned to an estimator. The population target is
E[Y(0, M1(0), M2(1, M1(0)))] = 0.48 in both cases.
"""

from types import MappingProxyType

import numpy as np


CASE_DIMENSIONS = {1: "12211", 2: "53321"}
DIMENSIONS = {1: (1, 2, 2, 1, 1), 2: (5, 3, 3, 2, 1)}

# Rows are source coordinates and columns are destination coordinates.
# Floating-point literals retain the matrices in the final saved runs.
_COEFFICIENTS = {
    1: {
        "xa": [[0.35]],
        "xm1": [[0.50]],
        "am1": [[0.50]],
        "xm2": [[0.30]],
        "am2": [[0.60]],
        "m1m2": [[0.50]],
        "xz": [[0.25, -0.20]],
        "az": [[-0.8, -0.5599999999999999]],
        "m1z": [[1.00, 0.70]],
        "xw": [[0.30, -0.15]],
        "m1w": [[0.90, 0.70]],
        "xy": [[0.50]],
        "ay": [[0.30]],
        "m1y": [[1.50]],
        "m2y": [[0.80]],
        "wy": [[0.25], [-0.15]],
    },
    2: {
        "xa": [[0.22], [-0.18], [0.14], [0.10], [-0.10]],
        "xm1": [[0.32, 0.08], [-0.18, 0.28], [0.12, -0.20],
                [0.20, 0.16], [-0.10, 0.12]],
        "am1": [[0.40, 0.30]],
        "xm2": [[0.18], [-0.14], [0.12], [0.10], [-0.08]],
        "am2": [[0.60]],
        "m1m2": [[0.40], [0.30]],
        "xz": [[0.20, -0.10, 0.08], [-0.12, 0.18, 0.06],
               [0.08, 0.12, -0.16], [0.10, -0.08, 0.12],
               [-0.06, 0.10, 0.08]],
        "az": [[-0.7120000000000001, -0.584, -0.16800000000000007]],
        "m1z": [[1.00, 0.20, 0.60], [0.15, 0.95, -0.45]],
        "xw": [[0.18, -0.08, 0.10], [-0.10, 0.16, 0.06],
               [0.08, 0.10, -0.12], [0.12, -0.06, 0.10],
               [-0.06, 0.08, 0.06]],
        "m1w": [[0.95, -0.15, 0.60], [0.20, 1.00, 0.45]],
        "xy": [[0.30], [-0.20], [0.15], [0.12], [-0.10]],
        "ay": [[0.30]],
        "m1y": [[1.2000000000000002], [0.8999999999999999]],
        "m2y": [[0.80]],
        "wy": [[0.20], [-0.12], [0.10]],
    },
}


class NonparametricDGP:
    """Final DGP with independent N(0, 0.5) structural errors.

    X has identity covariance; A | X is Bernoulli(expit(X @ xa)). All
    coefficients and intercepts are fixed, independent of simulation seeds.
    """

    def __init__(self, case=1):
        if case not in CASE_DIMENSIONS:
            raise ValueError("case must be 1 or 2")
        self.case = case
        self.dimension_code = CASE_DIMENSIONS[case]
        self.dimensions = DIMENSIONS[case]
        self.noise_var = 0.5
        self.proxy_noise_var = 0.5
        coefficients = {key: np.array(value, dtype=np.float64)
                        for key, value in _COEFFICIENTS[case].items()}
        for value in coefficients.values():
            value.setflags(write=False)
        self.coefficients = MappingProxyType(coefficients)

    @property
    def population_truth(self):
        """Exact population nested mean, averaged over X and all errors."""
        b = self.coefficients
        return float((b["am2"] @ b["m2y"])[0, 0])

    def sample(self, n, seed):
        """Return observed x, z, w, m2, a, y as two-dimensional arrays."""
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        rng = np.random.default_rng(seed)
        dx, dz, dw, dm1, dm2 = self.dimensions
        b = self.coefficients
        sigma = np.sqrt(self.noise_var)
        proxy_sigma = np.sqrt(self.proxy_noise_var)
        # The draw order is part of the published simulation specification.
        x = rng.normal(size=(n, dx))
        p = np.exp(-np.logaddexp(0.0, -(x @ b["xa"])))
        a = rng.binomial(1, p).astype(np.float64)
        e1 = rng.normal(scale=sigma, size=(n, dm1))
        e2 = rng.normal(scale=sigma, size=(n, dm2))
        ez = rng.normal(scale=proxy_sigma, size=(n, dz))
        ew = rng.normal(scale=proxy_sigma, size=(n, dw))
        ey = rng.normal(scale=sigma, size=(n, 1))
        m1 = x @ b["xm1"] + a @ b["am1"] + e1
        m2 = x @ b["xm2"] + a @ b["am2"] + m1 @ b["m1m2"] + e2
        z = x @ b["xz"] + a @ b["az"] + m1 @ b["m1z"] + ez
        w = x @ b["xw"] + m1 @ b["m1w"] + ew
        y = (x @ b["xy"] + a @ b["ay"] + m1 @ b["m1y"]
             + m2 @ b["m2y"] + w @ b["wy"] + ey)
        return {"x": x, "z": z, "w": w, "m2": m2, "a": a, "y": y}

    def config(self):
        """Complete population specification saved with each run."""
        return {
            "name": "linear_gaussian_sequential_mediation",
            "case": self.case,
            "dimension_code": self.dimension_code,
            "dimension_order": ["x", "z", "w", "m1", "m2"],
            "dimensions": list(self.dimensions),
            "x_distribution": "N(0, I)",
            "treatment": "Bernoulli(expit(X @ xa))",
            "intercepts": "all_zero",
            "independent_structural_errors": True,
            "noise_variance_m1_m2_y": self.noise_var,
            "noise_variance_z_w": self.proxy_noise_var,
            "coefficients": {key: value.tolist() for key, value in self.coefficients.items()},
            "estimand": "E[Y(M2(M1(0),1),M1(0),0)]",
            "population_truth": self.population_truth,
        }
