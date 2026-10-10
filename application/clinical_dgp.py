"""Real baseline/outcome calibration with assumed synthetic mediator mechanisms.

Only X and fitted natural-arm log-CD4 means come from ACTG175. Treatment,
M1, M2, proxies, and outcomes are generated. The target is
E[Y(M2(M1(0), 1), M1(0), 0)] - E[Y(0)], in log(1+CD4) units.
"""

from pathlib import Path

import numpy as np


COEFFICIENTS = {
    "xm1": [[0.30], [-0.20], [0.10], [-0.15], [-0.25], [0.15]],
    "xm2": [[-0.10], [0.10], [0.10], [0.15], [0.10], [-0.20]],
    "xz": [[0.20, -0.10], [-0.12, 0.18], [0.08, 0.12],
           [0.10, -0.08], [-0.06, 0.10], [0.0, 0.0]],
    "az": [[0.25, -0.15]],
    "m1z": [[1.00, 0.70]],
    "xw": [[0.18, -0.08], [-0.10, 0.16], [0.08, 0.10],
           [0.12, -0.06], [-0.06, 0.08], [0.0, 0.0]],
    "m1w": [[0.90, 0.70]],
}


class ClinicalAnchoredDGP:
    """Empirical-X Gaussian SEM, with independent structural errors."""

    alpha = 0.50
    d = 0.60
    gamma = -0.50
    gamma1 = -0.55
    c_y = 0.80
    noise_var = 0.50
    treatment_probability = 0.50
    proxy_noise_var = 0.50
    adherence_intercept = float(np.log(0.85 / 0.15))

    def __init__(self, calibration_path):
        with np.load(Path(calibration_path), allow_pickle=False) as archive:
            required = {"x", "mu0", "mu1", "clinical_center", "clinical_scale"}
            if required.difference(archive.files):
                raise ValueError("Calibration archive lacks required arrays")
            self.x_pool = np.array(archive["x"], dtype=np.float64, copy=True)
            self.mu0 = np.array(archive["mu0"], dtype=np.float64, copy=True).reshape(-1, 1)
            self.mu1 = np.array(archive["mu1"], dtype=np.float64, copy=True).reshape(-1, 1)
            self.clinical_center = float(archive["clinical_center"])
            self.clinical_scale = float(archive["clinical_scale"])
        if (self.x_pool.ndim != 2 or self.x_pool.shape[1] != 6
                or len(self.x_pool) < 1
                or self.mu0.shape != (len(self.x_pool), 1)
                or self.mu1.shape != (len(self.x_pool), 1)):
            raise ValueError("Calibration requires aligned means and six baseline covariates")
        if not all(np.isfinite(v).all() for v in (self.x_pool, self.mu0, self.mu1)):
            raise ValueError("Calibration arrays must be finite")
        if (not np.isfinite(self.clinical_center)
                or not np.isfinite(self.clinical_scale) or self.clinical_scale <= 0
                or not np.isclose(self.clinical_center, self.mu0.mean(), rtol=1e-10, atol=1e-12)):
            raise ValueError("Invalid clinical center or scale")
        self.g0_pool = (self.mu0 - self.clinical_center) / self.clinical_scale
        self.g1_pool = (self.mu1 - self.clinical_center) / self.clinical_scale
        self.coefficients = {k: np.asarray(v, dtype=np.float64) for k, v in COEFFICIENTS.items()}

    @property
    def clinical_truth(self):
        """Generated population contrast, conditional on the fixed calibration."""
        return self.clinical_scale * (self.c_y * self.d)

    def sample_with_m1(self, n, seed):
        """Return observed data and factual M1 for the observed-M1 benchmark."""
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be a positive integer")
        rng = np.random.default_rng(seed)
        index = rng.integers(0, len(self.x_pool), size=n)
        x = self.x_pool[index].copy()
        a = rng.binomial(1, self.treatment_probability, size=(n, 1)).astype(np.float64)
        e1, e2 = (rng.normal(scale=np.sqrt(self.noise_var), size=(n, 1)) for _ in range(2))
        ez, ew = (rng.normal(scale=np.sqrt(self.proxy_noise_var), size=(n, 2)) for _ in range(2))
        ey = rng.normal(scale=np.sqrt(self.noise_var), size=(n, 1))
        b = self.coefficients
        u = x @ b["xm1"]
        mediator_base = self.adherence_intercept + x @ b["xm2"]
        g0, g1 = self.g0_pool[index], self.g1_pool[index]
        tau = g1 - g0 - self.gamma1 * self.alpha - self.c_y * (self.d + self.gamma * self.alpha)
        m1 = u + self.alpha * a + e1
        m2 = mediator_base + self.d * a + self.gamma * m1 + e2
        z = x @ b["xz"] + a @ b["az"] + m1 @ b["m1z"] + ez
        w = x @ b["xw"] + m1 @ b["m1w"] + ew
        y = (g0 + tau * a + self.gamma1 * (m1 - u)
             + self.c_y * (m2 - mediator_base - self.gamma * u) + ey)
        return {"x": x, "z": z, "w": w, "m2": m2, "a": a, "y": y}, m1
