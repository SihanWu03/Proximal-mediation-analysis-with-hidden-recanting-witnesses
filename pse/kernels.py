"""CUDA-capable, finite-rank Gaussian/linear kernel features.

Only landmark-by-landmark and feature-by-feature systems are formed. In
particular, neither fitting nor prediction creates an n-by-n Gram matrix.
Regularization parameters use empirical *mean* loss, independent of sample size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def resolve_device(device="cpu"):
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch installation cannot use CUDA")
    return result


def resolve_dtype(dtype="float64"):
    if dtype not in ("float64", torch.float64):
        raise ValueError("The manuscript implementation uses float64")
    return torch.float64


def to_tensor(value, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


def rbf_kernel(x, y, gamma):
    distance = ((x * x).sum(1, keepdim=True)
                + (y * y).sum(1).unsqueeze(0) - 2 * (x @ y.T)).clamp_min(0)
    return torch.exp(-float(gamma) * distance)


def solve_spd(matrix, rhs, jitter=1e-10):
    """Solve an SPD system, with bounded diagonal-ridge retries if needed."""
    matrix = (matrix + matrix.T) * 0.5
    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info.max().item()) != 0:
        eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        scale = matrix.diag().abs().max().clamp_min(1.0)
        for retry in range(6):
            factor, info = torch.linalg.cholesky_ex(matrix + (jitter * 10**retry) * scale * eye)
            if int(info.max().item()) == 0:
                break
        else:
            raise RuntimeError("Positive-definite kernel solve failed after numerical ridge retries")
    vector = rhs.ndim == 1
    solution = torch.cholesky_solve(rhs[:, None] if vector else rhs, factor)
    if not bool(torch.isfinite(solution).all().item()):
        raise FloatingPointError("Kernel solve produced nonfinite coefficients")
    return solution[:, 0] if vector else solution


class FeatureMap:
    """Training-standardized Nystrom RBF features plus intercept and linear terms.

    The linear component is part of the chosen composite kernel, not a separate
    parametric bridge model. With r landmarks and d input coordinates, the map
    has at most r+d+1 columns. Gaussian landmark eigenmodes below a relative
    numerical tolerance are discarded instead of amplified.
    """

    def __init__(self, n_components=128, device="cpu", dtype="float64", seed=0):
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.device, self.dtype = resolve_device(device), resolve_dtype(dtype)
        self.seed = int(seed)

    @torch.no_grad()
    def fit(self, x):
        values = to_tensor(x, self.device, self.dtype)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("Kernel inputs must be a two-dimensional nonempty feature matrix")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("Kernel inputs contain NaN or infinity")
        self.n_features_in_ = values.shape[1]
        self.mean_ = values.mean(0)
        self.scale_ = values.std(0, unbiased=False).clamp_min(1e-8)
        standardized = (values - self.mean_) / self.scale_
        indices = np.random.default_rng(self.seed).choice(
            len(values), min(self.n_components, len(values)), replace=False)
        self.centers_ = standardized[torch.as_tensor(indices, device=self.device)].clone()
        distances = torch.pdist(self.centers_[:512]).square()
        distances = distances[distances > 1e-12]
        self.gamma_ = float((1 / distances.median()).item()) if distances.numel() else 1.0
        eigenvalues, eigenvectors = torch.linalg.eigh(rbf_kernel(self.centers_, self.centers_, self.gamma_))
        tolerance = 1e-10
        keep = eigenvalues > eigenvalues[-1].clamp_min(1.0) * tolerance
        self.whitener_ = eigenvectors[:, keep] / eigenvalues[keep].sqrt().unsqueeze(0)
        self.output_dim_ = 1 + self.n_features_in_ + int(keep.sum().item())
        return self

    @torch.no_grad()
    def transform(self, x):
        values = to_tensor(x, self.device, self.dtype)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature dimensions do not match the training inputs")
        standardized = (values - self.mean_) / self.scale_
        pieces = [torch.ones((len(values), 1), device=self.device, dtype=self.dtype)]
        pieces.append(standardized * np.sqrt(1.0 / self.n_features_in_))
        pieces.append(rbf_kernel(standardized, self.centers_, self.gamma_) @ self.whitener_)
        return torch.cat(pieces, dim=1)

    def fit_transform(self, x):
        return self.fit(x).transform(x)


@dataclass
class KernelFunction:
    feature_map: FeatureMap
    coefficient: torch.Tensor

    @torch.no_grad()
    def __call__(self, x):
        return self.feature_map.transform(x) @ self.coefficient

    def predict(self, x):
        return self(x).detach().cpu().numpy()


class KernelRidge:
    """Ridge regression in exactly the same feature family as the bridges."""

    def __init__(self, n_components=128, ridge=1e-3, device="cpu",
                 dtype="float64", seed=0):
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.ridge = float(ridge)
        self.feature_map = FeatureMap(n_components, device, dtype, seed)

    @torch.no_grad()
    def fit(self, x, y):
        features = self.feature_map.fit_transform(x)
        response = to_tensor(y, features.device, features.dtype)
        if response.shape[0] != features.shape[0]:
            raise ValueError("Kernel ridge response length does not match X")
        eye = torch.eye(features.shape[1], device=features.device, dtype=features.dtype)
        self.coefficient_ = solve_spd(features.T @ features / len(features) + self.ridge * eye,
                                      features.T @ response / len(features))
        return self

    @torch.no_grad()
    def predict_tensor(self, x):
        return self.feature_map.transform(x) @ self.coefficient_

    def predict(self, x):
        return self.predict_tensor(x).detach().cpu().numpy()
