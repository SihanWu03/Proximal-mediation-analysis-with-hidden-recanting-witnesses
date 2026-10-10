"""Separate first-stage and joint second-stage regularized minimax bridges."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .kernels import FeatureMap, KernelFunction, resolve_device, resolve_dtype, solve_spd, to_tensor


@dataclass
class BridgeConfig:
    n_components: int = 128
    lambda_bridge: float = 1e-4
    lambda_adv: float = 1e-2
    lambda_rkhs: float = 1e-4
    jitter: float = 1e-10
    device: str = "cpu"
    dtype: str = "float64"
    seed: int = 0


def observed_features(data):
    """Explicit allowlist: hidden m1 values cannot affect any feature matrix."""
    values = {}
    for key in ("x", "w", "z", "m2"):
        arr = np.asarray(data[key], dtype=float)
        values[key] = arr[:, None] if arr.ndim == 1 else arr
    return {"h0": np.hstack((values["w"], values["m2"], values["x"])),
            "h1": np.hstack((values["w"], values["x"])),
            "q0": np.hstack((values["z"], values["m2"], values["x"])),
            "q1": np.hstack((values["z"], values["x"]))}


def reduced_minimax(penalties, moments, jitter=1e-10):
    """Solve min_theta sum_j sup_beta_j [beta_j'(C_j theta+b_j)
    - beta_j'S_j beta_j] + theta'P theta via its positive-definite Schur form.

    ``moments`` holds (C_j, b_j, S_j), and ``penalties`` is a list of the
    block-diagonal primal penalty matrices. This gives
    (sum C' S^-1 C + 4P) theta = -sum C' S^-1 b.
    """
    primal_penalty = torch.block_diag(*penalties)
    lhs, rhs = 4.0 * primal_penalty, torch.zeros(
        primal_penalty.shape[0], device=primal_penalty.device, dtype=primal_penalty.dtype)
    for cross, offset, dual_penalty in moments:
        combined = solve_spd(dual_penalty, torch.cat((cross, offset[:, None]), dim=1), jitter)
        lhs = lhs + cross.T @ combined[:, :-1]
        rhs = rhs - cross.T @ combined[:, -1]
    return solve_spd(lhs, rhs, jitter)


class HiddenBridgeSet:
    def __init__(self, config=None):
        self.config = config or BridgeConfig()

    @torch.no_grad()
    def fit(self, train, p1_train):
        cfg = self.config
        if min(cfg.lambda_bridge, cfg.lambda_adv) <= 0 or cfg.lambda_rkhs < 0:
            raise ValueError("Bridge/adversary penalties must be positive and lambda_rkhs nonnegative")
        device, dtype = resolve_device(cfg.device), resolve_dtype(cfg.dtype)
        a = to_tensor(train["a"], device, dtype).reshape(-1)
        y = to_tensor(train["y"], device, dtype).reshape(-1)
        p1 = to_tensor(p1_train, device, dtype).reshape(-1)
        n = len(a)
        if not (len(y) == len(p1) == n) or not bool(((a == 0) | (a == 1)).all().item()):
            raise ValueError("Require aligned responses, propensities, and binary treatment")
        if not bool(((p1 > 0) & (p1 < 1) & torch.isfinite(p1)).all().item()):
            raise ValueError("Training propensities must be finite and strictly between 0 and 1")
        mask0, mask1 = a == 0, a == 1
        if min(int(mask0.sum().item()), int(mask1.sum().item())) < 2:
            raise ValueError("Each training fold needs at least two observations per treatment arm")
        raw_features = observed_features(train)
        maps, features = {}, {}
        for index, (name, raw) in enumerate(raw_features.items()):
            if len(raw) != n:
                raise ValueError("Observed feature lengths do not match treatment")
            maps[name] = FeatureMap(cfg.n_components, cfg.device, cfg.dtype, cfg.seed + index * 103)
            features[name] = maps[name].fit_transform(raw)
        lam_h, lam_f, lam_rkhs = cfg.lambda_bridge, cfg.lambda_adv, cfg.lambda_rkhs

        def eye(matrix):
            return torch.eye(matrix.shape[1], dtype=dtype, device=device)

        def primal(matrix):
            return lam_h * (matrix.T @ matrix) / n + lam_rkhs * eye(matrix)

        # Both empirical squared penalties use the entire training fold.
        dual = {name: phi.T @ phi / n + lam_f * eye(phi) for name, phi in features.items()}
        penalty = {name: primal(phi) for name, phi in features.items()}
        h0, h1, q0, q1 = (features[name] for name in ("h0", "h1", "q0", "q1"))
        n0, n1 = mask0.sum(), mask1.sum()
        h0_cross = -q0[mask0].T @ h0[mask0] / n0
        h0_offset = q0[mask0].T @ y[mask0] / n0
        q1_cross = q1.T @ ((a / p1)[:, None] * h1) / n
        # Cross matrices have dual-by-primal dimensions.
        q1_cross = q1_cross.T
        q1_offset = -(h1.T @ ((1 - a) / (1 - p1))) / n
        h0_coef = reduced_minimax([penalty["h0"]], [(h0_cross, h0_offset, dual["q0"])], cfg.jitter)
        q1_coef = reduced_minimax([penalty["q1"]], [(q1_cross, q1_offset, dual["h1"])], cfg.jitter)

        # Joint outcome stage: E1[(h0_aux-h1)f1], E0[(Y-h0_aux)f0].
        h_joint_cross1 = torch.cat((-q1[mask1].T @ h1[mask1] / n1,
                                    q1[mask1].T @ h0[mask1] / n1), dim=1)
        h_joint_cross0 = torch.cat((torch.zeros((q0.shape[1], h1.shape[1]), device=device, dtype=dtype),
                                    h0_cross), dim=1)
        h_joint_coef = reduced_minimax([penalty["h1"], penalty["h0"]], [
            (h_joint_cross1, torch.zeros(q1.shape[1], device=device, dtype=dtype), dual["q1"]),
            (h_joint_cross0, h0_offset, dual["q0"])], cfg.jitter)

        # Joint treatment stage: NO propensity factor belongs in q0's moment.
        # E[(A*q1_aux-(1-A)*q0) f0], E[(A/p1*q1_aux-(1-A)/p0) f1].
        q_joint_cross0 = torch.cat((-h0.T @ ((1 - a)[:, None] * q0) / n,
                                    h0.T @ (a[:, None] * q1) / n), dim=1)
        q_joint_cross1 = torch.cat((torch.zeros((h1.shape[1], q0.shape[1]), device=device, dtype=dtype),
                                    q1_cross), dim=1)
        q_joint_coef = reduced_minimax([penalty["q0"], penalty["q1"]], [
            (q_joint_cross0, torch.zeros(h0.shape[1], device=device, dtype=dtype), dual["h0"]),
            (q_joint_cross1, q1_offset, dual["h1"])], cfg.jitter)

        coefficients = {"h0": h0_coef, "h1": h_joint_coef[:h1.shape[1]].clone(),
                        "q0": q_joint_coef[:q0.shape[1]].clone(), "q1": q1_coef}
        self.models_ = {name: KernelFunction(maps[name], coef.detach()) for name, coef in coefficients.items()}
        return self

    def predict(self, evaluation):
        features = observed_features(evaluation)
        return {name: self.models_[name].predict(raw).reshape(-1) for name, raw in features.items()}
