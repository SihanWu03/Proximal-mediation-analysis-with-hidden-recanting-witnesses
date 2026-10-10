"""Observed-data moment fits for the five manuscript scenarios.

Outcome bridges are affine and treatment bridges are exponential affine.
A misspecified bridge omits its proxy coordinate. Misspecified propensity
omits X; eta omits X only in q_wrong. Auxiliary h0 and q1 parameters enter
the original joint estimating equations.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import root
from scipy.special import expit


CASES = ("p_q_wrong", "q_wrong", "second_wrong", "h_wrong", "all_correct")
SPECIFICATIONS = {
    "all_correct": dict(p=True, h0=True, h1=True, q0=True, q1=True, eta=True),
    "second_wrong": dict(p=True, h0=True, h1=False, q0=False, q1=True, eta=True),
    "h_wrong": dict(p=True, h0=False, h1=False, q0=True, q1=True, eta=True),
    "q_wrong": dict(p=True, h0=True, h1=True, q0=False, q1=False, eta=False),
    "p_q_wrong": dict(p=False, h0=True, h1=True, q0=False, q1=False, eta=True),
}


def _observed(data):
    out = {k: np.asarray(data[k], dtype=float) for k in ("x", "z", "w", "m2", "a", "y")}
    out = {k: v[:, None] if v.ndim == 1 else v for k, v in out.items()}
    n = len(out["a"])
    for key, width in (("x", 1), ("z", 2), ("w", 2), ("m2", 1), ("a", 1), ("y", 1)):
        if out[key].shape != (n, width) or not np.isfinite(out[key]).all():
            raise ValueError(f"Invalid observed array {key}")
    if not np.isin(out["a"], [0, 1]).all():
        raise ValueError("A must be binary")
    return out


def _linear_solve(lhs, rhs, label):
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise FloatingPointError(f"Nonfinite {label} system")
    if np.linalg.matrix_rank(lhs) != lhs.shape[1]:
        raise np.linalg.LinAlgError(f"Rank-deficient {label} moment system")
    solution = np.linalg.solve(lhs, rhs)
    if not np.isfinite(solution).all():
        raise FloatingPointError(f"Nonfinite {label} solution")
    return solution


@dataclass
class FittedModel:
    case: str
    theta: np.ndarray
    slices: dict[str, slice]

    def _design(self, data):
        data = _observed(data)
        x, m = data["x"], data["m2"]
        z, w = data["z"].mean(axis=1, keepdims=True), data["w"].mean(axis=1, keepdims=True)
        one = np.ones_like(x)
        spec = SPECIFICATIONS[self.case]

        def basis(proxy, correct, stage0=False):
            return np.hstack([one, x] + ([proxy] if correct else []) + ([m] if stage0 else []))

        f = {"p": np.hstack([one, x]) if spec["p"] else one,
             "eta": np.hstack([one, x]) if spec["eta"] else one}
        for name in ("h0", "h1", "q0", "q1"):
            is_h, stage0 = name[0] == "h", name[1] == "0"
            f[name] = basis(w if is_h else z, spec[name], stage0)
            f["iv_" + name] = basis(z if is_h else w, spec[name], stage0)
        f["h0_aux"] = f["h0"]
        f["q1_aux"] = f["q1"]
        return data, f

    def predict(self, data):
        _, f = self._design(data)
        p = expit(f["p"] @ self.theta[self.slices["p"]])
        if not ((p > 0) & (p < 1)).all():
            raise FloatingPointError("Estimated propensity is numerically 0 or 1; no clipping applied")
        result = {"p": p}
        with np.errstate(over="raise", invalid="raise"):
            for name in ("h0", "h1", "h0_aux", "q0", "q1", "q1_aux", "eta"):
                value = f[name] @ self.theta[self.slices[name]]
                result[name] = np.exp(value) if name.startswith("q") else value
        if not all(np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError("Nonfinite nuisance prediction")
        return result

    def moments(self, data):
        obs, f = self._design(data)
        v = self.predict(data)
        a, y = obs["a"].ravel(), obs["y"].ravel()
        p, i0 = v["p"], 1 - a
        residual = {
            "p": p - a,
            "h0": i0 * (v["h0"] - y),
            "h1": a * (v["h1"] - v["h0_aux"]),
            "h0_aux": i0 * (v["h0_aux"] - y),
            "q1": a / p * v["q1"] - i0 / (1 - p),
            "q0": i0 * v["q0"] - a * v["q1_aux"],
            "q1_aux": a / p * v["q1_aux"] - i0 / (1 - p),
            "eta": i0 * (v["eta"] - v["h1"]),
        }
        instruments = {"p": f["p"], "eta": f["eta"], "h0_aux": f["iv_h0"], "q1_aux": f["iv_q1"]}
        instruments.update({k: f["iv_" + k] for k in ("h0", "h1", "q0", "q1")})
        return np.hstack([residual[k][:, None] * instruments[k] for k in self.slices])

def fit(data, case="all_correct"):
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    model = FittedModel(case=case, theta=np.empty(0), slices={})
    obs, f = model._design(data)
    a, y = obs["a"].ravel(), obs["y"].ravel()
    n, i0 = len(a), 1 - a
    if min(np.sum(a), np.sum(i0)) < 10:
        raise ValueError("Fewer than 10 observations in one treatment arm")
    start = 0
    for name in ("p", "h0", "h1", "h0_aux", "q1", "q0", "q1_aux", "eta"):
        model.slices[name] = slice(start, start + f[name].shape[1])
        start += f[name].shape[1]
    model.theta = np.zeros(start)

    def solve_root(fun, jac, initial, label):
        answer = root(fun, initial, jac=jac, method="hybr", options={"xtol": 1e-10, "maxfev": 1000})
        residual = np.max(np.abs(fun(answer.x)))
        if not np.isfinite(answer.x).all() or not np.isfinite(residual) or residual > 1e-7:
            raise RuntimeError(f"{label} moment solve failed: residual={residual:.3g}; {answer.message}")
        return answer.x

    fp = f["p"]
    def p_moment(b):
        return fp.T @ (expit(fp @ b) - a) / n
    def p_jac(b):
        pr = expit(fp @ b)
        return fp.T @ ((pr * (1 - pr))[:, None] * fp) / n
    initial = np.zeros(fp.shape[1])
    initial[0] = np.log(np.mean(a) / np.mean(i0))
    model.theta[model.slices["p"]] = solve_root(p_moment, p_jac, initial, "p")
    p = expit(fp @ model.theta[model.slices["p"]])
    if not ((p > 0) & (p < 1)).all():
        raise FloatingPointError("Propensity numerical positivity failure")

    fh0, ih0 = f["h0"], f["iv_h0"]
    h0_lhs = ih0.T @ (i0[:, None] * fh0) / n
    h0_rhs = ih0.T @ (i0 * y) / n
    model.theta[model.slices["h0"]] = _linear_solve(h0_lhs, h0_rhs, "standalone h0")
    fh1, ih1 = f["h1"], f["iv_h1"]
    joint_lhs = np.block([[ih1.T @ (a[:, None] * fh1) / n,
                           -ih1.T @ (a[:, None] * fh0) / n],
                          [np.zeros((fh0.shape[1], fh1.shape[1])), h0_lhs]])
    joint_rhs = np.r_[np.zeros(fh1.shape[1]), h0_rhs]
    joint_h = _linear_solve(joint_lhs, joint_rhs, "joint h1,h0_aux")
    model.theta[model.slices["h1"]] = joint_h[:fh1.shape[1]]
    model.theta[model.slices["h0_aux"]] = joint_h[fh1.shape[1]:]

    fq1, iq1 = f["q1"], f["iv_q1"]
    fq0, iq0 = f["q0"], f["iv_q0"]
    wp, wm = a / p, i0 / (1 - p)
    def q1_moment(b):
        with np.errstate(over="raise", invalid="raise"):
            return iq1.T @ (wp * np.exp(fq1 @ b) - wm) / n
    def q1_jac(b):
        return iq1.T @ ((wp * np.exp(fq1 @ b))[:, None] * fq1) / n
    model.theta[model.slices["q1"]] = solve_root(q1_moment, q1_jac, np.zeros(fq1.shape[1]), "standalone q1")
    k0 = fq0.shape[1]
    def joint_q_moment(b):
        with np.errstate(over="raise", invalid="raise"):
            q0, q1a = np.exp(fq0 @ b[:k0]), np.exp(fq1 @ b[k0:])
            return np.r_[iq0.T @ (i0 * q0 - a * q1a) / n,
                         iq1.T @ (wp * q1a - wm) / n]
    def joint_q_jac(b):
        q0, q1a = np.exp(fq0 @ b[:k0]), np.exp(fq1 @ b[k0:])
        return np.block([[iq0.T @ ((i0 * q0)[:, None] * fq0) / n,
                          -iq0.T @ ((a * q1a)[:, None] * fq1) / n],
                         [np.zeros((fq1.shape[1], k0)), iq1.T @ ((wp * q1a)[:, None] * fq1) / n]])
    initial = np.zeros(k0 + fq1.shape[1])
    initial[0] = np.log(np.mean(a) / np.mean(i0))
    joint_q = solve_root(joint_q_moment, joint_q_jac, initial, "joint q0,q1_aux")
    model.theta[model.slices["q0"]] = joint_q[:k0]
    model.theta[model.slices["q1_aux"]] = joint_q[k0:]
    fe = f["eta"]
    h1 = fh1 @ model.theta[model.slices["h1"]]
    model.theta[model.slices["eta"]] = _linear_solve(fe.T @ (i0[:, None] * fe) / n,
                                                   fe.T @ (i0 * h1) / n, "eta")
    mean_moment = model.moments(data).mean(axis=0)
    max_moment = float(np.max(np.abs(mean_moment)))
    if max_moment > 1e-7:
        raise RuntimeError(f"Fitted stacked moments fail tolerance: {max_moment}")
    return model

