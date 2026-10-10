"""Reproduce the fixed n=8000, seed=42 ACTG175-calibrated application."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
import torch

from application.clinical_dgp import ClinicalAnchoredDGP
from application.mr import evaluate_fold as mr_scores
from pse.bridges import BridgeConfig, HiddenBridgeSet
from pse.estimators import proximal_scores
from pse.kernels import KernelRidge


METHODS = ("POR1", "POR2", "PH", "PIPW", "PMR", "MR")
N = 8000
BASE_SEED = 42
FOLDS = 3
PROPENSITY = 0.5
BRIDGE_CONFIG = dict(
    n_components=128, lambda_bridge=0.0001, lambda_adv=0.01, lambda_rkhs=0.0001,
    jitter=1e-10, device="cpu", dtype="float64", seed=0,
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(calibration, output_dir):
    torch.set_num_threads(2)
    calibration, output_dir = Path(calibration), Path(output_dir)
    if not calibration.is_file():
        raise FileNotFoundError(
            f"Missing {calibration}. First run: python -m application.prepare_data"
        )
    outputs = ("run_config.json", "observed_data.npz", "scores.npz", "result.json", "results.csv")
    if any((output_dir / name).exists() for name in outputs):
        raise FileExistsError("Output files already exist; choose a fresh --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = BridgeConfig(**BRIDGE_CONFIG)
    plan = dict(
        n=N, proxy_noise_variance=0.5, datasets=1, data_seed=BASE_SEED,
        split_seed=BASE_SEED, feature_seed=BASE_SEED, rf_seed=BASE_SEED, outer_folds=FOLDS,
        fold_seed_rule="base seed + 7919*(fold+1); control RF +101; eta +37",
        bridge_config=asdict(cfg), trees=100, eta_ridge=0.001, known_propensity=PROPENSITY,
        calibration_file=calibration.name,
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        target="PSE0 = nested mean psi - E[Y(0)]", units="log(1+CD4)",
        baseline="Common cross-fitted RF AIPW control score",
        interval="PMR paired score SD (ddof=1)/sqrt(n); normal 95% CI",
        scope="Real baseline/outcome calibration; generated treatment, mediators, proxies, outcome",
        mr="Observed-M1 benchmark; linear normal mediator and linear outcome working models",
    )
    save_json(output_dir / "run_config.json", plan)
    start = time.perf_counter()
    dgp = ClinicalAnchoredDGP(calibration)
    data, m1 = dgp.sample_with_m1(N, BASE_SEED)
    np.savez_compressed(output_dir / "observed_data.npz", **data)
    mr_data = {k: v if k == "x" else v.ravel() for k, v in data.items() if k in ("x", "a", "m2", "y")}
    mr_data["m1"] = m1.ravel()
    arrays = {k: np.full(N, np.nan) for k in (*METHODS, "control", "fold")}
    seen = np.zeros(N, dtype=int)
    a = data["a"].ravel()
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=BASE_SEED)
    for fold, (ti, vi) in enumerate(splitter.split(np.zeros(N), a)):
        train = {k: v[ti] for k, v in data.items()}
        evaluation = {k: v[vi] for k, v in data.items()}
        seed = BASE_SEED + 7919 * (fold + 1)
        mask0 = train["a"].ravel() == 0
        mu0 = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=max(10, math.ceil(np.sqrt(len(ti)))),
            max_features=1.0, n_jobs=1, random_state=seed + 101,
        ).fit(train["x"][mask0], train["y"].ravel()[mask0]).predict(evaluation["x"])
        control = mu0 + 2 * (1 - evaluation["a"].ravel()) * (evaluation["y"].ravel() - mu0)
        bridge = HiddenBridgeSet(replace(cfg, seed=seed)).fit(train, np.full(len(ti), PROPENSITY))
        nuisance = bridge.predict(evaluation)
        train0 = {k: v[mask0] for k, v in train.items()}
        target = bridge.predict(train0)["h1"]
        eta = KernelRidge(n_components=128, ridge=0.001, device="cpu", dtype="float64", seed=seed + 37
                          ).fit(train0["x"], target).predict(evaluation["x"])
        scores = proximal_scores(evaluation, nuisance, np.full(len(vi), PROPENSITY), eta)
        scores["MR"] = mr_scores({k: v[ti] for k, v in mr_data.items()},
                                 {k: v[vi] for k, v in mr_data.items()}, PROPENSITY)
        for key, value in {**scores, "control": control, "fold": np.full(len(vi), fold)}.items():
            value = np.asarray(value).ravel()
            if value.shape != (len(vi),) or not np.isfinite(value).all():
                raise FloatingPointError(f"Invalid held-out {key} in fold {fold}")
            arrays[key][vi] = value
        seen[vi] += 1
        print(f"Fold {fold + 1}/{FOLDS} complete; elapsed {time.perf_counter() - start:.1f}s", flush=True)
    if not np.all(seen == 1):
        raise AssertionError("Each individual must be evaluated exactly once")
    rows = []
    for method in METHODS:
        delta = arrays[method] - arrays["control"]
        arrays[method + "_delta"] = delta
        clinical = dgp.clinical_scale * delta
        row = dict(method=method, estimate=float(clinical.mean()), standardized_estimate=float(delta.mean()))
        if method == "PMR":
            se = float(clinical.std(ddof=1) / np.sqrt(N))
            row.update(se=se, ci_lower=row["estimate"] - norm.ppf(0.975) * se,
                       ci_upper=row["estimate"] + norm.ppf(0.975) * se)
        rows.append(row)
    np.savez_compressed(output_dir / "scores.npz", **arrays)
    result = dict(
        n=N, treatment_count=int(a.sum()), control_count=int((1 - a).sum()),
        units="log(1+CD4)", quantity="PSE0 paired contrast", results=rows,
        generated_population_truth=float(dgp.clinical_truth), clinical_scale=dgp.clinical_scale,
        elapsed_seconds=time.perf_counter() - start, every_individual_evaluated_once=True,
    )
    save_json(output_dir / "result.json", result)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "estimate", "standardized_estimate", "se", "ci_lower", "ci_upper"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("data/actg175_calibration.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/application"))
    args = parser.parse_args()
    run(args.calibration, args.output_dir)


if __name__ == "__main__":
    main()
