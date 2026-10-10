"""Regenerate the fixed calibration from the original ACTG175 release.

Run from the repository root with ``python -m application.prepare_data``.
The bundled ``data/ACTG175.csv`` is used when present; otherwise it is downloaded.
Use ``--source ACTG175.csv`` to select another copy of the same release.
"""

import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


SOURCE_URL = "https://cran.r-universe.dev/speff2trial/data/ACTG175/csv"
SOURCE_SHA256 = "0cd9133ef7e72c60dd08bbca60ed8939d600a87f4ae4d4b9f8261d8fbd37ba5c"
USER_AGENT = "ACTG175-reproduction/1.0"
CALIBRATION_SEED = 20261005


def calibrate(source_bytes):
    """Construct empirical covariates, arm-specific means, and outcome scale."""
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"ACTG175 source checksum mismatch: expected {SOURCE_SHA256}, got {digest}. "
            "Reproduction requires the original 2026-10-05 CSV release; "
            "provide that file with --source."
        )
    frame = pd.read_csv(io.BytesIO(source_bytes))
    required = {"pidnum", "age", "wtkg", "gender", "karnof", "cd40", "preanti", "cd496", "arms"}
    absent = required.difference(frame.columns)
    if absent:
        raise ValueError(f"Full 96-week ACTG175 release required; missing {sorted(absent)}")
    if len(frame) != 2139 or not frame.pidnum.is_unique:
        raise ValueError("Unexpected participant count or duplicate IDs")
    baseline = pd.DataFrame({
        "age": frame.age, "wtkg": frame.wtkg, "gender": frame.gender,
        "karnof": frame.karnof, "log1p_cd40": np.log1p(frame.cd40),
        "log1p_preanti": np.log1p(frame.preanti),
    })
    complete_x = np.isfinite(baseline.to_numpy(float)).all(axis=1)
    pool = baseline.loc[complete_x].copy()
    center_x = pool.mean().to_numpy()
    scale_x = pool.std(ddof=0).to_numpy()
    if np.any(scale_x <= 0):
        raise ValueError("Degenerate calibration covariate")
    x_pool = (pool.to_numpy() - center_x) / scale_x
    y = np.log1p(frame.cd496)
    if (frame.cd496.dropna() < 0).any():
        raise ValueError("CD4 counts must not be negative")
    mean_predictions, residuals = [], []
    for arm in (0, 1):
        mask = complete_x & (frame.arms == arm) & np.isfinite(y)
        x = (baseline.loc[mask].to_numpy() - center_x) / scale_x
        target = y.loc[mask].to_numpy(float)
        if len(target) < 100:
            raise ValueError("Insufficient observed late outcomes for calibration")
        cfg = dict(n_estimators=200, min_samples_leaf=max(10, int(np.ceil(np.sqrt(len(target))))),
                   max_features=1.0, n_jobs=1)
        oof = np.full(len(target), np.nan)
        splitter = KFold(5, shuffle=True, random_state=CALIBRATION_SEED + arm)
        for fold, (train, test) in enumerate(splitter.split(x)):
            model = RandomForestRegressor(**cfg, random_state=CALIBRATION_SEED + arm * 100 + fold)
            model.fit(x[train], target[train])
            oof[test] = model.predict(x[test])
        residuals.extend(target - oof)
        model = RandomForestRegressor(**cfg, random_state=CALIBRATION_SEED + arm * 100 + 99)
        model.fit(x, target)
        mean_predictions.append(model.predict(x_pool))
    synthetic_variance = (-0.55 + 0.8 * (-0.5)) ** 2 * 0.5 + 0.8 ** 2 * 0.5 + 0.5
    clinical_scale = float(np.std(residuals, ddof=1)) / np.sqrt(synthetic_variance)
    return dict(
        x=x_pool, mu0=mean_predictions[0], mu1=mean_predictions[1],
        clinical_center=float(mean_predictions[0].mean()), clinical_scale=clinical_scale,
        patient_id=frame.loc[complete_x, "pidnum"].to_numpy(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Copy of the original ACTG175 CSV")
    parser.add_argument("--output", type=Path, default=Path("data/actg175_calibration.npz"))
    args = parser.parse_args()
    source = args.source
    if source is None and Path("data/ACTG175.csv").is_file():
        source = Path("data/ACTG175.csv")
    if source is None:
        print(f"Downloading {SOURCE_URL}", flush=True)
        try:
            request = Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=60) as response:
                source_bytes = response.read()
        except OSError as exc:
            status = getattr(exc, "code", None)
            detail = f" (HTTP {status})" if status else ""
            raise SystemExit(
                f"Could not download the original ACTG175 CSV{detail}. "
                "Provide an offline copy with --source /path/to/ACTG175.csv."
            ) from None
    else:
        source_bytes = source.read_bytes()
    arrays = calibrate(source_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved {args.output}; {len(arrays['x'])} baseline rows; "
          f"clinical scale={arrays['clinical_scale']:.12f}", flush=True)


if __name__ == "__main__":
    main()
