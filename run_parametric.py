"""Run the five parametric manuscript scenarios on shared simulated datasets."""
import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy

from parametric.dgp import FixedLinearDGP
from parametric.estimators import ESTIMATORS, estimate_dataset
from parametric.models import CASES, SPECIFICATIONS

SIZES = (1000, 2000, 4000, 8000)
SCENARIOS = dict(enumerate(CASES, start=1))
CONFIDENCE = 0.95


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/parametric"))
    parser.add_argument("--sample-sizes", default="1000,2000,4000,8000")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        sizes = [int(value.strip()) for value in args.sample_sizes.split(",")]
    except ValueError:
        parser.error("sample-sizes must contain integers")
    if not sizes or len(set(sizes)) != len(sizes) or not set(sizes) <= set(SIZES):
        parser.error(f"sample-sizes must be distinct members of {SIZES}")
    if min(args.repetitions, args.workers) < 1:
        parser.error("repetitions and workers must be positive")
    args.sample_sizes = sorted(sizes)
    return args


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def task(job):
    n, replicate = job
    seed = 1 + 4 * replicate + SIZES.index(n)
    dgp = FixedLinearDGP()
    data = dgp.sample(n, seed)
    rows = []
    for scenario, case in SCENARIOS.items():
        row = dict(n=n, scenario=scenario, scenario_name=case, dimensions="12211",
                   replicate=replicate, seed=seed, true_psi=dgp.population_truth,
                   confidence=CONFIDENCE, status="ok", error="")
        try:
            result, _ = estimate_dataset(data, dgp.population_truth, case, CONFIDENCE)
            row.update(result)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def write_reports(records, out, sizes, repetitions):
    columns = ["n", "scenario", "scenario_name", "dimensions", "replicate", "seed",
               "true_psi", "confidence", "status", "error"]
    frame = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
    frame = frame.sort_values(["n", "scenario", "replicate"]).reset_index(drop=True)
    points, intervals = [], []
    for n in sizes:
        for scenario, case in SCENARIOS.items():
            cell = frame[(frame.n == n) & (frame.scenario == scenario)]
            ok = cell[cell.status == "ok"]
            count = len(ok)
            metadata = dict(n=n, scenario=scenario, scenario_name=case, dimensions="12211",
                            planned=repetitions, completed=count,
                            failed=len(cell)-count, pending=repetitions-len(cell))
            for name in ESTIMATORS:
                residual = ok[f"{name}_residual"].to_numpy() if count else np.array([])
                points.append({**metadata, "estimator": name,
                    "bias": float(residual.mean()) if count else np.nan,
                    "rmse": float(np.sqrt(np.mean(residual*residual))) if count else np.nan})
            coverage = float(ok.PMR_ci_cover.mean()) if count else np.nan
            intervals.append({**metadata, "estimator": "PMR", "confidence": CONFIDENCE,
                "ci_method": "original_EIF_only", "coverage": coverage,
                "mean_se": float(ok.PMR_se.mean()) if count else np.nan,
                "mean_ci_width": float(ok.PMR_ci_width.mean()) if count else np.nan})
    for name, table in (("detailed", frame), ("summary", pd.DataFrame(points)),
                        ("inference_summary", pd.DataFrame(intervals))):
        temporary = out / f"{name}.tmp"
        table.to_csv(temporary, index=False)
        temporary.replace(out / f"{name}.csv")


def run(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "run_config.json"
    experiment = dict(sample_sizes=args.sample_sizes, repetitions=args.repetitions,
        scenarios=SCENARIOS, confidence=CONFIDENCE,
        seed_rule="1 + 4*replicate + (1000,2000,4000,8000).index(n); replicate starts at 0",
        shared_data_across_scenarios=True, dimensions=[1, 2, 2, 1, 1], truth=0.48,
        nuisance_feature_specifications=SPECIFICATIONS)
    experiment = json.loads(json.dumps(experiment))
    if config_path.exists():
        if not args.resume:
            raise ValueError("Results already exist; use --resume or choose a new output-dir")
        if json.loads(config_path.read_text(encoding="utf-8"))["experiment"] != experiment:
            raise ValueError("Resume requires the original sample sizes and repetitions")
    else:
        if any(out.iterdir()):
            raise ValueError("The output directory must be empty for a new run")
        atomic_json(config_path, dict(experiment=experiment,
            runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__)))
    checkpoints = out / "checkpoints"
    saved = {}
    for path in sorted(checkpoints.glob("n*/rep*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        saved[(rows[0]["n"], rows[0]["replicate"])] = rows
    jobs = [(n, replicate) for replicate in range(args.repetitions)
            for n in args.sample_sizes if (n, replicate) not in saved]
    total = len(args.sample_sizes) * args.repetitions
    print(f"Parametric simulation: {len(saved)}/{total} datasets saved; workers={args.workers}", flush=True)

    def receive(rows):
        n, replicate = rows[0]["n"], rows[0]["replicate"]
        atomic_json(checkpoints / f"n{n}" / f"rep{replicate:04d}.json", rows)
        saved[(n, replicate)] = rows
        if len(saved) % 50 == 0 or len(saved) == total:
            print(f"Completed {len(saved)}/{total} datasets", flush=True)

    try:
        if args.workers == 1:
            for job in jobs:
                receive(task(job))
        elif jobs:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(task, job) for job in jobs]
                for future in as_completed(futures):
                    receive(future.result())
    finally:
        records = [row for rows in saved.values() for row in rows]
        write_reports(records, out, args.sample_sizes, args.repetitions)
    failures = sum(row["status"] == "failed" for row in records)
    print(f"Saved {out}; failed fits={failures}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("Interrupted; continue the saved run with --resume.", file=sys.stderr)
        raise SystemExit(130)
