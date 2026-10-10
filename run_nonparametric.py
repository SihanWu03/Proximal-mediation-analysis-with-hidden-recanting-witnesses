"""Reproduce the final nonparametric simulation (both cases by default)."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch

from pse.bridges import BridgeConfig
from pse.estimators import EstimationConfig, estimate_dataset
from pse.simulation import NonparametricDGP


SAMPLE_SIZES = (1000, 2000, 4000, 8000)
ESTIMATORS = ("POR1", "POR2", "PH", "PIPW", "PMR", "naive")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="1,2", help="comma-separated cases: 1,2")
    parser.add_argument("--sample-sizes", default="1000,2000,4000,8000")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cpu, or auto")
    parser.add_argument("--forest-jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nonparametric"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.cases = [int(value.strip()) for value in args.cases.split(",")]
        args.sample_sizes = [int(value.strip()) for value in args.sample_sizes.split(",")]
    except ValueError:
        parser.error("cases and sample-sizes must be comma-separated integers")
    if not args.cases or any(case not in (1, 2) for case in args.cases):
        parser.error("cases must be selected from 1,2")
    if not args.sample_sizes or any(n not in SAMPLE_SIZES for n in args.sample_sizes):
        parser.error("sample-sizes must be selected from 1000,2000,4000,8000")
    if len(set(args.cases)) != len(args.cases) or len(set(args.sample_sizes)) != len(args.sample_sizes):
        parser.error("duplicate cases or sample-sizes")
    if args.repetitions < 1 or args.threads < 1 or args.forest_jobs == 0:
        parser.error("repetitions and threads must be positive; forest-jobs cannot be zero")
    return args


def dataset_seed(case, n, replicate, repetitions):
    """Original seed rule, including the case-2 offset in separate runs."""
    return 1 + (case - 1) * (4 * repetitions) + 4 * replicate + SAMPLE_SIZES.index(n)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_reports(records, out, dimension_code, sample_sizes, repetitions):
    """Report all completed datasets, using the original summary arithmetic."""
    frame = pd.DataFrame(records)
    if len(frame):
        frame = frame.sort_values(["n", "replicate"]).reset_index(drop=True)
    else:
        frame = pd.DataFrame(columns=["case", "dimensions", "n", "replicate", "seed", "model_seed"])
    point_rows, inference_rows = [], []
    for n in sample_sizes:
        cell = frame[frame["n"] == n]
        counts = dict(dimensions=dimension_code, n=n, planned=repetitions,
                      completed=len(cell), pending=repetitions - len(cell))
        for name in ESTIMATORS:
            residual = cell[f"{name}_residual"].to_numpy() if len(cell) else np.array([])
            point_rows.append({**counts, "estimator": name,
                "bias": float(residual.mean()) if len(residual) else np.nan,
                "rmse": float(np.sqrt(np.mean(residual**2))) if len(residual) else np.nan})
            if name in ("PMR", "naive"):
                coverage = float(cell[f"{name}_ci_cover"].mean()) if len(cell) else np.nan
                inference_rows.append({**counts, "estimator": name,
                    "confidence": 0.95,
                    "coverage": coverage,
                    "mean_ci_width": float(cell[f"{name}_ci_width"].mean()) if len(cell) else np.nan,
                    "mean_se": float(cell[f"{name}_se"].mean()) if len(cell) else np.nan})
    atomic_csv(frame, out / "detailed.csv")
    atomic_csv(pd.DataFrame(point_rows), out / "summary.csv")
    atomic_csv(pd.DataFrame(inference_rows), out / "inference_summary.csv")


def source_hash():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("run_nonparametric.py", "pse/simulation.py", "pse/naive.py",
                 "pse/bridges.py", "pse/kernels.py", "pse/estimators.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def run_case(args, case, bridge, estimation):
    out = args.output_dir.resolve() / f"case{case}"
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Output is locked: {lock}; check its PID before removing a stale lock") from None
    with os.fdopen(fd, "w") as handle:
        json.dump(dict(pid=os.getpid(), host=platform.node()), handle)
    try:
        return run_case_locked(args, case, bridge, estimation, out)
    finally:
        lock.unlink(missing_ok=True)


def run_case_locked(args, case, bridge, estimation, out):
    dgp = NonparametricDGP(case)
    spec = dict(case=case, dimensions=dgp.dimension_code,
                sample_sizes=args.sample_sizes, repetitions=args.repetitions, seed=1,
                seed_rule="1 + (case-1)*4*repetitions + 4*replicate + index(n in [1000,2000,4000,8000])",
                model_seed_rule="10000 + dataset_seed",
                bridge=asdict(bridge), estimation=asdict(estimation),
                dgp=dgp.config(), code_sha256=source_hash())
    scientific = json.loads(json.dumps(spec))
    scientific["bridge"].pop("device")
    scientific["estimation"].pop("forest_jobs")
    fingerprint = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    config_path = out / "run_config.json"
    if config_path.exists():
        if not args.resume:
            raise RuntimeError(f"{out} already contains a run; use --resume or a new --output-dir")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError("Resume settings or source code differ; use the original settings or a new output directory")
    else:
        if any(path.name != ".run.lock" for path in out.iterdir()):
            raise RuntimeError(f"{out} contains files without run_config.json; choose an empty directory")
        atomic_json(config_path, dict(experiment=spec, fingerprint=fingerprint,
            created_utc=datetime.now(timezone.utc).isoformat(),
            runtime=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                         sklearn=sklearn.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                         device=bridge.device, torch_threads=args.threads)))
    records = {}
    for path in sorted((out / "checkpoints").glob("n*/rep*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row["n"], row["replicate"])
        if key in records:
            raise RuntimeError(f"Duplicate checkpoint: {key}")
        records[key] = row
    total = len(args.sample_sizes) * args.repetitions
    try:
        for rep in range(args.repetitions):
            for n in args.sample_sizes:
                if (n, rep) in records:
                    continue
                seed = dataset_seed(case, n, rep, args.repetitions)
                model_seed = 10000 + seed
                started = time.perf_counter()
                data = dgp.sample(n, seed)
                values, _ = estimate_dataset(data, dgp.population_truth, bridge, estimation, seed=model_seed)
                row = dict(case=case, dimensions=dgp.dimension_code, n=n, replicate=rep,
                           seed=seed, model_seed=model_seed, true_psi=dgp.population_truth,
                           confidence=estimation.confidence, **values,
                           elapsed_seconds=time.perf_counter() - started)
                atomic_json(out / "checkpoints" / f"n{n}" / f"rep{rep:04d}.json", row)
                records[(n, rep)] = row
                print(f"case{case} [{len(records)}/{total}] n={n} rep={rep} seed={seed} "
                      f"{row['elapsed_seconds']:.1f}s", flush=True)
                if len(records) % 50 == 0:
                    write_reports(list(records.values()), out, dgp.dimension_code, args.sample_sizes, args.repetitions)
    finally:
        write_reports(list(records.values()), out, dgp.dimension_code, args.sample_sizes, args.repetitions)
    print(f"case{case}: {len(records)} datasets completed. Results: {out}", flush=True)


def run(args):
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install CUDA-enabled PyTorch or use --device cpu")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    bridge = BridgeConfig(device=device, dtype="float64", seed=1)
    estimation = EstimationConfig(forest_jobs=args.forest_jobs)
    for case in args.cases:
        run_case(args, case, bridge, estimation)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("Interrupted; completed datasets are checkpointed. Restart with --resume.", file=sys.stderr)
        raise SystemExit(130)
