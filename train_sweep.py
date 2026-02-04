#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from experiment_config import load_config, ExperimentConfig

SCRIPT_DIR = Path(__file__).resolve().parent


def short_config_hash(cfg: ExperimentConfig) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def iter_config_files(config_dir: Path) -> List[Path]:
    return sorted([p for p in config_dir.glob("*.json") if p.is_file()])


def deterministic_parent_run_dir(cfg: ExperimentConfig, runs_root: Path, cfg_hash: str) -> Path:
    name = f"{cfg.run.experiment_name}__seed{cfg.run.seed}__{cfg_hash}"
    return runs_root / name


def is_done(parent_run_dir: Path, env_variant: str) -> bool:
    done_path = parent_run_dir / env_variant / "DONE.json"
    if not done_path.exists():
        return False
    try:
        data = json.loads(done_path.read_text(encoding="utf-8"))
        return data.get("status") == "ok"
    except Exception:
        return False


def _run_train_job(job: Tuple[str, str, str], train_py: str) -> Tuple[Tuple[str, str, str], int]:
    """
    Top-level worker for Windows pickling.
    job = (config_path_str, env_variant, parent_run_dir_str)
    """
    config_path, env_variant, parent_run_dir = job
    cmd = [
        sys.executable,
        train_py,
        "--config",
        config_path,
        "--env",
        env_variant,
        "--run-dir",
        parent_run_dir,
    ]
    res = subprocess.run(cmd)
    return job, int(res.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-dir",
        type=str,
        default=str(SCRIPT_DIR / "experiment_configs"),
        help="Folder with *.json configs",
    )
    ap.add_argument(
        "--runs-root",
        type=str,
        default=str(SCRIPT_DIR / "runs"),
        help="Runs output root folder",
    )
    ap.add_argument(
        "--train-py",
        type=str,
        default=str(SCRIPT_DIR / "train.py"),
        help="Path to train.py",
    )

    ap.add_argument("--env", choices=["2d", "3d", "both"], default="3d", help="Env variant(s) to run")
    ap.add_argument("--max-workers", type=int, default=1, help="Parallel workers")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run without launching")
    args = ap.parse_args()

    config_dir = Path(args.config_dir)
    runs_root = Path(args.runs_root)
    train_py = str(Path(args.train_py))

    configs = iter_config_files(config_dir)
    if not configs:
        print(f"No configs found in {config_dir}")
        return

    env_variants = ["2d", "3d"] if args.env == "both" else [args.env]

    jobs: List[Tuple[str, str, str]] = []
    for cfg_path in configs:
        cfg = load_config(cfg_path)
        h = short_config_hash(cfg)
        parent_run_dir = deterministic_parent_run_dir(cfg, runs_root, h)

        for ev in env_variants:
            if is_done(parent_run_dir, ev):
                print(f"SKIP (done): {cfg_path.name} [{ev}] -> {parent_run_dir}/{ev}")
                continue
            jobs.append((str(cfg_path), ev, str(parent_run_dir)))

    if not jobs:
        print("Nothing to do (all done).")
        return

    print(f"Jobs to run: {len(jobs)}")
    for cfg_path_str, ev, parent_run_dir_str in jobs:
        print(f"  RUN: {Path(cfg_path_str).name} [{ev}] -> {parent_run_dir_str}/{ev}")

    if args.dry_run:
        return

    if args.max_workers <= 1:
        failed = 0
        for job in jobs:
            (_, ev, _), rc = _run_train_job(job, train_py)
            if rc != 0:
                failed += 1
                print(f"FAILED: {Path(job[0]).name} [{ev}] (returncode={rc})")
            else:
                print(f"DONE: {Path(job[0]).name} [{ev}]")
        if failed:
            raise SystemExit(f"{failed} job(s) failed.")
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed

    failed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_run_train_job, job, train_py) for job in jobs]
        for fut in as_completed(futs):
            job, rc = fut.result()
            cfg_path_str, ev, _ = job
            if rc != 0:
                failed += 1
                print(f"FAILED: {Path(cfg_path_str).name} [{ev}] (returncode={rc})")
            else:
                print(f"DONE: {Path(cfg_path_str).name} [{ev}]")

    if failed:
        raise SystemExit(f"{failed} job(s) failed.")


if __name__ == "__main__":
    main()
