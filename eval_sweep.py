#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent


def iter_checkpoints(runs_root: Path, env_filter: str) -> List[Tuple[Path, str]]:
    """
    Returns list of (ckpt_path, env_variant).
    Expected layout:
      runs/<parent>/<2d|3d>/checkpoints/*.pt
    """
    out: List[Tuple[Path, str]] = []
    env_variants = ["2d", "3d"] if env_filter == "both" else [env_filter]
    for env_variant in env_variants:
        for ckpt in runs_root.glob(f"*/{env_variant}/checkpoints/*.pt"):
            out.append((ckpt, env_variant))
    out.sort(key=lambda x: str(x[0]))
    return out


def is_eval_done(ckpt_path: Path) -> bool:
    run_dir = ckpt_path.parent.parent
    out_dir = run_dir / "eval" / ckpt_path.stem
    return (out_dir / "summary.json").exists()


def _run_eval_job(
    job: Tuple[str, str],
    eval_py: str,
    seeds: str,
    tree_state: str,
    max_steps: int | None,
) -> Tuple[Tuple[str, str], int]:
    """
    Top-level worker for Windows pickling.
    job = (ckpt_path_str, env_variant)
    """
    ckpt_path_str, env_variant = job
    cmd = [
        sys.executable,
        eval_py,
        "--ckpt",
        ckpt_path_str,
        "--env",
        env_variant,
        "--seeds",
        seeds,
        "--tree-state",
        tree_state,
    ]
    if max_steps is not None:
        cmd += ["--max-steps", str(int(max_steps))]

    res = subprocess.run(cmd)
    return job, int(res.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default=str(SCRIPT_DIR / "runs"), help="Runs output root folder")
    ap.add_argument("--eval-py", type=str, default=str(SCRIPT_DIR / "eval.py"), help="Path to eval.py")

    ap.add_argument("--env", choices=["2d", "3d", "both"], default="both", help="Which env checkpoints to evaluate")
    ap.add_argument("--every-n", type=int, default=2, help="Evaluate every Nth checkpoint (2 = every second)")
    ap.add_argument("--max-workers", type=int, default=4, help="Parallel workers")

    ap.add_argument("--seeds", type=str, default="1000:1049", help="Seed range 'a:b' inclusive, or comma list")
    ap.add_argument("--tree-state", choices=["full", "agentpos", "none"], default="full")
    ap.add_argument("--max-steps", type=int, default=None, help="Override max steps per episode")

    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    eval_py = str(Path(args.eval_py))

    all_ckpts = iter_checkpoints(runs_root, args.env)
    if not all_ckpts:
        print(f"No checkpoints found under {runs_root} (env={args.env})")
        return

    every_n = max(1, int(args.every_n))

    # group checkpoints by run_dir
    by_run = defaultdict(list)
    for ckpt_path, env_variant in all_ckpts:
        run_dir = ckpt_path.parent.parent  # .../<run>/<env>
        by_run[(run_dir, env_variant)].append(ckpt_path)

    selected = []

    for (_run_dir, env_variant), ckpts in by_run.items():
        # ckpts are already sorted
        n = len(ckpts)
        idxs = list(range(n - 1, -1, -every_n))  # always include last
        idxs = sorted(idxs)
        for i in idxs:
            selected.append((ckpts[i], env_variant))

    jobs: List[Tuple[str, str]] = []
    for ckpt_path, env_variant in selected:
        if is_eval_done(ckpt_path):
            print(f"SKIP (done): {ckpt_path}")
            continue
        jobs.append((str(ckpt_path), env_variant))

    if not jobs:
        print("Nothing to do (all selected checkpoints already evaluated).")
        return

    print(f"Jobs to run: {len(jobs)} (selected {len(selected)} / total {len(all_ckpts)}, every_n={every_n}, env={args.env})")
    for ckpt_path_str, env_variant in jobs[:10]:
        print(f"  RUN: {ckpt_path_str} [{env_variant}]")
    if len(jobs) > 10:
        print(f"  ... and {len(jobs) - 10} more")

    if args.dry_run:
        return

    if args.max_workers <= 1:
        failed = 0
        for job in jobs:
            (_, env_variant), rc = _run_eval_job(job, eval_py, args.seeds, args.tree_state, args.max_steps)
            if rc != 0:
                failed += 1
                print(f"FAILED: {job[0]} (returncode={rc})")
            else:
                print(f"DONE: {job[0]}")
        if failed:
            raise SystemExit(f"{failed} job(s) failed.")
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed

    failed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [
            ex.submit(_run_eval_job, job, eval_py, args.seeds, args.tree_state, args.max_steps)
            for job in jobs
        ]
        for fut in as_completed(futs):
            job, rc = fut.result()
            ckpt_path_str, _env_variant = job
            if rc != 0:
                failed += 1
                print(f"FAILED: {ckpt_path_str} (returncode={rc})")
            else:
                print(f"DONE: {ckpt_path_str}")

    if failed:
        raise SystemExit(f"{failed} job(s) failed.")


if __name__ == "__main__":
    main()
