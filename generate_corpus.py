"""Generate an expert-trajectory corpus for the Bitcraze Crazyflie quadrotor using SSQP.

Features:
  - Parallel generation across CPU cores using multiprocessing
  - Multiple task families: point-to-point, figure-8, helix, hover
  - Rigorous validation gating (convergence, bounds, residual, sim replay)
  - Chunked HDF5 / NPZ storage format
  - CLI with --dry-run option to inspect accept rates and solve times

Usage:
  uv run python generate_corpus.py --dry-run
  uv run python generate_corpus.py --num-trajectories 100 --workers 4 --format h5
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

# Configure JAX before importing any jax modules: CPU mode avoids CUDA fork/VRAM conflicts
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quadrotor.params import NX, NU, SIM_DT
from quadrotor.sampler import sample_task, TaskType
from quadrotor.solver import solve_task

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _worker_generate_single(args_tuple) -> Dict[str, Any]:
    """Worker function for multiprocessing pool."""
    task_idx, seed, task_type_str, dt = args_tuple
    rng = np.random.default_rng(seed)
    task_type = TaskType(task_type_str)

    task = sample_task(task_type=task_type, rng=rng, dt=dt)
    res = solve_task(task, max_iter=100, tolerance=1e-5, use_analytic=True)

    val = res["validation"]
    summary = {
        "task_idx": task_idx,
        "seed": seed,
        "task_type": task_type_str,
        "success": res["success"],
        "solve_time_s": res["solve_time_s"],
        "iterations": res["iterations"],
        "solver_state": res["solver_state"],
        "max_residual": val["max_residual"],
        "jax_pos_drift": val["jax_pos_drift"],
        "max_track_err": val["max_track_err"],
        "final_track_err": val["final_track_err"],
        "max_tilt_deg": val["max_tilt_deg"],
        "T": task.T,
        "H": task.H,
        "dt": dt,
        "xs": res["xs"] if res["success"] else None,
        "us": res["us"] if res["success"] else None,
        "metadata": json.dumps(task.metadata),
    }
    return summary


def save_corpus_h5(filepath: str, successful_trajs: List[Dict[str, Any]], rejected_count: int):
    """Save batch of successful trajectories to a single chunked HDF5 file."""
    if not HAS_H5PY:
        raise RuntimeError("h5py is not installed. Install h5py or use --format npz.")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    N = len(successful_trajs)
    if N == 0:
        print("[Warning] No successful trajectories to save.")
        return

    # Check if horizons are equal or variable
    max_H = max(t["H"] for t in successful_trajs)
    all_same_H = all(t["H"] == max_H for t in successful_trajs)

    with h5py.File(filepath, "w") as f:
        f.attrs["num_trajectories"] = N
        f.attrs["num_rejected"] = rejected_count
        f.attrs["accept_rate"] = N / (N + rejected_count) if (N + rejected_count) > 0 else 0.0
        f.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if all_same_H:
            # Fixed horizon: store as 3D dense arrays (N, H+1, 17) and (N, H, 4)
            xs_arr = np.array([t["xs"] for t in successful_trajs], dtype=np.float32)
            us_arr = np.array([t["us"] for t in successful_trajs], dtype=np.float32)
            f.create_dataset("states", data=xs_arr, compression="gzip", chunks=True)
            f.create_dataset("controls", data=us_arr, compression="gzip", chunks=True)
        else:
            # Variable horizon: store individual trajectory groups
            traj_grp = f.create_group("trajectories")
            for i, t in enumerate(successful_trajs):
                g = traj_grp.create_group(f"traj_{i:06d}")
                g.create_dataset("states", data=t["xs"].astype(np.float32), compression="gzip")
                g.create_dataset("controls", data=t["us"].astype(np.float32), compression="gzip")
                g.attrs["task_type"] = t["task_type"]
                g.attrs["T"] = t["T"]
                g.attrs["dt"] = t["dt"]
                g.attrs["seed"] = t["seed"]
                g.attrs["metadata"] = t["metadata"]

        # Metadata table
        f.create_dataset("task_types", data=np.array([t["task_type"] for t in successful_trajs], dtype="S32"))
        f.create_dataset("seeds", data=np.array([t["seed"] for t in successful_trajs], dtype=np.int64))
        f.create_dataset("solve_times", data=np.array([t["solve_time_s"] for t in successful_trajs], dtype=np.float32))
        f.create_dataset("iterations", data=np.array([t["iterations"] for t in successful_trajs], dtype=np.int32))
        f.create_dataset("max_residuals", data=np.array([t["max_residual"] for t in successful_trajs], dtype=np.float32))
        f.create_dataset("jax_drifts", data=np.array([t["jax_pos_drift"] for t in successful_trajs], dtype=np.float32))

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved HDF5 corpus to: {filepath} ({size_mb:.2f} MB)")


def save_corpus_npz(out_dir: str, successful_trajs: List[Dict[str, Any]]):
    """Save trajectories as individual compressed NPZ files."""
    os.makedirs(out_dir, exist_ok=True)
    for i, t in enumerate(successful_trajs):
        fname = f"{t['task_type']}_{t['seed']}_{i:05d}.npz"
        path = os.path.join(out_dir, fname)
        np.savez_compressed(
            path,
            xs=t["xs"].astype(np.float32),
            us=t["us"].astype(np.float32),
            dt=t["dt"],
            T=t["T"],
            task_type=t["task_type"],
            seed=t["seed"],
            iterations=t["iterations"],
            solve_time_s=t["solve_time_s"],
            max_residual=t["max_residual"],
            jax_pos_drift=t["jax_pos_drift"],
            metadata=t["metadata"]
        )
    print(f"Saved {len(successful_trajs)} NPZ trajectories to: {out_dir}")


def generate_corpus(num_trajectories: int = 10,
                    tasks: Optional[List[str]] = None,
                    workers: int = 4,
                    dt: float = 0.02,
                    out_dir: str = "./corpus",
                    output_format: str = "h5",
                    seed: int = 42,
                    dry_run: bool = False):
    """Generate, validate, and store a batch of expert quadrotor trajectories."""
    if dry_run:
        num_trajectories = min(num_trajectories, 10)
        print(f"\n========================================================")
        print(f"  DRY RUN: Generating {num_trajectories} sample trajectories")
        print(f"========================================================\n")

    if tasks is None:
        tasks = ["point_to_point", "figure8", "helix", "hover"]

    rng = np.random.default_rng(seed)
    task_types = rng.choice(tasks, size=num_trajectories)
    seeds = rng.integers(0, 1000000, size=num_trajectories)

    job_args = [
        (i, int(seeds[i]), str(task_types[i]), dt)
        for i in range(num_trajectories)
    ]

    t0 = time.perf_counter()
    results = []

    # Choose pool or sequential
    if workers > 1 and num_trajectories > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        print(f"Running across {workers} worker processes (spawn)...")
        with ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(_worker_generate_single, job_args):
                results.append(res)
                print(f"  [{len(results):03d}/{num_trajectories}] Task: {res['task_type']:15s} | "
                      f"State: {res['solver_state']:8s} | Pass: {res['success']} | "
                      f"Solve: {res['solve_time_s']*1e3:5.1f} ms | Iters: {res['iterations']:2d}")
    else:
        print("Running sequentially...")
        for arg in job_args:
            res = _worker_generate_single(arg)
            results.append(res)
            print(f"  [{len(results):03d}/{num_trajectories}] Task: {res['task_type']:15s} | "
                  f"State: {res['solver_state']:8s} | Pass: {res['success']} | "
                  f"Solve: {res['solve_time_s']*1e3:5.1f} ms | Iters: {res['iterations']:2d}")

    total_time = time.perf_counter() - t0
    successful = [r for r in results if r["success"]]
    rejected = [r for r in results if not r["success"]]
    accept_rate = len(successful) / len(results) if results else 0.0

    print(f"\n====================== Summary ======================")
    print(f"Total tasks attempted  : {len(results)}")
    print(f"Accepted trajectories  : {len(successful)} ({accept_rate * 100:.1f}%)")
    print(f"Rejected trajectories  : {len(rejected)}")
    print(f"Total time             : {total_time:.2f} s ({total_time / len(results):.3f} s/task)")

    if successful:
        solve_times = [r["solve_time_s"] * 1e3 for r in successful]
        residuals = [r["max_residual"] for r in successful]
        jax_drifts = [r["jax_pos_drift"] * 100 for r in successful] # cm
        print(f"Mean solve time (pass) : {np.mean(solve_times):.1f} ms (min {np.min(solve_times):.1f}, max {np.max(solve_times):.1f})")
        print(f"Max dynamics residual  : {np.max(residuals):.2e}")
        print(f"Max JAX sim drift      : {np.max(jax_drifts):.2f} cm (mean {np.mean(jax_drifts):.2f} cm)")

    if not dry_run and successful:
        if output_format == "h5" and HAS_H5PY:
            h5_path = os.path.join(out_dir, f"crazyflie_corpus_{len(successful)}trajs.h5")
            save_corpus_h5(h5_path, successful, len(rejected))
        else:
            save_corpus_npz(out_dir, successful)

    return {
        "total": len(results),
        "successful": len(successful),
        "accept_rate": accept_rate,
        "total_time_s": total_time,
        "results": results
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate expert quadrotor trajectory corpus with SSQP.")
    parser.add_argument("--num-trajectories", "-n", type=int, default=10, help="Number of trajectories to generate")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--out-dir", "-o", type=str, default="./corpus", help="Output directory")
    parser.add_argument("--format", "-f", type=str, choices=["h5", "npz"], default="h5" if HAS_H5PY else "npz")
    parser.add_argument("--dt", type=float, default=0.02, help="OCP timestep (s)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--tasks", type=str, default="point_to_point,figure8,helix,hover",
                        help="Comma-separated list of tasks")
    parser.add_argument("--dry-run", action="store_true", help="Quick 10-trajectory test run with accept rate report")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    generate_corpus(
        num_trajectories=args.num_trajectories,
        tasks=tasks_list,
        workers=args.workers,
        dt=args.dt,
        out_dir=args.out_dir,
        output_format=args.format,
        seed=args.seed,
        dry_run=args.dry_run,
    )
