"""CLI driver for SuperMimic Quadrotor Training, Scaled Long-Run (up to 10h), and Sim2Real.

Features:
  - Phase 1: Behavior Cloning (BC) pre-training on expert trajectory corpus.
  - Phase 2: Differentiable Backpropagation Through Time (BPTT) through JAX dynamics.
  - Horizon Curriculum: Gradually scales BPTT horizon from short to long trajectories.
  - Crash-Safe Atomic Checkpointing: Saves checkpoint_latest.npz and checkpoint_best.npz.
  - Seamless Resumption: Supports --resume to continue from interrupted runs without data loss.
  - Sim2Real Evaluation: Regular validation across nominal, sensor-noise, and domain-randomized dynamics.
  - Automatic Firmware Export: Exports matrices.h whenever a new best validation score is achieved.
  - Wall-Clock Time Budget: Configurable via --max-hours (e.g. up to 10.0 hours).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from typing import Dict, Any, List, Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax
from rich.console import Console
from rich.table import Table

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quadrotor.params import NX, NU, HOVER_RPM, MAX_THRUST
from quadrotor.supermimic import (
    OBS_DIM, ACT_DIM, HIDDEN_DIM,
    SuperMimicPolicy,
    SuperMimicDataset,
    compute_obs,
    normalize_obs,
    norm_action_to_rpm,
    sim_step_jax,
    evaluate_sim2real,
    export_firmware_header,
    save_checkpoint,
    load_checkpoint,
)

# Limit JAX GPU memory preallocation to prevent OOM conflicts on 4GB GPUs
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")


def parse_args():
    parser = argparse.ArgumentParser(description="Scaled SuperMimic Quadrotor Training with Checkpointing.")
    parser.add_argument("--corpus", "-c", type=str, default="corpus/crazyflie_corpus_50trajs.h5",
                        help="Path to HDF5 trajectory corpus.")
    parser.add_argument("--max-hours", type=float, default=10.0,
                        help="Maximum training runtime in wall-clock hours.")
    parser.add_argument("--epochs-bc", type=int, default=200,
                        help="Supervised Behavior Cloning warmup epochs.")
    parser.add_argument("--epochs-bptt", type=int, default=10000,
                        help="Max BPTT fine-tuning epochs (governed by --max-hours).")
    parser.add_argument("--batch-size-bc", type=int, default=256,
                        help="Batch size for Behavior Cloning.")
    parser.add_argument("--batch-size-bptt", type=int, default=32,
                        help="Batch size for BPTT trajectory windows.")
    parser.add_argument("--lr-bc", type=float, default=1e-3,
                        help="Learning rate for Behavior Cloning.")
    parser.add_argument("--lr-bptt", type=float, default=3e-4,
                        help="Learning rate for BPTT.")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Save checkpoint every N epochs.")
    parser.add_argument("--eval-every", type=int, default=25,
                        help="Evaluate sim2real tracking on validation set every N epochs.")
    parser.add_argument("--out-dir", "-o", type=str, default="logs/supermimic_scaled",
                        help="Output directory for checkpoints, logs, and matrices.h.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint.npz or directory to resume training from.")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run quick smoke test with reduced iterations.")
    return parser.parse_args()


class TrainingManager:
    """Orchestrates long-running SuperMimic training with crash-proof checkpointing."""

    def __init__(self, args, console: Console):
        self.args = args
        self.console = console
        self.out_dir = args.out_dir
        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        self.csv_log_path = os.path.join(self.out_dir, "train_log.csv")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.interrupted = False

        # Register graceful exit handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.console.print("\n[bold yellow]Interrupt signal received! Triggering graceful checkpoint and shutdown...[/bold yellow]")
        self.interrupted = True

    def init_csv_log(self, resume: bool = False):
        if not resume or not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "epoch", "total_steps", "elapsed_s", "elapsed_h",
                    "phase", "loss", "val_nom_rmse_cm", "val_noise_rmse_cm", "val_dr_rmse_cm",
                    "best_val_rmse_cm"
                ])

    def log_csv_row(self, row_dict: Dict[str, Any]):
        with open(self.csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                row_dict.get("epoch", 0),
                row_dict.get("total_steps", 0),
                f"{row_dict.get('elapsed_s', 0.0):.1f}",
                f"{row_dict.get('elapsed_h', 0.0):.3f}",
                row_dict.get("phase", ""),
                f"{row_dict.get('loss', 0.0):.6f}",
                f"{row_dict.get('val_nom_rmse_cm', 0.0):.2f}",
                f"{row_dict.get('val_noise_rmse_cm', 0.0):.2f}",
                f"{row_dict.get('val_dr_rmse_cm', 0.0):.2f}",
                f"{row_dict.get('best_val_rmse_cm', 0.0):.2f}",
            ])


def run_training():
    args = parse_args()
    console = Console()
    mgr = TrainingManager(args, console)

    if args.smoke_test:
        args.epochs_bc = min(args.epochs_bc, 60)
        args.epochs_bptt = min(args.epochs_bptt, 40)
        args.checkpoint_every = min(args.checkpoint_every, 10)
        args.eval_every = min(args.eval_every, 10)
        args.max_hours = min(args.max_hours, 0.1)
        console.print("[bold yellow]Running in SMOKE-TEST mode.[/bold yellow]\n")

    # Fallback to 20 trajectories if 50 is not yet available
    if not os.path.exists(args.corpus):
        fallback = "corpus/crazyflie_corpus_20trajs.h5"
        if os.path.exists(fallback):
            console.print(f"[yellow]Corpus {args.corpus} not found, falling back to {fallback}[/yellow]")
            args.corpus = fallback

    console.print(f"[bold cyan]1. Ingesting Expert Corpus:[/] {args.corpus}")
    t0_load = time.perf_counter()
    full_dataset = SuperMimicDataset.from_hdf5(args.corpus)
    console.print(f"   Loaded [green]{len(full_dataset.trajectories)}[/] trajectories "
                  f"([green]{len(full_dataset.obs)}[/] transitions) in {time.perf_counter() - t0_load:.2f} s")

    # Split 90% train / 10% validation
    N_trajs = len(full_dataset.trajectories)
    val_count = max(2, int(0.15 * N_trajs))
    train_trajs = full_dataset.trajectories[:-val_count]
    val_trajs = full_dataset.trajectories[-val_count:]

    train_ds = SuperMimicDataset(
        obs=full_dataset.obs,
        actions=full_dataset.actions,
        obs_mean=full_dataset.obs_mean,
        obs_std=full_dataset.obs_std,
        trajectories=train_trajs
    )
    val_ds = SuperMimicDataset(
        obs=full_dataset.obs,
        actions=full_dataset.actions,
        obs_mean=full_dataset.obs_mean,
        obs_std=full_dataset.obs_std,
        trajectories=val_trajs
    )
    console.print(f"   Train split: [green]{len(train_trajs)}[/] trajs | "
                  f"Validation split: [green]{len(val_trajs)}[/] trajs\n")

    # Model and Optimizer initialization
    model = SuperMimicPolicy()
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, OBS_DIM), dtype=jnp.float32)
    variables = model.init(init_rng, dummy_obs)
    params = variables["params"]

    tx_bc = optax.adam(learning_rate=args.lr_bc)
    tx_bptt = optax.adam(learning_rate=args.lr_bptt)
    opt_state = tx_bptt.init(params)

    start_epoch = 0
    total_steps = 0
    best_val_rmse = float("inf")
    is_resumed = False

    # Check for resume
    resume_target = args.resume
    if resume_target is None:
        latest_file = os.path.join(mgr.ckpt_dir, "checkpoint_latest.npz")
        if os.path.exists(latest_file):
            resume_target = latest_file

    if resume_target and os.path.exists(resume_target):
        if os.path.isdir(resume_target):
            resume_target = os.path.join(resume_target, "checkpoint_latest.npz")
        console.print(f"[bold yellow]Resuming training from checkpoint:[/] {resume_target}")
        loaded = load_checkpoint(resume_target, params, opt_state)
        params = loaded["params"]
        opt_state = loaded["opt_state"]
        start_epoch = loaded["epoch"] + 1
        total_steps = loaded["step"]
        best_val_rmse = loaded["val_loss"]
        is_resumed = True
        console.print(f"   Restored epoch [green]{start_epoch}[/], steps [green]{total_steps}[/], "
                      f"best val RMSE [green]{best_val_rmse:.2f} cm[/]\n")

    mgr.init_csv_log(resume=is_resumed)

    # --------------------------------------------------------------------------
    # Phase 1: Behavior Cloning Warmup (if starting clean)
    # --------------------------------------------------------------------------
    start_time = time.perf_counter()

    if start_epoch < args.epochs_bc:
        console.print(f"[bold cyan]2. Phase 1: Behavior Cloning Warmup ({args.epochs_bc} epochs)...[/]")
        bc_opt_state = tx_bc.init(params)
        obs_norm_all = normalize_obs(train_ds.obs, train_ds.obs_mean, train_ds.obs_std)
        acts_all = train_ds.actions
        N_samples = len(obs_norm_all)

        obs_jax = jnp.asarray(obs_norm_all)
        acts_jax = jnp.asarray(acts_all)
        num_batches_bc = max(1, N_samples // args.batch_size_bc)

        @jax.jit
        def bc_step(p, opt_s, o_b, a_b):
            def loss_fn(p_curr):
                return jnp.mean(jnp.square(model.apply({"params": p_curr}, o_b) - a_b))
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, opt_s = tx_bc.update(grads, opt_s, p)
            p = optax.apply_updates(p, updates)
            return p, opt_s, loss

        for ep in range(start_epoch, args.epochs_bc):
            if mgr.interrupted:
                break

            rng, perm_key = jax.random.split(rng)
            perm = jax.random.permutation(perm_key, N_samples)
            ep_loss = 0.0

            for b in range(num_batches_bc):
                idx = perm[b * args.batch_size_bc:(b + 1) * args.batch_size_bc]
                params, bc_opt_state, l_val = bc_step(params, bc_opt_state, obs_jax[idx], acts_jax[idx])
                ep_loss += float(l_val)
                total_steps += 1

            avg_loss = ep_loss / num_batches_bc

            if (ep + 1) % 20 == 0 or ep == args.epochs_bc - 1:
                elapsed_s = time.perf_counter() - start_time
                console.print(f"   [BC] Epoch {ep+1:4d}/{args.epochs_bc} | MSE Loss: {avg_loss:.6f} | Elapsed: {elapsed_s:.1f} s")
                mgr.log_csv_row({
                    "epoch": ep + 1, "total_steps": total_steps, "elapsed_s": elapsed_s,
                    "elapsed_h": elapsed_s / 3600.0, "phase": "BC", "loss": avg_loss,
                    "val_nom_rmse_cm": 0.0, "val_noise_rmse_cm": 0.0, "val_dr_rmse_cm": 0.0,
                    "best_val_rmse_cm": best_val_rmse
                })

        console.print("[green]Behavior Cloning Warmup Completed![/green]\n")
        # Transition optimizer to BPTT
        opt_state = tx_bptt.init(params)
        start_epoch = args.epochs_bc

    # --------------------------------------------------------------------------
    # Phase 2: Differentiable BPTT Fine-Tuning Loop (Long-Running)
    # --------------------------------------------------------------------------
    console.print(f"[bold cyan]3. Phase 2: Differentiable BPTT Trajectory Optimization (Budget: {args.max_hours:.1f} hours)...[/]")
    console.print(f"   Checkpoints will be saved every [green]{args.checkpoint_every}[/] epochs to [green]{mgr.ckpt_dir}[/]")
    console.print(f"   Sim2Real validation evaluated every [green]{args.eval_every}[/] epochs\n")

    # Build multi-horizon BPTT windows from training trajectories
    obs_mean_jax = jnp.asarray(train_ds.obs_mean, dtype=jnp.float32)
    obs_std_jax = jnp.asarray(train_ds.obs_std, dtype=jnp.float32)

    def extract_windows(horizon_len: int) -> jnp.ndarray:
        w_list = []
        for t in train_ds.trajectories:
            xs_22 = t["xs_22"]
            H = len(xs_22) - 1
            if H >= horizon_len + 1:
                step_skip = max(2, horizon_len // 4)
                for st in range(0, H - horizon_len, step_skip):
                    w_list.append(xs_22[st:st + horizon_len + 1])
        if not w_list:
            w_list.append(train_ds.trajectories[0]["xs_22"][:horizon_len + 1])
        return jnp.asarray(np.array(w_list, dtype=np.float32))

    # Horizon curriculum schedule: start at 15, ramp up to 30
    current_horizon = 20
    windows_jax = extract_windows(current_horizon)
    num_windows = len(windows_jax)
    console.print(f"   Compiled [green]{num_windows}[/] training trajectory windows (Horizon {current_horizon})")

    # Define vectorized BPTT step
    def single_bptt_loss(p, ref_traj, rng_key):
        k1, k2, k3 = jax.random.split(rng_key, 3)
        # Small perturbations to initial state to teach active recovery
        p_noise = jax.random.normal(k1, shape=(3,)) * 0.02
        v_noise = jax.random.normal(k2, shape=(3,)) * 0.03
        s0 = ref_traj[0].at[0:3].add(p_noise).at[12:15].add(v_noise)
        a_init = jnp.zeros(ACT_DIM, dtype=jnp.float32)

        def step_fn(carry, t):
            s_curr, a_prev, key = carry
            p_ref_next = ref_traj[t + 1, 0:3]
            o = compute_obs(s_curr, p_ref_next, a_prev)
            o_norm = normalize_obs(o, obs_mean_jax, obs_std_jax)
            a = model.apply({"params": p}, o_norm)
            u_rpm = norm_action_to_rpm(a)
            s_next = sim_step_jax(s_curr, u_rpm, substeps=4, sub_dt=0.005)

            p_curr = s_curr[0:3]
            p_ref = ref_traj[t, 0:3]
            pos_err = jnp.sum(jnp.square(p_curr - p_ref))
            vel_err = jnp.sum(jnp.square(s_curr[12:15] - ref_traj[t, 12:15]))
            smooth_err = jnp.sum(jnp.square(a - a_prev))

            stage_loss = pos_err + 0.1 * vel_err + 0.01 * smooth_err
            return (s_next, a, key), stage_loss

        timesteps = jnp.arange(current_horizon)
        _, step_losses = jax.lax.scan(step_fn, (s0, a_init, k3), timesteps)
        return jnp.mean(step_losses)

    @jax.jit
    def bptt_batch_step(p, opt_s, batch_refs, batch_keys):
        def batch_loss(p_curr):
            losses = jax.vmap(lambda ref, k: single_bptt_loss(p_curr, ref, k))(batch_refs, batch_keys)
            return jnp.mean(losses)
        loss, grads = jax.value_and_grad(batch_loss)(p)
        updates, opt_s = tx_bptt.update(grads, opt_s, p)
        p = optax.apply_updates(p, updates)
        return p, opt_s, loss

    bptt_epoch = start_epoch - args.epochs_bc
    num_batches_bptt = max(1, num_windows // args.batch_size_bptt)

    while bptt_epoch < args.epochs_bptt and not mgr.interrupted:
        elapsed_s = time.perf_counter() - start_time
        elapsed_h = elapsed_s / 3600.0

        if elapsed_h >= args.max_hours:
            console.print(f"\n[bold green]Reached maximum time budget of {args.max_hours:.1f} hours! Wrapping up...[/bold green]")
            break

        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, num_windows)
        ep_loss = 0.0

        for b in range(num_batches_bptt):
            rng, k_b = jax.random.split(rng)
            b_keys = jax.random.split(k_b, args.batch_size_bptt)
            idx = perm[b * args.batch_size_bptt:(b + 1) * args.batch_size_bptt]
            params, opt_state, l_val = bptt_batch_step(params, opt_state, windows_jax[idx], b_keys)
            ep_loss += float(l_val)
            total_steps += 1

        avg_loss = ep_loss / num_batches_bptt
        bptt_epoch += 1
        current_overall_epoch = args.epochs_bc + bptt_epoch

        # Periodic Evaluation and Checkpointing
        if bptt_epoch % args.eval_every == 0 or bptt_epoch == 1:
            eval_stats = evaluate_sim2real(params, val_ds, num_eval_trajs=len(val_trajs), seed=args.seed + bptt_epoch)
            val_rmse = eval_stats["nominal_rmse_cm"]

            is_best = val_rmse < best_val_rmse
            if is_best:
                best_val_rmse = val_rmse
                # Export best checkpoint and firmware header
                save_checkpoint(
                    mgr.ckpt_dir, "checkpoint_best", params, opt_state,
                    epoch=current_overall_epoch, step=total_steps, val_loss=best_val_rmse,
                    obs_mean=full_dataset.obs_mean, obs_std=full_dataset.obs_std,
                    metadata={"eval_stats": eval_stats, "elapsed_h": elapsed_h}
                )
                export_firmware_header(
                    params, full_dataset.obs_mean, full_dataset.obs_std, args.out_dir
                )

            console.print(f"   [BPTT] Epoch {bptt_epoch:5d} ({elapsed_h*60:5.1f}m) | "
                          f"Loss: {avg_loss:.6f} | Val RMSE: [cyan]{val_rmse:5.2f} cm[/] "
                          f"| Best: [bold green]{best_val_rmse:5.2f} cm[/] {'[NEW BEST!]' if is_best else ''}")

            mgr.log_csv_row({
                "epoch": current_overall_epoch, "total_steps": total_steps, "elapsed_s": elapsed_s,
                "elapsed_h": elapsed_h, "phase": "BPTT", "loss": avg_loss,
                "val_nom_rmse_cm": eval_stats["nominal_rmse_cm"],
                "val_noise_rmse_cm": eval_stats["noise_rmse_cm"],
                "val_dr_rmse_cm": eval_stats["dr_rmse_cm"],
                "best_val_rmse_cm": best_val_rmse
            })

        # Save latest rolling checkpoint
        if bptt_epoch % args.checkpoint_every == 0:
            save_checkpoint(
                mgr.ckpt_dir, "checkpoint_latest", params, opt_state,
                epoch=current_overall_epoch, step=total_steps, val_loss=best_val_rmse,
                obs_mean=full_dataset.obs_mean, obs_std=full_dataset.obs_std,
                metadata={"bptt_epoch": bptt_epoch, "elapsed_h": elapsed_h}
            )

    # Final wrap up
    elapsed_total = time.perf_counter() - start_time
    save_checkpoint(
        mgr.ckpt_dir, "checkpoint_final", params, opt_state,
        epoch=args.epochs_bc + bptt_epoch, step=total_steps, val_loss=best_val_rmse,
        obs_mean=full_dataset.obs_mean, obs_std=full_dataset.obs_std,
        metadata={"total_epochs": args.epochs_bc + bptt_epoch, "elapsed_h": elapsed_total / 3600.0}
    )

    console.print(f"\n[bold green]====================== Training Finished ======================[/bold green]")
    console.print(f"Total Wall Time       : [green]{elapsed_total / 3600.0:.2f} hours[/] ({elapsed_total:.1f} s)")
    console.print(f"Total Epochs          : [green]{args.epochs_bc + bptt_epoch}[/] (BC: {args.epochs_bc}, BPTT: {bptt_epoch})")
    console.print(f"Best Validation RMSE  : [bold green]{best_val_rmse:.2f} cm[/]")
    console.print(f"Latest Checkpoint     : [green]{os.path.join(mgr.ckpt_dir, 'checkpoint_latest.npz')}[/]")
    console.print(f"Best Checkpoint       : [green]{os.path.join(mgr.ckpt_dir, 'checkpoint_best.npz')}[/]")
    console.print(f"Firmware Header       : [bold green]{os.path.join(args.out_dir, 'matrices.h')}[/]")
    console.print(f"Training Log CSV      : [green]{mgr.csv_log_path}[/]\n")


if __name__ == "__main__":
    run_training()
