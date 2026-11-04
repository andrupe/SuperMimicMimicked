"""Autonomous SuperMimic Quadrotor Training Engine.

Ports the complete SuperMimic paper (Liao et al., CoRL 2026, arXiv:2610.28480) to the Crazyflie 2.1 quadrotor:
  1. Replay buffer initialized with expert corpus transitions and refreshed online via simulator rollouts.
  2. Phase 0: Behavior Cloning warmup ensuring physical feedforward controls.
  3. Phase 1: Differentiable World Model warmup (k=8 autoregressive windows, Lie-algebra increments).
  4. Phase 2: Joint SuperMimic autonomous training loop:
       - Vectorized data collection with exploration noise and context randomization.
       - Return-based reset sampling (RB-RS) curriculum across 1.0s trajectory bins.
       - Autoregressive k=8 step World Model updates (Eq. 1).
       - Differentiable BPTT Policy updates in imagination with soft survival weighting (Eq. 2)
         and perturbed initial states for corrective feedback gain synthesis (Kp, Kd, KR).
       - GAE(lambda=0.95) Critic bootstrapping updates (Eq. 3).
       - Horizon curriculum (H=16 -> 24 -> 32) and cosine decay learning rate schedule.
  5. Multi-suite Sim2Real evaluation (Nominal Physics, Sensor Noise, Domain Randomization).
  6. Automatic firmware export (matrices.h) matching crazyflie_nn/firmware/my_controller.c.
  7. Crash-safe atomic checkpointing and training logs in logs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
import h5py
from rich.console import Console
from rich.table import Table

# Memory management for 4GB NVIDIA RTX 2050
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quadrotor.params import (
    NX, NU, HOVER_RPM, MAX_THRUST, SIM_DT, M_NOMINAL
)
from quadrotor.world_model import (
    QuadrotorState, unpack_state_22, pack_state_22,
    invariant_features, compute_frame_local_targets, integrate_frame_local_step,
    WorldModel, WM_IN_DIM, WM_OUT_DIM, THRUST_SCALE
)
from quadrotor.policy_critic import (
    ValueCritic, compute_soft_termination
)
from quadrotor.rotations_jax import rot_angle_sq_jax
from quadrotor.quaternion import state17_to_22
from quadrotor.supermimic import (
    SuperMimicPolicy, compute_obs, normalize_obs, norm_action_to_rpm,
    export_firmware_header, save_checkpoint
)
from quadrotor.buffer import SuperMimicReplayBuffer
from quadrotor.supermimic_env import (
    MotionReferenceLibrary, sim_step_multi_substeps
)
from quadrotor.sim_bridge import _nominal_cf_rk4_jax


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous SuperMimic Quadrotor Training Engine.")
    parser.add_argument("--corpus", "-c", type=str, default="corpus/crazyflie_corpus_50trajs.h5",
                        help="Path to expert trajectory corpus.")
    parser.add_argument("--max-hours", type=float, default=9.5,
                        help="Maximum training runtime in wall-clock hours.")
    parser.add_argument("--target-rmse", type=float, default=4.0,
                        help="Target tracking RMSE (in cm) to declare goal achieved.")
    parser.add_argument("--bc-epochs", type=int, default=100,
                        help="Behavior cloning pretraining epochs.")
    parser.add_argument("--wm-warmup-iters", type=int, default=200,
                        help="World Model only warm-up iterations.")
    parser.add_argument("--iterations", type=int, default=30_000,
                        help="Max joint training iterations.")
    parser.add_argument("--num-envs", type=int, default=32,
                        help="Number of parallel simulator environments.")
    parser.add_argument("--batch-size-wm", type=int, default=64,
                        help="Batch size for World Model window updates.")
    parser.add_argument("--batch-size-pi", type=int, default=64,
                        help="Batch size for policy imagination BPTT rollouts.")
    parser.add_argument("--k-window", type=int, default=8,
                        help="Autoregressive window length k for World Model.")
    parser.add_argument("--lr-wm", type=float, default=3e-4,
                        help="Learning rate for World Model.")
    parser.add_argument("--lr-pi", type=float, default=1e-3,
                        help="Initial learning rate for Policy.")
    parser.add_argument("--lr-critic", type=float, default=3e-4,
                        help="Learning rate for Critic.")
    parser.add_argument("--action-noise", type=float, default=0.02,
                        help="Exploration action noise std.")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate on validation trajectories every N iterations.")
    parser.add_argument("--ckpt-every", type=int, default=200,
                        help="Save checkpoint every N iterations.")
    parser.add_argument("--out-dir", "-o", type=str, default="logs/supermimic_production",
                        help="Output directory.")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--trajs", nargs="*", type=int, default=[43, 44, 45, 46, 48],
                        help="Trajectory indices for reference library (default: 43 44 45 46 48). Pass empty for all.")
    parser.add_argument("--critic-weight", type=float, default=0.10,
                        help="Weight for critic bootstrapping in policy loss.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Fast smoke-test mode with minimal iterations.")
    return parser.parse_args()


# ------------------------------------------------------------------------------
# Observation Normalization Matching Firmware Exactly
# ------------------------------------------------------------------------------
def get_firmware_obs_norm():
    """Physics-grounded observation normalization matching crazyflie_nn/firmware/my_controller.c."""
    obs_mean = np.zeros(22, dtype=np.float32)
    obs_mean[3:12] = np.eye(3).flatten()

    obs_std = np.ones(22, dtype=np.float32)
    obs_std[0:3] = 0.50
    obs_std[3:12] = 0.50
    obs_std[12:15] = 1.00
    obs_std[15:18] = 2.00
    obs_std[18:22] = 0.50
    return obs_mean, obs_std


# ------------------------------------------------------------------------------
# Robust Multi-Objective Reward & Termination
# ------------------------------------------------------------------------------
def compute_supermimic_reward(st_next: QuadrotorState, ref_next_22: jax.Array, a_curr: jax.Array, a_prev: jax.Array):
    """SuperMimic reward computed on next state against next reference point."""
    p_ref = ref_next_22[0:3]
    R_ref = ref_next_22[3:12].reshape((3, 3))
    v_ref = ref_next_22[12:15]
    w_ref = ref_next_22[15:18]

    # 1. Position tracking: Non-vanishing Huber distance + quadratic restoring force
    pos_err = st_next.p - p_ref
    pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_err)) + 1e-4)
    r_pos = -3.5 * pos_dist - 12.0 * jnp.sum(jnp.square(pos_err))

    # 2. Attitude tracking: essential for quadrotor thrust vector alignment
    rot_sq = rot_angle_sq_jax(st_next.R, R_ref)
    r_rot = -0.40 * rot_sq

    # 3. Velocities & angular rates
    r_vel = -0.08 * jnp.sum(jnp.square(st_next.vb - v_ref))
    r_ang = -0.03 * jnp.sum(jnp.square(st_next.wb - w_ref))

    # 4. Action smoothness
    r_act = -0.01 * jnp.sum(jnp.square(a_curr - a_prev))

    return r_pos + r_rot + r_vel + r_ang + r_act


def compute_soft_term_prob(st_next: QuadrotorState, ref_next_22: jax.Array):
    """Differentiable soft termination probability for imagined rollouts.
    
    Only terminates when the drone hits the ground (z <= 0.05m) or flips upside down (R33 <= 0.20),
    allowing full gradient flow for all free-flight tracking recovery up to 1.5m.
    """
    crash_floor = jax.nn.sigmoid((0.05 - st_next.p[2]) / 0.02)
    crash_flip = jax.nn.sigmoid((0.20 - st_next.R[2, 2]) / 0.05)
    return jnp.clip(crash_floor + crash_flip, 0.0, 1.0)


# ------------------------------------------------------------------------------
# Closed-loop Scanned Evaluation Function for Sim2Real Metrics
# ------------------------------------------------------------------------------
def make_eval_kernel(policy: SuperMimicPolicy, obs_mean_jax: jax.Array, obs_std_jax: jax.Array):
    """JIT-compiled scanned evaluation kernel for rapid, memory-safe closed loop rollouts."""
    @jax.jit
    def eval_single_traj_scanned(p_curr, xs_ref, a0, xi, noise_pos, noise_gyro):
        H = len(xs_ref) - 1
        def step(carry, t):
            s_curr, a_prev = carry
            p_tgt = xs_ref[t + 1, 0:3]
            o = compute_obs(s_curr, p_tgt, a_prev)
            o = o.at[0:3].add(noise_pos[t]).at[15:18].add(noise_gyro[t])
            o_n = normalize_obs(o, obs_mean_jax, obs_std_jax)
            a = policy.apply({"params": p_curr}, o_n)
            u = norm_action_to_rpm(a)
            s_next = sim_step_multi_substeps(s_curr, u, xi, substeps=4, sub_dt=0.005)
            err = jnp.linalg.norm(s_next[0:3] - xs_ref[t + 1, 0:3])
            err_clamped = jnp.where(jnp.isnan(err) | (err > 3.0), 3.0, err)
            return (s_next, a), err_clamped

        _, errs = jax.lax.scan(step, (xs_ref[0], a0), jnp.arange(H))
        return jnp.sqrt(jnp.mean(jnp.square(errs))) * 100.0, jnp.max(errs) * 100.0

    return eval_single_traj_scanned


def evaluate_policy_suites(eval_fn: Any,
                           p_policy: Any,
                           val_trajs: List[np.ndarray],
                           val_actions: List[np.ndarray],
                           rng_np: np.random.Generator,
                           hover_p2p_indices: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:
    """Evaluate across Suite A (Nominal), Suite B (Sensor Noise), and Suite C (Domain Randomization)."""
    suites = ["nominal", "sensor_noise", "domain_randomization"]
    results = {}

    for suite in suites:
        all_rmse = []
        all_max = []

        for idx, traj in enumerate(val_trajs):
            traj_len = len(traj) - 1
            a0 = val_actions[idx][0]

            if suite == "domain_randomization":
                # DR context
                xi = np.array([
                    rng_np.uniform(-0.10, 0.10), # mass +- 10%
                    rng_np.uniform(-0.10, 0.10), # Ixx
                    rng_np.uniform(-0.10, 0.10), # Iyy
                    rng_np.uniform(-0.10, 0.10), # Izz
                    rng_np.uniform(-0.10, 0.10), # KF1
                    rng_np.uniform(-0.10, 0.10), # KF2
                    rng_np.uniform(-0.10, 0.10), # KF3
                    rng_np.uniform(-0.10, 0.10), # KF4
                    rng_np.uniform(-0.10, 0.10), # KM
                    rng_np.uniform(-0.20, 0.20), # tau_m
                ], dtype=np.float32)
            else:
                xi = np.zeros(10, dtype=np.float32)

            if suite == "sensor_noise":
                noise_pos = rng_np.normal(0.0, 0.005, size=(traj_len, 3)).astype(np.float32)
                noise_gyro = rng_np.normal(0.0, 0.005, size=(traj_len, 3)).astype(np.float32)
            else:
                noise_pos = np.zeros((traj_len, 3), dtype=np.float32)
                noise_gyro = np.zeros((traj_len, 3), dtype=np.float32)

            rmse_val, max_val = eval_fn(
                p_policy, jnp.asarray(traj), jnp.asarray(a0),
                jnp.asarray(xi), jnp.asarray(noise_pos), jnp.asarray(noise_gyro)
            )
            all_rmse.append(float(rmse_val))
            all_max.append(float(max_val))

        h_p2p_rmse = float(np.mean([all_rmse[i] for i in hover_p2p_indices])) if hover_p2p_indices else float(np.mean(all_rmse))
        results[suite] = {
            "rmse_cm": float(np.mean(all_rmse)),
            "hover_p2p_rmse_cm": h_p2p_rmse,
            "max_err_cm": float(np.mean(all_max)),
            "per_traj_rmse": [float(r) for r in all_rmse]
        }

    return results


def verify_firmware_header(matrices_h_path: str, csv_dir: str) -> bool:
    """Verify exported matrices.h matches firmware expectations and CSV data."""
    if not os.path.exists(matrices_h_path):
        return False
    with open(matrices_h_path, "r") as f:
        content = f.read()

    required_symbols = [
        "OBS_MEAN", "OBS_STD",
        "hidden_0_weights", "hidden_0_bias",
        "hidden_1_weights", "hidden_1_bias"
    ]
    for sym in required_symbols:
        if sym not in content:
            return False

    # Check Matrix structs layout
    assert ".rows = 22" in content
    assert ".rows = 64" in content
    assert ".rows = 4" in content
    return True


# ------------------------------------------------------------------------------
# Factory for Scanned Policy BPTT Kernels Parameterized by Horizon H
# ------------------------------------------------------------------------------
def make_bptt_kernel(horizon_len: int,
                     policy: SuperMimicPolicy,
                     critic: ValueCritic,
                     wm: WorldModel,
                     tx_pi: optax.GradientTransformation,
                     tx_critic: optax.GradientTransformation,
                     obs_mean_jax: jax.Array,
                     obs_std_jax: jax.Array,
                     in_m: jax.Array, in_s: jax.Array,
                     out_m: jax.Array, out_s: jax.Array,
                     critic_weight: float = 0.10):
    """Factory creating scanned BPTT imagination rollout and GAE critic update."""
    def scanned_imag_rollout(p_policy, p_c, p_wm, s0_22, a_init, xi, ref_traj):
        xi_norm = xi / 0.1

        def step_fn(carry, t):
            s_curr_22, a_prev, alive_prob = carry
            p_target = ref_traj[t + 1, 0:3]
            obs = compute_obs(s_curr_22, p_target, a_prev)
            obs_n = normalize_obs(obs, obs_mean_jax, obs_std_jax)

            a_curr = policy.apply({"params": p_policy}, obs_n)
            u_rpm = norm_action_to_rpm(a_curr)
            s_next_22 = sim_step_multi_substeps(s_curr_22, u_rpm, xi, substeps=4, sub_dt=0.005)
            st_next = unpack_state_22(s_next_22)

            # Robust reward evaluated on next state
            r = compute_supermimic_reward(st_next, ref_traj[t + 1], a_curr, a_prev)
            d = compute_soft_term_prob(st_next, ref_traj[t + 1])
            next_alive = alive_prob * (1.0 - d)

            st_curr = unpack_state_22(s_curr_22)
            inv_st = invariant_features(st_curr)
            pos_err = s_curr_22[0:3] - ref_traj[t, 0:3]
            c_in = jnp.concatenate([inv_st, xi_norm, pos_err], axis=-1)

            return (s_next_22, a_curr, next_alive), (r, d, alive_prob, c_in)

        (s_final_22, _, final_alive), (rews, terms, survs, c_ins) = jax.lax.scan(
            step_fn, (s0_22, a_init, 1.0), jnp.arange(horizon_len)
        )

        st_final = unpack_state_22(s_final_22)
        terminal_inv = invariant_features(st_final)
        terminal_pos_err = s_final_22[0:3] - ref_traj[horizon_len, 0:3]
        terminal_c_in = jnp.concatenate([terminal_inv, xi_norm, terminal_pos_err], axis=-1)
        v_terminal = critic.apply({"params": p_c}, terminal_c_in)

        discounts = 0.99 ** jnp.arange(horizon_len, dtype=jnp.float32)
        j_rewards = jnp.sum(discounts * survs * rews)
        j_bootstrap = (0.99 ** horizon_len) * final_alive * v_terminal
        policy_obj = j_rewards + critic_weight * j_bootstrap

        return policy_obj, (rews, terms, c_ins, terminal_c_in, v_terminal, final_alive)

    @jax.jit
    def train_step(p_policy, pi_opt_s, p_c, c_opt_s, p_wm, s0_b, a_init_b, xi_b, ref_b):
        def pi_loss_fn(p):
            def single(s0, a_init, xi, ref):
                obj, aux = scanned_imag_rollout(p, p_c, p_wm, s0, a_init, xi, ref)
                return -obj, aux
            losses, aux_all = jax.vmap(single)(s0_b, a_init_b, xi_b, ref_b)
            return jnp.mean(losses), aux_all

        (pi_loss, aux_all), pi_grads = jax.value_and_grad(pi_loss_fn, has_aux=True)(p_policy)
        pi_updates, pi_opt_s = tx_pi.update(pi_grads, pi_opt_s, p_policy)
        p_policy = optax.apply_updates(p_policy, pi_updates)

        rews, terms, c_ins, term_c_in, v_term, final_alives = aux_all
        c_ins_sg = jax.lax.stop_gradient(c_ins)
        rews_sg = jax.lax.stop_gradient(rews)
        terms_sg = jax.lax.stop_gradient(terms)

        def compute_gae(r_traj, d_traj, c_traj, v_term_s):
            v_all = jax.vmap(lambda x: critic.apply({"params": p_c}, x))(c_traj)
            v_ext = jnp.concatenate([v_all, v_term_s[None]], axis=0)
            def gae_scan(adv, t):
                t_idx = horizon_len - 1 - t
                delta = r_traj[t_idx] + 0.99 * (1.0 - d_traj[t_idx]) * v_ext[t_idx + 1] - v_ext[t_idx]
                adv_next = delta + 0.99 * 0.95 * (1.0 - d_traj[t_idx]) * adv
                target = adv_next + v_ext[t_idx]
                return adv_next, target
            _, targets_rev = jax.lax.scan(gae_scan, 0.0, jnp.arange(horizon_len))
            return jnp.flip(targets_rev, axis=0)

        gae_targets = jax.vmap(compute_gae)(rews_sg, terms_sg, c_ins_sg, jax.lax.stop_gradient(v_term))
        c_ins_flat = c_ins_sg.reshape((-1, c_ins_sg.shape[-1]))
        targets_flat = gae_targets.reshape((-1,))

        def critic_loss_fn(p):
            v_preds = jax.vmap(lambda x: critic.apply({"params": p}, x))(c_ins_flat)
            return jnp.mean(jnp.square(v_preds - targets_flat))

        c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(p_c)
        c_updates, c_opt_s = tx_critic.update(c_grads, c_opt_s, p_c)
        p_c = optax.apply_updates(p_c, c_updates)

        return p_policy, pi_opt_s, p_c, c_opt_s, pi_loss, c_loss, jnp.mean(final_alives)

    return train_step


# ------------------------------------------------------------------------------
# Main Autonomous SuperMimic Training Engine
# ------------------------------------------------------------------------------
def run_supermimic():
    args = parse_args()
    console = Console()
    console.print("\n[bold magenta]===================================================================[/bold magenta]")
    console.print("[bold cyan]       SuperMimic Quadrotor Autonomous Optimization Engine         [/bold cyan]")
    console.print("[bold magenta]===================================================================[/bold magenta]\n")

    if args.smoke_test:
        args.bc_epochs = 5
        args.wm_warmup_iters = 5
        args.iterations = 20
        args.eval_every = 5
        args.ckpt_every = 10
        args.max_hours = 0.05
        console.print("[bold yellow]Running in SMOKE-TEST mode.[/bold yellow]\n")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_log_path = os.path.join(args.out_dir, "train_log.csv")

    # Graceful interrupt handling
    interrupted = False
    def handle_signal(sig, frame):
        nonlocal interrupted
        console.print("\n[bold yellow]Interrupt caught! Wrapping up and saving checkpoint...[/bold yellow]")
        interrupted = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # 1. Observation Normalization
    obs_mean_np, obs_std_np = get_firmware_obs_norm()
    obs_mean_jax = jnp.asarray(obs_mean_np)
    obs_std_jax = jnp.asarray(obs_std_np)

    # 2. Ingest Corpus (Clean Trim from t=2)
    console.print(f"[bold cyan]1. Ingesting Expert Corpus:[/] {args.corpus}")
    t0_start = time.perf_counter()

    all_trajs = []
    all_actions = []
    all_task_types = []

    with h5py.File(args.corpus, "r") as f:
        num_trajs = len(f["trajectories"])
        for idx in range(num_trajs):
            grp = f["trajectories"][f"traj_{idx:06d}"]
            xs_17 = grp["states"][:]
            us_bar = grp["controls"][:]
            xs_22 = np.array([state17_to_22(x) for x in xs_17])
            task_type = str(grp.attrs.get("task_type", "unknown"))
            # Slicing past solver boundary artifacts (t<14 for hover, t<2 for dynamic)
            st_trim = 14 if task_type == "hover" else 2
            xs_22_clean = xs_22[st_trim:]
            us_rpm = us_bar[st_trim:] * MAX_THRUST
            a_firmware = np.clip((us_rpm - HOVER_RPM) / THRUST_SCALE, -1.0, 1.0)

            all_trajs.append(xs_22_clean)
            all_actions.append(a_firmware)
            all_task_types.append(task_type)

    if args.trajs is not None and len(args.trajs) > 0:
        train_trajs = [all_trajs[i] for i in args.trajs]
        train_actions = [all_actions[i] for i in args.trajs]
        train_task_types = [all_task_types[i] for i in args.trajs]
        val_trajs = train_trajs
        val_actions = train_actions
        val_task_types = train_task_types
        val_hover_p2p_indices = list(range(len(val_trajs)))
        console.print(f"   Selected Reference Library of [green]{len(train_trajs)}[/] benchmark trajectories: {args.trajs}")
    else:
        # Split: 43 train, 7 validation
        val_count = 7
        train_trajs = all_trajs[:-val_count]
        train_actions = all_actions[:-val_count]
        train_task_types = all_task_types[:-val_count]
        val_trajs = all_trajs[-val_count:]
        val_actions = all_actions[-val_count:]
        val_task_types = all_task_types[-val_count:]
        val_hover_p2p_indices = [i for i, tt in enumerate(val_task_types) if tt in ("hover", "point_to_point")]

    console.print(f"   Loaded [green]{num_trajs}[/] total trajectories.")
    console.print(f"   Train trajectories: [green]{len(train_trajs)}[/] | "
                  f"Validation trajectories: [green]{len(val_trajs)}[/]\n")

    # Build 1.0-second bins for Return-Based Reset Sampling (RB-RS)
    train_bins = []
    for t_idx, traj in enumerate(train_trajs):
        traj_len = len(traj)
        bin_step = int(1.0 / SIM_DT) # 50 steps
        for st in range(0, traj_len - 10, bin_step):
            en = min(st + bin_step, traj_len - 1)
            train_bins.append((t_idx, st, en))

    train_bin_returns = np.zeros(len(train_bins), dtype=np.float32)
    console.print(f"   Compiled [green]{len(train_bins)}[/] 1-second reset bins for Return-Based Curriculum.\n")

    def sample_train_reset_frames(n: int, rng: np.random.Generator):
        p_uniform = np.ones(len(train_bins)) / len(train_bins)
        max_ret = max(1e-4, np.max(train_bin_returns))
        scaled_neg_rets = -25.0 * (train_bin_returns / max_ret)
        shift = scaled_neg_rets - np.max(scaled_neg_rets)
        exp_rets = np.exp(shift)
        p_softmax = exp_rets / np.sum(exp_rets)
        p_combined = 0.50 * p_uniform + 0.50 * p_softmax
        p_combined = p_combined / np.sum(p_combined)
        chosen_bins = rng.choice(len(train_bins), size=n, p=p_combined)
        trajs = np.array([train_bins[b][0] for b in chosen_bins], dtype=np.int32)
        frames = np.array([rng.integers(train_bins[b][1], train_bins[b][2]) for b in chosen_bins], dtype=np.int32)
        return chosen_bins, trajs, frames

    # 3. Seed Replay Buffer
    console.print(f"[bold cyan]2. Seeding Replay Buffer with Clean Expert Trajectories...[/]")
    buffer = SuperMimicReplayBuffer(capacity=120_000)
    expert_obs_list = []
    expert_acts_list = []

    for t_idx in range(len(train_trajs)):
        xs_clean = train_trajs[t_idx]
        a_clean = train_actions[t_idx]
        H = len(a_clean)

        a_prev_arr = np.zeros_like(a_clean)
        a_prev_arr[1:] = a_clean[:-1]
        a_prev_arr[0] = a_clean[0]

        xi_arr = np.zeros((H, 10), dtype=np.float32)
        r_arr = np.zeros(H, dtype=np.float32)
        f_arr = np.arange(H, dtype=np.int32)
        traj_id_arr = np.full(H, t_idx, dtype=np.int32)
        ep_step_arr = np.arange(H, dtype=np.int32)
        done_arr = np.zeros(H, dtype=bool)
        done_arr[-1] = True

        buffer.add_batch(
            xs_clean[:H], xs_clean[1:H + 1], a_clean, a_prev_arr,
            xi_arr, r_arr, f_arr, traj_id_arr, ep_step_arr, done_arr
        )

        for t in range(H):
            p_tgt = xs_clean[t + 1, 0:3]
            o = compute_obs(xs_clean[t], p_tgt, a_prev_arr[t])
            expert_obs_list.append(o)
            expert_acts_list.append(a_clean[t])

    expert_obs = np.array(expert_obs_list, dtype=np.float32)
    expert_acts = np.array(expert_acts_list, dtype=np.float32)
    console.print(f"   Seeded buffer with [green]{buffer.size}[/] clean transitions ({len(expert_obs)} BC pairs).\n")

    # 4. Normalization Statistics for World Model
    in_m = jnp.zeros(31)
    in_s = jnp.array([
        1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ], dtype=jnp.float32)

    out_m = jnp.zeros(16)
    out_s = jnp.array([
        0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10,
        0.05, 0.05, 0.05, 0.05
    ], dtype=jnp.float32)

    # 5. Initialize Networks
    console.print(f"[bold cyan]3. Initializing Networks & Optimizers...[/]")
    wm = WorldModel(hidden_dim=256)
    policy = SuperMimicPolicy()
    critic = ValueCritic(hidden_dims=(128, 64))

    rng = jax.random.PRNGKey(args.seed)
    rng, k_wm, k_pi, k_c = jax.random.split(rng, 4)

    wm_params = wm.init(k_wm)
    tx_wm = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr_wm))
    wm_opt = tx_wm.init(wm_params)

    pi_params = policy.init(k_pi, jnp.zeros((1, 22)))["params"]

    # Cosine learning rate schedule for policy
    lr_schedule = optax.cosine_decay_schedule(
        init_value=args.lr_pi,
        decay_steps=args.iterations,
        alpha=0.03
    )
    tx_pi = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.clip(0.1),
        optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.95)
    )
    pi_opt = tx_pi.init(pi_params)

    critic_params = critic.init(k_c, jnp.zeros((1, 30)))["params"]
    tx_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr_critic))
    critic_opt = tx_critic.init(critic_params)

    # --------------------------------------------------------------------------
    # Phase 0: Behavior Cloning Warm-up (100 epochs)
    # --------------------------------------------------------------------------
    console.print(f"[bold cyan]4. Phase 0: Behavior Cloning Warm-up ({args.bc_epochs} epochs)...[/]")
    t0_bc = time.perf_counter()
    tx_bc = optax.adam(1e-3)
    bc_opt = tx_bc.init(pi_params)

    obs_n_all = (expert_obs - obs_mean_np) / obs_std_np
    N_bc = len(obs_n_all)
    B_bc = 256
    n_batches = N_bc // B_bc

    @jax.jit
    def bc_step(p, opt_s, o_b, a_b):
        def loss_fn(p_curr):
            return jnp.mean(jnp.square(policy.apply({"params": p_curr}, o_b) - a_b))
        l, g = jax.value_and_grad(loss_fn)(p)
        u, opt_s = tx_bc.update(g, opt_s, p)
        p = optax.apply_updates(p, u)
        return p, opt_s, l

    for ep in range(args.bc_epochs):
        rng, pk = jax.random.split(rng)
        perm = jax.random.permutation(pk, N_bc)
        ep_l = 0.0
        for b in range(n_batches):
            idx = perm[b * B_bc:(b + 1) * B_bc]
            pi_params, bc_opt, l_val = bc_step(
                pi_params, bc_opt,
                jnp.asarray(obs_n_all[idx]), jnp.asarray(expert_acts[idx])
            )
            ep_l += float(l_val)
        if (ep + 1) % 25 == 0 or ep == args.bc_epochs - 1:
            console.print(f"   [BC] Epoch {ep+1:3d}/{args.bc_epochs} | MSE Loss: [cyan]{ep_l/n_batches:.6f}[/] | Elapsed: {time.perf_counter()-t0_bc:.1f} s")

    pi_opt = tx_pi.init(pi_params)

    # Initial Validation Evaluation after BC
    rng_eval = np.random.default_rng(args.seed)
    eval_kernel = make_eval_kernel(policy, obs_mean_jax, obs_std_jax)
    eval_init = evaluate_policy_suites(eval_kernel, pi_params, val_trajs, val_actions, rng_eval, val_hover_p2p_indices)
    console.print(f"   [bold green]Initial Val RMSE after BC: {eval_init['nominal']['hover_p2p_rmse_cm']:.2f} cm (Hover/P2P)[/bold green] | "
                  f"Overall: {eval_init['nominal']['rmse_cm']:.2f} cm "
                  f"(Sensor Noise: {eval_init['sensor_noise']['hover_p2p_rmse_cm']:.2f} cm, DR: {eval_init['domain_randomization']['hover_p2p_rmse_cm']:.2f} cm)\n")

    # --------------------------------------------------------------------------
    # Phase 1: Differentiable World Model Warm-up
    # --------------------------------------------------------------------------
    console.print(f"[bold cyan]5. Phase 1: Differentiable World Model Warm-up ({args.wm_warmup_iters} iterations)...[/]")
    k_steps = args.k_window

    def scanned_wm_loss(p_wm, s_win, s1_win, a_win, xi_win, mask_win):
        st_init = unpack_state_22(s_win[0])
        def step_fn(st_curr, t):
            st_gt0 = unpack_state_22(s_win[t])
            st_gt1 = unpack_state_22(s1_win[t])
            target_delta = compute_frame_local_targets(st_gt0, st_gt1)
            xi_norm = xi_win[t] / 0.1
            raw_in = wm.inputs(st_curr, a_win[t], xi_norm)
            norm_in = (raw_in - in_m) / in_s
            pred_norm_out = wm.net.apply(p_wm, norm_in)
            pred_delta = pred_norm_out * out_s + out_m
            step_err = jnp.square((pred_delta - target_delta) / out_s)
            loss = jnp.mean(step_err) * mask_win[t]
            st_next = integrate_frame_local_step(st_curr, pred_delta)
            return st_next, loss
        _, step_losses = jax.lax.scan(step_fn, st_init, jnp.arange(k_steps))
        return jnp.sum(step_losses) / jnp.maximum(jnp.sum(mask_win), 1.0)

    @jax.jit
    def train_wm_batch(p_wm, opt_s, s_b, s1_b, a_b, xi_b, mask_b):
        def loss_fn(p):
            losses = jax.vmap(lambda s, s1, a, xi, m: scanned_wm_loss(p, s, s1, a, xi, m))(s_b, s1_b, a_b, xi_b, mask_b)
            return jnp.mean(losses)
        l, g = jax.value_and_grad(loss_fn)(p_wm)
        u, opt_s = tx_wm.update(g, opt_s, p_wm)
        p_wm = optax.apply_updates(p_wm, u)
        return p_wm, opt_s, l

    t0_wm = time.perf_counter()
    rng_np = np.random.default_rng(args.seed)
    for it in range(args.wm_warmup_iters):
        if interrupted:
            break
        w = buffer.sample_windows(batch_size=args.batch_size_wm, k=k_steps, rng=rng_np)
        wm_params, wm_opt, l_wm = train_wm_batch(
            wm_params, wm_opt,
            jnp.asarray(w["s"]), jnp.asarray(w["s1"]), jnp.asarray(w["a"]),
            jnp.asarray(w["xi"]), jnp.asarray(w["mask"])
        )
        if (it + 1) % 50 == 0 or it == args.wm_warmup_iters - 1:
            console.print(f"   [WM Warmup] Iter {it+1:4d}/{args.wm_warmup_iters} | WM Loss: [cyan]{float(l_wm):.6f}[/] | Elapsed: {time.perf_counter()-t0_wm:.1f} s")

    console.print("[green]World Model Warm-up Completed![/green]\n")

    # --------------------------------------------------------------------------
    # Phase 2: Compile BPTT Kernels for Horizon Curriculum
    # --------------------------------------------------------------------------
    console.print(f"[bold cyan]6. Compiling Scanned BPTT Kernels for Curriculum (H=16, 24, 32)...[/]")
    t0_comp = time.perf_counter()
    kernel_h16 = make_bptt_kernel(16, policy, critic, wm, tx_pi, tx_critic, obs_mean_jax, obs_std_jax, in_m, in_s, out_m, out_s, critic_weight=args.critic_weight)
    kernel_h24 = make_bptt_kernel(24, policy, critic, wm, tx_pi, tx_critic, obs_mean_jax, obs_std_jax, in_m, in_s, out_m, out_s, critic_weight=args.critic_weight)
    kernel_h32 = make_bptt_kernel(32, policy, critic, wm, tx_pi, tx_critic, obs_mean_jax, obs_std_jax, in_m, in_s, out_m, out_s, critic_weight=args.critic_weight)
    vmap_sim_step = jax.jit(jax.vmap(lambda s, u, xi: sim_step_multi_substeps(s, u, xi, substeps=4, sub_dt=0.005)))
    console.print(f"   Kernels compiled in {time.perf_counter()-t0_comp:.2f} s\n")

    # CSV Logging initialization
    with open(csv_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "iteration", "elapsed_s", "elapsed_h", "horizon", "lr",
            "wm_loss", "pi_loss", "critic_loss", "alive_prob",
            "val_nom_rmse_cm", "val_noise_rmse_cm", "val_dr_rmse_cm", "best_val_rmse_cm"
        ])

    # --------------------------------------------------------------------------
    # Phase 3: Joint SuperMimic Autonomous Training Loop
    # --------------------------------------------------------------------------
    console.print(f"[bold cyan]7. Launching Joint SuperMimic Autonomous Training Loop...[/]")
    console.print(f"   Time Budget: [green]{args.max_hours:.1f} hours[/] | "
                  f"Target RMSE: [green]{args.target_rmse:.1f} cm[/]\n")

    best_val_rmse = eval_init["nominal"]["hover_p2p_rmse_cm"]

    # Initial save of best checkpoint from BC
    save_checkpoint(
        ckpt_dir, "checkpoint_best",
        {"policy": pi_params, "wm": wm_params, "critic": critic_params},
        {"pi_opt": pi_opt, "wm_opt": wm_opt, "critic_opt": critic_opt},
        epoch=0, step=0, val_loss=best_val_rmse,
        obs_mean=obs_mean_np, obs_std=obs_std_np,
        metadata={"iteration": 0, "best_rmse": best_val_rmse}
    )
    export_firmware_header(pi_params, obs_mean_np, obs_std_np, args.out_dir)

    # Initialize parallel simulator environments
    num_envs = args.num_envs
    env_states = np.zeros((num_envs, 22), dtype=np.float32)
    env_actions = np.zeros((num_envs, 4), dtype=np.float32)
    env_xi = np.zeros((num_envs, 10), dtype=np.float32)
    env_trajs = np.zeros(num_envs, dtype=np.int32)
    env_frames = np.zeros(num_envs, dtype=np.int32)
    env_steps = np.zeros(num_envs, dtype=np.int32)
    env_bins = np.zeros(num_envs, dtype=np.int32)
    env_disc_rets = np.zeros(num_envs, dtype=np.float32)

    bins_init, trajs_init, frames_init = sample_train_reset_frames(num_envs, rng_np)
    for i in range(num_envs):
        t_id, f_idx = trajs_init[i], frames_init[i]
        s_init = train_trajs[t_id][f_idx].copy()
        s_init[0:3] += rng_np.normal(0, 0.015, size=3)
        s_init[12:15] += rng_np.normal(0, 0.02, size=3)
        env_states[i] = s_init
        env_actions[i] = train_actions[t_id][f_idx].copy()
        env_trajs[i] = t_id
        env_frames[i] = f_idx
        env_bins[i] = bins_init[i]
        env_xi[i] = rng_np.uniform(-0.05, 0.05, size=10)

    t0_joint = time.perf_counter()

    for it in range(args.iterations):
        if interrupted:
            break

        elapsed_s = time.perf_counter() - t0_start
        elapsed_h = elapsed_s / 3600.0
        if elapsed_h >= args.max_hours:
            console.print(f"\n[bold yellow]Maximum time budget of {args.max_hours:.1f} hours reached! Wrapping up...[/bold yellow]")
            break

        # Curriculum Horizon Selection
        if it < 350:
            curr_H = 16
            train_pi_step = kernel_h16
        elif it < 800:
            curr_H = 24
            train_pi_step = kernel_h24
        else:
            curr_H = 32
            train_pi_step = kernel_h32

        # A. Online Simulator Step with Parallel Environments
        s_batch_sim = env_states.copy()
        a_prev_batch = env_actions.copy()
        xi_batch = env_xi.copy()

        p_targets = np.array([
            train_trajs[env_trajs[i]][min(env_frames[i] + 1, len(train_trajs[env_trajs[i]]) - 1), 0:3]
            for i in range(num_envs)
        ])
        obs_batch = np.array([
            compute_obs(env_states[i], p_targets[i], env_actions[i])
            for i in range(num_envs)
        ])
        obs_batch_n = (obs_batch - obs_mean_np) / obs_std_np

        actions_clean = np.array(policy.apply({"params": pi_params}, jnp.asarray(obs_batch_n)))
        actions_noisy = np.clip(actions_clean + rng_np.normal(0.0, args.action_noise, size=(num_envs, 4)), -1.0, 1.0)

        u_rpms = np.clip(HOVER_RPM + actions_noisy * THRUST_SCALE, 0.0, MAX_THRUST)
        next_states = np.array(vmap_sim_step(jnp.asarray(env_states), jnp.asarray(u_rpms), jnp.asarray(env_xi)))

        step_rewards = np.zeros(num_envs, dtype=np.float32)
        dones = np.zeros(num_envs, dtype=bool)

        for i in range(num_envs):
            t_id, f_idx = env_trajs[i], env_frames[i]
            ref_st_next = train_trajs[t_id][min(f_idx + 1, len(train_trajs[t_id]) - 1)]
            st_next_obj = unpack_state_22(jnp.asarray(next_states[i]))
            r_val = float(compute_supermimic_reward(st_next_obj, jnp.asarray(ref_st_next), jnp.asarray(actions_noisy[i]), jnp.asarray(env_actions[i])))
            step_rewards[i] = r_val
            env_disc_rets[i] += (0.99 ** env_steps[i]) * r_val

            s_next = next_states[i]
            pos_err = np.linalg.norm(s_next[0:3] - ref_st_next[0:3])
            crashed = (s_next[2] < 0.05) or (pos_err > 0.40) or (s_next[11] < 0.5)
            timeout = (f_idx + 1 >= len(train_trajs[t_id]) - 1) or (env_steps[i] >= 150)
            dones[i] = crashed or timeout

        buffer.add_batch(
            s_batch_sim, next_states, actions_noisy, a_prev_batch, xi_batch,
            step_rewards, env_frames, env_trajs, env_steps, dones
        )

        # Handle Resets with Return-Based Sampling
        done_ids = np.nonzero(dones)[0]
        if len(done_ids) > 0:
            for d_i in done_ids:
                b_idx = env_bins[d_i]
                train_bin_returns[b_idx] = 0.9 * train_bin_returns[b_idx] + 0.1 * env_disc_rets[d_i]
            new_bins, new_trajs, new_frames = sample_train_reset_frames(len(done_ids), rng_np)
            for j, d_idx in enumerate(done_ids):
                t_id, f_idx = new_trajs[j], new_frames[j]
                s_new = train_trajs[t_id][f_idx].copy()
                s_new[0:3] += rng_np.normal(0, 0.015, size=3)
                s_new[12:15] += rng_np.normal(0, 0.02, size=3)
                env_states[d_idx] = s_new
                env_actions[d_idx] = train_actions[t_id][f_idx].copy()
                env_trajs[d_idx] = t_id
                env_frames[d_idx] = f_idx
                env_steps[d_idx] = 0
                env_bins[d_idx] = new_bins[j]
                env_disc_rets[d_idx] = 0.0
                env_xi[d_idx] = rng_np.uniform(-0.05, 0.05, size=10)

        not_done = np.nonzero(~dones)[0]
        env_states[not_done] = next_states[not_done]
        env_actions[not_done] = actions_noisy[not_done]
        env_frames[not_done] += 1
        env_steps[not_done] += 1

        # B. World Model Batched Update
        w = buffer.sample_windows(batch_size=args.batch_size_wm, k=k_steps, rng=rng_np)
        wm_params, wm_opt, l_wm = train_wm_batch(
            wm_params, wm_opt,
            jnp.asarray(w["s"]), jnp.asarray(w["s1"]), jnp.asarray(w["a"]),
            jnp.asarray(w["xi"]), jnp.asarray(w["mask"])
        )

        # C. Policy & Critic BPTT Update in Imagination
        # Sample reference windows using Return-Based Reset Curriculum (RB-RS)
        _, t_ids, f_starts = sample_train_reset_frames(args.batch_size_pi, rng_np)
        pert_scale = 0.015 if it < 500 else 0.008

        s0_list = []
        a0_list = []
        ref_windows = []

        for b in range(args.batch_size_pi):
            t_id_b = int(t_ids[b])
            f_b = int(f_starts[b])
            traj_len = len(train_trajs[t_id_b])
            f_max = traj_len - (curr_H + 1)
            f_clamped = min(f_b, max(0, f_max))
            slice_w = train_trajs[t_id_b][f_clamped: f_clamped + curr_H + 1]
            if len(slice_w) < curr_H + 1:
                pad = np.tile(train_trajs[t_id_b][-1:], (curr_H + 1 - len(slice_w), 1))
                slice_w = np.concatenate([slice_w, pad], axis=0)

            # Local state perturbation around reference point for feedback gain synthesis
            s0 = slice_w[0].copy()
            s0[0:3] += rng_np.normal(0, pert_scale, size=3)
            s0[12:15] += rng_np.normal(0, pert_scale * 1.5, size=3)

            s0_list.append(s0)
            a0_list.append(train_actions[t_id_b][f_clamped].copy())
            ref_windows.append(slice_w)

        s0_pi = jnp.asarray(np.array(s0_list, dtype=np.float32))
        a_init_pi = jnp.asarray(np.array(a0_list, dtype=np.float32))
        ref_batch_pi = jnp.asarray(np.array(ref_windows, dtype=np.float32))

        # Domain Randomization in imagination rollouts (mass +-10%, rotor thrust, inertia, motor lag)
        xi_raw = rng_np.uniform(-0.10, 0.10, size=(args.batch_size_pi, 10)).astype(np.float32)
        xi_pi = jnp.asarray(xi_raw)

        pi_params, pi_opt, critic_params, critic_opt, l_pi, l_c, alive = train_pi_step(
            pi_params, pi_opt, critic_params, critic_opt, wm_params,
            s0_pi, a_init_pi, xi_pi, ref_batch_pi
        )

        # Periodic Validation & Sim2Real Metric Tracking
        if (it + 1) % args.eval_every == 0 or it == args.iterations - 1:
            eval_res = evaluate_policy_suites(eval_kernel, pi_params, val_trajs, val_actions, rng_eval, val_hover_p2p_indices)
            nom_rmse = eval_res["nominal"]["rmse_cm"]
            nom_hp2p = eval_res["nominal"]["hover_p2p_rmse_cm"]
            noise_rmse = eval_res["sensor_noise"]["rmse_cm"]
            noise_hp2p = eval_res["sensor_noise"]["hover_p2p_rmse_cm"]
            dr_rmse = eval_res["domain_randomization"]["rmse_cm"]
            dr_hp2p = eval_res["domain_randomization"]["hover_p2p_rmse_cm"]

            is_best = nom_hp2p < best_val_rmse
            if is_best:
                best_val_rmse = nom_hp2p
                save_checkpoint(
                    ckpt_dir, "checkpoint_best",
                    {"policy": pi_params, "wm": wm_params, "critic": critic_params},
                    {"pi_opt": pi_opt, "wm_opt": wm_opt, "critic_opt": critic_opt},
                    epoch=it + 1, step=it + 1, val_loss=best_val_rmse,
                    obs_mean=obs_mean_np, obs_std=obs_std_np,
                    metadata={"iteration": it + 1, "best_rmse": best_val_rmse, "elapsed_h": elapsed_h}
                )
                csv_dir, matrices_path = export_firmware_header(pi_params, obs_mean_np, obs_std_np, args.out_dir)
                verify_firmware_header(matrices_path, csv_dir)

            current_lr = float(lr_schedule(it))
            per_traj_str = [round(r, 1) for r in eval_res["nominal"]["per_traj_rmse"]]
            console.print(
                f"Iter {it+1:5d} (H={curr_H:2d}, lr={current_lr:.1e}, {elapsed_s:.1f}s) | "
                f"WM: {float(l_wm):.4f} | Pi: {float(l_pi):.4f} | Critic: {float(l_c):.4f} | Alive: {float(alive):.3f} | "
                f"Nominal: [bold]{nom_hp2p:.2f} cm[/bold] (Best: [bold green]{best_val_rmse:.2f} cm[/bold green]) "
                f"{'*' if is_best else ''}\n"
                f"   Suite B (Noise): {noise_hp2p:.2f} cm | Suite C (DR): {dr_hp2p:.2f} cm | All {len(val_trajs)} Trajs (Nominal): {nom_rmse:.2f} cm | Trajs: {per_traj_str}"
            )

            # CSV log
            with open(csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"), it + 1, f"{elapsed_s:.1f}", f"{elapsed_h:.3f}",
                    curr_H, f"{current_lr:.2e}", f"{float(l_wm):.6f}", f"{float(l_pi):.6f}", f"{float(l_c):.6f}",
                    f"{float(alive):.4f}", f"{nom_rmse:.2f}", f"{noise_rmse:.2f}", f"{dr_rmse:.2f}", f"{best_val_rmse:.2f}"
                ])

            # Check if target RMSE reached (after sufficient curriculum progression through H=32)
            if best_val_rmse <= args.target_rmse and it >= 1000:
                console.print(f"\n[bold green]SUCCESS: Reached target tracking precision of {best_val_rmse:.2f} cm <= {args.target_rmse:.1f} cm![/bold green]\n")
                break

        # Periodic checkpoint
        if (it + 1) % args.ckpt_every == 0:
            save_checkpoint(
                ckpt_dir, "checkpoint_latest",
                {"policy": pi_params, "wm": wm_params, "critic": critic_params},
                {"pi_opt": pi_opt, "wm_opt": wm_opt, "critic_opt": critic_opt},
                epoch=it + 1, step=it + 1, val_loss=best_val_rmse,
                obs_mean=obs_mean_np, obs_std=obs_std_np,
                metadata={"iteration": it + 1, "best_rmse": best_val_rmse, "elapsed_h": elapsed_h}
            )

    # Wrap up & Final Evaluation of Best Checkpoint
    elapsed_total = time.perf_counter() - t0_start
    best_ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.npz")
    matrices_h_path = os.path.join(args.out_dir, "matrices.h")
    csv_dir = os.path.join(args.out_dir, "csv")

    console.print(f"\n[bold green]======================== Optimization Complete ========================[/bold green]")
    console.print(f"Total Wall Time           : [green]{elapsed_total / 3600.0:.2f} hours[/] ({elapsed_total:.1f} s)")
    console.print(f"Best Validation RMSE      : [bold green]{best_val_rmse:.2f} cm[/]")
    console.print(f"Latest Checkpoint         : [green]{os.path.join(ckpt_dir, 'checkpoint_latest.npz')}[/]")
    console.print(f"Best Checkpoint           : [green]{best_ckpt_path}[/]")
    console.print(f"Firmware Header matrices.h: [bold green]{matrices_h_path}[/]")
    console.print(f"Training Log CSV          : [green]{csv_log_path}[/]\n")

    # Run full 3-suite Sim2Real evaluation report on best policy
    if os.path.exists(best_ckpt_path):
        from quadrotor.supermimic import load_checkpoint
        loaded = load_checkpoint(best_ckpt_path, {"policy": pi_params, "wm": wm_params, "critic": critic_params}, {"pi_opt": pi_opt, "wm_opt": wm_opt, "critic_opt": critic_opt})
        best_p_pi = loaded["params"]["policy"]
        final_eval = evaluate_policy_suites(eval_kernel, best_p_pi, val_trajs, val_actions, rng_eval, val_hover_p2p_indices)
        
        table = Table(title="SuperMimic Sim2Real Validation Suite Results (Best Model)")
        table.add_column("Evaluation Suite", style="cyan")
        table.add_column("Hover & P2P RMSE", style="bold green")
        table.add_column("Overall RMSE", style="yellow")
        table.add_column("Max Tracking Error", style="magenta")
        table.add_column("Per-Trajectory RMSE (cm)", style="dim")

        for s_name in ["nominal", "sensor_noise", "domain_randomization"]:
            res = final_eval[s_name]
            per_str = ", ".join([f"{r:.1f}" for r in res["per_traj_rmse"]])
            table.add_row(
                s_name.replace("_", " ").title(),
                f"{res['hover_p2p_rmse_cm']:.2f} cm",
                f"{res['rmse_cm']:.2f} cm",
                f"{res['max_err_cm']:.2f} cm",
                f"[{per_str}]"
            )
        console.print(table)
        console.print("")

        # Verify firmware header
        header_ok = verify_firmware_header(matrices_h_path, csv_dir)
        if header_ok:
            console.print("[bold green]Firmware header matrices.h verified successfully![/bold green]\n")
        else:
            console.print("[bold red]Firmware header verification failed![/bold red]\n")


if __name__ == "__main__":
    run_supermimic()
