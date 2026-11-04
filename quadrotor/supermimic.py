"""SuperMimic Quadrotor Training and Sim2Real Pipeline.

Implements the SuperMimic paradigm for the Bitcraze Crazyflie 2.1 quadrotor:
  1. Corpus ingestion: extracts (state, reference, action) pairs from SSQP corpus.
  2. Policy Architecture: matches firmware/my_controller.c (22 obs -> 64 SiLU -> 4 tanh).
  3. Hybrid Training:
       Phase 1: Supervised Behavior Cloning (BC) on expert transitions.
       Phase 2: Differentiable Backpropagation Through Time (BPTT) through JAX dynamics.
  4. Sim2Real Validation: Closed-loop rollouts under nominal, sensor-noise, and domain-randomized dynamics.
  5. Firmware Exporter: Generates CSV and matrices.h compatible with Crazyflie C firmware.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
import h5py
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from quadrotor.params import (
    NX, NU, HOVER_RPM, MAX_THRUST, SIM_DT,
    M_NOMINAL, I_NOMINAL, KF, KM, L, TAU_M
)
from quadrotor.quaternion import quat_to_R, state17_to_22
from crazyflie_nn.dynamics.dynamics import rk4


# Observation layout (dimension 22) matching firmware/my_controller.c
OBS_DIM = 22
ACT_DIM = 4
HIDDEN_DIM = 64
THRUST_SCALE = 0.4 * HOVER_RPM  # +/- 40% of hover thrust


class SuperMimicPolicy(nn.Module):
    """Firmware-compatible MLP policy: 22 obs -> 64 (SiLU) -> 4 (tanh)."""
    hidden_dim: int = HIDDEN_DIM
    act_dim: int = ACT_DIM

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        # Layer names match crazyflie_nn firmware expectation: hidden_0, hidden_1
        h = nn.Dense(self.hidden_dim, name="hidden_0")(obs)
        h = nn.silu(h)
        out = nn.Dense(self.act_dim, name="hidden_1")(h)
        return jnp.tanh(out)


def compute_obs(state_22: jax.Array | np.ndarray,
                target_pos: jax.Array | np.ndarray,
                action_prev: jax.Array | np.ndarray) -> jax.Array | np.ndarray:
    """Compute 22-dimensional observation matching crazyflie_nn/firmware/my_controller.c."""
    pos = state_22[0:3]
    R_wb = state_22[3:12]
    vb = state_22[12:15]
    wb = state_22[15:18]

    pos_err = pos - target_pos
    if isinstance(state_22, jax.Array):
        return jnp.concatenate([pos_err, R_wb, vb, wb, action_prev])
    return np.concatenate([pos_err, R_wb, vb, wb, action_prev]).astype(np.float32)


def rpm_to_norm_action(omega_rpm: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
    """Convert motor speed (RPM) to firmware normalized action a in [-1, 1]."""
    return (omega_rpm - HOVER_RPM) / THRUST_SCALE


def norm_action_to_rpm(action: jax.Array | np.ndarray) -> jax.Array | np.ndarray:
    """Convert normalized action to clamped motor RPM."""
    rpm = HOVER_RPM + action * THRUST_SCALE
    if isinstance(action, jax.Array):
        return jnp.clip(rpm, 0.0, MAX_THRUST)
    return np.clip(rpm, 0.0, MAX_THRUST)


@dataclass
class SuperMimicDataset:
    """Dataset extracted from HDF5 expert trajectory corpus."""
    obs: np.ndarray             # (N, 22)
    actions: np.ndarray         # (N, 4)
    obs_mean: np.ndarray        # (22,)
    obs_std: np.ndarray         # (22,)
    trajectories: List[Dict[str, np.ndarray]] # Raw trajectory states & controls

    @classmethod
    def from_hdf5(cls, filepath: str) -> "SuperMimicDataset":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Corpus file not found: {filepath}")

        all_obs = []
        all_acts = []
        trajs_list = []

        with h5py.File(filepath, "r") as f:
            if "trajectories" in f:
                traj_keys = sorted(list(f["trajectories"].keys()))
                for k in traj_keys:
                    grp = f["trajectories"][k]
                    xs_17 = grp["states"][:]     # (H+1, 17)
                    us_bar = grp["controls"][:]  # (H, 4)
                    dt = float(grp.attrs.get("dt", 0.02))
                    task_type = str(grp.attrs.get("task_type", "unknown"))

                    H = len(us_bar)
                    # Convert 17D states to 22D
                    xs_22 = np.zeros((H + 1, 22), dtype=np.float32)
                    for t in range(H + 1):
                        xs_22[t] = state17_to_22(xs_17[t])

                    # Convert normalized rotor controls to RPM and then to firmware actions
                    us_rpm = us_bar * MAX_THRUST
                    actions_firmware = rpm_to_norm_action(us_rpm)

                    trajs_list.append({
                        "xs_17": xs_17,
                        "xs_22": xs_22,
                        "us_bar": us_bar,
                        "us_rpm": us_rpm,
                        "actions": actions_firmware,
                        "dt": dt,
                        "task_type": task_type,
                    })

                    a_prev = np.zeros(ACT_DIM, dtype=np.float32)
                    for t in range(H):
                        p_target = xs_22[t + 1, 0:3]
                        o = compute_obs(xs_22[t], p_target, a_prev)
                        a = actions_firmware[t]
                        all_obs.append(o)
                        all_acts.append(a)
                        a_prev = a

            elif "states" in f and "controls" in f:
                xs_all = f["states"][:]     # (N, H+1, 17)
                us_all = f["controls"][:]   # (N, H, 4)
                N, H, _ = us_all.shape
                for i in range(N):
                    xs_17 = xs_all[i]
                    us_bar = us_all[i]
                    xs_22 = np.zeros((H + 1, 22), dtype=np.float32)
                    for t in range(H + 1):
                        xs_22[t] = state17_to_22(xs_17[t])

                    us_rpm = us_bar * MAX_THRUST
                    actions_firmware = rpm_to_norm_action(us_rpm)

                    trajs_list.append({
                        "xs_17": xs_17,
                        "xs_22": xs_22,
                        "us_bar": us_bar,
                        "us_rpm": us_rpm,
                        "actions": actions_firmware,
                        "dt": 0.02,
                        "task_type": "unknown",
                    })

                    a_prev = np.zeros(ACT_DIM, dtype=np.float32)
                    for t in range(H):
                        p_target = xs_22[t + 1, 0:3]
                        o = compute_obs(xs_22[t], p_target, a_prev)
                        a = actions_firmware[t]
                        all_obs.append(o)
                        all_acts.append(a)
                        a_prev = a
            else:
                raise ValueError("Unrecognized HDF5 dataset structure.")

        obs_arr = np.array(all_obs, dtype=np.float32)
        acts_arr = np.array(all_acts, dtype=np.float32)

        mean = np.mean(obs_arr, axis=0)
        std = np.std(obs_arr, axis=0)
        std = np.where(std < 1e-4, 1.0, std).astype(np.float32)

        return cls(
            obs=obs_arr,
            actions=acts_arr,
            obs_mean=mean,
            obs_std=std,
            trajectories=trajs_list
        )


def normalize_obs(obs: jax.Array | np.ndarray, mean: jax.Array | np.ndarray, std: jax.Array | np.ndarray) -> jax.Array | np.ndarray:
    """Normalize observation with clip [-10, 10] matching firmware."""
    norm = (obs - mean) / std
    if isinstance(obs, jax.Array):
        return jnp.clip(norm, -10.0, 10.0)
    return np.clip(norm, -10.0, 10.0)


# ==============================================================================
# Phase 1: Supervised Behavior Cloning (BC) Warmup
# ==============================================================================

def train_behavior_cloning(dataset: SuperMimicDataset,
                           epochs: int = 150,
                           batch_size: int = 256,
                           lr: float = 1e-3,
                           seed: int = 42) -> Tuple[Any, Dict[str, float]]:
    """Train SuperMimicPolicy on dataset via supervised Behavior Cloning."""
    model = SuperMimicPolicy()
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, OBS_DIM), dtype=jnp.float32)
    variables = model.init(init_rng, dummy_obs)
    params = variables["params"]

    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)

    obs_norm_all = normalize_obs(dataset.obs, dataset.obs_mean, dataset.obs_std)
    acts_all = dataset.actions
    N = len(obs_norm_all)

    obs_norm_jax = jnp.asarray(obs_norm_all, dtype=jnp.float32)
    acts_jax = jnp.asarray(acts_all, dtype=jnp.float32)

    @jax.jit
    def bc_step(p, opt_s, o_batch, a_batch):
        def loss_fn(p_curr):
            pred_a = model.apply({"params": p_curr}, o_batch)
            return jnp.mean(jnp.square(pred_a - a_batch))
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_s = tx.update(grads, opt_s, p)
        p = optax.apply_updates(p, updates)
        return p, opt_s, loss

    t0 = time.perf_counter()
    history = []
    num_batches = max(1, N // batch_size)

    for ep in range(epochs):
        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, N)

        ep_loss = 0.0
        for b in range(num_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            params, opt_state, loss_val = bc_step(params, opt_state, obs_norm_jax[idx], acts_jax[idx])
            ep_loss += float(loss_val)

        avg_loss = ep_loss / num_batches
        history.append(avg_loss)

    elapsed = time.perf_counter() - t0
    stats = {
        "initial_loss": float(history[0]),
        "final_loss": float(history[-1]),
        "epochs": epochs,
        "elapsed_s": elapsed,
    }
    return params, stats


# ==============================================================================
# Phase 2: Differentiable BPTT Fine-Tuning
# ==============================================================================

def sim_step_jax(state_22: jax.Array, u_rpm: jax.Array, substeps: int = 4, sub_dt: float = 0.005) -> jax.Array:
    """Integrate 4 substeps of 5 ms RK4 for one 50 Hz control step."""
    dummy_key = jax.random.PRNGKey(0)
    def body(i, s):
        return rk4(s, u_rpm, dummy_key, sub_dt)
    return jax.lax.fori_loop(0, substeps, body, state_22)


def train_bptt(params: Any,
               dataset: SuperMimicDataset,
               epochs: int = 80,
               horizon: int = 20,
               batch_size: int = 32,
               lr: float = 3e-4,
               seed: int = 123) -> Tuple[Any, Dict[str, float]]:
    """Fine-tune SuperMimicPolicy by unrolling through differentiable quadrotor dynamics."""
    model = SuperMimicPolicy()
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)

    # Pre-extract trajectory windows of length horizon
    windows = []
    for traj in dataset.trajectories:
        xs_22 = traj["xs_22"]
        H = len(xs_22) - 1
        if H >= horizon + 1:
            for start in range(0, H - horizon, max(1, horizon // 2)):
                windows.append(xs_22[start:start + horizon + 1])

    if not windows:
        # Fallback if horizon is longer than trajectories
        horizon = min(len(t["xs_22"]) - 2 for t in dataset.trajectories)
        for traj in dataset.trajectories:
            windows.append(traj["xs_22"][:horizon + 1])

    windows_arr = np.array(windows, dtype=np.float32)  # (M, horizon+1, 22)
    M = len(windows_arr)
    windows_jax = jnp.asarray(windows_arr)

    obs_mean_jax = jnp.asarray(dataset.obs_mean, dtype=jnp.float32)
    obs_std_jax = jnp.asarray(dataset.obs_std, dtype=jnp.float32)

    def single_trajectory_bptt_loss(p, ref_traj, rng_key):
        """Unroll policy for `horizon` steps and compute tracking loss with recovery perturbations."""
        k1, k2, k3 = jax.random.split(rng_key, 3)
        p_noise = jax.random.normal(k1, shape=(3,)) * 0.02 # 2cm pos perturbation
        v_noise = jax.random.normal(k2, shape=(3,)) * 0.03 # 3cm/s vel perturbation
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

            # Tracking stage loss
            p_curr = s_curr[0:3]
            p_ref = ref_traj[t, 0:3]
            pos_err = jnp.sum(jnp.square(p_curr - p_ref))
            vel_err = jnp.sum(jnp.square(s_curr[12:15] - ref_traj[t, 12:15]))
            smooth_err = jnp.sum(jnp.square(a - a_prev))

            stage_loss = pos_err + 0.1 * vel_err + 0.01 * smooth_err
            return (s_next, a, key), stage_loss

        timesteps = jnp.arange(horizon)
        _, step_losses = jax.lax.scan(step_fn, (s0, a_init, k3), timesteps)
        return jnp.mean(step_losses)

    @jax.jit
    def bptt_batch_step(p, opt_s, batch_refs, batch_keys):
        def batch_loss(p_curr):
            losses = jax.vmap(lambda ref, k: single_trajectory_bptt_loss(p_curr, ref, k))(batch_refs, batch_keys)
            return jnp.mean(losses)
        loss, grads = jax.value_and_grad(batch_loss)(p)
        updates, opt_s = tx.update(grads, opt_s, p)
        p = optax.apply_updates(p, updates)
        return p, opt_s, loss

    rng = jax.random.PRNGKey(seed)
    history = []
    num_batches = max(1, M // batch_size)
    t0 = time.perf_counter()

    for ep in range(epochs):
        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, M)

        ep_loss = 0.0
        for b in range(num_batches):
            rng, k_b = jax.random.split(rng)
            b_keys = jax.random.split(k_b, batch_size)
            idx = perm[b * batch_size:(b + 1) * batch_size]
            params, opt_state, l_val = bptt_batch_step(params, opt_state, windows_jax[idx], b_keys)
            ep_loss += float(l_val)

        avg_loss = ep_loss / num_batches
        history.append(avg_loss)

    elapsed = time.perf_counter() - t0
    stats = {
        "initial_loss": float(history[0]),
        "final_loss": float(history[-1]),
        "epochs": epochs,
        "elapsed_s": elapsed,
        "horizon": horizon,
    }
    return params, stats


# ==============================================================================
# Phase 3: Sim2Real Closed-Loop Evaluation
# ==============================================================================

def _closed_loop_scan(params: Any,
                       ref_traj: jax.Array,
                       obs_mean: jax.Array,
                       obs_std: jax.Array,
                       rng_key: jax.Array,
                       add_noise: bool = False,
                       dr: bool = False) -> Tuple[jax.Array, jax.Array]:
    """JAX-scanned fast closed-loop trajectory simulation."""
    model = SuperMimicPolicy()
    s0 = ref_traj[0]
    a0 = jnp.zeros(ACT_DIM, dtype=jnp.float32)
    H = len(ref_traj) - 1

    def step_fn(carry, t):
        s_curr, a_prev, key_curr = carry
        p_target = ref_traj[t + 1, 0:3]
        o = compute_obs(s_curr, p_target, a_prev)

        if add_noise:
            key_curr, k_n1, k_n2 = jax.random.split(key_curr, 3)
            pos_n = jax.random.normal(k_n1, shape=(3,)) * 0.005  # 5 mm pos noise
            gyro_n = jax.random.normal(k_n2, shape=(3,)) * 0.005 # 0.005 rad/s gyro noise
            o = o.at[0:3].add(pos_n).at[15:18].add(gyro_n)

        o_norm = normalize_obs(o, obs_mean, obs_std)
        a = model.apply({"params": params}, o_norm)
        u_rpm = norm_action_to_rpm(a)

        key_curr, k_dyn = jax.random.split(key_curr)
        dyn_key = k_dyn if dr else jax.random.PRNGKey(0)

        def sub(i, s):
            return rk4(s, u_rpm, dyn_key, 0.005)

        s_next = jax.lax.fori_loop(0, 4, sub, s_curr)
        raw_err = jnp.linalg.norm(s_curr[0:3] - ref_traj[t, 0:3])
        pos_err = jnp.where(jnp.isnan(raw_err) | (raw_err > 5.0), 5.0, raw_err)
        return (s_next, a, key_curr), pos_err

    timesteps = jnp.arange(H)
    (s_final, _, _), pos_errs = jax.lax.scan(step_fn, (s0, a0, rng_key), timesteps)
    raw_fin = jnp.linalg.norm(s_final[0:3] - ref_traj[H, 0:3])
    final_err = jnp.where(jnp.isnan(raw_fin) | (raw_fin > 5.0), 5.0, raw_fin)
    return pos_errs, final_err

_jit_closed_loop = jax.jit(_closed_loop_scan, static_argnames=("add_noise", "dr"))


def evaluate_sim2real(params: Any,
                      dataset: SuperMimicDataset,
                      num_eval_trajs: int = 5,
                      seed: int = 999) -> Dict[str, Any]:
    """Test policy across multiple trajectories under nominal, sensor noise, and domain randomization."""
    obs_mean_jax = jnp.asarray(dataset.obs_mean, dtype=jnp.float32)
    obs_std_jax = jnp.asarray(dataset.obs_std, dtype=jnp.float32)

    trajs = dataset.trajectories[:min(num_eval_trajs, len(dataset.trajectories))]
    results = {
        "nominal": [],
        "sensor_noise": [],
        "domain_randomized": [],
    }

    base_rng = jax.random.PRNGKey(seed)

    for i, traj in enumerate(trajs):
        xs_ref_jax = jnp.asarray(traj["xs_22"], dtype=jnp.float32)
        base_rng, k_nom, k_noise, k_dr = jax.random.split(base_rng, 4)

        # 1. Nominal rollout
        errs_nom, fin_nom = _jit_closed_loop(params, xs_ref_jax, obs_mean_jax, obs_std_jax, k_nom, add_noise=False, dr=False)
        errs_nom_np = np.asarray(errs_nom)
        results["nominal"].append({
            "rmse_m": float(np.sqrt(np.mean(np.square(errs_nom_np)))),
            "max_err_m": float(np.max(errs_nom_np)),
            "final_err_m": float(fin_nom),
        })

        # 2. Sensor noise rollout (5 mm pos noise, 0.005 rad/s gyro noise)
        errs_noise, fin_noise = _jit_closed_loop(params, xs_ref_jax, obs_mean_jax, obs_std_jax, k_noise, add_noise=True, dr=False)
        errs_noise_np = np.asarray(errs_noise)
        results["sensor_noise"].append({
            "rmse_m": float(np.sqrt(np.mean(np.square(errs_noise_np)))),
            "max_err_m": float(np.max(errs_noise_np)),
            "final_err_m": float(fin_noise),
        })

        # 3. Domain randomized rollout (mass +10%, motor lag, thrust perturbation)
        errs_dr, fin_dr = _jit_closed_loop(params, xs_ref_jax, obs_mean_jax, obs_std_jax, k_dr, add_noise=False, dr=True)
        errs_dr_np = np.asarray(errs_dr)
        results["domain_randomized"].append({
            "rmse_m": float(np.sqrt(np.mean(np.square(errs_dr_np)))),
            "max_err_m": float(np.max(errs_dr_np)),
            "final_err_m": float(fin_dr),
        })

    summary = {
        "nominal_rmse_cm": float(np.mean([r["rmse_m"] for r in results["nominal"]]) * 100),
        "nominal_max_cm": float(np.mean([r["max_err_m"] for r in results["nominal"]]) * 100),
        "noise_rmse_cm": float(np.mean([r["rmse_m"] for r in results["sensor_noise"]]) * 100),
        "noise_max_cm": float(np.mean([r["max_err_m"] for r in results["sensor_noise"]]) * 100),
        "dr_rmse_cm": float(np.mean([r["rmse_m"] for r in results["domain_randomized"]]) * 100),
        "dr_max_cm": float(np.mean([r["max_err_m"] for r in results["domain_randomized"]]) * 100),
        "trajectories_evaluated": len(trajs),
        "details": results,
    }
    return summary


# ==============================================================================
# Phase 4: Firmware Export to CSV and matrices.h
# ==============================================================================

def export_firmware_header(params: Any,
                           obs_mean: np.ndarray,
                           obs_std: np.ndarray,
                           out_dir: str) -> Tuple[str, str]:
    """Export network weights and normalizer to CSV and C header file."""
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # 1. Save OBS_MEAN and OBS_STD
    np.savetxt(os.path.join(csv_dir, "OBS_MEAN.csv"), obs_mean.reshape(1, -1), delimiter=",", fmt="%.18f")
    np.savetxt(os.path.join(csv_dir, "OBS_STD.csv"), obs_std.reshape(1, -1), delimiter=",", fmt="%.18f")

    # 2. Extract weights and biases (transposed for C row-major layout)
    # hidden_0: Dense(64) from input(22) -> kernel shape (22, 64) -> transpose to (64, 22)
    # hidden_1: Dense(4) from hidden(64) -> kernel shape (64, 4) -> transpose to (4, 64)
    h0_w = np.array(params["hidden_0"]["kernel"]).T
    h0_b = np.array(params["hidden_0"]["bias"])
    h1_w = np.array(params["hidden_1"]["kernel"]).T
    h1_b = np.array(params["hidden_1"]["bias"])

    np.savetxt(os.path.join(csv_dir, "hidden_0_weights.csv"), h0_w, delimiter=",", fmt="%.7f")
    np.savetxt(os.path.join(csv_dir, "hidden_0_bias.csv"), h0_b.reshape(1, -1), delimiter=",", fmt="%.7f")
    np.savetxt(os.path.join(csv_dir, "hidden_1_weights.csv"), h1_w, delimiter=",", fmt="%.7f")
    np.savetxt(os.path.join(csv_dir, "hidden_1_bias.csv"), h1_b.reshape(1, -1), delimiter=",", fmt="%.7f")

    # 3. Call csvtohead to generate matrices.h
    matrices_h_path = os.path.join(out_dir, "matrices.h")
    from crazyflie_nn.tools.csvtohead import convert_folder_to_single_header
    convert_folder_to_single_header(csv_dir, matrices_h_path)

    return csv_dir, matrices_h_path


# ==============================================================================
# Checkpoint Management (Atomic & Crash-Safe)
# ==============================================================================

def save_checkpoint(checkpoint_dir: str,
                    name: str,
                    params: Any,
                    opt_state: Any,
                    epoch: int,
                    step: int,
                    val_loss: float,
                    obs_mean: np.ndarray,
                    obs_std: np.ndarray,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
    """Atomically save training checkpoint to avoid corruption on OOM/interrupt."""
    from flax import serialization
    os.makedirs(checkpoint_dir, exist_ok=True)

    params_bytes = np.frombuffer(serialization.to_bytes(params), dtype=np.uint8)
    opt_bytes = np.frombuffer(serialization.to_bytes(opt_state), dtype=np.uint8)

    meta_str = ""
    if metadata is not None:
        import json
        meta_str = json.dumps(metadata)

    tmp_file = os.path.join(checkpoint_dir, f"{name}.tmp.npz")
    final_file = os.path.join(checkpoint_dir, f"{name}.npz")

    np.savez_compressed(
        tmp_file,
        params_bytes=params_bytes,
        opt_bytes=opt_bytes,
        epoch=np.int32(epoch),
        step=np.int32(step),
        val_loss=np.float32(val_loss),
        obs_mean=obs_mean.astype(np.float32),
        obs_std=obs_std.astype(np.float32),
        metadata=meta_str
    )
    os.replace(tmp_file, final_file)
    return final_file


def load_checkpoint(checkpoint_path: str,
                    dummy_params: Any,
                    dummy_opt_state: Any) -> Dict[str, Any]:
    """Atomically load checkpoint from npz."""
    from flax import serialization
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with np.load(checkpoint_path, allow_pickle=False) as data:
        params_bytes = data["params_bytes"].tobytes()
        opt_bytes = data["opt_bytes"].tobytes()

        params = serialization.from_bytes(dummy_params, params_bytes)
        opt_state = serialization.from_bytes(dummy_opt_state, opt_bytes)
        epoch = int(data["epoch"])
        step = int(data["step"])
        val_loss = float(data["val_loss"])
        obs_mean = data["obs_mean"]
        obs_std = data["obs_std"]
        meta_str = str(data["metadata"])

    metadata = {}
    if meta_str:
        import json
        metadata = json.loads(meta_str)

    return {
        "params": params,
        "opt_state": opt_state,
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "metadata": metadata
    }
