"""Vectorized Quadrotor Tracking Simulator and Return-Based Reset Sampling.

Matches SuperMimic Paper (Section 3.1 & Appendix D.1):
  - Fast JAX-vectorized (vmap) 50 Hz closed-loop simulation with 4 substeps of 5 ms RK4.
  - Parameterized dynamics with physical context xi (mass, inertia, rotor thrusts, rotor drag, motor lag).
  - True nominal dynamics when xi = 0 (no spurious gradient bias).
  - Configurable sensor noise (5 mm pos, 0.005 rad/s gyro, 1 cm/s vel).
  - Return-based bin reset sampling (50% uniform / 40% softmax(-25 G/Gmax) / 10% past-softmax).
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import h5py
import jax
import jax.numpy as jnp

from quadrotor.params import (
    M_NOMINAL, L, KF, KM, G, GRAV, I_NOMINAL, I_INV,
    Kaero, TAU_M, MAX_THRUST, HOVER_RPM, SIM_DT
)
from quadrotor.rotations_jax import (
    skew_jax, so3_exp_jax, yaw_mat_jax, mat_to_6d_jax
)
from quadrotor.world_model import (
    QuadrotorState, pack_state_22, unpack_state_22,
    THRUST_SCALE, XI_DIM
)
from quadrotor.policy_critic import (
    compute_proprio_obs, compute_goal_features, compute_analytic_reward,
    GOAL_K
)


# ------------------------------------------------------------------------------
# Core Parameterized JAX Continuous Dynamics & RK4
# ------------------------------------------------------------------------------

def parameterized_dynamics_jax(x: jax.Array, u_rpm: jax.Array, xi: jax.Array) -> jax.Array:
    """Continuous dynamics parameterized by physical context xi in R^10 or R^14.
    
    xi layout:
      0: dM (mass fraction, e.g. [-0.15, 0.15])
      1..3: dI (inertia diag fraction, e.g. [-0.2, 0.2]^3)
      4..7: dKF (rotor thrust fractions, e.g. [-0.12, 0.12]^4)
      8: dKM (rotor moment fraction, e.g. [-0.15, 0.15])
      9: dtau (motor lag fraction, e.g. [-0.25, 0.35])
      10..11: CoM dx, dy offset (m, e.g. [-0.003, 0.003])
      12: CoM dz offset (m, e.g. [-0.004, 0.004])
      13: Dynamic voltage sag coefficient (e.g. [0.0, 0.08])
    """
    # Ensure xi is at least 14 dims using static Python shape
    pad_len = max(0, 14 - int(xi.shape[-1]))
    if pad_len > 0:
        xi_pad = jnp.pad(xi, (0, pad_len))
    else:
        xi_pad = xi

    # Unpack state
    rw = x[0:3]
    Rwb = x[3:12].reshape((3, 3))
    vb = x[12:15]
    wb = x[15:18]
    omega = x[18:22]

    # Context perturbations
    M = M_NOMINAL * (1.0 + xi_pad[0])
    diag_pert = 1.0 + xi_pad[1:4]
    I_base = jnp.asarray(I_NOMINAL, dtype=jnp.float32)
    I = I_base.at[0, 0].multiply(diag_pert[0]).at[1, 1].multiply(diag_pert[1]).at[2, 2].multiply(diag_pert[2])
    I_inv = jnp.linalg.inv(I)

    # Dynamic battery sag: high throttle load temporarily drops motor thrust efficiency
    sag_factor = 1.0 - xi_pad[13] * (jnp.mean(omega) / MAX_THRUST)
    kf_rot = KF * (1.0 + xi_pad[4:8]) * sag_factor
    km_rot = KM * (1.0 + xi_pad[8]) * sag_factor
    tau_m = TAU_M * (1.0 + xi_pad[9])

    # Motor dynamics
    omega_dot = (1.0 / tau_m) * (u_rpm - omega)

    # World position rate
    rw_dot = Rwb @ vb

    # Forces in body frame
    F_gravity = Rwb.T @ jnp.array([0.0, 0.0, -M * G])
    F_thrust_mag = jnp.sum(kf_rot * (omega ** 2))
    F_thrust = jnp.array([0.0, 0.0, F_thrust_mag])
    Faero = jnp.sum(jnp.abs(omega)) * (Kaero @ vb)
    Fb = F_gravity + F_thrust - Faero
    vb_dot = (1.0 / M) * Fb - jnp.cross(wb, vb)

    # Moments in body frame
    sqrt2 = jnp.sqrt(2.0)
    om_sq = omega ** 2
    tau_phi   = L * (-kf_rot[0] * om_sq[0] - kf_rot[1] * om_sq[1] + kf_rot[2] * om_sq[2] + kf_rot[3] * om_sq[3]) / sqrt2
    tau_theta = L * (-kf_rot[0] * om_sq[0] + kf_rot[1] * om_sq[1] + kf_rot[2] * om_sq[2] - kf_rot[3] * om_sq[3]) / sqrt2
    tau_psi   = km_rot * (-om_sq[0] + om_sq[1] - om_sq[2] + om_sq[3])

    # Center of Mass (CoM) offset inducing parasitic pitching/rolling moments
    # tau_com = r_com x F_thrust = [dy * F_thrust_mag, -dx * F_thrust_mag, 0]
    dx_com = xi_pad[10]
    dy_com = xi_pad[11]
    tau_phi += dy_com * F_thrust_mag
    tau_theta -= dx_com * F_thrust_mag

    tau = jnp.array([tau_phi, tau_theta, tau_psi])

    wb_dot = I_inv @ (tau - jnp.cross(wb, I @ wb))
    Rwb_dot = (Rwb @ skew_jax(wb)).flatten()

    return jnp.concatenate([rw_dot, Rwb_dot, vb_dot, wb_dot, omega_dot])


def rk4_step_jax(x: jax.Array, u_rpm: jax.Array, xi: jax.Array, dt: float) -> jax.Array:
    """Exact Lie-group RK4 integrator for quadrotor SE(3) dynamics."""
    def split(s):
        return s[0:3], s[3:12].reshape((3, 3)), s[12:15], s[15:18], s[18:22]
    def pack(rw, R, vb, wb, om):
        return jnp.concatenate([rw, R.flatten(), vb, wb, om])

    rw_k, R_k, vb_k, wb_k, om_k = split(x)

    dx1 = parameterized_dynamics_jax(x, u_rpm, xi)
    rw_d1, _, vb_d1, wb_d1, om_d1 = split(dx1)

    wb_temp2 = wb_k + 0.5 * dt * wb_d1
    x2 = pack(rw_k + 0.5 * dt * rw_d1,
              R_k @ so3_exp_jax(0.5 * dt * wb_k),
              vb_k + 0.5 * dt * vb_d1,
              wb_temp2,
              om_k + 0.5 * dt * om_d1)
    dx2 = parameterized_dynamics_jax(x2, u_rpm, xi)
    rw_d2, _, vb_d2, wb_d2, om_d2 = split(dx2)

    wb_temp3 = wb_k + 0.5 * dt * wb_d2
    x3 = pack(rw_k + 0.5 * dt * rw_d2,
              R_k @ so3_exp_jax(0.5 * dt * wb_temp2),
              vb_k + 0.5 * dt * vb_d2,
              wb_temp3,
              om_k + 0.5 * dt * om_d2)
    dx3 = parameterized_dynamics_jax(x3, u_rpm, xi)
    rw_d3, _, vb_d3, wb_d3, om_d3 = split(dx3)

    wb_temp4 = wb_k + dt * wb_d3
    x4 = pack(rw_k + dt * rw_d3,
              R_k @ so3_exp_jax(dt * wb_temp3),
              vb_k + dt * vb_d3,
              wb_temp4,
              om_k + dt * om_d3)
    dx4 = parameterized_dynamics_jax(x4, u_rpm, xi)
    rw_d4, _, vb_d4, wb_d4, om_d4 = split(dx4)

    rw_next = rw_k + (dt / 6.0) * (rw_d1 + 2.0 * rw_d2 + 2.0 * rw_d3 + rw_d4)
    vb_next = vb_k + (dt / 6.0) * (vb_d1 + 2.0 * vb_d2 + 2.0 * vb_d3 + vb_d4)
    wb_next = wb_k + (dt / 6.0) * (wb_d1 + 2.0 * wb_d2 + 2.0 * wb_d3 + wb_d4)
    om_next = om_k + (dt / 6.0) * (om_d1 + 2.0 * om_d2 + 2.0 * om_d3 + om_d4)

    w_final = dt * wb_k + (dt ** 2 / 6.0) * (wb_d1 + wb_d2 + wb_d3)
    R_next = R_k @ so3_exp_jax(w_final)

    return pack(rw_next, R_next, vb_next, wb_next, om_next)


def sim_step_multi_substeps(x: jax.Array, u_rpm: jax.Array, xi: jax.Array, substeps: int = 4, sub_dt: float = 0.005) -> jax.Array:
    """Run `substeps` RK4 substeps for one 50 Hz control period."""
    def body(i, s):
        return rk4_step_jax(s, u_rpm, xi, sub_dt)
    return jax.lax.fori_loop(0, substeps, body, x)


# ------------------------------------------------------------------------------
# Trajectory Reference Library & Return-Based Reset Sampler
# ------------------------------------------------------------------------------

class MotionReferenceLibrary:
    """Loads and holds reference trajectories from HDF5 corpus with binning for reset sampling."""

    def __init__(self, corpus_path: str, bin_duration_s: float = 1.0, dt: float = 0.02):
        self.corpus_path = corpus_path
        self.dt = dt
        self.bin_steps = int(round(bin_duration_s / dt))

        self.trajectories = []
        self.traj_lengths = []
        self.task_types = []
        self.names = []
        self.bins = []  # List of (traj_id, start_frame, end_frame)

        with h5py.File(corpus_path, "r") as f:
            if "trajectories" in f:
                for k in sorted(list(f["trajectories"].keys())):
                    grp = f["trajectories"][k]
                    xs = grp["states"][:]
                    H = len(xs) - 1
                    if xs.shape[-1] == 22:
                        xs_22 = xs.astype(np.float32)
                    elif xs.shape[-1] == 17:
                        from quadrotor.quaternion import state17_to_22
                        xs_22 = np.zeros((H + 1, 22), dtype=np.float32)
                        for t in range(H + 1):
                            xs_22[t] = state17_to_22(xs[t])
                    else:
                        raise ValueError(f"Unsupported state dim {xs.shape[-1]}")

                    self.trajectories.append(xs_22)
                    self.traj_lengths.append(H)
                    self.task_types.append(grp.attrs.get("task_type", "generic"))
                    self.names.append(grp.attrs.get("name", k))

        self.num_trajs = len(self.trajectories)
        # Create 1-second bins across all trajectories
        for t_idx, H in enumerate(self.traj_lengths):
            for st in range(0, H - 15, max(1, self.bin_steps // 2)):
                end = min(st + self.bin_steps, H)
                self.bins.append((t_idx, st, end))

        self.num_bins = len(self.bins)
        # Track returns for each bin (for return-based reset sampling)
        self.bin_G = np.zeros(self.num_bins, dtype=np.float32)
        self.bin_w_avg = np.full(self.num_bins, 1.0 / self.num_bins, dtype=np.float32)
        self.G_max = 2.0 / (1.0 - 0.99) # Theoretical max return estimate

    def update_bin_returns(self, bin_indices: np.ndarray, returns: np.ndarray, old_weight: float = 0.8):
        """Update EMA returns for sampled bins."""
        for b, ret in zip(bin_indices, returns):
            if 0 <= b < self.num_bins:
                self.bin_G[b] = old_weight * self.bin_G[b] + (1.0 - old_weight) * ret

    def sample_reset_frames(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample reset points using SuperMimic mixture: 50% uniform, 40% softmax(-25 G/Gmax), 10% past-avg."""
        # Softmax on negative return
        scaled_G = -25.0 * (self.bin_G / max(self.G_max, 1e-3))
        scaled_G -= np.max(scaled_G)
        soft = np.exp(scaled_G)
        soft /= np.sum(soft)

        self.bin_w_avg = 0.999 * self.bin_w_avg + 0.001 * soft
        past = self.bin_w_avg / np.sum(self.bin_w_avg)

        # Mixture probability
        probs = 0.5 * (1.0 / self.num_bins) + 0.4 * soft + 0.1 * past
        probs /= np.sum(probs)

        chosen_bins = rng.choice(self.num_bins, size=n, p=probs)
        chosen_trajs = np.zeros(n, dtype=np.int32)
        chosen_frames = np.zeros(n, dtype=np.int32)

        for i, b_idx in enumerate(chosen_bins):
            t_idx, st, end = self.bins[b_idx]
            chosen_trajs[i] = t_idx
            chosen_frames[i] = rng.integers(st, max(st + 1, end - 10))

        return chosen_bins, chosen_trajs, chosen_frames

    def get_reference_window(self, traj_id: int, start_frame: int, length: int) -> np.ndarray:
        """Fetch length-step reference slice from trajectory."""
        traj = self.trajectories[traj_id]
        H = len(traj)
        if start_frame + length <= H:
            return traj[start_frame:start_frame + length]
        # Pad with final state if trajectory ends
        pad_count = (start_frame + length) - H
        window = np.zeros((length, 22), dtype=np.float32)
        avail = H - start_frame
        window[:avail] = traj[start_frame:]
        window[avail:] = traj[-1]
        return window
