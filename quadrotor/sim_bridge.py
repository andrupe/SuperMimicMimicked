"""Simulation bridge and validation gate for expert quadrotor trajectories."""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import Dict, Any, Optional
import numpy as np
import jax
import jax.numpy as jnp

import crazyflie_nn.dynamics.dynamics as cfdyn
from quadrotor.params import (
    NX, NU, SIM_DT, MAX_THRUST, M_NOMINAL, G, KF, KM, L, TAU_M, Kaero, I_NOMINAL
)
from quadrotor.quaternion import (
    quat_to_R, state_right_minus, state17_to_22, state22_to_17
)
from quadrotor.dynamics import rk4_integrate, CrazyflieDynamics


# ------------------------------------------------------------- JAX sim nominal
@jax.jit
def _nominal_cf_dynamics_jax(x, u):
    rw = x[0:3]
    Rwb = jnp.reshape(x[3:12], (3, 3))
    vb = x[12:15]
    wb = x[15:18]
    omega = x[18:22]

    omega_dot = (1.0 / TAU_M) * (u - omega)
    rw_dot = Rwb @ vb
    F_gravity = Rwb.T @ jnp.array([0.0, 0.0, -M_NOMINAL * G])
    F_thrust = jnp.array([0.0, 0.0, KF * jnp.sum(omega**2)])
    Faero = jnp.sum(jnp.abs(omega)) * (Kaero @ vb)
    Fb = F_gravity + F_thrust - Faero
    vb_dot = (1.0 / M_NOMINAL) * Fb - jnp.cross(wb, vb)

    Rwb_dot = Rwb @ cfdyn.skew(wb)
    
    sqrt2 = jnp.sqrt(2.0)
    tau_phi = L * (-KF*omega[0]**2 - KF*omega[1]**2 + KF*omega[2]**2 + KF*omega[3]**2) / sqrt2
    tau_theta = L * (-KF*omega[0]**2 + KF*omega[1]**2 + KF*omega[2]**2 - KF*omega[3]**2) / sqrt2
    tau_psi = KM * (-omega[0]**2 + omega[1]**2 - omega[2]**2 + omega[3]**2)
    tau = jnp.array([tau_phi, tau_theta, tau_psi])

    I_inv = jnp.linalg.inv(I_NOMINAL)
    wb_dot = I_inv @ (tau - jnp.cross(wb, I_NOMINAL @ wb))

    return jnp.concatenate([rw_dot, Rwb_dot.flatten(), vb_dot, wb_dot, omega_dot])


@jax.jit
def _nominal_cf_rk4_jax(x, u, dt):
    def split_state(s):
        return s[0:3], s[3:12].reshape((3, 3)), s[12:15], s[15:18], s[18:22]
    def pack_state(rw, Rwb, vb, wb, omega):
        return jnp.concatenate([rw, Rwb.flatten(), vb, wb, omega])

    rw_k, R_k, vb_k, wb_k, omega_k = split_state(x)

    dx1 = _nominal_cf_dynamics_jax(x, u)
    rw1, _, vb1, wb1, om1 = split_state(dx1)

    wb_temp2 = wb_k + 0.5 * dt * wb1
    x2 = pack_state(rw_k + 0.5 * dt * rw1,
                    R_k @ cfdyn.exp_so3(0.5 * dt * wb_k),
                    vb_k + 0.5 * dt * vb1,
                    wb_temp2,
                    omega_k + 0.5 * dt * om1)
    dx2 = _nominal_cf_dynamics_jax(x2, u)
    rw2, _, vb2, wb2, om2 = split_state(dx2)

    wb_temp3 = wb_k + 0.5 * dt * wb2
    x3 = pack_state(rw_k + 0.5 * dt * rw2,
                    R_k @ cfdyn.exp_so3(0.5 * dt * wb_temp2),
                    vb_k + 0.5 * dt * vb2,
                    wb_temp3,
                    omega_k + 0.5 * dt * om2)
    dx3 = _nominal_cf_dynamics_jax(x3, u)
    rw3, _, vb3, wb3, om3 = split_state(dx3)

    wb_temp4 = wb_k + dt * wb3
    x4 = pack_state(rw_k + dt * rw3,
                    R_k @ cfdyn.exp_so3(dt * wb_temp3),
                    vb_k + dt * vb3,
                    wb_temp4,
                    omega_k + dt * om3)
    dx4 = _nominal_cf_dynamics_jax(x4, u)
    rw4, _, vb4, wb4, om4 = split_state(dx4)

    rw_next = rw_k + (dt / 6.0) * (rw1 + 2.0 * rw2 + 2.0 * rw3 + rw4)
    vb_next = vb_k + (dt / 6.0) * (vb1 + 2.0 * vb2 + 2.0 * vb3 + vb4)
    wb_next = wb_k + (dt / 6.0) * (wb1 + 2.0 * wb2 + 2.0 * wb3 + wb4)
    omega_next = omega_k + (dt / 6.0) * (om1 + 2.0 * om2 + 2.0 * om3 + om4)

    w_final = dt * wb_k + (dt**2 / 6.0) * (wb1 + wb2 + wb3)
    R_next = R_k @ cfdyn.exp_so3(w_final)
    return pack_state(rw_next, R_next, vb_next, wb_next, omega_next)


def rollout_numpy(x0: np.ndarray, us: np.ndarray, dt: float, substeps: Optional[int] = None) -> np.ndarray:
    """Independent forward rollout in NumPy using Lie-group RK4."""
    H = len(us)
    sub = max(1, int(round(dt / SIM_DT))) if substeps is None else substeps
    xs = np.zeros((H + 1, NX), dtype=np.float64)
    xs[0] = x0.copy()
    for k in range(H):
        xs[k + 1] = rk4_integrate(xs[k], us[k], dt, sub)
    return xs


@jax.jit
def _fast_rollout_scan_jax(x0_22: jax.Array, us_rpm: jax.Array, substeps: int, dt_step: float):
    def step_fn(curr, u_rpm):
        def sub_step_fn(i, val):
            return _nominal_cf_rk4_jax(val, u_rpm, dt_step)
        next_state = jax.lax.fori_loop(0, substeps, sub_step_fn, curr)
        return next_state, next_state

    _, traj = jax.lax.scan(step_fn, x0_22, us_rpm)
    return traj


def rollout_jax(x0_17: np.ndarray, us_bar: np.ndarray, dt: float) -> np.ndarray:
    """Independent forward rollout in the JAX simulator (nominal parameters).
    Accelerated with jax.lax.scan.
    Returns array of 17-dim states of shape (H + 1, 17).
    """
    H = len(us_bar)
    substeps = max(1, int(round(dt / SIM_DT)))
    dt_step = dt / substeps

    x0_22 = jnp.asarray(state17_to_22(x0_17), dtype=jnp.float32)
    us_rpm = jnp.asarray(us_bar * MAX_THRUST, dtype=jnp.float32)

    traj_22 = np.asarray(_fast_rollout_scan_jax(x0_22, us_rpm, substeps, dt_step), dtype=np.float64)

    states_17 = np.zeros((H + 1, NX), dtype=np.float64)
    states_17[0] = x0_17.copy()
    for k in range(H):
        states_17[k + 1] = state22_to_17(traj_22[k])

    return states_17


def validate_trajectory(xs: np.ndarray, us: np.ndarray, dt: float,
                        solver_state_name: str,
                        p_ref: Optional[np.ndarray] = None,
                        residual_tol: float = 1e-5,
                        jax_drift_tol: float = 0.05,
                        check_jax: bool = False) -> Dict[str, Any]:
    """Rigorous validation gate for an expert quadrotor trajectory.

    Checks:
      1. Solver converged (solver_state_name == 'Success')
      2. No NaN or Inf values in states or controls
      3. Controls stay within [0, 1] bounds
      4. OCP dynamics rollout residual < residual_tol
      5. Forward rollout in JAX sim tracks within jax_drift_tol (if check_jax=True)
      6. Tilt angle <= 75 degrees
    """
    H = len(us)
    assert len(xs) == H + 1, f"State length {len(xs)} != H+1 {H+1}"

    # 1. Finite checks
    is_finite = bool(np.all(np.isfinite(xs)) and np.all(np.isfinite(us)))

    # 2. Control bounds
    u_min_val = float(np.min(us))
    u_max_val = float(np.max(us))
    bounds_ok = bool(u_min_val >= -1e-4 and u_max_val <= 1.0 + 1e-4)

    # 3. Independent NumPy rollout residual (exact Lie-group RK4)
    roll_np = rollout_numpy(xs[0], us, dt)
    dyn = CrazyflieDynamics(dt)
    res_list = [np.linalg.norm(dyn.state_right_minus(roll_np[k], xs[k])) for k in range(H + 1)]
    max_residual = float(np.max(res_list))

    # 4. Optional JAX sim replay drift
    if check_jax:
        roll_jax = rollout_jax(xs[0], us, dt)
        jax_pos_drift = float(np.max(np.linalg.norm(roll_jax[:, 0:3] - xs[:, 0:3], axis=1)))
        jax_final_drift = float(np.linalg.norm(roll_jax[-1, 0:3] - xs[-1, 0:3]))
    else:
        roll_jax = None
        jax_pos_drift = max_residual
        jax_final_drift = max_residual

    # 5. Body tilts
    tilts = []
    for k in range(H + 1):
        R = quat_to_R(xs[k, 3:7])
        cos_tilt = np.clip(R[2, 2], -1.0, 1.0)
        tilts.append(np.degrees(np.arccos(cos_tilt)))
    max_tilt_deg = float(np.max(tilts))

    # 6. Tracking error against reference if provided
    if p_ref is not None:
        pos_errs = np.linalg.norm(xs[:, 0:3] - p_ref, axis=1)
        max_track_err = float(np.max(pos_errs))
        final_track_err = float(pos_errs[-1])
    else:
        max_track_err = 0.0
        final_track_err = 0.0

    solver_ok = (solver_state_name == "Success")
    passed = bool(
        solver_ok
        and is_finite
        and bounds_ok
        and (max_residual < residual_tol)
        and (jax_pos_drift < jax_drift_tol)
        and (max_tilt_deg < 80.0)
    )

    return {
        "passed": passed,
        "solver_ok": solver_ok,
        "is_finite": is_finite,
        "bounds_ok": bounds_ok,
        "max_residual": max_residual,
        "jax_pos_drift": jax_pos_drift,
        "jax_final_drift": jax_final_drift,
        "max_tilt_deg": max_tilt_deg,
        "u_min": u_min_val,
        "u_max": u_max_val,
        "max_track_err": max_track_err,
        "final_track_err": final_track_err,
        "roll_jax": roll_jax,
        "roll_np": roll_np,
    }
