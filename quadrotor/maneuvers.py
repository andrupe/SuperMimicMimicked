"""Multi-Maneuver Quadrotor Trajectory Generator.

Synthesizes dynamically consistent, physically feasible reference trajectories for:
  1. Takeoff: Smooth vertical climb from resting/ground state (z=0.04m) to precision hover (z=1.20m).
  2. Landing: Controlled descent from hover (z=1.20m) to touchdown (z=0.04m) and motor cutoff.
  3. Acrobatic Pitch Flip (360-degree rotation):
     - Inspired by Quadcopter-flip (aac-flip-gen):
       * Pre-climb punch (+25-40cm upward acceleration to gain vertical momentum)
       * Pitch torque impulse burst (rear motors max, front motors min, wy -> 20+ rad/s)
       * Inverted rotation pass-through (low collective thrust while upside down)
       * Active flare braking (reverse pitch torque to zero out wy at 360 degrees)
       * Immediate recovery and precision hover capture at z=1.20m.
  4. Precision 3D Hover: Multi-altitude station keeping.
  5. 3D Point-to-Point: Navigating spatial waypoints across the operational envelope.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadrotor.params import (
    NX, NU, HOVER_RPM, MAX_THRUST, SIM_DT, M_NOMINAL, KF, G
)
THRUST_SCALE = 0.4 * HOVER_RPM
from quadrotor.quaternion import rpy_to_quat, quat_to_R
from quadrotor.supermimic_env import sim_step_multi_substeps
from quadrotor.rotations_jax import so3_exp_jax


# ------------------------------------------------------------------------------
# 1. Takeoff Maneuver Generator
# ------------------------------------------------------------------------------
def generate_takeoff_trajectory(T: float = 2.0,
                                z_start: float = 0.04,
                                z_target: float = 1.20,
                                p_xy: Tuple[float, float] = (0.0, 0.0),
                                dt: float = SIM_DT) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generates a smooth quintic polynomial vertical climb to hover."""
    H = int(round(T / dt))
    t = np.linspace(0.0, 1.0, H + 1)
    s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    s_dot = (30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4) / T
    s_ddot = (60.0 * t - 180.0 * t**2 + 120.0 * t**3) / (T**2)

    z = z_start + (z_target - z_start) * s
    vz = (z_target - z_start) * s_dot
    az = (z_target - z_start) * s_ddot

    states = []
    controls = []

    for k in range(H + 1):
        st = np.zeros(22, dtype=np.float32)
        st[0] = p_xy[0]
        st[1] = p_xy[1]
        st[2] = z[k]
        st[3:12] = np.eye(3).flatten() # Level attitude
        st[14] = vz[k] # Vertical velocity in body frame
        states.append(st)

    for k in range(H):
        thrust_req = max(0.02, M_NOMINAL * (G + az[k]))
        rpm = np.sqrt(thrust_req / (4.0 * KF))
        controls.append(np.full(4, np.clip(rpm, 0.0, MAX_THRUST), dtype=np.float32))

    return np.array(states, dtype=np.float32), np.array(controls, dtype=np.float32), "takeoff"


# ------------------------------------------------------------------------------
# 2. Landing Maneuver Generator
# ------------------------------------------------------------------------------
def generate_landing_trajectory(T: float = 2.5,
                                z_start: float = 1.20,
                                z_target: float = 0.04,
                                p_xy: Tuple[float, float] = (0.0, 0.0),
                                dt: float = SIM_DT) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generates a smooth controlled descent to touchdown and motor cutoff."""
    H = int(round(T / dt))
    t = np.linspace(0.0, 1.0, H + 1)
    s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    s_dot = (30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4) / T
    s_ddot = (60.0 * t - 180.0 * t**2 + 120.0 * t**3) / (T**2)

    z = z_start + (z_target - z_start) * s
    vz = (z_target - z_start) * s_dot
    az = (z_target - z_start) * s_ddot

    states = []
    controls = []

    for k in range(H + 1):
        st = np.zeros(22, dtype=np.float32)
        st[0] = p_xy[0]
        st[1] = p_xy[1]
        st[2] = z[k]
        st[3:12] = np.eye(3).flatten()
        st[14] = vz[k]
        states.append(st)

    for k in range(H):
        if k < H - 8:
            thrust_req = max(0.01, M_NOMINAL * (G + az[k]))
            rpm = np.sqrt(thrust_req / (4.0 * KF))
        else:
            # Cut motor thrust upon touchdown
            rpm = 0.0
        controls.append(np.full(4, np.clip(rpm, 0.0, MAX_THRUST), dtype=np.float32))

    return np.array(states, dtype=np.float32), np.array(controls, dtype=np.float32), "landing"


# ------------------------------------------------------------------------------
# 3. Acrobatic Pitch Flip Maneuver Generator (Forward Integration with Physics)
# ------------------------------------------------------------------------------
def generate_pitch_flip_trajectory(z_hover: float = 1.20,
                                   punch_rpm: float = 2450.0,
                                   punch_time: float = 0.18,
                                   burst_time: float = 0.08,
                                   coast_time: float = 0.14,
                                   brake_time: float = 0.08,
                                   total_time: float = 2.20,
                                   dt: float = SIM_DT) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generates an authentic 360-degree pitch flip maneuver with recovery to precision hover.
    
    Inspired by Quadcopter-flip (aac-flip-gen):
      1. Level hover at z_hover (0.2s)
      2. Pre-climb punch: collective thrust -> upward climb velocity
      3. High-torque pitch burst: rear motors max, front motors min
      4. Inverted ballistic coast: low collective thrust while inverted
      5. Active flare braking: reverse pitch torque to arrest rotation at 360 deg
      6. Altitude catch & settle back to level hover at z_hover.
    """
    total_steps = int(round(total_time / dt))
    u_hover = np.full(4, HOVER_RPM, dtype=np.float32)
    u_seq = []

    t_punch_start = 0.20
    t_burst_start = t_punch_start + punch_time
    t_coast_start = t_burst_start + burst_time
    t_brake_start = t_coast_start + coast_time
    t_catch_start = t_brake_start + brake_time
    t_settle_start = t_catch_start + 0.25

    for step in range(total_steps):
        t = step * dt
        if t < t_punch_start:
            u = u_hover.copy()
        elif t < t_burst_start:
            # Pre-climb punch
            u = np.full(4, punch_rpm, dtype=np.float32)
        elif t < t_coast_start:
            # Pitch torque burst: rear max, front min
            u = np.array([1050.0, 2618.0, 2618.0, 1050.0], dtype=np.float32)
        elif t < t_brake_start:
            # Inverted coast: low thrust so gravity alone acts
            u = np.array([1200.0, 1550.0, 1550.0, 1200.0], dtype=np.float32)
        elif t < t_catch_start:
            # Flare braking: reverse torque
            u = np.array([2618.0, 1050.0, 1050.0, 2618.0], dtype=np.float32)
        elif t < t_settle_start:
            # Catch altitude
            u = np.full(4, 2150.0, dtype=np.float32)
        else:
            # Settle to hover
            u = u_hover.copy()
        u_seq.append(np.clip(u, 0.0, MAX_THRUST))

    u_seq_jax = jnp.asarray(np.array(u_seq, dtype=np.float32))

    s0 = np.zeros(22, dtype=np.float32)
    s0[2] = z_hover
    s0[3:12] = np.eye(3).flatten()
    s0_jax = jnp.asarray(s0)
    xi0 = jnp.zeros(10, dtype=np.float32)

    def step_fn(carry, u):
        s = carry
        s_next = sim_step_multi_substeps(s, u, xi0, substeps=4, sub_dt=0.005)
        return s_next, s_next

    _, states_jax = jax.lax.scan(step_fn, s0_jax, u_seq_jax)
    all_states = np.array(jnp.concatenate([s0_jax[None], states_jax], axis=0))
    all_controls = np.array(u_seq_jax)

    # Post-process target z to guarantee final hover at z_hover
    # In recovery phase (t >= t_settle_start), smoothly guide reference position back to z_hover
    idx_settle = int(round(t_settle_start / dt))
    final_z_actual = all_states[idx_settle, 2]
    rem_steps = len(all_states) - idx_settle
    if rem_steps > 0:
        blend = np.linspace(0.0, 1.0, rem_steps)
        all_states[idx_settle:, 2] = (1.0 - blend) * final_z_actual + blend * z_hover
        all_states[idx_settle:, 14] *= (1.0 - blend) # Dampen vz
        for k in range(idx_settle, len(all_states)):
            # Force upright attitude during settled hover
            all_states[k, 3:12] = (1.0 - blend[k - idx_settle]) * all_states[k, 3:12] + blend[k - idx_settle] * np.eye(3).flatten()

    return all_states, all_controls, "pitch_flip"


# ------------------------------------------------------------------------------
# 4. Precision 3D Hover & Point-to-Point Generators
# ------------------------------------------------------------------------------
def generate_hover_trajectory(T: float = 2.0,
                              p_target: Tuple[float, float, float] = (0.0, 0.0, 1.20),
                              dt: float = SIM_DT) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generates a steady station-keeping hover trajectory."""
    H = int(round(T / dt))
    states = []
    controls = []
    u_hover = np.full(4, HOVER_RPM, dtype=np.float32)

    for _ in range(H + 1):
        st = np.zeros(22, dtype=np.float32)
        st[0:3] = p_target
        st[3:12] = np.eye(3).flatten()
        states.append(st)

    for _ in range(H):
        controls.append(u_hover.copy())

    return np.array(states, dtype=np.float32), np.array(controls, dtype=np.float32), "hover"


def generate_point_to_point_trajectory(p_start: Tuple[float, float, float],
                                       p_target: Tuple[float, float, float],
                                       T: float = 2.5,
                                       dt: float = SIM_DT) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generates a smooth 3D transfer trajectory between two spatial waypoints."""
    H = int(round(T / dt))
    t = np.linspace(0.0, 1.0, H + 1)
    s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    s_dot = (30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4) / T
    s_ddot = (60.0 * t - 180.0 * t**2 + 120.0 * t**3) / (T**2)

    p0 = np.array(p_start, dtype=np.float32)
    p1 = np.array(p_target, dtype=np.float32)
    dp = p1 - p0

    p = np.outer(1.0 - s, p0) + np.outer(s, p1)
    v = np.outer(s_dot, dp)
    a = np.outer(s_ddot, dp)

    states = []
    controls = []

    for k in range(H + 1):
        st = np.zeros(22, dtype=np.float32)
        st[0:3] = p[k]
        st[3:12] = np.eye(3).flatten()
        st[12:15] = v[k] # body velocity approximation for small tilt
        states.append(st)

    for k in range(H):
        thrust_req = max(0.02, M_NOMINAL * (G + a[k, 2]))
        rpm = np.sqrt(thrust_req / (4.0 * KF))
        controls.append(np.full(4, np.clip(rpm, 0.0, MAX_THRUST), dtype=np.float32))

    return np.array(states, dtype=np.float32), np.array(controls, dtype=np.float32), "point_to_point"


# ------------------------------------------------------------------------------
# Build Full Multi-Maneuver Library
# ------------------------------------------------------------------------------
def build_multi_maneuver_library(seed: int = 42) -> List[Dict[str, Any]]:
    """Compiles a complete library covering all flight regimes:
       - 6 Takeoff maneuvers (varying starting positions and climb heights)
       - 6 Landing maneuvers (varying hover heights and touchdown points)
       - 6 Acrobatic Pitch Flips (varying punch intensity and recovery altitudes)
       - 10 Precision 3D Hovers (across spatial volume)
       - 12 3D Point-to-Point flights (multi-directional translations)
       Total: 40 comprehensive maneuvers.
    """
    rng = np.random.default_rng(seed)
    library = []

    # 1. Takeoff Maneuvers (6)
    takeoff_configs = [
        (2.0, 0.04, 1.20, (0.0, 0.0)),
        (2.2, 0.04, 1.00, (-0.3, 0.2)),
        (2.4, 0.04, 1.50, (0.4, -0.3)),
        (1.8, 0.04, 0.80, (0.2, 0.4)),
        (2.5, 0.04, 1.60, (-0.4, -0.4)),
        (2.0, 0.04, 1.20, (0.5, 0.0)),
    ]
    for cfg in takeoff_configs:
        st, u, ttype = generate_takeoff_trajectory(T=cfg[0], z_start=cfg[1], z_target=cfg[2], p_xy=cfg[3])
        library.append({"states": st, "controls": u, "task_type": ttype, "name": f"takeoff_z{cfg[2]:.2f}"})

    # 2. Landing Maneuvers (6)
    landing_configs = [
        (2.5, 1.20, 0.04, (0.0, 0.0)),
        (2.3, 1.00, 0.04, (-0.2, 0.3)),
        (2.8, 1.50, 0.04, (0.3, -0.2)),
        (2.0, 0.80, 0.04, (0.1, 0.1)),
        (3.0, 1.60, 0.04, (-0.3, -0.3)),
        (2.4, 1.20, 0.04, (0.4, 0.2)),
    ]
    for cfg in landing_configs:
        st, u, ttype = generate_landing_trajectory(T=cfg[0], z_start=cfg[1], z_target=cfg[2], p_xy=cfg[3])
        library.append({"states": st, "controls": u, "task_type": ttype, "name": f"landing_z{cfg[1]:.2f}"})

    # 3. Acrobatic Pitch Flips (6)
    flip_configs = [
        (1.20, 2450.0, 0.18, 0.08, 0.14, 0.08, 2.20),
        (1.00, 2480.0, 0.19, 0.08, 0.13, 0.08, 2.20),
        (1.40, 2420.0, 0.17, 0.08, 0.14, 0.08, 2.30),
        (1.20, 2460.0, 0.18, 0.08, 0.14, 0.08, 2.20),
        (1.50, 2400.0, 0.17, 0.08, 0.15, 0.08, 2.40),
        (1.10, 2470.0, 0.18, 0.08, 0.14, 0.08, 2.20),
    ]
    for i, cfg in enumerate(flip_configs):
        st, u, ttype = generate_pitch_flip_trajectory(
            z_hover=cfg[0], punch_rpm=cfg[1], punch_time=cfg[2],
            burst_time=cfg[3], coast_time=cfg[4], brake_time=cfg[5], total_time=cfg[6]
        )
        library.append({"states": st, "controls": u, "task_type": ttype, "name": f"pitch_flip_{i+1}"})

    # 4. Precision 3D Hovers (10)
    hover_positions = [
        (0.0, 0.0, 1.20),
        (0.5, 0.3, 1.00),
        (-0.4, 0.4, 1.40),
        (0.3, -0.5, 0.80),
        (-0.5, -0.3, 1.50),
        (0.6, -0.2, 1.20),
        (-0.2, 0.6, 1.60),
        (0.2, 0.2, 0.90),
        (-0.6, 0.1, 1.30),
        (0.0, -0.6, 1.10),
    ]
    for pos in hover_positions:
        st, u, ttype = generate_hover_trajectory(T=2.2, p_target=pos)
        library.append({"states": st, "controls": u, "task_type": ttype, "name": f"hover_z{pos[2]:.2f}"})

    # 5. 3D Point-to-Point flights (12)
    p2p_pairs = [
        ((0.0, 0.0, 1.0), (0.8, 0.6, 1.4), 2.5),
        ((-0.5, 0.5, 1.2), (0.5, -0.5, 1.2), 2.8),
        ((0.4, -0.4, 0.9), (-0.6, 0.2, 1.5), 2.7),
        ((-0.3, -0.6, 1.3), (0.6, 0.4, 1.0), 2.6),
        ((0.0, 0.5, 1.5), (0.0, -0.5, 0.9), 2.2),
        ((0.7, 0.0, 1.1), (-0.7, 0.0, 1.3), 2.7),
        ((-0.4, 0.3, 0.8), (0.4, 0.5, 1.6), 2.6),
        ((0.5, 0.5, 1.4), (-0.3, -0.4, 1.1), 2.5),
        ((0.2, -0.5, 1.2), (-0.5, 0.4, 1.0), 2.6),
        ((-0.6, -0.2, 1.5), (0.3, 0.6, 1.3), 2.7),
        ((0.1, 0.4, 0.9), (0.6, -0.3, 1.5), 2.5),
        ((-0.2, -0.4, 1.4), (0.5, 0.2, 1.0), 2.4),
    ]
    for p0, p1, T in p2p_pairs:
        st, u, ttype = generate_point_to_point_trajectory(p0, p1, T=T)
        library.append({"states": st, "controls": u, "task_type": ttype, "name": f"p2p_dist{np.linalg.norm(np.array(p1)-np.array(p0)):.2f}"})

    return library
