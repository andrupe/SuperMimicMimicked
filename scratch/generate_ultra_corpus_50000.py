"""Ultra-Corpus Generator for SuperMimic Bitcraze Crazyflie 2.1.

Scales reference data to 50,000 diverse, dynamically plausible, physically verified
reference trajectories (~7.5 Million transition steps).

Categories (17 Trajectory Families, Percentage-Weighted):
  1.  point_to_point     (12%) - Multi-Waypoint 3D Splines
  2.  figure8            ( 8%) - Parametric 3D Lissajous & Lemniscates
  3.  helix              ( 8%) - Parametric 3D Helices & Spirals
  4.  hover              ( 7%) - Precision 3D Hovers & Recovery
  5.  takeoff            ( 4%) - Vertical Takeoffs
  6.  landing            ( 4%) - Controlled Touchdown Landings
  7.  slalom             ( 7%) - High-Speed Slaloms & Agile Dashes
  8.  circular_orbit     ( 5%) - Banked Constant-Radius Orbits & POI Circles
  9.  step_dash_brake    ( 5%) - Sprint-and-Brake Acceleration/Flare
  10. vertical_sprint    ( 5%) - Rapid Vertical Climbs, Drops & Waves
  11. split_s_dive       ( 5%) - Vertical-Plane Half-Loops & Pull-Ups
  12. polygon_patrol     ( 5%) - Rounded-Corner Polygon Perimeter Circuits
  13. conical_spiral     ( 5%) - Expanding/Contracting Conical Spirals
  14. rose_curve         ( 5%) - Multi-Petal 3D Rose & Trefoil Curves
  15. crabbed_strafe     ( 5%) - Yaw-Decoupled Lateral & Pirouette Flight
  16. perch_approach     ( 5%) - Inclined Glideslope Final Approaches
  17. recovery_stabilize ( 5%) - High-Tilt/High-Velocity Recovery to Hover

Output:
  corpus/crazyflie_corpus_50000trajs.npz (High-performance concatenated format)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np

from quadrotor.params import (
  M_NOMINAL, G, KF, KM, L, MAX_THRUST
)
from quadrotor.quaternion import R_to_quat


def flatness_solve(
  pos: np.ndarray,      # (H+1, 3)
  vel: np.ndarray,      # (H+1, 3)
  acc: np.ndarray,      # (H+1, 3)
  jerk: np.ndarray,     # (H+1, 3)
  yaw: np.ndarray,      # (H+1,)
  yaw_dot: np.ndarray,  # (H+1,)
  max_tilt_deg: float = 68.0,
  min_z: float = 0.038,
) -> tuple[np.ndarray, np.ndarray, bool]:
  """Computes full 17-state and 4-control vectors using differential flatness."""
  H = len(pos) - 1
  m = M_NOMINAL
  g = G

  T_w = m * (acc + np.array([0.0, 0.0, g]))
  thrust_mag = np.linalg.norm(T_w, axis=-1, keepdims=True)

  max_collective = 4.0 * KF * (MAX_THRUST ** 2)
  min_collective = 4.0 * KF * (150.0 ** 2)

  if np.any(thrust_mag > max_collective * 0.95) or np.any(thrust_mag < min_collective):
    return None, None, False

  z_b = T_w / np.maximum(thrust_mag, 1e-6)

  x_c = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=-1)
  y_b = np.cross(z_b, x_c)
  y_b_norm = np.linalg.norm(y_b, axis=-1, keepdims=True)
  if np.any(y_b_norm < 1e-3):
    return None, None, False
  y_b = y_b / y_b_norm
  x_b = np.cross(y_b, z_b)

  cos_tilt = z_b[:, 2]
  cos_max_tilt = np.cos(np.radians(max_tilt_deg))
  if np.any(cos_tilt < cos_max_tilt):
    return None, None, False

  if np.any(pos[:, 2] < min_z):
    return None, None, False

  R = np.stack([x_b, y_b, z_b], axis=-1)
  quat = np.array([R_to_quat(R[k]) for k in range(H + 1)], dtype=np.float32)

  T_dot = m * jerk
  z_b_dot = (T_dot - np.sum(z_b * T_dot, axis=-1, keepdims=True) * z_b) / np.maximum(thrust_mag, 1e-6)

  om_x = -np.sum(y_b * z_b_dot, axis=-1)
  om_y = np.sum(x_b * z_b_dot, axis=-1)
  om_z = yaw_dot * z_b[:, 2]
  omega_b = np.stack([om_x, om_y, om_z], axis=-1)

  v_b = np.einsum("nij,ni->nj", R, vel)

  # Full state: pos(3), quat(4), v_b(3), omega_b(3), rotor_speeds(4) -> 17 dim
  # Controls: c1, c2, c3, c4 in [0, 1]
  T_total = thrust_mag.squeeze(-1)
  f_per_motor = np.clip(T_total / 4.0, 1e-6, None)
  omega_nominal = np.sqrt(f_per_motor / KF)
  ctrl_ratio = np.clip(omega_nominal / MAX_THRUST, 0.05, 0.98)

  states = np.zeros((H + 1, 17), dtype=np.float32)
  states[:, :3] = pos
  states[:, 3:7] = quat
  states[:, 7:10] = v_b
  states[:, 10:13] = omega_b
  states[:, 13:17] = ctrl_ratio[:, None]

  ctrls = np.zeros((H, 4), dtype=np.float32)
  ctrls[:] = ctrl_ratio[:-1, None]

  return states, ctrls, True


def _smooth_tangent_yaw(
  vel: np.ndarray,
  dt: float,
  min_speed: float = 0.25,
  max_yaw_rate: float = 3.0,
) -> np.ndarray:
  """Compute smooth tangent yaw, holding heading at low speed and rate-limiting turns."""
  speed_xy = np.linalg.norm(vel[:, :2], axis=-1)
  valid = np.where(speed_xy >= min_speed)[0]
  if len(valid) == 0:
    return np.zeros(len(vel), dtype=np.float64)
  yaw = np.empty(len(vel), dtype=np.float64)
  cur = float(np.arctan2(vel[valid[0], 1], vel[valid[0], 0]))
  max_step = max_yaw_rate * dt
  for i in range(len(vel)):
    if speed_xy[i] >= min_speed:
      tgt = float(np.arctan2(vel[i, 1], vel[i, 0]))
      diff = (tgt - cur + np.pi) % (2.0 * np.pi) - np.pi
      cur += float(np.clip(diff, -max_step, max_step))
    yaw[i] = cur
  return yaw


def _solve_from_pos_yaw(
  pos: np.ndarray,
  yaw: np.ndarray,
  dt: float,
  max_tilt_deg: float = 68.0,
  min_z: float = 0.038,
) -> tuple[np.ndarray, np.ndarray, bool]:
  """Differentiate position and yaw trajectories and run differential flatness."""
  vel = np.gradient(pos, dt, axis=0)
  acc = np.gradient(vel, dt, axis=0)
  jerk = np.gradient(acc, dt, axis=0)
  yaw_dot = np.gradient(yaw, dt)
  return flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot, max_tilt_deg=max_tilt_deg, min_z=min_z)


def sample_spline_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    n_pts = np.random.randint(2, 6)
    p0 = np.random.uniform([-1.5, -1.5, 0.5], [1.5, 1.5, 2.0])
    v0 = np.random.uniform([-1.2, -1.2, -0.6], [1.2, 1.2, 0.8])
    a0 = np.random.uniform([-3.0, -3.0, -1.5], [3.0, 3.0, 1.5])

    total_dist = np.random.uniform(1.0, 3.5)
    speed = np.random.uniform(0.8, 1.8)
    total_time = max(1.5, total_dist / speed)
    H = int(total_time / dt)
    if H < 40:
      H = 40

    t = np.linspace(0, total_time, H + 1)
    tau = t / total_time

    # Generate target arrival waypoint
    target_disp = np.random.uniform([-1.5, -1.5, -0.6], [1.5, 1.5, 0.6])
    p1 = np.clip(p0 + target_disp, [-1.8, -1.8, 0.45], [1.8, 1.8, 2.2])
    v1 = np.random.uniform([-0.3, -0.3, -0.2], [0.3, 0.3, 0.2])

    if n_pts == 2:
      # Quintic polynomial with explicit initial (p0, v0, a0) inducing randomized initial attitude tilt & velocity
      c0 = p0
      c1 = v0 * total_time
      c2 = 0.5 * a0 * (total_time ** 2)
      c3 = 10.0 * (p1 - p0) - (6.0 * v0 + 4.0 * v1) * total_time - 1.5 * a0 * (total_time ** 2)
      c4 = -15.0 * (p1 - p0) + (8.0 * v0 + 7.0 * v1) * total_time + 1.5 * a0 * (total_time ** 2)
      c5 = 6.0 * (p1 - p0) - 3.0 * (v0 + v1) * total_time - 0.5 * a0 * (total_time ** 2)

      pos = (c0[None, :] + c1[None, :] * tau[:, None] + c2[None, :] * (tau**2)[:, None] +
             c3[None, :] * (tau**3)[:, None] + c4[None, :] * (tau**4)[:, None] + c5[None, :] * (tau**5)[:, None])
    else:
      # Multi-waypoint cubic spline with initial velocity constraint
      wps = [p0]
      for _ in range(n_pts - 2):
        wps.append(np.random.uniform([-1.6, -1.6, 0.45], [1.6, 1.6, 2.2]))
      wps.append(p1)
      wps = np.array(wps)
      t_knots = np.linspace(0, total_time, len(wps))
      from scipy.interpolate import CubicSpline
      cs = CubicSpline(t_knots, wps, bc_type=((1, v0), (1, v1)))
      pos = cs(t)

    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)

    speed_xy = np.linalg.norm(vel[:, :2], axis=-1)
    if np.mean(speed_xy) > 0.15:
      yaw = _smooth_tangent_yaw(vel, dt)
    else:
      yaw = np.full(H + 1, np.random.uniform(-np.pi, np.pi))
    yaw_dot = np.gradient(yaw, dt)

    s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    if ok:
      return s, c, "point_to_point"

  # Fallback: gentle line
  pos = np.linspace([0, 0, 1], [0.5, 0.5, 1], 100)
  vel = np.gradient(pos, dt, axis=0)
  acc = np.gradient(vel, dt, axis=0)
  jerk = np.gradient(acc, dt, axis=0)
  s, c, _ = flatness_solve(pos, vel, acc, jerk, np.zeros(100), np.zeros(100))
  return s, c, "point_to_point"


def sample_lissajous_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.5, 5.0)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)

    Ax = np.random.uniform(0.5, 1.4)
    Ay = np.random.uniform(0.5, 1.4)
    Az = np.random.uniform(0.15, 0.5)

    wx = np.random.choice([1.0, 1.5, 2.0])
    wy = np.random.choice([1.0, 2.0, 3.0])
    wz = np.random.choice([1.0, 2.0])

    phix = np.random.uniform(0, 2 * np.pi)
    phiy = np.random.uniform(0, 2 * np.pi)
    phiz = np.random.uniform(0, 2 * np.pi)
    center_xy = np.random.uniform([-0.7, -0.7], [0.7, 0.7])
    z0 = np.random.uniform(0.8, 1.7)

    pos_raw = np.stack([
      Ax * np.sin(wx * t + phix),
      Ay * np.sin(wy * t + phiy),
      Az * np.sin(wz * t + phiz),
    ], axis=-1)

    # Random horizontal rotation angle to span all orientations
    psi_rot = np.random.uniform(-np.pi, np.pi)
    cos_p, sin_p = np.cos(psi_rot), np.sin(psi_rot)
    R_z = np.array([[cos_p, -sin_p, 0], [sin_p, cos_p, 0], [0, 0, 1]])

    pos = (pos_raw @ R_z.T) + np.array([center_xy[0], center_xy[1], z0])

    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)

    yaw = _smooth_tangent_yaw(vel, dt)
    yaw_dot = np.gradient(yaw, dt)

    s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    if ok:
      return s, c, "figure8"

  pos = np.stack([np.sin(t), np.cos(t), np.ones_like(t)], axis=-1)
  vel = np.gradient(pos, dt, axis=0)
  acc = np.gradient(vel, dt, axis=0)
  jerk = np.gradient(acc, dt, axis=0)
  s, c, _ = flatness_solve(pos, vel, acc, jerk, np.zeros_like(t), np.zeros_like(t))
  return s, c, "figure8"


def sample_helix_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(3.0, 5.5)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)

    R_helix = np.random.uniform(0.5, 1.2)
    omega = np.random.uniform(1.2, 2.4) * np.random.choice([-1, 1])
    phi0 = np.random.uniform(0, 2 * np.pi)
    center_xy = np.random.uniform([-0.6, -0.6], [0.6, 0.6])
    z_start = np.random.uniform(0.5, 1.2)
    climb_rate = np.random.uniform(0.15, 0.45) * np.random.choice([-1, 1])

    pos = np.stack([
      center_xy[0] + R_helix * np.cos(omega * t + phi0),
      center_xy[1] + R_helix * np.sin(omega * t + phi0),
      z_start + climb_rate * t,
    ], axis=-1)

    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)

    yaw = np.unwrap(omega * t + phi0 + (np.pi / 2 if omega > 0 else -np.pi / 2))
    yaw_dot = np.gradient(yaw, dt)

    s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    if ok:
      return s, c, "helix"

  pos = np.stack([np.cos(t), np.sin(t), 1.0 + 0.1 * t], axis=-1)
  vel = np.gradient(pos, dt, axis=0)
  acc = np.gradient(vel, dt, axis=0)
  jerk = np.gradient(acc, dt, axis=0)
  s, c, _ = flatness_solve(pos, vel, acc, jerk, np.zeros_like(t), np.zeros_like(t))
  return s, c, "helix"


def sample_hover_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.2, 4.2)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)

    p_target = np.random.uniform([-1.4, -1.4, 0.55], [1.4, 1.4, 2.0])

    if np.random.rand() < 0.7:
      # Dynamic hover recovery: starts with wide random position offset, velocity, and tilt
      p_offset = np.random.uniform([-0.5, -0.5, -0.25], [0.5, 0.5, 0.25])
      v_offset = np.random.uniform([-1.0, -1.0, -0.5], [1.0, 1.0, 0.5])
      decay = np.exp(-2.2 * t)[:, None]
      pos = p_target[None, :] + (p_offset[None, :] + v_offset[None, :] * t[:, None]) * decay
    else:
      # Gentle station keeping with low-frequency micro-drift
      drift_amp = np.random.uniform(0.015, 0.045, size=(3,))
      drift_w = np.random.uniform(0.5, 1.5, size=(3,))
      drift_phi = np.random.uniform(0, 2 * np.pi, size=(3,))
      pos = p_target + drift_amp * np.sin(drift_w * t[:, None] + drift_phi)

    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)

    yaw0 = np.random.uniform(-np.pi, np.pi)
    yaw = np.full(H + 1, yaw0)
    yaw_dot = np.zeros(H + 1)

    s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    if ok:
      return s, c, "hover"

  # Fallback: ideal stationary hover
  H = int(3.0 / dt)
  pos = np.full((H + 1, 3), [0.0, 0.0, 1.0], dtype=np.float32)
  vel = np.zeros_like(pos)
  acc = np.zeros_like(pos)
  jerk = np.zeros_like(pos)
  s, c, _ = flatness_solve(pos, vel, acc, jerk, np.zeros(H + 1), np.zeros(H + 1))
  return s, c, "hover"


def sample_takeoff_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.0, 3.5)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)
    tau = t / duration

    z_ground = 0.04
    z_hover = np.random.uniform(0.6, 1.6)
    z = z_ground + (z_hover - z_ground) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)

    x0 = np.random.uniform(-1.2, 1.2)
    y0 = np.random.uniform(-1.2, 1.2)
    pos = np.stack([np.full(H + 1, x0), np.full(H + 1, y0), z], axis=-1)

    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    yaw0 = np.random.uniform(-np.pi, np.pi)
    yaw = np.full(H + 1, yaw0)
    yaw_dot = np.zeros(H + 1)

    s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot, min_z=0.038)
    if ok:
      return s, c, "takeoff"

  # Fallback: gentle takeoff
  duration = 2.5
  H = int(duration / dt)
  t = np.linspace(0, duration, H + 1)
  tau = t / duration
  z = 0.04 + (1.0 - 0.04) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
  pos = np.stack([np.zeros(H + 1), np.zeros(H + 1), z], axis=-1)
  vel = np.gradient(pos, dt, axis=0)
  acc = np.gradient(vel, dt, axis=0)
  jerk = np.gradient(acc, dt, axis=0)
  s, c, _ = flatness_solve(pos, vel, acc, jerk, np.zeros(H + 1), np.zeros(H + 1), min_z=0.038)
  return s, c, "takeoff"


def sample_landing_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.2, 3.8)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)
    tau = t / duration

    z_hover = np.random.uniform(0.6, 1.6)
    z_ground = 0.04
    z = z_hover + (z_ground - z_hover) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)

    x0 = np.random.uniform(-1.2, 1.2)
    y0 = np.random.uniform(-1.2, 1.2)
    # Include slight horizontal drift / glide approach
    vx_init = np.random.uniform(-0.5, 0.5)
    vy_init = np.random.uniform(-0.5, 0.5)
    flare = (1.0 - tau**2)
    x = x0 + vx_init * t * flare
    y = y0 + vy_init * t * flare
    pos = np.stack([x, y, z], axis=-1)

    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    yaw0 = np.random.uniform(-np.pi, np.pi)
    yaw = np.full(H + 1, yaw0)
    yaw_dot = np.zeros(H + 1)

    s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot, min_z=0.038)
    if ok:
      return s, c, "landing"

  # Fallback: gentle landing
  duration = 2.8
  H = int(duration / dt)
  t = np.linspace(0, duration, H + 1)
  tau = t / duration
  z = 1.0 + (0.04 - 1.0) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
  pos = np.stack([np.zeros(H + 1), np.zeros(H + 1), z], axis=-1)
  vel = np.gradient(pos, dt, axis=0)
  acc = np.gradient(vel, dt, axis=0)
  jerk = np.gradient(acc, dt, axis=0)
  s, c, _ = flatness_solve(pos, vel, acc, jerk, np.zeros(H + 1), np.zeros(H + 1), min_z=0.038)
  return s, c, "landing"


def sample_slalom_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.5, 4.5)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)

    v_fwd = np.random.uniform(1.0, 2.0)
    A_lateral = np.random.uniform(0.4, 1.0)
    w_lateral = np.random.uniform(1.8, 3.0)
    phi_lat = np.random.uniform(0, 2 * np.pi)
    z0 = np.random.uniform(0.8, 1.6)

    # Local slalom coordinates with arbitrary phase offset
    x_local = v_fwd * (t - duration / 2.0)
    y_local = A_lateral * np.sin(w_lateral * t + phi_lat)
    z_local = np.full(H + 1, z0)

    # Rotate course heading by arbitrary angle to cover all attitudes and directions
    heading_course = np.random.uniform(-np.pi, np.pi)
    c_hdg, s_hdg = np.cos(heading_course), np.sin(heading_course)
    x_w = c_hdg * x_local - s_hdg * y_local
    y_w = s_hdg * x_local + c_hdg * y_local
    pos = np.stack([x_w, y_w, z_local], axis=-1)

    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    yaw = _smooth_tangent_yaw(vel, dt)
    yaw_dot = np.gradient(yaw, dt)

    s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    if ok:
      return s, c, "slalom"

  # Fallback: standard slalom
  duration = 3.0
  H = int(duration / dt)
  t = np.linspace(0, duration, H + 1)
  pos = np.stack([1.2 * (t - 1.5), 0.6 * np.sin(2.0 * t), np.ones(H + 1)], axis=-1)
  vel = np.gradient(pos, dt, axis=0)
  acc = np.gradient(vel, dt, axis=0)
  jerk = np.gradient(acc, dt, axis=0)
  s, c, _ = flatness_solve(pos, vel, acc, jerk, np.zeros(H + 1), np.zeros(H + 1))
  return s, c, "slalom"


def sample_circular_orbit_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.8, 5.2)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)

    radius = np.random.uniform(0.45, 1.6)
    omega = np.random.uniform(1.0, 2.4) * np.random.choice([-1.0, 1.0])
    phi0 = np.random.uniform(0.0, 2 * np.pi)
    center = np.random.uniform([-0.6, -0.6, 0.7], [0.6, 0.6, 1.8])
    tilt_amp = np.random.uniform(0.0, 0.35)
    tilt_phase = np.random.uniform(0.0, 2 * np.pi)

    theta = omega * t + phi0
    pos = np.stack([
      center[0] + radius * np.cos(theta),
      center[1] + radius * np.sin(theta),
      center[2] + tilt_amp * np.sin(theta + tilt_phase),
    ], axis=-1)

    if np.random.rand() < 0.5:
      # Tangent-aligned (nose-forward)
      yaw = np.unwrap(theta + (np.pi / 2 if omega > 0 else -np.pi / 2))
    else:
      # Point-of-interest inward yaw
      yaw = np.unwrap(theta + np.pi)

    s, c, ok = _solve_from_pos_yaw(pos, yaw, dt)
    if ok:
      return s, c, "circular_orbit"

  t = np.linspace(0, 3.5, int(3.5 / dt) + 1)
  pos = np.stack([0.8 * np.cos(1.2 * t), 0.8 * np.sin(1.2 * t), np.ones_like(t)], axis=-1)
  s, c, _ = _solve_from_pos_yaw(pos, np.unwrap(1.2 * t + np.pi / 2), dt)
  return s, c, "circular_orbit"


def sample_step_dash_brake_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(1.8, 3.4)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)
    tau = t / duration

    p0 = np.random.uniform([-1.5, -1.5, 0.55], [1.5, 1.5, 1.8])
    dist = np.random.uniform(1.2, 3.2)
    az = np.random.uniform(-np.pi, np.pi)
    dz = np.random.uniform(-0.4, 0.5)
    p1 = np.clip(p0 + np.array([dist * np.cos(az), dist * np.sin(az), dz]), [-1.8, -1.8, 0.45], [1.8, 1.8, 2.2])

    # Smoothstep-7 (minimum-snap rest-to-rest) with asymmetric dash-brake warp
    power = np.random.uniform(0.75, 1.35)
    s_warp = np.clip(tau ** power, 0.0, 1.0)
    blend = 35 * s_warp**4 - 84 * s_warp**5 + 70 * s_warp**6 - 20 * s_warp**7
    pos = p0[None, :] + (p1 - p0)[None, :] * blend[:, None]

    yaw = np.full(H + 1, az)
    s, c, ok = _solve_from_pos_yaw(pos, yaw, dt)
    if ok:
      return s, c, "step_dash_brake"

  t = np.linspace(0, 2.5, int(2.5 / dt) + 1)
  tau = t / 2.5
  blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
  pos = np.array([-0.8, 0.0, 1.0])[None, :] + np.array([1.6, 0.0, 0.0])[None, :] * blend[:, None]
  s, c, _ = _solve_from_pos_yaw(pos, np.zeros(len(t)), dt)
  return s, c, "step_dash_brake"


def sample_vertical_sprint_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.2, 4.2)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)
    tau = t / duration

    x0 = np.random.uniform(-1.3, 1.3)
    y0 = np.random.uniform(-1.3, 1.3)
    z_low = np.random.uniform(0.35, 0.8)
    z_high = np.random.uniform(1.3, 2.3)

    mode = np.random.randint(0, 3)
    if mode == 0:
      # Rapid climb
      z = z_low + (z_high - z_low) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    elif mode == 1:
      # Rapid descent
      z = z_high + (z_low - z_high) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    else:
      # Sinusoidal altitude pump
      cycles = np.random.uniform(1.0, 2.5)
      phi = np.random.uniform(0, 2 * np.pi)
      z = 0.5 * (z_low + z_high) + 0.5 * (z_high - z_low) * np.sin(2 * np.pi * cycles * tau + phi)

    pos = np.stack([np.full(H + 1, x0), np.full(H + 1, y0), z], axis=-1)
    yaw = np.full(H + 1, np.random.uniform(-np.pi, np.pi))

    s, c, ok = _solve_from_pos_yaw(pos, yaw, dt)
    if ok:
      return s, c, "vertical_sprint"

  t = np.linspace(0, 3.0, int(3.0 / dt) + 1)
  z = 1.1 + 0.4 * np.sin(2.0 * t)
  pos = np.stack([np.zeros_like(t), np.zeros_like(t), z], axis=-1)
  s, c, _ = _solve_from_pos_yaw(pos, np.zeros_like(t), dt)
  return s, c, "vertical_sprint"


def sample_split_s_dive_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.2, 3.8)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)
    tau = t / duration

    az = np.random.uniform(-np.pi, np.pi)
    fwd_span = np.random.uniform(1.0, 2.4)
    z_amp = np.random.uniform(0.45, 0.95) * np.random.choice([-1.0, 1.0])
    z_mid = np.random.uniform(1.1, 1.7)
    p0 = np.random.uniform([-0.8, -0.8], [0.8, 0.8])

    # Vertical-plane S-curve / half-loop dive-and-pull-up
    s_arc = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    horiz = fwd_span * np.sin(np.pi * s_arc)
    vert = z_mid + z_amp * np.cos(np.pi * s_arc)

    pos = np.stack([
      p0[0] + horiz * np.cos(az),
      p0[1] + horiz * np.sin(az),
      vert,
    ], axis=-1)

    vel = np.gradient(pos, dt, axis=0)
    speed_xy = np.linalg.norm(vel[:, :2], axis=-1)
    yaw = _smooth_tangent_yaw(vel, dt) if np.mean(speed_xy) > 0.1 else np.full(H + 1, az)
    s, c, ok = _solve_from_pos_yaw(pos, yaw, dt)
    if ok:
      return s, c, "split_s_dive"

  t = np.linspace(0, 3.0, int(3.0 / dt) + 1)
  tau = t / 3.0
  s_arc = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
  pos = np.stack([1.2 * np.sin(np.pi * s_arc), np.zeros_like(t), 1.2 + 0.5 * np.cos(np.pi * s_arc)], axis=-1)
  s, c, _ = _solve_from_pos_yaw(pos, np.zeros_like(t), dt)
  return s, c, "split_s_dive"


def sample_polygon_patrol_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  from scipy.interpolate import CubicSpline
  for _ in range(50):
    n_sides = int(np.random.choice([3, 4, 5, 6]))
    duration = np.random.uniform(3.2, 5.5)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)

    radius = np.random.uniform(0.6, 1.5)
    center = np.random.uniform([-0.5, -0.5, 0.7], [0.5, 0.5, 1.7])
    rot0 = np.random.uniform(0.0, 2 * np.pi)
    direction = np.random.choice([-1.0, 1.0])

    angles = rot0 + direction * np.linspace(0, 2 * np.pi, n_sides + 1)
    r_jitter = radius * np.random.uniform(0.85, 1.15, size=n_sides)
    r_jitter = np.append(r_jitter, r_jitter[0])
    z_jitter = center[2] + np.random.uniform(-0.25, 0.25, size=n_sides)
    z_jitter = np.append(z_jitter, z_jitter[0])

    wps = np.stack([
      center[0] + r_jitter * np.cos(angles),
      center[1] + r_jitter * np.sin(angles),
      z_jitter,
    ], axis=-1)
    wps[-1] = wps[0]
    t_knots = np.linspace(0, duration, n_sides + 1)
    cs = CubicSpline(t_knots, wps, bc_type="periodic")
    pos = cs(t)

    vel = np.gradient(pos, dt, axis=0)
    yaw = _smooth_tangent_yaw(vel, dt)
    s, c, ok = _solve_from_pos_yaw(pos, yaw, dt)
    if ok:
      return s, c, "polygon_patrol"

  t = np.linspace(0, 4.0, int(4.0 / dt) + 1)
  wps = np.array([[0.8, 0.8, 1.0], [-0.8, 0.8, 1.0], [-0.8, -0.8, 1.0], [0.8, -0.8, 1.0], [0.8, 0.8, 1.0]])
  cs = CubicSpline(np.linspace(0, 4.0, 5), wps, bc_type="periodic")
  pos = cs(t)
  vel = np.gradient(pos, dt, axis=0)
  s, c, _ = _solve_from_pos_yaw(pos, _smooth_tangent_yaw(vel, dt), dt)
  return s, c, "polygon_patrol"


def sample_conical_spiral_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(3.0, 5.2)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)
    tau = t / duration

    r0 = np.random.uniform(0.25, 0.65)
    r1 = np.random.uniform(0.95, 1.55)
    if np.random.rand() < 0.5:
      r0, r1 = r1, r0

    z0 = np.random.uniform(0.45, 1.0)
    z1 = np.random.uniform(1.2, 2.0)
    if np.random.rand() < 0.5:
      z0, z1 = z1, z0

    omega = np.random.uniform(1.8, 3.5) * np.random.choice([-1.0, 1.0])
    phi0 = np.random.uniform(0, 2 * np.pi)
    center_xy = np.random.uniform([-0.5, -0.5], [0.5, 0.5])

    r = r0 + (r1 - r0) * tau
    z = z0 + (z1 - z0) * (3 * tau**2 - 2 * tau**3)
    theta = omega * t + phi0

    pos = np.stack([
      center_xy[0] + r * np.cos(theta),
      center_xy[1] + r * np.sin(theta),
      z,
    ], axis=-1)

    vel = np.gradient(pos, dt, axis=0)
    yaw = _smooth_tangent_yaw(vel, dt)
    s, c, ok = _solve_from_pos_yaw(pos, yaw, dt)
    if ok:
      return s, c, "conical_spiral"

  t = np.linspace(0, 3.5, int(3.5 / dt) + 1)
  tau = t / 3.5
  r = 0.4 + 0.7 * tau
  pos = np.stack([r * np.cos(2.0 * t), r * np.sin(2.0 * t), 0.7 + 0.8 * tau], axis=-1)
  vel = np.gradient(pos, dt, axis=0)
  s, c, _ = _solve_from_pos_yaw(pos, _smooth_tangent_yaw(vel, dt), dt)
  return s, c, "conical_spiral"


def sample_rose_curve_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(3.2, 5.5)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)

    k = int(np.random.choice([2, 3, 4]))
    R0 = np.random.uniform(0.55, 1.0)
    R1 = np.random.uniform(0.25, 0.55)
    Az = np.random.uniform(0.15, 0.45)
    omega = np.random.uniform(0.9, 1.6) * np.random.choice([-1.0, 1.0])
    phi0 = np.random.uniform(0, 2 * np.pi)
    center = np.random.uniform([-0.5, -0.5, 0.8], [0.5, 0.5, 1.6])

    theta = omega * t + phi0
    r = R0 + R1 * np.cos(k * theta)
    pos = np.stack([
      center[0] + r * np.cos(theta),
      center[1] + r * np.sin(theta),
      center[2] + Az * np.sin(k * theta),
    ], axis=-1)

    vel = np.gradient(pos, dt, axis=0)
    yaw = _smooth_tangent_yaw(vel, dt)
    s, c, ok = _solve_from_pos_yaw(pos, yaw, dt)
    if ok:
      return s, c, "rose_curve"

  t = np.linspace(0, 4.0, int(4.0 / dt) + 1)
  r = 0.7 + 0.3 * np.cos(3.0 * t)
  pos = np.stack([r * np.cos(t), r * np.sin(t), 1.1 + 0.2 * np.sin(3.0 * t)], axis=-1)
  vel = np.gradient(pos, dt, axis=0)
  s, c, _ = _solve_from_pos_yaw(pos, _smooth_tangent_yaw(vel, dt), dt)
  return s, c, "rose_curve"


def sample_crabbed_strafe_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.5, 4.5)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)
    tau = t / duration

    p0 = np.random.uniform([-1.4, -1.4, 0.6], [1.4, 1.4, 1.8])
    disp = np.random.uniform([-1.8, -1.8, -0.4], [1.8, 1.8, 0.5])
    p1 = np.clip(p0 + disp, [-1.8, -1.8, 0.45], [1.8, 1.8, 2.1])
    lat_amp = np.random.uniform(0.2, 0.6)
    lat_w = np.random.uniform(1.5, 3.0)

    blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    pos = p0[None, :] + (p1 - p0)[None, :] * blend[:, None]
    pos[:, 1] += lat_amp * np.sin(lat_w * t) * np.sin(np.pi * tau)

    yaw0 = np.random.uniform(-np.pi, np.pi)
    if np.random.rand() < 0.5:
      # Fixed decoupled heading (flying sideways / backward relative to travel)
      yaw = np.full(H + 1, yaw0)
    else:
      # Continuous pirouette while translating
      yaw_rate = np.random.uniform(1.0, 2.5) * np.random.choice([-1.0, 1.0])
      yaw = yaw0 + yaw_rate * t

    s, c, ok = _solve_from_pos_yaw(pos, yaw, dt)
    if ok:
      return s, c, "crabbed_strafe"

  t = np.linspace(0, 3.0, int(3.0 / dt) + 1)
  tau = t / 3.0
  pos = np.stack([np.zeros_like(t), 1.2 * (10 * tau**3 - 15 * tau**4 + 6 * tau**5), np.ones_like(t)], axis=-1)
  s, c, _ = _solve_from_pos_yaw(pos, np.zeros_like(t), dt)
  return s, c, "crabbed_strafe"


def sample_perch_approach_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.2, 3.8)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)
    tau = t / duration

    pad = np.random.uniform([-1.0, -1.0, 0.045], [1.0, 1.0, 0.07])
    glide_dist = np.random.uniform(1.0, 2.4)
    glide_angle = np.radians(np.random.uniform(15.0, 45.0))
    az = np.random.uniform(-np.pi, np.pi)

    p_start = np.array([
      pad[0] - glide_dist * np.cos(glide_angle) * np.cos(az),
      pad[1] - glide_dist * np.cos(glide_angle) * np.sin(az),
      pad[2] + glide_dist * np.sin(glide_angle),
    ])

    # Smooth glideslope descent with late flare to zero velocity at perch
    s_glide = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    pos = p_start[None, :] + (pad - p_start)[None, :] * s_glide[:, None]
    yaw = np.full(H + 1, az)

    s, c, ok = _solve_from_pos_yaw(pos, yaw, dt, min_z=0.038)
    if ok:
      return s, c, "perch_approach"

  t = np.linspace(0, 2.8, int(2.8 / dt) + 1)
  tau = t / 2.8
  s_glide = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
  p0 = np.array([-1.2, 0.0, 0.9])
  p1 = np.array([0.0, 0.0, 0.05])
  pos = p0[None, :] + (p1 - p0)[None, :] * s_glide[:, None]
  s, c, _ = _solve_from_pos_yaw(pos, np.zeros_like(t), dt, min_z=0.038)
  return s, c, "perch_approach"


def sample_recovery_stabilize_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
  for _ in range(50):
    duration = np.random.uniform(2.0, 3.5)
    H = int(duration / dt)
    t = np.linspace(0, duration, H + 1)
    tau = t / duration

    p_goal = np.random.uniform([-1.2, -1.2, 0.7], [1.2, 1.2, 1.8])
    p0 = np.clip(p_goal + np.random.uniform([-0.8, -0.8, -0.35], [0.8, 0.8, 0.4]), [-1.8, -1.8, 0.4], [1.8, 1.8, 2.2])
    v0 = np.random.uniform([-1.8, -1.8, -0.9], [1.8, 1.8, 0.9])
    a0 = np.random.uniform([-5.5, -5.5, -2.5], [5.5, 5.5, 3.5])

    # Quintic boundary-value polynomial from perturbed (p0, v0, a0) to stationary hover (p_goal, 0, 0)
    T = duration
    c0 = p0
    c1 = v0 * T
    c2 = 0.5 * a0 * (T ** 2)
    c3 = 10.0 * (p_goal - p0) - 6.0 * v0 * T - 1.5 * a0 * (T ** 2)
    c4 = -15.0 * (p_goal - p0) + 8.0 * v0 * T + 1.5 * a0 * (T ** 2)
    c5 = 6.0 * (p_goal - p0) - 3.0 * v0 * T - 0.5 * a0 * (T ** 2)

    pos = (
      c0[None, :]
      + c1[None, :] * tau[:, None]
      + c2[None, :] * (tau ** 2)[:, None]
      + c3[None, :] * (tau ** 3)[:, None]
      + c4[None, :] * (tau ** 4)[:, None]
      + c5[None, :] * (tau ** 5)[:, None]
    )

    yaw0 = np.random.uniform(-np.pi, np.pi)
    yaw_rate0 = np.random.uniform(-1.5, 1.5)
    yaw = yaw0 + yaw_rate0 * t * np.exp(-2.5 * t)

    s, c, ok = _solve_from_pos_yaw(pos, yaw, dt)
    if ok:
      return s, c, "recovery_stabilize"

  t = np.linspace(0, 2.5, int(2.5 / dt) + 1)
  pos = np.stack([0.3 * np.exp(-2.0 * t), np.zeros_like(t), np.ones_like(t)], axis=-1)
  s, c, _ = _solve_from_pos_yaw(pos, np.zeros_like(t), dt)
  return s, c, "recovery_stabilize"


TRAJECTORY_PERCENTAGES: dict[str, float] = {
  "point_to_point": 0.12,
  "figure8": 0.08,
  "helix": 0.08,
  "hover": 0.07,
  "takeoff": 0.04,
  "landing": 0.04,
  "slalom": 0.07,
  "circular_orbit": 0.05,
  "step_dash_brake": 0.05,
  "vertical_sprint": 0.05,
  "split_s_dive": 0.05,
  "polygon_patrol": 0.05,
  "conical_spiral": 0.05,
  "rose_curve": 0.05,
  "crabbed_strafe": 0.05,
  "perch_approach": 0.05,
  "recovery_stabilize": 0.05,
}


def compute_counts_from_percentages(total_trajs: int) -> dict[str, int]:
  assert abs(sum(TRAJECTORY_PERCENTAGES.values()) - 1.0) < 1e-6
  counts = {k: int(round(total_trajs * pct)) for k, pct in TRAJECTORY_PERCENTAGES.items()}
  diff = total_trajs - sum(counts.values())
  if diff != 0:
    first_key = next(iter(counts))
    counts[first_key] += diff
  return counts


def generate_ultra_corpus(total_trajs: int = 50000, output_path: str = "corpus/crazyflie_corpus_50000trajs.npz"):
  print(f"[INFO] Initializing generation of {total_trajs} randomized, plausible trajectories across {len(TRAJECTORY_PERCENTAGES)} families...", flush=True)
  t0 = time.time()

  counts = compute_counts_from_percentages(total_trajs)

  generators = {
    "point_to_point": sample_spline_trajectory,
    "figure8": sample_lissajous_trajectory,
    "helix": sample_helix_trajectory,
    "hover": sample_hover_trajectory,
    "takeoff": sample_takeoff_trajectory,
    "landing": sample_landing_trajectory,
    "slalom": sample_slalom_trajectory,
    "circular_orbit": sample_circular_orbit_trajectory,
    "step_dash_brake": sample_step_dash_brake_trajectory,
    "vertical_sprint": sample_vertical_sprint_trajectory,
    "split_s_dive": sample_split_s_dive_trajectory,
    "polygon_patrol": sample_polygon_patrol_trajectory,
    "conical_spiral": sample_conical_spiral_trajectory,
    "rose_curve": sample_rose_curve_trajectory,
    "crabbed_strafe": sample_crabbed_strafe_trajectory,
    "perch_approach": sample_perch_approach_trajectory,
    "recovery_stabilize": sample_recovery_stabilize_trajectory,
  }

  all_states_list = []
  all_ctrls_list = []
  state_offsets = [0]
  ctrl_offsets = [0]
  task_labels = []

  report_interval = 5000
  current_count = 0

  for task, target_count in counts.items():
    gen_fn = generators[task]
    print(f"[START] Synthesizing {target_count} {task} trajectories...", flush=True)
    for _ in range(target_count):
      s, c, ttype = gen_fn()
      assert s is not None and c is not None, f"Generator returned None for {ttype}"
      all_states_list.append(s.astype(np.float32))
      all_ctrls_list.append(c.astype(np.float32))

      state_offsets.append(state_offsets[-1] + len(s))
      ctrl_offsets.append(ctrl_offsets[-1] + len(c))
      task_labels.append(ttype)

      current_count += 1
      if current_count % report_interval == 0:
        el = time.time() - t0
        print(f"[PROGRESS] Generated {current_count}/{total_trajs} trajectories ({el:.1f}s, {el/current_count*1000:.1f} ms/traj)...", flush=True)

  print("[INFO] Concatenating arrays into contiguous buffer...", flush=True)
  states_concat = np.concatenate(all_states_list, axis=0)
  ctrls_concat = np.concatenate(all_ctrls_list, axis=0)
  state_offsets = np.array(state_offsets, dtype=np.int32)
  ctrl_offsets = np.array(ctrl_offsets, dtype=np.int32)
  tasks_arr = np.array(task_labels)

  print(f"[INFO] Total States: {len(states_concat):,}, Total Transitions: {len(ctrls_concat):,}", flush=True)
  print(f"[INFO] Compressing and saving into NPZ format ({output_path})...", flush=True)
  t_save = time.time()

  out_file = Path(output_path)
  out_file.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out_file,
    states=states_concat,
    ctrls=ctrls_concat,
    state_offsets=state_offsets,
    ctrl_offsets=ctrl_offsets,
    tasks=tasks_arr,
    num_trajs=np.array([total_trajs], dtype=np.int32),
  )

  elapsed = time.time() - t0
  save_time = time.time() - t_save
  file_size_mb = out_file.stat().st_size / (1024 * 1024)

  print("=" * 70, flush=True)
  print(f"SUCCESS: Generated {total_trajs} trajectories ({len(ctrls_concat):,} transition steps) in {elapsed:.1f}s (saved in {save_time:.1f}s)!", flush=True)
  print(f"File: {out_file} ({file_size_mb:.1f} MB)", flush=True)
  pct_breakdown = {k: f"{v} ({100.0 * v / total_trajs:.1f}%)" for k, v in counts.items()}
  print(f"Task Breakdown: {pct_breakdown}", flush=True)
  print("=" * 70, flush=True)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Generate Ultra-Corpus with 17 trajectory families.")
  parser.add_argument("--total-trajs", type=int, default=50000, help="Total trajectories to generate.")
  parser.add_argument("--output", type=str, default="corpus/crazyflie_corpus_50000trajs.npz", help="Output NPZ path.")
  args = parser.parse_args()
  generate_ultra_corpus(total_trajs=args.total_trajs, output_path=args.output)
