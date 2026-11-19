"""Mega-Corpus Generator for SuperMimic Bitcraze Crazyflie 2.1.

Generates 9,000 diverse, dynamically plausible, physically verified reference
trajectories (100x the initial 90-trajectory corpus, ~1.2 Million transitions).

Categories (9,000 Total):
  1. Multi-Waypoint 3D Splines (2,500 trajs)
  2. Parametric 3D Lissajous & Lemniscates (2,000 trajs)
  3. Parametric 3D Helices & Spirals (1,500 trajs)
  4. Precision 3D Hovers & Station Keeping (1,200 trajs)
  5. Vertical Takeoffs (600 trajs)
  6. Controlled Touchdown Landings (600 trajs)
  7. High-Speed Slaloms & Agile Dashes (600 trajs)

Output:
  corpus/crazyflie_corpus_9000trajs.npz (High-performance concatenated format)
"""

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

    # Total thrust vector in world: T_w = m * (a + g * z_w)
    T_w = m * (acc + np.array([0.0, 0.0, g]))
    thrust_mag = np.linalg.norm(T_w, axis=-1, keepdims=True)

    max_collective = 4.0 * KF * (MAX_THRUST ** 2)
    min_collective = 4.0 * KF * (150.0 ** 2)

    # Physical checks: collective thrust limits
    if np.any(thrust_mag > max_collective * 0.95) or np.any(thrust_mag < min_collective):
        return None, None, False

    z_b = T_w / np.maximum(thrust_mag, 1e-6)

    # Heading vector in horizontal plane
    x_c = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=-1)

    y_b_raw = np.cross(z_b, x_c)
    y_b_norm = np.linalg.norm(y_b_raw, axis=-1, keepdims=True)
    if np.any(y_b_norm < 1e-4):
        return None, None, False
    y_b = y_b_raw / y_b_norm

    x_b = np.cross(y_b, z_b)
    R = np.stack([x_b, y_b, z_b], axis=-1)

    # Tilt limit check
    tilt_cos = z_b[:, 2]
    if np.any(tilt_cos < np.cos(np.radians(max_tilt_deg))):
        return None, None, False

    # Altitude check (must not crash through ground)
    if np.any(pos[:, 2] < min_z):
        return None, None, False

    quats = np.array([R_to_quat(R[k]) for k in range(H + 1)], dtype=np.float32)
    v_b = np.einsum('nij,ni->nj', R, vel).astype(np.float32)

    # Angular velocity
    T_dot = m * jerk
    z_b_dot = (T_dot - np.sum(z_b * T_dot, axis=-1, keepdims=True) * z_b) / np.maximum(thrust_mag, 1e-6)
    p_body = -np.sum(y_b * z_b_dot, axis=-1)
    q_body = np.sum(x_b * z_b_dot, axis=-1)
    r_body = yaw_dot * z_b[:, 2]
    w_b = np.stack([p_body, q_body, r_body], axis=-1).astype(np.float32)

    thrust_per_rotor = thrust_mag[:, 0] / 4.0
    rpm_per_rotor = np.sqrt(np.maximum(thrust_per_rotor / KF, 0.0))
    rotor_ratio = np.clip(rpm_per_rotor / MAX_THRUST, 0.0, 1.0).astype(np.float32)

    controls = np.repeat(rotor_ratio[:H, None], 4, axis=-1)

    states = np.zeros((H + 1, 17), dtype=np.float32)
    states[:, 0:3] = pos.astype(np.float32)
    states[:, 3:7] = quats
    states[:, 7:10] = v_b
    states[:, 10:13] = w_b
    states[:, 13:17] = np.repeat(rotor_ratio[:, None], 4, axis=-1)

    return states, controls, True


# ------------------------------------------------------------------------------
# 1. Multi-Waypoint 3D Splines
# ------------------------------------------------------------------------------
def sample_spline_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
    from scipy.interpolate import CubicSpline
    for _ in range(40):
        n_pts = np.random.randint(2, 6)
        p0 = np.random.uniform([-1.5, -1.5, 0.5], [1.5, 1.5, 2.0])
        v0 = np.random.uniform([-1.2, -1.2, -0.6], [1.2, 1.2, 0.8])
        a0 = np.random.uniform([-3.0, -3.0, -1.5], [3.0, 3.0, 1.5])

        total_dist = np.random.uniform(1.0, 3.5)
        speed = np.random.uniform(0.8, 1.8)
        total_time = max(1.5, total_dist / speed)
        H = int(round(total_time / dt))
        H = np.clip(H, 50, 220)

        t = np.linspace(0.0, total_time, H + 1)
        tau = t / total_time

        target_disp = np.random.uniform([-1.5, -1.5, -0.6], [1.5, 1.5, 0.6])
        p1 = np.clip(p0 + target_disp, [-1.8, -1.8, 0.45], [1.8, 1.8, 2.2])
        v1 = np.random.uniform([-0.3, -0.3, -0.2], [0.3, 0.3, 0.2])

        if n_pts == 2:
            c0 = p0
            c1 = v0 * total_time
            c2 = 0.5 * a0 * (total_time ** 2)
            c3 = 10.0 * (p1 - p0) - (6.0 * v0 + 4.0 * v1) * total_time - 1.5 * a0 * (total_time ** 2)
            c4 = -15.0 * (p1 - p0) + (8.0 * v0 + 7.0 * v1) * total_time + 1.5 * a0 * (total_time ** 2)
            c5 = 6.0 * (p1 - p0) - 3.0 * (v0 + v1) * total_time - 0.5 * a0 * (total_time ** 2)

            pos = (c0[None, :] + c1[None, :] * tau[:, None] + c2[None, :] * (tau**2)[:, None] +
                   c3[None, :] * (tau**3)[:, None] + c4[None, :] * (tau**4)[:, None] + c5[None, :] * (tau**5)[:, None])
        else:
            wps = [p0]
            for _ in range(n_pts - 2):
                wps.append(np.random.uniform([-1.6, -1.6, 0.45], [1.6, 1.6, 2.2]))
            wps.append(p1)
            wps = np.array(wps)
            t_knots = np.linspace(0.0, total_time, len(wps))
            cs = CubicSpline(t_knots, wps, bc_type=((1, v0), (1, v1)))
            pos = cs(t)

        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)

        speed_xy = np.linalg.norm(vel[:, :2], axis=-1)
        if np.mean(speed_xy) > 0.15:
            yaw = np.unwrap(np.arctan2(vel[:, 1] + 1e-4, vel[:, 0] + 1e-4))
        else:
            yaw = np.full(H + 1, np.random.uniform(-np.pi, np.pi))
        yaw_dot = np.gradient(yaw, dt)

        s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
        if ok and s is not None:
            return s, c, "point_to_point"

    # Fallback point-to-point
    p0 = np.random.uniform([-1.0, -1.0, 0.5], [1.0, 1.0, 1.5])
    p1 = np.random.uniform([-1.0, -1.0, 0.5], [1.0, 1.0, 1.5])
    H = 100
    t = np.linspace(0, 1, H + 1)
    s_poly = 10 * t**3 - 15 * t**4 + 6 * t**5
    pos = p0 + (p1 - p0) * s_poly[:, None]
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    yaw = np.zeros(H + 1)
    yaw_dot = np.zeros(H + 1)
    s, c, _ = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    return s, c, "point_to_point"


# ------------------------------------------------------------------------------
# 2. Parametric 3D Lissajous & Lemniscates
# ------------------------------------------------------------------------------
def sample_lissajous_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
    for _ in range(40):
        H = np.random.randint(120, 240)
        total_time = H * dt
        t = np.linspace(0, total_time, H + 1)

        freq_pairs = [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2), (1, 3)]
        fx, fy = freq_pairs[np.random.randint(0, len(freq_pairs))]
        base_w = np.random.uniform(1.0, 2.0)
        wx = fx * base_w
        wy = fy * base_w
        wz = base_w * np.random.uniform(0.5, 1.5)

        Ax = np.random.uniform(0.4, 1.3)
        Ay = np.random.uniform(0.4, 1.3)
        Az = np.random.uniform(0.1, 0.45)
        phi_x = np.random.uniform(0.0, 2.0 * np.pi)
        phi_y = np.random.uniform(0.0, 2.0 * np.pi)
        phi_z = np.random.uniform(0.0, 2.0 * np.pi)
        center_xy = np.random.uniform([-0.7, -0.7], [0.7, 0.7])
        z0 = np.random.uniform(0.8, 1.6)

        x_raw = Ax * np.sin(wx * t + phi_x)
        y_raw = Ay * np.sin(wy * t + phi_y)
        z_raw = Az * np.sin(wz * t + phi_z)
        pos_raw = np.stack([x_raw, y_raw, z_raw], axis=-1)

        # Rotate course heading horizontally
        psi_rot = np.random.uniform(-np.pi, np.pi)
        cos_p, sin_p = np.cos(psi_rot), np.sin(psi_rot)
        R_z = np.array([[cos_p, -sin_p, 0], [sin_p, cos_p, 0], [0, 0, 1]])
        pos = (pos_raw @ R_z.T) + np.array([center_xy[0], center_xy[1], z0])

        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)

        yaw = np.arctan2(vel[:, 1] + 1e-4, vel[:, 0] + 1e-4)
        yaw = np.unwrap(yaw)
        yaw_dot = np.gradient(yaw, dt)

        s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
        if ok and s is not None:
            return s, c, "figure8"

    # Fallback figure8
    t = np.linspace(0, 4.0, 201)
    pos = np.stack([0.7 * np.sin(1.57 * t), 0.4 * np.sin(3.14 * t), np.full_like(t, 1.2)], axis=-1)
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    yaw = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    yaw_dot = np.gradient(yaw, dt)
    s, c, _ = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    return s, c, "figure8"


# ------------------------------------------------------------------------------
# 3. Parametric 3D Helices & Spirals
# ------------------------------------------------------------------------------
def sample_helix_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
    for _ in range(40):
        H = np.random.randint(140, 240)
        total_time = H * dt
        t = np.linspace(0, total_time, H + 1)

        R0 = np.random.uniform(0.4, 1.1)
        R1 = R0 + np.random.uniform(-0.25, 0.35)
        R_t = np.linspace(R0, R1, H + 1)
        R_dot = (R1 - R0) / total_time

        revolutions = np.random.uniform(1.2, 3.2)
        sign = np.random.choice([-1.0, 1.0])
        w = sign * 2.0 * np.pi * revolutions / total_time
        phi0 = np.random.uniform(0.0, 2.0 * np.pi)
        center_xy = np.random.uniform([-0.6, -0.6], [0.6, 0.6])

        z0 = np.random.uniform(0.6, 1.2)
        z1 = z0 + np.random.uniform(-0.4, 0.8)
        z_t = np.linspace(z0, z1, H + 1)
        vz = (z1 - z0) / total_time

        theta = w * t + phi0
        x = center_xy[0] + R_t * np.cos(theta)
        y = center_xy[1] + R_t * np.sin(theta)
        z = z_t

        vx = R_dot * np.cos(theta) - R_t * w * np.sin(theta)
        vy = R_dot * np.sin(theta) + R_t * w * np.cos(theta)
        vz_arr = np.full_like(t, vz)

        pos = np.stack([x, y, z], axis=-1)
        vel = np.stack([vx, vy, vz_arr], axis=-1)
        acc = np.gradient(vel, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)

        yaw = np.unwrap(np.arctan2(vel[:, 1] + 1e-4, vel[:, 0] + 1e-4))
        yaw_dot = np.gradient(yaw, dt)

        s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
        if ok and s is not None:
            return s, c, "helix"

    # Fallback helix
    t = np.linspace(0, 4.0, 201)
    pos = np.stack([0.6 * np.cos(3.14 * t), 0.6 * np.sin(3.14 * t), 0.8 + 0.2 * t], axis=-1)
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    yaw = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    yaw_dot = np.gradient(yaw, dt)
    s, c, _ = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    return s, c, "helix"


# ------------------------------------------------------------------------------
# 4. Precision 3D Hovers
# ------------------------------------------------------------------------------
def sample_hover_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
    H = np.random.randint(90, 180)
    pos_target = np.random.uniform([-1.5, -1.5, 0.55], [1.5, 1.5, 2.0])
    yaw_target = np.random.uniform(-np.pi, np.pi)

    t = np.linspace(0, H * dt, H + 1)
    if np.random.rand() < 0.7:
        # Dynamic recovery from offset attitude, velocity and position
        p_offset = np.random.uniform([-0.5, -0.5, -0.25], [0.5, 0.5, 0.25])
        v_offset = np.random.uniform([-1.0, -1.0, -0.5], [1.0, 1.0, 0.5])
        decay = np.exp(-2.2 * t)[:, None]
        pos = pos_target[None, :] + (p_offset[None, :] + v_offset[None, :] * t[:, None]) * decay
    else:
        # Precision micro-drift
        drift_amp = np.random.uniform(0.015, 0.045, size=(3,))
        drift_w = np.random.uniform(0.5, 1.5, size=(3,))
        drift_phi = np.random.uniform(0, 2 * np.pi, size=(3,))
        pos = pos_target + drift_amp * np.sin(drift_w * t[:, None] + drift_phi)

    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    yaw = np.full(H + 1, yaw_target)
    yaw_dot = np.zeros(H + 1)

    s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    if ok and s is not None:
        return s, c, "hover"

    pos = np.full((H + 1, 3), pos_target, dtype=np.float32)
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    jerk = np.zeros_like(pos)
    s, c, _ = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    return s, c, "hover"


# ------------------------------------------------------------------------------
# 5. Takeoffs & Landings
# ------------------------------------------------------------------------------
def sample_takeoff_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
    while True:
        z_target = np.random.uniform(0.6, 2.0)
        p_xy = np.random.uniform([-1.5, -1.5], [1.5, 1.5])
        T = np.random.uniform(1.8, 2.6)
        H = int(round(T / dt))
        t = np.linspace(0.0, 1.0, H + 1)
        s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        s_dot = (30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4) / T
        s_ddot = (60.0 * t - 180.0 * t**2 + 120.0 * t**3) / (T**2)
        s_jerk = (60.0 - 360.0 * t + 360.0 * t**2) / (T**3)

        pos = np.stack([np.full(H + 1, p_xy[0]), np.full(H + 1, p_xy[1]), 0.04 + (z_target - 0.04) * s], axis=-1)
        vel = np.stack([np.zeros(H + 1), np.zeros(H + 1), (z_target - 0.04) * s_dot], axis=-1)
        acc = np.stack([np.zeros(H + 1), np.zeros(H + 1), (z_target - 0.04) * s_ddot], axis=-1)
        jerk = np.stack([np.zeros(H + 1), np.zeros(H + 1), (z_target - 0.04) * s_jerk], axis=-1)
        yaw = np.full(H + 1, np.random.uniform(-np.pi, np.pi))
        yaw_dot = np.zeros(H + 1)
        s17, c17, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
        if ok and s17 is not None:
            return s17, c17, "takeoff"


def sample_landing_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
    while True:
        z_start = np.random.uniform(0.8, 2.0)
        p_xy = np.random.uniform([-1.5, -1.5], [1.5, 1.5])
        T = np.random.uniform(2.0, 3.0)
        H = int(round(T / dt))
        t = np.linspace(0.0, 1.0, H + 1)
        s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        s_dot = (30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4) / T
        s_ddot = (60.0 * t - 180.0 * t**2 + 120.0 * t**3) / (T**2)
        s_jerk = (60.0 - 360.0 * t + 360.0 * t**2) / (T**3)

        vx_init = np.random.uniform(-0.5, 0.5)
        vy_init = np.random.uniform(-0.5, 0.5)
        flare = 1.0 - s
        x = p_xy[0] + vx_init * t * T * flare
        y = p_xy[1] + vy_init * t * T * flare
        pos = np.stack([x, y, z_start + (0.04 - z_start) * s], axis=-1)
        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)
        yaw = np.full(H + 1, np.random.uniform(-np.pi, np.pi))
        yaw_dot = np.zeros(H + 1)
        s17, c17, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
        if ok and s17 is not None:
            return s17, c17, "landing"


# ------------------------------------------------------------------------------
# 6. High-Speed Slaloms & Agile Dashes
# ------------------------------------------------------------------------------
def sample_slalom_trajectory(dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, str]:
    for _ in range(40):
        H = np.random.randint(120, 200)
        T = H * dt
        t = np.linspace(0.0, 1.0, H + 1)
        s_poly = 10 * t**3 - 15 * t**4 + 6 * t**5

        v_fwd = np.random.uniform(1.0, 2.0)
        A_lat = np.random.uniform(0.4, 1.0)
        w_lat = np.random.uniform(1.8, 3.0)
        phi_lat = np.random.uniform(0, 2 * np.pi)
        z0 = np.random.uniform(0.8, 1.5)

        x_local = v_fwd * (t * T - T / 2.0)
        y_local = A_lat * np.sin(w_lat * t * T + phi_lat)
        z_local = np.full(H + 1, z0)

        course_hdg = np.random.uniform(-np.pi, np.pi)
        c_h, s_h = np.cos(course_hdg), np.sin(course_hdg)
        x_w = c_h * x_local - s_h * y_local
        y_w = s_h * x_local + c_h * y_local
        pos = np.stack([x_w, y_w, z_local], axis=-1)

        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)

        yaw = np.unwrap(np.arctan2(vel[:, 1] + 1e-4, vel[:, 0] + 1e-4))
        yaw_dot = np.gradient(yaw, dt)

        s, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot, max_tilt_deg=65.0)
        if ok and s is not None:
            return s, c, "slalom"

    # Fallback slalom
    t = np.linspace(0, 1.0, 141)
    s_poly = 10 * t**3 - 15 * t**4 + 6 * t**5
    pos = np.stack([-1.2 + 2.4 * s_poly, 0.6 * np.sin(2 * np.pi * s_poly), np.full_like(t, 1.2)], axis=-1)
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    yaw = np.unwrap(np.arctan2(vel[:, 1] + 1e-4, vel[:, 0] + 1e-4))
    yaw_dot = np.gradient(yaw, dt)
    s, c, _ = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
    return s, c, "slalom"


# ------------------------------------------------------------------------------
# Main Mega-Corpus Assembler
# ------------------------------------------------------------------------------
def generate_mega_corpus(total_trajs: int = 9000, output_path: str = "corpus/crazyflie_corpus_9000trajs.npz"):
    print(f"[INFO] Initializing generation of {total_trajs} randomized, plausible trajectories...")
    t0 = time.time()

    counts = {
        "point_to_point": 2500,
        "figure8": 2000,
        "helix": 1500,
        "hover": 1200,
        "takeoff": 600,
        "landing": 600,
        "slalom": 600,
    }

    generators = {
        "point_to_point": sample_spline_trajectory,
        "figure8": sample_lissajous_trajectory,
        "helix": sample_helix_trajectory,
        "hover": sample_hover_trajectory,
        "takeoff": sample_takeoff_trajectory,
        "landing": sample_landing_trajectory,
        "slalom": sample_slalom_trajectory,
    }

    all_states_list = []
    all_ctrls_list = []
    state_offsets = [0]
    ctrl_offsets = [0]
    task_labels = []

    report_interval = 1000
    current_count = 0

    for task, target_count in counts.items():
        gen_fn = generators[task]
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

    print("=" * 60, flush=True)
    print(f"SUCCESS: Generated {total_trajs} trajectories ({len(ctrls_concat):,} transition steps) in {elapsed:.1f}s (saved in {save_time:.1f}s)!", flush=True)
    print(f"File: {out_file} ({file_size_mb:.1f} MB)", flush=True)
    print(f"Task Breakdown: {counts}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    generate_mega_corpus()
