"""Test differential flatness trajectory generator for Crazyflie 2.1."""

import time
import numpy as np
from quadrotor.params import (
    M_NOMINAL, G, KF, KM, L, MAX_THRUST, I_NOMINAL
)
from quadrotor.quaternion import R_to_quat, quat_to_R

def generate_differential_flatness_trajectory(
    pos_traj: np.ndarray,      # (H+1, 3) position
    vel_traj: np.ndarray,      # (H+1, 3) velocity
    acc_traj: np.ndarray,      # (H+1, 3) acceleration
    jerk_traj: np.ndarray,     # (H+1, 3) jerk
    yaw_traj: np.ndarray,      # (H+1,) yaw angle
    yaw_dot_traj: np.ndarray,  # (H+1,) yaw rate
    dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Computes exact 17-dim quadcopter state and 4-dim control using differential flatness.
    
    Returns:
      states: (H+1, 17) [pos(3), quat(4, wxyz), lin_vel_b(3), ang_vel_b(3), rotor_ratio(4)]
      controls: (H, 4) rotor ratio in [0, 1]
      is_valid: bool (within motor and tilt limits)
    """
    H = len(pos_traj) - 1
    m = M_NOMINAL
    g = G
    
    # 1. Total thrust vector in world: T_w = m * (a + g * z_w)
    T_w = m * (acc_traj + np.array([0.0, 0.0, g]))  # (H+1, 3)
    thrust_mag = np.linalg.norm(T_w, axis=-1, keepdims=True)  # (H+1, 1)
    
    # Check max thrust limit: 4 * max_rotor_thrust
    max_collective = 4.0 * KF * (MAX_THRUST ** 2)
    min_collective = 4.0 * KF * (200.0 ** 2)
    if np.any(thrust_mag > max_collective * 0.95) or np.any(thrust_mag < min_collective):
        return None, None, False
        
    z_b = T_w / np.maximum(thrust_mag, 1e-6)  # (H+1, 3)
    
    # Heading vector in horizontal plane
    x_c = np.stack([np.cos(yaw_traj), np.sin(yaw_traj), np.zeros_like(yaw_traj)], axis=-1)
    
    # y_b = (z_b x x_c) / ||z_b x x_c||
    y_b_raw = np.cross(z_b, x_c)
    y_b_norm = np.linalg.norm(y_b_raw, axis=-1, keepdims=True)
    if np.any(y_b_norm < 1e-4):
        # Gimbal lock when z_b is parallel to x_c (pointing horizontally along heading)
        return None, None, False
    y_b = y_b_raw / y_b_norm
    
    # x_b = y_b x z_b
    x_b = np.cross(y_b, z_b)
    
    # Rotation matrix R = [x_b, y_b, z_b]
    # Shape: (H+1, 3, 3)
    R = np.stack([x_b, y_b, z_b], axis=-1)
    
    # Check tilt angle: z_b . z_w = cos(tilt)
    tilt_cos = z_b[:, 2]
    max_tilt_deg = 70.0
    if np.any(tilt_cos < np.cos(np.radians(max_tilt_deg))):
        return None, None, False
        
    # Convert R to quaternions (H+1, 4)
    quats = np.array([R_to_quat(R[k]) for k in range(H + 1)], dtype=np.float32)
    
    # Linear velocity in body frame: v_b = R^T @ v_w
    v_b = np.einsum('nij,ni->nj', R, vel_traj)
    
    # Angular velocity in body frame:
    # T_dot = m * jerk
    T_dot = m * jerk_traj
    # z_b_dot = (T_dot - (z_b . T_dot) * z_b) / thrust_mag
    z_b_dot = (T_dot - np.sum(z_b * T_dot, axis=-1, keepdims=True) * z_b) / thrust_mag
    
    # p_body = - y_b . z_b_dot
    # q_body =   x_b . z_b_dot
    # r_body =   yaw_dot * (z_w . z_b)
    p_body = - np.sum(y_b * z_b_dot, axis=-1)
    q_body =   np.sum(x_b * z_b_dot, axis=-1)
    r_body =   yaw_dot_traj * z_b[:, 2]
    w_b = np.stack([p_body, q_body, r_body], axis=-1)
    
    # Rotor thrust distribution
    # Collective thrust divided by 4
    thrust_per_rotor = thrust_mag[:, 0] / 4.0
    rpm_per_rotor = np.sqrt(np.maximum(thrust_per_rotor / KF, 0.0))
    rotor_ratio = np.clip(rpm_per_rotor / MAX_THRUST, 0.0, 1.0)
    
    # 4-dim control inputs (H, 4)
    controls = np.repeat(rotor_ratio[:H, None], 4, axis=-1).astype(np.float32)
    
    # Full 17-state
    states = np.zeros((H + 1, 17), dtype=np.float32)
    states[:, 0:3] = pos_traj
    states[:, 3:7] = quats
    states[:, 7:10] = v_b
    states[:, 10:13] = w_b
    states[:, 13:17] = np.repeat(rotor_ratio[:, None], 4, axis=-1)
    
    return states, controls, True


if __name__ == "__main__":
    t0 = time.time()
    valid_count = 0
    N = 100
    for i in range(N):
        H = 100
        dt = 0.02
        t = np.linspace(0, H * dt, H + 1)
        # Random Lissajous
        Ax, Ay, Az = np.random.uniform(0.4, 1.2, 3)
        wx, wy, wz = np.random.uniform(1.0, 2.5, 3)
        pos = np.stack([
            Ax * np.sin(wx * t),
            Ay * np.sin(wy * t),
            1.2 + Az * np.sin(wz * t)
        ], axis=-1)
        vel = np.stack([
            Ax * wx * np.cos(wx * t),
            Ay * wy * np.cos(wy * t),
            Az * wz * np.cos(wz * t)
        ], axis=-1)
        acc = np.stack([
            -Ax * (wx**2) * np.sin(wx * t),
            -Ay * (wy**2) * np.sin(wy * t),
            -Az * (wz**2) * np.sin(wz * t)
        ], axis=-1)
        jerk = np.stack([
            -Ax * (wx**3) * np.cos(wx * t),
            -Ay * (wy**3) * np.cos(wy * t),
            -Az * (wz**3) * np.cos(wz * t)
        ], axis=-1)
        yaw = np.arctan2(vel[:, 1], vel[:, 0])
        # Smooth unwrap
        yaw = np.unwrap(yaw)
        yaw_dot = np.gradient(yaw, dt)
        
        s, c, ok = generate_differential_flatness_trajectory(pos, vel, acc, jerk, yaw, yaw_dot, dt)
        if ok:
            valid_count += 1
            
    elapsed = time.time() - t0
    print(f"Generated {valid_count}/{N} valid trajectories in {elapsed:.3f}s ({elapsed*1000/N:.2f} ms/traj)!")
