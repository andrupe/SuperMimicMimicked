import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np, jax, jax.numpy as jnp
from quadrotor.params import HOVER_RPM, MAX_THRUST, SIM_DT, NX
from quadrotor.supermimic_env import sim_step_multi_substeps
from quadrotor.quaternion import rpy_to_quat, quat_to_R

# 1. Takeoff Trajectory (0.04m -> 1.20m, 2.0s = 100 steps)
def generate_takeoff_trajectory(T=2.0, z_start=0.04, z_target=1.20, dt=0.02):
    H = int(round(T / dt))
    t = np.linspace(0.0, 1.0, H + 1)
    # Quintic polynomial for position
    s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    s_dot = (30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4) / T
    s_ddot = (60.0 * t - 180.0 * t**2 + 120.0 * t**3) / (T**2)

    z = z_start + (z_target - z_start) * s
    vz = (z_target - z_start) * s_dot
    az = (z_target - z_start) * s_ddot

    states = []
    controls = []
    # Feedforward collective thrust: m*(g + az)
    # Total thrust = 4 * kF * u_rpm^2 = m*(g + az)
    # u_rpm = sqrt(m*(g + az) / (4 * kF))
    from quadrotor.params import M_NOMINAL, KF, G
    for k in range(H + 1):
        st = np.zeros(22, dtype=np.float32)
        st[2] = z[k]
        st[3:12] = np.eye(3).flatten() # Level
        st[14] = vz[k] # vb_z in body frame (level -> vb_z = vz)
        states.append(st)

    for k in range(H):
        thrust_req = M_NOMINAL * (G + az[k])
        thrust_req = max(0.01, thrust_req)
        rpm = np.sqrt(thrust_req / (4.0 * KF))
        controls.append(np.full(4, rpm, dtype=np.float32))

    return np.array(states), np.array(controls)

# 2. Landing Trajectory (1.20m -> 0.04m, 2.5s = 125 steps)
def generate_landing_trajectory(T=2.5, z_start=1.20, z_target=0.04, dt=0.02):
    H = int(round(T / dt))
    t = np.linspace(0.0, 1.0, H + 1)
    s = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    s_dot = (30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4) / T
    s_ddot = (60.0 * t - 180.0 * t**2 + 120.0 * t**3) / (T**2)

    z = z_start + (z_target - z_start) * s
    vz = (z_target - z_start) * s_dot
    az = (z_target - z_start) * s_ddot

    from quadrotor.params import M_NOMINAL, KF, G
    states = []
    controls = []
    for k in range(H + 1):
        st = np.zeros(22, dtype=np.float32)
        st[2] = z[k]
        st[3:12] = np.eye(3).flatten()
        st[14] = vz[k]
        states.append(st)

    for k in range(H):
        if k < H - 5:
            thrust_req = M_NOMINAL * (G + az[k])
            thrust_req = max(0.01, thrust_req)
            rpm = np.sqrt(thrust_req / (4.0 * KF))
        else:
            # Cut motor thrust on touchdown
            rpm = 0.0
        controls.append(np.full(4, rpm, dtype=np.float32))

    return np.array(states), np.array(controls)

# Test both
to_st, to_u = generate_takeoff_trajectory()
print(f"Takeoff: {len(to_st)} states, z={to_st[0, 2]:.2f}m -> {to_st[-1, 2]:.2f}m, RPM={to_u.mean(axis=1).min():.1f} - {to_u.mean(axis=1).max():.1f}")

land_st, land_u = generate_landing_trajectory()
print(f"Landing: {len(land_st)} states, z={land_st[0, 2]:.2f}m -> {land_st[-1, 2]:.2f}m, RPM={land_u.mean(axis=1).min():.1f} - {land_u.mean(axis=1).max():.1f}")
