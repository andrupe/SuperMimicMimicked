import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np, jax, jax.numpy as jnp
from quadrotor.params import HOVER_RPM, MAX_THRUST, SIM_DT
from quadrotor.supermimic_env import sim_step_multi_substeps
from quadrotor.quaternion import state17_to_22, quat_to_R
from quadrotor.rotations_jax import so3_exp_jax

# Test generating a clean pitch-flip trajectory using the Crazyflie dynamics
s0_22 = np.zeros(22, dtype=np.float32)
s0_22[0:3] = [0.0, 0.0, 1.2] # Initial hover at z = 1.2m
s0_22[3:12] = np.eye(3).flatten() # Level

u_hover = np.full(4, HOVER_RPM, dtype=np.float32)
xi_nominal = np.zeros(10, dtype=np.float32)

states = [s0_22.copy()]
controls = []

# Flip timeline (dt = 0.02s):
# 1. Steady hover: 0.0 - 0.2s (10 steps)
# 2. Pre-climb punch: 0.2 - 0.4s (10 steps) -> collective thrust +30% to gain climb velocity
# 3. Pitch torque burst: 0.4 - 0.5s (5 steps) -> rear motors max, front motors min
# 4. Inverted ballistic coast: 0.5 - 0.64s (7 steps) -> low collective thrust (50% hover)
# 5. Flare braking: 0.64 - 0.74s (5 steps) -> front motors max, rear motors min (reverse torque)
# 6. Altitude arrest & settle: 0.74 - 1.5s (38 steps) -> collective thrust +25%, leveling attitude
# Total steps: 75 steps (1.5 seconds)

s_curr = s0_22.copy()
total_pitch = 0.0

for step in range(85):
    t = step * SIM_DT
    R_curr = s_curr[3:12].reshape((3, 3))
    pitch_angle = np.arctan2(R_curr[0, 2], R_curr[2, 2])
    wy = s_curr[16]

    if t < 0.20:
        # Hover
        u = u_hover.copy()
    elif t < 0.38:
        # Pre-climb punch: boost thrust by 30%
        u = np.full(4, HOVER_RPM * 1.28, dtype=np.float32)
    elif t < 0.46:
        # Pitch burst: rear motors (1, 2) max, front motors (0, 3) min
        u = np.array([1200.0, 2600.0, 2600.0, 1200.0], dtype=np.float32)
    elif t < 0.66:
        # Inverted rotation: low collective thrust to preserve altitude while spinning
        u = np.array([1350.0, 1850.0, 1850.0, 1350.0], dtype=np.float32)
    elif t < 0.74:
        # Flare braking: reverse torque (front motors max, rear motors min)
        u = np.array([2550.0, 1200.0, 1200.0, 2550.0], dtype=np.float32)
    elif t < 0.95:
        # Catch and arrest descent: high collective thrust
        u = np.full(4, HOVER_RPM * 1.22, dtype=np.float32)
    else:
        # Settle to hover
        u = u_hover.copy()

    u = np.clip(u, 0.0, MAX_THRUST)
    controls.append(u.copy())

    s_next = np.array(sim_step_multi_substeps(jnp.asarray(s_curr), jnp.asarray(u), jnp.asarray(xi_nominal), substeps=4, sub_dt=0.005))
    states.append(s_next.copy())
    s_curr = s_next

states = np.array(states)
controls = np.array(controls)

print(f"Generated trajectory: {len(states)} states, {len(controls)} controls.")
print(f"Initial z: {states[0, 2]:.2f} m | Min z: {states[:, 2].min():.2f} m | Max z: {states[:, 2].max():.2f} m | Final z: {states[-1, 2]:.2f} m")
print(f"Max pitch rate wy: {states[:, 16].max():.2f} rad/s | Min pitch rate wy: {states[:, 16].min():.2f} rad/s")
print(f"Min R33 (inversion check): {states[:, 11].min():.3f} (expect < -0.8 for full upside down)")
print(f"Final attitude R33: {states[-1, 11]:.3f} (expect ~1.0 for upright level)")
