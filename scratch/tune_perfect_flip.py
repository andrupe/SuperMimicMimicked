import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np, jax, jax.numpy as jnp
from quadrotor.params import HOVER_RPM, MAX_THRUST, SIM_DT
from quadrotor.supermimic_env import sim_step_multi_substeps

# Build planned control sequence for 120 steps (2.4 seconds)
u_hover = np.full(4, HOVER_RPM, dtype=np.float32)

def generate_flip(punch_rpm=2400.0, punch_steps=9, burst_steps=4, coast_steps=7, brake_steps=4, catch_steps=12):
    u_seq = []
    # 1. Hover hold (10 steps = 0.20s)
    for _ in range(10):
        u_seq.append(u_hover.copy())
    # 2. Pre-climb punch
    for _ in range(punch_steps):
        u_seq.append(np.full(4, punch_rpm, dtype=np.float32))
    # 3. High pitch torque burst (rear max, front min)
    for _ in range(burst_steps):
        u_seq.append(np.array([1050.0, 2618.0, 2618.0, 1050.0], dtype=np.float32))
    # 4. Inverted coast
    for _ in range(coast_steps):
        u_seq.append(np.array([1200.0, 1500.0, 1500.0, 1200.0], dtype=np.float32))
    # 5. Flare braking (front max, rear min)
    for _ in range(brake_steps):
        u_seq.append(np.array([2618.0, 1050.0, 1050.0, 2618.0], dtype=np.float32))
    # 6. Catch & level
    for _ in range(catch_steps):
        u_seq.append(np.full(4, 2150.0, dtype=np.float32))
    # 7. Final settle to hover (rest of steps up to 120)
    rem = 120 - len(u_seq)
    for _ in range(rem):
        u_seq.append(u_hover.copy())

    return np.array(u_seq, dtype=np.float32)

u_seq = jnp.asarray(generate_flip())
s0_22 = np.zeros(22, dtype=np.float32)
s0_22[0:3] = [0.0, 0.0, 1.2]
s0_22[3:12] = np.eye(3).flatten()
s0_22 = jnp.asarray(s0_22)
xi0 = jnp.zeros(10, dtype=np.float32)

@jax.jit
def run_rollout(s_init, us, xi):
    def step(carry, u):
        s = carry
        s_next = sim_step_multi_substeps(s, u, xi, substeps=4, sub_dt=0.005)
        return s_next, s_next
    _, states = jax.lax.scan(step, s_init, us)
    return jnp.concatenate([s_init[None], states], axis=0)

all_states = np.array(run_rollout(s0_22, u_seq, xi0))

print("=" * 65)
print(f"Perfect Flip Trajectory: {len(all_states)} states ({len(u_seq)} controls, {len(u_seq)*SIM_DT:.2f}s)")
print(f"Altitude (z): start={all_states[0, 2]:.2f}m, min={all_states[:, 2].min():.2f}m, max={all_states[:, 2].max():.2f}m, final={all_states[-1, 2]:.2f}m")
print(f"Pitch rate (wy): min={all_states[:, 16].min():.2f} rad/s, max={all_states[:, 16].max():.2f} rad/s")
print(f"Attitude R33: min={all_states[:, 11].min():.3f} (inverted), final={all_states[-1, 11]:.3f} (upright)")
print(f"Final velocity norm: {np.linalg.norm(all_states[-1, 12:15]):.3f} m/s")
print(f"Final angular rate norm: {np.linalg.norm(all_states[-1, 15:18]):.3f} rad/s")
print("=" * 65)
