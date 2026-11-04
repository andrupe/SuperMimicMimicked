import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np, jax, jax.numpy as jnp
from quadrotor.params import HOVER_RPM, MAX_THRUST, SIM_DT
from quadrotor.supermimic_env import sim_step_multi_substeps

# Build planned control sequence for 90 steps (1.8 seconds)
u_hover = np.full(4, HOVER_RPM, dtype=np.float32)
u_seq = []
for step in range(90):
    t = step * SIM_DT
    if t < 0.20:
        # Initial hover
        u = u_hover.copy()
    elif t < 0.36:
        # Pre-climb punch: collective thrust +30%
        u = np.full(4, HOVER_RPM * 1.30, dtype=np.float32)
    elif t < 0.44:
        # Pitch burst: rear motors (1, 2) max, front motors (0, 3) min
        u = np.array([1150.0, 2600.0, 2600.0, 1150.0], dtype=np.float32)
    elif t < 0.60:
        # Inverted rotation: low collective thrust to avoid accelerating into ground
        u = np.array([1200.0, 1700.0, 1700.0, 1200.0], dtype=np.float32)
    elif t < 0.68:
        # Flare braking: reverse torque (front motors max, rear motors min)
        u = np.array([2550.0, 1150.0, 1150.0, 2550.0], dtype=np.float32)
    elif t < 0.90:
        # Catch and arrest descent
        u = np.full(4, HOVER_RPM * 1.25, dtype=np.float32)
    else:
        # Hover hold
        u = u_hover.copy()
    u_seq.append(np.clip(u, 0.0, MAX_THRUST))

u_seq = jnp.asarray(np.array(u_seq, dtype=np.float32))

s0_22 = np.zeros(22, dtype=np.float32)
s0_22[0:3] = [0.0, 0.0, 1.2] # Initial hover at z = 1.2m
s0_22[3:12] = np.eye(3).flatten() # Level
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
all_controls = np.array(u_seq)

print("=" * 60)
print(f"Rollout successfully finished: {len(all_states)} states ({len(all_controls)} steps).")
print(f"Altitude (z): start={all_states[0, 2]:.2f}m, min={all_states[:, 2].min():.2f}m, max={all_states[:, 2].max():.2f}m, end={all_states[-1, 2]:.2f}m")
print(f"Pitch rate (wy): min={all_states[:, 16].min():.2f} rad/s, max={all_states[:, 16].max():.2f} rad/s")
print(f"Attitude R33: min={all_states[:, 11].min():.3f} (< -0.8 means fully inverted!), end={all_states[-1, 11]:.3f} (upright=1.0)")
print("=" * 60)
