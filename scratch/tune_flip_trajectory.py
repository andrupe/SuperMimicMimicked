import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np, jax, jax.numpy as jnp
from quadrotor.params import HOVER_RPM, MAX_THRUST, SIM_DT
from quadrotor.supermimic_env import sim_step_multi_substeps

# Build planned control sequence for 100 steps (2.0 seconds)
u_hover = np.full(4, HOVER_RPM, dtype=np.float32)

def generate_flip_controls(punch_dur=0.20, burst_dur=0.08, brake_dur=0.08, catch_dur=0.25):
    u_seq = []
    for step in range(110):
        t = step * SIM_DT
        if t < 0.20:
            # Steady hover at 1.2m
            u = u_hover.copy()
        elif t < 0.20 + punch_dur:
            # Pre-climb punch: max upward acceleration (2400 RPM)
            u = np.full(4, 2450.0, dtype=np.float32)
        elif t < 0.20 + punch_dur + burst_dur:
            # High-torque pitch burst: rear motors max, front motors min
            u = np.array([1100.0, 2600.0, 2600.0, 1100.0], dtype=np.float32)
        elif t < 0.20 + punch_dur + burst_dur + 0.12:
            # Inverted pass-through: neutral low thrust
            u = np.array([1250.0, 1450.0, 1450.0, 1250.0], dtype=np.float32)
        elif t < 0.20 + punch_dur + burst_dur + 0.12 + brake_dur:
            # Flare braking: reverse torque (front motors max, rear motors min)
            u = np.array([2550.0, 1150.0, 1150.0, 2550.0], dtype=np.float32)
        elif t < 0.20 + punch_dur + burst_dur + 0.12 + brake_dur + catch_dur:
            # Arrest vertical velocity & level out
            u = np.full(4, 2200.0, dtype=np.float32)
        else:
            # Return to level hover
            u = u_hover.copy()
        u_seq.append(np.clip(u, 0.0, MAX_THRUST))
    return np.array(u_seq, dtype=np.float32)

u_seq = jnp.asarray(generate_flip_controls())
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
print(f"Refined Flip Trajectory: {len(all_states)} states ({len(u_seq)} controls, {len(u_seq)*SIM_DT:.2f}s)")
print(f"Altitude (z): start={all_states[0, 2]:.2f}m, min={all_states[:, 2].min():.2f}m, max={all_states[:, 2].max():.2f}m, final={all_states[-1, 2]:.2f}m")
print(f"Pitch rate (wy): min={all_states[:, 16].min():.2f} rad/s, max={all_states[:, 16].max():.2f} rad/s")
print(f"Attitude R33: min={all_states[:, 11].min():.3f} (inverted), final={all_states[-1, 11]:.3f} (upright)")
print(f"Final velocity norm: {np.linalg.norm(all_states[-1, 12:15]):.3f} m/s")
print("=" * 65)
