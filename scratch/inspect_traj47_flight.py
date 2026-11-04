import sys, os
sys.path.insert(0, os.path.abspath("."))
import h5py, numpy as np, jax, jax.numpy as jnp, optax
from quadrotor.params import HOVER_RPM, MAX_THRUST
from quadrotor.supermimic import SuperMimicPolicy, compute_obs, normalize_obs, norm_action_to_rpm
from quadrotor.supermimic_env import sim_step_multi_substeps
from quadrotor.quaternion import state17_to_22

THRUST_SCALE = 0.4 * HOVER_RPM
obs_mean = np.zeros(22, dtype=np.float32)
obs_mean[3:12] = np.eye(3).flatten()
obs_std = np.ones(22, dtype=np.float32)
obs_std[0:3] = 0.50
obs_std[3:12] = 0.50
obs_std[12:15] = 1.00
obs_std[15:18] = 2.00
obs_std[18:22] = 0.50
obs_m_j, obs_s_j = jnp.asarray(obs_mean), jnp.asarray(obs_std)

with h5py.File("corpus/crazyflie_corpus_50trajs.h5", "r") as f:
    grp47 = f["trajectories"]["traj_000047"]
    s17 = grp47["states"][:]
    u_bar = grp47["controls"][:]
    s22 = np.array([state17_to_22(x) for x in s17])
    u_rpm = u_bar * MAX_THRUST
    raw_act = (u_rpm - HOVER_RPM) / THRUST_SCALE
    xs_ref = s22[2:]
    acts_ref = np.clip(raw_act[2:], -1.0, 1.0)

print(f"Traj 47 duration: {len(xs_ref)*0.02:.2f}s ({len(xs_ref)} steps)")
print("Sample positions along Traj 47:")
for t in [0, 20, 40, 60, 80, 100, 120, 140, 160]:
    if t < len(xs_ref):
        print(f"  t={t*0.02:.2f}s (step {t:3d}): pos={xs_ref[t, 0:3]}, vel={xs_ref[t, 12:15]}, act={acts_ref[t]}")
