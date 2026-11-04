import sys, os
sys.path.insert(0, os.path.abspath("."))
import h5py, numpy as np, jax, jax.numpy as jnp, optax
from quadrotor.params import HOVER_RPM, MAX_THRUST
from quadrotor.rotations_jax import rot_angle_sq_jax
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
    dt = float(grp47.attrs["dt"])
    s22 = np.array([state17_to_22(x) for x in s17])
    u_rpm = u_bar * MAX_THRUST
    raw_act = (u_rpm - HOVER_RPM) / THRUST_SCALE
    xs_ref = s22[2:]
    acts_ref = np.clip(raw_act[2:], -1.0, 1.0)

print(f"Traj 47 clean len: {len(xs_ref)}, dt={dt}")
print(f"Ref start pos: {xs_ref[0, 0:3]}, end pos: {xs_ref[-1, 0:3]}")
print(f"Ref z range: [{xs_ref[:, 2].min():.3f}, {xs_ref[:, 2].max():.3f}]")
print(f"Ref x range: [{xs_ref[:, 0].min():.3f}, {xs_ref[:, 0].max():.3f}]")
print(f"Ref y range: [{xs_ref[:, 1].min():.3f}, {xs_ref[:, 1].max():.3f}]")
print(f"Expert acts min: {acts_ref.min(axis=0)}, max: {acts_ref.max(axis=0)}, mean: {acts_ref.mean(axis=0)}")

# Check open-loop RK4 on Traj 47 with jitted scan
@jax.jit
def open_loop_eval(xs, us):
    def step(s, u):
        s_next = sim_step_multi_substeps(s, u, jnp.zeros(10), substeps=4, sub_dt=0.005)
        return s_next, s_next
    _, s_sim = jax.lax.scan(step, xs[0], us)
    errs = jnp.linalg.norm(s_sim[:, 0:3] - xs[1:, 0:3], axis=-1)
    return jnp.sqrt(jnp.mean(jnp.square(errs))) * 100.0, errs

rmse_ol, errs_ol = open_loop_eval(jnp.asarray(xs_ref), jnp.asarray(u_rpm[2:]))
print(f"Traj 47 open-loop RMSE: {float(rmse_ol):.4f} cm, max err: {float(jnp.max(errs_ol))*100:.4f} cm")
