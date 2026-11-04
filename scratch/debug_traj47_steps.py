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
    grp = f["trajectories"]["traj_000047"]
    xs_22 = np.array([state17_to_22(x) for x in grp["states"][:]])[2:]
    us_rpm = (grp["controls"][:] * MAX_THRUST)[2:]
    actions = np.clip((us_rpm - HOVER_RPM) / THRUST_SCALE, -1.0, 1.0)

policy = SuperMimicPolicy()
rng = jax.random.PRNGKey(42)
rng, k_pi = jax.random.split(rng)
p_pi = policy.init(k_pi, jnp.zeros((1, 22)))["params"]

# Train BC
bc_obs = []
bc_acts = []
a_prev = np.zeros_like(actions)
a_prev[1:] = actions[:-1]
a_prev[0] = actions[0]
for t in range(len(actions)):
    o = compute_obs(xs_22[t], xs_22[t + 1, 0:3], a_prev[t])
    bc_obs.append(o)
    bc_acts.append(actions[t])

bc_obs_n = (np.array(bc_obs, dtype=np.float32) - obs_mean) / obs_std
bc_acts = np.array(bc_acts, dtype=np.float32)

tx_bc = optax.adam(1e-3)
bc_opt = tx_bc.init(p_pi)

@jax.jit
def bc_step(p, opt_s, o_b, a_b):
    l, g = jax.value_and_grad(lambda p_c: jnp.mean(jnp.square(policy.apply({"params": p_c}, o_b) - a_b)))(p)
    u, opt_s = tx_bc.update(g, opt_s, p)
    return optax.apply_updates(p, u), opt_s, l

for ep in range(100):
    p_pi, bc_opt, _ = bc_step(p_pi, bc_opt, jnp.asarray(bc_obs_n), jnp.asarray(bc_acts))

print("Testing step-by-step rollout with BC policy on Traj 47:")
s_curr = xs_22[0].copy()
a_p = actions[0].copy()
xi = jnp.zeros(10)

@jax.jit
def single_step(s, p_tgt, a_prev):
    o = compute_obs(s, p_tgt, a_prev)
    o_n = normalize_obs(o, obs_m_j, obs_s_j)
    a = policy.apply({"params": p_pi}, o_n)
    u = norm_action_to_rpm(a)
    s_next = sim_step_multi_substeps(s, u, xi, substeps=4, sub_dt=0.005)
    return s_next, a

for t in range(min(50, len(actions))):
    s_next, a_curr = single_step(s_curr, jnp.asarray(xs_22[t + 1, 0:3]), a_p)
    err = np.linalg.norm(np.array(s_next[0:3]) - xs_22[t + 1, 0:3])
    print(f"t={t:2d} | pos_err: {err*100:6.2f} cm | a_curr: {np.round(np.array(a_curr), 2)} | ref_a: {np.round(actions[t], 2)} | z={s_next[2]:.2f}")
    if err > 1.0 or np.isnan(err):
        print(f"Diverged at step {t}!")
        break
    s_curr = np.array(s_next)
    a_p = np.array(a_curr)
