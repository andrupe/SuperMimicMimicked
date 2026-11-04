import sys, os
sys.path.insert(0, os.path.abspath("."))
import h5py, numpy as np, jax, jax.numpy as jnp, optax, time
from quadrotor.params import HOVER_RPM, MAX_THRUST, SIM_DT
from quadrotor.world_model import (
    QuadrotorState, unpack_state_22, pack_state_22, invariant_features,
    WorldModel, THRUST_SCALE
)
from quadrotor.rotations_jax import rot_angle_sq_jax
from quadrotor.supermimic import SuperMimicPolicy, compute_obs, normalize_obs, norm_action_to_rpm
from quadrotor.sim_bridge import _nominal_cf_rk4_jax
from quadrotor.quaternion import state17_to_22

print("Testing Dynamic Trajectory Tracking (Point-to-Point, Figure-8, Helix)...")

# Observation normalization
obs_mean = np.zeros(22, dtype=np.float32)
obs_mean[3:12] = np.eye(3).flatten()
obs_std = np.ones(22, dtype=np.float32)
obs_std[0:3] = 0.50
obs_std[3:12] = 0.50
obs_std[12:15] = 1.00
obs_std[15:18] = 2.00
obs_std[18:22] = 0.50
obs_m_j, obs_s_j = jnp.asarray(obs_mean), jnp.asarray(obs_std)

# Select 3 clean trajectories: 43 (p2p), 47 (figure8), 49 (helix)
corpus_path = "corpus/crazyflie_corpus_50trajs.h5"
eval_trajs = []
eval_actions = []
with h5py.File(corpus_path, "r") as f:
    for idx in [43, 47, 49]:
        grp = f["trajectories"][f"traj_{idx:06d}"]
        xs_22 = np.array([state17_to_22(x) for x in grp["states"][:]])[2:]
        us_rpm = (grp["controls"][:] * MAX_THRUST)[2:]
        a_exp = (us_rpm - HOVER_RPM) / THRUST_SCALE
        eval_trajs.append(xs_22)
        eval_actions.append(a_exp)

print(f"Loaded {len(eval_trajs)} trajectories.")

policy = SuperMimicPolicy()
rng = jax.random.PRNGKey(42)
rng, k_pi = jax.random.split(rng)
p_pi = policy.init(k_pi, jnp.zeros((1, 22)))["params"]

tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3, b1=0.9, b2=0.95)
)
opt_state = tx.init(p_pi)

H = 20

@jax.jit
def single_rollout_loss(p_curr, s0, a_init, ref_w):
    def step(carry, t):
        s_curr, a_prev = carry
        p_tgt = ref_w[t + 1, 0:3]
        o = compute_obs(s_curr, p_tgt, a_prev)
        o_n = normalize_obs(o, obs_m_j, obs_s_j)
        a = policy.apply({"params": p_curr}, o_n)
        u = norm_action_to_rpm(a)
        def sub(i, st): return _nominal_cf_rk4_jax(st, u, 0.005)
        s_next = jax.lax.fori_loop(0, 4, sub, s_curr)
        
        # Loss terms:
        # 1. Position tracking
        pos_err = s_next[0:3] - ref_w[t + 1, 0:3]
        pos_loss = jnp.sum(jnp.square(pos_err))
        
        # 2. Attitude tracking: essential for quadrotor thrust vector alignment!
        R_curr = s_next[3:12].reshape((3, 3))
        R_ref = ref_w[t + 1, 3:12].reshape((3, 3))
        rot_loss = 0.2 * rot_angle_sq_jax(R_curr, R_ref)
        
        # 3. Velocities & smoothness
        vel_loss = 0.05 * jnp.sum(jnp.square(s_next[12:15] - ref_w[t + 1, 12:15]))
        ang_loss = 0.02 * jnp.sum(jnp.square(s_next[15:18] - ref_w[t + 1, 15:18]))
        act_loss = 0.01 * jnp.sum(jnp.square(a - a_prev))
        
        return (s_next, a), (pos_loss + rot_loss + vel_loss + ang_loss + act_loss)

    _, step_losses = jax.lax.scan(step, (s0, a_init), jnp.arange(H))
    discounts = 0.99 ** jnp.arange(H)
    return jnp.sum(discounts * step_losses)

@jax.jit
def bptt_step(p, opt_s, s0_b, a0_b, ref_b):
    def loss_fn(p_curr):
        losses = jax.vmap(lambda s0, a0, ref: single_rollout_loss(p_curr, s0, a0, ref))(s0_b, a0_b, ref_b)
        return jnp.mean(losses)
    l, g = jax.value_and_grad(loss_fn)(p)
    u, opt_s = tx.update(g, opt_s, p)
    p = optax.apply_updates(p, u)
    return p, opt_s, l

@jax.jit
def eval_traj(p, xs_ref, a0):
    traj_len = len(xs_ref) - 1
    def step(carry, t):
        s_curr, a_prev = carry
        o = compute_obs(s_curr, xs_ref[t + 1, 0:3], a_prev)
        o_n = normalize_obs(o, obs_m_j, obs_s_j)
        a = policy.apply({"params": p}, o_n)
        u = norm_action_to_rpm(a)
        def sub(i, st): return _nominal_cf_rk4_jax(st, u, 0.005)
        s_next = jax.lax.fori_loop(0, 4, sub, s_curr)
        err = jnp.linalg.norm(s_next[0:3] - xs_ref[t + 1, 0:3])
        return (s_next, a), err
    _, errs = jax.lax.scan(step, (xs_ref[0], a0), jnp.arange(traj_len))
    return jnp.sqrt(jnp.mean(jnp.square(errs))) * 100.0

def eval_all(p):
    rmses = [float(eval_traj(p, jnp.asarray(eval_trajs[i]), jnp.asarray(eval_actions[i][0]))) for i in range(len(eval_trajs))]
    return float(np.mean(rmses)), rmses

mean_rmse, rmses = eval_all(p_pi)
print(f"Initial RMSE: {mean_rmse:.2f} cm (per traj: {[round(r, 2) for r in rmses]})")

rng_np = np.random.default_rng(42)
t0 = time.perf_counter()
batch_size = 48

for it in range(400):
    s0_list, a0_list, ref_list = [], [], []
    for _ in range(batch_size):
        t_id = rng_np.integers(0, len(eval_trajs))
        traj = eval_trajs[t_id]
        acts = eval_actions[t_id]
        f_max = len(traj) - (H + 1)
        f_start = rng_np.integers(0, max(1, f_max))
        
        slice_w = traj[f_start: f_start + H + 1]
        if len(slice_w) < H + 1:
            pad = np.tile(traj[-1:], (H + 1 - len(slice_w), 1))
            slice_w = np.concatenate([slice_w, pad], axis=0)
            
        s0 = slice_w[0].copy()
        # Initial state perturbation for feedback gain learning
        s0[0:3] += rng_np.normal(0, 0.015, size=3)
        s0[12:15] += rng_np.normal(0, 0.02, size=3)
        
        a0 = acts[f_start].copy()
        
        s0_list.append(s0)
        a0_list.append(a0)
        ref_list.append(slice_w)
        
    s0_batch = jnp.asarray(np.array(s0_list, dtype=np.float32))
    a0_batch = jnp.asarray(np.array(a0_list, dtype=np.float32))
    ref_batch = jnp.asarray(np.array(ref_list, dtype=np.float32))
    
    p_pi, opt_state, loss = bptt_step(p_pi, opt_state, s0_batch, a0_batch, ref_batch)
    
    if (it + 1) % 50 == 0:
        mean_rmse, rmses = eval_all(p_pi)
        elapsed = time.perf_counter() - t0
        print(f"Iter {it+1:3d} ({elapsed:.1f}s) | Loss: {float(loss):.5f} | Mean Val: {mean_rmse:.2f} cm | Trajs: {[round(r, 2) for r in rmses]}")
