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

print("Testing BPTT with Perturbed Initial States for Feedback Gains...")

# 1. Observation normalization
obs_mean = np.zeros(22, dtype=np.float32)
obs_mean[3:12] = np.eye(3).flatten()
obs_std = np.ones(22, dtype=np.float32)
obs_std[0:3] = 0.50
obs_std[3:12] = 0.50
obs_std[12:15] = 1.00
obs_std[15:18] = 2.00
obs_std[18:22] = 0.50
obs_m_j, obs_s_j = jnp.asarray(obs_mean), jnp.asarray(obs_std)

# 2. Ingest Corpus (Trimmed from t=2)
corpus_path = "corpus/crazyflie_corpus_50trajs.h5"
train_trajs = []
val_trajs = []
with h5py.File(corpus_path, "r") as f:
    for idx in range(50):
        grp = f["trajectories"][f"traj_{idx:06d}"]
        xs_22 = np.array([state17_to_22(x) for x in grp["states"][:]])[2:]
        if idx < 43:
            train_trajs.append(xs_22)
        else:
            val_trajs.append(xs_22)

print(f"Loaded {len(train_trajs)} train trajs, {len(val_trajs)} val trajs.")

# 3. Policy & Optimizer
policy = SuperMimicPolicy()
rng = jax.random.PRNGKey(42)
rng, k_pi = jax.random.split(rng)
p_pi = policy.init(k_pi, jnp.zeros((1, 22)))["params"]

tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3, b1=0.9, b2=0.95)
)
opt_state = tx.init(p_pi)

# BPTT Horizon H=16
H = 16
batch_size = 64

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
        
        # Immediate tracking reward:
        # 1. Huber distance
        pos_err = s_next[0:3] - ref_w[t + 1, 0:3]
        pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_err)) + 1e-4)
        r_pos = -2.0 * pos_dist + 1.0 * jnp.exp(-jnp.sum(jnp.square(pos_err)) / (0.05 ** 2))
        
        # 2. Attitude error
        R_curr = s_next[3:12].reshape((3, 3))
        R_ref = ref_w[t + 1, 3:12].reshape((3, 3))
        e_rot = jnp.sqrt(rot_angle_sq_jax(R_curr, R_ref) + 1e-4)
        r_rot = -0.5 * e_rot
        
        # 3. Velocities
        r_vel = -0.1 * jnp.sum(jnp.square(s_next[12:15] - ref_w[t + 1, 12:15]))
        r_ang = -0.05 * jnp.sum(jnp.square(s_next[15:18] - ref_w[t + 1, 15:18]))
        r_act = -0.02 * jnp.sum(jnp.square(a - a_prev))
        
        step_r = r_pos + r_rot + r_vel + r_ang + r_act
        return (s_next, a), step_r

    _, step_rews = jax.lax.scan(step, (s0, a_init), jnp.arange(H))
    discounts = 0.99 ** jnp.arange(H)
    return -jnp.sum(discounts * step_rews)

@jax.jit
def bptt_batch_update(p, opt_s, s0_b, a_init_b, ref_b):
    def loss_fn(p_curr):
        losses = jax.vmap(lambda s0, a0, ref: single_rollout_loss(p_curr, s0, a0, ref))(s0_b, a_init_b, ref_b)
        return jnp.mean(losses)
    l, g = jax.value_and_grad(loss_fn)(p)
    u, opt_s = tx.update(g, opt_s, p)
    p = optax.apply_updates(p, u)
    return p, opt_s, l

@jax.jit
def eval_traj(p, xs_ref):
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
        err_clamped = jnp.where(jnp.isnan(err) | (err > 3.0), 3.0, err)
        return (s_next, a), err_clamped
    _, errs = jax.lax.scan(step, (xs_ref[0], jnp.zeros(4)), jnp.arange(traj_len))
    return jnp.sqrt(jnp.mean(jnp.square(errs))) * 100.0

def eval_all(p):
    val_rmses = [float(eval_traj(p, jnp.asarray(t))) for t in val_trajs]
    return float(np.mean(val_rmses)), val_rmses

# Initial eval
mean_val, val_list = eval_all(p_pi)
print(f"Initial Mean Val RMSE: {mean_val:.2f} cm (per traj: {[round(v, 1) for v in val_list]})")

# Sampling helper with perturbations
rng_np = np.random.default_rng(42)
t0 = time.perf_counter()

for it in range(300):
    # Sample 64 random trajectory windows of length H + 1
    s0_list = []
    a_init_list = []
    ref_list = []
    for _ in range(batch_size):
        t_id = rng_np.integers(0, len(train_trajs))
        traj = train_trajs[t_id]
        f_max = len(traj) - (H + 1)
        if f_max <= 0:
            f_start = 0
        else:
            f_start = rng_np.integers(0, f_max)
        
        slice_w = traj[f_start: f_start + H + 1]
        if len(slice_w) < H + 1:
            pad = np.tile(traj[-1:], (H + 1 - len(slice_w), 1))
            slice_w = np.concatenate([slice_w, pad], axis=0)
            
        s0 = slice_w[0].copy()
        # Add state perturbations: 2 cm in position, 3 cm/s in velocity
        s0[0:3] += rng_np.normal(0, 0.02, size=3)
        s0[12:15] += rng_np.normal(0, 0.03, size=3)
        
        # Initial action: expert action at this step
        u_rpm0 = slice_w[0, 18:22]
        a0 = np.clip((u_rpm0 - HOVER_RPM) / THRUST_SCALE, -1.0, 1.0)
        
        s0_list.append(s0)
        a_init_list.append(a0)
        ref_list.append(slice_w)
        
    s0_batch = jnp.asarray(np.array(s0_list, dtype=np.float32))
    a0_batch = jnp.asarray(np.array(a_init_list, dtype=np.float32))
    ref_batch = jnp.asarray(np.array(ref_list, dtype=np.float32))
    
    p_pi, opt_state, loss = bptt_batch_update(p_pi, opt_state, s0_batch, a0_batch, ref_batch)
    
    if (it + 1) % 50 == 0:
        mean_val, val_list = eval_all(p_pi)
        elapsed = time.perf_counter() - t0
        print(f"Iter {it+1:3d} ({elapsed:.1f}s) | Loss: {float(loss):.4f} | Mean Val: {mean_val:.2f} cm | Trajs: {[round(v, 1) for v in val_list]}")
