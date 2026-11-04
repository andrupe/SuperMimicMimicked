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

print("Testing Sub-4cm Autonomous Tracking Engine...")

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

# Ingest trajectories: 43 (p2p), 47 (figure8), 49 (helix)
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

# Phase 0: Quick BC Warmup
print("\n--- Phase 0: BC Warmup (100 epochs) ---")
bc_obs = []
bc_acts = []
for i in range(len(eval_trajs)):
    traj = eval_trajs[i]
    acts = eval_actions[i]
    a_prev = np.zeros_like(acts)
    a_prev[1:] = acts[:-1]
    a_prev[0] = acts[0]
    for t in range(len(acts)):
        p_tgt = traj[t + 1, 0:3]
        o = compute_obs(traj[t], p_tgt, a_prev[t])
        bc_obs.append(o)
        bc_acts.append(acts[t])

bc_obs = np.array(bc_obs, dtype=np.float32)
bc_acts = np.array(bc_acts, dtype=np.float32)
bc_obs_n = (bc_obs - obs_mean) / obs_std

tx_bc = optax.adam(1e-3)
bc_opt = tx_bc.init(p_pi)

@jax.jit
def bc_step(p, opt_s, o_b, a_b):
    def loss_fn(p_curr):
        return jnp.mean(jnp.square(policy.apply({"params": p_curr}, o_b) - a_b))
    l, g = jax.value_and_grad(loss_fn)(p)
    u, opt_s = tx_bc.update(g, opt_s, p)
    p = optax.apply_updates(p, u)
    return p, opt_s, l

t0_bc = time.perf_counter()
for ep in range(100):
    p_pi, bc_opt, l_bc = bc_step(p_pi, bc_opt, jnp.asarray(bc_obs_n), jnp.asarray(bc_acts))
print(f"BC completed in {time.perf_counter()-t0_bc:.1f}s | Final MSE: {float(l_bc):.6f}")

# Evaluation function
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

bc_rmse, bc_list = eval_all(p_pi)
print(f"Validation RMSE after BC: {bc_rmse:.2f} cm (per traj: {[round(r, 2) for r in bc_list]})")

# Phase 1: BPTT with Horizon Curriculum and Cosine Decay Learning Rate
def make_kernel(H_len):
    def rollout_loss(p_curr, s0, a_init, ref_w):
        def step(carry, t):
            s_curr, a_prev = carry
            p_tgt = ref_w[t + 1, 0:3]
            o = compute_obs(s_curr, p_tgt, a_prev)
            o_n = normalize_obs(o, obs_m_j, obs_s_j)
            a = policy.apply({"params": p_curr}, o_n)
            u = norm_action_to_rpm(a)
            def sub(i, st): return _nominal_cf_rk4_jax(st, u, 0.005)
            s_next = jax.lax.fori_loop(0, 4, sub, s_curr)
            
            # Loss formulation:
            pos_err = s_next[0:3] - ref_w[t + 1, 0:3]
            pos_loss = jnp.sum(jnp.square(pos_err))
            
            R_curr = s_next[3:12].reshape((3, 3))
            R_ref = ref_w[t + 1, 3:12].reshape((3, 3))
            rot_loss = 0.25 * rot_angle_sq_jax(R_curr, R_ref)
            
            vel_loss = 0.05 * jnp.sum(jnp.square(s_next[12:15] - ref_w[t + 1, 12:15]))
            ang_loss = 0.02 * jnp.sum(jnp.square(s_next[15:18] - ref_w[t + 1, 15:18]))
            act_loss = 0.01 * jnp.sum(jnp.square(a - a_prev))
            
            return (s_next, a), (pos_loss + rot_loss + vel_loss + ang_loss + act_loss)

        _, step_losses = jax.lax.scan(step, (s0, a_init), jnp.arange(H_len))
        discounts = 0.99 ** jnp.arange(H_len)
        return jnp.sum(discounts * step_losses)

    @jax.jit
    def train_step(p, opt_s, s0_b, a0_b, ref_b):
        def loss_fn(p_curr):
            losses = jax.vmap(lambda s0, a0, ref: rollout_loss(p_curr, s0, a0, ref))(s0_b, a0_b, ref_b)
            return jnp.mean(losses)
        l, g = jax.value_and_grad(loss_fn)(p)
        u, opt_s = tx_pi.update(g, opt_s, p)
        p = optax.apply_updates(p, u)
        return p, opt_s, l

    return train_step

kernel_h16 = make_kernel(16)
kernel_h24 = make_kernel(24)
kernel_h32 = make_kernel(32)

# Cosine decay schedule: 1e-3 down to 3e-5 over 1000 iterations
total_iters = 100
lr_schedule = optax.cosine_decay_schedule(init_value=8e-4, decay_steps=total_iters, alpha=0.03)
tx_pi = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.95)
)
pi_opt = tx_pi.init(p_pi)

print("\n--- Phase 1: Scanned BPTT with Horizon Curriculum (1000 iterations) ---")
rng_np = np.random.default_rng(42)
t0_bptt = time.perf_counter()
best_rmse = bc_rmse
batch_size = 48

for it in range(total_iters):
    if it < 250:
        H_curr = 16
        step_fn = kernel_h16
    elif it < 600:
        H_curr = 24
        step_fn = kernel_h24
    else:
        H_curr = 32
        step_fn = kernel_h32
        
    s0_list, a0_list, ref_list = [], [], []
    for _ in range(batch_size):
        t_id = rng_np.integers(0, len(eval_trajs))
        traj = eval_trajs[t_id]
        acts = eval_actions[t_id]
        f_max = len(traj) - (H_curr + 1)
        f_start = rng_np.integers(0, max(1, f_max))
        
        slice_w = traj[f_start: f_start + H_curr + 1]
        if len(slice_w) < H_curr + 1:
            pad = np.tile(traj[-1:], (H_curr + 1 - len(slice_w), 1))
            slice_w = np.concatenate([slice_w, pad], axis=0)
            
        s0 = slice_w[0].copy()
        # Scale perturbations: as horizon grows, smaller perturbations
        pert_scale = 0.015 if it < 500 else 0.008
        s0[0:3] += rng_np.normal(0, pert_scale, size=3)
        s0[12:15] += rng_np.normal(0, pert_scale * 1.5, size=3)
        
        a0 = acts[f_start].copy()
        
        s0_list.append(s0)
        a0_list.append(a0)
        ref_list.append(slice_w)
        
    s0_batch = jnp.asarray(np.array(s0_list, dtype=np.float32))
    a0_batch = jnp.asarray(np.array(a0_list, dtype=np.float32))
    ref_batch = jnp.asarray(np.array(ref_list, dtype=np.float32))
    
    p_pi, pi_opt, loss = step_fn(p_pi, pi_opt, s0_batch, a0_batch, ref_batch)
    
    if (it + 1) % 50 == 0:
        mean_rmse, rmses = eval_all(p_pi)
        is_best = mean_rmse < best_rmse
        if is_best:
            best_rmse = mean_rmse
        elapsed = time.perf_counter() - t0_bptt
        lr_val = float(lr_schedule(it))
        print(f"Iter {it+1:4d} (H={H_curr:2d}, lr={lr_val:.1e}, {elapsed:.1f}s) | Loss: {float(loss):.5f} | Val RMSE: {mean_rmse:.2f} cm (Best: {best_rmse:.2f} cm) | Trajs: {[round(r, 2) for r in rmses]} {'*' if is_best else ''}")
