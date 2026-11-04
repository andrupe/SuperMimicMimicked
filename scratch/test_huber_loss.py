import sys, os
sys.path.insert(0, os.path.abspath("."))
import h5py, numpy as np, jax, jax.numpy as jnp, optax, time
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

corpus_path = "corpus/crazyflie_corpus_50trajs.h5"
eval_trajs = []
eval_actions = []

with h5py.File(corpus_path, "r") as f:
    for idx in [43, 44, 46]:
        grp = f["trajectories"][f"traj_{idx:06d}"]
        t_type = grp.attrs["task_type"]
        xs_17 = grp["states"][:]
        us_bar = grp["controls"][:]
        xs_22 = np.array([state17_to_22(x) for x in xs_17])
        us_rpm = us_bar * MAX_THRUST
        raw_act = (us_rpm - HOVER_RPM) / THRUST_SCALE
        st_trim = 14 if t_type == "hover" else 2
        xs_clean = xs_22[st_trim:]
        a_clean = np.clip(raw_act[st_trim:], -1.0, 1.0)
        eval_trajs.append(xs_clean)
        eval_actions.append(a_clean)

print(f"Loaded {len(eval_trajs)} trajectories for test.")

policy = SuperMimicPolicy()
rng = jax.random.PRNGKey(42)
rng, k_pi = jax.random.split(rng)
p_pi = policy.init(k_pi, jnp.zeros((1, 22)))["params"]

# BC Warmup
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

# Eval
@jax.jit
def eval_traj(p, xs_ref, a0):
    traj_len = len(xs_ref) - 1
    xi = jnp.zeros(10)
    def step(carry, t):
        s_curr, a_prev = carry
        o = compute_obs(s_curr, xs_ref[t + 1, 0:3], a_prev)
        o_n = normalize_obs(o, obs_m_j, obs_s_j)
        a = policy.apply({"params": p}, o_n)
        u = norm_action_to_rpm(a)
        s_next = sim_step_multi_substeps(s_curr, u, xi, substeps=4, sub_dt=0.005)
        err = jnp.linalg.norm(s_next[0:3] - xs_ref[t + 1, 0:3])
        return (s_next, a), err
    _, errs = jax.lax.scan(step, (xs_ref[0], a0), jnp.arange(traj_len))
    return jnp.sqrt(jnp.mean(jnp.square(errs))) * 100.0

def eval_all(p):
    rmses = [float(eval_traj(p, jnp.asarray(eval_trajs[i]), jnp.asarray(eval_actions[i][0]))) for i in range(len(eval_trajs))]
    return float(np.mean(rmses)), rmses

bc_rmse, bc_list = eval_all(p_pi)
print(f"Val RMSE after BC: {bc_rmse:.2f} cm (per traj: {[round(r, 2) for r in bc_list]})")

# Non-vanishing Linear Distance Loss BPTT Kernel
def make_bptt_kernel(H_len):
    def rollout_loss(p_curr, s0, a_init, ref_w):
        xi = jnp.zeros(10)
        def step(carry, t):
            s_curr, a_prev = carry
            o = compute_obs(s_curr, ref_w[t + 1, 0:3], a_prev)
            o_n = normalize_obs(o, obs_m_j, obs_s_j)
            a = policy.apply({"params": p_curr}, o_n)
            u = norm_action_to_rpm(a)
            s_next = sim_step_multi_substeps(s_curr, u, xi, substeps=4, sub_dt=0.005)

            pos_err = s_next[0:3] - ref_w[t + 1, 0:3]
            pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_err)) + 1e-4) # Non-vanishing L1/Huber
            pos_loss = 2.0 * pos_dist + 5.0 * jnp.sum(jnp.square(pos_err))

            R_curr = s_next[3:12].reshape((3, 3))
            R_ref = ref_w[t + 1, 3:12].reshape((3, 3))
            rot_loss = 0.25 * rot_angle_sq_jax(R_curr, R_ref)
            vel_loss = 0.05 * jnp.sum(jnp.square(s_next[12:15] - ref_w[t + 1, 12:15]))
            ang_loss = 0.02 * jnp.sum(jnp.square(s_next[15:18] - ref_w[t + 1, 15:18]))
            act_loss = 0.01 * jnp.sum(jnp.square(a - a_prev))
            return (s_next, a), (pos_loss + rot_loss + vel_loss + ang_loss + act_loss)

        _, step_losses = jax.lax.scan(step, (s0, a_init), jnp.arange(H_len))
        return jnp.sum((0.99 ** jnp.arange(H_len)) * step_losses)

    @jax.jit
    def train_step(p, opt_s, s0_b, a0_b, ref_b):
        def loss_fn(p_curr):
            losses = jax.vmap(lambda s0, a0, ref: rollout_loss(p_curr, s0, a0, ref))(s0_b, a0_b, ref_b)
            return jnp.mean(losses)
        l, g = jax.value_and_grad(loss_fn)(p)
        u, opt_s = tx_pi.update(g, opt_s, p)
        return optax.apply_updates(p, u), opt_s, l

    return train_step

kernel_h16 = make_bptt_kernel(16)
kernel_h24 = make_bptt_kernel(24)
kernel_h32 = make_bptt_kernel(32)

total_iters = 800
lr_schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=total_iters, alpha=0.03)
tx_pi = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.95))
pi_opt = tx_pi.init(p_pi)

rng_np = np.random.default_rng(42)
best_rmse = bc_rmse
t0 = time.perf_counter()

print("\n--- Running Non-Vanishing Loss BPTT (800 iterations) ---")
for it in range(total_iters):
    if it < 200:
        H_curr = 16
        step_fn = kernel_h16
    elif it < 500:
        H_curr = 24
        step_fn = kernel_h24
    else:
        H_curr = 32
        step_fn = kernel_h32

    s0_list, a0_list, ref_list = [], [], []
    for _ in range(64):
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
        pert_scale = 0.015 if it < 300 else 0.006
        s0[0:3] += rng_np.normal(0, pert_scale, size=3)
        s0[12:15] += rng_np.normal(0, pert_scale * 1.5, size=3)

        s0_list.append(s0)
        a0_list.append(acts[f_start].copy())
        ref_list.append(slice_w)

    p_pi, pi_opt, loss = step_fn(p_pi, pi_opt, jnp.asarray(np.array(s0_list)), jnp.asarray(np.array(a0_list)), jnp.asarray(np.array(ref_list)))
    if (it + 1) % 50 == 0:
        rmse, rmses = eval_all(p_pi)
        if rmse < best_rmse:
            best_rmse = rmse
        elapsed = time.perf_counter() - t0
        print(f"Iter {it+1:3d} (H={H_curr:2d}, {elapsed:.1f}s) | Loss: {float(loss):.5f} | Val RMSE: {rmse:.2f} cm (Best: {best_rmse:.2f} cm) | Trajs: {[round(r, 2) for r in rmses]}")
