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

with h5py.File("corpus/crazyflie_corpus_50trajs.h5", "r") as f:
    grp = f["trajectories"]["traj_000047"]
    xs_22 = np.array([state17_to_22(x) for x in grp["states"][:]])[2:]
    us_rpm = (grp["controls"][:] * MAX_THRUST)[2:]
    actions = np.clip((us_rpm - HOVER_RPM) / THRUST_SCALE, -1.0, 1.0)

xs_22_j = jnp.asarray(xs_22)
actions_j = jnp.asarray(actions)

policy = SuperMimicPolicy()
rng = jax.random.PRNGKey(42)
rng, k_pi = jax.random.split(rng)
p_pi = policy.init(k_pi, jnp.zeros((1, 22)))["params"]

# BC Warmup
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

# Eval
@jax.jit
def eval_traj(p):
    H = len(actions)
    xi = jnp.zeros(10)
    def step(carry, t):
        s_curr, a_prev = carry
        o = compute_obs(s_curr, xs_22_j[t + 1, 0:3], a_prev)
        o_n = normalize_obs(o, obs_m_j, obs_s_j)
        a = policy.apply({"params": p}, o_n)
        u = norm_action_to_rpm(a)
        s_next = sim_step_multi_substeps(s_curr, u, xi, substeps=4, sub_dt=0.005)
        err = jnp.linalg.norm(s_next[0:3] - xs_22_j[t + 1, 0:3])
        err_clamped = jnp.where(jnp.isnan(err) | (err > 3.0), 3.0, err)
        return (s_next, a), err_clamped
    _, errs = jax.lax.scan(step, (xs_22_j[0], actions_j[0]), jnp.arange(H))
    return jnp.sqrt(jnp.mean(jnp.square(errs))) * 100.0

bc_rmse = float(eval_traj(p_pi))
print(f"Traj 47 Val RMSE after BC: {bc_rmse:.2f} cm")

# BPTT Kernel with high-precision position loss
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
            pos_loss = 10.0 * jnp.sum(jnp.square(pos_err))
            R_curr = s_next[3:12].reshape((3, 3))
            R_ref = ref_w[t + 1, 3:12].reshape((3, 3))
            rot_loss = 0.5 * rot_angle_sq_jax(R_curr, R_ref)
            vel_loss = 0.1 * jnp.sum(jnp.square(s_next[12:15] - ref_w[t + 1, 12:15]))
            ang_loss = 0.05 * jnp.sum(jnp.square(s_next[15:18] - ref_w[t + 1, 15:18]))
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

total_iters = 600
lr_schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=total_iters, alpha=0.03)
tx_pi = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.95))
pi_opt = tx_pi.init(p_pi)

rng_np = np.random.default_rng(42)
best_rmse = bc_rmse
t0 = time.perf_counter()

for it in range(total_iters):
    if it < 200:
        H_curr = 16
        step_fn = kernel_h16
    elif it < 450:
        H_curr = 24
        step_fn = kernel_h24
    else:
        H_curr = 32
        step_fn = kernel_h32

    s0_list, a0_list, ref_list = [], [], []
    for _ in range(64):
        f_max = len(xs_22) - (H_curr + 1)
        f_start = rng_np.integers(0, max(1, f_max))
        slice_w = xs_22[f_start: f_start + H_curr + 1]
        s0 = slice_w[0].copy()
        pert_scale = 0.012 if it < 300 else 0.005
        s0[0:3] += rng_np.normal(0, pert_scale, size=3)
        s0[12:15] += rng_np.normal(0, pert_scale * 1.5, size=3)
        s0_list.append(s0)
        a0_list.append(actions[f_start].copy())
        ref_list.append(slice_w)

    p_pi, pi_opt, loss = step_fn(p_pi, pi_opt, jnp.asarray(np.array(s0_list)), jnp.asarray(np.array(a0_list)), jnp.asarray(np.array(ref_list)))
    if (it + 1) % 25 == 0:
        rmse = float(eval_traj(p_pi))
        if rmse < best_rmse:
            best_rmse = rmse
        elapsed = time.perf_counter() - t0
        print(f"Iter {it+1:3d} (H={H_curr:2d}, {elapsed:.1f}s) | Loss: {float(loss):.5f} | Traj 47 RMSE: {rmse:.2f} cm (Best: {best_rmse:.2f} cm)")
