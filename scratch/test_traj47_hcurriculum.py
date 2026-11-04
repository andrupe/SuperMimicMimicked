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
    grp47 = f["trajectories"]["traj_000047"]
    s17 = grp47["states"][:]
    u_bar = grp47["controls"][:]
    s22 = np.array([state17_to_22(x) for x in s17])
    u_rpm = u_bar * MAX_THRUST
    raw_act = (u_rpm - HOVER_RPM) / THRUST_SCALE
    xs_ref = s22[2:]
    acts_ref = np.clip(raw_act[2:], -1.0, 1.0)

xs_ref_j = jnp.asarray(xs_ref)
acts_ref_j = jnp.asarray(acts_ref)

policy = SuperMimicPolicy()
p_pi = policy.init(jax.random.PRNGKey(42), jnp.zeros((1, 22)))["params"]

# BC warmup on Traj 47
bc_obs = []
a_prev = np.zeros_like(acts_ref)
a_prev[1:] = acts_ref[:-1]
a_prev[0] = acts_ref[0]
for t in range(len(acts_ref)):
    o = compute_obs(xs_ref[t], xs_ref[t + 1, 0:3], a_prev[t])
    bc_obs.append(o)

bc_obs_n = (np.array(bc_obs, dtype=np.float32) - obs_mean) / obs_std
tx_bc = optax.adam(3e-3)
opt_s = tx_bc.init(p_pi)

@jax.jit
def bc_step(p, opt_s, o_b, a_b):
    l, g = jax.value_and_grad(lambda p_c: jnp.mean(jnp.square(policy.apply({"params": p_c}, o_b) - a_b)))(p)
    u, opt_s = tx_bc.update(g, opt_s, p)
    return optax.apply_updates(p, u), opt_s, l

for _ in range(300):
    p_pi, opt_s, l = bc_step(p_pi, opt_s, jnp.asarray(bc_obs_n), acts_ref_j)

def make_kernel(H_len):
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
            pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_err)) + 1e-4)
            pos_loss = 3.0 * pos_dist + 5.0 * jnp.sum(jnp.square(pos_err))

            R_curr = s_next[3:12].reshape((3, 3))
            R_ref = ref_w[t + 1, 3:12].reshape((3, 3))
            rot_loss = 0.50 * rot_angle_sq_jax(R_curr, R_ref)
            vel_loss = 0.10 * jnp.sum(jnp.square(s_next[12:15] - ref_w[t + 1, 12:15]))
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

kernel_16 = make_kernel(16)
kernel_32 = make_kernel(32)
kernel_64 = make_kernel(64)

total_iters = 1000
lr_sched = optax.cosine_decay_schedule(1e-3, total_iters, alpha=0.03)
tx_pi = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_sched))
pi_opt = tx_pi.init(p_pi)

@jax.jit
def eval_traj47(p):
    def step(carry, t):
        s_curr, a_prev = carry
        o = compute_obs(s_curr, xs_ref_j[t + 1, 0:3], a_prev)
        o_n = normalize_obs(o, obs_m_j, obs_s_j)
        a = policy.apply({"params": p}, o_n)
        u = norm_action_to_rpm(a)
        s_next = sim_step_multi_substeps(s_curr, u, jnp.zeros(10), substeps=4, sub_dt=0.005)
        err = jnp.linalg.norm(s_next[0:3] - xs_ref_j[t + 1, 0:3])
        return (s_next, a), err
    _, errs = jax.lax.scan(step, (xs_ref_j[0], acts_ref_j[0]), jnp.arange(len(xs_ref)-1))
    return jnp.sqrt(jnp.mean(jnp.square(errs))) * 100.0

rng_np = np.random.default_rng(42)
best_rmse = float(eval_traj47(p_pi))
print(f"Initial BC RMSE on Traj 47: {best_rmse:.2f} cm")
t0 = time.perf_counter()

for it in range(total_iters):
    if it < 300:
        H_curr = 16
        k_fn = kernel_16
    elif it < 700:
        H_curr = 32
        k_fn = kernel_32
    else:
        H_curr = 64
        k_fn = kernel_64

    s0_list, a0_list, ref_list = [], [], []
    for _ in range(64):
        f_max = len(xs_ref) - (H_curr + 1)
        f_start = rng_np.integers(0, f_max)
        slice_w = xs_ref[f_start: f_start + H_curr + 1]
        s0 = slice_w[0].copy()
        pert = 0.01 if it < 400 else 0.004
        s0[0:3] += rng_np.normal(0, pert, size=3)
        s0[12:15] += rng_np.normal(0, pert * 1.5, size=3)
        s0_list.append(s0)
        a0_list.append(acts_ref[f_start].copy())
        ref_list.append(slice_w)

    p_pi, pi_opt, loss = k_fn(p_pi, pi_opt, jnp.asarray(np.array(s0_list)), jnp.asarray(np.array(a0_list)), jnp.asarray(np.array(ref_list)))
    if (it + 1) % 50 == 0:
        rmse = float(eval_traj47(p_pi))
        if rmse < best_rmse:
            best_rmse = rmse
        el = time.perf_counter() - t0
        print(f"Iter {it+1:3d} (H={H_curr:2d}, {el:.1f}s) | Loss: {float(loss):.4f} | Traj 47 RMSE: {rmse:.2f} cm (Best: {best_rmse:.2f} cm)")
