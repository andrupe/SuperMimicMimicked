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

eval_trajs = []
eval_actions = []
with h5py.File("corpus/crazyflie_corpus_50trajs.h5", "r") as f:
    for idx in [43, 44, 45, 46, 48]:
        grp = f["trajectories"][f"traj_{idx:06d}"]
        t_type = grp.attrs["task_type"]
        s17 = grp["states"][:]
        u_bar = grp["controls"][:]
        s22 = np.array([state17_to_22(x) for x in s17])
        u_rpm = u_bar * MAX_THRUST
        raw_act = (u_rpm - HOVER_RPM) / THRUST_SCALE
        st_trim = 14 if t_type == "hover" else 2
        eval_trajs.append(s22[st_trim:])
        eval_actions.append(np.clip(raw_act[st_trim:], -1.0, 1.0))

policy = SuperMimicPolicy()
rng = jax.random.PRNGKey(42)
p_pi = policy.init(rng, jnp.zeros((1, 22)))["params"]

bc_obs, bc_acts = [], []
for i in range(len(eval_trajs)):
    traj = eval_trajs[i]
    acts = eval_actions[i]
    a_prev = np.zeros_like(acts)
    a_prev[1:] = acts[:-1]
    a_prev[0] = acts[0]
    for t in range(len(acts)):
        bc_obs.append(compute_obs(traj[t], traj[t + 1, 0:3], a_prev[t]))
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

for _ in range(120):
    p_pi, bc_opt, _ = bc_step(p_pi, bc_opt, jnp.asarray(bc_obs_n), jnp.asarray(bc_acts))

def make_bptt_kernel(H_len):
    def rollout_loss(p_curr, s0, a_init, ref_w, xi):
        def step(carry, t):
            s_curr, a_prev = carry
            o = compute_obs(s_curr, ref_w[t + 1, 0:3], a_prev)
            o_n = normalize_obs(o, obs_m_j, obs_s_j)
            a = policy.apply({"params": p_curr}, o_n)
            u = norm_action_to_rpm(a)
            s_next = sim_step_multi_substeps(s_curr, u, xi, substeps=4, sub_dt=0.005)

            pos_err = s_next[0:3] - ref_w[t + 1, 0:3]
            pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_err)) + 1e-4)
            pos_loss = 2.5 * pos_dist + 5.0 * jnp.sum(jnp.square(pos_err))

            R_curr = s_next[3:12].reshape((3, 3))
            R_ref = ref_w[t + 1, 3:12].reshape((3, 3))
            rot_loss = 0.30 * rot_angle_sq_jax(R_curr, R_ref)
            vel_loss = 0.05 * jnp.sum(jnp.square(s_next[12:15] - ref_w[t + 1, 12:15]))
            ang_loss = 0.02 * jnp.sum(jnp.square(s_next[15:18] - ref_w[t + 1, 15:18]))
            act_loss = 0.01 * jnp.sum(jnp.square(a - a_prev))
            return (s_next, a), (pos_loss + rot_loss + vel_loss + ang_loss + act_loss)

        _, step_losses = jax.lax.scan(step, (s0, a_init), jnp.arange(H_len))
        return jnp.sum((0.99 ** jnp.arange(H_len)) * step_losses)

    @jax.jit
    def train_step(p, opt_s, s0_b, a0_b, ref_b, xi_b):
        def loss_fn(p_curr):
            losses = jax.vmap(lambda s0, a0, ref, xi: rollout_loss(p_curr, s0, a0, ref, xi))(s0_b, a0_b, ref_b, xi_b)
            return jnp.mean(losses)
        l, g = jax.value_and_grad(loss_fn)(p)
        u, opt_s = tx_pi.update(g, opt_s, p)
        return optax.apply_updates(p, u), opt_s, l

    return train_step

kernel_h16 = make_bptt_kernel(16)
kernel_h24 = make_bptt_kernel(24)
kernel_h32 = make_bptt_kernel(32)

total_iters = 1000
lr_schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=total_iters, alpha=0.03)
tx_pi = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.95))
pi_opt = tx_pi.init(p_pi)

rng_np = np.random.default_rng(42)
t0 = time.perf_counter()
for it in range(total_iters):
    H_curr = 16 if it < 300 else (24 if it < 700 else 32)
    step_fn = kernel_h16 if it < 300 else (kernel_h24 if it < 700 else kernel_h32)
    s0_list, a0_list, ref_list, xi_list = [], [], [], []
    for _ in range(64):
        t_id = rng_np.integers(0, len(eval_trajs))
        traj = eval_trajs[t_id]
        acts = eval_actions[t_id]
        f_max = len(traj) - (H_curr + 1)
        f_start = rng_np.integers(0, max(1, f_max))
        slice_w = traj[f_start: f_start + H_curr + 1]
        s0 = slice_w[0].copy()
        pert = 0.012 if it < 400 else 0.005
        s0[0:3] += rng_np.normal(0, pert, size=3)
        s0[12:15] += rng_np.normal(0, pert * 1.5, size=3)
        # Match evaluation domain randomization magnitude (+- 10%)
        xi = rng_np.uniform(-0.10, 0.10, size=10).astype(np.float32)
        s0_list.append(s0)
        a0_list.append(acts[f_start].copy())
        ref_list.append(slice_w)
        xi_list.append(xi)

    p_pi, pi_opt, loss = step_fn(p_pi, pi_opt, jnp.asarray(np.array(s0_list)), jnp.asarray(np.array(a0_list)), jnp.asarray(np.array(ref_list)), jnp.asarray(np.array(xi_list)))

# Evaluate across all 3 suites
@jax.jit
def eval_suite_traj(p, xs_ref, a0, xi, noise_pos, noise_gyro):
    traj_len = len(xs_ref) - 1
    def step(carry, t):
        s_curr, a_prev = carry
        o = compute_obs(s_curr, xs_ref[t + 1, 0:3], a_prev)
        o = o.at[0:3].add(noise_pos[t]).at[15:18].add(noise_gyro[t])
        o_n = normalize_obs(o, obs_m_j, obs_s_j)
        a = policy.apply({"params": p}, o_n)
        u = norm_action_to_rpm(a)
        s_next = sim_step_multi_substeps(s_curr, u, xi, substeps=4, sub_dt=0.005)
        err = jnp.linalg.norm(s_next[0:3] - xs_ref[t + 1, 0:3])
        return (s_next, a), err
    _, errs = jax.lax.scan(step, (xs_ref[0], a0), jnp.arange(traj_len))
    return jnp.sqrt(jnp.mean(jnp.square(errs))) * 100.0, jnp.max(errs) * 100.0

print("\n=== Evaluation across 3 Suites ===")
# Suite A: Nominal
nom_rmses = []
for i in range(len(eval_trajs)):
    T = len(eval_trajs[i]) - 1
    r, m = eval_suite_traj(p_pi, jnp.asarray(eval_trajs[i]), jnp.asarray(eval_actions[i][0]), jnp.zeros(10), jnp.zeros((T, 3)), jnp.zeros((T, 3)))
    nom_rmses.append(float(r))
print(f"Suite A (Nominal) Val RMSE: {np.mean(nom_rmses):.2f} cm | Per traj: {[round(r, 2) for r in nom_rmses]}")

# Suite B: Sensor Noise
noise_rmses = []
for i in range(len(eval_trajs)):
    T = len(eval_trajs[i]) - 1
    n_p = jnp.asarray(rng_np.normal(0, 0.005, size=(T, 3)).astype(np.float32))
    n_g = jnp.asarray(rng_np.normal(0, 0.005, size=(T, 3)).astype(np.float32))
    r, m = eval_suite_traj(p_pi, jnp.asarray(eval_trajs[i]), jnp.asarray(eval_actions[i][0]), jnp.zeros(10), n_p, n_g)
    noise_rmses.append(float(r))
print(f"Suite B (Sensor Noise) Val RMSE: {np.mean(noise_rmses):.2f} cm | Per traj: {[round(r, 2) for r in noise_rmses]}")

# Suite C: Domain Randomization
dr_rmses = []
for i in range(len(eval_trajs)):
    T = len(eval_trajs[i]) - 1
    xi = jnp.asarray(rng_np.uniform(-0.10, 0.10, size=10).astype(np.float32))
    r, m = eval_suite_traj(p_pi, jnp.asarray(eval_trajs[i]), jnp.asarray(eval_actions[i][0]), xi, jnp.zeros((T, 3)), jnp.zeros((T, 3)))
    dr_rmses.append(float(r))
print(f"Suite C (Domain Randomization) Val RMSE: {np.mean(dr_rmses):.2f} cm | Per traj: {[round(r, 2) for r in dr_rmses]}")
