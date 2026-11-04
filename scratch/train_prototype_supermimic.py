import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import h5py

from quadrotor.params import NX, NU, HOVER_RPM, MAX_THRUST, SIM_DT, M_NOMINAL
from quadrotor.world_model import (
    QuadrotorState, unpack_state_22, pack_state_22, invariant_features,
    compute_frame_local_targets, integrate_frame_local_step,
    WorldModel, THRUST_SCALE, ERR_FLOOR
)
from quadrotor.policy_critic import (
    ValueCritic, compute_proprio_obs, compute_analytic_reward, compute_soft_termination
)
from quadrotor.supermimic import (
    SuperMimicPolicy, SuperMimicDataset, compute_obs, normalize_obs, norm_action_to_rpm
)
from quadrotor.quaternion import state17_to_22
from quadrotor.buffer import SuperMimicReplayBuffer
from quadrotor.supermimic_env import (
    MotionReferenceLibrary, sim_step_multi_substeps, parameterized_dynamics_jax
)
from crazyflie_nn.dynamics.dynamics import rk4

print("===================================================================")
print("     SuperMimic Autonomous Drone Optimization Prototype Test       ")
print("===================================================================")

# 1. Physics-Grounded Observation Normalization
obs_mean_np = np.zeros(22, dtype=np.float32)
obs_mean_np[3:12] = np.eye(3).flatten() # R_wb at level hover

obs_std_np = np.ones(22, dtype=np.float32)
obs_std_np[0:3] = 0.50     # pos error scale: 50 cm
obs_std_np[3:12] = 0.50    # rotation matrix scale
obs_std_np[12:15] = 1.00   # body velocity scale: 1.0 m/s
obs_std_np[15:18] = 2.00   # angular velocity scale: 2.0 rad/s
obs_std_np[18:22] = 0.50   # action history scale

obs_mean_jax = jnp.asarray(obs_mean_np)
obs_std_jax = jnp.asarray(obs_std_np)

# 2. Ingest Corpus & Build Reference Library
corpus_path = "corpus/crazyflie_corpus_50trajs.h5"
motion_lib = MotionReferenceLibrary(corpus_path, bin_duration_s=1.0, dt=SIM_DT)
print(f"Loaded {motion_lib.num_trajs} trajectories, {motion_lib.num_bins} reset bins.")

# Split: 43 train, 7 validation
train_trajs = motion_lib.trajectories[:-7]
val_trajs = motion_lib.trajectories[-7:]
print(f"Train trajs: {len(train_trajs)}, Validation trajs: {len(val_trajs)}")

# Rebuild bins strictly over train trajectories to guarantee valid indexing
train_bins = [b for b in motion_lib.bins if b[0] < len(train_trajs)]
train_bin_returns = np.zeros(len(train_bins), dtype=np.float32)
print(f"Train reset bins: {len(train_bins)}")

def sample_train_reset_frames(n: int, rng: np.random.Generator):
    if len(train_bins) == 0:
        return np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    p_uniform = np.ones(len(train_bins)) / len(train_bins)
    max_ret = max(1e-4, np.max(train_bin_returns))
    scaled_neg_rets = -25.0 * (train_bin_returns / max_ret)
    shift = scaled_neg_rets - np.max(scaled_neg_rets)
    exp_rets = np.exp(shift)
    p_softmax = exp_rets / np.sum(exp_rets)
    p_combined = 0.50 * p_uniform + 0.40 * p_softmax + 0.10 * p_softmax
    p_combined = p_combined / np.sum(p_combined)
    chosen_bins = rng.choice(len(train_bins), size=n, p=p_combined)
    trajs = np.array([train_bins[b][0] for b in chosen_bins], dtype=np.int32)
    frames = np.array([rng.integers(train_bins[b][1], train_bins[b][2]) for b in chosen_bins], dtype=np.int32)
    return chosen_bins, trajs, frames

# 3. Seed Replay Buffer from Expert Corpus
buffer = SuperMimicReplayBuffer(capacity=100_000)
expert_obs_list = []
expert_acts_list = []

for t_idx, traj in enumerate(train_trajs):
    H = len(traj) - 1
    s_arr = traj[:H]
    s1_arr = traj[1:H + 1]
    a_arr = np.clip((s_arr[:, 18:22] - HOVER_RPM) / THRUST_SCALE, -1.0, 1.0)
    a_prev_arr = np.zeros_like(a_arr)
    a_prev_arr[1:] = a_arr[:-1]
    xi_arr = np.zeros((H, 10), dtype=np.float32)
    r_arr = np.zeros(H, dtype=np.float32)
    f_arr = np.arange(H, dtype=np.int32)
    traj_id_arr = np.full(H, t_idx, dtype=np.int32)
    ep_step_arr = np.arange(H, dtype=np.int32)
    done_arr = np.zeros(H, dtype=bool)
    done_arr[-1] = True
    buffer.add_batch(s_arr, s1_arr, a_arr, a_prev_arr, xi_arr, r_arr, f_arr, traj_id_arr, ep_step_arr, done_arr)
    
    # Collect for BC warmup
    for t in range(H):
        p_tgt = traj[t + 1, 0:3]
        o = compute_obs(s_arr[t], p_tgt, a_prev_arr[t])
        expert_obs_list.append(o)
        expert_acts_list.append(a_arr[t])

expert_obs = np.array(expert_obs_list, dtype=np.float32)
expert_acts = np.array(expert_acts_list, dtype=np.float32)
print(f"Seeded Replay Buffer with {buffer.size} transitions ({len(expert_obs)} BC pairs).")

# 4. Normalization Statistics for World Model
in_m = jnp.zeros(31)
in_s = jnp.array([
    1.0,                                   # z
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5,          # tilt_6d
    1.0, 1.0, 1.0,                         # vb
    2.0, 2.0, 2.0,                         # wb
    0.5, 0.5, 0.5, 0.5,                    # omega_norm
    0.5, 0.5, 0.5, 0.5,                    # action
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 # xi
], dtype=jnp.float32)

out_m = jnp.zeros(16)
out_s = jnp.array([
    0.02, 0.02, 0.02,                      # dp (2 cm per 20 ms)
    0.05, 0.05, 0.05,                      # dtheta (0.05 rad per 20 ms)
    0.05, 0.05, 0.05,                      # dvb (0.05 m/s per 20 ms)
    0.10, 0.10, 0.10,                      # dwb (0.1 rad/s per 20 ms)
    0.05, 0.05, 0.05, 0.05                 # domega (normalized)
], dtype=jnp.float32)

# 5. Initialize Networks
wm = WorldModel(hidden_dim=256)
policy = SuperMimicPolicy()
critic = ValueCritic(hidden_dims=(128, 64))

rng = jax.random.PRNGKey(42)
rng, k_wm, k_pi, k_c = jax.random.split(rng, 4)

wm_params = wm.init(k_wm)
tx_wm = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
wm_opt = tx_wm.init(wm_params)

pi_params = policy.init(k_pi, jnp.zeros((1, 22)))["params"]
tx_pi = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.clip(0.1),
    optax.adam(3e-4, b1=0.9, b2=0.95)
)
pi_opt = tx_pi.init(pi_params)

critic_params = critic.init(k_c, jnp.zeros((1, 30)))["params"]
tx_c = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
critic_opt = tx_c.init(critic_params)

# Phase 0: Fast Behavior Cloning Warmup (60 epochs)
print("\n--- Phase 0: Behavior Cloning Warmup (60 epochs) ---")
t0_bc = time.perf_counter()
tx_bc = optax.adam(1e-3)
bc_opt = tx_bc.init(pi_params)
obs_n_all = (expert_obs - obs_mean_np) / obs_std_np
N_bc = len(obs_n_all)
B_bc = 256
n_batches = N_bc // B_bc

@jax.jit
def bc_step(p, opt_s, o_b, a_b):
    def loss_fn(p_curr):
        return jnp.mean(jnp.square(policy.apply({"params": p_curr}, o_b) - a_b))
    l, g = jax.value_and_grad(loss_fn)(p)
    u, opt_s = tx_bc.update(g, opt_s, p)
    p = optax.apply_updates(p, u)
    return p, opt_s, l

for ep in range(60):
    rng, pk = jax.random.split(rng)
    perm = jax.random.permutation(pk, N_bc)
    ep_l = 0.0
    for b in range(n_batches):
        idx = perm[b * B_bc:(b + 1) * B_bc]
        pi_params, bc_opt, l_val = bc_step(pi_params, bc_opt, jnp.asarray(obs_n_all[idx]), jnp.asarray(expert_acts[idx]))
        ep_l += float(l_val)
    if (ep + 1) % 20 == 0:
        print(f"   [BC] Epoch {ep+1:2d}/60 | MSE Loss: {ep_l/n_batches:.6f} | Elapsed: {time.perf_counter()-t0_bc:.1f} s")

# Reset policy optimizer for BPTT
pi_opt = tx_pi.init(pi_params)

# 6. Evaluation Function (Closed-loop scan with crash handling)
@jax.jit
def eval_single_traj(p_policy, xs_ref):
    H = len(xs_ref) - 1
    def step(carry, t):
        s_curr, a_prev = carry
        p_tgt = xs_ref[t + 1, 0:3]
        o = compute_obs(s_curr, p_tgt, a_prev)
        o_n = normalize_obs(o, obs_mean_jax, obs_std_jax)
        a = policy.apply({"params": p_policy}, o_n)
        u = norm_action_to_rpm(a)
        def sub(i, st): return rk4(st, u, jax.random.PRNGKey(0), 0.005)
        s_next = jax.lax.fori_loop(0, 4, sub, s_curr)
        # Protect against NaN in integration
        s_safe = jnp.where(jnp.isnan(s_next), s_curr, s_next)
        err = jnp.linalg.norm(s_safe[0:3] - xs_ref[t + 1, 0:3])
        err_clamped = jnp.where(jnp.isnan(err) | (err > 3.0), 3.0, err)
        return (s_safe, a), err_clamped

    _, errs = jax.lax.scan(step, (xs_ref[0], jnp.zeros(4)), jnp.arange(H))
    return jnp.sqrt(jnp.mean(jnp.square(errs))) * 100.0, jnp.max(errs) * 100.0

def evaluate_all_val(p_policy):
    rmses = []
    maxes = []
    for traj in val_trajs:
        r, m = eval_single_traj(p_policy, jnp.asarray(traj))
        rmses.append(float(r))
        maxes.append(float(m))
    return float(np.mean(rmses)), float(np.mean(maxes))

bc_rmse, bc_max = evaluate_all_val(pi_params)
print(f"Validation RMSE after BC: {bc_rmse:.2f} cm (Mean Peak Error: {bc_max:.2f} cm)\n")

# 7. Scanned World Model & BPTT Kernels
k_steps = 8
horizon = 24

def scanned_wm_loss(p_wm, s_win, s1_win, a_win, xi_win, mask_win):
    st_init = unpack_state_22(s_win[0])
    def step_fn(st_curr, t):
        st_gt0 = unpack_state_22(s_win[t])
        st_gt1 = unpack_state_22(s1_win[t])
        target_delta = compute_frame_local_targets(st_gt0, st_gt1)
        xi_norm = xi_win[t] / 0.1
        raw_in = wm.inputs(st_curr, a_win[t], xi_norm)
        norm_in = (raw_in - in_m) / in_s
        pred_norm_out = wm.net.apply(p_wm, norm_in)
        pred_delta = pred_norm_out * out_s + out_m
        step_err = jnp.square((pred_delta - target_delta) / out_s)
        loss = jnp.mean(step_err) * mask_win[t]
        st_next = integrate_frame_local_step(st_curr, pred_delta)
        return st_next, loss
    _, step_losses = jax.lax.scan(step_fn, st_init, jnp.arange(k_steps))
    return jnp.sum(step_losses) / jnp.maximum(jnp.sum(mask_win), 1.0)

@jax.jit
def train_wm_batch(p_wm, opt_s, s_b, s1_b, a_b, xi_b, mask_b):
    def loss_fn(p):
        losses = jax.vmap(lambda s, s1, a, xi, m: scanned_wm_loss(p, s, s1, a, xi, m))(s_b, s1_b, a_b, xi_b, mask_b)
        return jnp.mean(losses)
    l, g = jax.value_and_grad(loss_fn)(p_wm)
    u, opt_s = tx_wm.update(g, opt_s, p_wm)
    p_wm = optax.apply_updates(p_wm, u)
    return p_wm, opt_s, l

def scanned_imag_rollout(p_policy, p_c, p_wm, s0_22, a_init, xi, ref_traj):
    st_init = unpack_state_22(s0_22)
    xi_norm = xi / 0.1
    
    def step_fn(carry, t):
        st_curr, a_prev, alive_prob = carry
        p_target = ref_traj[t + 1, 0:3]
        s_curr_22 = pack_state_22(st_curr)
        obs = compute_obs(s_curr_22, p_target, a_prev)
        obs_n = normalize_obs(obs, obs_mean_jax, obs_std_jax)
        
        a_curr = policy.apply({"params": p_policy}, obs_n)
        st_next = wm.step(p_wm, st_curr, a_curr, xi_norm, in_m, in_s, out_m, out_s)
        
        r = compute_analytic_reward(st_curr, ref_traj[t], a_curr, a_prev, dt=SIM_DT)
        d = compute_soft_termination(st_curr, ref_traj[t])
        next_alive = alive_prob * (1.0 - d)
        
        inv_st = invariant_features(st_curr)
        pos_err = st_curr.p - ref_traj[t, 0:3]
        c_in = jnp.concatenate([inv_st, xi_norm, pos_err], axis=-1)
        
        return (st_next, a_curr, next_alive), (r, d, alive_prob, c_in)

    (st_final, _, final_alive), (rews, terms, survs, c_ins) = jax.lax.scan(
        step_fn, (st_init, a_init, 1.0), jnp.arange(horizon)
    )
    
    terminal_inv = invariant_features(st_final)
    terminal_pos_err = st_final.p - ref_traj[horizon, 0:3]
    terminal_c_in = jnp.concatenate([terminal_inv, xi_norm, terminal_pos_err], axis=-1)
    v_terminal = critic.apply({"params": p_c}, terminal_c_in)
    
    discounts = 0.99 ** jnp.arange(horizon, dtype=jnp.float32)
    j_rewards = jnp.sum(discounts * survs * rews)
    j_bootstrap = (0.99 ** horizon) * final_alive * v_terminal
    policy_obj = j_rewards + j_bootstrap
    
    return policy_obj, (rews, terms, c_ins, terminal_c_in, v_terminal, final_alive)

@jax.jit
def train_policy_critic_batch(p_policy, pi_opt_s, p_c, c_opt_s, p_wm, s0_b, a_init_b, xi_b, ref_b):
    def pi_loss_fn(p):
        def single(s0, a_init, xi, ref):
            obj, aux = scanned_imag_rollout(p, p_c, p_wm, s0, a_init, xi, ref)
            return -obj, aux
        losses, aux_all = jax.vmap(single)(s0_b, a_init_b, xi_b, ref_b)
        return jnp.mean(losses), aux_all
        
    (pi_loss, aux_all), pi_grads = jax.value_and_grad(pi_loss_fn, has_aux=True)(p_policy)
    pi_updates, pi_opt_s = tx_pi.update(pi_grads, pi_opt_s, p_policy)
    p_policy = optax.apply_updates(p_policy, pi_updates)
    
    rews, terms, c_ins, term_c_in, v_term, final_alives = aux_all
    c_ins_sg = jax.lax.stop_gradient(c_ins)
    rews_sg = jax.lax.stop_gradient(rews)
    terms_sg = jax.lax.stop_gradient(terms)
    
    def compute_gae(r_traj, d_traj, c_traj, v_term_s):
        v_all = jax.vmap(lambda x: critic.apply({"params": p_c}, x))(c_traj)
        v_ext = jnp.concatenate([v_all, v_term_s[None]], axis=0)
        def gae_scan(adv, t):
            t_idx = horizon - 1 - t
            delta = r_traj[t_idx] + 0.99 * (1.0 - d_traj[t_idx]) * v_ext[t_idx + 1] - v_ext[t_idx]
            adv_next = delta + 0.99 * 0.95 * (1.0 - d_traj[t_idx]) * adv
            target = adv_next + v_ext[t_idx]
            return adv_next, target
        _, targets_rev = jax.lax.scan(gae_scan, 0.0, jnp.arange(horizon))
        return jnp.flip(targets_rev, axis=0)
        
    gae_targets = jax.vmap(compute_gae)(rews_sg, terms_sg, c_ins_sg, jax.lax.stop_gradient(v_term))
    c_ins_flat = c_ins_sg.reshape((-1, c_ins_sg.shape[-1]))
    targets_flat = gae_targets.reshape((-1,))
    
    def critic_loss_fn(p):
        v_preds = jax.vmap(lambda x: critic.apply({"params": p}, x))(c_ins_flat)
        return jnp.mean(jnp.square(v_preds - targets_flat))
        
    c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(p_c)
    c_updates, c_opt_s = tx_c.update(c_grads, c_opt_s, p_c)
    p_c = optax.apply_updates(p_c, c_updates)
    
    return p_policy, pi_opt_s, p_c, c_opt_s, pi_loss, c_loss, jnp.mean(final_alives)

vmap_sim_step = jax.jit(jax.vmap(lambda s, u, xi: sim_step_multi_substeps(s, u, xi, substeps=4, sub_dt=0.005)))

# 8. World Model Warm-up (100 iterations)
print("--- Phase 1: World Model Warm-up (100 iterations) ---")
rng_np = np.random.default_rng(42)
t0_wm = time.perf_counter()
for it in range(100):
    w = buffer.sample_windows(batch_size=64, k=k_steps, rng=rng_np)
    wm_params, wm_opt, l_wm = train_wm_batch(
        wm_params, wm_opt,
        jnp.asarray(w["s"]), jnp.asarray(w["s1"]), jnp.asarray(w["a"]),
        jnp.asarray(w["xi"]), jnp.asarray(w["mask"])
    )
    if (it + 1) % 50 == 0:
        print(f"   [WM Warmup] Iter {it+1:3d}/100 | WM Loss: {float(l_wm):.6f} | Elapsed: {time.perf_counter()-t0_wm:.1f} s")

# 9. Joint SuperMimic Training Loop
print("\n--- Phase 2: Joint SuperMimic Training (300 iterations) ---")
num_envs = 32
env_states = np.zeros((num_envs, 22), dtype=np.float32)
env_actions = np.zeros((num_envs, 4), dtype=np.float32)
env_xi = np.zeros((num_envs, 10), dtype=np.float32)
env_trajs_id = np.zeros(num_envs, dtype=np.int32)
env_frames = np.zeros(num_envs, dtype=np.int32)
env_steps = np.zeros(num_envs, dtype=np.int32)
env_bins = np.zeros(num_envs, dtype=np.int32)
env_disc_rets = np.zeros(num_envs, dtype=np.float32)

bins_init, trajs_init, frames_init = sample_train_reset_frames(num_envs, rng_np)
for i in range(num_envs):
    t_id, f_idx = trajs_init[i], frames_init[i]
    s_init = train_trajs[t_id][f_idx].copy()
    s_init[0:3] += rng_np.normal(0, 0.02, size=3)
    s_init[12:15] += rng_np.normal(0, 0.03, size=3)
    env_states[i] = s_init
    env_trajs_id[i] = t_id
    env_frames[i] = f_idx
    env_bins[i] = bins_init[i]
    env_xi[i] = rng_np.uniform(-0.05, 0.05, size=10)

t0_train = time.perf_counter()
best_rmse = bc_rmse

for it in range(300):
    # A. Simulator step in parallel
    s_batch_sim = env_states.copy()
    a_prev_batch = env_actions.copy()
    xi_batch = env_xi.copy()
    
    p_targets = np.array([train_trajs[env_trajs_id[i]][min(env_frames[i] + 1, len(train_trajs[env_trajs_id[i]]) - 1), 0:3] for i in range(num_envs)])
    obs_batch = np.array([compute_obs(env_states[i], p_targets[i], env_actions[i]) for i in range(num_envs)])
    obs_batch_n = (obs_batch - obs_mean_np) / obs_std_np
    
    actions_clean = np.array(policy.apply({"params": pi_params}, jnp.asarray(obs_batch_n)))
    actions_noisy = np.clip(actions_clean + rng_np.normal(0.0, 0.10, size=(num_envs, 4)), -1.0, 1.0)
    
    u_rpms = np.clip(HOVER_RPM + actions_noisy * THRUST_SCALE, 0.0, MAX_THRUST)
    next_states = np.array(vmap_sim_step(jnp.asarray(env_states), jnp.asarray(u_rpms), jnp.asarray(env_xi)))
    
    step_rewards = np.zeros(num_envs, dtype=np.float32)
    dones = np.zeros(num_envs, dtype=bool)
    
    for i in range(num_envs):
        t_id, f_idx = env_trajs_id[i], env_frames[i]
        ref_st = train_trajs[t_id][f_idx]
        st_obj = unpack_state_22(jnp.asarray(env_states[i]))
        r_val = float(compute_analytic_reward(st_obj, jnp.asarray(ref_st), jnp.asarray(actions_noisy[i]), jnp.asarray(env_actions[i])))
        step_rewards[i] = r_val
        env_disc_rets[i] += (0.99 ** env_steps[i]) * r_val
        
        s_next = next_states[i]
        pos_err = np.linalg.norm(s_next[0:3] - train_trajs[t_id][min(f_idx + 1, len(train_trajs[t_id]) - 1), 0:3])
        crashed = (s_next[2] < 0.05) or (pos_err > 0.40) or (s_next[11] < 0.5)
        timeout = (f_idx + 1 >= len(train_trajs[t_id]) - 1) or (env_steps[i] >= 150)
        dones[i] = crashed or timeout

    buffer.add_batch(
        s_batch_sim, next_states, actions_noisy, a_prev_batch, xi_batch,
        step_rewards, env_frames, env_trajs_id, env_steps, dones
    )
    
    # Reset handling with Return-Based Sampling
    done_ids = np.nonzero(dones)[0]
    if len(done_ids) > 0:
        for d_i in done_ids:
            b_idx = env_bins[d_i]
            train_bin_returns[b_idx] = 0.9 * train_bin_returns[b_idx] + 0.1 * env_disc_rets[d_i]
        new_bins, new_trajs, new_frames = sample_train_reset_frames(len(done_ids), rng_np)
        for j, d_idx in enumerate(done_ids):
            t_id, f_idx = new_trajs[j], new_frames[j]
            s_new = train_trajs[t_id][f_idx].copy()
            s_new[0:3] += rng_np.normal(0, 0.02, size=3)
            s_new[12:15] += rng_np.normal(0, 0.03, size=3)
            env_states[d_idx] = s_new
            env_actions[d_idx] = 0.0
            env_trajs_id[d_idx] = t_id
            env_frames[d_idx] = f_idx
            env_steps[d_idx] = 0
            env_bins[d_idx] = new_bins[j]
            env_disc_rets[d_idx] = 0.0
            env_xi[d_idx] = rng_np.uniform(-0.05, 0.05, size=10)

    not_done = np.nonzero(~dones)[0]
    env_states[not_done] = next_states[not_done]
    env_actions[not_done] = actions_noisy[not_done]
    env_frames[not_done] += 1
    env_steps[not_done] += 1
    
    # B. World Model Update
    w = buffer.sample_windows(batch_size=64, k=k_steps, rng=rng_np)
    wm_params, wm_opt, l_wm = train_wm_batch(
        wm_params, wm_opt,
        jnp.asarray(w["s"]), jnp.asarray(w["s1"]), jnp.asarray(w["a"]),
        jnp.asarray(w["xi"]), jnp.asarray(w["mask"])
    )
    
    # C. Policy & Critic BPTT Update in Imagination
    starts, _ = buffer.sample_starts(batch_size=48, hist_len=1, rng=rng_np)
    s0_pi = jnp.asarray(starts["s"])
    a_init_pi = jnp.asarray(starts["a_prev"])
    xi_pi = jnp.asarray(starts["xi"])
    f_starts = starts["f"]
    t_ids = starts["traj_id"]
    
    ref_windows = []
    for b in range(48):
        t_id_b = int(t_ids[b])
        f_b = int(f_starts[b])
        traj_len = len(train_trajs[t_id_b])
        end_idx = min(f_b + horizon + 1, traj_len)
        slice_w = train_trajs[t_id_b][f_b:end_idx]
        if len(slice_w) < horizon + 1:
            pad = np.tile(train_trajs[t_id_b][-1:], (horizon + 1 - len(slice_w), 1))
            slice_w = np.concatenate([slice_w, pad], axis=0)
        ref_windows.append(slice_w)
    ref_batch_pi = jnp.asarray(np.array(ref_windows, dtype=np.float32))
    
    pi_params, pi_opt, critic_params, critic_opt, l_pi, l_c, alive = train_policy_critic_batch(
        pi_params, pi_opt, critic_params, critic_opt, wm_params,
        s0_pi, a_init_pi, xi_pi, ref_batch_pi
    )
    
    if (it + 1) % 50 == 0:
        val_rmse, val_max = evaluate_all_val(pi_params)
        is_best = val_rmse < best_rmse
        if is_best:
            best_rmse = val_rmse
        elapsed = time.perf_counter() - t0_train
        print(f"Iter {it+1:3d} ({elapsed:.1f}s) | WM: {float(l_wm):.4f} | Pi: {float(l_pi):.4f} | Critic: {float(l_c):.4f} | Alive: {float(alive):.3f} | Val RMSE: {val_rmse:.2f} cm (Best: {best_rmse:.2f} cm) {'*' if is_best else ''}")

print("\n--- Prototype Run Completed Successfully! ---")
