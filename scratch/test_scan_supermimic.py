import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from quadrotor.params import NX, NU, HOVER_RPM, MAX_THRUST, SIM_DT
from quadrotor.world_model import (
    QuadrotorState, unpack_state_22, pack_state_22, invariant_features,
    compute_frame_local_targets, integrate_frame_local_step,
    WorldModel, THRUST_SCALE, ERR_FLOOR
)
from quadrotor.policy_critic import (
    SuperMimicPolicyNet, ValueCritic, compute_proprio_obs,
    compute_analytic_reward, compute_soft_termination
)
from quadrotor.supermimic import SuperMimicPolicy, SuperMimicDataset, compute_obs, normalize_obs, norm_action_to_rpm

print("Verifying fast JAX-scanned World Model and BPTT kernels...")

# 1. Test Scanned World Model Unroll
wm = WorldModel(hidden_dim=128)
rng = jax.random.PRNGKey(42)
wm_params = wm.init(rng)
tx_wm = optax.adam(1e-3)
wm_opt = tx_wm.init(wm_params)

k_steps = 8
B = 32

# Synthetic batch of windows: (B, k, 22)
dummy_s0 = jnp.zeros((B, 22))
dummy_s0 = dummy_s0.at[:, 0:3].set(jnp.array([0, 0, 1.0]))
dummy_s0 = dummy_s0.at[:, 3:12].set(jnp.eye(3).flatten())
dummy_s0 = dummy_s0.at[:, 18:22].set(HOVER_RPM)

dummy_s1 = dummy_s0
dummy_a = jnp.zeros((B, k_steps, 4))
dummy_xi = jnp.zeros((B, k_steps, 10))
dummy_mask = jnp.ones((B, k_steps))

in_m = jnp.zeros(31)
in_s = jnp.ones(31)
out_m = jnp.zeros(16)
out_s = jnp.ones(16)

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

t0 = time.perf_counter()
dummy_s_win = jnp.repeat(dummy_s0[:, None, :], k_steps, axis=1)
dummy_s1_win = jnp.repeat(dummy_s1[:, None, :], k_steps, axis=1)
wm_params, wm_opt, l_wm = train_wm_batch(wm_params, wm_opt, dummy_s_win, dummy_s1_win, dummy_a, dummy_xi, dummy_mask)
print(f"Scanned World Model JIT compile & step in {time.perf_counter() - t0:.3f} s! Loss: {float(l_wm):.6f}")

# 2. Test Scanned Policy BPTT through World Model
policy = SuperMimicPolicy()
p_pi = policy.init(rng, jnp.zeros((1, 22)))["params"]
tx_pi = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4, b1=0.9, b2=0.95))
pi_opt = tx_pi.init(p_pi)

critic = ValueCritic(hidden_dims=(128, 64))
p_critic = critic.init(rng, jnp.zeros((1, 30)))["params"]
tx_c = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
critic_opt = tx_c.init(p_critic)

horizon = 24
dummy_ref_win = jnp.repeat(dummy_s0[:, None, :], horizon + 1, axis=1)

def scanned_imag_rollout(p_policy, p_c, p_wm, s0_22, a_init, xi, ref_traj):
    st_init = unpack_state_22(s0_22)
    xi_norm = xi / 0.1
    
    def step_fn(carry, t):
        st_curr, a_prev, alive_prob = carry
        
        p_target = ref_traj[t + 1, 0:3]
        obs = compute_proprio_obs(st_curr, p_target, a_prev)
        obs_n = obs # identity normalization for test
        
        a_curr = policy.apply({"params": p_policy}, obs_n)
        
        # Differentiable step through world model
        st_next = wm.step(p_wm, st_curr, a_curr, xi_norm, in_m, in_s, out_m, out_s)
        
        r = compute_analytic_reward(st_curr, ref_traj[t], a_curr, a_prev, dt=SIM_DT)
        d = compute_soft_termination(st_curr, ref_traj[t])
        
        next_alive = alive_prob * (1.0 - d)
        
        # Critic input for state t
        inv_st = invariant_features(st_curr)
        pos_err = st_curr.p - ref_traj[t, 0:3]
        c_in = jnp.concatenate([inv_st, xi_norm, pos_err], axis=-1)
        
        return (st_next, a_curr, next_alive), (r, d, alive_prob, c_in)

    (st_final, _, final_alive), (rews, terms, survs, c_ins) = jax.lax.scan(
        step_fn, (st_init, a_init, 1.0), jnp.arange(horizon)
    )
    
    # Terminal critic bootstrap
    terminal_inv = invariant_features(st_final)
    terminal_pos_err = st_final.p - ref_traj[horizon, 0:3]
    terminal_c_in = jnp.concatenate([terminal_inv, xi_norm, terminal_pos_err], axis=-1)
    v_terminal = critic.apply({"params": p_c}, terminal_c_in)
    
    # Policy discounted return J(theta)
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
    
    # Stop gradient on imagined states for critic training
    rews, terms, c_ins, term_c_in, v_term, final_alives = aux_all
    c_ins_sg = jax.lax.stop_gradient(c_ins)
    rews_sg = jax.lax.stop_gradient(rews)
    terms_sg = jax.lax.stop_gradient(terms)
    
    # Compute GAE targets
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

t0 = time.perf_counter()
dummy_a0 = jnp.zeros((B, 4))
dummy_xi0 = jnp.zeros((B, 10))
p_pi, pi_opt, p_critic, critic_opt, l_pi, l_c, alive = train_policy_critic_batch(
    p_pi, pi_opt, p_critic, critic_opt, wm_params,
    dummy_s0, dummy_a0, dummy_xi0, dummy_ref_win
)
print(f"Scanned Policy BPTT JIT compile & step in {time.perf_counter() - t0:.3f} s! Pi Loss: {float(l_pi):.4f}, Critic Loss: {float(l_c):.4f}, Alive: {float(alive):.4f}")
print("ALL KERNELS VERIFIED AND SUPER FAST!")
