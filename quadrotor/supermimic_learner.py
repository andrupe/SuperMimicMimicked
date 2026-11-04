"""SuperMimic JAX Learner: Vectorized World Model, Policy BPTT, and Critic Updates.

Implements the core algorithmic updates of SuperMimic (Liao et al., CoRL 2026):
  1. Autoregressive k-step World Model update (Eq. 1).
  2. Differentiable BPTT Policy update in imagination with soft survival weighting (Eq. 2).
  3. GAE(lambda=0.95) Critic bootstrapping update (Eq. 3).
  4. Vectorized simulator stepping for online data collection.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, NamedTuple
import jax
import jax.numpy as jnp
import optax

from quadrotor.params import (
    NX, NU, HOVER_RPM, MAX_THRUST, SIM_DT
)
from quadrotor.rotations_jax import (
    so3_exp_jax, so3_log_jax, yaw_mat_jax, mat_to_6d_jax, rot_angle_sq_jax
)
from quadrotor.world_model import (
    QuadrotorState, unpack_state_22, pack_state_22, invariant_features,
    compute_frame_local_targets, integrate_frame_local_step,
    WorldModel, WorldModelMLP, WM_IN_DIM, WM_OUT_DIM, ERR_FLOOR
)
from quadrotor.policy_critic import (
    SuperMimicPolicyNet, ValueCritic,
    compute_goal_features, compute_proprio_obs,
    compute_analytic_reward, compute_soft_termination,
    GOAL_DIM, GOAL_K, OBS_SINGLE_DIM, LATENT_Z_DIM
)
from quadrotor.supermimic_env import sim_step_multi_substeps


class SuperMimicLearner:
    """Encapsulates the JIT-compiled optimization kernels for SuperMimic."""

    def __init__(self,
                 wm_hidden: int = 256,
                 policy_encoder_hidden: Tuple[int, ...] = (128, 64),
                 policy_decoder_hidden: Tuple[int, ...] = (128, 64),
                 critic_hidden: Tuple[int, ...] = (256, 128),
                 lr_wm: float = 3e-4,
                 lr_pi: float = 2e-4,
                 lr_critic: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 horizon: int = 24,
                 k_window: int = 8):
        self.gamma = gamma
        self.lam = lam
        self.horizon = horizon
        self.k_window = k_window

        self.wm = WorldModel(hidden_dim=wm_hidden)
        self.policy_net = SuperMimicPolicyNet(encoder_hidden=policy_encoder_hidden, decoder_hidden=policy_decoder_hidden)
        self.critic_net = ValueCritic(hidden_dims=critic_hidden)

        # Optimizers matching paper: Adam, grad clipping 1.0
        self.tx_wm = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr_wm)
        )
        self.tx_pi = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.clip(0.1), # element-wise clipping for policy
            optax.adam(learning_rate=lr_pi, b1=0.9, b2=0.95)
        )
        self.tx_critic = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr_critic)
        )

    # --------------------------------------------------------------------------
    # 1. Autoregressive World Model Loss and Update Step (Eq. 1)
    # --------------------------------------------------------------------------

    def compute_wm_window_loss(self,
                              wm_params: Any,
                              s_win: jax.Array,      # (k, 22)
                              s1_win: jax.Array,     # (k, 22)
                              a_win: jax.Array,      # (k, 4)
                              xi_win: jax.Array,     # (k, 10)
                              mask: jax.Array,       # (k,)
                              in_mean: jax.Array,
                              in_std: jax.Array,
                              out_mean: jax.Array,
                              out_std: jax.Array) -> jax.Array:
        """Unroll World Model autoregressively across k steps and compute normalized MSE."""
        st_curr = unpack_state_22(s_win[0])
        total_loss = 0.0

        for j in range(self.k_window):
            st_gt0 = unpack_state_22(s_win[j])
            st_gt1 = unpack_state_22(s1_win[j])
            target_delta = compute_frame_local_targets(st_gt0, st_gt1)

            # Frame input
            xi_norm = xi_win[j] / 0.1 # Normalized context
            raw_in = self.wm.inputs(st_curr, a_win[j], xi_norm)
            norm_in = (raw_in - in_mean) / jnp.maximum(in_std, 1e-4)

            pred_norm_out = self.wm.net.apply(wm_params, norm_in)
            pred_delta = pred_norm_out * jnp.maximum(out_std, ERR_FLOOR) + out_mean

            # Normalized target error
            step_err = jnp.square((pred_delta - target_delta) / jnp.maximum(out_std, ERR_FLOOR))
            step_loss = jnp.mean(step_err) * mask[j]
            total_loss = total_loss + step_loss

            # Advance autoregressively
            st_curr = integrate_frame_local_step(st_curr, pred_delta)

        return total_loss / jnp.maximum(jnp.sum(mask), 1.0)

    # --------------------------------------------------------------------------
    # 2. Differentiable BPTT Policy & Critic Update in Imagination (Eq. 2 & 3)
    # --------------------------------------------------------------------------

    def single_imagination_rollout(self,
                                  pi_params: Any,
                                  critic_params: Any,
                                  wm_params: Any,
                                  s0_22: jax.Array,
                                  a_init: jax.Array,
                                  xi: jax.Array,
                                  ref_traj_window: jax.Array, # (horizon + GOAL_K, 22)
                                  in_mean: jax.Array,
                                  in_std: jax.Array,
                                  out_mean: jax.Array,
                                  out_std: jax.Array) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Roll out imagined trajectory through differentiable World Model and compute returns."""
        st_curr = unpack_state_22(s0_22)
        a_prev = a_init
        xi_norm = xi / 0.1

        rewards = []
        dones = []
        critic_inputs = []
        alive_prob = 1.0
        survival_weights = []

        for t in range(self.horizon):
            # Compute goal representation K steps ahead
            future_slice = ref_traj_window[t + 1: t + 1 + GOAL_K]
            goal = compute_goal_features(st_curr, future_slice)

            # Observation matching firmware
            p_target = ref_traj_window[t + 1, 0:3]
            obs = compute_proprio_obs(st_curr, p_target, a_prev)

            # Policy forward
            a_curr = self.policy_net.apply(pi_params, goal, obs)

            # Differentiable step through learned World Model
            st_next = self.wm.step(wm_params, st_curr, a_curr, xi_norm, in_mean, in_std, out_mean, out_std)

            # Reward and soft termination
            r = compute_analytic_reward(st_curr, ref_traj_window[t], a_curr, a_prev, dt=SIM_DT)
            d = compute_soft_termination(st_curr, ref_traj_window[t])

            # Critic features: invariant state + goal + xi + tracking error
            inv_st = invariant_features(st_curr)
            pos_err = st_curr.p - ref_traj_window[t, 0:3]
            c_in = jnp.concatenate([inv_st, goal, xi_norm, pos_err], axis=-1)

            rewards.append(r)
            dones.append(d)
            survival_weights.append(alive_prob)
            critic_inputs.append(c_in)

            alive_prob = alive_prob * (1.0 - d)
            st_curr = st_next
            a_prev = a_curr

        # Terminal state critic value bootstrap
        terminal_goal = compute_goal_features(st_curr, ref_traj_window[self.horizon: self.horizon + GOAL_K])
        terminal_inv = invariant_features(st_curr)
        terminal_pos_err = st_curr.p - ref_traj_window[self.horizon, 0:3]
        terminal_c_in = jnp.concatenate([terminal_inv, terminal_goal, xi_norm, terminal_pos_err], axis=-1)
        v_terminal = self.critic_net.apply(critic_params, terminal_c_in)

        rews = jnp.array(rewards) # (H,)
        terms = jnp.array(dones)  # (H,)
        survs = jnp.array(survival_weights) # (H,)
        c_ins = jnp.stack(critic_inputs, axis=0) # (H, dim)

        # Discounted policy return J(theta)
        discounts = self.gamma ** jnp.arange(self.horizon, dtype=jnp.float32)
        j_rewards = jnp.sum(discounts * survs * rews)
        j_bootstrap = (self.gamma ** self.horizon) * alive_prob * v_terminal
        policy_objective = j_rewards + j_bootstrap

        # Compute GAE targets for critic
        # Evaluate values of all states in imagination
        v_all = jax.vmap(lambda x: self.critic_net.apply(critic_params, x))(c_ins)
        v_all_with_term = jnp.concatenate([v_all, v_terminal[None]], axis=0)

        # Reverse GAE scan
        def gae_scan(carry, t):
            r_future = carry
            t_idx = self.horizon - 1 - t
            delta = rews[t_idx] + self.gamma * (1.0 - terms[t_idx]) * v_all_with_term[t_idx + 1] - v_all_with_term[t_idx]
            adv = delta + self.gamma * self.lam * (1.0 - terms[t_idx]) * r_future
            target = adv + v_all_with_term[t_idx]
            return adv, target

        _, gae_targets_rev = jax.lax.scan(gae_scan, 0.0, jnp.arange(self.horizon))
        gae_targets = jnp.flip(gae_targets_rev, axis=0)

        aux = {
            "mean_reward": jnp.mean(rews),
            "final_alive": alive_prob,
            "critic_inputs": c_ins,
            "gae_targets": gae_targets,
            "v_pred": v_all,
        }
        return policy_objective, aux

    # --------------------------------------------------------------------------
    # 3. Batched JIT Training Kernels
    # --------------------------------------------------------------------------

    def wm_train_batch(self,
                       wm_params: Any,
                       wm_opt_state: Any,
                       s_batch: jax.Array,      # (B, k, 22)
                       s1_batch: jax.Array,     # (B, k, 22)
                       a_batch: jax.Array,      # (B, k, 4)
                       xi_batch: jax.Array,     # (B, k, 10)
                       mask_batch: jax.Array,   # (B, k)
                       in_mean: jax.Array,
                       in_std: jax.Array,
                       out_mean: jax.Array,
                       out_std: jax.Array) -> Tuple[Any, Any, jax.Array]:
        """Compute World Model gradient over batch of windows and update parameters."""
        def loss_fn(p):
            losses = jax.vmap(
                lambda s, s1, a, xi, m: self.compute_wm_window_loss(
                    p, s, s1, a, xi, m, in_mean, in_std, out_mean, out_std
                )
            )(s_batch, s1_batch, a_batch, xi_batch, mask_batch)
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(wm_params)
        updates, wm_opt_state = self.tx_wm.update(grads, wm_opt_state, wm_params)
        wm_params = optax.apply_updates(wm_params, updates)
        return wm_params, wm_opt_state, loss

    def policy_critic_train_batch(self,
                                 pi_params: Any,
                                 pi_opt_state: Any,
                                 critic_params: Any,
                                 critic_opt_state: Any,
                                 wm_params: Any,
                                 s0_batch: jax.Array,          # (B, 22)
                                 a_init_batch: jax.Array,      # (B, 4)
                                 xi_batch: jax.Array,          # (B, 10)
                                 ref_batch: jax.Array,         # (B, horizon + GOAL_K, 22)
                                 in_mean: jax.Array,
                                 in_std: jax.Array,
                                 out_mean: jax.Array,
                                 out_std: jax.Array) -> Tuple[Any, Any, Any, Any, Dict[str, jax.Array]]:
        """Compute Differentiable BPTT Policy gradient through World Model and update Policy & Critic."""
        # 1. Policy update through frozen World Model
        def policy_loss_fn(p_pi):
            def single(s0, a_init, xi, ref):
                obj, aux = self.single_imagination_rollout(
                    p_pi, critic_params, wm_params, s0, a_init, xi, ref,
                    in_mean, in_std, out_mean, out_std
                )
                return -obj, aux # Minimize negative return

            losses, aux_all = jax.vmap(single)(s0_batch, a_init_batch, xi_batch, ref_batch)
            return jnp.mean(losses), aux_all

        (pi_loss, aux_data), pi_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(pi_params)
        pi_updates, pi_opt_state = self.tx_pi.update(pi_grads, pi_opt_state, pi_params)
        pi_params = optax.apply_updates(pi_params, pi_updates)

        # 2. Critic update on GAE targets from imagination
        c_ins_flat = aux_data["critic_inputs"].reshape((-1, aux_data["critic_inputs"].shape[-1]))
        gae_targets_flat = aux_data["gae_targets"].reshape((-1,))

        def critic_loss_fn(p_critic):
            v_preds = jax.vmap(lambda x: self.critic_net.apply(p_critic, x))(c_ins_flat)
            return jnp.mean(jnp.square(v_preds - gae_targets_flat))

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_updates, critic_opt_state = self.tx_critic.update(critic_grads, critic_opt_state, critic_params)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        metrics = {
            "pi_loss": pi_loss,
            "critic_loss": critic_loss,
            "mean_reward": jnp.mean(aux_data["mean_reward"]),
            "final_alive": jnp.mean(aux_data["final_alive"]),
        }
        return pi_params, pi_opt_state, critic_params, critic_opt_state, metrics

