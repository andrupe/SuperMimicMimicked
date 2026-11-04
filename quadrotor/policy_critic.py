"""SuperMimic Policy (Motion Encoder + Decoder) and Value Critic Networks.

Implements the SuperMimic policy architecture (Liao et al., CoRL 2026, Section 3.3):
  - Motion Encoder: encodes future reference trajectory horizon g in R^(6*K) -> latent z in R^16 (tanh).
  - Policy Decoder: maps (z, o_hist) -> action a in [-1, 1]^4 (tanh).
  - Value Critic V_phi: estimates state-value with GAE(lambda=0.95) bootstrapping.
  - Analytic tracking reward and differentiable soft termination for imagination BPTT.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from quadrotor.params import (
    NX, NU, HOVER_RPM, MAX_THRUST, SIM_DT
)
from quadrotor.rotations_jax import (
    so3_exp_jax, so3_log_jax, yaw_mat_jax, mat_to_6d_jax, rot_angle_sq_jax
)
from quadrotor.world_model import (
    QuadrotorState, invariant_features, INV_STATE_DIM, ACT_DIM, XI_DIM
)

GOAL_K = 10                       # Number of future reference frames
GOAL_STEP_DIM = 6                 # Relative (dx, dy, dz, dvx, dvy, dvz) in body frame
GOAL_DIM = GOAL_K * GOAL_STEP_DIM # 60 dimensions
OBS_SINGLE_DIM = 22               # pos_err(3), Rwb(9), vb(3), wb(3), a_prev(4)
LATENT_Z_DIM = 16


def compute_goal_features(st: QuadrotorState, ref_window: jax.Array) -> jax.Array:
    """Compute K-step future reference trajectory relative to current drone state in body frame.
    
    ref_window: (K, 22) array of future reference states.
    Returns: (GOAL_DIM,) vector invariant to global position and yaw.
    """
    R_t = jnp.swapaxes(st.R, -1, -2) # Body frame projector
    ref_p = ref_window[..., 0:3]     # (K, 3)
    ref_v = ref_window[..., 12:15]   # (K, 3)

    # Relative position and velocity in current body frame
    p_diff = ref_p - st.p[..., None, :] # (..., K, 3)
    v_diff = ref_v - st.vb[..., None, :]

    rel_p = jnp.squeeze(R_t[..., None, :, :] @ p_diff[..., :, :, None], axis=-1)
    rel_v = jnp.squeeze(R_t[..., None, :, :] @ v_diff[..., :, :, None], axis=-1)

    rel_combined = jnp.concatenate([rel_p, rel_v], axis=-1) # (..., K, 6)
    return rel_combined.reshape(st.p.shape[:-1] + (GOAL_DIM,))


def compute_proprio_obs(st: QuadrotorState, p_target: jax.Array, a_prev: jax.Array) -> jax.Array:
    """Compute 22-dimensional observation matching firmware / my_controller.c."""
    pos_err = st.p - p_target
    R_flat = st.R.reshape(st.p.shape[:-1] + (9,))
    vb = st.vb
    wb = st.wb
    return jnp.concatenate([pos_err, R_flat, vb, wb, a_prev], axis=-1)


class MotionEncoder(nn.Module):
    """Encodes K-step future reference motion into compact latent vector z in R^16."""
    hidden_dims: Tuple[int, ...] = (128, 64)
    z_dim: int = LATENT_Z_DIM

    @nn.compact
    def __call__(self, goal: jax.Array) -> jax.Array:
        h = goal
        for hd in self.hidden_dims:
            h = nn.Dense(hd)(h)
            h = nn.silu(h)
        z = nn.Dense(self.z_dim)(h)
        return jnp.tanh(z)


class PolicyDecoder(nn.Module):
    """Maps latent z and observation history to normalized actions in [-1, 1]^4."""
    hidden_dims: Tuple[int, ...] = (128, 64)
    act_dim: int = ACT_DIM

    @nn.compact
    def __call__(self, z: jax.Array, obs: jax.Array) -> jax.Array:
        x = jnp.concatenate([z, obs], axis=-1)
        h = x
        for hd in self.hidden_dims:
            h = nn.Dense(hd)(h)
            h = nn.silu(h)
        out = nn.Dense(self.act_dim, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.zeros_init())(h)
        return jnp.tanh(out)


class SuperMimicPolicyNet(nn.Module):
    """Full SuperMimic Policy combining Motion Encoder and Decoder."""
    encoder_hidden: Tuple[int, ...] = (128, 64)
    decoder_hidden: Tuple[int, ...] = (128, 64)
    z_dim: int = LATENT_Z_DIM

    def setup(self):
        self.encoder = MotionEncoder(hidden_dims=self.encoder_hidden, z_dim=self.z_dim)
        self.decoder = PolicyDecoder(hidden_dims=self.decoder_hidden, act_dim=ACT_DIM)

    def __call__(self, goal: jax.Array, obs: jax.Array) -> jax.Array:
        z = self.encoder(goal)
        return self.decoder(z, obs)


class ValueCritic(nn.Module):
    """Estimates state-value V(S, xi, goal) for GAE(lambda) bootstrapping."""
    hidden_dims: Tuple[int, ...] = (256, 128)

    @nn.compact
    def __call__(self, critic_in: jax.Array) -> jax.Array:
        h = critic_in
        for hd in self.hidden_dims:
            h = nn.Dense(hd)(h)
            h = nn.silu(h)
        v = nn.Dense(1)(h)
        return jnp.squeeze(v, axis=-1)


# ------------------------------------------------------------------------------
# Analytic Reward and Differentiable Soft Termination Functions
# ------------------------------------------------------------------------------

def compute_analytic_reward(st: QuadrotorState,
                            ref_state_22: jax.Array,
                            action: jax.Array,
                            a_prev: jax.Array,
                            dt: float = 0.02) -> jax.Array:
    """Compute SuperMimic analytic tracking reward matching Appendix A."""
    p_ref = ref_state_22[..., 0:3]
    R_ref = ref_state_22[..., 3:12].reshape(p_ref.shape[:-1] + (3, 3))
    v_ref = ref_state_22[..., 12:15]
    w_ref = ref_state_22[..., 15:18]

    # Geodesic and Euclidean tracking errors
    e_pos = jnp.sum(jnp.square(st.p - p_ref), axis=-1)
    e_rot = rot_angle_sq_jax(st.R, R_ref)
    e_vel = jnp.sum(jnp.square(st.vb - v_ref), axis=-1)
    e_ang = jnp.sum(jnp.square(st.wb - w_ref), axis=-1)
    e_act = jnp.sum(jnp.square(action - a_prev), axis=-1)

    # Kernel widths
    r_pos = jnp.exp(-e_pos / (0.15 ** 2))
    r_rot = jnp.exp(-e_rot / (0.35 ** 2))
    r_vel = jnp.exp(-e_vel / (0.50 ** 2))
    r_ang = jnp.exp(-e_ang / (1.50 ** 2))

    r_track = 1.0 * r_pos + 0.5 * r_rot + 0.2 * r_vel + 0.1 * r_ang
    r_reg = -0.05 * e_act

    return (r_track + r_reg) * dt


def compute_soft_termination(st: QuadrotorState, ref_state_22: jax.Array) -> jax.Array:
    """Compute differentiable termination probability in [0, 1] for BPTT imagination."""
    p_ref = ref_state_22[..., 0:3]
    e_pos = jnp.linalg.norm(st.p - p_ref, axis=-1)
    
    # Incline/tilt error: R[2, 2] is z-component of body vertical axis
    e_tilt = 1.0 - st.R[..., 2, 2] # 0 when upright, > 0.5 when tilted > 60 deg
    
    # Soft sigmoid thresholds
    d_pos = jax.nn.sigmoid((e_pos - 0.40) / 0.04)
    d_tilt = jax.nn.sigmoid((e_tilt - 0.50) / 0.05)
    
    # Combined soft termination
    alive = (1.0 - d_pos) * (1.0 - d_tilt)
    return 1.0 - alive
