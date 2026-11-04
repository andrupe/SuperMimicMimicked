"""Differentiable Neural World Model for the Bitcraze Crazyflie Quadrotor.

Implements the SuperMimic World Model (Liao et al., CoRL 2026, Section 3.2):
  - Invariant state features: z (altitude), body-tilt in yaw frame (6D), v_b, w_b, rotor RPMs.
  - Predicts frame-local Lie-algebra increments: (dp_yaw, dtheta_so3, dv_b, dw_b, domega_rot).
  - Conditioned on simulation context xi (mass, inertia, motor constants, motor lag).
  - Autoregressive multi-step rollout for training (k=8 step windows, Eq. 1).
  - Exact frame-local state integration s_{t+1} = step(s_t, a_t, xi).
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
    so3_exp_jax, so3_log_jax, yaw_mat_jax, mat_to_6d_jax, skew_jax
)

THRUST_SCALE = 0.4 * HOVER_RPM
INV_STATE_DIM = 1 + 6 + 3 + 3 + 4  # 17: z(1), tilt_6d(6), vb(3), wb(3), omega(4)
ACT_DIM = 4
XI_DIM = 10                        # dM(1), dI(3), dKF(4), dKM(1), dtau(1)
WM_IN_DIM = INV_STATE_DIM + ACT_DIM + XI_DIM  # 31
WM_OUT_DIM = 3 + 3 + 3 + 3 + 4     # 16: dp(3), dtheta(3), dvb(3), dwb(3), domega(4)

# Error floors for normalized loss (prevents small residuals from dominating loss)
ERR_FLOOR = jnp.array([
    0.002, 0.002, 0.002,   # dp (2 mm)
    0.005, 0.005, 0.005,   # dtheta (5 mrad)
    0.010, 0.010, 0.010,   # dvb (1 cm/s)
    0.020, 0.020, 0.020,   # dwb (0.02 rad/s)
    0.020, 0.020, 0.020, 0.020 # domega_norm
], dtype=jnp.float32)


class QuadrotorState(NamedTuple):
    """Structured quadrotor state representation."""
    p: jax.Array          # (..., 3) position
    R: jax.Array          # (..., 3, 3) attitude rotation matrix
    vb: jax.Array         # (..., 3) body linear velocity
    wb: jax.Array         # (..., 3) body angular velocity
    omega: jax.Array      # (..., 4) motor rotor speeds (RPM)


def unpack_state_22(s: jax.Array) -> QuadrotorState:
    """Unpack flat 22-dimensional state vector into QuadrotorState."""
    p = s[..., 0:3]
    R = s[..., 3:12].reshape(s.shape[:-1] + (3, 3))
    vb = s[..., 12:15]
    wb = s[..., 15:18]
    omega = s[..., 18:22]
    return QuadrotorState(p=p, R=R, vb=vb, wb=wb, omega=omega)


def pack_state_22(st: QuadrotorState) -> jax.Array:
    """Pack QuadrotorState into flat 22-dimensional state vector."""
    return jnp.concatenate([
        st.p,
        st.R.reshape(st.p.shape[:-1] + (9,)),
        st.vb,
        st.wb,
        st.omega
    ], axis=-1)


def invariant_features(st: QuadrotorState) -> jax.Array:
    """Compute heading- and translation-invariant state features (dimension 17)."""
    # 1. Altitude
    z = st.p[..., 2:3]
    # 2. Tilt in heading-free yaw frame: H_yaw^T @ R
    H = yaw_mat_jax(st.R)
    R_tilt = jnp.swapaxes(H, -1, -2) @ st.R
    tilt_6d = mat_to_6d_jax(R_tilt)
    # 3. Body velocities
    vb = st.vb
    wb = st.wb
    # 4. Normalized rotor speeds
    omega_norm = (st.omega - HOVER_RPM) / THRUST_SCALE
    return jnp.concatenate([z, tilt_6d, vb, wb, omega_norm], axis=-1)


def compute_frame_local_targets(st0: QuadrotorState, st1: QuadrotorState) -> jax.Array:
    """Compute frame-local increment targets from st0 to st1 (dimension 16)."""
    H0 = yaw_mat_jax(st0.R)
    H0_t = jnp.swapaxes(H0, -1, -2)
    dp_yaw = jnp.squeeze(H0_t @ (st1.p - st0.p)[..., None], axis=-1)

    R0_t = jnp.swapaxes(st0.R, -1, -2)
    R_rel = R0_t @ st1.R
    dtheta = so3_log_jax(R_rel)

    dvb = st1.vb - st0.vb
    dwb = st1.wb - st0.wb
    domega = (st1.omega - st0.omega) / THRUST_SCALE

    return jnp.concatenate([dp_yaw, dtheta, dvb, dwb, domega], axis=-1)


def integrate_frame_local_step(st: QuadrotorState, delta: jax.Array) -> QuadrotorState:
    """Integrate predicted frame-local increment into next QuadrotorState."""
    dp_yaw = delta[..., 0:3]
    dtheta = delta[..., 3:6]
    dvb = delta[..., 6:9]
    dwb = delta[..., 9:12]
    domega = delta[..., 12:16]

    H = yaw_mat_jax(st.R)
    dp_world = jnp.squeeze(H @ dp_yaw[..., None], axis=-1)
    p1 = st.p + dp_world

    R_rel = so3_exp_jax(dtheta)
    R1 = st.R @ R_rel

    vb1 = st.vb + dvb
    wb1 = st.wb + dwb
    omega1 = jnp.clip(st.omega + domega * THRUST_SCALE, 0.0, MAX_THRUST)

    return QuadrotorState(p=p1, R=R1, vb=vb1, wb=wb1, omega=omega1)


class WorldModelMLP(nn.Module):
    """World Model neural network predicting frame-local increments."""
    hidden_dim: int = 256
    out_dim: int = WM_OUT_DIM

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h = nn.Dense(self.hidden_dim, name="fc1")(x)
        h = nn.LayerNorm()(h)
        h = nn.silu(h)

        h = nn.Dense(self.hidden_dim, name="fc2")(h)
        h = nn.LayerNorm()(h)
        h = nn.silu(h)

        h = nn.Dense(self.hidden_dim, name="fc3")(h)
        h = nn.LayerNorm()(h)
        h = nn.silu(h)

        out = nn.Dense(self.out_dim, kernel_init=nn.initializers.zeros_init(), bias_init=nn.initializers.zeros_init(), name="fc_out")(h)
        return out


class WorldModel:
    """Manages World Model network, normalization statistics, and autoregressive unrolls."""

    def __init__(self, hidden_dim: int = 256):
        self.net = WorldModelMLP(hidden_dim=hidden_dim)
        self.in_mean = jnp.zeros(WM_IN_DIM, dtype=jnp.float32)
        self.in_std = jnp.ones(WM_IN_DIM, dtype=jnp.float32)
        self.out_mean = jnp.zeros(WM_OUT_DIM, dtype=jnp.float32)
        self.out_std = jnp.ones(WM_OUT_DIM, dtype=jnp.float32)

    def init(self, rng_key: jax.Array) -> Any:
        dummy_in = jnp.zeros((1, WM_IN_DIM), dtype=jnp.float32)
        return self.net.init(rng_key, dummy_in)

    def inputs(self, st: QuadrotorState, action: jax.Array, xi_norm: jax.Array) -> jax.Array:
        """Construct full World Model input vector."""
        inv = invariant_features(st)
        return jnp.concatenate([inv, action, xi_norm], axis=-1)

    def step(self, params: Any, st: QuadrotorState, action: jax.Array, xi_norm: jax.Array,
             in_mean: jax.Array, in_std: jax.Array, out_mean: jax.Array, out_std: jax.Array) -> QuadrotorState:
        """Single differentiable forward simulation step through the World Model."""
        raw_in = self.inputs(st, action, xi_norm)
        norm_in = (raw_in - in_mean) / jnp.maximum(in_std, 1e-4)
        pred_norm_out = self.net.apply(params, norm_in)
        pred_delta = pred_norm_out * jnp.maximum(out_std, ERR_FLOOR) + out_mean
        return integrate_frame_local_step(st, pred_delta)
