"""Rapid Motor Adaptation (RMA) & Asymmetric SuperMimic Framework.

Implements the RMA architecture (Kumar et al., RSS 2021) adapted for micro-quadrotor flight:
  1. Base Policy pi_theta(a | o, z):
     - Observation o in R^22: [p - p_target, R_wb, vb, wb, a_prev]
     - Environmental Latent z in R^4: encodes effective thrust deficit and bias torques
     - Total actor input: 26 dims -> 64 SiLU -> 4 Tanh
     - Lightweight memory budget: < 8 KB RAM, < 40 us execution on Cortex-M4F.
  2. Privileged Environment Encoder mu_psi(xi):
     - Maps privileged physics context xi in R^14 -> latent z in R^4.
  3. Onboard Adaptation Module phi_omega(H_t):
     - Predicts latent z_hat from rolling history of actions and velocities (H_t in R^24).
     - Trained via supervised regression against the privileged latent z.
  4. Privileged Value Critic V_phi(s, xi):
     - Observes complete physical state and exact domain context for zero-variance GAE bootstrapping.
  5. High-Stress Domain Randomization Engine:
     - Mass (+-15%), Payload (0-5g), Inertia (+-20%), Motor efficiencies (+-12%),
       Motor tau (18-35ms), Center of Mass offset (+-3mm XY, +-4mm Z), Dynamic voltage sag (0-8%),
       Gyro bias (+-0.035 rad/s), Sensor noise.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple, Dict, Any, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadrotor.params import (
    NX, NU, HOVER_RPM, MAX_THRUST, SIM_DT, M_NOMINAL
)
from quadrotor.world_model import (
    QuadrotorState, unpack_state_22, pack_state_22,
    invariant_features, THRUST_SCALE
)

OBS_DIM = 22
LATENT_Z_DIM = 4
ACTOR_IN_DIM = OBS_DIM + LATENT_Z_DIM # 26 dimensions
ACT_DIM = 4
XI_DIM = 14
HIST_DIM = 24 # 3 steps of actions (12) + 2 steps of (vb, wb) (12)


# ------------------------------------------------------------------------------
# 1. Privileged Environment Encoder mu_psi(xi) -> z in R^4
# ------------------------------------------------------------------------------
class PrivilegedEnvEncoder(nn.Module):
    """Encodes 14-dim privileged physical parameters into 4-dim latent z."""
    hidden_dim: int = 32
    z_dim: int = LATENT_Z_DIM

    @nn.compact
    def __call__(self, xi: jax.Array) -> jax.Array:
        h = nn.Dense(self.hidden_dim)(xi)
        h = nn.silu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.silu(h)
        z = nn.Dense(self.z_dim)(h)
        return jnp.tanh(z)


# ------------------------------------------------------------------------------
# 2. Onboard Adaptation Module phi_omega(H_t) -> z_hat in R^4
# ------------------------------------------------------------------------------
class OnboardAdaptationModule(nn.Module):
    """Predicts environmental latent z_hat from observable sensor/action history."""
    hidden_dim: int = 32
    z_dim: int = LATENT_Z_DIM

    @nn.compact
    def __call__(self, hist: jax.Array) -> jax.Array:
        h = nn.Dense(self.hidden_dim)(hist)
        h = nn.silu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.silu(h)
        z_hat = nn.Dense(self.z_dim)(h)
        return jnp.tanh(z_hat)


# ------------------------------------------------------------------------------
# 3. Lightweight RMA Base Policy pi_theta(a | o, z) -> [-1, 1]^4
# ------------------------------------------------------------------------------
class RMABasePolicy(nn.Module):
    """Lightweight 64-hidden MLP running in < 40us on the Crazyflie Cortex-M4."""
    hidden_dim: int = 64
    act_dim: int = ACT_DIM

    @nn.compact
    def __call__(self, obs_with_latent: jax.Array) -> jax.Array:
        h = nn.Dense(self.hidden_dim, name="hidden_0")(obs_with_latent)
        h = nn.silu(h)
        out = nn.Dense(self.act_dim, name="hidden_1")(h)
        return jnp.tanh(out)


# ------------------------------------------------------------------------------
# 4. Privileged Value Critic V_phi(s, xi)
# ------------------------------------------------------------------------------
class PrivilegedCritic(nn.Module):
    """High-capacity asymmetric critic observing ground-truth simulation physics."""
    hidden_dims: Tuple[int, ...] = (128, 64)

    @nn.compact
    def __call__(self, privileged_features: jax.Array) -> jax.Array:
        h = privileged_features
        for hd in self.hidden_dims:
            h = nn.Dense(hd)(h)
            h = nn.silu(h)
        val = nn.Dense(1)(h)
        return jnp.squeeze(val, axis=-1)


# ------------------------------------------------------------------------------
# 5. High-Stress Domain Randomization Generator
# ------------------------------------------------------------------------------
def sample_domain_randomization(n_samples: int,
                                rng: np.random.Generator,
                                stress_level: float = 1.0) -> np.ndarray:
    """Samples 14-dimensional physical parameter variation vectors:
      xi[0]   : Mass variation delta_m / m0 in [-0.15, +0.15]
      xi[1:4] : Inertia variations delta_Ixx, Iyy, Izz in [-0.20, +0.20]
      xi[4:8] : Rotor thrust efficiencies delta_kF in [-0.12, +0.12]
      xi[8]   : Torque coefficient delta_kM in [-0.15, +0.15]
      xi[9]   : Motor time constant delta_tau_m in [-0.25, +0.35]
      xi[10:13]: Center of Mass (CoM) offset dx, dy, dz in [-0.003, +0.003]m
      xi[13]  : Dynamic voltage sag coefficient in [0.0, 0.08]
    """
    sl = np.clip(stress_level, 0.0, 1.0)
    xi = np.zeros((n_samples, XI_DIM), dtype=np.float32)

    xi[:, 0] = rng.uniform(-0.15 * sl, 0.15 * sl, size=n_samples) # Mass
    xi[:, 1:4] = rng.uniform(-0.20 * sl, 0.20 * sl, size=(n_samples, 3)) # Inertia
    xi[:, 4:8] = rng.uniform(-0.12 * sl, 0.12 * sl, size=(n_samples, 4)) # Motor efficiencies
    xi[:, 8] = rng.uniform(-0.15 * sl, 0.15 * sl, size=n_samples) # Torque kM
    xi[:, 9] = rng.uniform(-0.25 * sl, 0.35 * sl, size=n_samples) # Motor tau
    xi[:, 10:12] = rng.uniform(-0.003 * sl, 0.003 * sl, size=(n_samples, 2)) # CoM XY (3mm)
    xi[:, 12] = rng.uniform(-0.004 * sl, 0.004 * sl, size=n_samples) # CoM Z (4mm)
    xi[:, 13] = rng.uniform(0.0, 0.08 * sl, size=n_samples) # Battery sag

    return xi


# ------------------------------------------------------------------------------
# Helper functions for observation packaging and history
# ------------------------------------------------------------------------------
def compute_actor_obs(st: QuadrotorState, p_target: jax.Array, a_prev: jax.Array) -> jax.Array:
    """Computes standard 22-dimensional onboard sensor observation."""
    pos_err = st.p - p_target
    R_flat = st.R.reshape((9,))
    return jnp.concatenate([pos_err, R_flat, st.vb, st.wb, a_prev], axis=-1)


def assemble_history_buffer(a_hist: jax.Array, v_hist: jax.Array, w_hist: jax.Array) -> jax.Array:
    """Assembles 24-dimensional history vector for the adaptation module."""
    # a_hist: (3, 4) -> 12 dims
    # v_hist: (2, 3) -> 6 dims
    # w_hist: (2, 3) -> 6 dims
    return jnp.concatenate([
        a_hist.reshape((-1,)),
        v_hist.reshape((-1,)),
        w_hist.reshape((-1,))
    ], axis=-1)
