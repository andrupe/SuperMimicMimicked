"""Differentiable world model environment for Crazyflie quadcopter velocity tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import StateRepresentation


class CrazyflieWorldModelEnv:
  """Differentiable world model env for quadcopter velocity control.

  Unrolls learned dynamics autoregressively with analytical differentiable
  tracking rewards and soft termination probabilities for BPTT policy learning.
  """

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    history_length: int = 1,
    termination_softness: float = 0.05,
    device: str = "cpu",
  ) -> None:
    self.device = device
    self.dynamics = dynamics
    self.representation = representation
    self.num_envs = batch_size
    self.replay_buffer = replay_buffer
    self.history_length = history_length
    self.termination_softness = termination_softness

    self._num_actions = env.num_actions
    self.dt = env.unwrapped.step_dt

    # Maximum theoretical per-step reward ceiling (all tracking terms saturated)
    self.max_step_reward = self.dt * (2.5 + 1.2 + 0.6)

    # State variables
    self.rep_state: torch.Tensor | None = None
    self.aux_state: torch.Tensor | None = None
    self.command: torch.Tensor | None = None
    self.last_actions: torch.Tensor | None = None
    self.sim_ctx: torch.Tensor | None = None

  @property
  def num_actions(self) -> int:
    return self._num_actions

  def sample_initial_states(self) -> TensorDict:
    """Sample state snapshots from replay buffer for rollout initialization."""
    return self.replay_buffer.sample(1, self.num_envs).squeeze(1)

  def reset(self, batch: TensorDict | None = None) -> TensorDict:
    """Reset to initial batch of states and commands."""
    if batch is None:
      batch = self.sample_initial_states()

    full_states = batch["state"].to(self.device)
    self.aux_state = self.representation.extract_aux_state(full_states).detach()
    self.rep_state = self.representation.represent(full_states).detach()
    self.last_actions = batch["action"].to(self.device).detach()
    self.command = batch["command"].to(self.device).detach()
    self.sim_ctx = batch["sim_ctx"].to(self.device).detach()

    return self.get_observations()

  def get_observations(self) -> TensorDict:
    """Assemble observations for actor and critic."""
    # rep_state: [z(1), g_b(3), v_b(3), w_b(3)] -> shape (B, 10)
    g_b = self.rep_state[..., 1:4]
    v_b = self.rep_state[..., 4:7]
    w_b = self.rep_state[..., 7:10]

    # Actor observation: projected_gravity(3), lin_vel(3), ang_vel(3), last_action(4), command(4) = 17
    actor_obs = torch.cat(
      [g_b, v_b, w_b, self.last_actions, self.command], dim=-1
    )

    # Critic privileged observation: rep_state(10), command(4), sim_ctx(3) = 17
    critic_obs = torch.cat(
      [self.rep_state, self.command, self.sim_ctx], dim=-1
    )

    obs_dict = {
      "actor": actor_obs,
      "critic": critic_obs,
    }
    return TensorDict(obs_dict, batch_size=[self.num_envs], device=self.device)

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Execute one differentiable transition step in the learned World Model."""
    # 1. Forward dynamics: (rep_state, action, aux_state, sim_ctx) -> (next_aux, next_rep)
    next_aux_state, next_rep_state = self.dynamics(
      self.rep_state, actions, self.aux_state, self.sim_ctx
    )

    # 2. Differentiable tracking rewards
    # Velocity tracking
    v_b = next_rep_state[..., 4:7]
    v_cmd = self.command[..., 0:3]
    lin_vel_err = torch.sum((v_b - v_cmd) ** 2, dim=-1)
    r_lin = 2.5 * torch.exp(-lin_vel_err / 0.25)

    # Yaw rate tracking
    w_z = next_rep_state[..., 9]
    yaw_cmd = self.command[..., 3]
    yaw_err = (w_z - yaw_cmd) ** 2
    r_yaw = 1.2 * torch.exp(-yaw_err / 0.25)

    # Upright reward (g_b = [gx, gy, gz]; upright means gz = -1.0)
    g_z = next_rep_state[..., 3]
    upright_err = (g_z - (-1.0)) ** 2
    r_upright = 0.6 * torch.exp(-upright_err / 0.04)

    # Altitude safety reward (discourage dropping below 0.4 m)
    z = next_rep_state[..., 0]
    r_alt = 0.8 * torch.exp(-torch.clamp(0.4 - z, min=0.0) ** 2 / 0.04)

    # Smoothness & effort penalties (actions=0.0 is hover)
    r_action_rate = -0.05 * torch.sum((actions - self.last_actions) ** 2, dim=-1)
    r_hover = -0.01 * torch.sum(actions ** 2, dim=-1)

    reward = (r_lin + r_yaw + r_upright + r_alt + r_action_rate + r_hover) * self.dt

    # 3. Differentiable soft termination probability
    p_crash_low = torch.sigmoid((0.05 - z) / self.termination_softness)
    p_crash_high = torch.sigmoid((z - 4.5) / 0.1)
    # Tilt > 80 deg corresponds to gz > cos(80 deg) * (-1.0) = -0.1736
    p_tilt = torch.sigmoid((g_z - (-0.1736)) / self.termination_softness)
    term_prob = torch.clamp(p_crash_low + p_crash_high + p_tilt, 0.0, 1.0)

    # Update state buffers
    self.rep_state = next_rep_state
    self.aux_state = next_aux_state
    self.last_actions = actions

    return self.get_observations(), reward, term_prob, {}

