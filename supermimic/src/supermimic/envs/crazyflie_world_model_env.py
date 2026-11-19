"""Differentiable world model environment for Crazyflie quadcopter reference tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.utils.lab_api.math import quat_apply
from tensordict import TensorDict

from supermimic.tasks.crazyflie_mdp import tracking_command_body

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import StateRepresentation


class CrazyflieWorldModelEnv:
  """Differentiable world model env for quadcopter reference tracking.

  Unrolls learned dynamics autoregressively for BPTT. Each imagined rollout
  starts from a contiguous, done-free window sampled from the replay buffer,
  and the *recorded* world-frame reference sequence of that window
  ([ref_pos(3), ref_vel_w(3), ref_yaw_rate(1)] per step) drives the
  tracking command and rewards, so the reference evolves exactly like it
  did in simulation over the whole BPTT horizon.
  """

  GOAL_DIM = 7

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    horizon: int = 32,
    history_length: int = 1,
    termination_softness: float = 0.05,
    roll_pitch_spin_weight: float = -0.005,
    tilt_margin_weight: float = 0.0,
    sim_context: Any | None = None,
    device: str = "cpu",
  ) -> None:
    self.device = device
    self.dynamics = dynamics
    self.representation = representation
    self.num_envs = batch_size
    self.replay_buffer = replay_buffer
    self.horizon = horizon
    self.history_length = history_length
    self.termination_softness = termination_softness
    self.roll_pitch_spin_weight = roll_pitch_spin_weight
    self.tilt_margin_weight = tilt_margin_weight

    self._num_actions = env.num_actions
    self.dt = env.unwrapped.step_dt
    self._decimation = int(getattr(env.unwrapped.cfg, "decimation", 4))
    self._dt_phys = float(env.unwrapped.cfg.sim.mujoco.timestep)
    self._sim_context = sim_context
    self._motor_cols = (
      sim_context.find("crazyflie_motor_dr") if sim_context is not None else None
    )

    # Reward weights (positive terms set the per-step ceiling)
    self.w_pos, self.w_pos_coarse, self.w_vel, self.w_yaw = 2.0, 1.0, 1.0, 0.4
    self.w_upright, self.w_alt = 0.3, 0.5
    self.max_step_reward = self.dt * (
      self.w_pos + self.w_pos_coarse + self.w_vel + self.w_yaw + self.w_upright + self.w_alt
    )

    self.rep_state: torch.Tensor | None = None
    self.aux_state: torch.Tensor | None = None
    self.goal_seq: torch.Tensor | None = None
    self.last_actions: torch.Tensor | None = None
    self.sim_ctx: torch.Tensor | None = None
    self._actor_hist: torch.Tensor | None = None
    self.t = 0

  @property
  def num_actions(self) -> int:
    return self._num_actions

  @property
  def window_length(self) -> int:
    # frame 0 supplies the previous action; frames 1..H+1 are the rollout states/references
    return self.horizon + 2

  def sample_initial_states(self) -> TensorDict:
    """Sample done-free windows (B, horizon+2) from the replay buffer.

    Falls back to single-frame starts with a constant-velocity reference
    extrapolation (same shapes) if no done-free window of that length exists yet.
    """
    L = self.window_length
    try:
      return self.replay_buffer.sample(L, self.num_envs)
    except RuntimeError:
      td = self.replay_buffer.sample(1, self.num_envs)
      td = td.expand(self.num_envs, L).clone()
      cmd = td["command"]
      steps = torch.arange(L, device=cmd.device, dtype=torch.float32).view(1, L, 1)
      cmd[..., 0:3] = cmd[:, :1, 0:3] + cmd[:, :1, 3:6] * steps * self.dt
      td["command"] = cmd
      return td

  def reset(self, batch: TensorDict | None = None) -> TensorDict:
    if batch is None:
      batch = self.sample_initial_states()
    full_states = batch["state"][:, 1].to(self.device)
    self.aux_state = self.representation.extract_aux_state(full_states).detach()
    self.rep_state = self.representation.represent(full_states).detach()
    self.last_actions = batch["action"][:, 0].to(self.device).detach()
    self.goal_seq = batch["command"][:, 1:].to(self.device).detach()  # (B, H+1, 7)
    self.sim_ctx = batch["sim_ctx"][:, 1].to(self.device).detach()
    self.t = 0
    if self.history_length > 1:
      obs0 = self._raw_actor_obs()
      self._actor_hist = obs0.unsqueeze(1).repeat(1, self.history_length, 1)
    else:
      self._actor_hist = None
    return self.get_observations()

  def _command_b(self) -> torch.Tensor:
    pos = self.aux_state[..., 0:3]
    quat = self.aux_state[..., 3:7]
    return tracking_command_body(pos, quat, self.goal_seq[:, self.t])

  def _raw_actor_obs(self) -> torch.Tensor:
    pos = self.aux_state[..., 0:3]
    g_b = self.rep_state[..., 1:4]
    v_b = self.rep_state[..., 4:7]
    w_b = self.rep_state[..., 7:10]
    cmd_b = self._command_b()
    return torch.cat([pos, g_b, v_b, w_b, self.last_actions, cmd_b], dim=-1)

  def get_observations(self) -> TensorDict:
    """Actor obs: root_pos(3), g_b(3), v_b(3), w_b(3), last_action(4), tracking_cmd_b(7) = 23 (x history_length)."""
    cmd_b = self._command_b()
    if self.history_length > 1:
      actor_obs = self._actor_hist.reshape(self.num_envs, -1)
    else:
      actor_obs = self._raw_actor_obs()
    critic_obs = torch.cat([self.rep_state, cmd_b, self.sim_ctx], dim=-1)
    return TensorDict({"actor": actor_obs, "critic": critic_obs}, batch_size=[self.num_envs], device=self.device)

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    next_aux_state, next_rep_state = self.dynamics(
      self.rep_state, actions, self.aux_state, self.sim_ctx
    )
    self.t = min(self.t + 1, self.goal_seq.shape[1] - 1)
    goal = self.goal_seq[:, self.t]

    pos = next_aux_state[..., 0:3]
    quat = next_aux_state[..., 3:7]
    v_b = next_rep_state[..., 4:7]
    w_z = next_rep_state[..., 9]
    g_z = next_rep_state[..., 3]
    z = next_rep_state[..., 0]

    # Position tracking: sharp + coarse Gaussian kernels and a linear term for gradient far away
    d2 = torch.sum((pos - goal[..., 0:3]) ** 2, dim=-1)
    dist = torch.sqrt(d2 + 1e-6)
    r_pos = self.w_pos * torch.exp(-d2 / 0.04) + self.w_pos_coarse * torch.exp(-d2 / 0.5) - 0.5 * dist

    # World-frame velocity tracking
    v_w = quat_apply(quat, v_b)
    dv2 = torch.sum((v_w - goal[..., 3:6]) ** 2, dim=-1)
    r_vel = self.w_vel * torch.exp(-dv2 / 0.25)

    # Yaw-rate tracking
    yaw_err = w_z - goal[..., 6]
    r_yaw = self.w_yaw * torch.exp(-(yaw_err ** 2) / 0.25) - 0.10 * torch.sqrt(yaw_err ** 2 + 1e-4)

    # Attitude / altitude safety
    r_upright = self.w_upright * torch.exp(-((g_z + 1.0) ** 2) / 0.04)
    r_alt = self.w_alt * torch.exp(-torch.clamp(0.3 - z, min=0.0) ** 2 / 0.01)

    # Regularization (action 0.0 is hover thrust)
    r_action_rate = -0.05 * torch.sum((actions - self.last_actions) ** 2, dim=-1)
    r_effort = -0.01 * torch.sum(actions ** 2, dim=-1)
    r_spin = self.roll_pitch_spin_weight * torch.sum(next_rep_state[..., 7:9] ** 2, dim=-1) - 0.025 * (yaw_err ** 2)
    r_tilt_margin = -self.tilt_margin_weight * torch.relu(g_z - (-0.342))

    reward = (r_pos + r_vel + r_yaw + r_upright + r_alt + r_action_rate + r_effort + r_spin + r_tilt_margin) * self.dt

    # Differentiable soft termination
    p_crash_low = torch.sigmoid((0.01 - z) / self.termination_softness)
    p_crash_high = torch.sigmoid((z - 4.5) / 0.1)
    p_tilt = torch.sigmoid((g_z - (-0.1736)) / self.termination_softness)
    term_prob = torch.clamp(p_crash_low + p_crash_high + p_tilt, 0.0, 1.0)

    if self._motor_cols is not None and self._sim_context is not None and self.sim_ctx is not None:
      m0 = self._motor_cols.start
      target = self.dynamics.proc(actions)
      raw_motor = self._sim_context.denormalize(self.sim_ctx, self._motor_cols)
      tau_up = raw_motor[..., 0:1]
      tau_down = raw_motor[..., 1:2]
      thrust_scale = raw_motor[..., 2:3]
      sag_rate = raw_motor[..., 3:4]
      thrust_state = raw_motor[..., 12:16]
      for _ in range(self._decimation):
        tau = torch.where(target >= thrust_state, tau_up, tau_down)
        alpha = self._dt_phys / (tau + self._dt_phys)
        thrust_state = thrust_state + alpha * (target - thrust_state)
        thrust_scale = torch.clamp(thrust_scale - sag_rate * self._dt_phys, min=0.70)
      norm_scale = (
        thrust_scale - self._sim_context.mean[m0 + 2 : m0 + 3]
      ) / self._sim_context.scale[m0 + 2 : m0 + 3]
      norm_state = (
        thrust_state - self._sim_context.mean[m0 + 12 : m0 + 16]
      ) / self._sim_context.scale[m0 + 12 : m0 + 16]
      updated_motor = torch.cat(
        [
          self.sim_ctx[..., m0 : m0 + 2],
          norm_scale,
          self.sim_ctx[..., m0 + 3 : m0 + 12],
          norm_state,
        ],
        dim=-1,
      )
      parts = []
      if m0 > 0:
        parts.append(self.sim_ctx[..., :m0])
      parts.append(updated_motor)
      if self._motor_cols.stop < self.sim_ctx.shape[-1]:
        parts.append(self.sim_ctx[..., self._motor_cols.stop :])
      self.sim_ctx = torch.cat(parts, dim=-1) if len(parts) > 1 else updated_motor

    self.rep_state = next_rep_state
    self.aux_state = next_aux_state
    self.last_actions = actions
    if self.history_length > 1:
      self._actor_hist = torch.cat([self._actor_hist[:, 1:], self._raw_actor_obs().unsqueeze(1)], dim=1)
    return self.get_observations(), reward, term_prob, {}
