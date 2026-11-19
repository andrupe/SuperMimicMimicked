"""Crazyflie Quadcopter Velocity Runner with SuperMimic BPTT & World Model training."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from mjlab.utils.lab_api.math import quat_apply
from supermimic.envs.crazyflie_world_model_env import CrazyflieWorldModelEnv
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import build_mlp
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.supermimic_runner import BaseRunnerCfg, build_world_model


class QuadcopterPolicy(nn.Module):
  """Actor policy mapping 23-dim actor observations (incl. xyz root_pos & 7-dim tracking cmd) to 4-dim rotor actions."""

  def __init__(self, obs_dim: int = 23, action_dim: int = 4, hidden_dims: tuple[int, ...] = (256, 256, 128)):
    super().__init__()
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.hidden_dims = tuple(hidden_dims)
    self.mlp = build_mlp(obs_dim, hidden_dims, action_dim)

  def forward(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
    if isinstance(obs, TensorDict):
      x = obs["actor"]
    else:
      x = obs
    return torch.tanh(self.mlp(x))


@dataclass
class CrazyflieRunnerCfg(BaseRunnerCfg):
  """Configuration for Crazyflie SuperMimic training."""

  max_iterations: int = 50000
  num_collect_steps: int = 8
  buffer_size: int = 2000
  replay_ratio: int = 4
  dynamics_hidden_dims: tuple[int, ...] = (512, 512, 512)
  actor_hidden_dims: tuple[int, ...] = (64, 64)
  critic_hidden_dims: tuple[int, ...] = (256, 256, 128)
  policy: BpttPolicyLearnerCfg = field(default_factory=BpttPolicyLearnerCfg)
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)
  actor_history_length: int = 1
  roll_pitch_spin_weight: float = -0.005
  tilt_margin_weight: float = 0.0
  checkpoint_dir: str = "logs/crazyflie_supermimic"


class CrazyflieRunner:
  """Training runner for Crazyflie 2.1 quadcopter on SuperMimic."""

  def __init__(
    self,
    env: Any,
    cfg: CrazyflieRunnerCfg,
    task_id: str = "SuperMimic-Crazyflie-Velocity",
    device: str = "cpu",
  ) -> None:
    self.env = env
    self.cfg = cfg
    self.task_id = task_id
    self.device = device

    # 1. World model and representation
    self.wm = build_world_model(cfg, env, device=device)
    self.representation = self.wm.representation
    self.dynamics = self.wm.dynamics
    self.sim_context = self.wm.sim_context

    # 2. Replay buffer
    self.replay_buffer = ReplayBuffer(
      schema=self._buffer_schema(),
      buffer_size=cfg.buffer_size,
      num_envs=env.num_envs,
      device=device,
    )

    # 3. Differentiable World Model Environment
    self.wm_env = CrazyflieWorldModelEnv(
      dynamics=self.dynamics,
      representation=self.representation,
      env=env,
      batch_size=cfg.policy.batch_size,
      replay_buffer=self.replay_buffer,
      horizon=cfg.policy.horizon,
      history_length=cfg.actor_history_length,
      termination_softness=cfg.termination_softness,
      roll_pitch_spin_weight=cfg.roll_pitch_spin_weight,
      tilt_margin_weight=cfg.tilt_margin_weight,
      sim_context=self.sim_context,
      device=device,
    )

    # 4. Learners
    self.wm_learner = WorldModelLearner(
      dynamics=self.dynamics,
      representation=self.representation,
      cfg=cfg.wm,
      device=device,
    )

    obs_dim = int(env.get_observations()["actor"].shape[-1]) * cfg.actor_history_length
    self._sim_actor_hist: torch.Tensor | None = None
    # Actor policy (23 * actor_history_length obs -> 4 actions)
    self.actor = QuadcopterPolicy(
      obs_dim=obs_dim,
      action_dim=self.wm.num_actions,
      hidden_dims=cfg.actor_hidden_dims,
    ).to(device)

    # Policy learner
    self.policy_learner = BpttPolicyLearner(
      policy=self.actor,
      wm_env=self.wm_env,
      cfg=cfg.policy,
      device=device,
    )

    self._pending_norm_states: list[torch.Tensor] = []
    self._pending_norm_actions: list[torch.Tensor] = []
    self._init_noise = float(cfg.policy.noise)
    self._init_policy_lr = float(cfg.policy.lr)
    self._init_wm_lr = float(cfg.wm.lr)
    self.checkpoint_path = Path(cfg.checkpoint_dir)
    self.checkpoint_path.mkdir(parents=True, exist_ok=True)

  def seed_replay_buffer_from_corpus(
    self,
    corpus_path: str = "corpus/crazyflie_corpus_50trajs.npz",
    num_steps: int = 80,
  ) -> int:
    """Pre-seed the replay buffer with contiguous expert transitions from the corpus.

    Converts corpus body-frame velocities [v_b, w_b] to world-frame [v_w, w_w]
    to match the full_state representation, and populates the 7-dim world goal
    [ref_pos(3), ref_vel_w(3), ref_yaw_rate(1)].
    """
    import numpy as np

    data = np.load(corpus_path)
    num_trajs = int(data["num_trajs"][0])
    num_envs = self.env.num_envs
    seeded_transitions = 0

    print(f"[INFO] Seeding replay buffer with expert transitions from {corpus_path} ({num_trajs} trajs)...")

    if "state_offsets" in data:
      all_s = data["states"]
      all_c = data["ctrls"]
      s_off = data["state_offsets"]
      c_off = data["ctrl_offsets"]
      def get_traj(i):
        return all_s[s_off[i]:s_off[i+1]], all_c[c_off[i]:c_off[i+1]]
    else:
      def get_traj(i):
        return data[f"state_{i}"], data[f"ctrl_{i}"]

    # Each env tracks a distinct trajectory contiguously so long BPTT windows (horizon+2) are done-free
    env_trajs = [get_traj((e * 197) % num_trajs) for e in range(num_envs)]

    for step in range(num_steps):
      raw_s = []
      raw_c = []
      truncated_flags = []

      for e in range(num_envs):
        s, c = env_trajs[e]
        H = len(c)
        t = step % H
        raw_s.append(s[t, :13])
        raw_c.append(c[t])
        truncated_flags.append(t == H - 1)

      s_t = torch.from_numpy(np.stack(raw_s, axis=0)).float().to(self.device)
      c_t = torch.from_numpy(np.stack(raw_c, axis=0)).float().to(self.device)
      truncated_t = torch.tensor(truncated_flags, dtype=torch.bool, device=self.device)

      pos = s_t[:, 0:3]
      quat = s_t[:, 3:7]
      v_b = s_t[:, 7:10]
      w_b = s_t[:, 10:13]
      v_w = quat_apply(quat, v_b)
      w_w = quat_apply(quat, w_b)

      # full_state expects world-frame linear and angular velocities
      states_t = torch.cat([pos, quat, v_w, w_w], dim=-1)

      # Rotor speed ratio c in [0, 1] -> thrust F = 0.1542 * c^2 -> hover-centered action [-1, 1]
      thrust_n = 0.1542 * (c_t ** 2)
      actions_t = torch.clamp((thrust_n - 0.0809325) / 0.0732675, -1.0, 1.0)

      if self.wm.num_goals == 7:
        commands_t = torch.cat([pos, v_w, w_b[:, 2:3]], dim=-1)
      else:
        commands_t = torch.cat([v_b, w_b[:, 2:3]], dim=-1)

      td = TensorDict(
        {
          "state": states_t,
          "action": actions_t,
          "command": commands_t,
          "sim_ctx": torch.zeros(num_envs, self.sim_context.output_dim, device=self.device),
          "terminated": torch.zeros(num_envs, dtype=torch.bool, device=self.device),
          "truncated": truncated_t,
        },
        batch_size=[num_envs],
      )
      self.replay_buffer.insert(td)
      self._pending_norm_states.append(states_t)
      self._pending_norm_actions.append(actions_t)
      seeded_transitions += num_envs

    # Immediately initialize WM normalization with expert distribution
    if self._pending_norm_states:
      states_cat = torch.cat(self._pending_norm_states, dim=0)
      actions_cat = torch.cat(self._pending_norm_actions, dim=0)
      self.wm_learner.update_normalization(states_cat, actions_cat)
      self._pending_norm_states.clear()
      self._pending_norm_actions.clear()

    print(f"[INFO] Successfully seeded {seeded_transitions} transitions ({num_steps} steps x {num_envs} envs) into replay buffer.")
    return seeded_transitions

  def _buffer_schema(self) -> dict[str, FieldSpec]:
    return {
      "state": FieldSpec(self.representation.full_state_dim),
      "action": FieldSpec(self.wm.num_actions),
      "command": FieldSpec(self.wm.num_goals),
      "sim_ctx": FieldSpec(self.sim_context.output_dim),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    }

  def collect_rollout(self) -> dict[str, float]:
    """Collect real physics transitions using current policy."""
    total_reward = 0.0
    ep_len = 0

    with torch.inference_mode():
      obs = self.env.get_observations()
      if self.cfg.actor_history_length > 1 and self._sim_actor_hist is None:
        self._sim_actor_hist = obs["actor"].to(self.device).unsqueeze(1).repeat(1, self.cfg.actor_history_length, 1)
      for _ in range(self.cfg.num_collect_steps):
        actor_obs = obs["actor"].to(self.device)
        full_state = obs["full_state"].to(self.device)
        goal = obs["goal"].to(self.device)
        sim_ctx = self.sim_context.read(self.env.unwrapped)

        policy_in = (
          self._sim_actor_hist.reshape(self.env.num_envs, -1)
          if self.cfg.actor_history_length > 1
          else actor_obs
        )
        actions = self.actor(policy_in)
        half = actions.shape[0] // 2
        if half > 0:
          noise = torch.randn_like(actions[half:]) * self.cfg.policy.noise
          actions = torch.cat([
            torch.clamp(actions[:half], -1.0, 1.0),
            torch.clamp(actions[half:] + noise, -1.0, 1.0),
          ], dim=0)
        else:
          noise = torch.randn_like(actions) * self.cfg.policy.noise
          actions = torch.clamp(actions + noise, -1.0, 1.0)

        next_obs, rewards, dones, extras = self.env.step(actions)

        truncated = extras.get(
          "time_outs",
          torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
        )
        terminated = dones.bool() & ~truncated

        finite_mask = torch.isfinite(full_state).all(dim=-1) & torch.isfinite(actions).all(dim=-1)
        safe_state = torch.where(finite_mask.unsqueeze(-1), full_state, torch.zeros_like(full_state))
        safe_actions = torch.where(finite_mask.unsqueeze(-1), actions, torch.zeros_like(actions))

        td = TensorDict(
          {
            "state": safe_state,
            "action": safe_actions,
            "command": torch.nan_to_num(goal, nan=0.0, posinf=0.0, neginf=0.0),
            "sim_ctx": torch.nan_to_num(sim_ctx, nan=0.0, posinf=0.0, neginf=0.0),
            "terminated": terminated | ~finite_mask,
            "truncated": truncated,
          },
          batch_size=[self.env.num_envs],
        )
        self.replay_buffer.insert(td)

        if finite_mask.any():
          self._pending_norm_states.append(full_state[finite_mask])
          self._pending_norm_actions.append(actions[finite_mask])

        total_reward += torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
        ep_len += 1
        obs = next_obs
        if self.cfg.actor_history_length > 1:
          next_actor_obs = next_obs["actor"].to(self.device)
          shifted = torch.cat([self._sim_actor_hist[:, 1:], next_actor_obs.unsqueeze(1)], dim=1)
          done_mask = dones.bool()
          if done_mask.any():
            reset_hist = next_actor_obs.unsqueeze(1).repeat(1, self.cfg.actor_history_length, 1)
            shifted = torch.where(done_mask.view(-1, 1, 1), reset_hist, shifted)
          self._sim_actor_hist = shifted

    return {
      "rollout/mean_step_reward": total_reward / max(ep_len, 1),
    }

  def train_step(self, iteration: int) -> dict[str, float]:
    """Execute one training cycle: collect, update WM, update policy."""
    metrics: dict[str, float] = {}

    # Linearly decay exploration noise (0.1 -> 0.02) and Adam LR (lr -> 0.1*lr)
    frac = min(1.0, max(0.0, float(iteration - 1) / max(self.cfg.max_iterations - 1, 1)))
    self.cfg.policy.noise = self._init_noise * (1.0 - 0.8 * frac)
    lr_scale = 1.0 - 0.9 * frac
    for opt in self.policy_learner.actor_optimizers:
      for g in opt.param_groups:
        if torch.is_tensor(g["lr"]):
          g["lr"].fill_(self._init_policy_lr * lr_scale)
    for opt in self.wm_learner.optimizers:
      for g in opt.param_groups:
        if torch.is_tensor(g["lr"]):
          g["lr"].fill_(self._init_wm_lr * lr_scale)

    # 1. Collect rollouts
    t0 = time.time()
    rollout_metrics = self.collect_rollout()
    metrics.update(rollout_metrics)
    metrics["time/rollout_s"] = time.time() - t0

    # 2. Update normalizers
    if self._pending_norm_states:
      states_cat = torch.cat(self._pending_norm_states, dim=0)
      actions_cat = torch.cat(self._pending_norm_actions, dim=0)
      self.wm_learner.update_normalization(states_cat, actions_cat)
      self._pending_norm_states.clear()
      self._pending_norm_actions.clear()

    # 3. Train World Model if buffer has sufficient data
    min_steps = self.cfg.wm.horizon + 1
    if self.replay_buffer.num_transitions >= min_steps:
      t0 = time.time()
      wm_batches = [
        self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
        for _ in range(self.cfg.replay_ratio)
      ]
      wm_losses = self.wm_learner.update(wm_batches)
      for k, v in wm_losses.items():
        metrics[f"wm/{k}"] = v.item() if torch.is_tensor(v) else float(v)
      metrics["time/wm_s"] = time.time() - t0

      # 4. Train Policy with BPTT in differentiable World Model
      t0 = time.time()
      policy_metrics = self.policy_learner.update()
      for k, v in policy_metrics.items():
        metrics[f"policy/{k}"] = v.item() if torch.is_tensor(v) else float(v)
      metrics["time/policy_s"] = time.time() - t0

    return metrics

  def load_checkpoint(self, path: str) -> None:
    ckpt = torch.load(path, map_location=self.device, weights_only=False)
    self.actor.load_state_dict(ckpt["actor"])
    self.dynamics.load_state_dict(ckpt["dynamics"])
    if "sim_context" in ckpt:
      self.sim_context.load_state_dict(ckpt["sim_context"])
    print(f"[INFO] Loaded actor/dynamics/sim_context from {path} (iter={ckpt.get('iteration', '?')})")

  def save_checkpoint(self, iteration: int) -> None:
    path = self.checkpoint_path / f"checkpoint_{iteration:06d}.pt"
    torch.save(
      {
        "iteration": iteration,
        "obs_dim": self.actor.obs_dim,
        "actor_hidden_dims": self.actor.hidden_dims,
        "dynamics_hidden_dims": self.cfg.dynamics_hidden_dims,
        "actor": self.actor.state_dict(),
        "dynamics": self.dynamics.state_dict(),
        "sim_context": self.sim_context.state_dict(),
      },
      path,
    )
    print(f"[INFO] Saved checkpoint to {path}")
