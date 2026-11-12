"""Crazyflie Quadcopter Velocity Runner with SuperMimic BPTT & World Model training."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

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
  """Actor policy mapping 17-dim actor observations to 4-dim rotor actions."""

  def __init__(self, obs_dim: int = 17, action_dim: int = 4, hidden_dims: tuple[int, ...] = (128, 64)):
    super().__init__()
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
  dynamics_hidden_dims: tuple[int, ...] = (256, 256)
  actor_hidden_dims: tuple[int, ...] = (128, 64)
  critic_hidden_dims: tuple[int, ...] = (128, 64)
  policy: BpttPolicyLearnerCfg = field(default_factory=BpttPolicyLearnerCfg)
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)
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
      termination_softness=cfg.termination_softness,
      device=device,
    )

    # 4. Learners
    self.wm_learner = WorldModelLearner(
      dynamics=self.dynamics,
      representation=self.representation,
      cfg=cfg.wm,
      device=device,
    )

    # Actor policy (17 obs -> 4 actions)
    self.actor = QuadcopterPolicy(
      obs_dim=17,
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
    self.checkpoint_path = Path(cfg.checkpoint_dir)
    self.checkpoint_path.mkdir(parents=True, exist_ok=True)

  def seed_replay_buffer_from_corpus(
    self,
    corpus_path: str = "corpus/crazyflie_corpus_50trajs.npz",
    num_steps: int = 50,
  ) -> int:
    """Pre-seed the replay buffer with expert transitions from the SSQP corpus.

    This ensures the neural World Model has high-quality, dynamically valid
    quadcopter transitions from iteration 1 across all flight regimes.
    """
    import numpy as np

    data = np.load(corpus_path)
    num_trajs = int(data["num_trajs"][0])
    num_envs = self.env.num_envs
    seeded_transitions = 0

    print(f"[INFO] Seeding replay buffer with expert transitions from {corpus_path} ({num_trajs} trajs)...")

    for step in range(num_steps):
      states = []
      actions = []
      commands = []
      truncated_flags = []

      for e in range(num_envs):
        traj_id = (e + (step // 20) * num_envs) % num_trajs
        s = data[f"state_{traj_id}"]
        c = data[f"ctrl_{traj_id}"]
        H = len(c)
        t = step % H

        states.append(torch.from_numpy(s[t, :13]))
        # Rotor speed ratio c in [0, 1] -> thrust F = 0.1542 * c^2 -> hover-centered action [-1, 1]
        thrust_n = 0.1542 * (torch.from_numpy(c[t]) ** 2)
        u = torch.clamp((thrust_n - 0.0809325) / 0.0732675, -1.0, 1.0)
        actions.append(u)
        # Reference command: [vx, vy, vz, wz]
        commands.append(torch.tensor([s[t, 7], s[t, 8], s[t, 9], s[t, 12]]))
        truncated_flags.append(t == H - 1)

      states_t = torch.stack(states).float().to(self.device)
      actions_t = torch.stack(actions).float().to(self.device)
      commands_t = torch.stack(commands).float().to(self.device)
      truncated_t = torch.tensor(truncated_flags, dtype=torch.bool, device=self.device)

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
      "command": FieldSpec(4),
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
      for _ in range(self.cfg.num_collect_steps):
        actor_obs = obs["actor"].to(self.device)
        full_state = obs["full_state"].to(self.device)
        goal = obs["goal"].to(self.device)
        sim_ctx = self.sim_context.read(self.env.unwrapped)

        actions = self.actor(actor_obs)
        noise = torch.randn_like(actions) * self.cfg.policy.noise
        actions = torch.clamp(actions + noise, -1.0, 1.0)

        next_obs, rewards, dones, extras = self.env.step(actions)

        truncated = extras.get(
          "time_outs",
          torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
        )
        terminated = dones.bool() & ~truncated

        td = TensorDict(
          {
            "state": full_state,
            "action": actions,
            "command": goal,
            "sim_ctx": sim_ctx,
            "terminated": terminated,
            "truncated": truncated,
          },
          batch_size=[self.env.num_envs],
        )
        self.replay_buffer.insert(td)

        self._pending_norm_states.append(full_state)
        self._pending_norm_actions.append(actions)

        total_reward += rewards.mean().item()
        ep_len += 1
        obs = next_obs

    return {
      "rollout/mean_step_reward": total_reward / max(ep_len, 1),
    }

  def train_step(self, iteration: int) -> dict[str, float]:
    """Execute one training cycle: collect, update WM, update policy."""
    metrics: dict[str, float] = {}

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

  def save_checkpoint(self, iteration: int) -> None:
    path = self.checkpoint_path / f"checkpoint_{iteration:06d}.pt"
    torch.save(
      {
        "iteration": iteration,
        "actor": self.actor.state_dict(),
        "dynamics": self.dynamics.state_dict(),
        "sim_context": self.sim_context.state_dict(),
      },
      path,
    )
    print(f"[INFO] Saved checkpoint to {path}")
