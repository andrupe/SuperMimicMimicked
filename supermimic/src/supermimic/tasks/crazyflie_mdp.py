"""MDP terms (commands, observations, rewards, terminations) for Bitcraze Crazyflie."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# 1. Expert Corpus Trajectory Command & Brownian Velocity Command
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class CorpusTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for Expert Trajectory Tracking Command from normal corpus."""

    resampling_time_range: tuple[float, float] = (1000.0, 1000.0)
    entity_name: str = "robot"
    corpus_path: str = "corpus/crazyflie_corpus_50trajs.npz"
    rsi: bool = True               # Reference State Initialization on reset
    rsi_prob: float = 0.8          # Probability of RSI vs starting from default hover
    loop_trajectories: bool = True # If trajectory reaches end, loop back

    def build(self, env: ManagerBasedRlEnv) -> CorpusTrajectoryCommand:
        return CorpusTrajectoryCommand(self, env)


class CorpusTrajectoryCommand(CommandTerm):
    """Command term that tracks expert trajectories from SSQP corpus.

    Each parallel environment tracks a sampled trajectory from the 50-trajectory
    corpus (hover, point-to-point, helix, figure8). Provides reference body
    velocity [vx, vy, vz, wz] as command for the policy, and reference position
    and attitude for tracking rewards and Reference State Initialization (RSI).
    """

    cfg: CorpusTrajectoryCommandCfg

    def __init__(self, cfg: CorpusTrajectoryCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        # Load corpus
        data = np.load(cfg.corpus_path)
        num_trajs = int(data["num_trajs"][0])
        self.trajectories: list[dict[str, torch.Tensor | str | int]] = []
        for i in range(num_trajs):
            s = torch.from_numpy(data[f"state_{i}"]).float().to(self.device)
            c = torch.from_numpy(data[f"ctrl_{i}"]).float().to(self.device)
            tt = str(data[f"task_{i}"])
            self.trajectories.append({
                "pos": s[:, 0:3],
                "quat": s[:, 3:7],  # wxyz
                "lin_vel_b": s[:, 7:10],
                "ang_vel_b": s[:, 10:13],
                "ctrl": c,
                "H": len(c),
                "task_type": tt,
            })
        self.num_trajs = len(self.trajectories)

        # Per-env tracking state
        self.traj_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 4, device=self.device)
        self._ref_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._ref_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._ref_action = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._vel_command

    @property
    def ref_pos(self) -> torch.Tensor:
        return self._ref_pos

    @property
    def ref_quat(self) -> torch.Tensor:
        return self._ref_quat

    @property
    def ref_action(self) -> torch.Tensor:
        return self._ref_action

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        for env_id in env_ids:
            e = int(env_id.item())
            k = int(torch.randint(0, self.num_trajs, (1,)).item())
            H = int(self.trajectories[k]["H"])
            # Sample starting time step along trajectory
            t0 = int(torch.randint(0, max(1, H - 16), (1,)).item())

            self.traj_indices[e] = k
            self.time_indices[e] = t0

            traj = self.trajectories[k]
            # Command: [vx_b, vy_b, vz_b, wz_b]
            self._vel_command[e, :3] = traj["lin_vel_b"][t0]
            self._vel_command[e, 3] = traj["ang_vel_b"][t0, 2]
            self._ref_pos[e] = traj["pos"][t0]
            self._ref_quat[e] = traj["quat"][t0]
            if t0 < H:
                self._ref_action[e] = traj["ctrl"][t0]

            # Reference State Initialization (RSI)
            if self.cfg.rsi and torch.rand(1).item() < self.cfg.rsi_prob:
                q = traj["quat"][t0].unsqueeze(0)
                vb = traj["lin_vel_b"][t0].unsqueeze(0)
                wb = traj["ang_vel_b"][t0].unsqueeze(0)
                vw = quat_apply(q, vb).squeeze(0)
                ww = quat_apply(q, wb).squeeze(0)
                origin = self._env.scene.env_origins[e]
                root_pos = traj["pos"][t0] + origin

                root_state = torch.cat([root_pos, q.squeeze(0), vw, ww], dim=-1).unsqueeze(0)
                self.robot.write_root_state_to_sim(root_state, env_ids=torch.tensor([e], device=self.device))

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        ids = range(self.num_envs) if env_ids is None else [int(i.item()) for i in env_ids]

        for e in ids:
            k = int(self.traj_indices[e].item())
            H = int(self.trajectories[k]["H"])
            t = int(self.time_indices[e].item()) + 1

            if t >= H:
                if self.cfg.loop_trajectories:
                    t = 0
                else:
                    k = int(torch.randint(0, self.num_trajs, (1,)).item())
                    self.traj_indices[e] = k
                    H = int(self.trajectories[k]["H"])
                    t = 0

            self.time_indices[e] = t
            traj = self.trajectories[k]
            self._vel_command[e, :3] = traj["lin_vel_b"][t]
            self._vel_command[e, 3] = traj["ang_vel_b"][t, 2]
            self._ref_pos[e] = traj["pos"][t]
            self._ref_quat[e] = traj["quat"][t]
            if t < H:
                self._ref_action[e] = traj["ctrl"][t]


@dataclass(kw_only=True)
class BrownianVelocityCommandCfg(CommandTermCfg):
    """Configuration for Brownian / Ornstein-Uhlenbeck 3D velocity command."""

    resampling_time_range: tuple[float, float] = (3.0, 5.0)
    entity_name: str = "robot"
    # Command bounds: [vx (m/s), vy (m/s), vz (m/s), wz (rad/s)]
    ranges: tuple[tuple[float, float], ...] = (
        (-1.5, 1.5),   # vx body
        (-1.0, 1.0),   # vy body
        (-0.8, 0.8),   # vz body
        (-3.14, 3.14), # wz body (yaw rate)
    )
    # Ornstein-Uhlenbeck parameters
    theta: float = 0.5       # mean-reversion rate
    sigma: float = 0.4       # Brownian noise scale
    resample_time_s: float = 4.0 # Resample target every N seconds

    def build(self, env: ManagerBasedRlEnv) -> BrownianVelocityCommand:
        return BrownianVelocityCommand(self, env)


class BrownianVelocityCommand(CommandTerm):
    """Ornstein-Uhlenbeck Brownian random walk in 3D body velocity + yaw rate."""

    cfg: BrownianVelocityCommandCfg

    def __init__(self, cfg: BrownianVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        # Current command: (num_envs, 4) -> [vx, vy, vz, wz]
        self._vel_command = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_command = torch.zeros(self.num_envs, 4, device=self.device)

        # Min/max ranges
        self.ranges_min = torch.tensor([r[0] for r in cfg.ranges], device=self.device)
        self.ranges_max = torch.tensor([r[1] for r in cfg.ranges], device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._vel_command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        rand = torch.rand(n, 4, device=self.device)
        self.target_command[env_ids] = self.ranges_min + rand * (self.ranges_max - self.ranges_min)
        # Small probability (15%) of hovering (0 velocity) to ensure hover robustness
        zero_mask = torch.rand(n, device=self.device) < 0.15
        self.target_command[env_ids[zero_mask]] = 0.0
        self._vel_command[env_ids] = self.target_command[env_ids].clone()

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        dt = self._env.step_dt
        ids = slice(None) if env_ids is None else env_ids

        # Ornstein-Uhlenbeck step: dc = -theta*(c - target)*dt + sigma*sqrt(dt)*randn
        cmd = self._vel_command[ids]
        target = self.target_command[ids]
        noise = torch.randn_like(cmd) * (self.cfg.sigma * math.sqrt(dt))
        drift = -self.cfg.theta * (cmd - target) * dt
        self._vel_command[ids] = torch.clamp(cmd + drift + noise, self.ranges_min, self.ranges_max)


# ---------------------------------------------------------------------------
# 2. Observation Terms
# ---------------------------------------------------------------------------


def quadcopter_projected_gravity(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Gravity vector projected into body frame: R^T @ [0, 0, -1]. Shape: (num_envs, 3)."""
    robot: Entity = env.scene[entity_name]
    quat = robot.data.root_link_quat_w
    grav = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    return quat_apply_inverse(quat, grav)


def quadcopter_body_lin_vel(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Linear velocity in body frame: R^T @ v_w. Shape: (num_envs, 3)."""
    robot: Entity = env.scene[entity_name]
    quat = robot.data.root_link_quat_w
    v_w = robot.data.root_link_lin_vel_w
    return quat_apply_inverse(quat, v_w)


def quadcopter_body_ang_vel(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Angular velocity in body frame: omega_b. Shape: (num_envs, 3)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.root_link_ang_vel_w


def quadcopter_root_pos(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """World position relative to origin. Shape: (num_envs, 3)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.root_link_pos_w - env.scene.env_origins


def quadcopter_root_ori(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """World quaternion (w, x, y, z). Shape: (num_envs, 4)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.root_link_quat_w


def quadcopter_last_action(env: ManagerBasedRlEnv, action_name: str = "thrust") -> torch.Tensor:
    """Last applied rotor action. Shape: (num_envs, 4)."""
    return env.action_manager.get_term(action_name).raw_action


def quadcopter_velocity_command(env: ManagerBasedRlEnv, command_name: str = "velocity") -> torch.Tensor:
    """Active 3D velocity command [vx, vy, vz, wz]. Shape: (num_envs, 4)."""
    term = env.command_manager.get_term(command_name)
    return term.command


# ---------------------------------------------------------------------------
# 3. Reward Terms
# ---------------------------------------------------------------------------


def quadcopter_lin_vel_tracking_reward(
    env: ManagerBasedRlEnv, std: float = 0.5, entity_name: str = "robot", command_name: str = "velocity"
) -> torch.Tensor:
    """Gaussian reward for tracking commanded body 3D linear velocity [vx, vy, vz]."""
    term = env.command_manager.get_term(command_name)
    cmd_lin_vel = term.command[:, :3]
    actual_lin_vel = quadcopter_body_lin_vel(env, entity_name)
    error_sq = torch.sum(torch.square(actual_lin_vel - cmd_lin_vel), dim=-1)
    return torch.exp(-error_sq / (2.0 * std * std))


def quadcopter_yaw_rate_tracking_reward(
    env: ManagerBasedRlEnv, std: float = 0.8, entity_name: str = "robot", command_name: str = "velocity"
) -> torch.Tensor:
    """Gaussian reward for tracking commanded yaw rate wz."""
    term = env.command_manager.get_term(command_name)
    cmd_wz = term.command[:, 3]
    actual_wz = quadcopter_body_ang_vel(env, entity_name)[:, 2]
    error_sq = torch.square(actual_wz - cmd_wz)
    return torch.exp(-error_sq / (2.0 * std * std))


def quadcopter_corpus_ori_tracking_reward(
    env: ManagerBasedRlEnv, std: float = 0.5, entity_name: str = "robot", command_name: str = "velocity"
) -> torch.Tensor:
    """Reward for tracking the reference orientation of the corpus trajectory."""
    term = env.command_manager.get_term(command_name)
    if not hasattr(term, "ref_quat"):
        return torch.ones(env.num_envs, device=env.device)
    robot: Entity = env.scene[entity_name]
    actual_quat = robot.data.root_link_quat_w
    ref_quat = term.ref_quat
    # Absolute dot product between quaternions (since q and -q represent same SO(3) rotation)
    dot = torch.abs(torch.sum(actual_quat * ref_quat, dim=-1))
    dot = torch.clamp(dot, 0.0, 1.0)
    angle_err = 2.0 * torch.acos(torch.clamp(dot, -1.0, 1.0))
    return torch.exp(-torch.square(angle_err) / (2.0 * std * std))


def quadcopter_corpus_pos_tracking_reward(
    env: ManagerBasedRlEnv, std: float = 0.5, entity_name: str = "robot", command_name: str = "velocity"
) -> torch.Tensor:
    """Reward for tracking reference position in point-to-point / hover / trajectory."""
    term = env.command_manager.get_term(command_name)
    if not hasattr(term, "ref_pos"):
        return torch.ones(env.num_envs, device=env.device)
    actual_pos = quadcopter_root_pos(env, entity_name)
    ref_pos = term.ref_pos
    pos_err_sq = torch.sum(torch.square(actual_pos - ref_pos), dim=-1)
    return torch.exp(-pos_err_sq / (2.0 * std * std))


def quadcopter_upright_reward(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Reward for keeping the quadcopter upright: (1 + g_bz) / 2 in [0, 1]."""
    gb = quadcopter_projected_gravity(env, entity_name)
    # gb_z = -1 when perfectly level
    return 0.5 * (1.0 - gb[:, 2])


def quadcopter_action_rate_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalty on abrupt rotor command rate: ||a_t - a_{t-1}||^2."""
    diff = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(diff), dim=-1)


def quadcopter_hover_effort_penalty(env: ManagerBasedRlEnv, hover_norm: float = 0.0) -> torch.Tensor:
    """Minor regularization encouraging thrust near nominal hover (0.0)."""
    raw = env.action_manager.action
    return torch.sum(torch.square(raw - hover_norm), dim=-1)


# ---------------------------------------------------------------------------
# 4. Termination Terms
# ---------------------------------------------------------------------------


def quadcopter_crash_termination(
    env: ManagerBasedRlEnv, min_altitude: float = 0.05, max_altitude: float = 5.0, entity_name: str = "robot"
) -> torch.Tensor:
    """Terminate if the quadcopter hits the ground or exceeds ceiling."""
    robot: Entity = env.scene[entity_name]
    z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (z < min_altitude) | (z > max_altitude)


def quadcopter_tilt_termination(
    env: ManagerBasedRlEnv, max_tilt_angle_deg: float = 80.0, entity_name: str = "robot"
) -> torch.Tensor:
    """Terminate if the quadcopter tilts beyond threshold."""
    gb = quadcopter_projected_gravity(env, entity_name)
    # cos(theta) = -gb_z. If theta > max_tilt -> -gb_z < cos(max_tilt)
    cos_threshold = math.cos(math.radians(max_tilt_angle_deg))
    return -gb[:, 2] < cos_threshold
