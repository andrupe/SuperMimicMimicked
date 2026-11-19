"""MDP terms (commands, observations, rewards, terminations) for Bitcraze Crazyflie."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.actions.actions import SiteEffortAction, SiteEffortActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# 1. Expert Corpus Trajectory Command & Brownian Velocity Command
# ---------------------------------------------------------------------------


def _euler_to_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Batched ZYX Euler -> unit quaternion (w, x, y, z)."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    q = torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=-1)
    return q / torch.norm(q, dim=-1, keepdim=True)


def sample_random_root_states(origins: torch.Tensor) -> torch.Tensor:
    """Batched wide-envelope root states [p(3), q(4), v_w(3), w_w(3)] within safe physical limits."""
    n, dev = origins.shape[0], origins.device
    p = origins.clone()
    p[:, 0:2] += (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * 1.2
    p[:, 2] += 0.45 + torch.rand(n, device=dev) * (2.0 - 0.45)
    max_tilt = math.radians(35.0)
    roll = (torch.rand(n, device=dev) * 2.0 - 1.0) * max_tilt
    pitch = (torch.rand(n, device=dev) * 2.0 - 1.0) * max_tilt
    yaw = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi
    q = _euler_to_quat(roll, pitch, yaw)
    v = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * torch.tensor([1.2, 1.2, 0.7], device=dev)
    v[:, 2] += 0.1
    w = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * torch.tensor([2.0, 2.0, 1.5], device=dev)
    return torch.cat([p, q, v, w], dim=-1)


def sample_random_root_state(origin: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Single-env wrapper kept for backward compatibility. Returns (1, 13)."""
    return sample_random_root_states(origin.to(device).unsqueeze(0))


def _load_corpus_flat(path: str, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Load a corpus into flat GPU tensors: states (N, 13), ctrls (M, 4), state/ctrl offsets, task labels."""
    data = np.load(path)
    num_trajs = int(data["num_trajs"][0])
    if "state_offsets" in data:
        states = torch.from_numpy(np.ascontiguousarray(data["states"][:, :13])).float()
        ctrls = torch.from_numpy(data["ctrls"]).float()
        s_off = torch.from_numpy(data["state_offsets"].astype(np.int64))
        c_off = torch.from_numpy(data["ctrl_offsets"].astype(np.int64))
        tasks = [str(t) for t in data["tasks"]]
    else:
        s_list = [data[f"state_{i}"][:, :13] for i in range(num_trajs)]
        c_list = [data[f"ctrl_{i}"] for i in range(num_trajs)]
        tasks = [str(data[f"task_{i}"]) for i in range(num_trajs)]
        states = torch.from_numpy(np.concatenate(s_list, 0)).float()
        ctrls = torch.from_numpy(np.concatenate(c_list, 0)).float()
        s_off = torch.from_numpy(np.cumsum([0] + [len(s) for s in s_list]).astype(np.int64))
        c_off = torch.from_numpy(np.cumsum([0] + [len(c) for c in c_list]).astype(np.int64))
    return states.to(device), ctrls.to(device), s_off.to(device), c_off.to(device), tasks


class _TrajectoryView:
    """List-like accessor returning per-trajectory dicts (views into the flat corpus)."""

    def __init__(self, term: "JointCorpusAndVelocityCommand"):
        self._t = term

    def __len__(self) -> int:
        return self._t.num_trajs

    def __getitem__(self, k: int) -> dict:
        t = self._t
        s0, s1 = int(t._s_off[k]), int(t._s_off[k + 1])
        c0, c1 = int(t._c_off[k]), int(t._c_off[k + 1])
        s = t._states[s0:s1]
        return {
            "pos": s[:, 0:3],
            "quat": s[:, 3:7],
            "lin_vel_b": s[:, 7:10],
            "ang_vel_b": s[:, 10:13],
            "ctrl": t._ctrls[c0:c1],
            "H": c1 - c0,
            "task_type": t._tasks[k],
        }


@dataclass(kw_only=True)
class JointCorpusAndVelocityCommandCfg(CommandTermCfg):
    """Configuration for Joint Corpus + Brownian reference tracking command.

    A fraction ``corpus_ratio`` of environments track expert corpus trajectories
    (with Reference State Initialization); the rest track a smooth Brownian
    (Ornstein-Uhlenbeck) world-frame velocity reference whose position is
    integrated. Every env exposes a full reference: position, world velocity
    and yaw rate.

    Resampling (which teleports the robot) happens ONLY on episode reset; a
    finished corpus trajectory holds its final pose (zero velocity) instead
    of looping, so neither the robot state nor the reference position ever
    jumps inside an episode.
    """

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    entity_name: str = "robot"
    corpus_path: str = "corpus/crazyflie_corpus_all90trajs.npz"
    corpus_ratio: float = 0.5
    rsi: bool = True
    rsi_prob: float = 0.8
    loop_trajectories: bool = False  # kept for config compatibility; trajectories hold their end pose
    ranges: tuple[tuple[float, float], ...] = (
        (-1.5, 1.5),   # vx world
        (-1.5, 1.5),   # vy world
        (-0.8, 0.8),   # vz world
        (-3.14, 3.14), # wz (yaw rate)
    )
    theta: float = 0.5
    sigma: float = 0.4
    ref_box_min: tuple[float, float, float] = (-1.8, -1.8, 0.4)
    ref_box_max: tuple[float, float, float] = (1.8, 1.8, 2.2)

    def build(self, env: ManagerBasedRlEnv) -> JointCorpusAndVelocityCommand:
        return JointCorpusAndVelocityCommand(self, env)


class JointCorpusAndVelocityCommand(CommandTerm):
    """Vectorized reference generator providing corpus and Brownian tracking references."""

    cfg: JointCorpusAndVelocityCommandCfg

    def __init__(self, cfg: JointCorpusAndVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        (self._states, self._ctrls, self._s_off, self._c_off, self._tasks) = _load_corpus_flat(
            cfg.corpus_path, self.device
        )
        self._traj_len = self._c_off[1:] - self._c_off[:-1]  # H per trajectory (states have H+1 rows)
        self.num_trajs = int(self._traj_len.shape[0])
        self.trajectories = _TrajectoryView(self)

        n = self.num_envs
        self.num_corpus_envs = int(round(n * cfg.corpus_ratio))
        self.is_corpus = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.is_corpus[: self.num_corpus_envs] = True

        self.traj_indices = torch.zeros(n, dtype=torch.long, device=self.device)
        self.time_indices = torch.zeros(n, dtype=torch.long, device=self.device)
        self._vel_command = torch.zeros(n, 4, device=self.device)  # legacy [v_b ref (3), wz ref]
        self._ref_pos = torch.zeros(n, 3, device=self.device)       # relative to env origin
        self._ref_vel_w = torch.zeros(n, 3, device=self.device)     # world-frame reference velocity
        self._ref_quat = torch.zeros(n, 4, device=self.device)
        self._ref_quat[:, 0] = 1.0
        self._ref_action = torch.zeros(n, 4, device=self.device)

        self.target_command = torch.zeros(n, 4, device=self.device)
        self.ranges_min = torch.tensor([r[0] for r in cfg.ranges], device=self.device)
        self.ranges_max = torch.tensor([r[1] for r in cfg.ranges], device=self.device)
        self._box_min = torch.tensor(cfg.ref_box_min, device=self.device)
        self._box_max = torch.tensor(cfg.ref_box_max, device=self.device)

    # --- properties --------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        return self._vel_command

    @property
    def ref_pos(self) -> torch.Tensor:
        return self._ref_pos

    @property
    def ref_vel_w(self) -> torch.Tensor:
        return self._ref_vel_w

    @property
    def ref_yaw_rate(self) -> torch.Tensor:
        return self._vel_command[:, 3]

    @property
    def ref_quat(self) -> torch.Tensor:
        return self._ref_quat

    @property
    def ref_action(self) -> torch.Tensor:
        return self._ref_action

    def _update_metrics(self) -> None:
        pass

    # --- corpus helpers ----------------------------------------------------
    def _lookup(self, k: torch.Tensor, t: torch.Tensor):
        H = self._traj_len[k]
        tc = torch.minimum(t.clamp(min=0), H)
        st = self._states[self._s_off[k] + tc]
        ctrl = self._ctrls[self._c_off[k] + torch.minimum(tc, H - 1)]
        return st[:, 0:3], st[:, 3:7], st[:, 7:10], st[:, 10:13], ctrl, t >= H

    def _set_corpus_reference(self, ids: torch.Tensor, k: torch.Tensor, t: torch.Tensor) -> None:
        pos, quat, vb, wb, ctrl, hold = self._lookup(k, t)
        live = (~hold).float().unsqueeze(-1)
        self._ref_pos[ids] = pos
        self._ref_quat[ids] = quat
        self._ref_vel_w[ids] = quat_apply(quat, vb) * live
        self._vel_command[ids, :3] = vb * live
        self._vel_command[ids, 3] = wb[:, 2] * live.squeeze(-1)
        self._ref_action[ids] = ctrl

    def set_trajectory(self, env_id: int, traj_id: int, t: int = 0) -> None:
        """Pin ``env_id`` to corpus trajectory ``traj_id`` at time ``t`` (used by evaluation)."""
        ids = torch.tensor([env_id], device=self.device)
        k = torch.tensor([traj_id], device=self.device)
        tt = torch.tensor([t], device=self.device)
        self.is_corpus[env_id] = True
        self.traj_indices[ids] = k
        self.time_indices[ids] = tt
        self._set_corpus_reference(ids, k, tt)

    # --- CommandTerm API -----------------------------------------------------
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(self.device).long()
        n = env_ids.shape[0]
        dev = self.device
        origins = self._env.scene.env_origins[env_ids]
        corpus = self.is_corpus[env_ids]

        root = sample_random_root_states(origins)

        # ---- corpus envs: pick trajectory + start time (t0 >= 1) ----
        k = torch.randint(0, self.num_trajs, (n,), device=dev)
        H = self._traj_len[k]
        span = (H - 16).clamp(min=2)
        t0 = 1 + (torch.rand(n, device=dev) * (span - 1).float()).long()
        pos, quat, vb, wb, ctrl, _ = self._lookup(k, t0)

        if self.cfg.rsi:
            rsi = corpus & (torch.rand(n, device=dev) < self.cfg.rsi_prob)
            p_pert = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * torch.tensor([0.10, 0.10, 0.06], device=dev)
            v_pert = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * torch.tensor([0.20, 0.20, 0.15], device=dev)
            w_pert = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * torch.tensor([0.6, 0.6, 0.4], device=dev)
            rp = pos + origins + p_pert
            rp[:, 2] = torch.maximum(torch.minimum(rp[:, 2], origins[:, 2] + 2.3), origins[:, 2] + 0.25)
            rsi_root = torch.cat([rp, quat, quat_apply(quat, vb) + v_pert, quat_apply(quat, wb) + w_pert], dim=-1)
            root = torch.where(rsi.unsqueeze(-1), rsi_root, root)

        self.robot.write_root_state_to_sim(root, env_ids=env_ids)

        # Corpus references (time index t0-1 so the same-step _update_command lands on t0).
        c_ids = env_ids[corpus]
        if c_ids.numel() > 0:
            self.traj_indices[c_ids] = k[corpus]
            self.time_indices[c_ids] = t0[corpus] - 1
            self._set_corpus_reference(c_ids, k[corpus], t0[corpus])

        # Brownian references start at the robot's initial position.
        b_mask = ~corpus
        b_ids = env_ids[b_mask]
        if b_ids.numel() > 0:
            nb = b_ids.shape[0]
            target = self.ranges_min + torch.rand(nb, 4, device=dev) * (self.ranges_max - self.ranges_min)
            target = torch.where((torch.rand(nb, device=dev) < 0.15).unsqueeze(-1), torch.zeros_like(target), target)
            self.target_command[b_ids] = target
            self._vel_command[b_ids] = target
            self._ref_vel_w[b_ids] = target[:, :3]
            self._ref_pos[b_ids] = root[b_mask, :3] - origins[b_mask]
            self._ref_pos[b_ids] = torch.maximum(torch.minimum(self._ref_pos[b_ids], self._box_max), self._box_min)
            self._ref_quat[b_ids] = root[b_mask, 3:7]
            self._ref_action[b_ids] = 0.0

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        dt = self._env.step_dt
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(self.device).long()
        else:
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if ids.numel() == 0:
            return

        corpus = self.is_corpus[ids]
        c_ids = ids[corpus]
        if c_ids.numel() > 0:
            k = self.traj_indices[c_ids]
            t = torch.minimum(self.time_indices[c_ids] + 1, self._traj_len[k])  # hold final pose at t == H
            self.time_indices[c_ids] = t
            self._set_corpus_reference(c_ids, k, t)

        b_ids = ids[~corpus]
        if b_ids.numel() > 0:
            cmd = self._vel_command[b_ids]
            target = self.target_command[b_ids]
            noise = torch.randn_like(cmd) * (self.cfg.sigma * math.sqrt(dt))
            drift = -self.cfg.theta * (cmd - target) * dt
            cmd = torch.clamp(cmd + drift + noise, self.ranges_min, self.ranges_max)
            new_pos = self._ref_pos[b_ids] + cmd[:, :3] * dt
            clamped = torch.maximum(torch.minimum(new_pos, self._box_max), self._box_min)
            hit = (clamped != new_pos)
            cmd[:, :3] = torch.where(hit, torch.zeros_like(cmd[:, :3]), cmd[:, :3])
            self._vel_command[b_ids] = cmd
            self._ref_vel_w[b_ids] = cmd[:, :3]
            self._ref_pos[b_ids] = clamped


@dataclass(kw_only=True)
class CorpusTrajectoryCommandCfg(JointCorpusAndVelocityCommandCfg):
    """Pure corpus-tracking command (all envs track expert trajectories)."""

    corpus_path: str = "corpus/crazyflie_corpus_50trajs.npz"
    corpus_ratio: float = 1.0

    def build(self, env: ManagerBasedRlEnv) -> CorpusTrajectoryCommand:
        return CorpusTrajectoryCommand(self, env)


class CorpusTrajectoryCommand(JointCorpusAndVelocityCommand):
    """Command term that tracks expert trajectories from the corpus (all envs)."""

    cfg: CorpusTrajectoryCommandCfg


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
    quat = robot.data.root_link_quat_w
    w_w = robot.data.root_link_ang_vel_w
    return quat_apply_inverse(quat, w_w)


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


def quadcopter_world_lin_vel(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Linear velocity in WORLD frame. Shape: (num_envs, 3).

    The full_state group must carry world-frame twists: the gencoord
    representation rotates them into the body frame itself.
    """
    robot: Entity = env.scene[entity_name]
    return robot.data.root_link_lin_vel_w


def quadcopter_world_ang_vel(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Angular velocity in WORLD frame. Shape: (num_envs, 3)."""
    robot: Entity = env.scene[entity_name]
    return robot.data.root_link_ang_vel_w


TRACKING_POS_ERR_CLIP = 1.5  # metres, per axis (shared with the world-model env)


def quadcopter_tracking_goal(env: ManagerBasedRlEnv, command_name: str = "velocity") -> torch.Tensor:
    """World-frame tracking reference [ref_pos(3), ref_vel_w(3), ref_yaw_rate(1)]. Shape: (num_envs, 7)."""
    term = env.command_manager.get_term(command_name)
    return torch.cat([term.ref_pos, term.ref_vel_w, term.ref_yaw_rate.unsqueeze(-1)], dim=-1)


def tracking_command_body(
    pos: torch.Tensor, quat: torch.Tensor, goal: torch.Tensor
) -> torch.Tensor:
    """Body-frame tracking command [pos_err_b(3), ref_vel_b(3), ref_yaw_rate(1)] from a 7-dim world goal."""
    pos_err_b = quat_apply_inverse(quat, goal[..., 0:3] - pos)
    pos_err_b = torch.clamp(pos_err_b, -TRACKING_POS_ERR_CLIP, TRACKING_POS_ERR_CLIP)
    ref_vel_b = quat_apply_inverse(quat, goal[..., 3:6])
    return torch.cat([pos_err_b, ref_vel_b, goal[..., 6:7]], dim=-1)


def quadcopter_tracking_command(
    env: ManagerBasedRlEnv, entity_name: str = "robot", command_name: str = "velocity"
) -> torch.Tensor:
    """Actor tracking command in body frame. Shape: (num_envs, 7)."""
    robot: Entity = env.scene[entity_name]
    goal = quadcopter_tracking_goal(env, command_name)
    return tracking_command_body(quadcopter_root_pos(env, entity_name), robot.data.root_link_quat_w, goal)


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
    """Terminate if the quadcopter hits the ground, exceeds ceiling, or becomes non-finite."""
    robot: Entity = env.scene[entity_name]
    pos = robot.data.root_link_pos_w
    vel = robot.data.root_link_lin_vel_w
    z = pos[:, 2] - env.scene.env_origins[:, 2]
    non_finite = ~torch.isfinite(pos).all(dim=-1) | ~torch.isfinite(vel).all(dim=-1)
    return (z < min_altitude) | (z > max_altitude) | non_finite


def quadcopter_tilt_termination(
    env: ManagerBasedRlEnv, max_tilt_angle_deg: float = 80.0, entity_name: str = "robot"
) -> torch.Tensor:
    """Terminate if the quadcopter tilts beyond threshold."""
    gb = quadcopter_projected_gravity(env, entity_name)
    # cos(theta) = -gb_z. If theta > max_tilt -> -gb_z < cos(max_tilt)
    cos_threshold = math.cos(math.radians(max_tilt_angle_deg))
    return -gb[:, 2] < cos_threshold


# ---------------------------------------------------------------------------
# 5. Realistic Crazyflie 2.1 Motor Lag, Battery Sag, Drag & Wind Action
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class CrazyflieMotorActionCfg(SiteEffortActionCfg):
    """Site effort action with Crazyflie 2.1 motor lag, battery sag, rotor asymmetry, and aero drag/wind."""

    max_thrust: float = 0.1542
    enable_domain_randomization: bool = True
    tau_up_range: tuple[float, float] = (0.022, 0.029)
    tau_down_range: tuple[float, float] = (0.023, 0.032)
    nominal_tau: float = 0.025
    thrust_scale_range: tuple[float, float] = (0.96, 1.02)
    sag_rate_range: tuple[float, float] = (0.0, 0.0025)
    rotor_gain_range: tuple[float, float] = (0.985, 1.015)
    wind_xy_range: tuple[float, float] = (-0.10, 0.10)
    wind_z_range: tuple[float, float] = (-0.04, 0.04)
    drag_scale_range: tuple[float, float] = (0.95, 1.08)
    rotor_drag_coeff: float = 0.0085   # N / (m/s) at hover thrust
    quad_drag_coeff: float = 0.0035    # N / (m/s)^2 fuselage drag

    def build(self, env: ManagerBasedRlEnv) -> CrazyflieMotorAction:
        return CrazyflieMotorAction(self, env)


class CrazyflieMotorAction(SiteEffortAction):
    """200 Hz Crazyflie 2.1 motor dynamics + battery sag + aero drag/wind."""

    cfg: CrazyflieMotorActionCfg

    def __init__(self, cfg: CrazyflieMotorActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._dt_phys = float(env.cfg.sim.mujoco.timestep)
        self._base_body_ids, _ = self._entity.find_bodies("base")
        hover_f = float(cfg.offset) if isinstance(cfg.offset, (int, float)) else 0.0809325
        self._hover_thrust = hover_f
        self._thrust_state = torch.full((self.num_envs, self.action_dim), hover_f, device=self.device)
        self._tau_up = torch.full((self.num_envs, 1), cfg.nominal_tau, device=self.device)
        self._tau_down = torch.full((self.num_envs, 1), cfg.nominal_tau, device=self.device)
        self._thrust_scale = torch.ones((self.num_envs, 1), device=self.device)
        self._sag_rate = torch.zeros((self.num_envs, 1), device=self.device)
        self._rotor_gains = torch.ones((self.num_envs, self.action_dim), device=self.device)
        self._wind_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._drag_scale = torch.ones((self.num_envs, 1), device=self.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            n = self.num_envs
        elif isinstance(env_ids, slice):
            n = self.num_envs
        else:
            n = len(env_ids)
            if n == 0:
                return

        if self._thrust_state.is_inference():
            self._thrust_state = self._thrust_state.clone()
        if self._thrust_scale.is_inference():
            self._thrust_scale = self._thrust_scale.clone()
        self._thrust_state[env_ids] = self._hover_thrust
        if self.cfg.enable_domain_randomization:
            if self.num_envs > 1:
                if env_ids is None or isinstance(env_ids, slice):
                    all_ids = torch.arange(self.num_envs, device=self.device)
                else:
                    all_ids = env_ids
                half = self.num_envs // 2
                nom_ids = all_ids[all_ids < half]
                dr_ids = all_ids[all_ids >= half]
            else:
                nom_ids = None
                dr_ids = env_ids

            if nom_ids is not None and len(nom_ids) > 0:
                self._tau_up[nom_ids] = self.cfg.nominal_tau
                self._tau_down[nom_ids] = self.cfg.nominal_tau
                self._thrust_scale[nom_ids] = 1.0
                self._sag_rate[nom_ids] = 0.0
                self._rotor_gains[nom_ids] = 1.0
                self._wind_w[nom_ids] = 0.0
                self._drag_scale[nom_ids] = 1.0

            n_dr = self.num_envs if dr_ids is None or isinstance(dr_ids, slice) else len(dr_ids)
            if n_dr > 0:
                u = lambda lo, hi, shape: lo + (hi - lo) * torch.rand(shape, device=self.device)
                self._tau_up[dr_ids] = u(*self.cfg.tau_up_range, (n_dr, 1))
                self._tau_down[dr_ids] = u(*self.cfg.tau_down_range, (n_dr, 1))
                self._thrust_scale[dr_ids] = u(*self.cfg.thrust_scale_range, (n_dr, 1))
                self._sag_rate[dr_ids] = u(*self.cfg.sag_rate_range, (n_dr, 1))
                self._rotor_gains[dr_ids] = u(*self.cfg.rotor_gain_range, (n_dr, self.action_dim))
                wind_xy = u(*self.cfg.wind_xy_range, (n_dr, 2))
                wind_z = u(*self.cfg.wind_z_range, (n_dr, 1))
                self._wind_w[dr_ids] = torch.cat([wind_xy, wind_z], dim=-1)
                self._drag_scale[dr_ids] = u(*self.cfg.drag_scale_range, (n_dr, 1))
        else:
            self._tau_up[env_ids] = self.cfg.nominal_tau
            self._tau_down[env_ids] = self.cfg.nominal_tau
            self._thrust_scale[env_ids] = 1.0
            self._sag_rate[env_ids] = 0.0
            self._rotor_gains[env_ids] = 1.0
            self._wind_w[env_ids] = 0.0
            self._drag_scale[env_ids] = 1.0

    def apply_actions(self) -> None:
        # 1. First-order asymmetric motor spin-up/spin-down lag (200 Hz)
        target = self._processed_actions
        tau = torch.where(target >= self._thrust_state, self._tau_up, self._tau_down)
        alpha = self._dt_phys / (tau + self._dt_phys)
        self._thrust_state = self._thrust_state + alpha * (target - self._thrust_state)

        # 2. Dynamic battery voltage sag + per-rotor thrust asymmetry
        self._thrust_scale = torch.clamp(self._thrust_scale - self._sag_rate * self._dt_phys, min=0.70)
        applied_thrust = torch.clamp(
            self._thrust_state * self._thrust_scale * self._rotor_gains,
            min=0.0,
            max=self.cfg.max_thrust,
        )
        self._entity.set_site_effort_target(applied_thrust, site_ids=self._target_ids)

        # 3. Body-frame rotor/induced drag + quadratic fuselage drag + wind
        v_rel_w = self._entity.data.root_link_lin_vel_w - self._wind_w
        q_w = self._entity.data.root_link_quat_w
        v_rel_b = quat_apply_inverse(q_w, v_rel_w)
        thrust_ratio = applied_thrust.sum(dim=-1, keepdim=True) / max(self._hover_thrust * 4.0, 1e-6)
        f_drag_b = -(
            self.cfg.rotor_drag_coeff * thrust_ratio * v_rel_b
            + self.cfg.quad_drag_coeff * torch.abs(v_rel_b) * v_rel_b
        ) * self._drag_scale
        f_drag_b = torch.clamp(f_drag_b, min=-2.0, max=2.0)
        f_drag_w = torch.nan_to_num(quat_apply(q_w, f_drag_b), nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(1)
        self._entity.write_external_wrench_to_sim(
            forces=f_drag_w,
            torques=torch.zeros_like(f_drag_w),
            body_ids=self._base_body_ids,
        )
