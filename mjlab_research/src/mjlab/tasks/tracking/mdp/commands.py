from __future__ import annotations

import copy
import glob as globmod
import json
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


def _resolve_motion_file(source: str | list[str]) -> list[str]:
  """Resolve motion source to list of file paths."""
  if isinstance(source, list):
    return source
  if any(c in source for c in "*?[]"):
    files = sorted(globmod.glob(source))
    if not files:
      raise ValueError(f"No files matched pattern: {source}")
    return files
  return [source]


class MotionLoader:
  """Loads reference motion as concatenated per-frame tensors on ``device``: the
  root link's world pose and twist and the joint positions and velocities.
  ``body_state`` computes the tracked bodies from them with ``fk``. Motion files
  hold them as ``root_pos_w``, ``root_quat_w`` (wxyz), ``root_lin_vel_w``,
  ``root_ang_vel_w``, ``joint_pos`` and ``joint_vel``, with ``joint_names``, which
  must equal ``fk.joint_names``.

  ``motion_file`` is either per-clip ``.npz`` files (one file, a comma list, or a
  glob, concatenated in sorted order) or a single packed directory written by
  ``scripts/pack_motions.py`` (one ``<field>.npy`` plus ``motion_ends.npy`` and
  ``joint_names.json``). The packed directory loads with a few sequential reads
  instead of opening tens of thousands of clips, so it is the path for large
  datasets; ``.npz`` clips remain for registry motions, single clips, and tests.
  """

  def __init__(
    self, motion_file: str | list[str], fk: ForwardKinematics, device: str = "cpu"
  ) -> None:
    files = _resolve_motion_file(motion_file)

    packed = len(files) == 1 and os.path.isdir(files[0])
    if packed:
      root = files[0]
      clips: list[dict] = []
      motion_ends = np.load(os.path.join(root, "motion_ends.npy"))
      with open(os.path.join(root, "joint_names.json")) as fh:
        clip_joints = [json.load(fh)]
      names_path = os.path.join(root, "motion_names.json")
      names = None
      if os.path.exists(names_path):
        with open(names_path) as fh:
          names = json.load(fh)
    else:
      root = ""
      clips = [dict(np.load(path)) for path in files]
      clip_joints = [clip["joint_names"].tolist() for clip in clips]
      motion_ends = np.cumsum([clip["joint_pos"].shape[0] for clip in clips])
      names = [os.path.splitext(os.path.basename(p))[0] for p in files]

    for joints in clip_joints:
      if joints != fk.joint_names:
        raise ValueError(f"motion joints {joints} are not the robot's {fk.joint_names}")

    def field(name: str) -> np.ndarray:
      if packed:
        return np.load(os.path.join(root, f"{name}.npy"))
      return np.concatenate([clip[name] for clip in clips])

    def to_device(arr: np.ndarray) -> torch.Tensor:
      return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    self.root_pos_w = to_device(field("root_pos_w"))
    self.root_quat_w = to_device(field("root_quat_w"))
    self.root_lin_vel_w = to_device(field("root_lin_vel_w"))
    self.root_ang_vel_w = to_device(field("root_ang_vel_w"))
    self.joint_pos = to_device(field("joint_pos"))
    self.joint_vel = to_device(field("joint_vel"))
    self.fk = fk
    self.time_step_total = self.joint_pos.shape[0]
    self.motion_ends = torch.as_tensor(motion_ends, dtype=torch.long, device=device)
    if names is None:
      names = [f"clip_{i}" for i in range(len(self.motion_ends))]
    self.motion_names: list[str] = names

  def body_state(
    self, frames: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """World positions, wxyz quaternions, linear and angular velocities of the
    tracked bodies at ``frames``, without env origins."""
    return self.fk(
      self.root_pos_w[frames],
      self.root_quat_w[frames],
      self.joint_pos[frames],
      self.root_lin_vel_w[frames],
      self.root_ang_vel_w[frames],
      self.joint_vel[frames],
    )


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    fk = ForwardKinematics(
      env.sim.mj_model,
      self.robot.indexing.body_ids,
      self.robot.indexing.joint_q_adr,
      self.body_indexes,
      device=self.device,
    )
    self.motion = MotionLoader(self.cfg.motion_file, fk, device=self.device)
    # One fused kernel per call; env subsets of two or more share one graph.
    self._body_state = torch.compile(
      self.motion.body_state,
      dynamic=True,
      options={"max_fusion_size": 512},
      disable=not self.device.startswith("cuda"),
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # The tracked bodies' reference states at ``time_steps``, without env origins.
    (
      self._body_pos_w,
      self._body_quat_w,
      self._body_lin_vel_w,
      self._body_ang_vel_w,
    ) = self._body_state(self.time_steps)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (0.2 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    # Per-bin distribution for sampling_mode="external"; the runner pushes this in
    # via set_sampling_distribution(). Uniform until the first update.
    self._external_probs = torch.full(
      (self.bin_count,), 1.0 / self.bin_count, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self._body_pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w[:, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _sample_bins(self, env_ids: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Draw reset frames from a normalized per-bin distribution."""
    sampled_bins = torch.multinomial(probs, len(env_ids), replacement=True)
    # Integer, multiply-first bin->frame map. With ~1M bins the old float32
    # ``ratio * total`` lost precision, and a multinomial overflow to bin_count
    # could push the frame index out of range. int64 multiply-first is exact,
    # cannot overflow (b*total < 1e14, well under int64 max), and stays bounded.
    total = self.motion.time_step_total - 2
    b = sampled_bins.clamp_(max=self.bin_count - 1)
    lo = (b * total) // self.bin_count
    span = (((b + 1) * total) // self.bin_count - lo).clamp_(min=1)
    jitter = (
      sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * span
    ).long()
    entropy = -(probs * (probs + 1e-12).log()).sum()
    self.metrics["sampling_entropy"][:] = (
      entropy / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    )
    pmax, imax = probs.max(dim=0)
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
    return (lo + jitter).clamp_(max=total)

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample. Clip the per-bin failure count at adaptive_clip*mean so no single
    # bin can dominate the distribution (prevents runaway concentration/forgetting).
    failed = self.bin_failed_count
    if self.cfg.adaptive_clip > 0:
      failed = torch.clamp(failed, max=self.cfg.adaptive_clip * failed.mean())
    sampling_probabilities = failed + self.cfg.adaptive_uniform_ratio / float(
      self.bin_count
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    return self._sample_bins(env_ids, sampling_probabilities)

  def _uniform_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.
    return torch.randint(
      0, self.motion.time_step_total - 1, (len(env_ids),), device=self.device
    )

  def set_sampling_distribution(self, probs: torch.Tensor) -> None:
    """Set the per-bin reset distribution used by ``sampling_mode='external'``.

    ``probs`` must have length ``bin_count``; it is moved to device, clamped
    non-negative and renormalized. Invalid inputs are ignored so the previous
    (or uniform) distribution stays in effect.
    """
    probs = probs.detach().to(self.device, dtype=torch.float).clamp_min(0.0)
    total = probs.sum()
    if probs.numel() != self.bin_count or not torch.isfinite(total) or total <= 0:
      return
    self._external_probs = probs / total

  def _external_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Sample reset frames from the externally-provided distribution."""
    return self._sample_bins(env_ids, self._external_probs)

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      frames = torch.zeros_like(env_ids)
    elif self.cfg.sampling_mode == "uniform":
      frames = self._uniform_sampling(env_ids)
    elif self.cfg.sampling_mode == "external":
      frames = self._external_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      frames = self._adaptive_sampling(env_ids)

    on_last = torch.isin(frames, self.motion.motion_ends - 1)
    frames = torch.where(on_last, frames - 1, frames)
    self.set_time_steps(env_ids, frames)

    root_pos = self.motion.root_pos_w[frames] + self._env.scene.env_origins[env_ids]
    root_ori = self.motion.root_quat_w[frames]
    root_lin_vel = self.motion.root_lin_vel_w[frames]
    root_ang_vel = self.motion.root_ang_vel_w[frames]

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._pending_forward = True

  def set_time_steps(self, env_ids: torch.Tensor | slice, frames: torch.Tensor) -> None:
    """Move ``env_ids`` to ``frames`` and compute their tracked bodies there."""
    self.time_steps[env_ids] = frames
    (
      self._body_pos_w[env_ids],
      self._body_quat_w[env_ids],
      self._body_lin_vel_w[env_ids],
      self._body_ang_vel_w[env_ids],
    ) = self._body_state(frames)

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    """
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    # Reset-path resamples are followed by the env's own forward(); only
    # compute-path resamples (wraparound or timer expiry) need our refresh.
    self._pending_forward = False
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None):
    ids = slice(None) if env_ids is None else env_ids
    self.time_steps[ids] += 1
    wrap_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)
    self.set_time_steps(ids, self.time_steps[ids])

    # _resample_command writes qpos/qvel but does not refresh derived
    # quantities; forward() so update_relative_body_poses reads the
    # post-teleport robot anchor instead of the stale pre-resample pose.
    # Covers both the wraparound above and a timer-expiry resample that ran
    # in compute before this call.
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()

    self.update_relative_body_poses()

    # Fold failure counts into the EMA only on the per-step update so
    # manual resets do not decay it faster.
    if env_ids is None and self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        frame = self.time_steps[batch]
        root_pos = self.motion.root_pos_w[frame] + self._env.scene.env_origins[batch]
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = root_pos.cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.motion.root_quat_w[frame].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_entity(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = int(self.motion.time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        env_ids = torch.tensor([get_env_idx()], device=self.device)
        self.set_time_steps(env_ids, torch.full_like(env_ids, int(scrubber.value)))
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    # reset_to_frame writes qpos/qvel; forward so update_relative_body_poses
    # reads the scrubbed pose instead of the stale pre-scrub kinematics.
    self._env.sim.forward()
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Reset to exact reference state at a specific frame.

    Like ``_resample_command`` but deterministic: no random
    perturbations to pose, velocity, or joint positions.
    """
    frames = torch.full_like(env_ids, frame)
    self.set_time_steps(env_ids, frames)
    self._write_reference_state_to_sim(
      env_ids,
      self.motion.root_pos_w[frames] + self._env.scene.env_origins[env_ids],
      self.motion.root_quat_w[frames],
      self.motion.root_lin_vel_w[frames],
      self.motion.root_ang_vel_w[frames],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str | list[str] = ""
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  adaptive_clip: float = 100.0
  sampling_mode: Literal["adaptive", "uniform", "start", "external"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
