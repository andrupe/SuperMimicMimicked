"""Render 3 harder 3D quadcopter-body GIFs: High-Speed Corkscrew, Extreme Recovery, and a 360-deg Pitch Flip."""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
from pathlib import Path

import numpy as np
import torch

import supermimic.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from supermimic.tasks.crazyflie_mdp import CorpusTrajectoryCommand
from scratch.generate_ultra_corpus_50000 import _solve_from_pos_yaw
from scratch.render_quadcopter_3d_gifs import rollout_and_render_quad


def build_hard_trajectories(dt: float = 0.02) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
  np.random.seed(20261113)
  hard = []

  # 1. Hard High-Speed Tight Corkscrew Slalom (aggressive banking up to ~60-65 deg)
  dur1 = 3.8
  t1 = np.linspace(0, dur1, int(dur1 / dt) + 1)
  pos1 = np.stack([
    1.15 * np.sin(2.3 * t1) + 0.35 * t1 - 0.6,
    1.05 * np.cos(2.7 * t1),
    1.30 + 0.45 * np.sin(3.1 * t1),
  ], axis=-1)
  vel1 = np.gradient(pos1, dt, axis=0)
  yaw1 = np.unwrap(np.arctan2(vel1[:, 1] + 1e-4, vel1[:, 0] + 1e-4))
  s1, c1, ok1 = _solve_from_pos_yaw(pos1, yaw1, dt, max_tilt_deg=67.5)
  assert ok1, "Hard corkscrew failed flatness_solve"
  hard.append(("hard_highspeed_corkscrew", "Hard High-Speed Corkscrew", s1, c1))

  # 2. Hard Extreme Perturbed Attitude & Velocity Recovery
  dur2 = 2.6
  t2 = np.linspace(0, dur2, int(dur2 / dt) + 1)
  tau = t2 / dur2
  p_goal = np.array([0.0, 0.0, 1.35])
  p0 = np.array([-0.65, 0.55, 1.15])
  v0 = np.array([1.75, -1.55, -0.75])
  a0 = np.array([-5.8, 5.2, 2.8])
  T = dur2
  c0, c1_p, c2_p = p0, v0 * T, 0.5 * a0 * (T ** 2)
  c3_p = 10.0 * (p_goal - p0) - 6.0 * v0 * T - 1.5 * a0 * (T ** 2)
  c4_p = -15.0 * (p_goal - p0) + 8.0 * v0 * T + 1.5 * a0 * (T ** 2)
  c5_p = 6.0 * (p_goal - p0) - 3.0 * v0 * T - 0.5 * a0 * (T ** 2)
  pos2 = (
    c0[None, :] + c1_p[None, :] * tau[:, None] + c2_p[None, :] * (tau**2)[:, None]
    + c3_p[None, :] * (tau**3)[:, None] + c4_p[None, :] * (tau**4)[:, None] + c5_p[None, :] * (tau**5)[:, None]
  )
  yaw2 = 0.8 + 1.8 * t2 * np.exp(-2.2 * t2)
  s2, c2, ok2 = _solve_from_pos_yaw(pos2, yaw2, dt, max_tilt_deg=67.5)
  assert ok2, "Hard extreme recovery failed flatness_solve"
  hard.append(("hard_extreme_recovery", "Hard Extreme Recovery", s2, c2))

  # 3. 360-Degree Pitch Flip (out-of-distribution aerobatic maneuver from 90-traj corpus)
  d90 = np.load("corpus/crazyflie_corpus_all90trajs.npz")
  flip_idx = 62
  if "state_offsets" in d90:
    s3 = d90["states"][d90["state_offsets"][flip_idx] : d90["state_offsets"][flip_idx + 1]]
    c3 = d90["ctrls"][d90["ctrl_offsets"][flip_idx] : d90["ctrl_offsets"][flip_idx + 1]]
  else:
    s3 = d90[f"state_{flip_idx}"]
    c3 = d90[f"ctrl_{flip_idx}"]
  hard.append(("hard_pitch_flip", "360-deg Pitch Flip (OOD)", s3, c3))

  return hard


def append_traj(cmd_term: CorpusTrajectoryCommand, name: str, s_np: np.ndarray, c_np: np.ndarray, device: str) -> int:
  s_t = torch.from_numpy(s_np[:, :13]).float().to(device)
  c_t = torch.from_numpy(c_np).float().to(device)
  tid = cmd_term.num_trajs
  s_start, c_start = int(cmd_term._s_off[-1]), int(cmd_term._c_off[-1])
  cmd_term._states = torch.cat([cmd_term._states, s_t], dim=0)
  cmd_term._ctrls = torch.cat([cmd_term._ctrls, c_t], dim=0)
  cmd_term._s_off = torch.cat([cmd_term._s_off, torch.tensor([s_start + len(s_t)], dtype=torch.long, device=device)])
  cmd_term._c_off = torch.cat([cmd_term._c_off, torch.tensor([c_start + len(c_t)], dtype=torch.long, device=device)])
  cmd_term._traj_len = cmd_term._c_off[1:] - cmd_term._c_off[:-1]
  cmd_term._tasks.append(name)
  cmd_term.num_trajs += 1
  return tid


def main():
  device = "cuda"
  ckpt = torch.load("logs/crazyflie_17fam_23d_50k/checkpoint_050000.pt", map_location=device)
  actor_sd = ckpt["actor"]
  actor = QuadcopterPolicy(
    obs_dim=int(actor_sd["mlp.0.weight"].shape[1]),
    action_dim=4,
    hidden_dims=tuple(ckpt["actor_hidden_dims"]),
  ).to(device)
  actor.load_state_dict(actor_sd)
  actor.eval()

  hard_trajs = build_hard_trajectories()
  out_dir = Path("gif/exp4")

  # First two with standard 80 deg tilt termination
  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Corpus", play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50trajs.npz"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term: CorpusTrajectoryCommand = env.unwrapped.command_manager.get_term("velocity")
  for name, title, s_np, c_np in hard_trajs[:2]:
    tid = append_traj(cmd_term, name, s_np, c_np, device)
    rollout_and_render_quad(actor, env, cmd_term, tid, title, out_dir / f"quad3d_{name}.gif", device=device)
  env.close()

  # Pitch flip with tilt termination disabled (180 deg) so we can see full flip / fall behavior
  env_cfg_flip = load_env_cfg("SuperMimic-Crazyflie-Corpus", play=True)
  env_cfg_flip.scene.num_envs = 1
  env_cfg_flip.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50trajs.npz"
  env_cfg_flip.terminations["tilt"].params["max_tilt_angle_deg"] = 180.0
  env_flip = ManagerBasedRlEnv(cfg=env_cfg_flip, device=device)
  cmd_flip: CorpusTrajectoryCommand = env_flip.unwrapped.command_manager.get_term("velocity")
  name, title, s_np, c_np = hard_trajs[2]
  tid = append_traj(cmd_flip, name, s_np, c_np, device)
  rollout_and_render_quad(actor, env_flip, cmd_flip, tid, title, out_dir / f"quad3d_{name}.gif", device=device)
  env_flip.close()


if __name__ == "__main__":
  main()
