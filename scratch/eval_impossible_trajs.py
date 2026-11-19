"""Evaluate checkpoint_050000.pt on 3 physically impossible trajectories and render 3D quadcopter GIFs."""

from __future__ import annotations

import os
import sys
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import mujoco

import supermimic.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from supermimic.tasks.crazyflie_mdp import CorpusTrajectoryCommand
from scratch.render_hard_quad3d_gifs import append_traj
from scratch.render_quadcopter_3d_gifs import rollout_and_render_quad


def build_impossible_trajectories(dt: float = 0.02) -> list[tuple[str, str, np.ndarray, np.ndarray, float]]:
  """Return (name, title, states(T,13), ctrls(T-1,4), max_tilt_deg)."""
  trajs = []

  # 1. Instantaneous Teleport Jump (infinite velocity/acceleration discontinuity)
  dur1 = 3.2
  T1 = int(dur1 / dt) + 1
  s1 = np.zeros((T1, 13), dtype=np.float32)
  s1[:, 3] = 1.0  # upright quat [1,0,0,0]
  for i in range(T1):
    t = i * dt
    if t < 0.8:
      s1[i, :3] = [0.0, 0.0, 1.2]
    elif t < 2.0:
      s1[i, :3] = [1.4, -0.9, 1.7]
    else:
      s1[i, :3] = [-0.9, 1.0, 1.0]
  c1 = np.full((T1 - 1, 4), 0.265, dtype=np.float32)
  trajs.append((
    "impossible_teleport_jump",
    "Impossible 1: Teleport Step Jump",
    s1,
    c1,
    80.0,
  ))

  # 2. ~3.1G High-Speed Circle with 0-deg Flat Attitude (exceeds TWR & tilt physics)
  dur2 = 3.0
  T2 = int(dur2 / dt) + 1
  t2 = np.linspace(0.0, dur2, T2)
  R = 0.9
  omega = 5.8  # v = 5.22 m/s, a_c = R * omega^2 = 30.28 m/s^2 (~3.09 g)
  s2 = np.zeros((T2, 13), dtype=np.float32)
  s2[:, 0] = R * np.cos(omega * t2) - R  # starts at (0, 0, 1.3)
  s2[:, 1] = R * np.sin(omega * t2)
  s2[:, 2] = 1.3
  s2[:, 3] = 1.0  # flat quat [1,0,0,0]
  s2[:, 7] = -R * omega * np.sin(omega * t2)
  s2[:, 8] = R * omega * np.cos(omega * t2)
  c2 = np.full((T2 - 1, 4), 0.265, dtype=np.float32)
  trajs.append((
    "impossible_3g_flat_circle",
    "Impossible 2: 3.1G Flat Circle (5.2 m/s)",
    s2,
    c2,
    80.0,
  ))

  # 3. Upside-Down Gravity-Defying Climb (roll=180 deg, vz=+0.6 m/s)
  dur3 = 2.5
  T3 = int(dur3 / dt) + 1
  t3 = np.linspace(0.0, dur3, T3)
  s3 = np.zeros((T3, 13), dtype=np.float32)
  s3[:, 2] = 1.2 + 0.6 * t3
  s3[:, 4] = 1.0  # quat [0, 1, 0, 0] -> 180 deg roll (inverted)
  # body z points down in world frame, so vz_w = +0.6 => vb_z = -0.6
  s3[:, 9] = -0.6
  c3 = np.full((T3 - 1, 4), 0.265, dtype=np.float32)
  trajs.append((
    "impossible_inverted_climb",
    "Impossible 3: Inverted Upward Climb",
    s3,
    c3,
    180.0,
  ))

  return trajs


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

  out_dir = Path("gif/exp4")
  out_dir.mkdir(parents=True, exist_ok=True)
  trajs = build_impossible_trajectories()

  for name, title, s_np, c_np, max_tilt in trajs:
    env_cfg = load_env_cfg("SuperMimic-Crazyflie-Corpus", play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50trajs.npz"
    env_cfg.terminations["tilt"].params["max_tilt_angle_deg"] = max_tilt
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    cmd_term: CorpusTrajectoryCommand = env.unwrapped.command_manager.get_term("velocity")
    tid = append_traj(cmd_term, name, s_np, c_np, device)
    rollout_and_render_quad(actor, env, cmd_term, tid, title, out_dir / f"quad3d_{name}.gif", device=device)
    env.close()


if __name__ == "__main__":
  main()
