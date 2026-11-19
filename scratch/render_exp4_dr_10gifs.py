"""Render 10 3D quadcopter GIFs of exp4 (checkpoint_050000.pt) under DR into gif/exp4/dr/."""

from __future__ import annotations

import os
import sys
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np
import torch

import supermimic.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from scratch.eval_all_families_and_oof import build_oof_trajectories
from scratch.render_hard_quad3d_gifs import append_traj
from scratch.render_quadcopter_3d_gifs import rollout_and_render_quad


def find_family_tids(cmd_term, families: list[str]) -> dict[str, int]:
  tids = {}
  for idx, t in enumerate(cmd_term.trajectories):
    fam = t["task_type"]
    if fam in families and fam not in tids:
      if fam == "hover" or float(torch.std(t["lin_vel_b"]).item()) > 0.05:
        tids[fam] = idx
  for idx, t in enumerate(cmd_term.trajectories):
    fam = t["task_type"]
    if fam in families and fam not in tids:
      tids[fam] = idx
  return tids


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

  out_dir = Path("gif/exp4/dr")
  out_dir.mkdir(parents=True, exist_ok=True)

  torch.manual_seed(20261113)
  np.random.seed(20261113)

  # 1. 5 GIFs under Full Combined DR (play=False)
  full_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=False)
  full_cfg.scene.num_envs = 1
  full_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50000trajs.npz"
  full_cfg.commands["velocity"].corpus_ratio = 1.0
  full_cfg.commands["velocity"].rsi = False
  env_full = ManagerBasedRlEnv(cfg=full_cfg, device=device)
  cmd_full = env_full.unwrapped.command_manager.get_term("velocity")

  full_fams = ["figure8", "point_to_point", "crabbed_strafe", "split_s_dive"]
  tids_full = find_family_tids(cmd_full, full_fams)
  oof_name, oof_s, oof_c = build_oof_trajectories()[0]
  tids_full[oof_name] = append_traj(cmd_full, oof_name, oof_s, oof_c, device)

  for fam, tid in tids_full.items():
    rollout_and_render_quad(
      actor,
      env_full,
      cmd_full,
      tid,
      f"EXP4 Full DR: {fam}",
      out_dir / f"full_dr_{fam}.gif",
      device=device,
    )
  env_full.close()

  # 2. 5 GIFs under Moderate Realistic DR (20 ms lag, 0.85x sag, ±8% asym, wind, drag, noise)
  mod_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=True)
  mod_cfg.scene.num_envs = 1
  mod_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50000trajs.npz"
  mod_cfg.commands["velocity"].corpus_ratio = 1.0
  mod_cfg.commands["velocity"].rsi = False
  mod_cfg.observations["actor"].enable_corruption = True
  mod_cfg.actions["thrust"].enable_domain_randomization = True
  mod_cfg.actions["thrust"].tau_up_range = (0.015, 0.022)
  mod_cfg.actions["thrust"].tau_down_range = (0.018, 0.025)
  mod_cfg.actions["thrust"].thrust_scale_range = (0.85, 0.95)
  mod_cfg.actions["thrust"].sag_rate_range = (0.005, 0.010)
  mod_cfg.actions["thrust"].wind_xy_range = (-0.35, 0.35)
  mod_cfg.actions["thrust"].wind_z_range = (-0.10, 0.10)
  mod_cfg.events["mass"].params["ranges"] = (0.90, 1.10)
  mod_cfg.events["base_com"].params["ranges"] = {0: (-0.003, 0.003), 1: (-0.003, 0.003), 2: (-0.002, 0.002)}

  env_mod = ManagerBasedRlEnv(cfg=mod_cfg, device=device)
  cmd_mod = env_mod.unwrapped.command_manager.get_term("velocity")

  mod_fams = ["hover", "figure8", "slalom", "split_s_dive"]
  tids_mod = find_family_tids(cmd_mod, mod_fams)
  tids_mod[oof_name] = append_traj(cmd_mod, oof_name, oof_s, oof_c, device)

  for fam, tid in tids_mod.items():
    rollout_and_render_quad(
      actor,
      env_mod,
      cmd_mod,
      tid,
      f"EXP4 Mod DR (20ms): {fam}",
      out_dir / f"mod_dr_{fam}.gif",
      device=device,
    )
  env_mod.close()


if __name__ == "__main__":
  main()
