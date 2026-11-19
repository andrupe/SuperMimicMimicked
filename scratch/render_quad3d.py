"""Render one quad3d GIF per trajectory family (17 in-fold + 3 OOF) under full DR."""

from __future__ import annotations

import argparse
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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt", type=str, required=True)
  parser.add_argument("--out-dir", type=str, required=True)
  parser.add_argument("--label", type=str, required=True)
  parser.add_argument("--device", type=str, default="cuda")
  args = parser.parse_args()

  device = args.device
  ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
  actor_sd = ckpt["actor"]
  actor = QuadcopterPolicy(
    obs_dim=int(actor_sd["mlp.0.weight"].shape[1]),
    action_dim=4,
    hidden_dims=tuple(ckpt["actor_hidden_dims"]),
  ).to(device)
  actor.load_state_dict(actor_sd)
  actor.eval()

  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50000trajs.npz"
  env_cfg.commands["velocity"].corpus_ratio = 1.0
  env_cfg.commands["velocity"].rsi = False
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term = env.unwrapped.command_manager.get_term("velocity")

  by_family: dict[str, list[int]] = {}
  for idx, t in enumerate(cmd_term.trajectories):
    by_family.setdefault(str(t["task_type"]), []).append(idx)

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  rng = np.random.default_rng(42)
  torch.manual_seed(42)
  np.random.seed(42)

  for fam in sorted(by_family.keys()):
    tid = int(rng.choice(by_family[fam]))
    out_gif = out_dir / f"quad3d_{fam}.gif"
    print(f"Rendering {out_gif} (traj_id={tid})...", flush=True)
    rollout_and_render_quad(actor, env, cmd_term, tid, f"{args.label} DR: {fam}", out_gif, device=device)

  for key, s_np, c_np in build_oof_trajectories(dt=0.02):
    tid = append_traj(cmd_term, key, s_np, c_np, device=device)
    out_gif = out_dir / f"quad3d_{key}.gif"
    print(f"Rendering {out_gif} (OOF traj_id={tid})...", flush=True)
    rollout_and_render_quad(actor, env, cmd_term, tid, f"{args.label} DR: {key}", out_gif, device=device)

  env.close()
  print(f"[DONE] Rendered all quad3d GIFs to {out_dir}")


if __name__ == "__main__":
  main()
