"""Evaluate exp4 model (checkpoint_050000.pt) with full Crazyflie 2.1 domain randomization enabled."""

from __future__ import annotations

import json
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
from scratch.eval_all_families_and_oof import build_oof_trajectories, rollout_trajectory


def main():
  device = "cuda"
  ckpt_path = "logs/crazyflie_17fam_23d_50k/checkpoint_050000.pt"
  ckpt = torch.load(ckpt_path, map_location=device)
  actor_sd = ckpt["actor"]
  actor = QuadcopterPolicy(
    obs_dim=int(actor_sd["mlp.0.weight"].shape[1]),
    action_dim=4,
    hidden_dims=tuple(ckpt["actor_hidden_dims"]),
  ).to(device)
  actor.load_state_dict(actor_sd)
  actor.eval()

  # play=False enables full Crazyflie 2.1 DR: motor lag, battery sag, rotor asymmetry,
  # aero drag + wind, mass/inertia/CoM DR, velocity pushes, and actor sensor noise.
  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50000trajs.npz"
  env_cfg.commands["velocity"].corpus_ratio = 1.0
  env_cfg.commands["velocity"].rsi = False
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term = env.unwrapped.command_manager.get_term("velocity")

  # Pick 1 trajectory per family (matching eval_all_families_and_oof.py)
  family_tids: dict[str, int] = {}
  for idx, t in enumerate(cmd_term.trajectories):
    fam = t["task_type"]
    if fam not in family_tids:
      if fam == "hover" or float(torch.std(t["lin_vel_b"]).item()) > 0.05:
        family_tids[fam] = idx
  for idx, t in enumerate(cmd_term.trajectories):
    fam = t["task_type"]
    if fam not in family_tids:
      family_tids[fam] = idx

  torch.manual_seed(20261113)
  np.random.seed(20261113)

  print(f"\n=== EXP4 Model ({ckpt_path}) UNDER FULL DOMAIN RANDOMIZATION ===")
  fam_results = []
  for fam in sorted(family_tids.keys()):
    tid = family_tids[fam]
    traj = cmd_term.trajectories[tid]
    res = rollout_trajectory(actor, env, traj, cmd_term, tid, device=device, gif_path=None, title_name=fam)
    res["family"] = fam
    fam_results.append(res)
    print(f"  [CORPUS+DR] {fam:22s} | Surv: {res['survival_ratio']*100:5.1f}% | Pos RMSE: {res['pos_rmse_cm']:6.2f} cm | Vel RMSE: {res['vel_rmse_ms']:5.2f} m/s")

  # Append & evaluate the 3 OOF trajectories under full DR
  oof_specs = build_oof_trajectories()
  oof_results = []
  for name, s_np, c_np in oof_specs:
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

    traj = cmd_term.trajectories[tid]
    res = rollout_trajectory(
      actor,
      env,
      traj,
      cmd_term,
      tid,
      device=device,
      gif_path=Path(f"gif/exp4/{name}_with_dr.gif"),
      title_name=f"{name} (DR)",
    )
    res["name"] = name
    oof_results.append(res)
    print(f"  [OOF+DR]    {name:22s} | Surv: {res['survival_ratio']*100:5.1f}% | Pos RMSE: {res['pos_rmse_cm']:6.2f} cm | Vel RMSE: {res['vel_rmse_ms']:5.2f} m/s")

  out = {"checkpoint": ckpt_path, "domain_randomization": True, "corpus_families": fam_results, "oof_trajectories": oof_results}
  out_path = Path("logs/eval_exp4_with_dr.json")
  out_path.write_text(json.dumps(out, indent=2))
  print(f"\nSaved results to {out_path}")
  env.close()


if __name__ == "__main__":
  main()
