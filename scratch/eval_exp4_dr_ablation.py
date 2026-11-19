"""Test exp4 (checkpoint_050000.pt) under individual & mild DR conditions to find where it still performs well."""

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
from scratch.eval_all_families_and_oof import build_oof_trajectories, rollout_trajectory


def run_case(label: str, actor, env_cfg, device: str):
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term = env.unwrapped.command_manager.get_term("velocity")

  # Pick hover, figure8, and append oof_compound_lissajous
  tids = {}
  for idx, t in enumerate(cmd_term.trajectories):
    fam = t["task_type"]
    if fam in ("hover", "figure8") and fam not in tids:
      if fam == "hover" or float(torch.std(t["lin_vel_b"]).item()) > 0.05:
        tids[fam] = idx

  name, s_np, c_np = build_oof_trajectories()[0]
  s_t = torch.from_numpy(s_np[:, :13]).float().to(device)
  c_t = torch.from_numpy(c_np).float().to(device)
  oof_tid = cmd_term.num_trajs
  s_start, c_start = int(cmd_term._s_off[-1]), int(cmd_term._c_off[-1])
  cmd_term._states = torch.cat([cmd_term._states, s_t], dim=0)
  cmd_term._ctrls = torch.cat([cmd_term._ctrls, c_t], dim=0)
  cmd_term._s_off = torch.cat([cmd_term._s_off, torch.tensor([s_start + len(s_t)], dtype=torch.long, device=device)])
  cmd_term._c_off = torch.cat([cmd_term._c_off, torch.tensor([c_start + len(c_t)], dtype=torch.long, device=device)])
  cmd_term._traj_len = cmd_term._c_off[1:] - cmd_term._c_off[:-1]
  cmd_term._tasks.append(name)
  cmd_term.num_trajs += 1
  tids[name] = oof_tid

  print(f"\n--- {label} ---")
  for k, tid in tids.items():
    res = rollout_trajectory(actor, env, cmd_term.trajectories[tid], cmd_term, tid, device=device)
    print(f"  {k:24s} | Surv: {res['survival_ratio']*100:5.1f}% | Pos RMSE: {res['pos_rmse_cm']:6.2f} cm | Vel RMSE: {res['vel_rmse_ms']:5.2f} m/s")
  env.close()


def make_base_cfg():
  cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=True)
  cfg.scene.num_envs = 1
  cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50000trajs.npz"
  cfg.commands["velocity"].corpus_ratio = 1.0
  cfg.commands["velocity"].rsi = False
  return cfg


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

  torch.manual_seed(42)
  np.random.seed(42)

  # Case 1: Sensor Noise + Aero Drag + Wind (0 ms motor lag)
  c1 = make_base_cfg()
  c1.observations["actor"].enable_corruption = True
  c1.actions["thrust"].enable_domain_randomization = True
  c1.actions["thrust"].tau_up_range = (0.0, 0.0)
  c1.actions["thrust"].tau_down_range = (0.0, 0.0)
  c1.actions["thrust"].thrust_scale_range = (1.0, 1.0)
  c1.actions["thrust"].sag_rate_range = (0.0, 0.0)
  c1.actions["thrust"].rotor_gain_range = (1.0, 1.0)
  run_case("Case 1: Sensor Noise + Drag + Wind (0ms lag)", actor, c1, device)

  # Case 2: Battery Sag (0.85x thrust) + Rotor Asymmetry (±8%) + Mass/CoM DR (0ms lag)
  c2 = make_base_cfg()
  c2.actions["thrust"].enable_domain_randomization = True
  c2.actions["thrust"].tau_up_range = (0.0, 0.0)
  c2.actions["thrust"].tau_down_range = (0.0, 0.0)
  c2.actions["thrust"].thrust_scale_range = (0.85, 0.85)
  c2.actions["thrust"].sag_rate_range = (0.01, 0.01)
  c2.actions["thrust"].wind_xy_range = (0.0, 0.0)
  c2.actions["thrust"].wind_z_range = (0.0, 0.0)
  c2.events["mass"].params["ranges"] = (0.90, 1.12)
  c2.events["base_com"].params["ranges"] = {0: (-0.004, 0.004), 1: (-0.004, 0.004), 2: (-0.003, 0.003)}
  run_case("Case 2: 0.85x Battery Sag + ±8% Rotor Asym + Mass/CoM DR (0ms lag)", actor, c2, device)

  # Case 3: Small Motor Lag (10 ms) + Drag (no wind/sag)
  c3 = make_base_cfg()
  c3.actions["thrust"].enable_domain_randomization = False
  c3.actions["thrust"].nominal_tau = 0.010
  run_case("Case 3: 10 ms Motor Lag + Nominal Aero Drag", actor, c3, device)

  # Case 4: Realistic 25 ms Motor Lag alone
  c4 = make_base_cfg()
  c4.actions["thrust"].enable_domain_randomization = False
  c4.actions["thrust"].nominal_tau = 0.025
  run_case("Case 4: 25 ms Motor Lag + Nominal Aero Drag", actor, c4, device)


if __name__ == "__main__":
  main()
