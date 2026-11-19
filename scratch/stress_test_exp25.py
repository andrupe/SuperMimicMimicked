"""Stress-test exp25_50k (100 Hz, stride=2) under harsher out-of-distribution DR regimes."""

from __future__ import annotations

import json
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import supermimic.tasks
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from scratch.eval_100hz import rollout_100hz
from scratch.eval_all_families_and_oof import build_oof_trajectories


def apply_regime(env_cfg, regime: str):
  # Restore original exp17 observation noise for both stress regimes
  actor_terms = env_cfg.observations["actor"].terms
  actor_terms["root_pos"].noise = Unoise(n_min=-0.015, n_max=0.015)
  actor_terms["projected_gravity"].noise = Unoise(n_min=-0.02, n_max=0.02)
  actor_terms["body_lin_vel"].noise = Unoise(n_min=-0.04, n_max=0.04)
  actor_terms["body_ang_vel"].noise = Unoise(n_min=-0.05, n_max=0.05)

  act_cfg = env_cfg.actions["thrust"]
  if regime == "exp17_dr":
    env_cfg.events["base_com"].params["ranges"] = {0: (-0.002, 0.002), 1: (-0.002, 0.002), 2: (-0.0015, 0.0015)}
    env_cfg.events["mass"].params["ranges"] = (0.94, 1.075)
    act_cfg.tau_up_range = (0.020, 0.0325)
    act_cfg.tau_down_range = (0.0225, 0.0375)
    act_cfg.thrust_scale_range = (0.94, 1.025)
    act_cfg.sag_rate_range = (0.0, 0.005)
    act_cfg.rotor_gain_range = (0.97, 1.03)
    act_cfg.wind_xy_range = (-0.20, 0.20)
    act_cfg.wind_z_range = (-0.075, 0.075)
    act_cfg.drag_scale_range = (0.90, 1.15)
  elif regime == "extreme_dr":
    env_cfg.events["base_com"].params["ranges"] = {0: (-0.004, 0.004), 1: (-0.004, 0.004), 2: (-0.0025, 0.0025)}
    env_cfg.events["mass"].params["ranges"] = (0.88, 1.15)
    act_cfg.tau_up_range = (0.020, 0.040)
    act_cfg.tau_down_range = (0.022, 0.045)
    act_cfg.thrust_scale_range = (0.85, 1.02)
    act_cfg.sag_rate_range = (0.0, 0.010)
    act_cfg.rotor_gain_range = (0.95, 1.05)
    act_cfg.wind_xy_range = (-0.30, 0.30)
    act_cfg.wind_z_range = (-0.10, 0.10)
    act_cfg.drag_scale_range = (0.85, 1.25)


def run_regime(actor, regime: str, trajs_per_family: int = 5, device: str = "cuda") -> dict:
  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.decimation = 2  # 100 Hz
  env_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50000trajs_clean_yaw.npz"
  apply_regime(env_cfg, regime)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term = env.unwrapped.command_manager.get_term("velocity")
  by_family: dict[str, list[int]] = {}
  for idx, t in enumerate(cmd_term.trajectories):
    by_family.setdefault(str(t["task_type"]), []).append(idx)

  oof_ids = []
  for name, s_np, c_np in build_oof_trajectories():
    s_t = torch.from_numpy(s_np[:, :13]).float().to(device)
    c_t = torch.from_numpy(c_np).float().to(device)
    new_tid = cmd_term.num_trajs
    s_start, c_start = int(cmd_term._s_off[-1]), int(cmd_term._c_off[-1])
    cmd_term._states = torch.cat([cmd_term._states, s_t], dim=0)
    cmd_term._ctrls = torch.cat([cmd_term._ctrls, c_t], dim=0)
    cmd_term._s_off = torch.cat([cmd_term._s_off, torch.tensor([s_start + len(s_t)], dtype=torch.long, device=device)])
    cmd_term._c_off = torch.cat([cmd_term._c_off, torch.tensor([c_start + len(c_t)], dtype=torch.long, device=device)])
    cmd_term._traj_len = cmd_term._c_off[1:] - cmd_term._c_off[:-1]
    cmd_term._tasks.append(name)
    cmd_term.num_trajs += 1
    oof_ids.append((name, new_tid))

  rng = np.random.default_rng(20261113)
  family_results = {}
  for fam in sorted(by_family.keys()):
    chosen = rng.choice(by_family[fam], size=min(trajs_per_family, len(by_family[fam])), replace=False)
    runs = [rollout_100hz(actor, env, cmd_term.trajectories[int(tid)], cmd_term, int(tid), stride=2, device=device) for tid in chosen]
    family_results[fam] = {
      "mean_pos_rmse_cm": round(float(np.mean([r["pos_rmse_cm"] for r in runs])), 2),
      "mean_survival_pct": round(float(np.mean([r["survival_ratio"] for r in runs]) * 100.0), 1),
    }

  oof_results = []
  for name, tid in oof_ids:
    r = rollout_100hz(actor, env, cmd_term.trajectories[tid], cmd_term, tid, stride=2, device=device)
    r["name"] = name
    oof_results.append(r)

  env.close()
  return {
    "regime": regime,
    "overall_family_mean_pos_rmse_cm": round(float(np.mean([v["mean_pos_rmse_cm"] for v in family_results.values()])), 2),
    "overall_family_mean_survival_pct": round(float(np.mean([v["mean_survival_pct"] for v in family_results.values()])), 1),
    "oof_mean_pos_rmse_cm": round(float(np.mean([r["pos_rmse_cm"] for r in oof_results])), 2),
    "oof_mean_survival_pct": round(float(np.mean([r["survival_ratio"] * 100.0 for r in oof_results])), 1),
    "families": family_results,
    "out_of_fold": oof_results,
  }


def main():
  device = "cuda"
  ckpt = torch.load("logs/crazyflie_exp25_50k/checkpoint_050000.pt", map_location=device, weights_only=False)
  actor_sd = ckpt["actor"]
  actor = QuadcopterPolicy(
    obs_dim=int(actor_sd["mlp.0.weight"].shape[1]),
    action_dim=4,
    hidden_dims=tuple(ckpt["actor_hidden_dims"]),
  ).to(device)
  actor.load_state_dict(actor_sd)
  actor.eval()

  out = {}
  for regime in ["exp17_dr", "extreme_dr"]:
    res = run_regime(actor, regime, trajs_per_family=5, device=device)
    out[regime] = res
    print(
      f"[{regime}] 17-fam: {res['overall_family_mean_pos_rmse_cm']:.2f} cm ({res['overall_family_mean_survival_pct']:.1f}%) | "
      f"3-OOF: {res['oof_mean_pos_rmse_cm']:.2f} cm ({res['oof_mean_survival_pct']:.1f}%)",
      flush=True,
    )

  Path("logs/stress_test_exp25.json").write_text(json.dumps(out, indent=2))
  print("[DONE] Saved logs/stress_test_exp25.json")


if __name__ == "__main__":
  main()
