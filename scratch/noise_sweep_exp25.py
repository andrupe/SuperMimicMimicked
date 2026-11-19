"""Sweep observation noise multiplier on exp25_50k (100 Hz, stride=2) until survival < 100%."""

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


BASE_NOISE = {
  "root_pos": 0.0075,
  "projected_gravity": 0.01,
  "body_lin_vel": 0.02,
  "body_ang_vel": 0.025,
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

  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.decimation = 2  # 100 Hz
  env_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50000trajs_clean_yaw.npz"
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

  obs_mgr = env.unwrapped.observation_manager
  actor_cfgs = obs_mgr._group_obs_term_cfgs["actor"]
  actor_names = obs_mgr._group_obs_term_names["actor"]

  out_path = Path("logs/noise_sweep_exp25.json")
  results = json.loads(out_path.read_text()) if out_path.exists() else []
  done_scales = {r["noise_scale"] for r in results}
  scales = [20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]
  for sc in scales:
    if sc in done_scales:
      continue
    for term_name, term_cfg in zip(actor_names, actor_cfgs):
      if term_name in BASE_NOISE:
        b = BASE_NOISE[term_name] * sc
        term_cfg.noise = Unoise(n_min=-b, n_max=b)

    rng = np.random.default_rng(20261113)
    fam_rmse, fam_surv = {}, {}
    for fam in sorted(by_family.keys()):
      chosen = rng.choice(by_family[fam], size=3, replace=False)
      runs = [rollout_100hz(actor, env, cmd_term.trajectories[int(tid)], cmd_term, int(tid), stride=2, device=device) for tid in chosen]
      fam_rmse[fam] = round(float(np.mean([r["pos_rmse_cm"] for r in runs])), 2)
      fam_surv[fam] = round(float(np.mean([r["survival_ratio"] for r in runs]) * 100.0), 1)

    oof_runs = [rollout_100hz(actor, env, cmd_term.trajectories[tid], cmd_term, tid, stride=2, device=device) for _, tid in oof_ids]
    mean_rmse = round(float(np.mean(list(fam_rmse.values()))), 2)
    mean_surv = round(float(np.mean(list(fam_surv.values()))), 1)
    oof_rmse = round(float(np.mean([r["pos_rmse_cm"] for r in oof_runs])), 2)
    oof_surv = round(float(np.mean([r["survival_ratio"] * 100.0 for r in oof_runs])), 1)

    failed_fams = {k: v for k, v in fam_surv.items() if v < 100.0}
    rec = {
      "noise_scale": sc,
      "noise_bounds": {k: round(v * sc, 5) for k, v in BASE_NOISE.items()},
      "mean_pos_rmse_cm": mean_rmse,
      "mean_survival_pct": mean_surv,
      "oof_mean_pos_rmse_cm": oof_rmse,
      "oof_mean_survival_pct": oof_surv,
      "failed_families": failed_fams,
    }
    results.append(rec)
    print(
      f"[scale={sc:4.1f}x] pos=±{BASE_NOISE['root_pos']*sc*100:.2f}cm vel=±{BASE_NOISE['body_lin_vel']*sc:.3f}m/s "
      f"gyro=±{BASE_NOISE['body_ang_vel']*sc:.3f}rad/s | 17-fam: {mean_rmse:5.2f}cm ({mean_surv:5.1f}%) | "
      f"3-OOF: {oof_rmse:5.2f}cm ({oof_surv:5.1f}%) | fails={failed_fams}",
      flush=True,
    )
    Path("logs/noise_sweep_exp25.json").write_text(json.dumps(results, indent=2))
    if mean_surv < 100.0 or oof_surv < 100.0:
      break

  env.close()
  print("[DONE] Saved logs/noise_sweep_exp25.json")


if __name__ == "__main__":
  main()
