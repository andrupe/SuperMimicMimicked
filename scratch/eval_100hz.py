"""Test running the 50 Hz-trained exp17 policy at 100 Hz (decimation=2, dt=0.01s) in MuJoCo."""

from __future__ import annotations

import argparse
import json
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from pathlib import Path

import numpy as np
import torch

import supermimic.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from scratch.eval_all_families_and_oof import build_oof_trajectories


def rollout_100hz(
  actor: QuadcopterPolicy,
  env: ManagerBasedRlEnv,
  traj: dict,
  cmd_term,
  traj_id: int,
  stride: int = 1,
  device: str = "cuda",
) -> dict:
  env.reset()
  cmd_term.set_trajectory(0, traj_id, t=0)
  H = int(traj["H"])
  total_steps = 2 * H

  q0 = traj["quat"][0].unsqueeze(0).to(device)
  vb0 = traj["lin_vel_b"][0].unsqueeze(0).to(device)
  wb0 = traj["ang_vel_b"][0].unsqueeze(0).to(device)
  vw0 = quat_apply(q0, vb0).squeeze(0)
  ww0 = quat_apply(q0, wb0).squeeze(0)
  origin = env.unwrapped.scene.env_origins[0]
  p0 = traj["pos"][0].to(device) + origin

  root_state0 = torch.cat([p0, q0.squeeze(0), vw0, ww0], dim=-1).unsqueeze(0)
  env.unwrapped.scene["robot"].write_root_state_to_sim(root_state0)
  env.unwrapped.sim.forward()
  env.unwrapped.observation_manager._obs_buffer = None
  obs = env.get_observations()

  history_length = max(1, int(actor.obs_dim) // 23)
  ring_len = (history_length - 1) * stride + 1
  obs_ring = obs["actor"].unsqueeze(1).repeat(1, ring_len, 1)  # (1, ring_len, 23)

  ref_pos_hist, act_pos_hist = [], []
  term_step = total_steps

  with torch.inference_mode():
    for k in range(total_steps):
      traj_t = min(k // 2, H - 1)
      # Pin 50 Hz trajectory index at real-time 1x speed (advance every 2nd 100 Hz step)
      cmd_term.set_trajectory(0, traj_id, t=traj_t)
      env.unwrapped.observation_manager._obs_buffer = None
      obs = env.get_observations()
      obs_ring[:, -1, :] = obs["actor"]

      if history_length > 1:
        idx = list(range(ring_len - 1 - (history_length - 1) * stride, ring_len, stride))
        policy_in = obs_ring[:, idx, :].reshape(1, -1)
      else:
        policy_in = obs["actor"]
      action = actor(policy_in)

      qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy()
      act_pos_hist.append(qpos[:3].copy())
      ref_pos_hist.append(traj["pos"][traj_t].cpu().numpy().copy())

      obs, _, terminated, _, _ = env.step(action)
      if terminated.any():
        term_step = k + 1
        break
      obs_ring = torch.cat([obs_ring[:, 1:], obs["actor"].unsqueeze(1)], dim=1)

  act_p = np.array(act_pos_hist)
  ref_p = np.array(ref_pos_hist)
  pos_errs = np.linalg.norm(act_p - ref_p, axis=-1)
  return {
    "pos_rmse_cm": float(np.sqrt(np.mean(pos_errs ** 2)) * 100.0),
    "pos_max_cm": float(np.max(pos_errs) * 100.0),
    "survival_ratio": float(term_step / total_steps),
  }


def eval_mode(actor, corpus_path: str, play: bool, stride: int, trajs_per_family: int, seed: int, device: str) -> dict:
  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=play)
  env_cfg.scene.num_envs = 1
  env_cfg.decimation = 2  # 2 * 0.005s = 0.01s (100 Hz control)
  env_cfg.commands["velocity"].corpus_path = corpus_path
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term = env.unwrapped.command_manager.get_term("velocity")

  by_family: dict[str, list[int]] = {}
  for idx, t in enumerate(cmd_term.trajectories):
    by_family.setdefault(str(t["task_type"]), []).append(idx)

  oofs = build_oof_trajectories()
  oof_ids = []
  for name, s_np, c_np in oofs:
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

  rng = np.random.default_rng(seed)
  family_results = {}
  for fam in sorted(by_family.keys()):
    ids = by_family[fam]
    chosen = rng.choice(ids, size=min(trajs_per_family, len(ids)), replace=False)
    runs = [rollout_100hz(actor, env, cmd_term.trajectories[int(tid)], cmd_term, int(tid), stride=stride, device=device) for tid in chosen]
    family_results[fam] = {
      "mean_pos_rmse_cm": round(float(np.mean([r["pos_rmse_cm"] for r in runs])), 2),
      "mean_survival_pct": round(float(np.mean([r["survival_ratio"] for r in runs]) * 100.0), 1),
    }

  oof_results = []
  for name, tid in oof_ids:
    r = rollout_100hz(actor, env, cmd_term.trajectories[tid], cmd_term, tid, stride=stride, device=device)
    r["name"] = name
    oof_results.append(r)

  env.close()
  return {
    "mode": "nominal" if play else "full_dr",
    "stride": stride,
    "overall_family_mean_pos_rmse_cm": round(float(np.mean([v["mean_pos_rmse_cm"] for v in family_results.values()])), 2),
    "overall_family_mean_survival_pct": round(float(np.mean([v["mean_survival_pct"] for v in family_results.values()])), 1),
    "oof_mean_pos_rmse_cm": round(float(np.mean([r["pos_rmse_cm"] for r in oof_results])), 2),
    "oof_mean_survival_pct": round(float(np.mean([r["survival_ratio"] * 100.0 for r in oof_results])), 1),
    "families": family_results,
    "out_of_fold": oof_results,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt", type=str, default="logs/crazyflie_exp17_50k/checkpoint_050000.pt")
  parser.add_argument("--corpus-path", type=str, default="corpus/crazyflie_corpus_50000trajs.npz")
  parser.add_argument("--trajs-per-family", type=int, default=5)
  parser.add_argument("--out-json", type=str, default="logs/eval_exp17_100hz.json")
  parser.add_argument("--device", type=str, default="cuda")
  args = parser.parse_args()

  ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
  actor_sd = ckpt["actor"]
  actor = QuadcopterPolicy(
    obs_dim=int(actor_sd["mlp.0.weight"].shape[1]),
    action_dim=4,
    hidden_dims=tuple(ckpt["actor_hidden_dims"]),
  ).to(args.device)
  actor.load_state_dict(actor_sd)
  actor.eval()

  out = {}
  for label, stride in [("naive_100hz_stride1", 1), ("matched_80ms_100hz_stride2", 2)]:
    nom = eval_mode(actor, args.corpus_path, True, stride, args.trajs_per_family, 42, args.device)
    dr = eval_mode(actor, args.corpus_path, False, stride, args.trajs_per_family, 20261113, args.device)
    out[label] = {"nominal": nom, "full_dr": dr}
    print(
      f"[{label}] Nom: {nom['overall_family_mean_pos_rmse_cm']:.2f} cm ({nom['overall_family_mean_survival_pct']:.1f}%) | "
      f"DR: {dr['overall_family_mean_pos_rmse_cm']:.2f} cm ({dr['overall_family_mean_survival_pct']:.1f}%) | "
      f"OOF Nom/DR: {nom['oof_mean_pos_rmse_cm']:.2f} / {dr['oof_mean_pos_rmse_cm']:.2f} cm",
      flush=True,
    )

  Path(args.out_json).write_text(json.dumps(out, indent=2))
  print(f"[DONE] Saved {args.out_json}")


if __name__ == "__main__":
  main()
