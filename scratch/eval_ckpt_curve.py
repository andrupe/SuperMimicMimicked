"""Evaluate a run's checkpoints across full training on CPU without disturbing GPU training."""

from __future__ import annotations

import argparse
import json
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import supermimic.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from scratch.eval_all_families_and_oof import build_oof_trajectories, rollout_trajectory


def build_env(corpus_path: str, play: bool, device: str):
  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=play)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["velocity"].corpus_path = corpus_path
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term = env.unwrapped.command_manager.get_term("velocity")

  by_family: dict[str, list[int]] = {}
  for idx, t in enumerate(cmd_term.trajectories):
    by_family.setdefault(str(t["task_type"]), []).append(idx)

  oofs = build_oof_trajectories()
  oof_entries = []
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
    oof_entries.append((name, new_tid))

  return env, cmd_term, by_family, oof_entries


def eval_on_env(
  actor: QuadcopterPolicy,
  env: ManagerBasedRlEnv,
  cmd_term,
  by_family: dict[str, list[int]],
  oof_entries: list[tuple[str, int]],
  play: bool,
  trajs_per_family: int,
  seed: int,
  device: str,
) -> dict:
  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)

  family_results = {}
  for fam in sorted(by_family.keys()):
    ids = by_family[fam]
    chosen = rng.choice(ids, size=min(trajs_per_family, len(ids)), replace=False)
    runs = []
    for tid in chosen:
      tid = int(tid)
      res = rollout_trajectory(
        actor, env, cmd_term.trajectories[tid], cmd_term, tid,
        device=device, gif_path=None, title_name=fam,
      )
      res["traj_id"] = tid
      runs.append(res)
    family_results[fam] = {
      "mean_pos_rmse_cm": round(float(np.mean([r["pos_rmse_cm"] for r in runs])), 2),
      "worst_pos_max_cm": round(float(np.max([r["pos_max_cm"] for r in runs])), 2),
      "mean_vel_rmse_ms": round(float(np.mean([r["vel_rmse_ms"] for r in runs])), 3),
      "mean_survival_pct": round(float(np.mean([r["survival_ratio"] for r in runs]) * 100.0), 1),
    }

  oof_results = []
  for name, oof_id in oof_entries:
    res = rollout_trajectory(
      actor, env, cmd_term.trajectories[oof_id], cmd_term, oof_id,
      device=device, gif_path=None, title_name=name,
    )
    res["name"] = name
    oof_results.append(res)

  return {
    "mode": "nominal" if play else "full_dr",
    "overall_family_mean_pos_rmse_cm": round(float(np.mean([v["mean_pos_rmse_cm"] for v in family_results.values()])), 2),
    "overall_family_mean_survival_pct": round(float(np.mean([v["mean_survival_pct"] for v in family_results.values()])), 1),
    "oof_mean_pos_rmse_cm": round(float(np.mean([r["pos_rmse_cm"] for r in oof_results])), 2),
    "oof_mean_survival_pct": round(float(np.mean([r["survival_ratio"] * 100.0 for r in oof_results])), 1),
    "families": family_results,
    "out_of_fold": oof_results,
  }


def plot_curve(records: list[dict], out_png: str, title: str):
  iters = [r["iter"] for r in records]
  nom_rmse = [r["nominal"]["overall_family_mean_pos_rmse_cm"] for r in records]
  dr_rmse = [r["full_dr"]["overall_family_mean_pos_rmse_cm"] for r in records]
  nom_surv = [r["nominal"]["overall_family_mean_survival_pct"] for r in records]
  dr_surv = [r["full_dr"]["overall_family_mean_survival_pct"] for r in records]
  nom_oof = [r["nominal"]["oof_mean_pos_rmse_cm"] for r in records]
  dr_oof = [r["full_dr"]["oof_mean_pos_rmse_cm"] for r in records]

  fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
  fig.suptitle(title, fontsize=13, fontweight="bold")

  axes[0].plot(iters, nom_rmse, "o-", label="Nominal (17-fam)")
  axes[0].plot(iters, dr_rmse, "s-", label="Full DR (17-fam)")
  axes[0].set_title("17-Family Mean Pos RMSE (cm)")
  axes[0].set_xlabel("Iteration")
  axes[0].set_ylabel("Pos RMSE (cm)")
  axes[0].grid(True, alpha=0.3)
  axes[0].legend()

  axes[1].plot(iters, nom_surv, "o-", label="Nominal (17-fam)")
  axes[1].plot(iters, dr_surv, "s-", label="Full DR (17-fam)")
  axes[1].set_title("17-Family Mean Survival (%)")
  axes[1].set_xlabel("Iteration")
  axes[1].set_ylabel("Survival (%)")
  axes[1].set_ylim(0, 103)
  axes[1].grid(True, alpha=0.3)
  axes[1].legend()

  axes[2].plot(iters, nom_oof, "o-", label="Nominal OOF")
  axes[2].plot(iters, dr_oof, "s-", label="Full DR OOF")
  axes[2].set_title("3-OOF Unseen Mean Pos RMSE (cm)")
  axes[2].set_xlabel("Iteration")
  axes[2].set_ylabel("Pos RMSE (cm)")
  axes[2].grid(True, alpha=0.3)
  axes[2].legend()

  plt.tight_layout()
  plt.savefig(out_png)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt-dir", type=str, required=True)
  parser.add_argument("--iters", type=int, nargs="+", required=True)
  parser.add_argument("--corpus-path", type=str, default="corpus/crazyflie_corpus_50000trajs.npz")
  parser.add_argument("--trajs-per-family", type=int, default=3)
  parser.add_argument("--out-json", type=str, required=True)
  parser.add_argument("--out-png", type=str, required=True)
  parser.add_argument("--title", type=str, default="Checkpoint Evaluation Progression")
  parser.add_argument("--device", type=str, default="cpu")
  args = parser.parse_args()

  env_nom, cmd_nom, fam_nom, oof_nom = build_env(args.corpus_path, play=True, device=args.device)
  env_dr, cmd_dr, fam_dr, oof_dr = build_env(args.corpus_path, play=False, device=args.device)

  records = []
  for it in args.iters:
    ckpt_path = Path(args.ckpt_dir) / f"checkpoint_{it:06d}.pt"
    ckpt = torch.load(ckpt_path, map_location=args.device)
    actor_sd = ckpt["actor"]
    actor = QuadcopterPolicy(
      obs_dim=int(actor_sd["mlp.0.weight"].shape[1]),
      action_dim=4,
      hidden_dims=tuple(ckpt["actor_hidden_dims"]),
    ).to(args.device)
    actor.load_state_dict(actor_sd)
    actor.eval()

    nom_res = eval_on_env(actor, env_nom, cmd_nom, fam_nom, oof_nom, True, args.trajs_per_family, 42, args.device)
    dr_res = eval_on_env(actor, env_dr, cmd_dr, fam_dr, oof_dr, False, args.trajs_per_family, 20261113, args.device)
    rec = {"iter": it, "checkpoint": str(ckpt_path), "nominal": nom_res, "full_dr": dr_res}
    records.append(rec)
    print(
      f"[iter {it:5d}] Nom: {nom_res['overall_family_mean_pos_rmse_cm']:5.2f} cm ({nom_res['overall_family_mean_survival_pct']:5.1f}%) | "
      f"DR: {dr_res['overall_family_mean_pos_rmse_cm']:5.2f} cm ({dr_res['overall_family_mean_survival_pct']:5.1f}%) | "
      f"OOF Nom/DR: {nom_res['oof_mean_pos_rmse_cm']:5.2f} / {dr_res['oof_mean_pos_rmse_cm']:5.2f} cm",
      flush=True,
    )
    Path(args.out_json).write_text(json.dumps(records, indent=2))
    plot_curve(records, args.out_png, args.title)

  env_nom.close()
  env_dr.close()
  print(f"[DONE] Saved {args.out_json} and {args.out_png}")


if __name__ == "__main__":
  main()
