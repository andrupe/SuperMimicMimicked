"""Evaluate a Crazyflie checkpoint across all 17 corpus families + 3 OOF trajectories, both nominal and under full DR."""

from __future__ import annotations

import argparse
import json
import os
import sys
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
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


def run_eval_suite(
  actor: QuadcopterPolicy,
  corpus_path: str,
  play: bool,
  gif_dir: Path | None,
  device: str = "cuda",
  seed: int = 42,
  trajs_per_family: int = 3,
  gif_all_families: bool = False,
) -> dict:
  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=play)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["velocity"].corpus_path = corpus_path
  env_cfg.commands["velocity"].corpus_ratio = 1.0
  env_cfg.commands["velocity"].rsi = False
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term = env.unwrapped.command_manager.get_term("velocity")

  by_family: dict[str, list[int]] = {}
  for idx, t in enumerate(cmd_term.trajectories):
    by_family.setdefault(str(t["task_type"]), []).append(idx)

  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)

  mode_label = "NOMINAL (play=True)" if play else "FULL DR (play=False)"
  print(f"\n=== Evaluating {mode_label}: {trajs_per_family} trajs/family across {len(by_family)} families ===")
  family_results = {}
  for fam in sorted(by_family.keys()):
    ids = by_family[fam]
    chosen = rng.choice(ids, size=min(trajs_per_family, len(ids)), replace=False)
    runs = []
    for i, tid in enumerate(chosen):
      tid = int(tid)
      gif_out = (gif_dir / f"{fam}_telemetry.gif") if (gif_dir is not None and gif_all_families and i == 0) else None
      res = rollout_trajectory(
        actor, env, cmd_term.trajectories[tid], cmd_term, tid,
        device=device, gif_path=gif_out, title_name=f"{fam} ({'nom' if play else 'DR'})"
      )
      res["traj_id"] = tid
      if gif_out is not None:
        res["gif_path"] = str(gif_out)
      runs.append(res)
    mean_pos = float(np.mean([r["pos_rmse_cm"] for r in runs]))
    max_pos = float(np.max([r["pos_max_cm"] for r in runs]))
    mean_vel = float(np.mean([r["vel_rmse_ms"] for r in runs]))
    mean_surv = float(np.mean([r["survival_ratio"] for r in runs]) * 100.0)
    family_results[fam] = {
      "mean_pos_rmse_cm": round(mean_pos, 2),
      "worst_pos_max_cm": round(max_pos, 2),
      "mean_vel_rmse_ms": round(mean_vel, 3),
      "mean_survival_pct": round(mean_surv, 1),
      "runs": runs,
    }
    print(f"  [{mode_label}] {fam:20s} | Pos RMSE: {mean_pos:6.2f} cm | Max: {max_pos:6.2f} cm | Surv: {mean_surv:5.1f}%")

  oofs = build_oof_trajectories()
  oof_results = []
  print(f"\n=== Evaluating {mode_label}: 3 Out-of-Fold (unseen) trajectories ===")
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

    gif_out = (gif_dir / f"{name}_telemetry.gif") if gif_dir is not None else None
    res = rollout_trajectory(
      actor, env, cmd_term.trajectories[new_tid], cmd_term, new_tid,
      device=device, gif_path=gif_out, title_name=f"{name} ({'nom' if play else 'DR'})"
    )
    res["name"] = name
    if gif_out is not None:
      res["gif_path"] = str(gif_out)
    oof_results.append(res)
    print(f"  [{mode_label}] {name:24s} | Pos RMSE: {res['pos_rmse_cm']:6.2f} cm | Surv: {res['survival_ratio']*100:5.1f}%")

  env.close()
  return {
    "mode": "nominal" if play else "full_dr",
    "overall_family_mean_pos_rmse_cm": round(float(np.mean([v["mean_pos_rmse_cm"] for v in family_results.values()])), 2),
    "overall_family_mean_survival_pct": round(float(np.mean([v["mean_survival_pct"] for v in family_results.values()])), 1),
    "oof_mean_pos_rmse_cm": round(float(np.mean([r["pos_rmse_cm"] for r in oof_results])), 2),
    "oof_mean_survival_pct": round(float(np.mean([r["survival_ratio"] * 100.0 for r in oof_results])), 1),
    "families": family_results,
    "out_of_fold": oof_results,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt", type=str, required=True)
  parser.add_argument("--name", type=str, required=True)
  parser.add_argument("--corpus-path", type=str, default="corpus/crazyflie_corpus_50000trajs.npz")
  parser.add_argument("--out-json", type=str, required=True)
  parser.add_argument("--gif-dir", type=str, required=True)
  parser.add_argument("--trajs-per-family", type=int, default=3)
  parser.add_argument("--gif-all-families", action="store_true")
  parser.add_argument("--dr-gifs-only", action="store_true")
  parser.add_argument("--device", type=str, default="cuda")
  args = parser.parse_args()

  ckpt = torch.load(args.ckpt, map_location=args.device)
  actor_sd = ckpt["actor"]
  actor = QuadcopterPolicy(
    obs_dim=int(actor_sd["mlp.0.weight"].shape[1]),
    action_dim=4,
    hidden_dims=tuple(ckpt["actor_hidden_dims"]),
  ).to(args.device)
  actor.load_state_dict(actor_sd)
  actor.eval()

  gif_base = Path(args.gif_dir)
  nom_gif_dir = None if args.dr_gifs_only else (gif_base / "nominal")
  dr_gif_dir = gif_base if args.dr_gifs_only else (gif_base / "dr")
  nom_res = run_eval_suite(
    actor, args.corpus_path, play=True, gif_dir=nom_gif_dir,
    device=args.device, seed=42, trajs_per_family=args.trajs_per_family,
    gif_all_families=args.gif_all_families,
  )
  dr_res = run_eval_suite(
    actor, args.corpus_path, play=False, gif_dir=dr_gif_dir,
    device=args.device, seed=20261113, trajs_per_family=args.trajs_per_family,
    gif_all_families=args.gif_all_families,
  )

  report = {
    "name": args.name,
    "checkpoint": args.ckpt,
    "nominal": nom_res,
    "full_dr": dr_res,
  }
  out_path = Path(args.out_json)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(report, indent=2))
  print(f"\n[SUCCESS] Saved full eval report to {out_path}")

  import mlflow
  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  mlflow.set_experiment("SuperMimic-Crazyflie-MegaCorpus")
  with mlflow.start_run(run_name=f"eval_{args.name}"):
    mlflow.log_metrics({
      "eval_nom/fam_pos_rmse_cm": nom_res["overall_family_mean_pos_rmse_cm"],
      "eval_nom/fam_survival_pct": nom_res["overall_family_mean_survival_pct"],
      "eval_nom/oof_pos_rmse_cm": nom_res["oof_mean_pos_rmse_cm"],
      "eval_nom/oof_survival_pct": nom_res["oof_mean_survival_pct"],
      "eval_dr/fam_pos_rmse_cm": dr_res["overall_family_mean_pos_rmse_cm"],
      "eval_dr/fam_survival_pct": dr_res["overall_family_mean_survival_pct"],
      "eval_dr/oof_pos_rmse_cm": dr_res["oof_mean_pos_rmse_cm"],
      "eval_dr/oof_survival_pct": dr_res["oof_mean_survival_pct"],
    })
    mlflow.log_artifact(str(out_path), artifact_path="eval")


if __name__ == "__main__":
  main()
