"""Sweep Domain Randomization scale on exp25_50k (100 Hz, stride=2) and plot linear-scale degradation curves."""

from __future__ import annotations

import json
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from scratch.eval_100hz import rollout_100hz
from scratch.eval_all_families_and_oof import build_oof_trajectories


def configure_dr_scale(env_cfg, sc: float):
  act_cfg = env_cfg.actions["thrust"]
  if sc == 0.0:
    env_cfg.events["base_com"].params["ranges"] = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
    env_cfg.events["mass"].params["ranges"] = (1.0, 1.0)
    act_cfg.enable_domain_randomization = False
    return

  env_cfg.events["base_com"].params["ranges"] = {
    0: (-0.001 * sc, 0.001 * sc),
    1: (-0.001 * sc, 0.001 * sc),
    2: (-0.0008 * sc, 0.0008 * sc),
  }
  env_cfg.events["mass"].params["ranges"] = (max(0.5, 1.0 - 0.04 * sc), 1.0 + 0.04 * sc)
  act_cfg.tau_up_range = (max(0.005, 0.025 - 0.003 * sc), 0.025 + 0.004 * sc)
  act_cfg.tau_down_range = (max(0.005, 0.025 - 0.002 * sc), 0.025 + 0.007 * sc)
  act_cfg.thrust_scale_range = (max(0.35, 1.0 - 0.04 * sc), 1.0 + 0.02 * sc)
  act_cfg.sag_rate_range = (0.0, 0.0025 * sc)
  act_cfg.rotor_gain_range = (max(0.5, 1.0 - 0.015 * sc), 1.0 + 0.015 * sc)
  act_cfg.wind_xy_range = (-0.10 * sc, 0.10 * sc)
  act_cfg.wind_z_range = (-0.04 * sc, 0.04 * sc)
  act_cfg.drag_scale_range = (max(0.2, 1.0 - 0.05 * sc), 1.0 + 0.08 * sc)


def plot_linear(records: list[dict], out_png: str):
  scales = [r["dr_scale"] for r in records]
  rmse_17 = [r["mean_pos_rmse_cm"] for r in records]
  rmse_oof = [r["oof_mean_pos_rmse_cm"] for r in records]
  surv_17 = [r["mean_survival_pct"] for r in records]
  surv_oof = [r["oof_mean_survival_pct"] for r in records]

  fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=150)
  fig.suptitle("exp25_50k (100 Hz stride=2): Degradation vs Domain Randomization Scale (Linear Scales)", fontsize=12.5, fontweight="bold")

  axes[0].plot(scales, rmse_17, "o-", linewidth=2.2, label="17-Family Mean Pos RMSE")
  axes[0].plot(scales, rmse_oof, "s--", linewidth=2.0, label="3-OOF Unseen Pos RMSE")
  axes[0].axvline(1.0, color="gray", linestyle=":", alpha=0.8, label="1.0x (exp25 train DR)")
  axes[0].axvline(2.0, color="#ff7f0e", linestyle=":", alpha=0.8, label="~2.0x (exp17 DR level)")
  axes[0].set_title("Tracking Error vs DR Scale (Linear)")
  axes[0].set_xlabel("DR Scale Multiplier (x exp25 DR)")
  axes[0].set_ylabel("Mean Pos RMSE (cm)")
  axes[0].grid(True, alpha=0.3)
  axes[0].legend(fontsize=8.5)

  axes[1].plot(scales, surv_17, "o-", linewidth=2.2, label="17-Family Mean Survival")
  axes[1].plot(scales, surv_oof, "s--", linewidth=2.0, label="3-OOF Unseen Survival")
  axes[1].axvline(1.0, color="gray", linestyle=":", alpha=0.8, label="1.0x (exp25 train DR)")
  axes[1].axvline(2.0, color="#ff7f0e", linestyle=":", alpha=0.8, label="~2.0x (exp17 DR level)")
  axes[1].set_title("Survival Rate vs DR Scale (Linear)")
  axes[1].set_xlabel("DR Scale Multiplier (x exp25 DR)")
  axes[1].set_ylabel("Survival (%)")
  axes[1].set_ylim(0, 103)
  axes[1].grid(True, alpha=0.3)
  axes[1].legend(fontsize=8.5)

  plt.tight_layout()
  plt.savefig(out_png)
  plt.close(fig)


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

  oofs = build_oof_trajectories()
  scales = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
  results = []
  out_json = Path("logs/dr_sweep_exp25.json")
  out_png = "/home/andrupe/.gemini/antigravity-cli/brain/b2d41161-778e-44e8-a60a-c51d518b1e94/exp25_dr_sweep_linear.png"

  for sc in scales:
    env_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=(sc == 0.0))
    env_cfg.scene.num_envs = 1
    env_cfg.decimation = 2  # 100 Hz
    env_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50000trajs_clean_yaw.npz"
    configure_dr_scale(env_cfg, sc)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    cmd_term = env.unwrapped.command_manager.get_term("velocity")
    by_family: dict[str, list[int]] = {}
    for idx, t in enumerate(cmd_term.trajectories):
      by_family.setdefault(str(t["task_type"]), []).append(idx)

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

    rng = np.random.default_rng(20261113)
    fam_rmse, fam_surv = {}, {}
    for fam in sorted(by_family.keys()):
      chosen = rng.choice(by_family[fam], size=3, replace=False)
      runs = [rollout_100hz(actor, env, cmd_term.trajectories[int(tid)], cmd_term, int(tid), stride=2, device=device) for tid in chosen]
      fam_rmse[fam] = round(float(np.mean([r["pos_rmse_cm"] for r in runs])), 2)
      fam_surv[fam] = round(float(np.mean([r["survival_ratio"] for r in runs]) * 100.0), 1)

    oof_runs = [rollout_100hz(actor, env, cmd_term.trajectories[tid], cmd_term, tid, stride=2, device=device) for _, tid in oof_ids]
    env.close()

    mean_rmse = round(float(np.mean(list(fam_rmse.values()))), 2)
    mean_surv = round(float(np.mean(list(fam_surv.values()))), 1)
    oof_rmse = round(float(np.mean([r["pos_rmse_cm"] for r in oof_runs])), 2)
    oof_surv = round(float(np.mean([r["survival_ratio"] * 100.0 for r in oof_runs])), 1)
    failed = {k: v for k, v in fam_surv.items() if v < 100.0}

    results.append({
      "dr_scale": sc,
      "mass_range": [round(max(0.5, 1.0 - 0.04 * sc), 3), round(1.0 + 0.04 * sc, 3)],
      "thrust_scale_range": [round(max(0.35, 1.0 - 0.04 * sc), 3), round(1.0 + 0.02 * sc, 3)],
      "rotor_gain_range": [round(max(0.5, 1.0 - 0.015 * sc), 3), round(1.0 + 0.015 * sc, 3)],
      "mean_pos_rmse_cm": mean_rmse,
      "mean_survival_pct": mean_surv,
      "oof_mean_pos_rmse_cm": oof_rmse,
      "oof_mean_survival_pct": oof_surv,
      "failed_families": failed,
    })
    print(
      f"[DR={sc:3.1f}x] mass={results[-1]['mass_range']} thrust={results[-1]['thrust_scale_range']} | "
      f"17-fam: {mean_rmse:5.2f}cm ({mean_surv:5.1f}%) | 3-OOF: {oof_rmse:5.2f}cm ({oof_surv:5.1f}%) | fails={failed}",
      flush=True,
    )
    out_json.write_text(json.dumps(results, indent=2))
    plot_linear(results, out_png)

  print(f"[DONE] Saved {out_json} and {out_png}")


if __name__ == "__main__":
  main()
