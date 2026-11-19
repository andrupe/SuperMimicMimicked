"""Evaluate checkpoint_050000.pt across all 17 corpus families + 3 out-of-fold (OOF) unseen trajectories."""

from __future__ import annotations

import json
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import supermimic.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from supermimic.tasks.crazyflie_mdp import CorpusTrajectoryCommand
from scratch.generate_ultra_corpus_50000 import _solve_from_pos_yaw


def rollout_trajectory(
  actor: QuadcopterPolicy,
  env: ManagerBasedRlEnv,
  traj: dict[str, torch.Tensor | str | int],
  cmd_term: CorpusTrajectoryCommand,
  traj_id: int,
  device: str = "cuda",
  gif_path: Path | None = None,
  title_name: str = "",
) -> dict[str, float | str | int]:
  env.reset()
  cmd_term.set_trajectory(0, traj_id, t=0)
  H = int(traj["H"])

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

  ref_pos_hist, act_pos_hist = [], []
  ref_vel_hist, act_vel_hist = [], []
  act_hist, time_hist = [], []

  term_step = H
  term_reason = "completed"

  history_length = max(1, int(actor.obs_dim) // 23)
  actor_hist = obs["actor"].unsqueeze(1).repeat(1, history_length, 1) if history_length > 1 else None

  with torch.inference_mode():
    for t in range(H):
      actor_obs = obs["actor"]
      policy_in = actor_hist.reshape(1, -1) if history_length > 1 else actor_obs
      action = actor(policy_in)

      qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy()
      act_pos_hist.append(qpos[:3].copy())
      ref_pos_hist.append(traj["pos"][t].cpu().numpy().copy())
      act_vel_hist.append(actor_obs[0, 6:9].cpu().numpy().copy())
      ref_vel_hist.append(traj["lin_vel_b"][t].cpu().numpy().copy())
      act_hist.append(action[0].cpu().numpy().copy())
      time_hist.append(t * 0.02)

      obs, _, terminated, _, _ = env.step(action)
      if terminated.any():
        term_step = t + 1
        term_reason = "tilt_exceeded_or_crash"
        break
      if history_length > 1:
        actor_hist = torch.cat([actor_hist[:, 1:], obs["actor"].unsqueeze(1)], dim=1)

  act_p = np.array(act_pos_hist)
  ref_p = np.array(ref_pos_hist)
  act_v = np.array(act_vel_hist)
  ref_v = np.array(ref_vel_hist)
  act_u = np.array(act_hist)
  t_arr = np.array(time_hist)

  pos_errs = np.linalg.norm(act_p - ref_p, axis=-1)
  vel_errs = np.linalg.norm(act_v - ref_v, axis=-1)
  survival = float(term_step) / float(H)

  if gif_path is not None and len(act_p) > 10:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    stride = max(1, len(act_p) // 35)
    for i in range(5, len(act_p), stride):
      fig = plt.figure(figsize=(11, 4.5), dpi=75)
      ax1 = fig.add_subplot(1, 2, 1, projection="3d")
      ax1.plot(ref_p[:, 0], ref_p[:, 1], ref_p[:, 2], "k--", linewidth=1.5, alpha=0.6, label="Expert Target")
      ax1.plot(act_p[:i, 0], act_p[:i, 1], act_p[:i, 2], color="#00bcd4", linewidth=2.5, label="Policy Flight")
      ax1.scatter([act_p[i-1, 0]], [act_p[i-1, 1]], [act_p[i-1, 2]], color="#ff5722", s=70, label="Crazyflie")
      ax1.scatter([act_p[0, 0]], [act_p[0, 1]], [act_p[0, 2]], color="#4caf50", s=40, marker="^", label="Start")

      all_pts = np.vstack([ref_p, act_p])
      mid = np.mean([all_pts.min(axis=0), all_pts.max(axis=0)], axis=0)
      half_span = max(0.5, float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.55))
      ax1.set_xlim(mid[0] - half_span, mid[0] + half_span)
      ax1.set_ylim(mid[1] - half_span, mid[1] + half_span)
      ax1.set_zlim(max(0.0, mid[2] - half_span), max(0.0, mid[2] - half_span) + 2.0 * half_span)
      ax1.set_box_aspect([1.0, 1.0, 1.0])
      ax1.set_xlabel("X (m)", fontsize=8)
      ax1.set_ylabel("Y (m)", fontsize=8)
      ax1.set_zlabel("Z (m)", fontsize=8)
      ax1.set_title(f"OOF: {title_name}\nSurv: {survival*100:.0f}% | Err: {pos_errs[i-1]*100:.1f} cm", fontsize=9, fontweight="bold")
      ax1.legend(loc="upper right", fontsize=7)
      ax1.view_init(elev=26, azim=45 + (i * 0.4) % 360)

      ax2 = fig.add_subplot(2, 2, 2)
      ax2.plot(t_arr[:i], ref_v[:i, 0], "r--", alpha=0.7, label=r"$v_x^*$")
      ax2.plot(t_arr[:i], act_v[:i, 0], "r-", label=r"$v_x$ act")
      ax2.plot(t_arr[:i], ref_v[:i, 1], "g--", alpha=0.7, label=r"$v_y^*$")
      ax2.plot(t_arr[:i], act_v[:i, 1], "g-", label=r"$v_y$ act")
      ax2.plot(t_arr[:i], ref_v[:i, 2], "b--", alpha=0.7, label=r"$v_z^*$")
      ax2.plot(t_arr[:i], act_v[:i, 2], "b-", label=r"$v_z$ act")
      ax2.set_xlim([0, t_arr[-1]])
      ax2.set_ylabel("Velocity (m/s)", fontsize=8)
      ax2.set_title("Body Velocity Tracking", fontsize=8, fontweight="bold")
      ax2.grid(True, alpha=0.3)
      ax2.legend(loc="upper right", fontsize=6, ncol=3)

      ax3 = fig.add_subplot(2, 2, 4)
      for r, col in enumerate(["#e91e63", "#9c27b0", "#3f51b5", "#009688"]):
        ax3.plot(t_arr[:i], act_u[:i, r], color=col, label=f"M{r+1}")
      ax3.set_xlim([0, t_arr[-1]])
      ax3.set_ylim([-1.1, 1.1])
      ax3.set_xlabel("Time (s)", fontsize=8)
      ax3.set_ylabel("Thrust", fontsize=8)
      ax3.set_title("Motor Commands", fontsize=8, fontweight="bold")
      ax3.grid(True, alpha=0.3)
      ax3.legend(loc="upper right", fontsize=6, ncol=4)

      fig.tight_layout()
      fig.canvas.draw()
      rgba = np.asarray(fig.canvas.buffer_rgba())
      frames.append(Image.fromarray(rgba).convert("RGB"))
      plt.close(fig)
    if frames:
      imageio.mimsave(str(gif_path), [np.array(f) for f in frames], fps=20, loop=0)

  return {
    "steps_survived": int(term_step),
    "total_steps": int(H),
    "survival_ratio": float(survival),
    "term_reason": term_reason,
    "pos_rmse_cm": float(np.sqrt(np.mean(pos_errs ** 2)) * 100.0),
    "pos_max_cm": float(np.max(pos_errs) * 100.0),
    "vel_rmse_ms": float(np.sqrt(np.mean(vel_errs ** 2))),
  }


def build_oof_trajectories(dt: float = 0.02) -> list[tuple[str, np.ndarray, np.ndarray]]:
  """Generate 3 unseen out-of-fold trajectories never present in the 50k corpus."""
  np.random.seed(999999)
  oofs = []

  # 1. OOF Compound 3D Corkscrew-Lissajous (coupled 3:2:5 harmonics + yaw sweep)
  dur1 = 4.2
  t1 = np.linspace(0, dur1, int(dur1 / dt) + 1)
  pos1 = np.stack([
    0.95 * np.sin(1.4 * t1 + 0.3) + 0.25 * np.cos(2.8 * t1),
    0.85 * np.cos(1.9 * t1 - 0.5),
    1.15 + 0.35 * np.sin(2.3 * t1 + 0.8),
  ], axis=-1)
  vel1 = np.gradient(pos1, dt, axis=0)
  yaw1 = np.unwrap(np.arctan2(vel1[:, 1] + 1e-4, vel1[:, 0] + 1e-4))
  s1, c1, ok1 = _solve_from_pos_yaw(pos1, yaw1, dt)
  assert ok1, "OOF 1 failed flatness_solve"
  oofs.append(("oof_compound_lissajous", s1, c1))

  # 2. OOF Multi-Gate Waypoint Circuit with Altitude Chicane (unseen spline)
  from scipy.interpolate import CubicSpline
  dur2 = 4.6
  t2 = np.linspace(0, dur2, int(dur2 / dt) + 1)
  wps = np.array([
    [-1.1, -0.8, 0.75],
    [0.2, -1.0, 1.35],
    [1.1, 0.1, 0.85],
    [0.3, 1.1, 1.65],
    [-0.9, 0.7, 1.15],
    [-1.2, -0.3, 0.90],
  ])
  cs = CubicSpline(np.linspace(0, dur2, len(wps)), wps, bc_type="natural")
  pos2 = cs(t2)
  vel2 = np.gradient(pos2, dt, axis=0)
  yaw2 = np.unwrap(np.arctan2(vel2[:, 1] + 1e-4, vel2[:, 0] + 1e-4))
  s2, c2, ok2 = _solve_from_pos_yaw(pos2, yaw2, dt)
  assert ok2, "OOF 2 failed flatness_solve"
  oofs.append(("oof_waypoint_chicane", s2, c2))

  # 3. OOF High-Tilt Perturbed Recovery into Decoupled Crabbed Orbit
  dur3 = 4.0
  t3 = np.linspace(0, dur3, int(dur3 / dt) + 1)
  tau = t3 / dur3
  # Perturbed start decaying smoothly into a tilted circular orbit with independent yaw spin
  orbit = np.stack([
    0.85 * np.cos(1.8 * t3),
    0.85 * np.sin(1.8 * t3),
    1.25 + 0.22 * np.sin(1.8 * t3),
  ], axis=-1)
  pert = np.stack([
    0.45 * np.exp(-3.0 * t3) * np.cos(4.0 * t3),
    -0.40 * np.exp(-3.0 * t3),
    0.30 * np.exp(-3.0 * t3),
  ], axis=-1)
  pos3 = orbit + pert
  yaw3 = 0.5 + 1.4 * t3  # decoupled continuous yaw rotation
  s3, c3, ok3 = _solve_from_pos_yaw(pos3, yaw3, dt)
  assert ok3, "OOF 3 failed flatness_solve"
  oofs.append(("oof_recovery_crab_orbit", s3, c3))

  return oofs


def main():
  device = "cuda"
  ckpt_path = Path("logs/crazyflie_17fam_23d_50k/checkpoint_050000.pt")
  corpus_path = "corpus/crazyflie_corpus_50000trajs.npz"
  gif_dir = Path("gif/exp4")
  artifact_dir = Path("/home/andrupe/.gemini/antigravity-cli/brain/b2d41161-778e-44e8-a60a-c51d518b1e94")

  ckpt = torch.load(ckpt_path, map_location=device)
  actor_sd = ckpt["actor"]
  obs_dim = int(actor_sd["mlp.0.weight"].shape[1])
  hidden_dims = tuple(ckpt["actor_hidden_dims"])
  actor = QuadcopterPolicy(obs_dim=obs_dim, action_dim=4, hidden_dims=hidden_dims).to(device)
  actor.load_state_dict(actor_sd)
  actor.eval()

  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Corpus", play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["velocity"].corpus_path = corpus_path
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term: CorpusTrajectoryCommand = env.unwrapped.command_manager.get_term("velocity")

  # Group trajectories by task_type
  by_family: dict[str, list[int]] = {}
  for idx, t in enumerate(cmd_term.trajectories):
    by_family.setdefault(str(t["task_type"]), []).append(idx)

  rng = np.random.default_rng(42)
  family_results = {}
  print(f"\n[INFO] Evaluating 3 trajectories per family across all {len(by_family)} families...")
  for fam in sorted(by_family.keys()):
    ids = by_family[fam]
    chosen = rng.choice(ids, size=min(3, len(ids)), replace=False)
    runs = []
    for tid in chosen:
      tid = int(tid)
      res = rollout_trajectory(actor, env, cmd_term.trajectories[tid], cmd_term, tid, device=device)
      res["traj_id"] = tid
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
    print(f"  {fam:20s} | Pos RMSE: {mean_pos:6.2f} cm | Max: {max_pos:6.2f} cm | Surv: {mean_surv:5.1f}%")

  # 3 Out-of-Fold trajectories: append into cmd_term flat buffers so set_trajectory works identically
  oofs = build_oof_trajectories()
  oof_results = []
  print("\n[INFO] Evaluating 3 Out-of-Fold (unseen) trajectories...")
  for name, s_np, c_np in oofs:
    s_t = torch.from_numpy(s_np[:, :13]).float().to(device)
    c_t = torch.from_numpy(c_np).float().to(device)
    new_tid = cmd_term.num_trajs

    s_start = int(cmd_term._s_off[-1])
    c_start = int(cmd_term._c_off[-1])
    cmd_term._states = torch.cat([cmd_term._states, s_t], dim=0)
    cmd_term._ctrls = torch.cat([cmd_term._ctrls, c_t], dim=0)
    cmd_term._s_off = torch.cat([cmd_term._s_off, torch.tensor([s_start + len(s_t)], dtype=torch.long, device=device)])
    cmd_term._c_off = torch.cat([cmd_term._c_off, torch.tensor([c_start + len(c_t)], dtype=torch.long, device=device)])
    cmd_term._traj_len = cmd_term._c_off[1:] - cmd_term._c_off[:-1]
    cmd_term._tasks.append(name)
    cmd_term.num_trajs += 1

    gif_out = gif_dir / f"{name}_telemetry.gif"
    res = rollout_trajectory(
      actor, env, cmd_term.trajectories[new_tid], cmd_term, new_tid,
      device=device, gif_path=gif_out, title_name=name
    )
    res["name"] = name
    res["gif_path"] = str(gif_out)
    oof_results.append(res)
    print(f"  {name:24s} | Pos RMSE: {res['pos_rmse_cm']:6.2f} cm | Max: {res['pos_max_cm']:6.2f} cm | Surv: {res['survival_ratio']*100:5.1f}% | GIF: {gif_out}")

  env.close()

  overall = {
    "checkpoint": str(ckpt_path),
    "num_families": len(family_results),
    "overall_family_mean_pos_rmse_cm": round(float(np.mean([v["mean_pos_rmse_cm"] for v in family_results.values()])), 2),
    "overall_family_mean_survival_pct": round(float(np.mean([v["mean_survival_pct"] for v in family_results.values()])), 1),
    "families": family_results,
    "out_of_fold": oof_results,
  }
  out_json = artifact_dir / "eval_all17_and_oof_checkpoint_050000.json"
  with open(out_json, "w") as f:
    json.dump(overall, f, indent=2)
  print(f"\n[SUCCESS] Saved full report to {out_json}")

  import mlflow
  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  mlflow.set_experiment("SuperMimic-Crazyflie-17Families-23D-Scaled")
  with mlflow.start_run(run_name="eval_all17_and_3oof_checkpoint_050000"):
    mlflow.log_metrics({
      "eval17/mean_pos_rmse_cm": overall["overall_family_mean_pos_rmse_cm"],
      "eval17/mean_survival_pct": overall["overall_family_mean_survival_pct"],
      "oof/mean_pos_rmse_cm": float(np.mean([r["pos_rmse_cm"] for r in oof_results])),
      "oof/mean_survival_pct": float(np.mean([r["survival_ratio"] * 100.0 for r in oof_results])),
    })
    for fam, v in family_results.items():
      mlflow.log_metrics({
        f"fam/{fam}_pos_rmse_cm": v["mean_pos_rmse_cm"],
        f"fam/{fam}_survival_pct": v["mean_survival_pct"],
      })
    mlflow.log_artifact(str(out_json), artifact_path="eval")
    for r in oof_results:
      mlflow.log_artifact(r["gif_path"], artifact_path="oof_gifs")


if __name__ == "__main__":
  main()
