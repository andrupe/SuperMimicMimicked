"""Comprehensive evaluation and MLflow logging script for SuperMimic Crazyflie quadcopter."""

from __future__ import annotations

import argparse
import json
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import sys
import time
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from PIL import Image, ImageDraw

import supermimic.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from supermimic.tasks.crazyflie_mdp import CorpusTrajectoryCommand
import mujoco


def evaluate_single_trajectory(
  actor: QuadcopterPolicy,
  env: ManagerBasedRlEnv,
  traj_id: int,
  task_name: str,
  output_dir: Path,
  render_gif: bool = True,
  device: str = "cuda",
  prefix: str = "",
) -> dict[str, float | str]:
  print(f"\n[EVAL] Evaluating Trajectory {traj_id} ({task_name})...")
  # Cleanly reset environment state, action history, and buffers before rollout
  env.reset()
  cmd_term: CorpusTrajectoryCommand = env.unwrapped.command_manager.get_term("velocity")

  cmd_term.set_trajectory(0, traj_id, t=0)
  traj = cmd_term.trajectories[traj_id]
  H = int(traj["H"])

  # Initial condition
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

  # Tracking histories
  ref_pos_hist = []
  act_pos_hist = []
  ref_vel_hist = []
  act_vel_hist = []
  act_hist = []
  time_hist = []

  term_step = H
  term_reason = "completed"

  with torch.inference_mode():
    for t in range(H):
      actor_obs = obs["actor"]
      action = actor(actor_obs)

      qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy()

      act_p = qpos[:3].copy()
      ref_p = traj["pos"][t].cpu().numpy().copy()
      act_v = actor_obs[0, 6:9].cpu().numpy().copy()
      ref_v = traj["lin_vel_b"][t].cpu().numpy().copy()
      u = action[0].cpu().numpy().copy()

      act_pos_hist.append(act_p)
      ref_pos_hist.append(ref_p)
      act_vel_hist.append(act_v)
      ref_vel_hist.append(ref_v)
      act_hist.append(u)
      time_hist.append(t * 0.02)

      obs, rewards, terminated, time_outs, extras = env.step(action)
      if terminated.any():
        term_step = t + 1
        term_reason = "tilt_exceeded_or_crash"
        print(f"[WARN] Trajectory {traj_id} terminated early at step {term_step}/{H} ({term_reason})")
        break

  act_p_arr = np.array(act_pos_hist)
  ref_p_arr = np.array(ref_pos_hist)
  act_v_arr = np.array(act_vel_hist)
  ref_v_arr = np.array(ref_vel_hist)
  act_u_arr = np.array(act_hist)
  t_arr = np.array(time_hist)

  pos_errs = np.linalg.norm(act_p_arr - ref_p_arr, axis=-1)
  vel_errs = np.linalg.norm(act_v_arr - ref_v_arr, axis=-1)
  survival_ratio = float(term_step) / float(H)

  # Action rate
  if len(act_u_arr) > 1:
    act_diff = np.diff(act_u_arr, axis=0)
    act_rate = float(np.mean(np.square(act_diff)))
  else:
    act_rate = 0.0

  results = {
    "task": task_name,
    "traj_id": traj_id,
    "steps_survived": int(term_step),
    "total_steps": int(H),
    "survival_ratio": float(survival_ratio),
    "term_reason": term_reason,
    "pos_rmse_m": float(np.sqrt(np.mean(pos_errs ** 2))),
    "pos_mean_m": float(np.mean(pos_errs)),
    "pos_max_m": float(np.max(pos_errs)),
    "pos_final_m": float(pos_errs[-1]),
    "vel_rmse_ms": float(np.sqrt(np.mean(vel_errs ** 2))),
    "vel_mean_ms": float(np.mean(vel_errs)),
    "action_rate": float(act_rate),
    "gif_path": "",
  }

  print(f"  Pos RMSE: {results['pos_rmse_m']*100:.2f} cm | Max: {results['pos_max_m']*100:.2f} cm | Vel RMSE: {results['vel_rmse_ms']:.3f} m/s | Survival: {survival_ratio*100:.1f}%")

  # Render animated GIF telemetry
  if render_gif and len(act_p_arr) > 10:
    p_tag = f"{prefix}_" if prefix else ""
    gif_file = output_dir / f"crazyflie_{p_tag}{task_name}_telemetry.gif"
    telemetry_frames = []
    step_stride = max(1, len(act_p_arr) // 35)

    for i in range(5, len(act_p_arr), step_stride):
      fig = plt.figure(figsize=(11, 4.5), dpi=75)

      # Left: 3D Trajectory with equal-scale bounding box
      ax1 = fig.add_subplot(1, 2, 1, projection="3d")
      ax1.plot(ref_p_arr[:, 0], ref_p_arr[:, 1], ref_p_arr[:, 2], "k--", linewidth=1.5, alpha=0.6, label="Expert Target")
      ax1.plot(act_p_arr[:i, 0], act_p_arr[:i, 1], act_p_arr[:i, 2], color="#00bcd4", linewidth=2.5, label="Policy Flight")
      ax1.scatter([act_p_arr[i-1, 0]], [act_p_arr[i-1, 1]], [act_p_arr[i-1, 2]], color="#ff5722", s=70, marker="o", label="Crazyflie")
      ax1.scatter([act_p_arr[0, 0]], [act_p_arr[0, 1]], [act_p_arr[0, 2]], color="#4caf50", s=40, marker="^", label="Start")

      all_pts = np.vstack([ref_p_arr, act_p_arr])
      mid = np.mean([all_pts.min(axis=0), all_pts.max(axis=0)], axis=0)
      half_span = max(0.5, float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.55))
      ax1.set_xlim(mid[0] - half_span, mid[0] + half_span)
      ax1.set_ylim(mid[1] - half_span, mid[1] + half_span)
      ax1.set_zlim(max(0.0, mid[2] - half_span), max(0.0, mid[2] - half_span) + 2.0 * half_span)
      ax1.set_box_aspect([1.0, 1.0, 1.0])

      ax1.set_xlabel("X (m)", fontsize=8)
      ax1.set_ylabel("Y (m)", fontsize=8)
      ax1.set_zlabel("Z (m)", fontsize=8)
      ax1.set_title(f"3D Flight Path: {task_name.title()}\nSurv: {survival_ratio*100:.0f}% | Err: {pos_errs[i-1]*100:.1f} cm", fontsize=9, fontweight="bold")
      ax1.legend(loc="upper right", fontsize=7)
      ax1.view_init(elev=26, azim=45 + (i * 0.4) % 360)

      # Right: Velocity and Action Tracking
      ax2 = fig.add_subplot(2, 2, 2)
      ax2.plot(t_arr[:i], ref_v_arr[:i, 0], "r--", alpha=0.7, label=r"$v_x^*$")
      ax2.plot(t_arr[:i], act_v_arr[:i, 0], "r-", label=r"$v_x$ act")
      ax2.plot(t_arr[:i], ref_v_arr[:i, 1], "g--", alpha=0.7, label=r"$v_y^*$")
      ax2.plot(t_arr[:i], act_v_arr[:i, 1], "g-", label=r"$v_y$ act")
      ax2.plot(t_arr[:i], ref_v_arr[:i, 2], "b--", alpha=0.7, label=r"$v_z^*$")
      ax2.plot(t_arr[:i], act_v_arr[:i, 2], "b-", label=r"$v_z$ act")
      ax2.set_xlim([0, t_arr[-1]])
      ax2.set_ylabel("Velocity (m/s)", fontsize=8)
      ax2.set_title("Body Velocity Tracking", fontsize=8, fontweight="bold")
      ax2.grid(True, alpha=0.3)
      ax2.legend(loc="upper right", fontsize=6, ncol=3)

      ax3 = fig.add_subplot(2, 2, 4)
      colors = ["#e91e63", "#9c27b0", "#3f51b5", "#009688"]
      for r in range(4):
        ax3.plot(t_arr[:i], act_u_arr[:i, r], color=colors[r], label=f"Motor {r+1}")
      ax3.set_xlim([0, t_arr[-1]])
      ax3.set_ylim([-1.1, 1.1])
      ax3.set_xlabel("Time (s)", fontsize=8)
      ax3.set_ylabel("Thrust Command", fontsize=8)
      ax3.set_title("Motor Commands u ∈ [-1, 1]", fontsize=8, fontweight="bold")
      ax3.grid(True, alpha=0.3)
      ax3.legend(loc="upper right", fontsize=6, ncol=4)

      fig.tight_layout()
      fig.canvas.draw()
      rgba = np.asarray(fig.canvas.buffer_rgba())
      telemetry_frames.append(Image.fromarray(rgba).convert("RGB"))
      plt.close(fig)

    if telemetry_frames:
      imageio.mimsave(str(gif_file), [np.array(f) for f in telemetry_frames], fps=20, loop=0)
      results["gif_path"] = str(gif_file)
      print(f"[SUCCESS] Saved telemetry GIF: {gif_file}")

  return results


def main():
  parser = argparse.ArgumentParser(description="Evaluate SuperMimic Crazyflie Policy on Benchmark Suite and Log to MLflow.")
  parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to policy checkpoint (.pt).")
  parser.add_argument("--corpus-path", type=str, default="corpus/crazyflie_corpus_9000trajs.npz", help="Path to corpus npz.")
  parser.add_argument("--output-dir", type=str, default=None, help="Directory to save eval artifacts.")
  parser.add_argument("--use-mlflow", action="store_true", help="Log results and artifacts to MLflow.")
  parser.add_argument("--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow URI.")
  parser.add_argument("--experiment-name", type=str, default="SuperMimic-Crazyflie-MegaCorpus", help="MLflow Experiment Name.")
  parser.add_argument("--run-id", type=str, default=None, help="Existing MLflow run ID to attach to.")
  parser.add_argument("--render-gifs", action="store_true", default=True, help="Render 3D telemetry GIFs.")
  parser.add_argument("--device", type=str, default="cuda", help="Device ('cuda' or 'cpu').")
  parser.add_argument("--prefix", type=str, default=None, help="Prefix tag for output artifacts (GIFs/JSON).")
  args = parser.parse_args()

  ckpt_path = Path(args.checkpoint_path).resolve()
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

  prefix = args.prefix or ckpt_path.parent.name

  # Output directory defaults to artifact brain dir
  if args.output_dir:
    output_dir = Path(args.output_dir).resolve()
  else:
    output_dir = Path("/home/andrupe/.gemini/antigravity-cli/brain/b2d41161-778e-44e8-a60a-c51d518b1e94").resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  # Load actor policy (auto-detect obs_dim & hidden_dims from checkpoint weights)
  print(f"[INFO] Loading checkpoint: {ckpt_path} (prefix: {prefix})")
  ckpt = torch.load(ckpt_path, map_location=args.device)
  actor_sd = ckpt["actor"]
  obs_dim = int(actor_sd["mlp.0.weight"].shape[1])
  weight_keys = sorted(
    [k for k in actor_sd if k.startswith("mlp.") and k.endswith(".weight") and actor_sd[k].ndim == 2],
    key=lambda k: int(k.split(".")[1]),
  )
  hidden_dims = tuple(ckpt.get("actor_hidden_dims") or (int(actor_sd[k].shape[0]) for k in weight_keys[:-1]))
  actor = QuadcopterPolicy(obs_dim=obs_dim, action_dim=4, hidden_dims=hidden_dims).to(args.device)
  actor.load_state_dict(actor_sd)
  actor.eval()

  # Create evaluation environment (1 env, corpus-based)
  print(f"[INFO] Initializing SuperMimic-Crazyflie-Corpus env with {args.corpus_path}...")
  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Corpus", play=True)
  env_cfg.scene.num_envs = 1
  if "velocity" in env_cfg.commands and hasattr(env_cfg.commands["velocity"], "corpus_path"):
    env_cfg.commands["velocity"].corpus_path = args.corpus_path

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  cmd_term: CorpusTrajectoryCommand = env.unwrapped.command_manager.get_term("velocity")
  num_trajs = len(cmd_term.trajectories)
  print(f"[INFO] Loaded corpus with {num_trajs} trajectories.")

  # Dynamically pick representative trajectories by actual task_type in priority order
  def _find_traj(target_types: tuple[str, ...], fallback_id: int) -> int:
    for tt in target_types:
      for idx, t in enumerate(cmd_term.trajectories):
        if t["task_type"] == tt:
          # Avoid degenerate constant-velocity fallback lines when non-hover
          if tt != "hover":
            vel_std = float(torch.std(t["lin_vel_b"]).item())
            if vel_std < 0.05:
              continue
          return idx
    return min(fallback_id, num_trajs - 1)

  test_specs = [
    (_find_traj(("figure8",), 0), "figure8_spline"),
    (_find_traj(("helix",), 1), "ascending_helix"),
    (_find_traj(("slalom", "point_to_point"), 2), "agile_slalom"),
    (_find_traj(("hover",), 3), "precision_hover"),
  ]

  all_results = []
  for traj_id, task_name in test_specs:
    res = evaluate_single_trajectory(
      actor=actor,
      env=env,
      traj_id=traj_id,
      task_name=task_name,
      output_dir=output_dir,
      render_gif=args.render_gifs,
      device=args.device,
      prefix=prefix,
    )
    all_results.append(res)

  env.close()

  # Aggregated metrics
  overall_pos_rmse = float(np.mean([r["pos_rmse_m"] for r in all_results]))
  overall_survival = float(np.mean([r["survival_ratio"] for r in all_results]))
  overall_vel_rmse = float(np.mean([r["vel_rmse_ms"] for r in all_results]))

  summary = {
    "checkpoint": str(ckpt_path),
    "iteration": ckpt.get("iteration", -1),
    "prefix": prefix,
    "overall_pos_rmse_m": overall_pos_rmse,
    "overall_survival_ratio": overall_survival,
    "overall_vel_rmse_ms": overall_vel_rmse,
    "results_by_course": all_results,
  }

  summary_path = output_dir / f"eval_summary_{prefix}_{ckpt_path.stem}.json"
  with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
  print(f"\n[INFO] Saved evaluation summary JSON to {summary_path}")

  # MLflow logging
  if args.use_mlflow:
    import mlflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Use existing run or create new
    if args.run_id:
      with mlflow.start_run(run_id=args.run_id):
        _log_to_mlflow(summary, all_results, summary_path)
    else:
      with mlflow.start_run(run_name=f"eval_{prefix}_{ckpt_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}"):
        _log_to_mlflow(summary, all_results, summary_path)

    print(f"[MLflow] Successfully logged evaluation results and artifacts to {args.mlflow_tracking_uri}")


def _log_to_mlflow(summary: dict, all_results: list[dict], summary_path: Path):
  import mlflow
  mlflow.log_metrics({
    "eval/overall_pos_rmse_m": summary["overall_pos_rmse_m"],
    "eval/overall_survival_ratio": summary["overall_survival_ratio"],
    "eval/overall_vel_rmse_ms": summary["overall_vel_rmse_ms"],
  })

  for r in all_results:
    prefix = f"eval/{r['task']}"
    mlflow.log_metrics({
      f"{prefix}/pos_rmse_m": r["pos_rmse_m"],
      f"{prefix}/pos_max_m": r["pos_max_m"],
      f"{prefix}/vel_rmse_ms": r["vel_rmse_ms"],
      f"{prefix}/survival_ratio": r["survival_ratio"],
      f"{prefix}/action_rate": r["action_rate"],
    })
    if r["gif_path"] and Path(r["gif_path"]).exists():
      mlflow.log_artifact(r["gif_path"], artifact_path="eval_gifs")

  mlflow.log_artifact(str(summary_path), artifact_path="eval_reports")


if __name__ == "__main__":
  main()
