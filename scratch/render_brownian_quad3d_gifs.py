"""Render 3 Brownian (Ornstein-Uhlenbeck) motion trajectories as trained on in SuperMimic-Crazyflie-Joint."""

from __future__ import annotations

import os
import sys
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import mujoco
from PIL import Image

import supermimic.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from scratch.render_quadcopter_3d_gifs import draw_quadcopter


def render_brownian_episode(
  actor: QuadcopterPolicy,
  env: ManagerBasedRlEnv,
  cmd_term,
  ep_idx: int,
  out_gif: Path,
  steps: int = 180,
  device: str = "cuda",
):
  torch.manual_seed(100 + ep_idx * 101)
  obs, _ = env.reset()
  pos_hist, quat_hist, ref_hist, tilt_hist = [], [], [], []
  vel_cmd_hist = []

  with torch.inference_mode():
    for _ in range(steps):
      qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy()
      q_t = torch.from_numpy(qpos[3:7]).float()
      R_np = matrix_from_quat(q_t.unsqueeze(0)).squeeze(0).numpy()
      tilt_deg = float(np.degrees(np.arccos(np.clip(R_np[2, 2], -1.0, 1.0))))

      pos_hist.append(qpos[:3].copy())
      quat_hist.append(R_np)
      ref_hist.append(cmd_term._ref_pos[0].cpu().numpy().copy())
      vel_cmd_hist.append(cmd_term.vel_command_b[0].cpu().numpy().copy() if hasattr(cmd_term, "vel_command_b") else cmd_term.command[0, :3].cpu().numpy().copy())
      tilt_hist.append(tilt_deg)

      action = actor(obs["actor"])
      obs, _, terminated, _, _ = env.step(action)
      if terminated.any():
        break

  pos_arr = np.array(pos_hist)
  ref_arr = np.array(ref_hist)
  err_arr = np.linalg.norm(pos_arr - ref_arr, axis=-1)
  surv = len(pos_arr) / float(steps) * 100.0
  rmse_cm = float(np.sqrt(np.mean(err_arr**2)) * 100.0)
  max_tilt = float(max(tilt_hist))

  frames = []
  stride = max(1, len(pos_arr) // 45)
  indices = list(range(0, len(pos_arr), stride))
  if indices[-1] != len(pos_arr) - 1:
    indices.append(len(pos_arr) - 1)

  all_pts = np.vstack([ref_arr, pos_arr])
  mid = np.mean([all_pts.min(axis=0), all_pts.max(axis=0)], axis=0)
  half_span = max(0.75, float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.55))

  for i in indices:
    fig = plt.figure(figsize=(11, 5.0), dpi=80)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(ref_arr[:, 0], ref_arr[:, 1], ref_arr[:, 2], "k--", linewidth=1.3, alpha=0.55, label="Brownian Target")
    ax1.plot(pos_arr[: i + 1, 0], pos_arr[: i + 1, 1], pos_arr[: i + 1, 2], color="#00bcd4", linewidth=2.0, label="Flight")
    draw_quadcopter(ax1, pos_arr[i], quat_hist[i], arm_len=max(0.18, half_span * 0.14), rotor_r=max(0.07, half_span * 0.055))
    ax1.set_xlim(mid[0] - half_span, mid[0] + half_span)
    ax1.set_ylim(mid[1] - half_span, mid[1] + half_span)
    z0 = max(0.0, mid[2] - half_span)
    ax1.set_zlim(z0, z0 + 2.0 * half_span)
    ax1.set_box_aspect([1.0, 1.0, 1.0])
    ax1.set_xlabel("X (m)", fontsize=8)
    ax1.set_ylabel("Y (m)", fontsize=8)
    ax1.set_zlabel("Z (m)", fontsize=8)
    ax1.set_title(f"Brownian OU #{ep_idx} (Global)\nSurv: {surv:.0f}% | Err: {err_arr[i]*100:.1f} cm", fontsize=9, fontweight="bold")
    ax1.view_init(elev=24, azim=40 + (i * 0.35) % 360)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    p_cur = pos_arr[i]
    draw_quadcopter(ax2, p_cur, quat_hist[i], arm_len=0.28, rotor_r=0.11)
    j0 = max(0, i - 15)
    ax2.plot(pos_arr[j0 : i + 1, 0], pos_arr[j0 : i + 1, 1], pos_arr[j0 : i + 1, 2], color="#00bcd4", linewidth=2.0)
    ax2.scatter([ref_arr[i, 0]], [ref_arr[i, 1]], [ref_arr[i, 2]], color="#4caf50", s=55, marker="x")
    zoom = 0.55
    ax2.set_xlim(p_cur[0] - zoom, p_cur[0] + zoom)
    ax2.set_ylim(p_cur[1] - zoom, p_cur[1] + zoom)
    ax2.set_zlim(p_cur[2] - zoom, p_cur[2] + zoom)
    ax2.set_box_aspect([1.0, 1.0, 1.0])
    ax2.set_xlabel("X (m)", fontsize=8)
    ax2.set_ylabel("Y (m)", fontsize=8)
    ax2.set_zlabel("Z (m)", fontsize=8)
    ax2.set_title(f"Chase Cam (Body Attitude)\nTilt: {tilt_hist[i]:.1f} deg | Z: {p_cur[2]:.2f} m | t={i*0.02:.2f}s", fontsize=9, fontweight="bold")
    ax2.view_init(elev=20, azim=45)

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frames.append(Image.fromarray(rgba).convert("RGB"))
    plt.close(fig)

  out_gif.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(str(out_gif), [np.array(f) for f in frames], fps=18, loop=0)
  print(f"[SUCCESS] Saved Brownian GIF: {out_gif} (Pos RMSE: {rmse_cm:.2f} cm, Max Tilt: {max_tilt:.1f} deg, Surv: {surv:.1f}%)")


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

  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Joint", play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["velocity"].corpus_ratio = 0.0
  env_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50trajs.npz"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term = env.unwrapped.command_manager.get_term("velocity")

  out_dir = Path("gif/exp4")
  for ep in (1, 2, 3):
    render_brownian_episode(actor, env, cmd_term, ep, out_dir / f"quad3d_brownian_{ep}.gif", device=device)

  env.close()


if __name__ == "__main__":
  main()
