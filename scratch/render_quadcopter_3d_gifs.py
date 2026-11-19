"""Render 3D quadcopter-body GIF visualizations showing full 3D attitude (arms, rotors, nose, tilt angle)."""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
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
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from supermimic.tasks.crazyflie_mdp import CorpusTrajectoryCommand
from scratch.eval_all_families_and_oof import build_oof_trajectories


def draw_quadcopter(ax, pos: np.ndarray, R: np.ndarray, arm_len: float = 0.22, rotor_r: float = 0.085):
  """Draw a 3D X-frame quadcopter at `pos` with rotation matrix `R` (body->world)."""
  # X-frame arm directions in body frame (45 deg diagonals)
  angles = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
  rotor_colors = ["#ff5722", "#ff5722", "#2196f3", "#2196f3"]  # front rotors orange, rear rotors blue
  theta = np.linspace(0, 2 * np.pi, 20)
  circle_b = np.stack([rotor_r * np.cos(theta), rotor_r * np.sin(theta), np.zeros_like(theta)], axis=0)  # (3, 20)

  for ang, col in zip(angles, rotor_colors):
    arm_b = np.array([arm_len * np.cos(ang), arm_len * np.sin(ang), 0.0])
    tip_w = pos + R @ arm_b
    ax.plot([pos[0], tip_w[0]], [pos[1], tip_w[1]], [pos[2], tip_w[2]], color="#263238", linewidth=2.6)
    ring_w = tip_w[:, None] + R @ circle_b
    ax.plot(ring_w[0], ring_w[1], ring_w[2], color=col, linewidth=1.8)

  # Central body hub
  ax.scatter([pos[0]], [pos[1]], [pos[2]], color="#111111", s=35, zorder=10)

  # Forward nose arrow (+X body) and thrust/up axis (+Z body)
  nose_w = pos + R @ np.array([arm_len * 1.15, 0.0, 0.0])
  ax.plot([pos[0], nose_w[0]], [pos[1], nose_w[1]], [pos[2], nose_w[2]], color="#ff1744", linewidth=2.5)
  up_w = pos + R @ np.array([0.0, 0.0, arm_len * 1.1])
  ax.plot([pos[0], up_w[0]], [pos[1], up_w[1]], [pos[2], up_w[2]], color="#4caf50", linewidth=2.0)


def rollout_and_render_quad(
  actor: QuadcopterPolicy,
  env: ManagerBasedRlEnv,
  cmd_term: CorpusTrajectoryCommand,
  traj_id: int,
  title: str,
  out_gif: Path,
  device: str = "cuda",
):
  env.reset()
  cmd_term.set_trajectory(0, traj_id, t=0)
  traj = cmd_term.trajectories[traj_id]
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

  pos_hist, quat_hist, ref_hist, tilt_hist = [], [], [], []
  history_length = max(1, int(actor.obs_dim) // 23)
  actor_hist = obs["actor"].unsqueeze(1).repeat(1, history_length, 1) if history_length > 1 else None
  with torch.inference_mode():
    for t in range(H):
      qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy()
      q_t = torch.from_numpy(qpos[3:7]).float()
      R_np = matrix_from_quat(q_t.unsqueeze(0)).squeeze(0).numpy()
      tilt_deg = float(np.degrees(np.arccos(np.clip(R_np[2, 2], -1.0, 1.0))))

      pos_hist.append(qpos[:3].copy())
      quat_hist.append(R_np)
      ref_hist.append(traj["pos"][t].cpu().numpy().copy())
      tilt_hist.append(tilt_deg)

      policy_in = actor_hist.reshape(1, -1) if history_length > 1 else obs["actor"]
      action = actor(policy_in)
      obs, _, terminated, _, _ = env.step(action)
      if terminated.any():
        break
      if history_length > 1:
        actor_hist = torch.cat([actor_hist[:, 1:], obs["actor"].unsqueeze(1)], dim=1)

  pos_arr = np.array(pos_hist)
  ref_arr = np.array(ref_hist)
  err_arr = np.linalg.norm(pos_arr - ref_arr, axis=-1)
  surv = len(pos_arr) / float(H) * 100.0

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

    # Left: Full 3D course with quadcopter X-frame
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(ref_arr[:, 0], ref_arr[:, 1], ref_arr[:, 2], "k--", linewidth=1.3, alpha=0.55, label="Target")
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
    ax1.set_title(f"{title} (Global View)\nSurv: {surv:.0f}% | Err: {err_arr[i]*100:.1f} cm", fontsize=9, fontweight="bold")
    ax1.view_init(elev=24, azim=40 + (i * 0.35) % 360)

    # Right: Drone-centered close-up chase camera so tilt/roll/pitch/yaw are crystal clear
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    p_cur = pos_arr[i]
    draw_quadcopter(ax2, p_cur, quat_hist[i], arm_len=0.28, rotor_r=0.11)
    # Show short local trail + target marker
    j0 = max(0, i - 15)
    ax2.plot(pos_arr[j0 : i + 1, 0], pos_arr[j0 : i + 1, 1], pos_arr[j0 : i + 1, 2], color="#00bcd4", linewidth=2.0)
    ax2.scatter([ref_arr[i, 0]], [ref_arr[i, 1]], [ref_arr[i, 2]], color="#4caf50", s=55, marker="x", label="Target Now")
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
  print(f"[SUCCESS] Saved 3D quadcopter GIF: {out_gif} (Pos RMSE: {np.sqrt(np.mean(err_arr**2))*100:.2f} cm, Max Tilt: {max(tilt_hist):.1f} deg, Surv: {surv:.1f}%)")


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

  env_cfg = load_env_cfg("SuperMimic-Crazyflie-Corpus", play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["velocity"].corpus_path = "corpus/crazyflie_corpus_50000trajs.npz"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term: CorpusTrajectoryCommand = env.unwrapped.command_manager.get_term("velocity")

  def find_first(task_type: str) -> int:
    for idx, t in enumerate(cmd_term.trajectories):
      if t["task_type"] == task_type and float(torch.std(t["lin_vel_b"]).item()) > 0.1:
        return idx
    return 0

  # Append 1 OOF trajectory too
  oof_name, s_np, c_np = build_oof_trajectories()[0]
  s_t = torch.from_numpy(s_np[:, :13]).float().to(device)
  c_t = torch.from_numpy(c_np).float().to(device)
  oof_tid = cmd_term.num_trajs
  s_start, c_start = int(cmd_term._s_off[-1]), int(cmd_term._c_off[-1])
  cmd_term._states = torch.cat([cmd_term._states, s_t], dim=0)
  cmd_term._ctrls = torch.cat([cmd_term._ctrls, c_t], dim=0)
  cmd_term._s_off = torch.cat([cmd_term._s_off, torch.tensor([s_start + len(s_t)], dtype=torch.long, device=device)])
  cmd_term._c_off = torch.cat([cmd_term._c_off, torch.tensor([c_start + len(c_t)], dtype=torch.long, device=device)])
  cmd_term._traj_len = cmd_term._c_off[1:] - cmd_term._c_off[:-1]
  cmd_term._tasks.append(oof_name)
  cmd_term.num_trajs += 1

  out_dir = Path("gif/exp4")
  specs = [
    (find_first("split_s_dive"), "Split-S Dive", out_dir / "quad3d_split_s_dive.gif"),
    (find_first("slalom"), "Agile Slalom", out_dir / "quad3d_agile_slalom.gif"),
    (oof_tid, "OOF Compound Lissajous", out_dir / "quad3d_oof_lissajous.gif"),
  ]
  for tid, title, path in specs:
    rollout_and_render_quad(actor, env, cmd_term, tid, title, path, device=device)

  env.close()


if __name__ == "__main__":
  main()
