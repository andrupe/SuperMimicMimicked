"""Render 20 Full DR 3D quadcopter GIFs for exp23 (100 Hz strided controller on exp17)."""

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
from PIL import Image

import supermimic.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from scratch.eval_all_families_and_oof import build_oof_trajectories
from scratch.render_hard_quad3d_gifs import append_traj
from scratch.render_quadcopter_3d_gifs import draw_quadcopter


def rollout_and_render_100hz(
  actor: QuadcopterPolicy,
  env: ManagerBasedRlEnv,
  cmd_term,
  traj_id: int,
  title: str,
  out_gif: Path,
  stride: int = 2,
  device: str = "cuda",
):
  env.reset()
  cmd_term.set_trajectory(0, traj_id, t=0)
  traj = cmd_term.trajectories[traj_id]
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
  obs_ring = obs["actor"].unsqueeze(1).repeat(1, ring_len, 1)

  pos_hist, ref_hist, quat_hist = [], [], []
  with torch.inference_mode():
    for k in range(total_steps):
      traj_t = min(k // 2, H - 1)
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
      q_t = torch.from_numpy(qpos[3:7]).unsqueeze(0).float()
      R = matrix_from_quat(q_t)[0].numpy()
      pos_hist.append(qpos[:3].copy())
      ref_hist.append(traj["pos"][traj_t].cpu().numpy().copy())
      quat_hist.append(R)

      obs, _, terminated, _, _ = env.step(action)
      if terminated.any():
        break
      obs_ring = torch.cat([obs_ring[:, 1:], obs["actor"].unsqueeze(1)], dim=1)

  pos_arr = np.array(pos_hist)
  ref_arr = np.array(ref_hist)
  err_arr = np.linalg.norm(pos_arr - ref_arr, axis=-1)
  surv = len(pos_arr) / float(total_steps) * 100.0
  rmse_cm = float(np.sqrt(np.mean(err_arr ** 2)) * 100.0)

  frames = []
  f_stride = max(1, len(pos_arr) // 45)
  indices = list(range(0, len(pos_arr), f_stride))
  if indices[-1] != len(pos_arr) - 1:
    indices.append(len(pos_arr) - 1)

  all_pts = np.vstack([ref_arr, pos_arr])
  mid = np.mean([all_pts.min(axis=0), all_pts.max(axis=0)], axis=0)
  half_span = max(0.75, float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.55))

  for i in indices:
    fig = plt.figure(figsize=(11, 5.0), dpi=80)
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
    ax1.set_title(f"{title} (Global)\nSurv: {surv:.0f}% | Err: {err_arr[i]*100:.1f} cm", fontsize=9, fontweight="bold")
    ax1.view_init(elev=24, azim=40 + (i * 0.35) % 360)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    p_cur = pos_arr[i]
    draw_quadcopter(ax2, p_cur, quat_hist[i], arm_len=0.28, rotor_r=0.11)
    i0 = max(0, i - 25)
    ax2.plot(pos_arr[i0 : i + 1, 0], pos_arr[i0 : i + 1, 1], pos_arr[i0 : i + 1, 2], color="#00bcd4", linewidth=2.0)
    ax2.scatter([ref_arr[i, 0]], [ref_arr[i, 1]], [ref_arr[i, 2]], color="#4caf50", s=55, marker="x", label="Target")
    ax2.set_xlim(p_cur[0] - 0.55, p_cur[0] + 0.55)
    ax2.set_ylim(p_cur[1] - 0.55, p_cur[1] + 0.55)
    ax2.set_zlim(max(0.0, p_cur[2] - 0.55), p_cur[2] + 0.55)
    ax2.set_box_aspect([1.0, 1.0, 1.0])
    ax2.set_title(f"Close-Up (RMSE: {rmse_cm:.2f} cm)", fontsize=9, fontweight="bold")
    ax2.view_init(elev=20, azim=45)

    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    frames.append(Image.fromarray(buf))
    plt.close(fig)

  out_gif.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(out_gif, frames, fps=15, loop=0)
  print(f"[SAVED] {out_gif} (Pos RMSE: {rmse_cm:.2f} cm, Surv: {surv:.1f}%)", flush=True)


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt", type=str, default="logs/crazyflie_exp17_50k/checkpoint_050000.pt")
  parser.add_argument("--corpus-path", type=str, default="corpus/crazyflie_corpus_50000trajs.npz")
  parser.add_argument("--out-dir", type=str, default="gif/exp23_quad3d")
  parser.add_argument("--label", type=str, default="exp23_100hz")
  args = parser.parse_args()

  device = "cuda"
  ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
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
  env_cfg.commands["velocity"].corpus_path = args.corpus_path
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  cmd_term = env.unwrapped.command_manager.get_term("velocity")

  by_family: dict[str, list[int]] = {}
  for idx, t in enumerate(cmd_term.trajectories):
    by_family.setdefault(str(t["task_type"]), []).append(idx)

  rng = np.random.default_rng(20261113)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  for fam in sorted(by_family.keys()):
    tid = int(rng.choice(by_family[fam], size=1)[0])
    rollout_and_render_100hz(
      actor, env, cmd_term, tid,
      title=f"{args.label} DR: {fam}",
      out_gif=out_dir / f"quad3d_{fam}.gif",
      stride=2, device=device,
    )

  for name, s_np, c_np in build_oof_trajectories():
    tid = append_traj(cmd_term, name, s_np, c_np, device)
    rollout_and_render_100hz(
      actor, env, cmd_term, tid,
      title=f"{args.label} DR: {name}",
      out_gif=out_dir / f"quad3d_{name}.gif",
      stride=2, device=device,
    )

  env.close()
  print(f"[DONE] Rendered all {args.label} 100 Hz quad3d GIFs to {args.out_dir}")


if __name__ == "__main__":
  main()
