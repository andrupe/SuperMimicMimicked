"""Render high-speed agile slalom flight demonstration for the SuperMimic Quadrotor.

Produces:
- scratch/video_high_speed_slalom.gif
- /home/andrupe/.gemini/antigravity-cli/brain/b2d41161-778e-44e8-a60a-c51d518b1e94/video_high_speed_slalom.gif

Accurately displays:
1. 3D Slalom reference path with slalom waypoint markers.
2. Actual Crazyflie drone attitude, carbon arms, spinning rotor discs, and ground shadow.
3. Live telemetry: Tracking error (cm), 4 motor RPMs, tilt angle, and speed.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import msgpack_restore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from quadrotor.params import HOVER_RPM, MAX_THRUST, M_NOMINAL, G
from quadrotor.supermimic import SuperMimicPolicy, compute_obs, normalize_obs, norm_action_to_rpm
from quadrotor.supermimic_env import sim_step_multi_substeps
from quadrotor.quaternion import state17_to_22
from scratch.generate_ultra_corpus_50000 import sample_slalom_trajectory

THRUST_SCALE = 0.4 * HOVER_RPM
ARTIFACT_DIR = "/home/andrupe/.gemini/antigravity-cli/brain/b2d41161-778e-44e8-a60a-c51d518b1e94"
LOCAL_DIR = "scratch"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(LOCAL_DIR, exist_ok=True)

# Load production model
ckpt_path = "logs/supermimic_production/checkpoints/checkpoint_best.npz"
data = np.load(ckpt_path, allow_pickle=True)
params = msgpack_restore(data["params_bytes"].tobytes())
p_pi = params["policy"]
obs_mean = data["obs_mean"]
obs_std = data["obs_std"]
obs_m_j, obs_s_j = jnp.asarray(obs_mean), jnp.asarray(obs_std)

policy = SuperMimicPolicy()


@jax.jit
def run_closed_loop_sim(p, xs_ref, a0, xi, noise_pos, noise_gyro):
  traj_len = len(xs_ref) - 1
  def step(carry, t):
    s_curr, a_prev = carry
    o = compute_obs(s_curr, xs_ref[t + 1, 0:3], a_prev)
    o = o.at[0:3].add(noise_pos[t]).at[15:18].add(noise_gyro[t])
    o_n = normalize_obs(o, obs_m_j, obs_s_j)
    a = policy.apply({"params": p}, o_n)
    u = norm_action_to_rpm(a)
    s_next = sim_step_multi_substeps(s_curr, u, xi, substeps=4, sub_dt=0.005)

    pos_err = jnp.linalg.norm(s_next[0:3] - xs_ref[t + 1, 0:3])
    z_err = s_next[2] - xs_ref[t + 1, 2]

    R = s_next[3:12].reshape((3, 3))
    cos_tilt = jnp.clip(R[2, 2], -1.0, 1.0)
    tilt_deg = jnp.rad2deg(jnp.arccos(cos_tilt))

    step_data = {
      "state": s_next,
      "pos": s_next[0:3],
      "R": R,
      "pos_err": pos_err,
      "z_err": z_err,
      "rpm": u,
      "tilt_deg": tilt_deg,
      "action": a,
    }
    return (s_next, a), step_data

  _, history = jax.lax.scan(step, (xs_ref[0], a0), jnp.arange(traj_len))
  return history


def draw_crazyflie(ax, pos, R, arm_len=0.046, rotor_r=0.025, col_front="#e74c3c", col_rear="#2980b9"):
  """Draws a 3D Crazyflie with arms, rotors, and ground projection shadow."""
  d = arm_len / np.sqrt(2.0)
  motors_body = np.array([
    [ d, -d, 0.0],  # M1: front-right
    [-d, -d, 0.0],  # M2: rear-right
    [-d,  d, 0.0],  # M3: rear-left
    [ d,  d, 0.0],  # M4: front-left
  ])
  motors_world = pos + (R @ motors_body.T).T

  # Draw carbon fiber X arms
  ax.plot([motors_world[1, 0], motors_world[3, 0]],
          [motors_world[1, 1], motors_world[3, 1]],
          [motors_world[1, 2], motors_world[3, 2]],
          color="#2c3e50", lw=3.0, zorder=6)
  ax.plot([motors_world[0, 0], motors_world[2, 0]],
          [motors_world[0, 1], motors_world[2, 1]],
          [motors_world[0, 2], motors_world[2, 2]],
          color="#2c3e50", lw=3.0, zorder=6)

  # Central PCB body
  ax.scatter([pos[0]], [pos[1]], [pos[2]], color="#111111", s=60, edgecolors="#f39c12", linewidths=1.5, zorder=7)

  # Rotor discs
  theta = np.linspace(0, 2 * np.pi, 20)
  circle_pts = np.stack([rotor_r * np.cos(theta), rotor_r * np.sin(theta), np.zeros_like(theta)], axis=0)
  for i in range(4):
    col = col_front if i in [0, 3] else col_rear
    disc_world = motors_world[i][:, None] + (R @ circle_pts)
    ax.plot(disc_world[0], disc_world[1], disc_world[2], color=col, lw=2.0, zorder=8)
    ax.scatter([motors_world[i, 0]], [motors_world[i, 1]], [motors_world[i, 2]], color=col, s=20, zorder=9)

  # Ground shadow & drop-line
  ax.scatter([pos[0]], [pos[1]], [0.0], color="#7f8c8d", alpha=0.35, s=50, zorder=3)
  ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0.0, pos[2]], color="#95a5a6", ls=":", lw=1.2, alpha=0.6, zorder=4)


def render_slalom_video():
  print("\n[Rendering Video] High-Speed 3D Slalom Trajectory Tracking...", flush=True)

  # Sample a clean, dynamic slalom trajectory
  np.random.seed(1337)
  # Standard agile slalom: forward speed 1.4 m/s, lateral amplitude 0.8m, period 2.0s
  dt = 0.02
  duration = 3.6
  H = int(duration / dt)
  t = np.linspace(0, duration, H + 1)

  v_fwd = 1.3
  A_lateral = 0.75
  w_lateral = 2.4
  z0 = 1.1

  pos = np.stack([
    v_fwd * (t - duration / 2.0),
    A_lateral * np.sin(w_lateral * t),
    np.full(H + 1, z0),
  ], axis=-1)

  vel = np.gradient(pos, dt, axis=0)
  acc = np.gradient(vel, dt, axis=0)
  jerk = np.gradient(acc, dt, axis=0)
  yaw = np.unwrap(np.arctan2(vel[:, 1] + 1e-4, vel[:, 0] + 1e-4))
  yaw_dot = np.gradient(yaw, dt)

  from scratch.generate_ultra_corpus_50000 import flatness_solve
  s17, c, ok = flatness_solve(pos, vel, acc, jerk, yaw, yaw_dot)
  if not ok:
    print("[WARN] Slalom flatness solve fallback triggered.")
    s17, c, _ = sample_slalom_trajectory(dt=0.02)

  s22 = np.array([state17_to_22(x) for x in s17])
  u_rpm = c * MAX_THRUST
  raw_act = np.clip((u_rpm - HOVER_RPM) / THRUST_SCALE, -1.0, 1.0)

  zero_noise = jnp.zeros((len(s22) - 1, 3))
  zero_xi = jnp.zeros(10)

  hist = run_closed_loop_sim(
    p_pi, jnp.asarray(s22), jnp.asarray(raw_act[0]),
    zero_xi, zero_noise, zero_noise
  )

  pos_actual = np.array(hist["pos"])
  pos_ref = s22[1:, 0:3]
  errors_cm = np.array(hist["pos_err"]) * 100.0
  rpms = np.array(hist["rpm"])
  tilts = np.array(hist["tilt_deg"])
  Rs = np.array(hist["R"])
  traj_len = len(pos_actual)

  # Axis limits
  pad = 0.25
  xlim = [min(pos_ref[:, 0].min(), pos_actual[:, 0].min()) - pad, max(pos_ref[:, 0].max(), pos_actual[:, 0].max()) + pad]
  ylim = [min(pos_ref[:, 1].min(), pos_actual[:, 1].min()) - pad, max(pos_ref[:, 1].max(), pos_actual[:, 1].max()) + pad]
  zlim = [0.0, 1.6]

  frames = []
  stride = 2
  for step in range(0, traj_len, stride):
    fig = plt.figure(figsize=(12, 6.5), dpi=80)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1, 1], hspace=0.35, wspace=0.25)

    # 3D Slalom View
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_xlim(xlim); ax3d.set_ylim(ylim); ax3d.set_zlim(zlim)
    ax3d.set_xlabel("Forward X (m)", fontsize=10, labelpad=4)
    ax3d.set_ylabel("Lateral Y (m)", fontsize=10, labelpad=4)
    ax3d.set_zlabel("Altitude Z (m)", fontsize=10, labelpad=4)
    ax3d.view_init(elev=24, azim=-60)

    # Reference slalom spline
    ax3d.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], color="#3498db", ls="--", lw=2.0, label="Expert Slalom Path", alpha=0.85)

    # Slalom gates / waypoints
    slalom_gate_indices = [0, len(pos_ref)//4, len(pos_ref)//2, 3*len(pos_ref)//4, -1]
    for g_i, gi in enumerate(slalom_gate_indices):
      lbl = "Slalom Gates" if g_i == 0 else None
      ax3d.scatter([pos_ref[gi, 0]], [pos_ref[gi, 1]], [pos_ref[gi, 2]], color="#e74c3c", s=65, marker="^", label=lbl, zorder=5)

    # Actual history trail up to current step
    ax3d.plot(pos_actual[:step+1, 0], pos_actual[:step+1, 1], pos_actual[:step+1, 2], color="#27ae60", lw=2.6, label="SuperMimic Trajectory")

    # Draw Crazyflie drone at current position with attitude tilt R
    draw_crazyflie(ax3d, pos_actual[step], Rs[step])

    ax3d.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax3d.set_title(f"Agile Slalom 3D Tracking (t = {step*0.02:.2f} s)", fontsize=11, fontweight="bold")

    # Telemetry Panel 1: Position Tracking Error (cm)
    ax_err = fig.add_subplot(gs[0, 1])
    t_axis = np.arange(step + 1) * 0.02
    ax_err.plot(t_axis, errors_cm[:step+1], color="#27ae60", lw=2.0)
    ax_err.axhline(4.0, color="#e74c3c", ls="--", lw=1.2, label="Paper Target (4.0 cm)")
    ax_err.set_xlim([0, traj_len * 0.02])
    ax_err.set_ylim([0, 8.0])
    ax_err.set_ylabel("Tracking Error (cm)", fontsize=9)
    rmse_now = np.sqrt(np.mean(errors_cm[:step+1]**2))
    ax_err.set_title(f"Tracking Error: {errors_cm[step]:.2f} cm (RMSE: {rmse_now:.2f} cm)", fontsize=10, fontweight="bold", color="#27ae60" if errors_cm[step] < 4.0 else "#e74c3c")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="upper right", fontsize=8)

    # Telemetry Panel 2: Motor Thrusts & Body Tilt
    ax_rpm = fig.add_subplot(gs[1, 1])
    motor_names = ["M1 (FR)", "M2 (RR)", "M3 (RL)", "M4 (FL)"]
    colors = ["#e74c3c", "#2980b9", "#2980b9", "#e74c3c"]
    bars = ax_rpm.bar(motor_names, rpms[step], color=colors, alpha=0.85, width=0.55)
    ax_rpm.axhline(HOVER_RPM, color="#f39c12", ls="--", lw=1.2, label=f"Hover RPM ({HOVER_RPM:.0f})")
    ax_rpm.set_ylim([1000, 2700])
    ax_rpm.set_ylabel("Motor RPM", fontsize=9)
    ax_rpm.set_title(f"Active Motor Thrusts | Body Tilt: {tilts[step]:.1f}°", fontsize=10, fontweight="bold")
    ax_rpm.grid(True, alpha=0.3, axis="y")
    ax_rpm.legend(loc="upper right", fontsize=8)

    fig.canvas.draw()
    rgba = fig.canvas.buffer_rgba()
    frame = Image.frombuffer("RGBA", fig.canvas.get_width_height(), rgba, "raw", "RGBA", 0, 1)
    frames.append(frame.convert("RGB"))
    plt.close(fig)

  out_art = os.path.join(ARTIFACT_DIR, "video_high_speed_slalom.gif")
  out_loc = os.path.join(LOCAL_DIR, "video_high_speed_slalom.gif")
  frames[0].save(out_art, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
  frames[0].save(out_loc, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
  print(f"[SUCCESS] Saved {out_art} ({len(frames)} frames, {os.path.getsize(out_art)/(1024*1024):.2f} MB)", flush=True)


if __name__ == "__main__":
  render_slalom_video()
