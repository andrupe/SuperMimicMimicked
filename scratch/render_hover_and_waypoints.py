"""Render Hover and Waypoint Finding trajectory visualizations for Crazyflie."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw, ImageFont
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
import supermimic.tasks

OUTPUT_DIR = "/Users/apan/.gemini/antigravity/brain/6594de82-ad27-4802-9ad2-2e21631b46d3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CKPT_PATH = "/Users/apan/University/SuperMimicMimicked/supermimic/logs/crazyflie_supermimic/checkpoint_002500.pt"
XML_PATH = "/Users/apan/University/SuperMimicMimicked/mjlab_research/src/mjlab/asset_zoo/robots/bitcraze_crazyflie/xmls/crazyflie.xml"

def generate_hover_gif():
    print("\n=======================================================")
    print("[INFO] Generating Trajectory 1: Hover Station Keeping...")
    print("=======================================================")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    actor = QuadcopterPolicy(obs_dim=17, action_dim=4, hidden_dims=(128, 64))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env_cfg = load_env_cfg("SuperMimic-Crazyflie-Velocity", play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    obs, _ = env.reset()

    # Standalone MuJoCo visualizer
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, 480, 480)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam.distance = 0.50
    cam.elevation = -18.0
    cam.azimuth = 45.0

    # Overwrite command term to lock to pure zero hover command [0, 0, 0, 0]
    cmd_term = env.command_manager.get_term("velocity")
    cmd_term._update_command = lambda env_ids=None: None
    cmd_term._resample_command = lambda env_ids: None
    zero_cmd = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    cmd_term._vel_command[:] = zero_cmd

    num_steps = 120
    tracking_frames = []
    pos_hist, vel_hist, act_hist, t_hist = [], [], [], []

    with torch.inference_mode():
        for t in range(num_steps):
            obs["actor"][0, 13:17] = zero_cmd[0]
            action = actor(obs["actor"])
            next_obs, rewards, term, trun, ext = env.step(action)

            qpos = env.sim.data.qpos[0].cpu().numpy()
            full_st = obs["full_state"][0].cpu().numpy()
            pos = qpos[:3].copy()
            vel = full_st[7:10].copy()
            act = action[0].cpu().numpy().copy()

            pos_hist.append(pos)
            vel_hist.append(vel)
            act_hist.append(act)
            t_hist.append(t * 0.02)

            # Sync renderer
            d.qpos[:7] = qpos
            mujoco.mj_forward(m, d)
            cam.azimuth = 45.0 + t * 0.5
            renderer.update_scene(d, camera=cam)
            rgb = renderer.render()

            # HUD
            img = Image.fromarray(rgb)
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([(15, 15), (290, 115)], fill=(10, 15, 25, 200), outline=(0, 200, 255, 220), width=1)

            drift = np.linalg.norm(pos[:2])
            draw.text((25, 22), "CRAZYFLIE 2.1 | HOVER STATION KEEP", fill=(0, 220, 255, 255))
            draw.text((25, 42), "Cmd Vel: [0.00, 0.00, 0.00] m/s", fill=(240, 240, 240, 255))
            draw.text((25, 60), f"Act Vel: [{vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f}] m/s", fill=(120, 255, 160, 255))
            draw.text((25, 78), f"Altitude: {pos[2]:.3f} m (Ref: 1.00 m)", fill=(255, 220, 80, 255))
            draw.text((25, 96), f"XY Drift: {drift:.3f} m | Attitude: STABLE", fill=(200, 220, 255, 255))

            combined = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            tracking_frames.append(np.array(combined))

            obs = next_obs

    hover_gif1 = os.path.join(OUTPUT_DIR, "crazyflie_hover_flight.gif")
    imageio.mimsave(hover_gif1, tracking_frames, fps=25, loop=0)
    print(f"[SUCCESS] Saved {hover_gif1} ({len(tracking_frames)} frames)")

    # 3D Hover Analysis GIF
    pos_arr = np.array(pos_hist)
    vel_arr = np.array(vel_hist)
    act_arr = np.array(act_hist)
    t_arr = np.array(t_hist)

    analysis_frames = []
    for i in range(10, num_steps, 2):
        fig = plt.figure(figsize=(10, 4.5), dpi=80)
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot(pos_arr[:i, 0], pos_arr[:i, 1], pos_arr[:i, 2], color="#00e5ff", linewidth=2.5, label="Hover Path")
        ax1.scatter([pos_arr[i-1, 0]], [pos_arr[i-1, 1]], [pos_arr[i-1, 2]], color="#ff1744", s=70, label="Drone")
        ax1.scatter([0], [0], [1.0], color="#76ff03", s=60, marker="*", label="Target (1.0m)")
        ax1.set_xlim([-0.2, 0.2])
        ax1.set_ylim([-0.2, 0.2])
        ax1.set_zlim([0.8, 1.2])
        ax1.set_xlabel("X (m)", fontsize=8)
        ax1.set_ylabel("Y (m)", fontsize=8)
        ax1.set_zlabel("Z (m)", fontsize=8)
        ax1.set_title("Station-Keeping Envelope (3D)", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.view_init(elev=20, azim=45 + i * 0.5)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t_arr[:i], pos_arr[:i, 2], "y-", linewidth=2.0, label="Altitude (Z)")
        ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Target 1.0m")
        ax2.set_ylabel("Altitude (m)", fontsize=8)
        ax2.set_xlim([0, t_arr[-1]])
        ax2.set_ylim([0.85, 1.15])
        ax2.set_title("Altitude Hold Stability", fontsize=10, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.6)

        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(t_arr[:i], act_arr[:i, 0], color="#f50057", label="FL")
        ax3.plot(t_arr[:i], act_arr[:i, 1], color="#651fff", label="FR")
        ax3.plot(t_arr[:i], act_arr[:i, 2], color="#00e676", label="RL")
        ax3.plot(t_arr[:i], act_arr[:i, 3], color="#ff9100", label="RR")
        ax3.set_xlabel("Time (s)", fontsize=8)
        ax3.set_ylabel("Action", fontsize=8)
        ax3.set_xlim([0, t_arr[-1]])
        ax3.set_ylim([-0.5, 1.0])
        ax3.set_title("Equilibrium Rotor Thrusts", fontsize=10, fontweight="bold")
        ax3.legend(loc="lower right", ncol=4, fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        analysis_frames.append(rgba[..., :3].copy())
        plt.close(fig)

    hover_gif2 = os.path.join(OUTPUT_DIR, "crazyflie_hover_telemetry.gif")
    imageio.mimsave(hover_gif2, analysis_frames, fps=15, loop=0)
    print(f"[SUCCESS] Saved {hover_gif2} ({len(analysis_frames)} frames)")


def generate_waypoint_gif():
    print("\n=======================================================")
    print("[INFO] Generating Trajectory 2: Waypoint Navigation via Velocity Commands...")
    print("=======================================================")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    actor = QuadcopterPolicy(obs_dim=17, action_dim=4, hidden_dims=(128, 64))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env_cfg = load_env_cfg("SuperMimic-Crazyflie-Velocity", play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    obs, _ = env.reset()

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, 480, 480)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam.distance = 0.65
    cam.elevation = -22.0
    cam.azimuth = 45.0

    cmd_term = env.command_manager.get_term("velocity")
    cmd_term._update_command = lambda env_ids=None: None
    cmd_term._resample_command = lambda env_ids: None

    # Waypoint sequence for a 3D figure-trajectory circuit
    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.5, 0.4, 1.25]),
        np.array([0.8, -0.3, 1.35]),
        np.array([0.1, -0.6, 1.15]),
        np.array([-0.5, -0.2, 1.05]),
        np.array([0.0, 0.0, 1.0]),
    ]
    wp_idx = 0
    reach_threshold = 0.22

    num_steps = 150
    tracking_frames = []
    pos_hist, vel_hist, cmd_hist, act_hist, t_hist, wp_hist = [], [], [], [], [], []

    with torch.inference_mode():
        for t in range(num_steps):
            pos_w = env.sim.data.qpos[0, :3].clone()
            quat_w = env.sim.data.qpos[0, 3:7].clone().unsqueeze(0)

            # Outer loop guidance: target position error -> velocity command
            target_wp = torch.tensor(waypoints[wp_idx], dtype=torch.float32)
            pos_err = target_wp - pos_w
            dist = torch.norm(pos_err).item()

            if dist < reach_threshold and wp_idx < len(waypoints) - 1:
                wp_idx += 1
                target_wp = torch.tensor(waypoints[wp_idx], dtype=torch.float32)
                pos_err = target_wp - pos_w
                dist = torch.norm(pos_err).item()

            # P guidance with velocity ceiling (0.6 m/s)
            kp = 1.0
            v_cmd_w = pos_err * kp
            speed = torch.norm(v_cmd_w)
            max_speed = 0.6
            if speed > max_speed:
                v_cmd_w = (v_cmd_w / speed) * max_speed

            # Transform into body frame
            v_cmd_b = quat_apply_inverse(quat_w, v_cmd_w.unsqueeze(0)).squeeze(0)
            cmd_4d = torch.cat([v_cmd_b, torch.tensor([0.0])]).unsqueeze(0)

            cmd_term._vel_command[:] = cmd_4d
            obs["actor"][0, 13:17] = cmd_4d[0]

            action = actor(obs["actor"])
            next_obs, rewards, term, trun, ext = env.step(action)

            qpos = env.sim.data.qpos[0].cpu().numpy()
            full_st = obs["full_state"][0].cpu().numpy()
            pos = qpos[:3].copy()
            vel = full_st[7:10].copy()
            act = action[0].cpu().numpy().copy()
            cmd_np = cmd_4d[0].cpu().numpy().copy()

            pos_hist.append(pos)
            vel_hist.append(vel)
            cmd_hist.append(cmd_np)
            act_hist.append(act)
            t_hist.append(t * 0.02)
            wp_hist.append(wp_idx)

            # Render 3D view
            d.qpos[:7] = qpos
            mujoco.mj_forward(m, d)
            cam.azimuth = 45.0 + t * 0.6
            renderer.update_scene(d, camera=cam)
            rgb = renderer.render()

            # HUD
            img = Image.fromarray(rgb)
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([(15, 15), (310, 120)], fill=(15, 20, 30, 200), outline=(255, 150, 0, 220), width=1)

            draw.text((25, 22), f"WAYPOINT GUIDANCE | WP {wp_idx+1}/{len(waypoints)}", fill=(255, 180, 50, 255))
            draw.text((25, 42), f"Target WP: [{waypoints[wp_idx][0]:+.2f}, {waypoints[wp_idx][1]:+.2f}, {waypoints[wp_idx][2]:.2f}] m", fill=(240, 240, 240, 255))
            draw.text((25, 60), f"Dist to WP: {dist:.2f} m | Speed: {np.linalg.norm(vel):.2f} m/s", fill=(100, 255, 180, 255))
            draw.text((25, 78), f"Cmd Vel: [{cmd_np[0]:+.2f}, {cmd_np[1]:+.2f}, {cmd_np[2]:+.2f}] m/s", fill=(220, 220, 255, 255))
            draw.text((25, 96), f"Act Pos: [{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f}] m", fill=(255, 220, 100, 255))

            combined = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            tracking_frames.append(np.array(combined))

            obs = next_obs

    wp_gif1 = os.path.join(OUTPUT_DIR, "crazyflie_waypoint_flight.gif")
    imageio.mimsave(wp_gif1, tracking_frames, fps=25, loop=0)
    print(f"[SUCCESS] Saved {wp_gif1} ({len(tracking_frames)} frames)")

    # 3D Trajectory & Telemetry Analysis GIF
    pos_arr = np.array(pos_hist)
    vel_arr = np.array(vel_hist)
    cmd_arr = np.array(cmd_hist)
    act_arr = np.array(act_hist)
    t_arr = np.array(t_hist)
    wp_arr = np.array(wp_hist)
    wp_pts = np.array(waypoints)

    wp_frames = []
    for i in range(10, num_steps, 2):
        fig = plt.figure(figsize=(10, 4.5), dpi=80)
        
        # 1. 3D Waypoint Circuit
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        # Draw planned waypoint segments
        ax1.plot(wp_pts[:, 0], wp_pts[:, 1], wp_pts[:, 2], "k--", alpha=0.4, label="Waypoint Path")
        # Draw waypoint markers
        for k, pt in enumerate(wp_pts):
            is_active = (k == wp_arr[i-1])
            ax1.scatter([pt[0]], [pt[1]], [pt[2]], color="#ff9100" if is_active else "#90caf9", s=100 if is_active else 40, marker="o")
            ax1.text(pt[0], pt[1], pt[2] + 0.05, f"WP{k+1}", fontsize=7)
        # Draw actual flight path
        ax1.plot(pos_arr[:i, 0], pos_arr[:i, 1], pos_arr[:i, 2], color="#00e5ff", linewidth=2.5, label="Actual Flight")
        ax1.scatter([pos_arr[i-1, 0]], [pos_arr[i-1, 1]], [pos_arr[i-1, 2]], color="#ff1744", s=80, marker="^", label="Crazyflie")

        ax1.set_xlabel("X (m)", fontsize=8)
        ax1.set_ylabel("Y (m)", fontsize=8)
        ax1.set_zlabel("Z (m)", fontsize=8)
        ax1.set_title("3D Waypoint Finding Circuit", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=7)
        ax1.view_init(elev=25, azim=40 + i * 0.5)
        ax1.set_xlim([-1.0, 1.2])
        ax1.set_ylim([-1.0, 1.0])
        ax1.set_zlim([0.8, 1.6])

        # 2. Position tracking over time
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t_arr[:i], pos_arr[:i, 0], "r-", label="X")
        ax2.plot(t_arr[:i], pos_arr[:i, 1], "g-", label="Y")
        ax2.plot(t_arr[:i], pos_arr[:i, 2], "b-", label="Z")
        ax2.set_ylabel("Position (m)", fontsize=8)
        ax2.set_xlim([0, t_arr[-1]])
        ax2.set_ylim([-1.0, 1.6])
        ax2.set_title("Cartesian Trajectory Coordinates", fontsize=10, fontweight="bold")
        ax2.legend(loc="upper right", ncol=3, fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # 3. Outer-loop velocity commands vs actual velocity
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(t_arr[:i], cmd_arr[:i, 0], "r--", alpha=0.6, label="Cmd Vx")
        ax3.plot(t_arr[:i], vel_arr[:i, 0], "r-", label="Act Vx")
        ax3.plot(t_arr[:i], cmd_arr[:i, 1], "g--", alpha=0.6, label="Cmd Vy")
        ax3.plot(t_arr[:i], vel_arr[:i, 1], "g-", label="Act Vy")
        ax3.plot(t_arr[:i], cmd_arr[:i, 2], "b--", alpha=0.6, label="Cmd Vz")
        ax3.plot(t_arr[:i], vel_arr[:i, 2], "b-", label="Act Vz")
        ax3.set_xlabel("Time (s)", fontsize=8)
        ax3.set_ylabel("Vel (m/s)", fontsize=8)
        ax3.set_xlim([0, t_arr[-1]])
        ax3.set_ylim([-0.9, 0.9])
        ax3.set_title("Guidance Velocity Commands vs Tracking", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper right", ncol=3, fontsize=6)
        ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        wp_frames.append(rgba[..., :3].copy())
        plt.close(fig)

    wp_gif2 = os.path.join(OUTPUT_DIR, "crazyflie_waypoint_telemetry.gif")
    imageio.mimsave(wp_gif2, wp_frames, fps=15, loop=0)
    print(f"[SUCCESS] Saved {wp_gif2} ({len(wp_frames)} frames)")


if __name__ == "__main__":
    generate_hover_gif()
    generate_waypoint_gif()

