"""Generate high-quality animated GIFs of the trained Crazyflie quadcopter tracking trajectories."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import sys
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
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
import supermimic.tasks

OUTPUT_DIR = "/Users/apan/.gemini/antigravity/brain/6594de82-ad27-4802-9ad2-2e21631b46d3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CKPT_PATH = "/Users/apan/University/SuperMimicMimicked/supermimic/logs/crazyflie_supermimic/checkpoint_002500.pt"
XML_PATH = "/Users/apan/University/SuperMimicMimicked/mjlab_research/src/mjlab/asset_zoo/robots/bitcraze_crazyflie/xmls/crazyflie.xml"

def main():
    print("[INFO] Loading trained policy checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    actor = QuadcopterPolicy(obs_dim=17, action_dim=4, hidden_dims=(128, 64))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    print("[INFO] Creating simulation environment...")
    env_cfg = load_env_cfg("SuperMimic-Crazyflie-Velocity", play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    obs, _ = env.reset()

    # Load standalone MuJoCo model for camera rendering
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, 480, 480)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam.distance = 0.55
    cam.elevation = -22.0
    cam.azimuth = 45.0

    num_steps = 150
    tracking_frames = []
    
    # Telemetry histories
    pos_hist = []
    vel_hist = []
    cmd_hist = []
    act_hist = []
    time_hist = []

    print(f"[INFO] Simulating and recording {num_steps} flight steps...")
    with torch.inference_mode():
        for t in range(num_steps):
            action = actor(obs["actor"])
            next_obs, rewards, terminated, time_outs, extras = env.step(action)

            # Read simulation states
            qpos = env.sim.data.qpos[0].cpu().numpy()
            full_st = obs["full_state"][0].cpu().numpy()
            pos = qpos[:3].copy()
            quat = qpos[3:7].copy()
            vel = full_st[7:10].copy() # lin vel
            cmd = obs["goal"][0].cpu().numpy().copy() # cmd [vx, vy, vz, wz]
            act = action[0].cpu().numpy().copy()

            pos_hist.append(pos)
            vel_hist.append(vel)
            cmd_hist.append(cmd)
            act_hist.append(act)
            time_hist.append(t * 0.02)

            # Sync to MuJoCo visualizer
            d.qpos[:7] = qpos
            mujoco.mj_forward(m, d)

            # Orbit camera slightly around the drone for dynamic visual perspective
            cam.azimuth = 45.0 + t * 0.4
            renderer.update_scene(d, camera=cam)
            rgb = renderer.render()

            # Add sleek HUD overlay using Pillow
            img = Image.fromarray(rgb)
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            # HUD background box
            draw.rectangle([(15, 15), (280, 115)], fill=(10, 15, 25, 190), outline=(50, 130, 240, 220), width=1)
            
            # HUD text
            g_z = -obs["actor"][0, 2].item()
            speed = np.linalg.norm(vel)
            
            draw.text((25, 22), "CRAZYFLIE 2.1 | SUPERMIMIC BPTT", fill=(100, 200, 255, 255))
            draw.text((25, 42), f"Cmd Vel: [{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}] m/s", fill=(240, 240, 240, 255))
            draw.text((25, 60), f"Act Vel: [{vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f}] m/s", fill=(120, 255, 150, 255))
            draw.text((25, 78), f"Altitude: {pos[2]:.2f} m | Speed: {speed:.2f} m/s", fill=(220, 220, 220, 255))
            draw.text((25, 96), f"Attitude: {'UPRIGHT' if g_z > 0.75 else 'TILTING'} (gz: {g_z:.2f})", fill=(255, 200, 80, 255))

            combined = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            tracking_frames.append(np.array(combined))

            obs = next_obs

    gif_path1 = os.path.join(OUTPUT_DIR, "crazyflie_tracking_flight.gif")
    print(f"[INFO] Writing 3D tracking GIF to {gif_path1}...")
    imageio.mimsave(gif_path1, tracking_frames, fps=25, loop=0)
    print(f"[SUCCESS] Saved {gif_path1} ({len(tracking_frames)} frames)")

    # -------------------------------------------------------------
    # GIF 2: 3D Trajectory Ribbon & Telemetry Graphs
    # -------------------------------------------------------------
    print("[INFO] Generating 3D Trajectory Ribbon & Telemetry GIF...")
    pos_arr = np.array(pos_hist)
    vel_arr = np.array(vel_hist)
    cmd_arr = np.array(cmd_hist)
    act_arr = np.array(act_hist)
    t_arr = np.array(time_hist)

    traj_frames = []
    # Render every 2nd frame for trajectory plot to keep GIF compact
    for i in range(10, num_steps, 2):
        fig = plt.figure(figsize=(10, 4.5), dpi=80)
        
        # 1. 3D Trajectory space
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot(pos_arr[:i, 0], pos_arr[:i, 1], pos_arr[:i, 2], color="#00bcd4", linewidth=2.5, label="Flight Path")
        # Draw drone marker
        ax1.scatter([pos_arr[i-1, 0]], [pos_arr[i-1, 1]], [pos_arr[i-1, 2]], color="#ff5722", s=80, marker="o", label="Crazyflie")
        # Start marker
        ax1.scatter([pos_arr[0, 0]], [pos_arr[0, 1]], [pos_arr[0, 2]], color="#4caf50", s=40, marker="^", label="Origin")
        
        ax1.set_xlabel("X (m)", fontsize=9)
        ax1.set_ylabel("Y (m)", fontsize=9)
        ax1.set_zlabel("Z (m)", fontsize=9)
        ax1.set_title("3D Flight Trajectory Trail", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.view_init(elev=25, azim=45 + i * 0.4)
        
        # Keep consistent limits
        pad = 0.3
        ax1.set_xlim([min(pos_arr[:, 0].min(), -0.5) - pad, max(pos_arr[:, 0].max(), 0.5) + pad])
        ax1.set_ylim([min(pos_arr[:, 1].min(), -0.5) - pad, max(pos_arr[:, 1].max(), 0.5) + pad])
        ax1.set_zlim([0.0, max(pos_arr[:, 2].max(), 1.5) + pad])

        # 2. Velocity Tracking Plot
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t_arr[:i], cmd_arr[:i, 0], "r--", alpha=0.7, label="Vx Cmd")
        ax2.plot(t_arr[:i], vel_arr[:i, 0], "r-", linewidth=1.8, label="Vx Act")
        ax2.plot(t_arr[:i], cmd_arr[:i, 1], "g--", alpha=0.7, label="Vy Cmd")
        ax2.plot(t_arr[:i], vel_arr[:i, 1], "g-", linewidth=1.8, label="Vy Act")
        ax2.plot(t_arr[:i], cmd_arr[:i, 2], "b--", alpha=0.7, label="Vz Cmd")
        ax2.plot(t_arr[:i], vel_arr[:i, 2], "b-", linewidth=1.8, label="Vz Act")
        ax2.set_ylabel("Vel (m/s)", fontsize=8)
        ax2.set_xlim([0, t_arr[-1]])
        ax2.set_ylim([-1.6, 1.6])
        ax2.set_title("3D Velocity Tracking Performance", fontsize=10, fontweight="bold")
        ax2.legend(loc="upper right", ncol=3, fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # 3. 4-Rotor Thrust Control Actions
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(t_arr[:i], act_arr[:i, 0], color="#e91e63", label="Rotor FL")
        ax3.plot(t_arr[:i], act_arr[:i, 1], color="#9c27b0", label="Rotor FR")
        ax3.plot(t_arr[:i], act_arr[:i, 2], color="#3f51b5", label="Rotor RL")
        ax3.plot(t_arr[:i], act_arr[:i, 3], color="#ff9800", label="Rotor RR")
        ax3.set_xlabel("Time (s)", fontsize=8)
        ax3.set_ylabel("Action [-1, 1]", fontsize=8)
        ax3.set_xlim([0, t_arr[-1]])
        ax3.set_ylim([-1.1, 1.1])
        ax3.set_title("Rotor Thrust Control Demands", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper right", ncol=4, fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        
        # Convert fig to numpy frame
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frame = rgba[..., :3].copy()
        traj_frames.append(frame)
        plt.close(fig)

    gif_path2 = os.path.join(OUTPUT_DIR, "crazyflie_trajectory_plot.gif")
    print(f"[INFO] Writing 3D telemetry plot GIF to {gif_path2}...")
    imageio.mimsave(gif_path2, traj_frames, fps=15, loop=0)
    print(f"[SUCCESS] Saved {gif_path2} ({len(traj_frames)} frames)")


if __name__ == "__main__":
    main()
