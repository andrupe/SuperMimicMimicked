"""Generate long-horizon (8.0 second, 400 step) trajectory GIFs for Crazyflie 2.1."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import supermimic.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
from supermimic.runners.crazyflie_runner import QuadcopterPolicy

OUTPUT_DIR = "/Users/apan/.gemini/antigravity/brain/6594de82-ad27-4802-9ad2-2e21631b46d3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CKPT_PATH = "/Users/apan/University/SuperMimicMimicked/supermimic/logs/crazyflie_supermimic/checkpoint_002500.pt"
XML_PATH = "/Users/apan/University/SuperMimicMimicked/mjlab_research/src/mjlab/asset_zoo/robots/bitcraze_crazyflie/xmls/crazyflie.xml"

TOTAL_STEPS = 400 # 400 steps * 0.02s = 8.0 seconds of continuous flight
FRAME_STRIDE = 2  # Render every 2nd step -> 200 frames at 25 fps = 8.0s playback

def run_long_waypoint_trajectory():
    print("\n=======================================================")
    print(f"[INFO] Trajectory 1: Long 8.0-Second Multi-Waypoint Circuit ({TOTAL_STEPS} steps)...")
    print("=======================================================")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    actor = QuadcopterPolicy(obs_dim=17, action_dim=4, hidden_dims=(128, 64))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env_cfg = load_env_cfg("SuperMimic-Crazyflie-Velocity", play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 20.0 # Prevent timeout reset
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

    # Comprehensive 9-waypoint 3D aerial navigation circuit
    waypoints = [
        np.array([0.0, 0.0, 1.0]),      # WP1: Hover Start
        np.array([0.6, 0.45, 1.25]),    # WP2: Climb forward-right
        np.array([0.8, -0.35, 1.45]),   # WP3: Bank right high
        np.array([0.2, -0.65, 1.20]),   # WP4: Descend rear-right
        np.array([-0.45, -0.40, 1.05]), # WP5: Cross left sweep
        np.array([-0.70, 0.25, 1.35]),  # WP6: Climb rear-left
        np.array([-0.20, 0.55, 1.40]),  # WP7: Forward bank high
        np.array([0.25, 0.20, 1.15]),   # WP8: Approach home
        np.array([0.0, 0.0, 1.0]),      # WP9: Final precision hover
    ]
    wp_idx = 0
    reach_threshold = 0.20

    tracking_frames = []
    pos_hist, vel_hist, cmd_hist, act_hist, t_hist, wp_hist = [], [], [], [], [], []

    with torch.inference_mode():
        for t in range(TOTAL_STEPS):
            pos_w = env.sim.data.qpos[0, :3].clone()
            quat_w = env.sim.data.qpos[0, 3:7].clone().unsqueeze(0)

            # Outer-loop P-guidance
            target_wp = torch.tensor(waypoints[wp_idx], dtype=torch.float32)
            pos_err = target_wp - pos_w
            dist = torch.norm(pos_err).item()

            if dist < reach_threshold and wp_idx < len(waypoints) - 1:
                wp_idx += 1
                target_wp = torch.tensor(waypoints[wp_idx], dtype=torch.float32)
                pos_err = target_wp - pos_w
                dist = torch.norm(pos_err).item()

            # Velocity command saturated at 0.55 m/s
            kp = 1.0
            v_cmd_w = pos_err * kp
            speed = torch.norm(v_cmd_w)
            max_speed = 0.55
            if speed > max_speed:
                v_cmd_w = (v_cmd_w / speed) * max_speed

            # Transform to body frame
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

            # Render frame on stride
            if t % FRAME_STRIDE == 0:
                d.qpos[:7] = qpos
                mujoco.mj_forward(m, d)
                cam.azimuth = 45.0 + (t * 0.35) # Smooth 360 camera pan
                renderer.update_scene(d, camera=cam)
                rgb = renderer.render()

                img = Image.fromarray(rgb)
                overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)

                # HUD box
                draw.rectangle([(15, 15), (320, 122)], fill=(12, 18, 28, 205), outline=(255, 160, 20, 230), width=1)
                sim_time = t * 0.02
                draw.text((25, 22), f"LONG WAYPOINT CIRCUIT | T = {sim_time:.2f}s / 8.0s", fill=(255, 180, 50, 255))
                draw.text((25, 42), f"Target WP {wp_idx+1}/9: [{waypoints[wp_idx][0]:+.2f}, {waypoints[wp_idx][1]:+.2f}, {waypoints[wp_idx][2]:.2f}] m", fill=(240, 240, 240, 255))
                draw.text((25, 60), f"Dist to WP: {dist:.2f} m | Speed: {np.linalg.norm(vel):.2f} m/s", fill=(100, 255, 180, 255))
                draw.text((25, 78), f"Cmd Vel: [{cmd_np[0]:+.2f}, {cmd_np[1]:+.2f}, {cmd_np[2]:+.2f}] m/s", fill=(220, 220, 255, 255))
                draw.text((25, 96), f"Position: [{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:.2f}] m", fill=(255, 230, 100, 255))

                combined = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                tracking_frames.append(np.array(combined))

            obs = next_obs

    gif_path1 = os.path.join(OUTPUT_DIR, "crazyflie_long_waypoint_flight.gif")
    print(f"[INFO] Writing {len(tracking_frames)} frames to {gif_path1}...")
    imageio.mimsave(gif_path1, tracking_frames, fps=25, loop=0)
    print(f"[SUCCESS] Saved {gif_path1}")

    # Telemetry Animation
    pos_arr = np.array(pos_hist)
    vel_arr = np.array(vel_hist)
    cmd_arr = np.array(cmd_hist)
    t_arr = np.array(t_hist)
    wp_arr = np.array(wp_hist)
    wp_pts = np.array(waypoints)

    telemetry_frames = []
    # Render every 4th step for telemetry GIF (100 frames @ 15 fps = 6.6s)
    for i in range(12, TOTAL_STEPS, 4):
        fig = plt.figure(figsize=(10, 4.5), dpi=80)

        # 1. 3D Waypoint Path
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot(wp_pts[:, 0], wp_pts[:, 1], wp_pts[:, 2], "k--", alpha=0.35, label="Planned Circuit")
        for k, pt in enumerate(wp_pts):
            is_active = (k == wp_arr[i-1])
            ax1.scatter([pt[0]], [pt[1]], [pt[2]], color="#ff9100" if is_active else "#90caf9", s=90 if is_active else 35)
            ax1.text(pt[0], pt[1], pt[2] + 0.04, f"W{k+1}", fontsize=7)
        ax1.plot(pos_arr[:i, 0], pos_arr[:i, 1], pos_arr[:i, 2], color="#00e5ff", linewidth=2.5, label="Flight Ribbon")
        ax1.scatter([pos_arr[i-1, 0]], [pos_arr[i-1, 1]], [pos_arr[i-1, 2]], color="#ff1744", s=75, marker="^", label="Crazyflie")

        ax1.set_xlabel("X (m)", fontsize=8)
        ax1.set_ylabel("Y (m)", fontsize=8)
        ax1.set_zlabel("Z (m)", fontsize=8)
        ax1.set_title("8-Second 3D Waypoint Trajectory", fontsize=10, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=7)
        ax1.view_init(elev=24, azim=38 + i * 0.35)
        ax1.set_xlim([-1.0, 1.1])
        ax1.set_ylim([-0.9, 0.9])
        ax1.set_zlim([0.8, 1.6])

        # 2. 3D Position tracking over 8 seconds
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t_arr[:i], pos_arr[:i, 0], "r-", label="X")
        ax2.plot(t_arr[:i], pos_arr[:i, 1], "g-", label="Y")
        ax2.plot(t_arr[:i], pos_arr[:i, 2], "b-", label="Z")
        ax2.set_ylabel("Position (m)", fontsize=8)
        ax2.set_xlim([0, t_arr[-1]])
        ax2.set_ylim([-1.0, 1.6])
        ax2.set_title("Cartesian Coordinates (0 - 8.0s)", fontsize=10, fontweight="bold")
        ax2.legend(loc="upper right", ncol=3, fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # 3. Outer-loop velocity command vs body velocity
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(t_arr[:i], cmd_arr[:i, 0], "r--", alpha=0.6, label="Cmd Vx")
        ax3.plot(t_arr[:i], vel_arr[:i, 0], "r-", label="Act Vx")
        ax3.plot(t_arr[:i], cmd_arr[:i, 1], "g--", alpha=0.6, label="Cmd Vy")
        ax3.plot(t_arr[:i], vel_arr[:i, 1], "g-", label="Act Vy")
        ax3.plot(t_arr[:i], cmd_arr[:i, 2], "b--", alpha=0.6, label="Cmd Vz")
        ax3.plot(t_arr[:i], vel_arr[:i, 2], "b-", label="Act Vz")
        ax3.set_xlabel("Time (s)", fontsize=8)
        ax3.set_ylabel("Velocity (m/s)", fontsize=8)
        ax3.set_xlim([0, t_arr[-1]])
        ax3.set_ylim([-0.8, 0.8])
        ax3.set_title("Outer-Loop Velocity Command Tracking", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper right", ncol=3, fontsize=6)
        ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        telemetry_frames.append(rgba[..., :3].copy())
        plt.close(fig)

    gif_path2 = os.path.join(OUTPUT_DIR, "crazyflie_long_waypoint_telemetry.gif")
    print(f"[INFO] Writing {len(telemetry_frames)} frames to {gif_path2}...")
    imageio.mimsave(gif_path2, telemetry_frames, fps=15, loop=0)
    print(f"[SUCCESS] Saved {gif_path2}")


def run_long_brownian_trajectory():
    print("\n=======================================================")
    print(f"[INFO] Trajectory 2: Long 8.0-Second Brownian Velocity Random Walk ({TOTAL_STEPS} steps)...")
    print("=======================================================")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    actor = QuadcopterPolicy(obs_dim=17, action_dim=4, hidden_dims=(128, 64))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env_cfg = load_env_cfg("SuperMimic-Crazyflie-Velocity", play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 20.0
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    obs, _ = env.reset()

    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, 480, 480)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam.distance = 0.60
    cam.elevation = -20.0
    cam.azimuth = 45.0

    tracking_frames = []
    pos_hist, vel_hist, cmd_hist, act_hist, t_hist = [], [], [], [], []

    with torch.inference_mode():
        for t in range(TOTAL_STEPS):
            action = actor(obs["actor"])
            next_obs, rewards, term, trun, ext = env.step(action)

            qpos = env.sim.data.qpos[0].cpu().numpy()
            full_st = obs["full_state"][0].cpu().numpy()
            pos = qpos[:3].copy()
            vel = full_st[7:10].copy()
            cmd = obs["goal"][0].cpu().numpy().copy()
            act = action[0].cpu().numpy().copy()

            pos_hist.append(pos)
            vel_hist.append(vel)
            cmd_hist.append(cmd)
            act_hist.append(act)
            t_hist.append(t * 0.02)

            if t % FRAME_STRIDE == 0:
                d.qpos[:7] = qpos
                mujoco.mj_forward(m, d)
                cam.azimuth = 45.0 + (t * 0.4)
                renderer.update_scene(d, camera=cam)
                rgb = renderer.render()

                img = Image.fromarray(rgb)
                overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)

                draw.rectangle([(15, 15), (310, 118)], fill=(15, 15, 25, 200), outline=(0, 180, 255, 220), width=1)
                sim_time = t * 0.02
                g_z = -obs["actor"][0, 2].item()
                draw.text((25, 22), f"BROWNIAN FLIGHT | T = {sim_time:.2f}s / 8.0s", fill=(0, 220, 255, 255))
                draw.text((25, 42), f"Cmd Vel: [{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}] m/s", fill=(240, 240, 240, 255))
                draw.text((25, 60), f"Act Vel: [{vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f}] m/s", fill=(120, 255, 160, 255))
                draw.text((25, 78), f"Altitude: {pos[2]:.2f} m | Speed: {np.linalg.norm(vel):.2f} m/s", fill=(240, 220, 80, 255))
                draw.text((25, 96), f"Attitude: {'UPRIGHT' if g_z > 0.75 else 'BANKING'} (gz: {g_z:.2f})", fill=(255, 200, 120, 255))

                combined = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                tracking_frames.append(np.array(combined))

            obs = next_obs

    gif_path3 = os.path.join(OUTPUT_DIR, "crazyflie_long_brownian_flight.gif")
    print(f"[INFO] Writing {len(tracking_frames)} frames to {gif_path3}...")
    imageio.mimsave(gif_path3, tracking_frames, fps=25, loop=0)
    print(f"[SUCCESS] Saved {gif_path3}")

    # Telemetry Animation
    pos_arr = np.array(pos_hist)
    vel_arr = np.array(vel_hist)
    cmd_arr = np.array(cmd_hist)
    act_arr = np.array(act_hist)
    t_arr = np.array(t_hist)

    telemetry_frames = []
    for i in range(12, TOTAL_STEPS, 4):
        fig = plt.figure(figsize=(10, 4.5), dpi=80)

        # 1. 3D Flight Ribbon
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot(pos_arr[:i, 0], pos_arr[:i, 1], pos_arr[:i, 2], color="#00e5ff", linewidth=2.5, label="Stochastic Ribbon")
        ax1.scatter([pos_arr[i-1, 0]], [pos_arr[i-1, 1]], [pos_arr[i-1, 2]], color="#ff1744", s=75, marker="o", label="Crazyflie")
        ax1.scatter([pos_arr[0, 0]], [pos_arr[0, 1]], [pos_arr[0, 2]], color="#76ff03", s=40, marker="^", label="Origin")

        ax1.set_xlabel("X (m)", fontsize=8)
        ax1.set_ylabel("Y (m)", fontsize=8)
        ax1.set_zlabel("Z (m)", fontsize=8)
        ax1.set_title("8-Second 3D Brownian Walk Trail", fontsize=10, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=7)
        ax1.view_init(elev=25, azim=40 + i * 0.35)
        pad = 0.3
        ax1.set_xlim([min(pos_arr[:, 0].min(), -0.5) - pad, max(pos_arr[:, 0].max(), 0.5) + pad])
        ax1.set_ylim([min(pos_arr[:, 1].min(), -0.5) - pad, max(pos_arr[:, 1].max(), 0.5) + pad])
        ax1.set_zlim([0.0, max(pos_arr[:, 2].max(), 1.5) + pad])

        # 2. Velocity Tracking Performance
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t_arr[:i], cmd_arr[:i, 0], "r--", alpha=0.6, label="Cmd Vx")
        ax2.plot(t_arr[:i], vel_arr[:i, 0], "r-", label="Act Vx")
        ax2.plot(t_arr[:i], cmd_arr[:i, 1], "g--", alpha=0.6, label="Cmd Vy")
        ax2.plot(t_arr[:i], vel_arr[:i, 1], "g-", label="Act Vy")
        ax2.plot(t_arr[:i], cmd_arr[:i, 2], "b--", alpha=0.6, label="Cmd Vz")
        ax2.plot(t_arr[:i], vel_arr[:i, 2], "b-", label="Act Vz")
        ax2.set_ylabel("Velocity (m/s)", fontsize=8)
        ax2.set_xlim([0, t_arr[-1]])
        ax2.set_ylim([-1.6, 1.6])
        ax2.set_title("3D Velocity Random Walk Tracking", fontsize=10, fontweight="bold")
        ax2.legend(loc="upper right", ncol=3, fontsize=6)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # 3. 4-Rotor Thrust Control Demands
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(t_arr[:i], act_arr[:i, 0], color="#f50057", label="FL")
        ax3.plot(t_arr[:i], act_arr[:i, 1], color="#651fff", label="FR")
        ax3.plot(t_arr[:i], act_arr[:i, 2], color="#00e676", label="RL")
        ax3.plot(t_arr[:i], act_arr[:i, 3], color="#ff9100", label="RR")
        ax3.set_xlabel("Time (s)", fontsize=8)
        ax3.set_ylabel("Action", fontsize=8)
        ax3.set_xlim([0, t_arr[-1]])
        ax3.set_ylim([-1.0, 1.0])
        ax3.set_title("4-Rotor Control Actions", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper right", ncol=4, fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        telemetry_frames.append(rgba[..., :3].copy())
        plt.close(fig)

    gif_path4 = os.path.join(OUTPUT_DIR, "crazyflie_long_brownian_telemetry.gif")
    print(f"[INFO] Writing {len(telemetry_frames)} frames to {gif_path4}...")
    imageio.mimsave(gif_path4, telemetry_frames, fps=15, loop=0)
    print(f"[SUCCESS] Saved {gif_path4}")


if __name__ == "__main__":
    run_long_waypoint_trajectory()
    run_long_brownian_trajectory()

