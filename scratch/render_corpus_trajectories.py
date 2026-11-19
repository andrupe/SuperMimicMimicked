"""Render high-fidelity GIFs of the Crazyflie policy trained on the normal corpus."""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import sys
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
from mjlab.utils.lab_api.math import quat_apply
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
from supermimic.tasks.crazyflie_mdp import CorpusTrajectoryCommand

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/andrupe/.gemini/antigravity-cli/brain/b2d41161-778e-44e8-a60a-c51d518b1e94")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEFAULT_CKPT_DIR = os.path.join(WORKSPACE_ROOT, "logs/crazyflie_supermimic")
XML_PATH = os.path.join(WORKSPACE_ROOT, "mjlab_research/src/mjlab/asset_zoo/robots/bitcraze_crazyflie/xmls/crazyflie.xml")


def render_trajectory(actor: QuadcopterPolicy, env: ManagerBasedRlEnv, traj_id: int, task_name: str):
    print(f"\n[INFO] Starting rollout for trajectory {traj_id} ({task_name})...")
    obs, _ = env.reset()
    cmd_term: CorpusTrajectoryCommand = env.unwrapped.command_manager.get_term("velocity")

    # Lock environment to the selected trajectory
    cmd_term.traj_indices[0] = traj_id
    cmd_term.time_indices[0] = 0
    traj = cmd_term.trajectories[traj_id]
    H = int(traj["H"])

    # Reference State Initialization to trajectory start
    q0 = traj["quat"][0].unsqueeze(0)
    vb0 = traj["lin_vel_b"][0].unsqueeze(0)
    wb0 = traj["ang_vel_b"][0].unsqueeze(0)
    vw0 = quat_apply(q0, vb0).squeeze(0)
    ww0 = quat_apply(q0, wb0).squeeze(0)
    origin = env.unwrapped.scene.env_origins[0]
    p0 = traj["pos"][0] + origin

    # Initialize step 0 commands in command manager
    cmd_term._vel_command[0, :3] = traj["lin_vel_b"][0]
    cmd_term._vel_command[0, 3] = traj["ang_vel_b"][0, 2]
    cmd_term._ref_pos[0] = traj["pos"][0]
    cmd_term._ref_quat[0] = traj["quat"][0]

    root_state0 = torch.cat([p0, q0.squeeze(0), vw0, ww0], dim=-1).unsqueeze(0)
    env.unwrapped.scene["robot"].write_root_state_to_sim(root_state0)
    env.unwrapped.sim.forward()

    # Invalidate cached obs buffer so actor gets true step 0 observation
    env.unwrapped.observation_manager._obs_buffer = None
    obs = env.get_observations()

    # Setup MuJoCo renderer (optional, falls back gracefully in headless envs)
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    renderer = None
    try:
        renderer = mujoco.Renderer(m, 480, 480)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        cam.distance = 0.65
        cam.elevation = -20.0
        cam.azimuth = 45.0
    except Exception as e:
        print(f"[WARN] Headless 3D visual renderer unavailable ({e}). Continuing with telemetry and quantitative metrics.")

    # Histories for telemetry and trajectory plot
    ref_pos_hist = []
    act_pos_hist = []
    ref_vel_hist = []
    act_vel_hist = []
    act_hist = []
    time_hist = []
    flight_frames = []

    print(f"[INFO] Rolling out {H} steps ({H * 0.02:.2f}s flight)...")
    term_step = H
    term_reason = "completed"

    with torch.inference_mode():
        for t in range(H):
            actor_obs = obs["actor"]
            action = actor(actor_obs)

            # Store states before step
            qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy()
            full_st = obs["full_state"][0].cpu().numpy()

            act_p = qpos[:3].copy()
            ref_p = traj["pos"][t].cpu().numpy().copy()
            act_v = full_st[7:10].copy()  # body lin vel
            ref_v = traj["lin_vel_b"][t].cpu().numpy().copy()
            u = action[0].cpu().numpy().copy()

            act_pos_hist.append(act_p)
            ref_pos_hist.append(ref_p)
            act_vel_hist.append(act_v)
            ref_vel_hist.append(ref_v)
            act_hist.append(u)
            time_hist.append(t * 0.02)

            # Render 3D view if renderer available
            if renderer is not None:
                d.qpos[:7] = qpos
                mujoco.mj_forward(m, d)
                cam.azimuth = 45.0 + t * 0.35
                renderer.update_scene(d, camera=cam)
                rgb = renderer.render()

                # HUD overlay
                img = Image.fromarray(rgb)
                overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)

                # Background HUD panel
                draw.rectangle([(15, 15), (320, 125)], fill=(12, 18, 30, 200), outline=(40, 140, 255, 230), width=1)
                pos_err = np.linalg.norm(act_p - ref_p)
                speed = np.linalg.norm(act_v)

                draw.text((25, 22), f"CRAZYFLIE | CORPUS {task_name.upper()}", fill=(110, 210, 255, 255))
                draw.text((25, 42), f"Step {t:03d}/{H} | Time: {t*0.02:.2f}s | Speed: {speed:.2f} m/s", fill=(240, 240, 240, 255))
                draw.text((25, 62), f"Ref Vel: [{ref_v[0]:+.2f}, {ref_v[1]:+.2f}, {ref_v[2]:+.2f}] m/s", fill=(180, 220, 255, 255))
                draw.text((25, 82), f"Act Vel: [{act_v[0]:+.2f}, {act_v[1]:+.2f}, {act_v[2]:+.2f}] m/s", fill=(130, 255, 160, 255))
                draw.text((25, 102), f"Pos Error: {pos_err*100:.1f} cm | Alt: {act_p[2]:.2f} m", fill=(255, 215, 90, 255))

                combined = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                flight_frames.append(np.array(combined))

            obs, rewards, terminated, time_outs, extras = env.step(action)
            if terminated.any():
                term_step = t + 1
                term_reason = "terminated (tilt/crash)"
                print(f"[WARN] Flight terminated at step {term_step}/{H} ({term_step * 0.02:.2f}s) on {term_reason}!")
                break

    # Quantitative summary
    act_p_arr = np.array(act_pos_hist)
    ref_p_arr = np.array(ref_pos_hist)
    act_v_arr = np.array(act_vel_hist)
    ref_v_arr = np.array(ref_vel_hist)
    act_u_arr = np.array(act_hist)
    t_arr = np.array(time_hist)

    pos_errs = np.linalg.norm(act_p_arr - ref_p_arr, axis=-1)
    vel_errs = np.linalg.norm(act_v_arr - ref_v_arr, axis=-1)
    mean_act = np.mean(act_u_arr, axis=0)

    print(f"\n--- {task_name.upper()} RESULTS ---")
    print(f"Steps survived: {term_step}/{H} ({term_reason})")
    print(f"Altitude range z: [{act_p_arr[:, 2].min():.3f}, {act_p_arr[:, 2].max():.3f}] m")
    print(f"Mean position error: {pos_errs.mean():.3f} m (final: {pos_errs[-1]:.3f} m)")
    print(f"Mean velocity error: {vel_errs.mean():.3f} m/s (final: {vel_errs[-1]:.3f} m/s)")
    print(f"Mean actions: [{mean_act[0]:.3f}, {mean_act[1]:.3f}, {mean_act[2]:.3f}, {mean_act[3]:.3f}]")

    # 1. Save Flight GIF if frames exist
    flight_gif_path = None
    if flight_frames:
        flight_gif_path = os.path.join(OUTPUT_DIR, f"crazyflie_corpus_{task_name}_flight.gif")
        print(f"[INFO] Saving flight animation GIF to {flight_gif_path}...")
        imageio.mimsave(flight_gif_path, flight_frames, fps=25, loop=0)
        print(f"[SUCCESS] Saved {flight_gif_path}")

    # 2. Render Telemetry & Trajectory Overlay GIF
    print(f"[INFO] Generating 3D Trajectory & Telemetry comparison GIF...")
    act_p_arr = np.array(act_pos_hist)
    ref_p_arr = np.array(ref_pos_hist)
    act_v_arr = np.array(act_vel_hist)
    ref_v_arr = np.array(ref_vel_hist)
    act_u_arr = np.array(act_hist)
    t_arr = np.array(time_hist)

    telemetry_frames = []
    step_stride = 2
    for i in range(10, H, step_stride):
        fig = plt.figure(figsize=(11, 4.5), dpi=80)

        # 3D Trajectory: Ground Truth Expert vs Actual Flight Path
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot(ref_p_arr[:, 0], ref_p_arr[:, 1], ref_p_arr[:, 2], "k--", linewidth=1.5, alpha=0.6, label="Expert SSQP Target")
        ax1.plot(act_p_arr[:i, 0], act_p_arr[:i, 1], act_p_arr[:i, 2], color="#00bcd4", linewidth=2.5, label="Policy Flight Path")
        # Current drone position
        ax1.scatter([act_p_arr[i-1, 0]], [act_p_arr[i-1, 1]], [act_p_arr[i-1, 2]], color="#ff5722", s=70, marker="o", label="Crazyflie")
        ax1.scatter([act_p_arr[0, 0]], [act_p_arr[0, 1]], [act_p_arr[0, 2]], color="#4caf50", s=40, marker="^", label="Start")

        ax1.set_xlabel("X (m)", fontsize=8)
        ax1.set_ylabel("Y (m)", fontsize=8)
        ax1.set_zlabel("Z (m)", fontsize=8)
        ax1.set_title(f"3D Path: Expert vs Policy ({task_name.title()})", fontsize=10, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=7)
        # Fixed isometric view for stable orientation perspective
        ax1.view_init(elev=28, azim=-65)

        pad = 0.2
        x_min = min(ref_p_arr[:, 0].min(), act_p_arr[:, 0].min()) - pad
        x_max = max(ref_p_arr[:, 0].max(), act_p_arr[:, 0].max()) + pad
        y_min = min(ref_p_arr[:, 1].min(), act_p_arr[:, 1].min()) - pad
        y_max = max(ref_p_arr[:, 1].max(), act_p_arr[:, 1].max()) + pad
        z_min = max(0.0, min(ref_p_arr[:, 2].min(), act_p_arr[:, 2].min()) - pad)
        z_max = max(ref_p_arr[:, 2].max(), act_p_arr[:, 2].max()) + pad

        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])
        ax1.set_zlim([z_min, z_max])
        try:
            ax1.set_box_aspect([max(0.1, x_max - x_min), max(0.1, y_max - y_min), max(0.1, z_max - z_min)])
        except Exception:
            pass

        # Velocity tracking: Vx, Vy, Vz
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t_arr[:i], ref_v_arr[:i, 0], "r--", alpha=0.7, label="Vx Ref")
        ax2.plot(t_arr[:i], act_v_arr[:i, 0], "r-", linewidth=1.6, label="Vx Act")
        ax2.plot(t_arr[:i], ref_v_arr[:i, 1], "g--", alpha=0.7, label="Vy Ref")
        ax2.plot(t_arr[:i], act_v_arr[:i, 1], "g-", linewidth=1.6, label="Vy Act")
        ax2.plot(t_arr[:i], ref_v_arr[:i, 2], "b--", alpha=0.7, label="Vz Ref")
        ax2.plot(t_arr[:i], act_v_arr[:i, 2], "b-", linewidth=1.6, label="Vz Act")
        ax2.set_ylabel("Body Vel (m/s)", fontsize=8)
        ax2.set_xlim([0, t_arr[-1]])
        ax2.set_ylim([-1.5, 1.5])
        ax2.set_title("Expert Velocity Command Tracking", fontsize=9, fontweight="bold")
        ax2.legend(loc="upper right", ncol=3, fontsize=6.5)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # 4-Rotor Thrust Control Actions
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(t_arr[:i], act_u_arr[:i, 0], color="#e91e63", label="FL")
        ax3.plot(t_arr[:i], act_u_arr[:i, 1], color="#9c27b0", label="FR")
        ax3.plot(t_arr[:i], act_u_arr[:i, 2], color="#3f51b5", label="RL")
        ax3.plot(t_arr[:i], act_u_arr[:i, 3], color="#ff9800", label="RR")
        ax3.set_xlabel("Time (s)", fontsize=8)
        ax3.set_ylabel("Rotor Control", fontsize=8)
        ax3.set_xlim([0, t_arr[-1]])
        ax3.set_ylim([-1.1, 1.1])
        ax3.set_title("4-Rotor BPTT Thrust Demands", fontsize=9, fontweight="bold")
        ax3.legend(loc="upper right", ncol=4, fontsize=6.5)
        ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        telemetry_frames.append(rgba[..., :3].copy())
        plt.close(fig)

    telemetry_gif_path = os.path.join(OUTPUT_DIR, f"crazyflie_corpus_{task_name}_telemetry.gif")
    print(f"[INFO] Saving telemetry comparison GIF to {telemetry_gif_path}...")
    imageio.mimsave(telemetry_gif_path, telemetry_frames, fps=15, loop=0)
    print(f"[SUCCESS] Saved {telemetry_gif_path}")

    return flight_gif_path, telemetry_gif_path


def main():
    import glob
    ckpt_path = os.environ.get("CKPT_PATH")
    if not ckpt_path:
        ckpts = sorted(glob.glob(os.path.join(DEFAULT_CKPT_DIR, "checkpoint_*.pt")))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {DEFAULT_CKPT_DIR}")
        ckpt_path = ckpts[-1]
    print(f"[INFO] Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actor = QuadcopterPolicy(obs_dim=17, action_dim=4, hidden_dims=(128, 64))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    print("[INFO] Creating SuperMimic-Crazyflie-Corpus simulation environment...")
    env_cfg = load_env_cfg("SuperMimic-Crazyflie-Corpus", play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")

    # Render 1: Figure-8 trajectory (Traj 5)
    render_trajectory(actor, env, traj_id=5, task_name="figure8")

    # Render 2: Helix trajectory (Traj 3)
    render_trajectory(actor, env, traj_id=3, task_name="helix")

    print("\n[ALL DONE] All corpus trajectory GIFs successfully generated!")


if __name__ == "__main__":
    main()
