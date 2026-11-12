"""Test waypoint navigation with velocity commands on trained Crazyflie policy."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import numpy as np
import torch
import mujoco

from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse
from supermimic.runners.crazyflie_runner import QuadcopterPolicy
import supermimic.tasks

CKPT_PATH = "/Users/apan/University/SuperMimicMimicked/supermimic/logs/crazyflie_supermimic/checkpoint_002500.pt"

def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    actor = QuadcopterPolicy(obs_dim=17, action_dim=4, hidden_dims=(128, 64))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env_cfg = load_env_cfg("SuperMimic-Crazyflie-Velocity", play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    obs, _ = env.reset()

    # Disable random Brownian resampling on the command term so we drive it
    cmd_term = env.command_manager.get_term("velocity")
    cmd_term._update_command = lambda env_ids=None: None
    cmd_term._resample_command = lambda env_ids: None

    # Waypoints: square / circuit
    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.5, 1.2]),
        np.array([0.0, 1.0, 1.4]),
        np.array([-0.8, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    wp_idx = 0
    reach_dist = 0.25

    print(f"[INFO] Running 200 steps of waypoint guidance...")
    with torch.inference_mode():
        for t in range(200):
            pos_w = env.sim.data.qpos[0, :3].clone()
            quat_w = env.sim.data.qpos[0, 3:7].clone().unsqueeze(0) # (1, 4)

            target = torch.tensor(waypoints[wp_idx], dtype=torch.float32)
            err_w = target - pos_w
            dist = torch.norm(err_w).item()

            if dist < reach_dist and wp_idx < len(waypoints) - 1:
                wp_idx += 1
                target = torch.tensor(waypoints[wp_idx], dtype=torch.float32)
                err_w = target - pos_w
                print(f"Step {t:3d}: Reached WP {wp_idx-1}, switching to WP {wp_idx} {waypoints[wp_idx]}")

            # Outer loop P velocity command in world frame
            kp = 1.2
            v_cmd_w = err_w * kp
            speed = torch.norm(v_cmd_w)
            max_speed = 1.0
            if speed > max_speed:
                v_cmd_w = (v_cmd_w / speed) * max_speed

            # Transform into body frame
            v_cmd_b = quat_apply_inverse(quat_w, v_cmd_w.unsqueeze(0)).squeeze(0) # (3,)
            cmd_4d = torch.cat([v_cmd_b, torch.tensor([0.0])]).unsqueeze(0) # (1, 4)

            cmd_term._vel_command[:] = cmd_4d
            obs["actor"][0, 13:17] = cmd_4d[0]

            action = actor(obs["actor"])
            next_obs, rewards, terminated, time_outs, extras = env.step(action)
            obs = next_obs

            if t % 20 == 0 or dist < reach_dist:
                print(f"Step {t:3d} | WP {wp_idx} | Pos: [{pos_w[0]:+.2f}, {pos_w[1]:+.2f}, {pos_w[2]:.2f}] | Dist: {dist:.2f}m | Cmd_b: [{v_cmd_b[0]:+.2f}, {v_cmd_b[1]:+.2f}, {v_cmd_b[2]:+.2f}]")

    print("[SUCCESS] Waypoint guidance test finished!")

if __name__ == "__main__":
    main()

