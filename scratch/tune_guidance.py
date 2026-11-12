"""Tune outer-loop velocity guidance for waypoint tracking."""

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

def test_guidance(kp=1.0, max_speed=0.6):
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    actor = QuadcopterPolicy(obs_dim=17, action_dim=4, hidden_dims=(128, 64))
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env_cfg = load_env_cfg("SuperMimic-Crazyflie-Velocity", play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    obs, _ = env.reset()

    cmd_term = env.command_manager.get_term("velocity")
    cmd_term._update_command = lambda env_ids=None: None
    cmd_term._resample_command = lambda env_ids: None

    waypoints = [
        np.array([0.5, 0.3, 1.2]),
        np.array([0.0, 0.6, 1.3]),
        np.array([-0.4, 0.2, 1.1]),
        np.array([0.0, 0.0, 1.0]),
    ]
    wp_idx = 0
    reach_dist = 0.15

    print(f"\n--- Testing Guidance kp={kp}, max_speed={max_speed} ---")
    with torch.inference_mode():
        for t in range(120):
            pos_w = env.sim.data.qpos[0, :3].clone()
            quat_w = env.sim.data.qpos[0, 3:7].clone().unsqueeze(0)

            target = torch.tensor(waypoints[wp_idx], dtype=torch.float32)
            err_w = target - pos_w
            dist = torch.norm(err_w).item()

            if dist < reach_dist and wp_idx < len(waypoints) - 1:
                print(f"Step {t:3d}: Reached WP {wp_idx} -> switching to WP {wp_idx+1}")
                wp_idx += 1
                target = torch.tensor(waypoints[wp_idx], dtype=torch.float32)
                err_w = target - pos_w

            # P guidance
            v_cmd_w = err_w * kp
            speed = torch.norm(v_cmd_w)
            if speed > max_speed:
                v_cmd_w = (v_cmd_w / speed) * max_speed

            v_cmd_b = quat_apply_inverse(quat_w, v_cmd_w.unsqueeze(0)).squeeze(0)
            cmd_4d = torch.cat([v_cmd_b, torch.tensor([0.0])]).unsqueeze(0)

            cmd_term._vel_command[:] = cmd_4d
            obs["actor"][0, 13:17] = cmd_4d[0]

            action = actor(obs["actor"])
            next_obs, rewards, term, trun, ext = env.step(action)
            obs = next_obs

            if term or trun:
                print(f"Step {t:3d}: Reset/Terminated! pos={pos_w.tolist()}")
                return False

            if t % 25 == 0:
                print(f"Step {t:3d} | WP {wp_idx} | Pos: [{pos_w[0]:+.2f}, {pos_w[1]:+.2f}, {pos_w[2]:.2f}] | Dist: {dist:.2f}m")

    print("[SUCCESS] Flight completed smoothly!")
    return True

if __name__ == "__main__":
    test_guidance(kp=0.8, max_speed=0.5)

