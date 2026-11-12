import os
import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.tasks.tracking.mdp.commands import MotionLoader

model = mujoco.MjModel.from_xml_path(str(G1_XML))

# In G1 XML: joint 0 is root (free joint with 7 qpos, 6 qvel).
# The remaining 29 joints are hinges.
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
print(f"Loaded {len(joint_names)} joints from G1_XML:")
print(joint_names)

num_frames = 500
os.makedirs("scratch/mock_motion", exist_ok=True)
motion_path = os.path.abspath("scratch/mock_motion/g1_idle_motion.npz")

root_pos = np.zeros((num_frames, 3), dtype=np.float32)
root_pos[:, 2] = 0.79  # standard standing height
root_quat = np.zeros((num_frames, 4), dtype=np.float32)
root_quat[:, 0] = 1.0  # wxyz identity
root_lin_vel = np.zeros((num_frames, 3), dtype=np.float32)
root_ang_vel = np.zeros((num_frames, 3), dtype=np.float32)

joint_pos = np.zeros((num_frames, len(joint_names)), dtype=np.float32)
joint_vel = np.zeros((num_frames, len(joint_names)), dtype=np.float32)

np.savez(
    motion_path,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    root_pos_w=root_pos,
    root_quat_w=root_quat,
    root_lin_vel_w=root_lin_vel,
    root_ang_vel_w=root_ang_vel,
    joint_names=joint_names,
)
print("Successfully generated motion file at:", motion_path)
