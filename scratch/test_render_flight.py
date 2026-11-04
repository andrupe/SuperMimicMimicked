"""
Prototype renderer for Crazyflie 3D flight visualization.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

def draw_drone_3d(ax, pos, R, arm_len=0.046, rotor_r=0.023):
    # pos: (3,), R: (3, 3)
    # Crazyflie arm vectors in body frame (X-configuration at 45 deg)
    d = arm_len / np.sqrt(2.0)
    motors_body = np.array([
        [ d, -d, 0.0],  # M1: front-right
        [-d, -d, 0.0],  # M2: rear-right
        [-d,  d, 0.0],  # M3: rear-left
        [ d,  d, 0.0],  # M4: front-left
    ])
    # Transform to world frame
    motors_world = pos + (R @ motors_body.T).T

    # Draw arms (M2 to M4, M1 to M3)
    ax.plot([motors_world[1, 0], motors_world[3, 0]],
            [motors_world[1, 1], motors_world[3, 1]],
            [motors_world[1, 2], motors_world[3, 2]],
            color="black", lw=2.5)
    ax.plot([motors_world[0, 0], motors_world[2, 0]],
            [motors_world[0, 1], motors_world[2, 1]],
            [motors_world[0, 2], motors_world[2, 2]],
            color="black", lw=2.5)

    # Draw central body
    ax.scatter([pos[0]], [pos[1]], [pos[2]], color="darkblue", s=40, zorder=5)

    # Draw rotor discs
    theta = np.linspace(0, 2 * np.pi, 16)
    circle_pts = np.stack([rotor_r * np.cos(theta), rotor_r * np.sin(theta), np.zeros_like(theta)], axis=0) # (3, 16)
    
    for i in range(4):
        # Front motors (M1, M4) orange/red, rear motors (M2, M3) blue/cyan
        col = "#e74c3c" if i in [0, 3] else "#3498db"
        disc_world = motors_world[i][:, None] + (R @ circle_pts)
        ax.plot(disc_world[0], disc_world[1], disc_world[2], color=col, lw=1.5)
        ax.scatter([motors_world[i, 0]], [motors_world[i, 1]], [motors_world[i, 2]], color=col, s=15)

    # Draw ground projection shadow
    ax.scatter([pos[0]], [pos[1]], [0.0], color="gray", alpha=0.3, s=35)
    ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0.0, pos[2]], color="gray", ls=":", lw=1.0, alpha=0.5)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
pos = np.array([0.5, 0.5, 1.0])
R = np.eye(3)
draw_drone_3d(ax, pos, R)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1.5])
ax.set_title("Crazyflie Drone Test Render")
fig.canvas.draw()
rgba = fig.canvas.buffer_rgba()
img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), rgba, "raw", "RGBA", 0, 1)
img.save("scratch/test_drone.png")
print("Rendered scratch/test_drone.png successfully!")
