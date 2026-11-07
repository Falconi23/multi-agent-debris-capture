"""
Monte Carlo workspace of the three 3-DOF capture arms on the chaser front face.

Plots the reachable workspace (WS1) and a margin-shrunk subset (WS2) against
the hovering debris position used in adr_marl_env.py (2.1 m ahead of the
chaser centre). WS2 here is not the strict dextrous workspace: it keeps only
samples at least `margin` rad from every joint limit, as a proxy for poses
far from limits and singularities.

Run: python workspace.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

dims = np.array([0.8, 1.5, 0.8])  # X width, Y length, Z height [m]
arm_lengths = [0.8, 0.8, 0.4]

# Arm bases on the +Y face
base_top = np.array([ 0.0, dims[1]/2,  0.4])
base_sx  = np.array([-0.3, dims[1]/2, -0.3])
base_dx  = np.array([ 0.3, dims[1]/2, -0.3])

limit = np.pi / 2.0
margin = 0.20  # same joint-limit margin used in the arm_operator reward

target_pos = np.array([0.0, 2.1, 0.0])

# q1: yaw about the body Z axis, q2: shoulder pitch, q3: elbow pitch.
# Links 2 and 3 are collinear (both at q2+q3), same as the environment.
def fk_vectorized(bases, q1, q2, q3):
    x = bases[0] + arm_lengths[0]*np.sin(q1)*np.cos(q2) + arm_lengths[1]*np.sin(q1)*np.cos(q2+q3) + arm_lengths[2]*np.sin(q1)*np.cos(q2+q3)
    y = bases[1] + arm_lengths[0]*np.cos(q1)*np.cos(q2) + arm_lengths[1]*np.cos(q1)*np.cos(q2+q3) + arm_lengths[2]*np.cos(q1)*np.cos(q2+q3)
    z = bases[2] + arm_lengths[0]*np.sin(q2) + arm_lengths[1]*np.sin(q2+q3) + arm_lengths[2]*np.sin(q2+q3)
    return x, y, z

N = 100000

# Top arm shoulder bends up only, side arms down only
q1_t = np.random.uniform(-limit, limit, N)
q2_t = np.random.uniform(0.0, limit, N)
q3_t = np.random.uniform(-limit, limit, N)

q1_s = np.random.uniform(-limit, limit, N)
q2_s = np.random.uniform(-limit, 0.0, N)
q3_s = np.random.uniform(-limit, limit, N)

x_top, y_top, z_top = fk_vectorized(base_top, q1_t, q2_t, q3_t)
x_sx, y_sx, z_sx = fk_vectorized(base_sx, q1_s, q2_s, q3_s)
x_dx, y_dx, z_dx = fk_vectorized(base_dx, q1_s, q2_s, q3_s)

# WS2 masks: every joint at least `margin` rad from its limits
m_top = (np.abs(q1_t) < limit-margin) & (q2_t > margin) & (q2_t < limit-margin) & (np.abs(q3_t) < limit-margin)
m_side = (np.abs(q1_s) < limit-margin) & (q2_s < -margin) & (q2_s > -limit+margin) & (np.abs(q3_s) < limit-margin)

fig = plt.figure(figsize=(15, 8))
fig.canvas.manager.set_window_title("Kinematic Workspace Analysis - 3 Arms")

ax1 = fig.add_subplot(2, 2, 1)

# Thin slice around each base's X so the side view is a section, not a projection
sl_t = np.abs(x_top - base_top[0]) < 0.1
sl_s = np.abs(x_sx - base_sx[0]) < 0.1
sl_d = np.abs(x_dx - base_dx[0]) < 0.1

ax1.scatter(y_top[sl_t], z_top[sl_t], c='lightgray', s=1)
ax1.scatter(y_sx[sl_s], z_sx[sl_s], c='lightgray', s=1)
ax1.scatter(y_dx[sl_d], z_dx[sl_d], c='lightgray', s=1, label='WS1 (Reachable Area)')

ax1.scatter(y_top[sl_t & m_top], z_top[sl_t & m_top], c='blue', alpha=0.3, s=1)
ax1.scatter(y_sx[sl_s & m_side], z_sx[sl_s & m_side], c='blue', alpha=0.3, s=1)
ax1.scatter(y_dx[sl_d & m_side], z_dx[sl_d & m_side], c='blue', alpha=0.3, s=1, label='WS2 (Joint-Margin Workspace)')

ax1.add_patch(Rectangle((-dims[1]/2, -dims[2]/2), dims[1], dims[2], color='dimgray', alpha=0.8, zorder=5, label='Chaser Body'))
ax1.scatter([base_top[1], base_sx[1]], [base_top[2], base_sx[2]], c='black', s=40, zorder=6)
ax1.scatter(target_pos[1], target_pos[2], c='magenta', s=150, marker='*', zorder=10, edgecolor='black', label='Target')

ax1.set_title("Side View (Y-Z Section)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Y (Longitudinal Axis) [m]"); ax1.set_ylabel("Z (Height) [m]")
ax1.grid(True, linestyle='--', alpha=0.6); ax1.axis('equal')
ax1.legend(loc='upper left', fontsize=9)

ax2 = fig.add_subplot(2, 2, 3)

ax2.scatter(y_top, x_top, c='lightgray', s=0.1)
ax2.scatter(y_sx, x_sx, c='lightgray', s=0.1)
ax2.scatter(y_dx, x_dx, c='lightgray', s=0.1)

ax2.scatter(y_top[m_top], x_top[m_top], c='blue', alpha=0.05, s=0.1)
ax2.scatter(y_sx[m_side], x_sx[m_side], c='blue', alpha=0.05, s=0.1)
ax2.scatter(y_dx[m_side], x_dx[m_side], c='blue', alpha=0.05, s=0.1)

ax2.add_patch(Rectangle((-dims[1]/2, -dims[0]/2), dims[1], dims[0], color='dimgray', alpha=0.8, zorder=5))
ax2.scatter([base_top[1], base_sx[1], base_dx[1]], [base_top[0], base_sx[0], base_dx[0]], c='black', s=40, zorder=6)
ax2.scatter(target_pos[1], target_pos[0], c='magenta', s=150, marker='*', zorder=10, edgecolor='black')

ax2.set_title("Top View (Y-X Projection)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Y (Longitudinal Axis) [m]"); ax2.set_ylabel("X (Lateral Axis) [m]")
ax2.grid(True, linestyle='--', alpha=0.6); ax2.axis('equal')

ax3 = fig.add_subplot(1, 2, 2, projection='3d')

x_c, y_c, z_c = [-dims[0]/2, dims[0]/2], [-dims[1]/2, dims[1]/2], [-dims[2]/2, dims[2]/2]
xx, yy = np.meshgrid(x_c, y_c)
for z in z_c: ax3.plot_surface(xx, yy, np.full_like(xx, z), color='dimgray', alpha=0.6)
yy, zz = np.meshgrid(y_c, z_c)
for x in x_c: ax3.plot_surface(np.full_like(yy, x), yy, zz, color='dimgray', alpha=0.6)
xx, zz = np.meshgrid(x_c, z_c)
for y in y_c: ax3.plot_surface(xx, np.full_like(xx, y), zz, color='dimgray', alpha=0.6)

# Decimate for the 3D scatter, otherwise rendering crawls
step = 20

ax3.scatter(x_top[::step], y_top[::step], z_top[::step], c='cyan', alpha=0.02, s=1)
ax3.scatter(x_top[m_top][::step], y_top[m_top][::step], z_top[m_top][::step], c='blue', alpha=0.1, s=2)

ax3.scatter(x_sx[::step], y_sx[::step], z_sx[::step], c='lime', alpha=0.02, s=1)
ax3.scatter(x_sx[m_side][::step], y_sx[m_side][::step], z_sx[m_side][::step], c='darkgreen', alpha=0.1, s=2)

ax3.scatter(x_dx[::step], y_dx[::step], z_dx[::step], c='orange', alpha=0.02, s=1)
ax3.scatter(x_dx[m_side][::step], y_dx[m_side][::step], z_dx[m_side][::step], c='red', alpha=0.1, s=2)

ax3.scatter(*base_top, c='black', s=50); ax3.scatter(*base_sx, c='black', s=50); ax3.scatter(*base_dx, c='black', s=50)
ax3.scatter(*target_pos, c='magenta', s=300, marker='*', zorder=20, edgecolor='black')

ax3.set_title("Combined 3D Workspace (WS1 and WS2)", fontsize=12, fontweight='bold')
ax3.set_xlabel('X [m]'); ax3.set_ylabel('Y [m]'); ax3.set_zlabel('Z [m]')
ax3.set_xlim([-2.5, 2.5]); ax3.set_ylim([-1, 4.5]); ax3.set_zlim([-2, 2])
ax3.view_init(elev=20, azim=-50)

# 3D scatter handles don't produce usable legend entries, so build them by hand
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=8, label='Top Arm (WS1)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Top Arm (WS2)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=8, label='Left Arm (WS1)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', markersize=8, label='Left Arm (WS2)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Right Arm (WS1)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Right Arm (WS2)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', markeredgecolor='black', markersize=15, label='Target')
]
ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()