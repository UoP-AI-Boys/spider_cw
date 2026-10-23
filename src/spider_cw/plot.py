import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from spider_cw.kinematics import BODY_A, BODY_B, LEG_ORDER, leg_joints, BASE_ANGLES


def animate_gait(chromosome, interval=33, save=None, ground_z=None):
    genes = np.array(chromosome.genes)
    if genes.ndim == 1:
        genes = genes[None, :]
    joints = leg_joints(genes)
    num_frames = len(joints)

    # Copied from plot_spider_pose.m
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set(
        xlim=(-4, 4), ylim=(-4, 4), zlim=(-2, 2),
        xlabel="X", ylabel="Y", zlabel="Z",
    )
    ax.set_box_aspect((1, 1, 0.5))

    # Body
    t = np.linspace(0, 2 * np.pi, 100)
    body_x = BODY_A * np.cos(t)
    body_y = BODY_B * np.sin(t)
    ax.plot(body_x, body_y, np.zeros_like(t), "k-", lw=3)

    # Head marker
    # ax.plot(BODY_A + 0.2, 0, 0, "r^", markersize=10, markerfacecolor="r")

    if ground_z is not None:
        gx, gy = np.meshgrid([-4, 4], [-4, 4])
        ax.plot_surface(gx, gy, np.full_like(gx, ground_z, dtype=float), alpha=0.15, color="tab:green")

    for name, base_angle, leg in zip(LEG_ORDER, BASE_ANGLES, joints[0]):
        pos = leg[0] + 0.2 * np.array([np.cos(base_angle), np.sin(base_angle), 0])
        ax.text(pos[0], pos[1], pos[2] + 0.05, name, fontsize=10, fontweight="bold")

    segments = []
    for _ in LEG_ORDER:
        segments.append([ax.plot([], [], [], color=c, lw=2)[0] for c in ("k", "b", "r")])
    feet = ax.scatter([], [], [], color="r", s=20)
    title = ax.set_title("")

    def update(frame):
        for lines, leg in zip(segments, joints[frame]):
            for s, line in enumerate(lines):
                line.set_data_3d(leg[s:s + 2, 0], leg[s:s + 2, 1], leg[s:s + 2, 2])
        foot = joints[frame, :, -1, :]
        feet._offsets3d = (foot[:, 0], foot[:, 1], foot[:, 2])
        title.set_text(f"Spider gait - frame {frame + 1} of {num_frames}")
        return [line for lines in segments for line in lines] + [feet, title]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval)

    if save:
        print(f"Saving {num_frames} frames to {save}...")
        anim.save(save, fps=round(1000 / interval))
        print("Done.")
    else:
        plt.show()
    return anim
