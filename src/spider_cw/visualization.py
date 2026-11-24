import os

import matplotlib.pyplot as plt
import numpy as np

import spider_cw.new as GA
from spider_cw.kinematics import foot_positions

LEG_NAMES = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
LEG_ORDER = [0, 6, 2, 4, 7, 1, 5, 3]


def plot_fitness(history, file_name="fitness.png"):
    """
    Best/Mean fitness over generations, plus each score of the best chromosome
    """
    save_path = os.path.join("visualisations", file_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["best"], label="best")
    ax1.plot(history["mean"], label="population mean")
    ax1.set(title="Fitness over generations", xlabel="generation", ylabel="fitness")
    ax1.legend()
    for k in GA.WEIGHTS:
        ax2.plot([b[k] for b in history["breakdown"]], label=k)
    ax2.set(
        title="Best chromosome's scores",
        xlabel="generation",
        ylabel="score (0-1)",
        ylim=(0, 1.05),
    )
    ax2.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)


def plot_gait(gait, file_name="gait.png"):
    """
    Coxa and Femur angles of every leg over the gait.
    """
    save_path = os.path.join("visualisations", file_name)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for leg in range(GA.N_LEGS):
        style = "-" if leg % 2 == 0 else "--"
        ax1.plot(gait[:, 3 * leg], style, label=LEG_NAMES[leg])
        ax2.plot(gait[:, 3 * leg + 1], style, label=LEG_NAMES[leg])
        ax3.plot(gait[:, 3 * leg + 2], style, label=LEG_NAMES[leg])
    ax1.set(title="Coxa angles (forward/back swing)", ylabel="radians")
    ax2.set(title="Femur angles (lift)", xlabel="frame", ylabel="radians")
    ax3.set(title="Tibia angles", xlabel="frame", ylabel="radians")
    ax1.legend(ncols=8, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)


def plot_footfall(gait, file_name="footfall.png"):
    """
    Stance (pushing back) vs swing for each leg over the gait.
    Stance = coxa moving backwards, matching lift_fitness.
    """
    save_path = os.path.join("visualisations", file_name)

    coxa = gait[:, 0::3] * GA.COXA_FORWARD_SIGN
    velocity = np.roll(coxa, -1, axis=0) - coxa
    stance = (velocity < 0)[:, LEG_ORDER]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(stance.T, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.axhline(3.5, color="tab:red", lw=1)
    ax.set(
        yticks=range(GA.N_LEGS),
        yticklabels=[LEG_NAMES[i] for i in LEG_ORDER],
        xlabel="frame",
        title="Footfall pattern (black = stance / pushing back)",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)


def plot_foot_paths(gait, file_name="foot_paths.png"):
    """
    Foot tip paths through one gait cycle, from above and from the side.
    """
    save_path = os.path.join("visualisations", file_name)

    feet = foot_positions(gait)

    fig, (top, side) = plt.subplots(1, 2, figsize=(12, 5))
    for leg in range(GA.N_LEGS):
        style = "-" if leg % 2 == 0 else "--"
        top.plot(feet[:, leg, 0], feet[:, leg, 1], style, label=LEG_NAMES[leg])
        side.plot(feet[:, leg, 0], feet[:, leg, 2], style, label=LEG_NAMES[leg])
        top.plot(*feet[0, leg, :2], "k.")

    top.set(title="Foot paths (top)", xlabel="x (forward)", ylabel="y", aspect="equal")
    side.set(title="Foot paths (side)", xlabel="x (forward)", ylabel="z (height)")
    side.legend(ncols=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)


def plot_diversity(history, file_name="diversity.png"):
    save_path = os.path.join("visualisations", file_name)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history["diversity"])
    ax.set(
        title="Population diversity",
        xlabel="generation",
        ylabel="mean gene std / joint range",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)


def plot_leg_spline(chromosome, leg=0, file_name="leg_spline.png"):
    """
    One leg's three joints: B-spline curve, keyframe control points, joint limits
    """
    save_path = os.path.join("visualisations", file_name.split(".")[0] + f"_{LEG_NAMES[leg]}.png")

    gait = GA.decode(chromosome)
    key_frames = np.arange(GA.N_KEYFRAMES) * GA.N_FRAMES / GA.N_KEYFRAMES

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for j, (ax, name) in enumerate(zip(axes, ["coxa", "femur", "tibia"])):
        col = 3 * leg + j
        ax.plot(gait[:, col], label="B-spline")
        ax.plot(key_frames, chromosome[:, col], "o", label="keyframes")
        ax.axhline(GA.JOINT_MIN[col], color="grey", ls=":")
        ax.axhline(GA.JOINT_MAX[col], color="grey", ls=":")
        ax.set(ylabel=f"{name} (rad)")

    axes[0].set_title(f"{LEG_NAMES[leg]} joints")
    axes[0].legend()
    axes[-1].set(xlabel="frame")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)

def plot_joint_trajectories(ga_gait, nn_gait, file_name="nn_joint_trajectories.png"):
    save_path = os.path.join("visualisations", file_name)

    frames = np.arange(GA.N_FRAMES)
    joint_names = ["Coxa (Swing)", "Femur (Lift)", "Tibia (Extension)"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Leg 1 Joint Trajectories: GA vs. Neural Network", fontsize=13)

    for j in range(3):
        axes[j].plot(frames, ga_gait[:, j], "--", label="GA", alpha=0.8)
        axes[j].plot(frames, nn_gait[:, j], "-", label="Neural Network", linewidth=2)
        axes[j].set_ylabel(f"{joint_names[j]}\n(rad)")
        axes[j].grid(True, alpha=0.3)
        if j == 0:
            axes[j].legend(loc="upper right")

    axes[-1].set_xlabel("Frame (0 - 299)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
