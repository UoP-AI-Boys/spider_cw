import matplotlib.pyplot as plt

import spider_cw.new as GA

LEG_NAMES = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]


def plot_fitness(history):
    """
    Best/Mean fitness over generations, plus each score of the best chromosome
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["best"], label="best")
    ax1.plot(history["mean"], label="population mean")
    ax1.set(title="Gitness over generations", xlabel="generation", ylabel="fitness")
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
    fig.savefig("visualisations/fitness.png", dpi=120)


def plot_gait(gait):
    """
    Coxa and Femur angles of every leg over the gait.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for leg in range(GA.N_LEGS):
        style = "-" if leg % 2 == 0 else "--"
        ax1.plot(gait[:, 3 * leg], style, label=LEG_NAMES[leg])
        ax2.plot(gait[:, 3 * leg + 1], style, label=LEG_NAMES[leg])
    ax1.set(title="Coxa angles (forward/back swing)", ylabel="radians")
    ax2.set(title="Femur angles (lift)", xlabel="frame", ylabel="radians")
    ax1.legend(ncols=8, fontsize=8)
    fig.tight_layout()
    fig.savefig("visualisations/gait.png", dpi=120)
