import os
import threading
import time
from datetime import datetime

import numpy as np

from spider_cw.plot import animate_gait
from spider_cw.visualization import plot_fitness, plot_gait

# Script Params
GAITS_SAVE_DIR = os.getenv("GAITS_SAVE_DIR", "gaits")
DEBUG_GAIT_PROGRESS = os.getenv("DEBUG_GAIT_PROGRESS", "False").lower() == "true"
MAX_RENDER_THREADS = int(os.getenv("MAX_RENDER_THREADS", "4"))

# Chromosome Encoding
N_FRAMES = 300  # Frames in final gait
N_KEYFRAMES = 8  # Keyframes in the chromosome
N_LEGS = 8  # Number of legs on the spider
N_JOINTS = 24  # Total number of joints

JOINT_MIN = np.tile([-0.35, -1.10, -1.60], N_LEGS)  # shape (24,)
JOINT_MAX = np.tile([0.35, -0.50, -0.70], N_LEGS)  # shape (24,)

# Fitness Params
COXA_FORWARD_SIGN = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
FEMUR_UP_SIGN = 1
TARGET_STRIDE = 0.6
SMOOTH_SCALE = 1e-3  # bigger = allows more jerky movement
WEIGHTS = {"stepping": 1.0, "coordination": 2.0, "lift": 1.0, "smoothness": 1.0}

# Genetic Algorithm Params
POPULATION_SIZE = 150
MAX_GENERATIONS = 500
TOURNAMENT_SIZE = 3  # bigger = stronger selection pressure
CROSSOVER_RATE = 0.9  # chance two parents are crossed (else copied)
MUTATION_RATE = 0.05  # chance each gene is mutated
MUTATION_STRENGTH = 0.1  # mutation std, as a fraction of the joint's range
NUM_ELITE = 3  # best N chromosomes copied unchanged each generation
INITIAL_SPREAD = 0.25  # initial angles within +-25% of range around the centre
STALL_GENERATIONS = 100  # stop if the best fitness hasn't improved for this long


# Chromosome Stuff
def random_chromosome(rng):
    """
    Random Chromosome

    Angles are spread around the middle of the joint's ranges.
    """
    center = (JOINT_MIN + JOINT_MAX) / 2
    spread = INITIAL_SPREAD * (JOINT_MAX - JOINT_MIN) / 2
    return rng.uniform(center - spread, center + spread, size=(N_KEYFRAMES, N_JOINTS))


def decode(chromosome):
    """
    Decodes a keyframe'd chromosome to a full gait

    (N_KEYFRAMES, 24) chromosome -> (N_FRAMES, 24) gait
    Keyframe k sits at frame k * (300 / N_KEYFRAMES). For each joint,
    frames in between are linearly interpolated. period=N_FRAMES makes it
    wrap round, so the last frame blends into the first frame
    """
    key_times = np.arange(N_KEYFRAMES) * (N_FRAMES / N_KEYFRAMES)
    frames = np.arange(N_FRAMES)

    gait = np.empty((N_FRAMES, N_JOINTS))
    for j in range(N_JOINTS):
        gait[:, j] = np.interp(frames, key_times, chromosome[:, j], period=N_FRAMES)

    return gait


# Fitness Stuff
def correlation(x, y):
    """
    How similarly 2 values move over time, from -1 to 1.

    1 = rise and fall together (in sync)
    -1 = one rises while the other falls (opposite)
    0 = no relation
    """
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def stepping_fitness(coxa):
    """
    Each leg's coxa swing (max - min) should match TARGET_STRIDE
    prevents the GA from settling on a gait that stands still.
    """
    stride = coxa.max(axis=0) - coxa.min(axis=0)
    error = np.abs(stride - TARGET_STRIDE) / TARGET_STRIDE
    return float(np.mean(np.exp(-4 * error**2)))


def coordination_fitness(coxa):
    """
    Neighbouring legs should be out of sync
    Correlation -1 -> score 1
    Correlation 1  -> score 0
    """
    corrs = [correlation(coxa[:, i], coxa[:, (i + 1) % N_LEGS]) for i in range(N_LEGS)]
    return float(np.mean([(1 - c) / 2 for c in corrs]))


def lift_fitness(coxa, femur):
    """
    A leg should be raised while swinging forward and down
    while pushing back. Correlate coxa velocity with femur
    height. np.roll wraps around because the gait loops.
    """
    velocity = np.roll(coxa, -1, axis=0) - coxa
    corrs = [correlation(velocity[:, i], femur[:, i]) for i in range(N_LEGS)]
    return float(np.mean([(1 + c) / 2 for c in corrs]))


def smoothness_fitness(gait):
    """
    Penalise sudden jerks.
    Average squared acceleration between frames.
    """
    accel = np.roll(gait, -1, axis=0) - 2 * gait + np.roll(gait, 1, axis=0)
    return float(np.exp(-np.mean(accel**2) / SMOOTH_SCALE))


def fitness_breakdown(chromosome):
    """
    All 4 Scores plus weighted total
    """
    gait = decode(chromosome)

    coxa = gait[:, 0::3] * COXA_FORWARD_SIGN
    femur = gait[:, 1::3] * FEMUR_UP_SIGN

    scores = {
        "stepping": stepping_fitness(coxa),
        "coordination": coordination_fitness(coxa),
        "lift": lift_fitness(coxa, femur),
        "smoothness": smoothness_fitness(gait),
    }

    scores["total"] = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
    return scores


def fitness(chromosome):
    """Overall fitness for the chromosome. Best possible = sum of WEIGHTS"""
    return fitness_breakdown(chromosome)["total"]


# GA Stuff
def tournament_select(population, fitnesses, rng):
    """
    Pick TOURNAMENT_SIZE random chromosomes; return fittest.
    """
    contenders = rng.choice(len(population), TOURNAMENT_SIZE, replace=False)
    winner = contenders[np.argmax(fitnesses[contenders])]
    return population[winner]


def crossover(parent1, parent2, rng):
    """
    Leg-wise uniform crossover.
    """
    child1, child2 = parent1.copy(), parent2.copy()

    if rng.random() > CROSSOVER_RATE:
        return child1, child2

    for leg in range(N_LEGS):
        if rng.random() < 0.5:
            cols = slice(3 * leg, 3 * leg + 3)
            child1[:, cols] = parent2[:, cols]
            child2[:, cols] = parent1[:, cols]

    return child1, child2


def mutate(chromosome, rng):
    """
    Gaussian mutation
    Each gene, with chance MUTATION_RATE, gets random noise of
    size MUTATION_STRENGTH * the joints range. Clipped to the joints limits
    """
    mask = rng.random(chromosome.shape) < MUTATION_RATE

    noise = rng.normal(
        0.0, MUTATION_STRENGTH * (JOINT_MAX - JOINT_MIN), size=chromosome.shape
    )

    return np.clip(chromosome + mask * noise, JOINT_MIN, JOINT_MAX)


def train_ga():
    """
    Train the genetic algorithm

    returns the best chromosome and generation history
    """
    rng = np.random.default_rng()
    max_fitness = sum(WEIGHTS.values())

    # Initialisation
    population = [random_chromosome(rng) for _ in range(POPULATION_SIZE)]
    history = {"best": [], "mean": [], "breakdown": []}
    best_ever, best_ever_fit, last_improvement = None, -np.inf, 0

    threads = []

    for gen in range(MAX_GENERATIONS):
        fitnesses = np.array([fitness(c) for c in population])
        best_idx = int(np.argmax(fitnesses))

        if fitnesses[best_idx] > best_ever_fit + 1e-6:
            best_ever = population[best_idx].copy()
            best_ever_fit = fitnesses[best_idx]
            last_improvement = gen

        history["best"].append(fitnesses[best_idx])
        history["mean"].append(fitnesses.mean())
        history["breakdown"].append(fitness_breakdown(population[best_idx]))

        if gen % 25 == 0:
            b = history["breakdown"][-1]
            if DEBUG_GAIT_PROGRESS:
                run_date = str(datetime.now()).replace(" ", "_")
                gait = decode(population[best_idx])

                threads = [x for x in threads if x.is_alive()]

                wait_start = None
                while len(threads) >= MAX_RENDER_THREADS:
                    if wait_start is None:
                        wait_start = time.perf_counter()
                        print(
                            "[DEBUG | gait_animation]: Too many render threads alive waiting for thread to finish..."
                        )
                    threads[0].join()
                    threads = [x for x in threads if x.is_alive()]

                if wait_start is not None:
                    print(
                        f"[DEBUG | gait_animation]: Blocked for {time.perf_counter() - wait_start:.2f}s waiting for a render slot"
                    )

                animation_thread = threading.Thread(
                    target=animate_gait,
                    kwargs={
                        "chromosome": gait,
                        "title": f"generation {gen}",
                        "save": f"gaits/gen-{gen}_{run_date}.gif",
                    },
                )
                animation_thread.start()
                threads.append(animation_thread)

            print(
                f"gen {gen:4d} | best {fitnesses[best_idx]:.3f} / {max_fitness:.1f} "
                f"| mean {fitnesses.mean():.3f} | "
                + " ".join(f"{k[:5]} {b[k]:.2f}" for k in WEIGHTS)
            )

        # Termination: near-perfect, or no improvement for STALL_GENERATIONS
        if best_ever_fit >= 0.99 * max_fitness:
            print(f"Stopping at gen {gen}: reached 99% of max fitness.")
            break
        if gen - last_improvement >= STALL_GENERATIONS:
            print(
                f"Stopping at gen {gen}: no improvement for {STALL_GENERATIONS} generations."
            )
            break

        elite_idx = np.argsort(fitnesses)[-NUM_ELITE:]
        next_population = [population[i].copy() for i in elite_idx]

        while len(next_population) < POPULATION_SIZE:
            p1 = tournament_select(population, fitnesses, rng)
            p2 = tournament_select(population, fitnesses, rng)
            c1, c2 = crossover(p1, p2, rng)
            next_population.append(mutate(c1, rng))
            if len(next_population) < POPULATION_SIZE:
                next_population.append(mutate(c2, rng))

        population = next_population

    for thread in threads:
        thread.join()

    print(f"Best fitness: {best_ever_fit:.3f} / {max_fitness:.1f}")
    return best_ever, history


def main():
    os.makedirs("gaits", exist_ok=True)
    run_date = str(datetime.now()).replace(" ", "_")

    best, history = train_ga()
    best_gait = decode(best)

    animate_gait(best_gait, save=f"gaits/best-{run_date}.gif")

    os.makedirs("visualisations", exist_ok=True)
    plot_fitness(history)
    plot_gait(best_gait)


if __name__ == "__main__":
    main()
