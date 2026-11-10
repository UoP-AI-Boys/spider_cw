from datetime import datetime
from spider_cw.plot import animate_gait
import numpy as np

# Chromosome Encoding
N_FRAMES = 300  # Frames in final gait
N_KEYFRAMES = 8  # Keyframes in the chromosome
N_LEGS = 8  # Number of legs on the spider
N_JOINTS = 24  # Total number of joints

# JOINT_MIN = np.tile([-0.5, -1.2, -2.2], N_LEGS)  # shape (24,)
# JOINT_MAX = np.tile([0.5, 0.6, -0.6], N_LEGS)  # shape (24,)

JOINT_MIN = np.tile([-0.35, -1.10, -1.60], N_LEGS)  # shape (24,)
JOINT_MAX = np.tile([0.35, -0.50, -0.70], N_LEGS)  # shape (24,)

# Fitness Params
COXA_FORWARD_SIGN = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
FEMUR_UP_SIGN = 1
TARGET_STRIDE = 0.6
SMOOTH_SCALE = 1e-3
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
    center = (JOINT_MIN + JOINT_MAX) / 2
    spread = INITIAL_SPREAD * (JOINT_MAX - JOINT_MIN) / 2
    return rng.uniform(center - spread, center + spread, size=(N_KEYFRAMES, N_JOINTS))


def decode(chromosome):
    key_times = np.arange(N_KEYFRAMES) * (N_FRAMES / N_KEYFRAMES)
    frames = np.arange(N_FRAMES)

    gait = np.empty((N_FRAMES, N_JOINTS))
    for j in range(N_JOINTS):
        gait[:, j] = np.interp(frames, key_times, chromosome[:, j], period=N_FRAMES)

    return gait


# Fitness Stuff


def correlation(x, y):
    x, y = x - x.mean(), y - y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    return 0.0 if denom < 1e-12 else float((x * y).sum() / denom)


def stepping_fitness(coxa):
    stride = coxa.max(axis=0) - coxa.min(axis=0)
    error = np.abs(stride - TARGET_STRIDE) / TARGET_STRIDE
    return float(np.mean(np.exp(-4 * error**2)))


def coordination_fitness(coxa):
    corrs = [correlation(coxa[:, i], coxa[i, (i + 1) % N_LEGS]) for i in range(N_LEGS)]
    return float(np.mean([(1 - c) / 2 for c in corrs]))


def lift_fitness(coxa, femur):
    velocity = np.roll(coxa, -1, axis=0) - coxa
    corrs = [correlation(velocity[:, i], femur[:, i]) for i in range(N_LEGS)]
    return float(np.mean([(1 + c) / 2 for c in corrs]))


def smoothness_fitness(gait):
    accel = np.roll(gait, -1, axis=0) - 2 * gait + np.roll(gait, 1, axis=0)
    return float(np.exp(-np.mean(accel**2) / SMOOTH_SCALE))


def fitness_breakdown(chromosome):
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
    return fitness_breakdown(chromosome)["total"]


# GA Stuff


def tournament_select(population, fitnesses, rng):
    contenders = rng.choice(len(population), TOURNAMENT_SIZE, replace=False)
    winner = contenders[np.argmax(fitnesses[contenders])]
    return population[winner]


def crossover(parent1, parent2, rng):
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
    mask = rng.random(chromosome.shape) < MUTATION_RATE

    noise = rng.normal(
        0.0, MUTATION_STRENGTH * (JOINT_MAX - JOINT_MIN), size=chromosome.shape
    )

    return np.clip(chromosome + mask * noise, JOINT_MIN, JOINT_MAX)


def train_ga():
    rng = np.random.default_rng()
    max_fitness = sum(WEIGHTS.values())

    # Initialisation
    population = [random_chromosome(rng) for _ in range(POPULATION_SIZE)]
    history = {"best": [], "mean": [], "breakdown": []}
    best_ever, best_ever_fit, last_improvement = None, -np.inf, 0

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

    print(f"Best fitness: {best_ever_fit:.3f} / {max_fitness:.1f}")
    return best_ever, history


def main():
    run_date = datetime.now()

    best, history = train_ga()
    gait = decode(best)
    animate_gait(gait, save=f"gaits/best-{run_date}.gif")


if __name__ == "__main__":
    main()
