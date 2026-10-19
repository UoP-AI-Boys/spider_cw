import copy
import random as rn

import numpy as np

from spider_cw.kinematics import BASE_POSITIONS, foot_positions

LEG_ORDER = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINT_BOUNDS = [(-0.5, 0.5), (-1.2, 0.6), (-2.2, -0.6)]
JOINT_WIDTHS = [high - low for low, high in JOINT_BOUNDS]
NUM_GENES = len(LEG_ORDER) * 3

GROUND_Z = -0.7
TARGET_REACH = 1.9
WEIGHTS = {"ground": 1, "reach": 0.5, "symmetry": 0.5}

# Parameters
POPULATION_SIZE = 150
MAX_GENERATIONS = 100
TOURNAMENT_SIZE = 3
NUM_ELITE = 2
BASE_MUTATION_RATE = 0.05
MAX_MUTATION_RATE = 0.3
MUTATION_SIGMA = 0.1
RANDOM_DIVERSITY = 0.3


class Genome:
    def __init__(self, genes):
        self.genes = list(genes)

    @classmethod
    def random(cls):
        return cls(rn.uniform(*JOINT_BOUNDS[i % 3]) for i in range(NUM_GENES))

    def clamp(self):
        for i, gene in enumerate(self.genes):
            low_bound, high_bound = JOINT_BOUNDS[i % 3]
            self.genes[i] = max(low_bound, min(high_bound, gene))
        return self

    def leg(self, name):
        i = LEG_ORDER.index(name) * 3
        return self.genes[i : i + 3]

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, index):
        return self.genes[index]


def generate_genome() -> Genome:
    """Generates a random genome."""
    return Genome.random()


def generate_population(size=150):
    """generates a single generation of genomes"""
    population = []
    for i in range(size):
        genome = generate_genome()
        population.append(genome)
    return population


def fitness_term(genome: Genome):
    feet = foot_positions(genome)

    ground = np.mean((feet[:, 2] - GROUND_Z) ** 2)

    reach_dist = np.linalg.norm(feet[:, :2] - BASE_POSITIONS[:, :2], axis=1)
    reach = np.mean((reach_dist - TARGET_REACH) ** 2)

    left = feet[:4]
    right = feet[4:][::-1] * [1, -1, 1]

    symmetry = np.mean((left - right) ** 2)

    return {"ground": ground, "reach": reach, "symmetry": symmetry}


def fitness(genome: Genome):
    terms = fitness_term(genome)
    total = sum(WEIGHTS[k] * v for k, v in terms.items())
    return 1.0 / (1.0 + total)


def tournament_selection(population, fitness_values, tournament_size=TOURNAMENT_SIZE):
    contenders = rn.sample(range(len(population)), tournament_size)

    return population[max(contenders, key=lambda i: fitness_values[i])]


def population_diversity(population):
    genes = np.array([genome.genes for genome in population])
    widths = np.array([JOINT_WIDTHS[i % 3] for i in range(NUM_GENES)])
    return float(np.mean(genes.std() / widths))


def mutation_rate(population, base=BASE_MUTATION_RATE, top=MAX_MUTATION_RATE):
    convergence = max(0.0, 1 - population_diversity(population) / RANDOM_DIVERSITY)
    return base + (top - base) * convergence

def mutate(genome, mutation_rate, sigma=MUTATION_SIGMA):
    for i in range(NUM_GENES):
        if rn.random() < mutation_rate:
            genome.genes[i] += rn.gauss(0, sigma * JOINT_WIDTHS[i % 3])
    return genome.clamp()

def crossover(parent_one, parent_two, mutation_rate):
    if len(parent_one) != len(parent_two):
        raise ValueError("parent genomes must have the same length")

    crossover_point = rn.randint(1, NUM_GENES - 1)
    child_one = Genome(parent_one.genes[:crossover_point] + parent_two.genes[crossover_point:])
    child_two = Genome(parent_two.genes[:crossover_point] + parent_one.genes[crossover_point:])
    return mutate(child_one, mutation_rate), mutate(child_two, mutation_rate)

def next_generation(population, fitness_values):
    if not population:
        return []

    mutate_rate = mutation_rate(population)

    order = np.argsort(fitness_values)[::-1]
    next_population = [copy.deepcopy(population[i]) for i in order[:NUM_ELITE]]

    while len(next_population) < len(population):
        parent_one = tournament_selection(population, fitness_values)
        parent_two = tournament_selection(population, fitness_values)
        children = crossover(parent_one, parent_two, mutate_rate)
        next_population.extend(children)

    return next_population[:len(population)]

def main():
    population = generate_population()
    history = []

    for generation in range(MAX_GENERATIONS + 1):
        fitness_values = [fitness(genome) for genome in population]
        best_index = int(np.argmax(fitness_values))
        fitness_max = fitness_values[best_index]
        best_genome = population[best_index]
        history.append((fitness_max, float(np.mean(fitness_values))))

        print(f"Generation {generation} max {fitness_max} mean {history[-1][1]} diversity {population_diversity(population)}")

        if generation == MAX_GENERATIONS:
            break
        population = next_generation(population, fitness_values)

    print("Best Genome:", best_genome.genes)
    print("Fitness Max", fitness_max)
    print("Error terms:", {k: round(float(v), 5) for k, v in fitness_term(best_genome).items()})
    # return best_genome, history



def test_main():
    # known_good_angles = [-0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2];
    # known_good_angles = [0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1];

    genome1 = Genome([0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1])
    genome2 = Genome([0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2])

    genome1_fitness = fitness(genome1)
    # genome2_fitness = fitness(genome2)
    print("genome1_fitness", genome1_fitness)
    # print("genome2_fitness", genome2_fitness)
