from spider_cw.plot import animate_gait
import time
import copy
import pprint
import random as rn

import numpy as np

from spider_cw.kinematics import leg_joints

LEG_ORDER = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINT_BOUNDS = [(-0.5, 0.5), (-1.2, 0.6), (-2.2, -0.6)]
JOINT_WIDTHS = [high - low for low, high in JOINT_BOUNDS]
NUM_GENES = len(LEG_ORDER) * 3
LOW = np.array([low for low, high in JOINT_BOUNDS] * len(LEG_ORDER))
HIGH = np.array([high for low, high in JOINT_BOUNDS] * len(LEG_ORDER))
WIDTHS = HIGH - LOW

# Parameters
FRAMES = 300
POPULATION_SIZE = 150
MAX_GENERATIONS = 100
TOURNAMENT_SIZE = 3
NUM_ELITE = 2
BASE_MUTATION_RATE = 0.05
MAX_MUTATION_RATE = 0.3
MUTATION_SIGMA = 0.1
RANDOM_DIVERSITY = 0.3

"""
1. Create new chromosomes with 300 frames of 24 joint angles. (DONE)
2. Fitness function
2.1. Fitness for one frame.
2.2. Fitness for sequential frames respecting previous frames.
2.3. Fitness for spider gait.
2.4. Process all frames.
"""

class Chromosome:
    def __init__(self, genes):
        self.genes = np.array(genes, dtype=float)

    @classmethod
    def random(cls):
        genes = np.random.uniform(low=LOW, high=HIGH, size=(FRAMES, NUM_GENES))
        return cls(genes)

    def clamp(self):
        np.clip(self.genes, LOW, HIGH, out=self.genes)
        return self

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, index):
        return self.genes[index]


def generate_chromosome() -> Chromosome:
    """Generates a random chromosome."""
    return Chromosome.random()


def generate_population(size=150):
    """generates a single generation of chromosomes"""
    population = []
    for i in range(size):
        chromosome = generate_chromosome()
        population.append(chromosome)
    return population

def bounded_fitness(x, hi, lo):
    below = np.maximum(lo - x, 0.0)
    above = np.maximum(x - hi, 0.0)
    return 1.0 / (1.0 + below + above)


def better_compare_joint_vectors(current, previous):
    current_no_base = current[..., 1:, :]
    previous_no_base = previous[..., 1:, :]
    dists = np.linalg.norm(current_no_base - previous_no_base, axis=-1)
    return bounded_fitness(dists, 1, 0.2)

def better_fitness_sequential(chromosome: Chromosome):
    joint_vectors = leg_joints(chromosome)
    scores = better_compare_joint_vectors(joint_vectors[1:], joint_vectors[:-1])
    return np.mean(scores)

def fitness(chromosome: Chromosome):
    return better_fitness_sequential(chromosome)

def tournament_selection(population, fitness_values, tournament_size=TOURNAMENT_SIZE):
    contenders = rn.sample(range(len(population)), tournament_size)

    return population[max(contenders, key=lambda i: fitness_values[i])]


def population_diversity(population):
    genes = np.array([chromosome.genes for chromosome in population])
    widths = np.array([JOINT_WIDTHS[i % 3] for i in range(NUM_GENES)])
    return float(np.mean(genes.std() / widths))


def mutation_rate(population, base=BASE_MUTATION_RATE, top=MAX_MUTATION_RATE):
    convergence = max(0.0, 1 - population_diversity(population) / RANDOM_DIVERSITY)
    return base + (top - base) * convergence

def mutate(chromosome, mutation_rate, sigma=MUTATION_SIGMA):
    shape = chromosome.genes.shape
    mask = np.random.random(shape) < mutation_rate
    noise = np.random.normal(0, sigma, shape) * WIDTHS
    chromosome.genes += mask * noise
    return chromosome.clamp()

def crossover(parent_one, parent_two, mutation_rate):
    if len(parent_one) != len(parent_two):
        raise ValueError("parent chromosomes must have the same length")

    point = rn.randint(1, len(parent_one) - 1)
    child_one = Chromosome(np.concat([parent_one.genes[:point], parent_two.genes[point:]]))
    child_two = Chromosome(np.concat([parent_two.genes[:point], parent_one.genes[point:]]))
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
        fitness_values = [fitness(chromosome) for chromosome in population]
        best_index = int(np.argmax(fitness_values))
        fitness_max = fitness_values[best_index]
        best_chromosome = population[best_index]
        history.append((fitness_max, float(np.mean(fitness_values))))

        print(f"Generation {generation} max {fitness_max} mean {history[-1][1]} diversity {population_diversity(population)}")

        if generation == MAX_GENERATIONS:
            break
        population = next_generation(population, fitness_values)

    print("Best chromosome:", best_chromosome.genes)
    print("Fitness Max", fitness_max)

    animate_gait(best_chromosome, save="gaits/gait_best.gif")
    # return best_chromosome, history


def test_main():
    # known_good_angles = [-0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2];
    # known_good_angles = [0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1];

    chromosome1 = Chromosome([0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1])
    chromosome2 = Chromosome([0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2])

    chromosome1_fitness = fitness(chromosome1)
    # chromosome2_fitness = fitness(chromosome2)
    print("chromosome1_fitness", chromosome1_fitness)
    # print("chromosome2_fitness", chromosome2_fitness)
