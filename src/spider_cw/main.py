import copy
import pprint
import random as rn

import numpy as np

from spider_cw.kinematics import BASE_POSITIONS, foot_positions, leg_joints
import sys


LEG_ORDER = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINT_BOUNDS = [(-0.5, 0.5), (-1.2, 0.6), (-2.2, -0.6)]
JOINT_WIDTHS = [high - low for low, high in JOINT_BOUNDS]
NUM_GENES = len(LEG_ORDER) * 3
FRAMES = 300

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
        self.genes = list(genes)

    @classmethod
    def random(cls):
        genes = []
        for _ in range(FRAMES):
            genes.append([rn.uniform(*JOINT_BOUNDS[i % 3]) for i in range(NUM_GENES)])
        return cls(genes)

    def clamp(self, frame):
        for i, gene in enumerate(self.genes[frame]):
            low_bound, high_bound = JOINT_BOUNDS[i % 3]
            self.genes[frame][i] = max(low_bound, min(high_bound, gene))
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


def fitness_term(chromosome: Chromosome):
    feet = foot_positions(chromosome[0])

    ground = np.mean((feet[:, 2] - GROUND_Z) ** 2)

    reach_dist = np.linalg.norm(feet[:, :2] - BASE_POSITIONS[:, :2], axis=1)
    reach = np.mean((reach_dist - TARGET_REACH) ** 2)

    left = feet[:4]
    right = feet[4:][::-1] * [1, -1, 1]

    symmetry = np.mean((left - right) ** 2)

    return {"ground": ground, "reach": reach, "symmetry": symmetry}

def bounded_fitness(x, hi, lo):
    if lo <= x <= hi:
        return 1.0
    dist = lo - x if x < lo else x - hi
    return 1.0 / (1.0 + dist)

def compare_joint_vectors(current, previous):
    """Compare joint vectors of current frame to previous frame."""
    frame_fitness = []
    for leg in range(len(LEG_ORDER)):
        coxa_dist  = np.linalg.norm(current[leg][1] - previous[leg][1])
        femur_dist = np.linalg.norm(current[leg][2] - previous[leg][2])
        tibia_dist = np.linalg.norm(current[leg][3] - previous[leg][3])

        coxa_fitness = bounded_fitness(coxa_dist, 1, 0.2)
        femur_fitness = bounded_fitness(femur_dist, 1, 0.2)
        tibia_fitness = bounded_fitness(tibia_dist, 1, 0.2)

        total_fitness = np.mean([coxa_fitness, femur_fitness, tibia_fitness])
        frame_fitness.append(total_fitness)

        # print(f"leg {LEG_ORDER[leg]}: coxa={coxa_dist}, femur={femur_dist}, tibia={tibia_dist}")
        # print(f"leg fitness {LEG_ORDER[leg]}: coxa={coxa_fitness}, femur={femur_fitness}, tibia={tibia_fitness}")
        # print(f"total fitness {LEG_ORDER[leg]}: {total_fitness}")
        # print()

    return np.mean(frame_fitness)


def fitness_sequential(chromosome: Chromosome):
    """Get the fitness of a chromosome by comparing joint vectors frame by frame."""
    frame_fitness = []
    for frame in range(1, len(chromosome)):
        frame_current = chromosome[frame]
        frame_previous = chromosome[frame - 1]

        # Get joint vectors of current frame and previous frame.
        vector_current = leg_joints(frame_current)
        vector_previous = leg_joints(frame_previous)

        # Compare joint vectors of current frame to previous frame.
        fitness = compare_joint_vectors(vector_current, vector_previous)

        # print(f"frame {frame}: fitness={fitness}")

        # Add frame fitness to list of entire chromosome frames fitnesses
        frame_fitness.append(fitness)

    # Calculate entire chromosomes overall fitness
    chromosome_fitness = np.mean(frame_fitness)
    # print(f"chromosome fitness: {chromosome_fitness}")
    return chromosome_fitness


def fitness(chromosome: Chromosome):
    # terms = fitness_term(chromosome)
    total = fitness_sequential(chromosome)
    # total = sum(WEIGHTS[k] * v for k, v in terms.items())
    return 1.0 / (1.0 + total)


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
    for frame in range(len(chromosome)):
        for i in range(NUM_GENES):
            if rn.random() < mutation_rate:
                chromosome.genes[frame][i] += rn.gauss(0, sigma * JOINT_WIDTHS[i % 3])
        chromosome.clamp(frame)
    return chromosome

def crossover(parent_one, parent_two, mutation_rate):
    if len(parent_one) != len(parent_two):
        raise ValueError("parent chromosomes must have the same length")

    crossover_point = rn.randint(1, NUM_GENES - 1)
    child_one = Chromosome(parent_one.genes[:crossover_point] + parent_two.genes[crossover_point:])
    child_two = Chromosome(parent_two.genes[:crossover_point] + parent_one.genes[crossover_point:])
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
    print("Error terms:", {k: round(float(v), 5) for k, v in fitness_term(best_chromosome).items()})
    # return best_chromosome, history


def _main():
    chromo = Chromosome.random()
    genes = chromo.genes

    pprint.pprint(genes)
    print("number of frames", len(genes))


def test_main():
    # known_good_angles = [-0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2, -0.2, -0.4, -1.2];
    # known_good_angles = [0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1];

    chromosome1 = Chromosome([0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1])
    chromosome2 = Chromosome([0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2, 0, -0.6, -1.2])

    chromosome1_fitness = fitness(chromosome1)
    # chromosome2_fitness = fitness(chromosome2)
    print("chromosome1_fitness", chromosome1_fitness)
    # print("chromosome2_fitness", chromosome2_fitness)
