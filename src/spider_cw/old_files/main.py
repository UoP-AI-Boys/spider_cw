import copy
import random as rn

import numpy as np

from spider_cw.kinematics import leg_joints
from spider_cw.plot import animate_gait

# Legs in the order their angles appear in a gene vector
LEG_ORDER = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
# (min, max) angle in radians for each of a leg's three joints: coxa, femur, tibia.
JOINT_BOUNDS = [(-0.5, 0.5), (-1.2, 0.6), (-2.2, -0.6)]
JOINT_WIDTHS = [high - low for low, high in JOINT_BOUNDS]
# One gene per joint: 8 legs * 3 joints = 24 angles per frame.
NUM_GENES = len(LEG_ORDER) * 3
# Per-gene lower/upper bounds and ranges. The 3-joint pattern is tiled once per leg
# so these line up with the 24 genes in a frame and broadcast over all frames.
LOW = np.array([low for low, high in JOINT_BOUNDS] * len(LEG_ORDER))
HIGH = np.array([high for low, high in JOINT_BOUNDS] * len(LEG_ORDER))
WIDTHS = HIGH - LOW

# Parameters
FRAMES = 300  # Length of the gait animation (rows per chromosome)
POPULATION_SIZE = 150
MAX_GENERATIONS = 100
TOURNAMENT_SIZE = 3  # Number of random contenders per selection
NUM_ELITE = 2  # Best chromosomes copied unchanged into the next generation
BASE_MUTATION_RATE = 0.05  # Per-gene mutation chance when the population is diverse
MAX_MUTATION_RATE = 0.3  # Per-gene mutation chance when the population has converged
MUTATION_SIGMA = 0.1  # Std. dev. of mutation noise, as a fraction of a joint's range
# Diversity of a freshly randomised population; used to normalise the current diversity
# when deciding how far the population has converged.
RANDOM_DIVERSITY = 0.3

# Band of per-frame joint movement (distance between a joint's position in
# consecutive frames, in the same units as the leg segment lengths) that is not penalised.
MIN_JOINT_STEP = 0.2
MAX_JOINT_STEP = 1

"""
1. Create new chromosomes with 300 frames of 24 joint angles. (DONE)
2. Fitness function
2.1. Fitness for one frame.
2.2. Fitness for sequential frames respecting previous frames.
2.3. Fitness for spider gait.
2.4. Process all frames.
"""

class Chromosome:
    """A candidate gait: a (FRAMES, NUM_GENES) array of joint angles in radians.

    Each row is one animation frame holding 24 angles (3 per leg, in LEG_ORDER).
    """

    def __init__(self, genes):
        self.genes = np.array(genes, dtype=float)

    @classmethod
    def random(cls):
        genes = np.random.uniform(low=LOW, high=HIGH, size=(FRAMES, NUM_GENES))
        return cls(genes)

    def clamp(self):
        """Force all angles back inside the joint limits (in place) and return self."""
        np.clip(self.genes, LOW, HIGH, out=self.genes)
        return self

    # Length and indexing are in terms of frames, so a chromosome behaves like
    # a sequence of frames
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
    """Score x in (0, 1]: 1.0 when lo <= x <= hi, decaying as x leaves that band.

    The penalty is the distance outside the band, so the score falls off smoothly
    rather than being a hard pass/fail.
    """
    below = np.maximum(lo - x, 0.0)
    above = np.maximum(x - hi, 0.0)
    return 1.0 / (1.0 + below + above)


def compare_joint_vectors(current, previous):
    """Score how far each joint moved between two sets of frames.

    Inputs are joint positions with shape (..., legs, points, 3), where point 0 of
    each leg is its fixed base. The base is skipped since it never moves. Movement
    per joint is rewarded when between MIN_JOINT_STEP and MAX_JOINT_STEP: enough to
    actually move, but not so much that it jerks or teleports between frames.
    Returns scores with shape (..., legs, points - 1).
    """
    current_no_base = current[..., 1:, :]
    previous_no_base = previous[..., 1:, :]
    dists = np.linalg.norm(current_no_base - previous_no_base, axis=-1)
    return bounded_fitness(dists, MAX_JOINT_STEP, MIN_JOINT_STEP)

def four_feet_on_floor (chromosome):
    for joint in chromosome:
        pass

def fitness_sequential(chromosome: Chromosome):
    """Mean frame-to-frame movement score over the whole gait (higher is better)."""
    # Forward kinematics: joint angles -> 3D joint positions for every frame.
    joint_vectors = leg_joints(chromosome)
    # Compare each frame (1..N) against the frame before it (0..N-1).
    scores = compare_joint_vectors(joint_vectors[1:], joint_vectors[:-1])
    # checks if four total feet are touching the ground in-line with the biological tendancies for aracnid gaits
    four_feet_on_floor(chromosome)
    return np.mean(scores)

def fitness(chromosome: Chromosome):
    # Single entry point for the fitness function so the metric can be swapped easily.
    return fitness_sequential(chromosome)

def tournament_selection(population, fitness_values, tournament_size=TOURNAMENT_SIZE):
    """Pick tournament_size distinct random chromosomes and return the fittest."""
    contenders = rn.sample(range(len(population)), tournament_size)

    return population[max(contenders, key=lambda i: fitness_values[i])]


def population_diversity(population):
    """How spread out the population is (0 = all identical, ~0.29 = uniform random)."""
    genes = np.array([chromosome.genes for chromosome in population])  # (pop, frames, genes)
    # Standard deviation across the population only (axis 0), separately for every
    # frame and gene, so a population of identical gaits scores 0 even if each gait
    # varies over time. Each gene is normalised by its joint's range (WIDTHS), then
    # everything is averaged. A uniform distribution has std = width / sqrt(12) ~ 0.29,
    # which is where RANDOM_DIVERSITY comes from.
    return float(np.mean(genes.std(axis=0) / WIDTHS))


def mutation_rate(population, base=BASE_MUTATION_RATE, top=MAX_MUTATION_RATE):
    """Adaptive mutation rate: rises from `base` to `top` as diversity collapses."""
    # 0 when as diverse as a random population, approaching 1 as it converges.
    convergence = max(0.0, 1 - population_diversity(population) / RANDOM_DIVERSITY)
    return base + (top - base) * convergence

def mutate(chromosome, mutation_rate, sigma=MUTATION_SIGMA):
    """Gaussian mutation applied in place; returns the clamped chromosome."""
    shape = chromosome.genes.shape
    # Each gene is independently selected for mutation with probability mutation_rate.
    mask = np.random.random(shape) < mutation_rate
    # Noise is scaled by each joint's range so all joints mutate proportionally.
    noise = np.random.normal(0, sigma, shape) * WIDTHS
    chromosome.genes += mask * noise
    return chromosome.clamp()

def crossover(parent_one, parent_two, mutation_rate):
    """Single-point crossover along the time (frame) axis, then mutate both children."""
    if len(parent_one) != len(parent_two):
        raise ValueError("parent chromosomes must have the same length")

    # Cut between frames (never at the ends, so both parents contribute); children
    # swap the tails of the parents.
    point = rn.randint(1, len(parent_one) - 1)
    child_one = Chromosome(np.concat([parent_one.genes[:point], parent_two.genes[point:]]))
    child_two = Chromosome(np.concat([parent_two.genes[:point], parent_one.genes[point:]]))
    return mutate(child_one, mutation_rate), mutate(child_two, mutation_rate)

def next_generation(population, fitness_values):
    """Build the next population using elitism, tournament selection, crossover and mutation."""
    if not population:
        return []

    # Computed once per generation from the current population's diversity.
    mutate_rate = mutation_rate(population)

    # Indices sorted best-first; deep-copy elites so later mutation can't alter them.
    order = np.argsort(fitness_values)[::-1]
    next_population = [copy.deepcopy(population[i]) for i in order[:NUM_ELITE]]

    while len(next_population) < len(population):
        parent_one = tournament_selection(population, fitness_values)
        parent_two = tournament_selection(population, fitness_values)
        children = crossover(parent_one, parent_two, mutate_rate)
        next_population.extend(children)

    # Children are added in pairs, so trim in case that overshot the population size.
    return next_population[:len(population)]

def main():
    """Run the genetic algorithm, then animate and save the best gait found."""
    population = generate_population()
    history = []  # (max fitness, mean fitness) per generation

    # +1 so the final generation is evaluated (and reported) before stopping.
    for generation in range(MAX_GENERATIONS + 1):
        fitness_values = [fitness(chromosome) for chromosome in population]
        best_index = int(np.argmax(fitness_values))
        fitness_max = fitness_values[best_index]
        best_chromosome = population[best_index]
        history.append((fitness_max, float(np.mean(fitness_values))))

        print(f"Generation {generation} max {fitness_max} mean {history[-1][1]} diversity {population_diversity(population)}")

        # Stop before breeding so best_chromosome comes from the final evaluated population.
        if generation == MAX_GENERATIONS:
            break
        population = next_generation(population, fitness_values)

    print("Best chromosome:", best_chromosome.genes)
    print("Fitness Max", fitness_max)

    animate_gait(best_chromosome, save="gaits/gait_best.gif")
    # return best_chromosome, history
