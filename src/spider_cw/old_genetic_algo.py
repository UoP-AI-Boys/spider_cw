import random as rn
import math


def generate_genome(k):
    genome = list()
    for i  in range (0,k):
        r = rn.randint(0,1)
        genome.append(r)
    return genome

def generate_population ():
    k = 10
    population = list()
    for i in range (150):
        genome = generate_genome(k)
        population.append (genome)
    return (population,k)

def fitness_assination(population):
    fitness_populated = list()
    for i in (population):
        fitness = 0
        for x in (i):
            fitness += x
        fitness_populated.append (fitness)
    return (fitness_populated)

def ranked(fitnessed_population,population):
    ranked = sorted(
    zip(fitnessed_population,population),
    key=lambda pair: pair[0],
    reverse=True
    )
    return ranked

def roulette_selection(population, fitnessed_population):
    if not population:
        raise ValueError("population cannot be empty")
    if len(population) != len(fitnessed_population):
        raise ValueError("population and fitness list must have the same length")

    total_fitness = sum(fitnessed_population)
    if total_fitness <= 0:
        return rn.choice(population).copy()

    selection_point = rn.random() * total_fitness
    cumulative_fitness = 0
    for genome, fitness in zip(population, fitnessed_population):
        cumulative_fitness += fitness
        if selection_point < cumulative_fitness:
            return genome.copy()
    return population[-1].copy()

def population_entropy(population):
    if not population or not population[0]:
        return 0

    entropy_total = 0
    genome_length = len(population[0])
    for gene_index in range(genome_length):
        one_frequency = sum(genome[gene_index] for genome in population) / len(population)
        if 0 < one_frequency < 1:
            entropy_total -= (
                one_frequency * math.log2(one_frequency)
                + (1 - one_frequency) * math.log2(1 - one_frequency)
            )
    return entropy_total / genome_length

def crossover(parent_one, parent_two, population, base_mutation_rate=0.001, max_mutation_rate=0.001):
    if len(parent_one) != len(parent_two):
        raise ValueError("parent genomes must have the same length")
    if not 0 <= base_mutation_rate <= max_mutation_rate <= 1:
        raise ValueError("mutation rates must satisfy 0 <= base <= max <= 1")

    if len(parent_one) > 1:
        crossover_point = rn.randint(1, len(parent_one) - 1)
        child_one = parent_one[:crossover_point] + parent_two[crossover_point:]
        child_two = parent_two[:crossover_point] + parent_one[crossover_point:]
    else:
        child_one = parent_one.copy()
        child_two = parent_two.copy()

    entropy = population_entropy(population)
    mutation_rate = base_mutation_rate + (1 - entropy) * (max_mutation_rate - base_mutation_rate)
    for child in (child_one, child_two):
        for gene_index in range(len(child)):
            if rn.random() < mutation_rate:
                child[gene_index] = 1 - child[gene_index]

    return child_one, child_two

def next_generation(population, fitnessed_population):
    if not population:
        return []

    next_population = [ranked(fitnessed_population, population)[0][1].copy()]
    while len(next_population) < len(population):
        parent_one = roulette_selection(population, fitnessed_population)
        parent_two = roulette_selection(population, fitnessed_population)
        children = crossover(parent_one, parent_two, population)
        next_population.extend(children[:len(population) - len(next_population)])
    return next_population

def run_genetic_algorithm(max_generations=100):
    population,high_score = generate_population()
    genome_length = len(population[0])

    for generation in range(max_generations + 1):
        fitnessed_population = fitness_assination(population)
        best_fitness = max(fitnessed_population)
        if generation == max_generations:
            break
        population = next_generation(population, fitnessed_population)
        best_genome = ranked(fitness_assination(population), population)[0][1]
        best_genome_count = 0
        best_genome_count = 0
        for fitness, genome in ranked(fitness_assination(population), population):
            if fitness == high_score:
                best_genome_count += 1
        print ( best_genome_count)
    return best_genome, best_fitness, generation

if __name__ == "__main__":
    best_genome, best_fitness, generations = run_genetic_algorithm()
    print("Best genome:", best_genome)
    print("Fitness:", best_fitness)
    print("Generations:", generations)
