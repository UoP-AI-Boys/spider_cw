import random as rn

LEG_ORDER = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]
JOINT_BOUNDS = [(-0.5, 0.5), (-1.2, 0.6), (-2.2, -0.6)]
NUM_GENES = len(LEG_ORDER) * 3

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
        return self.genes[i:i+3]

def generate_genome() -> Genome:
    """Generates a random genome."""
    return Genome.random()

def generate_population():
    """generates a single generation of genomes"""
    population = []
    for i in range(150):
        genome = generate_genome()
        population.append(genome)
    return (population)


def main():
    genome = generate_genome()

    print(f"L1: {genome.leg("L1")}")
    print(f"L2: {genome.leg("L2")}")
    print(f"L3: {genome.leg("L3")}")
    print(f"L4: {genome.leg("L4")}")
    print(f"R4: {genome.leg("R4")}")
    print(f"R3: {genome.leg("R3")}")
    print(f"R2: {genome.leg("R2")}")
    print(f"R1: {genome.leg("R1")}")

    genome.clamp()

    print(f"L1: {genome.leg("L1")}")
    print(f"L2: {genome.leg("L2")}")
    print(f"L3: {genome.leg("L3")}")
    print(f"L4: {genome.leg("L4")}")
    print(f"R4: {genome.leg("R4")}")
    print(f"R3: {genome.leg("R3")}")
    print(f"R2: {genome.leg("R2")}")
    print(f"R1: {genome.leg("R1")}")
    pass
