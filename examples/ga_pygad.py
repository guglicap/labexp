"""
PyGAD (https://pygad.readthedocs.io): one bit per parcel.

The chromosome is the solution itself: gene i = 1 if parcel i is taken. There is
no decoder, so a chromosome can break the limits of the van; here such a
chromosome gets a fitness below every feasible one (the more it exceeds, the
worse), which pushes the search back to the feasible side.

    python -m examples.ga_pygad
"""

import numpy as np
import pygad

from examples.toy_problem import (MAX_VOLUME, MAX_WEIGHT, N, VOLUME, WEIGHT, optimum,
                                             total_value)


def fitness(ga, bits, index):
    value = total_value(bits)
    if value >= 0:
        return value
    take = np.asarray(bits, dtype=bool)
    excess = max(0, WEIGHT[take].sum() - MAX_WEIGHT) + max(0, VOLUME[take].sum() - MAX_VOLUME)
    return -float(excess)


if __name__ == "__main__":
    ga = pygad.GA(num_generations=80, sol_per_pop=60, num_parents_mating=20, num_genes=N,
                  gene_space=[0, 1], gene_type=int, fitness_func=fitness,
                  parent_selection_type="tournament", crossover_type="uniform",
                  mutation_type="random", mutation_percent_genes=5, keep_elitism=2,
                  random_seed=1, suppress_warnings=True)
    ga.run()
    bits, value, _ = ga.best_solution()
    print("PyGAD, one bit per parcel:", int(value), "  optimum:", optimum())
