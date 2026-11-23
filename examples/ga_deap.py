"""
DEAP (https://deap.readthedocs.io): a permutation.

The chromosome is an order of the parcels (a permutation of 0..N-1), the decoder
loads them in that order. A permutation needs its own operators: a crossover
that keeps it a permutation (here the ordered crossover, cxOrdered) and a
mutation that swaps positions (mutShuffleIndexes). In DEAP you assemble the
algorithm yourself from these pieces, which is more work than pymoo and gives
more control.

    python -m examples.ga_deap
"""

import random

from deap import algorithms, base, creator, tools

from examples.toy_problem import N, load_in_order, optimum, total_value

creator.create("Fitness", base.Fitness, weights=(1.0,))        # 1.0 = maximise
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("order", random.sample, range(N), N)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.order)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda order: (total_value(load_in_order(order)),))   # a tuple
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == "__main__":
    random.seed(1)
    population = toolbox.population(n=60)
    best = tools.HallOfFame(1)
    algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.3, ngen=80, halloffame=best, verbose=False)
    print("DEAP, permutation:", total_value(load_in_order(best[0])), "  optimum:", optimum())
