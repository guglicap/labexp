"""
pymoo (https://pymoo.org): random keys.

The chromosome is one real number in [0, 1] per parcel. The decoder sorts the
parcels by their number and loads them in that order. Any vector of numbers is
a valid chromosome, so the standard crossover and mutation of pymoo for real
variables can be used as they are.

    python -m examples.ga_pymoo
"""

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from examples.toy_problem import N, load_in_order, optimum, total_value


class Van(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=N, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, keys, out, *args, **kwargs):
        take = load_in_order(np.argsort(keys))
        out["F"] = -total_value(take)              # pymoo minimises


if __name__ == "__main__":
    result = minimize(Van(), GA(pop_size=60), ("n_gen", 80), seed=1, verbose=False)
    take = load_in_order(np.argsort(result.X))
    print("pymoo, random keys:", total_value(take), "  optimum:", optimum())
