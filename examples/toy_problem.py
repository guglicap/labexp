"""
A toy problem for the examples of the libraries (it is NOT the project).

A van has to be loaded with parcels. Parcel i has a value, a weight and a
volume; the van carries at most MAX_WEIGHT kg and MAX_VOLUME litres. Which
parcels should it take, to carry the largest total value?

ilp_pulp.py solves it exactly as an integer linear program. The three GA examples
use two ways to describe a solution to a genetic algorithm:

* DIRECT: one bit per parcel (1 = take it). Simple, but most bit strings break a
  limit, so the fitness must punish them (see ga_pygad.py).
* INDIRECT: the chromosome is not the solution, it is the input of a small
  greedy procedure (a DECODER) that builds a solution which is always feasible.
  ``load_in_order`` below goes through the parcels in a given order and takes
  each one if it still fits. The chromosome only decides the order: a
  permutation (ga_deap.py) or one number per parcel, sorted (ga_pymoo.py,
  "random keys").
"""

import numpy as np

N = 40
_rng = np.random.default_rng(1)
VALUE = _rng.integers(10, 100, N)
WEIGHT = _rng.integers(5, 40, N)
VOLUME = _rng.integers(10, 80, N)
MAX_WEIGHT = 300
MAX_VOLUME = 600


def total_value(take) -> int:
    """The value of a set of parcels (a list of 0/1), or -1 if it breaks a limit."""
    take = np.asarray(take, dtype=bool)
    if WEIGHT[take].sum() > MAX_WEIGHT or VOLUME[take].sum() > MAX_VOLUME:
        return -1
    return int(VALUE[take].sum())


def load_in_order(order) -> np.ndarray:
    """The decoder: take the parcels in this order, each one if it still fits.
    Returns the 0/1 list of the parcels taken (always feasible)."""
    take = np.zeros(N, dtype=int)
    weight = volume = 0
    for i in order:
        if weight + WEIGHT[i] <= MAX_WEIGHT and volume + VOLUME[i] <= MAX_VOLUME:
            take[i] = 1
            weight += WEIGHT[i]
            volume += VOLUME[i]
    return take


def optimum() -> int:
    """The best possible value, with an integer linear program (scipy.optimize.milp),
    to see how close the GAs get."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    res = milp(c=-VALUE.astype(float),                                   # milp minimises
               constraints=LinearConstraint(np.vstack([WEIGHT, VOLUME]), ub=[MAX_WEIGHT, MAX_VOLUME]),
               integrality=np.ones(N), bounds=Bounds(0, 1))
    return int(round(-res.fun))


if __name__ == "__main__":
    by_ratio = np.argsort(-VALUE / (WEIGHT / MAX_WEIGHT + VOLUME / MAX_VOLUME))
    print("greedy (best value per unit of room first):", total_value(load_in_order(by_ratio)))
    print("optimum (ILP):", optimum())
