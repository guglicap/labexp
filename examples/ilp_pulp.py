"""
PuLP (https://coin-or.github.io/pulp/): the toy problem as an integer linear program.

One binary variable per parcel, x[i] = 1 if the parcel is taken. The objective and the
constraints are written as in a book, and solve_ilp passes the model to a solver (HiGHS or
CBC). The solver returns the optimum, with the proof that nothing better exists. The price is
the time: on large problems it may not finish.

    python -m examples.ilp_pulp          (in the folder PyNet2Plan)

This is how my_ilp_planner.py writes its model. Examples on the web often make a variable with
pulp.LpVariable(name, cat="Binary"); in this tool write prob.add_variable(name, cat="Binary").
"""

import pulp

from examples.toy_problem import MAX_VOLUME, MAX_WEIGHT, N, VALUE, VOLUME, WEIGHT, total_value
from pynet2plan.helpers import solve_ilp, value

if __name__ == "__main__":
    prob = pulp.LpProblem("van", pulp.LpMaximize)
    x = [prob.add_variable(f"x_{i}", cat="Binary") for i in range(N)]

    prob += pulp.lpSum(int(VALUE[i]) * x[i] for i in range(N))                       # the objective
    prob += pulp.lpSum(int(WEIGHT[i]) * x[i] for i in range(N)) <= MAX_WEIGHT, "weight"
    prob += pulp.lpSum(int(VOLUME[i]) * x[i] for i in range(N)) <= MAX_VOLUME, "volume"

    status = solve_ilp(prob, time_limit=60)                                          # "Optimal" or "Feasible"
    take = [round(value(v)) for v in x]
    print("PuLP:", status, total_value(take), "  parcels taken:", [i for i in range(N) if take[i]])
