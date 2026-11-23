"""
My ILP planner                                              [TO WRITE: groups of three]
================================================================================================

The same problem as my_ga_planner.py, solved with an integer linear program (ILP): the
decisions are integer variables, the rules are linear constraints, the score is the
objective, and a solver (HiGHS or CBC, through PuLP) finds the best plan, or the best it can
within the time limit.

The part "1. The inputs" below is done for you, as in my_ga_planner.py. "2. Tools" shows how
to write a model with PuLP in this tool. The model, part 3, is yours: which variables, which
constraints, and what to do on an instance where the full model is too large.

Run it in the window (python -m pynet2plan eon, File > Multi-year examples, algorithm
'my ILP planner', Run) or from the command line, in the folder PyNet2Plan:

    python -m pynet2plan eon-run -t data/topologies/spain7.json -d data/traffic_eon/spain7_multiyear.csv -a my_ilp_planner

The rules and the score are those of my_ga_planner.py. It must run: on every instance of this
kind it must return a valid plan within 10 minutes. When the solver stops at the time limit,
use the best plan it found; when it found none, return a plan made in another way (your GA, or
the baseline: ``from formulations.eon.multiyear_greedy import solve as baseline``). On spain7
the optimum is 15 000. The solver may use its threads; we run it with the default values of the
parameters of solve().
"""

import pulp

from pynet2plan.eon.helpers import (Spectrum, best_combination, budgets, k_shortest_paths, lightpath_cost,
                                    num_slots, path_length, segments_of, transponder_table)
from pynet2plan.eon.solution import EonSolution
from pynet2plan.errors import NotCompletedError, TimeLimitError
from pynet2plan.helpers import solve_ilp, value

NAME = "my ILP planner"
PROTECTION = False
REGENERATION = True
MULTI_PERIOD = True

PARAMETER_HELP = {
    "k": "Number of candidate paths of every demand (the k shortest in km).",
    "time_limit": "Seconds the solver may use (keep the whole run under 10 minutes).",
}


def solve(G, demands, k=3, time_limit=300):
    # ------------------------------------------------------------------------------------------
    # 1. The inputs (as in my_ga_planner.py)
    # ------------------------------------------------------------------------------------------
    table = transponder_table(G)       # the transponder types: tp.name, tp.rate (Gb/s), tp.slots,
                                       # tp.reach (km), tp.cost (one transponder)
    S = num_slots(G)                   # slots per link, numbered 0 .. S-1
    B = budgets(G)                     # [B1, B2, B3]: the money that arrives in every year
    T = len(B)                         # the number of years (3)
    links = list(G.edges)              # the links (u, v), one per direction; G.edges[u, v]["length"] in km
    # demands: d.id, d.origin, d.destination, d.traffic (Gb/s), d.year (served in that year or never)
    of_year = {y: [d for d in demands if d.year == y] for y in range(1, T + 1)}
    worth = {d.id: d.traffic * (T - d.year + 1) for d in demands}     # what it adds to the score if served
    paths = {d.id: k_shortest_paths(G, d.origin, d.destination, k) for d in demands}

    # ------------------------------------------------------------------------------------------
    # 2. Tools
    # ------------------------------------------------------------------------------------------
    #   prob = pulp.LpProblem("plan", pulp.LpMaximize)
    #   x = prob.add_variable("x_3_0", cat="Binary")          also lowBound=..., upBound=..., cat="Integer"
    #       (examples on the web often write pulp.LpVariable(...): in this tool use prob.add_variable)
    #   prob += pulp.lpSum(worth[d.id] * x[d.id] for d in demands)      the first expression: the objective
    #   prob += pulp.lpSum(...) <= B[0], "budget_year_1"                a constraint, with a name
    #   status = solve_ilp(prob, time_limit)
    #       "Optimal"   the best plan, proven
    #       "Feasible"  stopped at time_limit with a plan that may not be the best
    #       raises TimeLimitError if it found no plan in time (catch it and return a plan made otherwise)
    #   value(x)    the value of a variable (or of an expression) in the solution
    # Spectrum, path_length, segments_of, lightpath_cost, best_combination: see my_ga_planner.py.
    # examples/ilp_pulp.py solves a toy problem in the same way.

    # ------------------------------------------------------------------------------------------
    # 3. Your model
    # ------------------------------------------------------------------------------------------
    sol = EonSolution()
    #   from the values of your variables:
    #   sol.add_lightpath(d, path, tp, starts, regenerators=[...], year=d.year)   (see my_ga_planner.py)
    raise NotCompletedError("TODO: write your ILP (see the text at the top of this file)")
    return sol
