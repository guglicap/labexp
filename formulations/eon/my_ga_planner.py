"""
My GA planner                                                       [TO WRITE: every group]
================================================================================================

Your planner, built on a genetic algorithm. ``solve(G, demands)`` receives one instance (the
network, the demands of the three years, the budgets, the slots per link and the transponder
table) and returns the plan of the three years: an EonSolution in which every lightpath has
the year in which it is installed.

The part "1. The inputs" below is done for you: it reads everything the instance gives and
explains it. "2. Tools" lists what pynet2plan offers. The algorithm, part 3, is yours.

Run it in the window (python -m pynet2plan eon, File > Multi-year examples, algorithm
'my GA planner', Run) or from the command line, in the folder PyNet2Plan:

    python -m pynet2plan eon-run -t data/topologies/europe18.json -d data/traffic_eon/europe18_A.csv -a my_ga_planner

The rules (the checker of the tool tests all of them; the result shows INVALID otherwise):

* a lightpath: a path without loops, one transponder type, 3R where you want them; every
  segment between two 3R is within the reach of the type and uses the same adjacent slots on
  all its links (two segments may use different slots); two lightpaths never share a slot on
  a link;
* a demand is served in the year in which it arrives, or never: all its lightpaths are of the
  same type, installed in year d.year, and together carry at least d.traffic Gb/s (they may
  follow different paths and have different 3R);
* nothing is ever moved or removed;
* the hardware installed up to year y costs at most B1 + ... + By.

Score: a served demand counts d.traffic in every year from d.year to T (Gb/s·years). Between
two plans with the same score, the cheaper wins.

Requirements: a valid plan on any instance of this kind within 10 minutes on a laptop, no
parallel processes, and the same plan every time it runs with the same seed: make every random
number from ``seed`` (the generator ``rng`` below; for a library that uses the module random,
as DEAP does, call random.seed(seed); for numpy, np.random.default_rng(seed)), and stop after a
fixed number of generations, not after a fixed time. We run it with the default values of the
parameters of solve(): put your best settings there. Do not write the numbers of an instance in
the code.

The GA libraries pymoo, DEAP and PyGAD are installed; examples/ shows each of them on a toy
problem. You can put more code in other files of this folder whose name starts with "_" (they
are not listed as algorithms; python -m pynet2plan eon-list shows what is) and import them, e.g.
``from formulations.eon._my_decoder import decode``. You may also import from the baseline
(``from formulations.eon.multiyear_greedy import segment_path``, which puts the 3R where the
reach ends) and from examples/.
"""

import random

from pynet2plan.eon.helpers import (Spectrum, best_combination, budgets, k_shortest_paths, lightpath_cost,
                                    num_slots, path_length, segments_of, transponder_table)
from pynet2plan.eon.solution import EonSolution
from pynet2plan.errors import NotCompletedError

NAME = "my GA planner"
PROTECTION = False
REGENERATION = True
MULTI_PERIOD = True              # the demands have a year, every lightpath gets the year it is installed

PARAMETER_HELP = {
    "seed": "Seed of the random numbers: the same seed must give the same plan.",
    "k": "Number of candidate paths of every demand (the k shortest in km).",
}


def plan_score(sol, demands, T):
    """The score of a plan in Gb/s·years, as the tool computes it: every demand with at least
    one lightpath counts d.traffic in the years d.year .. T. (It does not check the rules: a
    plan whose lightpaths do not carry the whole demand is not valid at all.)"""
    served = {lp.demand_id for lp in sol.lightpaths}
    return sum(d.traffic * (T - d.year + 1) for d in demands if d.id in served)


def solve(G, demands, seed=1, k=3):
    # ------------------------------------------------------------------------------------------
    # 1. The inputs
    # ------------------------------------------------------------------------------------------
    table = transponder_table(G)       # the 9 transponder types. For every tp in table:
                                       #   tp.name    e.g. "400G PCS 16QAM" (what add_lightpath wants)
                                       #   tp.rate    Gb/s carried by one lightpath of this type
                                       #   tp.slots   adjacent slots it occupies on every link (4 or 8)
                                       #   tp.reach   km it can go without a 3R
                                       #   tp.cost    price of ONE transponder (a lightpath has 2, a 3R 2 more)
    S = num_slots(G)                   # slots per link, numbered 0 .. S-1
    B = budgets(G)                     # [B1, B2, B3]: the money that arrives in every year
    T = len(B)                         # the number of years (3)

    # The network: G is a networkx DiGraph. G.nodes are the cities; G.edges are the links, one per
    # direction (a fibre route is two links); G.edges[u, v]["length"] is the length in km.

    # The demands: a list of Demand, each with
    #   d.id            its number (the same as in the traffic file)
    #   d.origin        the city where it starts, d.destination where it ends (one direction only)
    #   d.traffic       the Gb/s it asks
    #   d.year          the year in which it arrives: it can be served only in that year
    of_year = {y: [d for d in demands if d.year == y] for y in range(1, T + 1)}
    worth = {d.id: d.traffic * (T - d.year + 1) for d in demands}     # what it adds to the score if served

    # The candidate paths of every demand: the k shortest in km, the shortest first (lists of cities).
    paths = {d.id: k_shortest_paths(G, d.origin, d.destination, k) for d in demands}

    rng = random.Random(seed)          # your only source of randomness (numpy: np.random.default_rng(seed))

    # ------------------------------------------------------------------------------------------
    # 2. Tools (pynet2plan/eon/helpers.py)
    # ------------------------------------------------------------------------------------------
    #   spectrum = Spectrum(G)               the slots in use on every link, all free at the start
    #     spectrum.first_fit(seg, n)         the lowest s such that slots s .. s+n-1 are free on every
    #                                        link of the segment seg (a list of cities), or None
    #     spectrum.is_free(seg, s, n)        are slots s .. s+n-1 free on every link of seg?
    #     spectrum.occupy(seg, s, n)         take them
    #     spectrum.copy()                    an independent copy (to try something and maybe undo it)
    #   path_length(G, path)                 km
    #   segments_of(path, regenerators)      the path cut at its 3R: ["A","B","C"], ["B"] -> [["A","B"], ["B","C"]]
    #   lightpath_cost(tp, q)                price of one lightpath of type tp with q 3R: tp.cost * (2 + 2q)
    #   best_combination(candidates, d.traffic, "slots" or "cost")
    #                                        n lightpaths of one type for a demand, chosen among
    #                                        candidates = [(tp, number of 3R), ...]
    #   plan_score(sol, demands, T)          above
    # multiyear_greedy.py builds a whole plan with these tools, one demand after the other: read
    # it first. Its loop is a natural starting point for the decoder of your GA.

    # ------------------------------------------------------------------------------------------
    # 3. Your genetic algorithm
    # ------------------------------------------------------------------------------------------
    sol = EonSolution()
    #   sol.add_lightpath(d, path, tp, starts, regenerators=[...], year=d.year)
    #       path          the list of cities from d.origin to d.destination
    #       tp            a transponder type of table (or its name)
    #       starts        (4th argument; its keyword is start=) the first slot of every segment:
    #                     [s] without 3R, [s1, s2] with one 3R, ...
    #       regenerators  the cities of the path where a 3R is placed
    #   sol.lightpaths is the list of the lightpaths added so far; each lp has lp.demand_id,
    #   lp.transponder (the name), lp.path, lp.starts, lp.regenerators, lp.year
    #   sol.block(d, "why")    optional: a demand you do not serve, with the reason (shown in the results)
    raise NotCompletedError("TODO: write your genetic algorithm (see the text at the top of this file)")
    return sol
