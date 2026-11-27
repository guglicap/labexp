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


import math
import random
import pygad

from pynet2plan.eon.helpers import (Spectrum, best_combination, budgets, k_shortest_paths, lightpath_cost,
                                    num_slots, path_length, segments_of, transponder_table)
from pynet2plan.eon.solution import EonSolution

NAME = "my GA planner"
PROTECTION = False
REGENERATION = True
MULTI_PERIOD = True              # the demands have a year, every lightpath gets the year it is installed

PARAMETER_HELP = {
    "seed": "Seed of the random numbers: the same seed must give the same plan.",
    "k": "Number of candidate paths of every demand (the k shortest in km).",
    "pop_size": "Number of individuals in the PyGAD population.",
    "n_gen": "Fixed number of generations to run.",
}


def plan_score(sol, demands, T):
    """The score of a plan in Gb/s·years, as the tool computes it: every demand with at least
    one lightpath counts d.traffic in the years d.year .. T. (It does not check the rules: a
    plan whose lightpaths do not carry the whole demand is not valid at all.)"""
    served = {lp.demand_id for lp in sol.lightpaths}
    return sum(d.traffic * (T - d.year + 1) for d in demands if d.id in served)


def solve(G, demands, seed=1, k=3, pop_size=40, n_gen=60):
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

    rng = random.Random(seed)          # your only source of randomness

    # ------------------------------------------------------------------------------------------
    # 2. Tools (pynet2plan/eon/helpers.py)
    # ------------------------------------------------------------------------------------------
    #   spectrum = Spectrum(G)               the slots in use on every link, all free at the start
    #     spectrum.first_fit(seg, n)         the lowest s such that slots s .. s+n-1 are free on every
    #                                        link of the segment seg (a list of cities), or None
    #     spectrum.is_free(seg, s, n)        are slots s .. s+n-1 free on every link of seg?
    #     spectrum.occupy(seg, s, n)         take them
    #     spectrum.copy()                    an independent copy (to try something and maybe undo it)
    #   lightpath_cost(tp, q)                price of one lightpath of type tp with q 3R: tp.cost * (2 + 2q)
    #   plan_score(sol, demands, T)          above

    # ------------------------------------------------------------------------------------------
    # 3. Your genetic algorithm (PyGAD)
    # ------------------------------------------------------------------------------------------
    N = len(demands)
    if N == 0:
        return EonSolution()
    idx_of = {d.id: i for i, d in enumerate(demands)}

    # A) Taglia un percorso in segmenti <= reach_km (piazzando un 3R dove finisce la portata)
    def cut_path(path, reach_km):
        segs, start, acc = [], 0, 0.0
        for i in range(len(path) - 1):
            link = G.edges[path[i], path[i + 1]]["length"]
            if link > reach_km:
                return None
            if acc + link > reach_km:
                segs.append(path[start:i + 1])
                start, acc = i, link
            else:
                acc += link
        segs.append(path[start:])
        return segs

    # Calcoliamo una volta sola i segmenti 3R e i costi per ogni (domanda, transponder, percorso)
    options = {}
    for d in demands:
        opts_d = []
        for tp in table:
            if tp.slots > S:
                continue
            n_lp = math.ceil(d.traffic / tp.rate)
            valid_paths = []
            for p_idx, path in enumerate(paths[d.id]):
                segs = cut_path(path, tp.reach)
                if segs is not None:
                    lp_cost = lightpath_cost(tp, len(segs) - 1)
                    lp_slots = tp.slots * (len(path) - 1)
                    regens = [seg[0] for seg in segs[1:]]
                    valid_paths.append((p_idx, path, segs, regens, lp_cost, lp_slots))
            if valid_paths:
                min_c = min(vp[4] for vp in valid_paths) * n_lp
                min_s = min(vp[5] for vp in valid_paths) * n_lp
                opts_d.append((tp, n_lp, min_c, min_s, valid_paths))
        options[d.id] = opts_d

    # B) Trasforma un cromosoma (lista di 2*N + 1 numeri in [0, 1]) in un piano valido
    def build_plan(chrom):
        spectrum = Spectrum(G)
        sol = EonSolution()
        money = 0.0
        total_cost = 0.0
        w = float(chrom[2 * N])  # 0 = preferisci costo basso, 1 = preferisci pochi slot

        for y in range(1, T + 1):
            money += B[y - 1]
            year_demands = sorted(of_year[y], key=lambda d: chrom[idx_of[d.id]], reverse=True)

            for d in year_demands:
                i = idx_of[d.id]
                # Negli anni prima dell'ultimo, un gene < 0.15 salta la domanda per risparmiare
                if y < T and chrom[i] < 0.15:
                    sol.block(d, "skipped")
                    continue

                # Percorso candidato preferito per questa domanda (scelto dal gene chrom[N + i])
                n_paths = len(paths[d.id])
                if n_paths == 0:
                    sol.block(d, "no path")
                    continue
                p0 = min(n_paths - 1, int(chrom[N + i] * n_paths))

                # Prova i transponder in ordine dal migliore secondo il peso w
                ordered_opts = sorted(options[d.id], key=lambda o: (1.0 - w) * o[2] + w * o[3])
                served = False

                for tp, n_lp, min_c, _, valid_paths in ordered_opts:
                    if round(money - min_c, 9) < 0:
                        continue

                    # Mette il percorso preferito p0 per primo, poi gli altri come ripiego
                    ordered_paths = sorted(valid_paths, key=lambda vp: (0 if vp[0] == p0 else 1, vp[0]))

                    trial = spectrum.copy() if n_lp > 1 else spectrum
                    placed = []
                    cost_d = 0.0

                    for _ in range(n_lp):
                        found = False
                        for _, path, segs, regens, lp_cost, _ in ordered_paths:
                            if round(money - (cost_d + lp_cost), 9) < 0:
                                continue
                            starts = [trial.first_fit(seg, tp.slots) for seg in segs]
                            if None in starts:
                                continue
                            if n_lp > 1:
                                for seg, s in zip(segs, starts):
                                    trial.occupy(seg, s, tp.slots)
                            placed.append((path, segs, regens, starts))
                            cost_d += lp_cost
                            found = True
                            break
                        if not found:
                            break

                    if len(placed) == n_lp:
                        if n_lp > 1:
                            spectrum = trial
                        else:
                            _, segs, _, starts = placed[0]
                            for seg, s in zip(segs, starts):
                                spectrum.occupy(seg, s, tp.slots)
                        money -= cost_d
                        total_cost += cost_d
                        for path, _, regens, starts in placed:
                            sol.add_lightpath(d, path, tp, starts, regenerators=regens, year=d.year)
                        served = True
                        break

                if not served:
                    sol.block(d, "no spectrum or budget")

        return sol, total_cost

    # C) Configurazione ed esecuzione di PyGAD
    # Popolazione iniziale: 3 piani base (percorsi piu' corti con w=0, 0.5, 1) + individui casuali
    by_worth = sorted(range(N), key=lambda i: worth[demands[i].id], reverse=True)
    prio_base = [0.0] * N
    for rank, i in enumerate(by_worth):
        prio_base[i] = 1.0 - 0.7 * (rank / N)
    shortest_paths = [0.0] * N

    initial_pop = [
        prio_base + shortest_paths + [0.0],
        prio_base + shortest_paths + [0.5],
        prio_base + shortest_paths + [1.0],
    ][:pop_size]
    while len(initial_pop) < pop_size:
        initial_pop.append([rng.random() for _ in range(2 * N + 1)])

    def fitness_func(ga_instance, solution, solution_idx):
        sol, total_cost = build_plan(solution)
        return plan_score(sol, demands, T) - 1e-4 * total_cost

    ga = pygad.GA(
        num_generations=n_gen,
        num_parents_mating=max(2, pop_size // 2),
        fitness_func=fitness_func,
        initial_population=initial_pop,
        gene_space={"low": 0.0, "high": 1.0},
        parent_selection_type="tournament",
        K_tournament=3,
        keep_elitism=max(2, pop_size // 5),
        crossover_type="uniform",
        mutation_type="random",
        mutation_probability=0.05,
        mutation_by_replacement=True,
        random_seed=seed,
        parallel_processing=None,
        suppress_warnings=True,
    )
    ga.run()

    best_chrom, _, _ = ga.best_solution()
    sol, _ = build_plan(best_chrom)
    return sol
