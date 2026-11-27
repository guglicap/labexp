import math
import random
import pygad

from pynet2plan.eon.helpers import (Spectrum, budgets, k_shortest_paths, lightpath_cost,
                                    num_slots, transponder_table)
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
    table = transponder_table(G)
    S = num_slots(G)
    B = budgets(G)
    T = len(B)

    of_year = {y: [d for d in demands if d.year == y] for y in range(1, T + 1)}
    worth = {d.id: d.traffic * (T - d.year + 1) for d in demands}
    paths = {d.id: k_shortest_paths(G, d.origin, d.destination, k) for d in demands}

    rng = random.Random(seed)

    N = len(demands)
    if N == 0:
        return EonSolution()
    idx_of = {d.id: i for i, d in enumerate(demands)}
    total_budget = sum(B) if sum(B) > 0 else 1.0
    all_links = list(G.edges)

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

    # Precalcolo opzioni per (domanda, transponder, percorso)
    options = {}
    all_unique_paths = {}
    for d in demands:
        opts_d = []
        unique_p = {}
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
                    if p_idx not in unique_p:
                        unique_p[p_idx] = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
            if valid_paths:
                min_c = min(vp[4] for vp in valid_paths) * n_lp
                min_s = min(vp[5] for vp in valid_paths) * n_lp
                opts_d.append((tp, n_lp, min_c, min_s, valid_paths))
        options[d.id] = opts_d
        all_unique_paths[d.id] = list(unique_p.values())

    # Cromosoma di lunghezza 2*N + 2:
    #   0 .. N-1      : priorita' (intatto)
    #   N .. 2*N-1    : indice percorso preferito (intatto)
    #   2*N           : alpha_cost  -> sensibilita' al budget residuo / costo
    #   2*N + 1       : beta_slots  -> sensibilita' alla congestione degli slot sui link
    chrom_len = 2 * N + 2

    def build_plan(chrom):
        spectrum = Spectrum(G)
        sol = EonSolution()
        money = 0.0
        total_cost = 0.0
        # Contatore veloce O(1) degli slot occupati su ciascun link
        used_slots = {e: 0 for e in all_links}

        alpha_cost = float(chrom[2 * N])
        beta_slots = float(chrom[2 * N + 1])

        for y in range(1, T + 1):
            money += B[y - 1]
            year_demands = sorted(of_year[y], key=lambda d: chrom[idx_of[d.id]], reverse=True)

            for d in year_demands:
                i = idx_of[d.id]
                if y < T and chrom[i] < 0.15:
                    sol.block(d, "skipped")
                    continue

                n_paths = len(paths[d.id])
                if n_paths == 0:
                    sol.block(d, "no path")
                    continue
                p0 = min(n_paths - 1, int(chrom[N + i] * n_paths))

                if not options[d.id]:
                    sol.block(d, "no spectrum or budget")
                    continue

                # Tensione sul budget residuo
                money_left_ratio = max(0.0, min(1.0, money / total_budget))
                budget_pressure = 1.0 - money_left_ratio

                # Congestione media sui percorsi validi della domanda d (calcolo istantaneo O(1))
                u_paths = all_unique_paths[d.id]
                if u_paths and S > 0:
                    congestion = sum(
                        max((used_slots[e] / S for e in p_links), default=0.0)
                        for p_links in u_paths
                    ) / len(u_paths)
                else:
                    congestion = 0.0

                max_min_c = max(o[2] for o in options[d.id])
                cost_pull = alpha_cost * max(budget_pressure, 0.25 if money <= max_min_c else 0.0)
                slot_pull = beta_slots * congestion

                denom = cost_pull + slot_pull
                if denom > 1e-9:
                    w_dyn = slot_pull / denom
                else:
                    w_dyn = 0.5

                ordered_opts = sorted(
                    options[d.id],
                    key=lambda o: (1.0 - w_dyn) * o[2] + w_dyn * o[3]
                )
                served = False

                for tp, n_lp, min_c, _, valid_paths in ordered_opts:
                    if round(money - min_c, 9) < 0:
                        continue

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
                        # Aggiorna il contatore veloce degli slot occupati per ogni link
                        for _, segs, _, _ in placed:
                            for seg in segs:
                                for j in range(len(seg) - 1):
                                    used_slots[(seg[j], seg[j + 1])] += tp.slots
                        money -= cost_d
                        total_cost += cost_d
                        for path, _, regens, starts in placed:
                            sol.add_lightpath(d, path, tp, starts, regenerators=regens, year=d.year)
                        served = True
                        break

                if not served:
                    sol.block(d, "no spectrum or budget")

        return sol, total_cost

    # Popolazione iniziale
    by_worth = sorted(range(N), key=lambda i: worth[demands[i].id], reverse=True)
    prio_base = [0.0] * N
    for rank, i in enumerate(by_worth):
        prio_base[i] = 1.0 - 0.7 * (rank / N)
    shortest_paths = [0.0] * N

    initial_pop = [
        prio_base + shortest_paths + [0.8, 0.2],
        prio_base + shortest_paths + [0.2, 0.8],
        prio_base + shortest_paths + [0.5, 0.5],
    ][:pop_size]
    while len(initial_pop) < pop_size:
        initial_pop.append([rng.random() for _ in range(chrom_len)])

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