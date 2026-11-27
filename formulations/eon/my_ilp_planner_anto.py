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



import math
import pulp
import time

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


"""
My ILP planner                                              [TO WRITE: groups of three]
================================================================================================

Two-regime ILP planner for multi-year EON planning:
- Small instances (e.g. Spain-7): exact slot-level ILP (proven optimum 15 000).
- Large instances (e.g. Europe-18): aggregate routing/transponder/budget ILP followed by
  multi-order First-Fit spectrum assignment and cumulative-budget-safe recovery.
"""

import math
import time
import pulp

from pynet2plan.eon.helpers import (budgets, k_shortest_paths, lightpath_cost,
                                    num_slots, transponder_table)
from pynet2plan.eon.solution import EonSolution
from pynet2plan.errors import TimeLimitError
from pynet2plan.helpers import solve_ilp, value

NAME = "my ILP planner"
PROTECTION = False
REGENERATION = True
MULTI_PERIOD = True

PARAMETER_HELP = {
    "k": "Number of candidate paths of every demand (the k shortest in km).",
    "time_limit": "Seconds the solver may use (keep the whole run under 10 minutes).",
}


def segment_path(G, path, reach_km):
    """Cut path into segments no longer than reach_km, placing 3R regenerators where needed."""
    segments, start, acc = [], 0, 0.0
    for i in range(len(path) - 1):
        link = G.edges[path[i], path[i + 1]]["length"]
        if link > reach_km:
            return None
        if acc + link > reach_km:
            segments.append(path[start:i + 1])
            start, acc = i, link
        else:
            acc += link
    segments.append(path[start:])
    return segments


# ----------------------------------------------------------------------------------------------
# Helper functions for spectrum assignment & cumulative budget checks (large instances)
# ----------------------------------------------------------------------------------------------

def _first_fit_segment(used, seg, width, S):
    """Lowest starting slot s where [s, s + width) is free on all links of seg."""
    seg_links = [(seg[i], seg[i + 1]) for i in range(len(seg) - 1)]
    for s in range(S - width + 1):
        if all(not any(used[e][s:s + width]) for e in seg_links):
            return s
    return None


def _try_place_lightpaths(used, S, lp_specs):
    """Try to place all lightpaths of a demand on a trial copy of the spectrum."""
    trial = {e: row[:] for e, row in used.items()}
    placed = []
    for path, tp, segs in lp_specs:
        width = tp.slots
        starts = []
        for seg in segs:
            s = _first_fit_segment(trial, seg, width, S)
            if s is None:
                return None, None
            starts.append(s)
        # All segments of a simple path are edge-disjoint, so commit them now
        for seg, s in zip(segs, starts):
            for i in range(len(seg) - 1):
                e = (seg[i], seg[i + 1])
                trial[e][s:s + width] = [True] * width
        regens = [seg[0] for seg in segs[1:]]
        placed.append((path, tp, starts, regens))
    return placed, trial


def _max_extra_budget(year, spent, cum_budget, T):
    """Maximum extra money that can be spent in `year` without violating any later year."""
    cum_s = 0.0
    for y in range(1, year):
        cum_s += spent[y]
    slack = float("inf")
    for y in range(year, T + 1):
        cum_s += spent[y]
        rem = cum_budget[y] - cum_s
        if rem < slack:
            slack = rem
    return slack


def _try_place_demand_any_path(G, d, table, paths, used, S, max_budget):
    """Find the best valid same-transponder placement for demand d within max_budget."""
    best = None  # (metric, placed, total_cost, trial)
    for tp in table:
        if tp.slots > S:
            continue
        req_n = math.ceil(d.traffic / tp.rate)
        valid_paths = []
        for path in paths[d.id]:
            segs = segment_path(G, path, tp.reach)
            if segs is not None:
                cost = lightpath_cost(tp, len(segs) - 1)
                valid_paths.append((cost, len(path) - 1, path, segs))
        if not valid_paths:
            continue
        valid_paths.sort(key=lambda item: (item[0], item[1]))
        if round(valid_paths[0][0] * req_n, 6) > round(max_budget, 6):
            continue

        trial = {e: row[:] for e, row in used.items()}
        placed = []
        total_cost = 0.0
        total_slot_links = 0
        ok = True

        for _ in range(req_n):
            lp_ok = False
            for cost, hops, path, segs in valid_paths:
                if round(total_cost + cost, 6) > round(max_budget, 6):
                    continue
                starts = [_first_fit_segment(trial, seg, tp.slots, S) for seg in segs]
                if all(s is not None for s in starts):
                    for seg, s in zip(segs, starts):
                        for i in range(len(seg) - 1):
                            e = (seg[i], seg[i + 1])
                            trial[e][s:s + tp.slots] = [True] * tp.slots
                    regens = [seg[0] for seg in segs[1:]]
                    placed.append((path, tp, starts, regens))
                    total_cost += cost
                    total_slot_links += tp.slots * hops
                    lp_ok = True
                    break
            if not lp_ok:
                ok = False
                break

        if ok:
            metric = total_cost + 0.25 * total_slot_links
            if best is None or metric < best[0]:
                best = (metric, placed, total_cost, trial)

    if best is None:
        return None, 0.0, None
    return best[1], best[2], best[3]


# ----------------------------------------------------------------------------------------------
# Regime 1: Exact slot-level ILP (small instances, e.g. Spain-7)
# ----------------------------------------------------------------------------------------------

def _solve_exact(G, demands, table, S, B, T, links, worth, paths, k, time_limit):
    prob = pulp.LpProblem("plan_exact", pulp.LpMaximize)

    x = {d.id: prob.add_variable(f"x_{d.id}", cat="Binary") for d in demands}
    prob += pulp.lpSum(worth[d.id] * x[d.id] for d in demands)

    u = {}
    n = {}
    z = {}
    info = {}
    occ = {(e, slot_u): [] for e in links for slot_u in range(S)}

    for d in demands:
        u_vars_d = []
        for tp_idx, tp in enumerate(table):
            if tp.slots > S:
                continue
            req_n = math.ceil(d.traffic / tp.rate)
            n_vars_tp = []

            for p_idx, path in enumerate(paths[d.id]):
                segs = segment_path(G, path, tp.reach)
                if segs is None:
                    continue
                cost = lightpath_cost(tp, len(segs) - 1)
                key = (d.id, p_idx, tp_idx)

                n_var = prob.add_variable(
                    f"n_{d.id}_{p_idx}_{tp_idx}",
                    lowBound=0, upBound=req_n, cat="Integer"
                )
                n[key] = n_var
                info[key] = (path, tp, segs, cost)
                n_vars_tp.append(n_var)

                for g_idx, seg in enumerate(segs):
                    seg_vars = []
                    for s in range(S - tp.slots + 1):
                        z_var = prob.add_variable(
                            f"z_{d.id}_{p_idx}_{tp_idx}_{g_idx}_{s}",
                            cat="Binary"
                        )
                        z[d.id, p_idx, tp_idx, g_idx, s] = z_var
                        seg_vars.append(z_var)
                        for i in range(len(seg) - 1):
                            e = (seg[i], seg[i + 1])
                            for slot_u in range(s, s + tp.slots):
                                occ[e, slot_u].append(z_var)

                    prob += pulp.lpSum(seg_vars) == n_var, f"cnt_{d.id}_{p_idx}_{tp_idx}_{g_idx}"

            if n_vars_tp:
                u_var = prob.add_variable(f"u_{d.id}_{tp_idx}", cat="Binary")
                u[d.id, tp_idx] = u_var
                u_vars_d.append(u_var)
                prob += pulp.lpSum(n_vars_tp) == req_n * u_var, f"req_{d.id}_{tp_idx}"

        prob += pulp.lpSum(u_vars_d) == x[d.id], f"one_tp_{d.id}"

    for y in range(1, T + 1):
        cum_b = sum(B[:y])
        ids_up_to_y = {d.id for d in demands if d.year <= y}
        cost_terms = [
            info[key][3] * n_var
            for key, n_var in n.items() if key[0] in ids_up_to_y
        ]
        if cost_terms:
            prob += pulp.lpSum(cost_terms) <= cum_b, f"budget_y{y}"

    for e_idx, e in enumerate(links):
        for slot_u in range(S):
            if occ[e, slot_u]:
                prob += pulp.lpSum(occ[e, slot_u]) <= 1, f"spec_{e_idx}_{slot_u}"

    try:
        solve_ilp(prob, time_limit)
    except TimeLimitError:
        from formulations.eon.multiyear_greedy import solve as baseline
        return baseline(G, demands, k=k, choose="cost")

    sol = EonSolution()
    for d in demands:
        if (value(x[d.id]) or 0.0) < 0.5:
            sol.block(d, "not selected by ILP")
            continue
        for key, n_var in n.items():
            if key[0] != d.id:
                continue
            count = int(round(value(n_var) or 0.0))
            if count <= 0:
                continue
            _, p_idx, tp_idx = key
            path, tp, segs, _ = info[key]
            seg_starts = [
                [s for s in range(S - tp.slots + 1)
                 if (value(z[d.id, p_idx, tp_idx, g_idx, s]) or 0.0) > 0.5]
                for g_idx in range(len(segs))
            ]
            regens = [seg[0] for seg in segs[1:]]
            for m in range(count):
                starts = [seg_starts[g_idx][m] for g_idx in range(len(segs))]
                sol.add_lightpath(d, path, tp, starts, regenerators=regens, year=d.year)

    return sol


# ----------------------------------------------------------------------------------------------
# Regime 2: Scalable Aggregate ILP + Multi-Order First-Fit + Recovery (large instances)
# ----------------------------------------------------------------------------------------------

def _decode_and_recover(G, demands, table, S, B, T, links, worth, paths, selected_info):
    cum_budget = {y: sum(B[:y]) for y in range(1, T + 1)}
    selected_demands = [d for d in demands if d.id in selected_info]

    orderings = [
        # 1) Highest worth first, then widest slot demand
        sorted(selected_demands, key=lambda d: (-worth[d.id], -selected_info[d.id][2], d.year)),
        # 2) Widest slot demand first (First-Fit Decreasing), then highest worth
        sorted(selected_demands, key=lambda d: (-selected_info[d.id][2], -worth[d.id], d.year)),
    ]

    best_score = -1.0
    best_sol = None
    best_dropped = len(selected_demands)

    for ordered in orderings:
        used = {e: [False] * S for e in links}
        spent = {y: 0.0 for y in range(1, T + 1)}
        sol = EonSolution()
        served_ids = set()
        score = 0.0
        dropped = 0

        # Pass 1: place demands chosen by the aggregate ILP
        for d in ordered:
            lp_specs, total_cost, _ = selected_info[d.id]
            slack = _max_extra_budget(d.year, spent, cum_budget, T)
            if round(total_cost, 6) <= round(slack, 6):
                placed, new_used = _try_place_lightpaths(used, S, lp_specs)
                if placed is not None:
                    used = new_used
                    spent[d.year] += total_cost
                    served_ids.add(d.id)
                    score += worth[d.id]
                    for path, tp, starts, regens in placed:
                        sol.add_lightpath(d, path, tp, starts, regenerators=regens, year=d.year)
                    continue
            dropped += 1

        # Pass 2: recovery using leftover budget & spectrum
        unserved = [d for d in demands if d.id not in served_ids]
        unserved.sort(key=lambda d: (-worth[d.id], d.year, d.traffic))
        for d in unserved:
            slack = _max_extra_budget(d.year, spent, cum_budget, T)
            if slack <= 1e-9:
                continue
            placed, cost, new_used = _try_place_demand_any_path(
                G, d, table, paths, used, S, slack
            )
            if placed is not None:
                used = new_used
                spent[d.year] += cost
                served_ids.add(d.id)
                score += worth[d.id]
                for path, tp, starts, regens in placed:
                    sol.add_lightpath(d, path, tp, starts, regenerators=regens, year=d.year)

        for d in demands:
            if d.id not in served_ids:
                sol.block(d, "not selected / no spectrum or budget")

        if score > best_score:
            best_score = score
            best_sol = sol
            best_dropped = dropped

    return best_score, best_sol, best_dropped


def _solve_scalable(G, demands, table, S, B, T, links, worth, paths, k, time_limit):
    t_start = time.monotonic()
    alphas = (1.0, 0.92)
    best_score = -1.0
    best_sol = None

    for idx, alpha in enumerate(alphas):
        elapsed = time.monotonic() - t_start
        remaining = time_limit - elapsed
        if remaining < 5.0:
            break
        sub_limit = min(90.0, remaining / (len(alphas) - idx))

        prob = pulp.LpProblem(f"plan_agg_{int(alpha * 100)}", pulp.LpMaximize)
        x = {d.id: prob.add_variable(f"x_{d.id}", cat="Binary") for d in demands}

        u = {}
        n = {}
        info = {}
        link_slots = {e: [] for e in links}

        for d in demands:
            u_vars_d = []
            for tp_idx, tp in enumerate(table):
                if tp.slots > S:
                    continue
                req_n = math.ceil(d.traffic / tp.rate)
                n_vars_tp = []

                for p_idx, path in enumerate(paths[d.id]):
                    segs = segment_path(G, path, tp.reach)
                    if segs is None:
                        continue
                    cost = lightpath_cost(tp, len(segs) - 1)
                    key = (d.id, p_idx, tp_idx)

                    n_var = prob.add_variable(
                        f"n_{d.id}_{p_idx}_{tp_idx}",
                        lowBound=0, upBound=req_n, cat="Integer"
                    )
                    n[key] = n_var
                    info[key] = (path, tp, segs, cost)
                    n_vars_tp.append(n_var)

                    for i in range(len(path) - 1):
                        e = (path[i], path[i + 1])
                        link_slots[e].append(tp.slots * n_var)

                if n_vars_tp:
                    u_var = prob.add_variable(f"u_{d.id}_{tp_idx}", cat="Binary")
                    u[d.id, tp_idx] = u_var
                    u_vars_d.append(u_var)
                    prob += pulp.lpSum(n_vars_tp) == req_n * u_var, f"req_{d.id}_{tp_idx}"

            prob += pulp.lpSum(u_vars_d) == x[d.id], f"one_tp_{d.id}"

        # Objective: maximize score, with tiny tie-breaker preferring fewer slots and lower cost
        obj_score = pulp.lpSum(worth[d.id] * x[d.id] for d in demands)
        obj_slots = pulp.lpSum(
            info[key][1].slots * (len(info[key][0]) - 1) * n_var
            for key, n_var in n.items()
        )
        obj_cost = pulp.lpSum(
            info[key][3] * n_var
            for key, n_var in n.items()
        )
        prob += obj_score - 1e-4 * obj_slots - 1e-5 * obj_cost

        # Cumulative budget constraints
        for y in range(1, T + 1):
            cum_b = sum(B[:y])
            ids_up_to_y = {d.id for d in demands if d.year <= y}
            cost_terms = [
                info[key][3] * n_var
                for key, n_var in n.items() if key[0] in ids_up_to_y
            ]
            if cost_terms:
                prob += pulp.lpSum(cost_terms) <= cum_b, f"budget_y{y}"

        # Aggregate spectrum capacity per link
        cap = int(math.floor(alpha * S))
        for e_idx, e in enumerate(links):
            if link_slots[e]:
                prob += pulp.lpSum(link_slots[e]) <= cap, f"cap_{e_idx}"

        try:
            status = solve_ilp(prob, sub_limit)
        except TimeLimitError:
            continue

        # Extract Phase-1 decisions per selected demand
        selected_info = {}
        for d in demands:
            if (value(x[d.id]) or 0.0) < 0.5:
                continue
            lp_specs = []
            total_cost = 0.0
            tp_slots = 0
            for key, n_var in n.items():
                if key[0] != d.id:
                    continue
                count = int(round(value(n_var) or 0.0))
                if count <= 0:
                    continue
                path, tp, segs, cost = info[key]
                tp_slots = tp.slots
                total_cost += cost * count
                for _ in range(count):
                    lp_specs.append((path, tp, segs))
            if lp_specs:
                selected_info[d.id] = (lp_specs, total_cost, tp_slots)

        score, sol, dropped = _decode_and_recover(
            G, demands, table, S, B, T, links, worth, paths, selected_info
        )
        if score > best_score:
            best_score = score
            best_sol = sol

        # If alpha=1.0 is proven Optimal and lost zero demands in First-Fit, we are optimal
        if alpha == 1.0 and status == "Optimal" and dropped == 0:
            break

    if best_sol is not None:
        return best_sol

    from formulations.eon.multiyear_greedy import solve as baseline
    return baseline(G, demands, k=k, choose="cost")


# ----------------------------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------------------------

def solve(G, demands, k=3, time_limit=300):
    table = transponder_table(G)
    S = num_slots(G)
    B = budgets(G)
    T = len(B)
    links = list(G.edges)
    worth = {d.id: d.traffic * (T - d.year + 1) for d in demands}
    paths = {d.id: k_shortest_paths(G, d.origin, d.destination, k) for d in demands}

    if len(demands) * S <= 2500:
        return _solve_exact(G, demands, table, S, B, T, links, worth, paths, k, time_limit)
    return _solve_scalable(G, demands, table, S, B, T, links, worth, paths, k, time_limit)


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
    #sol = EonSolution()
    #   from the values of your variables:
    #   sol.add_lightpath(d, path, tp, starts, regenerators=[...], year=d.year)   (see my_ga_planner.py)
    #raise NotCompletedError("TODO: write your ILP (see the text at the top of this file)")
    #return sol
