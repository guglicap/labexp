"""
Multi-year greedy with 3R and a budget per year                          [GIVEN - complete]
================================================================================================

Plans a network over several years. The demands of the traffic file carry the year in
which they arrive (column ``year``); every year the operator has a hardware budget
(``budgets(G)``), and what it does not spend stays available. A lightpath installed in a
year stays where it is until the end: it is never moved or removed, so the spectrum only
fills up. A demand can only be served in the year in which it arrives: if it is not served
then, it is lost. The score of a plan is in Gb/s·years: a demand counts its Gb/s in every year
from the one in which it arrives (and is served).

This is the simplest multi-year planner, the baseline of the project. Every year it goes
through the demands of the year, in the order of the file. For a demand it
tries the k shortest paths; on a path it takes the lightpaths with the fewest slots
(``choose="slots"``) or the cheapest ones (``choose="cost"``), with a 3R where the reach
ends, and first-fit slots. If the money left cannot pay them, or they do not fit, it tries
the next path. A demand that no path can take is blocked (lost).

It never looks ahead: it does not choose which demands deserve the money, it does not weigh
hardware against spectrum, and it does not care where it leaves the free slots. Compare
``choose="slots"`` and ``choose="cost"`` on europe18_A.csv and look at the table of the
years in the report.
"""

from pynet2plan.eon.helpers import (Spectrum, best_combination, budgets, k_shortest_paths, lightpath_cost,
                                    transponder_table)
from pynet2plan.eon.solution import EonSolution

NAME = "multi-year greedy (3R, budget per year)"
PROTECTION = False
REGENERATION = True
MULTI_PERIOD = True          # the demands have a year, every lightpath gets the year it is installed

PARAMETER_HELP = {
    "k": "Number of candidate paths tried for each demand (k shortest by km).",
    "choose": "'slots' (fewest slots, then fewest 3R) or 'cost' (cheapest hardware, then fewest slots).",
}
PARAMETER_CHOICES = {"choose": ["slots", "cost"]}


def segment_path(G, path, reach_km):
    """Cut ``path`` into segments no longer than ``reach_km``, with a 3R at the last node
    before the reach would be exceeded. None if one link alone is longer than the reach."""
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


def solve(G, demands, k=3, choose="slots"):
    table = transponder_table(G)
    money_of_year = budgets(G)              # [B1, B2, B3]
    sol = EonSolution()
    spectrum = Spectrum(G)                  # the slots in use: it only fills up, year after year
    money = 0.0                             # what can still be spent

    for year in range(1, len(money_of_year) + 1):
        money += money_of_year[year - 1]
        for d in [d for d in demands if d.year == year]:     # served now, or never
            served = False
            for path in k_shortest_paths(G, d.origin, d.destination, k):
                candidates = []
                for tp in table:
                    segs = segment_path(G, path, tp.reach)
                    if segs is not None:
                        candidates.append((tp, len(segs) - 1))
                combo = best_combination(candidates, d.traffic, choose)
                if combo is None:
                    continue
                tp, regens = combo[0]       # n lightpaths of one type: this greedy puts them all on this path with the same 3R
                                    # (the rules would allow different paths and different 3R)
                n = len(combo)
                cost = n * lightpath_cost(tp, regens)
                if cost > money + 1e-9:
                    continue                # too expensive on this path: try the next one
                segs = segment_path(G, path, tp.reach)
                trial = spectrum.copy()
                placed = []
                for _ in range(n):
                    starts = []
                    for seg in segs:
                        start = trial.first_fit(seg, tp.slots)
                        if start is None:
                            break
                        trial.occupy(seg, start, tp.slots)
                        starts.append(start)
                    if len(starts) < len(segs):
                        break
                    placed.append(starts)
                if len(placed) < n:
                    continue                # it does not fit: try the next path
                spectrum = trial
                money -= cost
                for starts in placed:
                    sol.add_lightpath(d, path, tp, starts, regenerators=[seg[0] for seg in segs[1:]], year=year)
                served = True
                break
            if not served:
                sol.block(d, f"budget or spectrum in year {year}")
    return sol
