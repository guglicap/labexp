# PyNet2Plan for the Lab Experience project

Open a terminal in this folder and run everything from here.

```
pip install -r requirements.txt                       once
python -m pynet2plan eon                              the EON window
python -m pynet2plan eon-list                         the algorithms the tool sees
```

The tool is also used in another course: the menus and data files that the project does not
mention (Teaching examples, protection, other networks) can be ignored.

In the window: File > Multi-year examples opens an instance (A, B, C, D, small, spain7);
choose the algorithm, Run. The results show the score, the table of the three years (Years
tab) and, with 'Year' above the map, the network at the end of every year.

| File | What it is |
|---|---|
| `formulations/eon/multiyear_greedy.py` | the baseline planner, complete: read it first |
| `formulations/eon/my_ga_planner.py` | yours: the genetic algorithm (every group) |
| `formulations/eon/my_ilp_planner.py` | yours: the ILP (groups of three) |
| `examples/` | a toy problem solved with pymoo, DEAP, PyGAD and as an ILP with PuLP |
| `data/topologies/` | `europe18.json`, `spain7.json` |
| `data/traffic_eon/` | the instances: `europe18_A.csv` ... `europe18_D.csv`, `europe18_small.csv`, `spain7_multiyear.csv` |
| `data/eon/transponders_cost.csv` | the transponder types with their costs |
| `pynet2plan/` | the tool: do not change it |

From the command line (the results are saved in `results/eon/` and open in the window with
Results > Browse results...):

```
python -m pynet2plan eon-run -t data/topologies/europe18.json -d data/traffic_eon/europe18_A.csv -a multiyear_greedy
python -m pynet2plan eon-run -t data/topologies/europe18.json -d data/traffic_eon/europe18_A.csv -a multiyear_greedy -p choose=cost
python -m pynet2plan eon-run -t data/topologies/europe18.json -d data/traffic_eon/europe18_small.csv -a my_ga_planner -p seed=2
python -m pynet2plan eon-run -t data/topologies/spain7.json -d data/traffic_eon/spain7_multiyear.csv -a my_ilp_planner
```

A traffic file carries the settings of its instance in its first lines (`# budgets: 300, 450,
600`, `# slots per link: 80`). To try other settings, add `--budgets 330,495,660` or
`--slots 88` to `eon-run`, or change the fields 'Budget per year' and 'Slots per link' in the window.

The examples of the libraries:

```
python -m examples.toy_problem        the toy problem: a greedy and the optimum
python -m examples.ga_pymoo           GA with pymoo, random keys
python -m examples.ga_deap            GA with DEAP, a permutation
python -m examples.ga_pygad           GA with PyGAD, one bit per item
python -m examples.ilp_pulp           the same problem as an ILP with PuLP
```
